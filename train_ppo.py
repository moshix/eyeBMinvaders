#!/usr/bin/env python3
"""
eyeBMinvaders PPO Training Script
==================================
Proximal Policy Optimization with parallel Rust game environments.
On-policy learning — every update uses fresh experience from the current policy.

Usage:
    python train_ppo.py                         # Train from scratch
    python train_ppo.py --resume model_ppo.pt   # Resume from checkpoint
    python train_ppo.py --auto-stop             # Stop when peaked

Requirements:
    pip install torch numpy
    Rust game_sim must be compiled: cd game_sim && maturin develop --release
"""

import math
import random
import time
import json
import argparse
import os
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from game_sim import BatchedGames as RustBatchedGames
    HAS_RUST_SIM = True
except ImportError:
    HAS_RUST_SIM = False
    print("ERROR: Rust game_sim not found. PPO requires it.")
    print("Build with: cd game_sim && maturin develop --release")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

torch.set_float32_matmul_precision('high')


# =============================================================================
# Reuse shared infrastructure from train.py
# =============================================================================
from train import FrameStack, auto_scale_for_gpu, TrainingConfig


# =============================================================================
# PPO Configuration
# =============================================================================
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.02
    value_coeff: float = 0.5
    n_epochs: int = 4
    minibatch_size: int = 4096
    rollout_length: int = 256       # steps per env per rollout
    max_grad_norm: float = 1.0
    n_frames: int = 4
    lr_min: float = 1e-5
    lr_warmup_updates: int = 10
    lr_decay_updates: int = 5000
    obs_norm: bool = True
    hidden_sizes: list = None       # backbone sizes
    head_hidden: int = 64           # policy/value head hidden size
    gru_hidden: int = 64            # GRU side-channel hidden size (0 to disable)
    chunk_length: int = 16          # sequential chunk size for GRU training
    use_curriculum: bool = False    # aggressive curriculum (off by default — hurts PPO)
    mixed_starts: bool = True       # progressive curriculum: ramps up high-level starts as model improves
    entropy_coeff_end: float = 0.005  # entropy decays from entropy_coeff → this
    target_kl: float = 0.03          # KL early stop threshold (None to disable)
    sil_capacity: int = 500          # self-imitation buffer: top-K episodes
    sil_min_level: int = 7           # only store episodes reaching this level
    sil_weight: float = 0.1          # SIL loss weight relative to PPO loss
    sil_batch_size: int = 256        # transitions sampled from SIL buffer per update

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [512, 256, 128]


# =============================================================================
# Aggressive Curriculum — stop training on mastered levels
# =============================================================================
class AggressiveCurriculum:
    """Once the agent reliably clears level N, never train below N again.

    Tracks clear rate per level. When clear_rate >= threshold over a window
    of episodes, advances the minimum start level.
    """

    def __init__(self, threshold=0.90, window=500, warmup=10000, max_level=9):
        self.threshold = threshold
        self.window = window
        self.warmup = warmup
        self.max_level = max_level  # don't advance past this start level
        self.min_level = 1
        self.level_clears = {}  # level -> deque of bools (cleared or not)

    def record(self, start_level, reached_level):
        """Record whether the agent cleared start_level (reached start_level+1)."""
        if start_level not in self.level_clears:
            self.level_clears[start_level] = deque(maxlen=self.window)
        self.level_clears[start_level].append(reached_level > start_level)

    def maybe_advance(self, episode_count):
        """Check if we should advance min_level. Returns new min_level or None."""
        if episode_count < self.warmup:
            return None
        lvl = self.min_level
        if lvl not in self.level_clears or len(self.level_clears[lvl]) < self.window // 2:
            return None
        clear_rate = sum(self.level_clears[lvl]) / len(self.level_clears[lvl])
        if clear_rate >= self.threshold and lvl < self.max_level:
            self.min_level = lvl + 1
            return self.min_level
        return None

    def sample_level(self):
        """Sample a starting level. 70% at min_level, 30% at min_level+1 (challenge)."""
        if random.random() < 0.70:
            return self.min_level
        return self.min_level + 1

    def state_dict(self):
        return {'min_level': self.min_level}

    def load_state_dict(self, d):
        self.min_level = d.get('min_level', 1)


# =============================================================================
# Self-Imitation Learning Buffer
# =============================================================================
class SILBuffer:
    """Stores top-K episodes by reward for self-imitation learning.

    When the agent reaches a high level, the full episode trajectory is stored.
    During PPO updates, transitions from these successful episodes are sampled
    and an auxiliary loss encourages the policy to imitate them.
    """

    def __init__(self, capacity=500, min_level=5):
        self.capacity = capacity
        self.min_level = min_level
        self.episodes = []  # sorted by reward ascending (worst first, best last)
        self.total_transitions = 0

    def maybe_add(self, obs_list, actions_list, returns_list, total_reward, max_level):
        """Add an episode if it reached min_level and is good enough."""
        if max_level < self.min_level:
            return False
        if len(self.episodes) >= self.capacity and total_reward <= self.episodes[0]['reward']:
            return False

        # Store as numpy arrays
        ep = {
            'obs': np.array(obs_list, dtype=np.float32),
            'actions': np.array(actions_list, dtype=np.int64),
            'returns': np.array(returns_list, dtype=np.float32),
            'reward': total_reward,
            'level': max_level,
            'n': len(obs_list),
        }
        self.episodes.append(ep)
        self.episodes.sort(key=lambda e: e['reward'])
        if len(self.episodes) > self.capacity:
            removed = self.episodes.pop(0)
            self.total_transitions -= removed['n']
        self.total_transitions += ep['n']
        return True

    def sample(self, batch_size):
        """Sample random transitions from stored episodes."""
        if self.total_transitions == 0:
            return None, None, None

        obs_all = []
        act_all = []
        ret_all = []
        for _ in range(batch_size):
            ep = self.episodes[random.randint(0, len(self.episodes) - 1)]
            idx = random.randint(0, ep['n'] - 1)
            obs_all.append(ep['obs'][idx])
            act_all.append(ep['actions'][idx])
            ret_all.append(ep['returns'][idx])

        return (np.array(obs_all, dtype=np.float32),
                np.array(act_all, dtype=np.int64),
                np.array(ret_all, dtype=np.float32))

    def __len__(self):
        return len(self.episodes)

    def stats(self):
        if not self.episodes:
            return "empty"
        best = self.episodes[-1]
        return f"{len(self.episodes)} eps, {self.total_transitions} trans, best lvl {best['level']} score {best['reward']:.0f}"


# =============================================================================
# Actor-Critic Network
# =============================================================================
class ActorCritic(nn.Module):
    """Dual-path Actor-Critic: feedforward backbone + GRU side-channel.

    Path A (proven, fast): [stacked_frames] → backbone [512→256→128] → 128-dim
    Path B (memory):       [raw_frame 62] → Linear(62→64) → GRU(64→64) → 64-dim
    Merge: concat([128 backbone, 64 memory, 1 level]) = 193 → heads [193→64→6]

    The feedforward path handles reactive dodging (frame-level precision).
    The GRU path handles temporal patterns (drift tracking, Monster2 arcs).
    If gru_hidden=0, falls back to feedforward-only (128+1=129 → heads).
    """

    def __init__(self, state_size, action_size, hidden_sizes=(512, 256, 128),
                 head_hidden=64, n_frames=4, gru_hidden=64):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_frames = n_frames
        self.raw_state_size = state_size // n_frames if n_frames > 1 else state_size
        self.hidden_sizes = list(hidden_sizes)
        self.head_hidden = head_hidden
        self.gru_hidden = gru_hidden

        # Path A: Shared feedforward backbone (processes stacked frames)
        layers = []
        prev = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        backbone_out = prev  # 128

        # Path B: GRU side-channel (processes raw single frame)
        if gru_hidden > 0:
            self.gru_proj = nn.Linear(self.raw_state_size, gru_hidden)
            self.gru = nn.GRUCell(gru_hidden, gru_hidden)
            head_in = backbone_out + gru_hidden + 1  # 128 + 64 + 1 = 193
        else:
            self.gru_proj = None
            self.gru = None
            head_in = backbone_out + 1  # 128 + 1 = 129

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, action_size),
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Orthogonal initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        # Small init for GRU projection so it starts near-zero (additive, not disruptive)
        if self.gru_proj is not None:
            nn.init.orthogonal_(self.gru_proj.weight, gain=0.1)

    def _level_feature(self, x):
        """Extract continuous level from state (feature index 2 of last frame)."""
        level_idx = (self.n_frames - 1) * self.raw_state_size + 2
        return x[:, level_idx:level_idx + 1]  # [batch, 1]

    def _raw_frame(self, x):
        """Extract the last raw frame from stacked input."""
        start = (self.n_frames - 1) * self.raw_state_size
        return x[:, start:start + self.raw_state_size]  # [batch, 62]

    def _merge(self, backbone_feat, gru_out, level):
        """Merge backbone + GRU + level into conditioned features."""
        if gru_out is not None:
            return torch.cat([backbone_feat, gru_out, level], dim=-1)
        return torch.cat([backbone_feat, level], dim=-1)

    def forward(self, x, hx=None):
        """Forward pass. Returns (logits, value, new_hx)."""
        backbone_feat = self.backbone(x)
        level = self._level_feature(x)

        gru_out = None
        new_hx = hx
        if self.gru is not None:
            raw = self._raw_frame(x)
            proj = torch.relu(self.gru_proj(raw))
            if hx is None:
                hx = torch.zeros(x.shape[0], self.gru_hidden, device=x.device)
            new_hx = self.gru(proj, hx)
            gru_out = new_hx

        conditioned = self._merge(backbone_feat, gru_out, level)
        logits = self.policy_head(conditioned)
        value = self.value_head(conditioned)
        return logits, value, new_hx

    def get_action_and_value(self, x, hx=None, action_mask_enabled=False):
        logits, value, new_hx = self.forward(x, hx)
        # Action masking: at level 7+, bias toward fire actions when cooldown ready
        # Only active after warmup to let the model learn basic behaviors first
        if action_mask_enabled:
            ss = x.shape[-1] // (self.n_frames if hasattr(self, 'n_frames') else 4)
            current = x[:, -ss:]  # last frame
            level_feat = current[:, 2]     # feature [2] = level / 10
            fire_feat = current[:, 52] if ss > 52 else torch.zeros(x.shape[0], device=x.device)
            mask = (level_feat >= 0.7) & (fire_feat > 0.9)  # level 7+ and fire ready
            if mask.any():
                logits[mask, :3] -= 5.0  # soft bias against non-fire, not hard mask
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1), new_hx

    def get_value(self, x, hx=None):
        backbone_feat = self.backbone(x)
        level = self._level_feature(x)
        gru_out = None
        if self.gru is not None:
            raw = self._raw_frame(x)
            proj = torch.relu(self.gru_proj(raw))
            if hx is None:
                hx = torch.zeros(x.shape[0], self.gru_hidden, device=x.device)
            new_hx = self.gru(proj, hx)
            gru_out = new_hx
        conditioned = self._merge(backbone_feat, gru_out, level)
        return self.value_head(conditioned).squeeze(-1)

    def evaluate_actions_sequential(self, obs_chunks, actions_chunks, init_hx):
        """Evaluate actions on sequential chunks for GRU training.

        obs_chunks: [n_chunks, chunk_len, features]
        actions_chunks: [n_chunks, chunk_len]
        init_hx: [n_chunks, gru_hidden] — GRU state at chunk start
        Returns: log_probs, entropy, values (all [n_chunks * chunk_len])
        """
        n_chunks, chunk_len, _ = obs_chunks.shape
        all_log_probs = []
        all_entropy = []
        all_values = []

        for c in range(n_chunks):
            hx = init_hx[c:c+1].expand(1, -1).squeeze(0) if init_hx is not None else None
            for t in range(chunk_len):
                x = obs_chunks[c, t:t+1]  # [1, features]
                logits, value, hx = self.forward(x, hx.unsqueeze(0) if hx is not None and hx.dim() == 1 else hx)
                dist = Categorical(logits=logits)
                all_log_probs.append(dist.log_prob(actions_chunks[c, t:t+1]))
                all_entropy.append(dist.entropy())
                all_values.append(value.squeeze(-1))
                if hx is not None:
                    hx = hx.squeeze(0)

        return torch.cat(all_log_probs), torch.cat(all_entropy), torch.cat(all_values)

    def evaluate_actions(self, x, actions, hx=None):
        """Non-sequential evaluate (for backward compat / no-GRU mode)."""
        logits, value, _ = self.forward(x, hx)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)


# =============================================================================
# Running Mean/Std for Observation Normalization
# =============================================================================
class RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""
    def __init__(self, shape, device='cpu'):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.device = device

    def update(self, x):
        """Update with a batch of observations [batch, features]."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        """Normalize observation, clamp to [-10, 10]."""
        std = np.sqrt(self.var + 1e-8)
        return np.clip((x - self.mean) / std, -10.0, 10.0).astype(np.float32)

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


# =============================================================================
# GAE Computation
# =============================================================================
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    gae = np.zeros(num_envs, dtype=np.float32)

    for t in reversed(range(steps)):
        if t == steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# =============================================================================
# LR Schedule
# =============================================================================
def scheduled_lr(update_step, lr_max, lr_min, warmup, decay_end):
    """Linear warmup → cosine decay → flat minimum."""
    if update_step < warmup:
        return lr_min + (update_step / max(warmup, 1)) * (lr_max - lr_min)
    elif update_step < decay_end:
        progress = (update_step - warmup) / max(decay_end - warmup, 1)
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return lr_min + cosine * (lr_max - lr_min)
    else:
        return lr_min


# =============================================================================
# PPO Update
# =============================================================================
def ppo_update(agent, optimizer, obs_flat, actions_flat, old_log_probs_flat,
               returns_flat, advantages_flat, cfg, device, entropy_coeff_override=None,
               sil_buffer=None):
    """Run PPO update epochs over collected rollout data."""
    ent_coeff = entropy_coeff_override if entropy_coeff_override is not None else cfg.entropy_coeff
    n = len(obs_flat)

    # Normalize advantages (full rollout)
    adv_mean = advantages_flat.mean()
    adv_std = advantages_flat.std() + 1e-8
    advantages_norm = (advantages_flat - adv_mean) / adv_std

    # Convert to tensors
    obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions_flat, dtype=torch.long, device=device)
    old_lp_t = torch.as_tensor(old_log_probs_flat, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns_flat, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(advantages_norm, dtype=torch.float32, device=device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    n_updates = 0
    kl_early_stopped = False

    for epoch in range(cfg.n_epochs):
        if kl_early_stopped:
            break
        indices = np.random.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            end = min(start + cfg.minibatch_size, n)
            mb = indices[start:end]

            new_log_probs, entropy, new_values = agent.evaluate_actions(
                obs_t[mb], actions_t[mb])

            # KL early stop: if policy drifted too far, abort remaining epochs
            with torch.no_grad():
                approx_kl = (old_lp_t[mb] - new_log_probs).mean().item()
            if cfg.target_kl and approx_kl > cfg.target_kl:
                kl_early_stopped = True
                break

            # Policy loss (clipped surrogate)
            ratio = (new_log_probs - old_lp_t[mb]).exp()
            surr1 = ratio * adv_t[mb]
            surr2 = ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns_t[mb])

            # Entropy bonus
            entropy_mean = entropy.mean()

            # Total loss
            loss = policy_loss + cfg.value_coeff * value_loss - ent_coeff * entropy_mean

            # Self-Imitation Learning: auxiliary loss from successful episodes
            if sil_buffer and len(sil_buffer) > 0:
                sil_data = sil_buffer.sample(cfg.sil_batch_size)
                if sil_data[0] is not None:
                    sil_obs = torch.as_tensor(sil_data[0], dtype=torch.float32, device=device)
                    sil_acts = torch.as_tensor(sil_data[1], dtype=torch.long, device=device)
                    sil_rets = torch.as_tensor(sil_data[2], dtype=torch.float32, device=device)
                    sil_logits, sil_vals, _ = agent.forward(sil_obs)
                    sil_dist = Categorical(logits=sil_logits)
                    # Only imitate when stored return > current value (clipped positive advantage)
                    sil_adv = (sil_rets - sil_vals.squeeze(-1).detach()).clamp(min=0)
                    sil_lp = sil_dist.log_prob(sil_acts)
                    sil_loss = -(sil_lp * sil_adv).mean()
                    loss = loss + cfg.sil_weight * sil_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            total_kl += approx_kl
            n_updates += 1

    return {
        'policy_loss': total_policy_loss / max(n_updates, 1),
        'value_loss': total_value_loss / max(n_updates, 1),
        'entropy': total_entropy / max(n_updates, 1),
        'kl': total_kl / max(n_updates, 1),
        'kl_stopped': kl_early_stopped,
    }


def ppo_update_chunked(agent, optimizer, roll_obs, roll_actions, roll_log_probs,
                       returns, advantages, roll_hx, roll_dones,
                       cfg, device, entropy_coeff_override=None, chunk_length=16):
    """PPO update with batched sequential chunks for GRU training.

    Splits each env's rollout into chunks, then processes all chunks' timestep t
    in parallel before moving to t+1. This keeps the GRU sequential within each
    chunk while leveraging GPU parallelism across chunks.
    """
    ent_coeff = entropy_coeff_override if entropy_coeff_override is not None else cfg.entropy_coeff
    rollout_len, num_envs, state_size = roll_obs.shape

    # Normalize advantages
    adv_flat = advantages.reshape(-1)
    adv_mean = adv_flat.mean()
    adv_std = adv_flat.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    # Build chunks: reshape [rollout, envs, ...] → [n_chunks, chunk_len, ...]
    n_chunks_per_env = rollout_len // chunk_length
    total_chunks = n_chunks_per_env * num_envs

    # Reshape: for each env, split rollout into consecutive chunks
    # Result: [total_chunks, chunk_length, ...]
    chunk_obs = np.zeros((total_chunks, chunk_length, state_size), dtype=np.float32)
    chunk_actions = np.zeros((total_chunks, chunk_length), dtype=np.int64)
    chunk_old_lp = np.zeros((total_chunks, chunk_length), dtype=np.float32)
    chunk_returns = np.zeros((total_chunks, chunk_length), dtype=np.float32)
    chunk_advantages = np.zeros((total_chunks, chunk_length), dtype=np.float32)
    chunk_init_hx = np.zeros((total_chunks, cfg.gru_hidden), dtype=np.float32)

    idx = 0
    for env in range(num_envs):
        for c in range(n_chunks_per_env):
            t0 = c * chunk_length
            t1 = t0 + chunk_length
            chunk_obs[idx] = roll_obs[t0:t1, env]
            chunk_actions[idx] = roll_actions[t0:t1, env]
            chunk_old_lp[idx] = roll_log_probs[t0:t1, env]
            chunk_returns[idx] = returns[t0:t1, env].astype(np.float32)
            chunk_advantages[idx] = advantages[t0:t1, env]
            chunk_init_hx[idx] = roll_hx[t0, env]
            idx += 1

    # Convert to tensors
    chunk_obs_t = torch.as_tensor(chunk_obs, dtype=torch.float32, device=device)
    chunk_actions_t = torch.as_tensor(chunk_actions, dtype=torch.long, device=device)
    chunk_old_lp_t = torch.as_tensor(chunk_old_lp, dtype=torch.float32, device=device)
    chunk_returns_t = torch.as_tensor(chunk_returns, dtype=torch.float32, device=device)
    chunk_adv_t = torch.as_tensor(chunk_advantages, dtype=torch.float32, device=device)
    chunk_hx_t = torch.as_tensor(chunk_init_hx, dtype=torch.float32, device=device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    n_updates = 0
    kl_early_stopped = False

    chunks_per_batch = max(1, cfg.minibatch_size // chunk_length)

    for epoch in range(cfg.n_epochs):
        if kl_early_stopped:
            break
        perm = torch.randperm(total_chunks, device=device)

        for batch_start in range(0, total_chunks, chunks_per_batch):
            batch_end = min(batch_start + chunks_per_batch, total_chunks)
            batch_idx = perm[batch_start:batch_end]
            nb = len(batch_idx)

            # Batched sequential: process all chunks' step t in parallel
            hx = chunk_hx_t[batch_idx]  # [nb, gru_hidden]
            step_log_probs = []
            step_entropy = []
            step_values = []

            for t in range(chunk_length):
                x = chunk_obs_t[batch_idx, t]  # [nb, state_size]
                logits, value, hx = agent.forward(x, hx)
                dist = Categorical(logits=logits)
                step_log_probs.append(dist.log_prob(chunk_actions_t[batch_idx, t]))
                step_entropy.append(dist.entropy())
                step_values.append(value.squeeze(-1))

            # Stack: [chunk_length, nb] → [nb * chunk_length]
            new_log_probs = torch.stack(step_log_probs, dim=1).reshape(-1)
            entropy_all = torch.stack(step_entropy, dim=1).reshape(-1)
            new_values = torch.stack(step_values, dim=1).reshape(-1)

            old_lp_batch = chunk_old_lp_t[batch_idx].reshape(-1)
            returns_batch = chunk_returns_t[batch_idx].reshape(-1)
            adv_batch = chunk_adv_t[batch_idx].reshape(-1)

            # KL early stop
            with torch.no_grad():
                approx_kl = (old_lp_batch - new_log_probs).mean().item()
            if cfg.target_kl and approx_kl > cfg.target_kl:
                kl_early_stopped = True
                break

            # Policy loss
            ratio = (new_log_probs - old_lp_batch).exp()
            surr1 = ratio * adv_batch
            surr2 = ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_values, returns_batch)
            entropy_mean = entropy_all.mean()
            loss = policy_loss + cfg.value_coeff * value_loss - ent_coeff * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            total_kl += approx_kl
            n_updates += 1

    return {
        'policy_loss': total_policy_loss / max(n_updates, 1),
        'value_loss': total_value_loss / max(n_updates, 1),
        'entropy': total_entropy / max(n_updates, 1),
        'kl': total_kl / max(n_updates, 1),
        'kl_stopped': kl_early_stopped,
    }


# =============================================================================
# GPU Auto-Scaling for PPO
# =============================================================================
def auto_scale_ppo(cfg, device, num_envs):
    """Scale PPO parameters based on GPU memory."""
    if device == 'cpu':
        return cfg, num_envs

    try:
        if device == 'cuda' and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
            gpu_name = props.name
        elif device == 'mps':
            gpu_mem_gb = 8
            gpu_name = "Apple MPS"
        else:
            return cfg, num_envs

        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

        if gpu_mem_gb >= 24:
            cfg.minibatch_size = max(cfg.minibatch_size, 4096)
            num_envs = max(num_envs, 2048)
        elif gpu_mem_gb >= 16:
            cfg.minibatch_size = max(cfg.minibatch_size, 2048)
            num_envs = max(num_envs, 1024)
        elif gpu_mem_gb >= 8:
            cfg.minibatch_size = max(cfg.minibatch_size, 1024)
            num_envs = max(num_envs, 512)

        print(f"Auto-scaled: minibatch_size={cfg.minibatch_size}, num_envs={num_envs}")

    except Exception as e:
        print(f"Auto-scale skipped: {e}")

    return cfg, num_envs


# =============================================================================
# Training Loop
# =============================================================================
NUM_ENVS = 256


def train_ppo(episodes=1_000_000, resume_path=None, save_dir="models",
              device_override=None, num_envs=NUM_ENVS, config=None,
              auto_scale=True, auto_stop=False):
    os.makedirs(save_dir, exist_ok=True)

    if device_override:
        device = device_override
    else:
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'

    cfg = config or PPOConfig()

    if auto_scale:
        cfg, num_envs = auto_scale_ppo(cfg, device, num_envs)

    if not HAS_RUST_SIM:
        print("ERROR: Rust game_sim required for PPO training")
        return

    envs = RustBatchedGames(num_envs, seed=42)
    raw_state_size = envs.state_size
    action_size = envs.action_size
    print("Using Rust game simulation")

    # Frame stacking
    frame_stack = None
    if cfg.n_frames > 1:
        frame_stack = FrameStack(num_envs, raw_state_size, cfg.n_frames)
        state_size = frame_stack.stacked_size
    else:
        state_size = raw_state_size

    # Actor-Critic network
    agent = ActorCritic(state_size, action_size,
                        hidden_sizes=cfg.hidden_sizes,
                        head_hidden=cfg.head_hidden,
                        n_frames=cfg.n_frames,
                        gru_hidden=cfg.gru_hidden).to(device)

    # torch.compile for PyTorch 2.x+
    try:
        agent = torch.compile(agent)
    except Exception:
        pass

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # Observation normalization
    obs_rms = RunningMeanStd(state_size, device) if cfg.obs_norm else None

    # Aggressive curriculum
    curriculum = AggressiveCurriculum() if cfg.use_curriculum else None
    if curriculum:
        print("Aggressive curriculum: enabled (advances when 80% clear rate)")

    # Resume from checkpoint
    update_count = 0
    total_episodes = 0
    best_avg_score = 0.0
    best_avg_episode = 0

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        agent_mod = agent._orig_mod if hasattr(agent, '_orig_mod') else agent
        agent_mod.load_state_dict(checkpoint['agent'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        update_count = checkpoint.get('update_count', 0)
        total_episodes = checkpoint.get('total_episodes', 0)
        best_avg_score = checkpoint.get('best_avg_score', 0.0)
        best_avg_episode = checkpoint.get('best_avg_episode', 0)
        if obs_rms and 'obs_rms' in checkpoint:
            obs_rms.load_state_dict(checkpoint['obs_rms'])
        if curriculum and 'curriculum' in checkpoint:
            curriculum.load_state_dict(checkpoint['curriculum'])
        print(f"Resumed from {resume_path} (updates={update_count}, episodes={total_episodes})")

    print(f"Training on device: {device}")
    print(f"Parallel environments: {num_envs}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Episodes target: {episodes:,}")
    print(f"PPO: rollout={cfg.rollout_length} steps × {num_envs} envs = "
          f"{cfg.rollout_length * num_envs:,} transitions/update")
    gru_str = f", GRU={cfg.gru_hidden} (chunks={cfg.chunk_length})" if cfg.gru_hidden > 0 else ""
    print(f"Network: backbone={cfg.hidden_sizes}, heads={cfg.head_hidden}, "
          f"frames={cfg.n_frames}{gru_str}")
    print(f"LR: {cfg.lr} → {cfg.lr_min} (warmup={cfg.lr_warmup_updates}, "
          f"decay={cfg.lr_decay_updates})")
    print(f"Clip={cfg.clip_epsilon}, Entropy={cfg.entropy_coeff}, "
          f"Value={cfg.value_coeff}, GAE λ={cfg.gae_lambda}")
    print(f"Obs normalization: {'ON' if cfg.obs_norm else 'OFF'}")
    print("-" * 70)

    # Pre-allocate rollout buffers
    roll_obs = np.zeros((cfg.rollout_length, num_envs, state_size), dtype=np.float32)
    roll_actions = np.zeros((cfg.rollout_length, num_envs), dtype=np.int64)
    roll_log_probs = np.zeros((cfg.rollout_length, num_envs), dtype=np.float32)
    roll_rewards = np.zeros((cfg.rollout_length, num_envs), dtype=np.float32)
    roll_dones = np.zeros((cfg.rollout_length, num_envs), dtype=np.float32)
    roll_values = np.zeros((cfg.rollout_length, num_envs), dtype=np.float32)
    # GRU hidden states per step (for chunk-based training)
    use_gru = cfg.gru_hidden > 0
    if use_gru:
        roll_hx = np.zeros((cfg.rollout_length, num_envs, cfg.gru_hidden), dtype=np.float32)
        gru_hx = torch.zeros(num_envs, cfg.gru_hidden, device=device)  # current hidden state
    else:
        roll_hx = None
        gru_hx = None

    # Stats
    scores = deque(maxlen=1000)
    levels = deque(maxlen=1000)
    best_score = 0
    best_level = 0
    peak_detected = False
    start_time = time.time()

    # Event log
    log_path = os.path.join(save_dir, "training_events_ppo.jsonl")
    log_file = open(log_path, "a")

    # Initialize environments
    raw_states = envs.reset_all()
    states = frame_stack.reset_all(raw_states) if frame_stack else raw_states
    if obs_rms:
        obs_rms.update(states)
        states = obs_rms.normalize(states)
    env_start_levels = np.ones(num_envs, dtype=np.int32)  # track start level per env
    env_fire_actions = np.zeros(num_envs, dtype=np.int32)  # count fire actions per episode
    env_total_actions = np.zeros(num_envs, dtype=np.int32)  # count total actions per episode

    # Self-Imitation Learning buffer
    sil_buffer = SILBuffer(cfg.sil_capacity, cfg.sil_min_level) if cfg.sil_weight > 0 else None
    # Per-env episode trajectory tracking for SIL
    env_ep_obs = [[] for _ in range(num_envs)]
    env_ep_actions = [[] for _ in range(num_envs)]
    env_ep_rewards = [[] for _ in range(num_envs)]
    if sil_buffer:
        print(f"Self-Imitation Learning: ON (min_level={cfg.sil_min_level}, "
              f"capacity={cfg.sil_capacity}, weight={cfg.sil_weight})")

    episode_count = total_episodes

    while episode_count < episodes:
        # === Collect rollout ===
        for step in range(cfg.rollout_length):
            states_t = torch.as_tensor(states, dtype=torch.float32, device=device)

            with torch.no_grad():
                use_mask = episode_count > 100_000  # warmup: no masking for first 100k episodes
                actions, log_probs, _, values, new_hx = agent.get_action_and_value(states_t, gru_hx, action_mask_enabled=use_mask)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()

            # Store in rollout buffer
            roll_obs[step] = states if not isinstance(states, np.ndarray) else states
            roll_actions[step] = actions_np
            roll_log_probs[step] = log_probs_np
            if use_gru:
                roll_hx[step] = gru_hx.cpu().numpy()  # store hidden state BEFORE this step
                gru_hx = new_hx.detach()
            roll_values[step] = values_np

            # Step environments
            raw_next_states, rewards, dones = envs.step_all_fast(actions_np.tolist())
            next_states = frame_stack.push(raw_next_states) if frame_stack else raw_next_states

            roll_rewards[step] = np.asarray(rewards, dtype=np.float32)
            roll_dones[step] = np.asarray(dones, dtype=np.float32)

            # Track fire vs non-fire actions per env
            env_total_actions += 1
            env_fire_actions += (actions_np >= 3).astype(np.int32)

            # Track per-env trajectories for SIL
            if sil_buffer:
                rewards_np = roll_rewards[step]
                for ei in range(num_envs):
                    env_ep_obs[ei].append(states[ei].copy() if isinstance(states, np.ndarray) else list(states[ei]))
                    env_ep_actions[ei].append(int(actions_np[ei]))
                    env_ep_rewards[ei].append(float(rewards_np[ei]))

            # Handle done environments
            done_mask = np.asarray(dones)
            done_idxs = np.where(done_mask)[0]
            for i in done_idxs:
                i = int(i)
                episode_count += 1
                ep_score, ep_level, ep_lives, ep_steps, ep_ekills, ep_kkills, ep_mshots, ep_hits = envs.get_stats(i)
                scores.append(ep_score)
                levels.append(ep_level)
                if ep_score > best_score:
                    best_score = ep_score
                if ep_level > best_level:
                    best_level = ep_level

                start_lvl = int(env_start_levels[i])

                # Record curriculum result
                if curriculum:
                    curriculum.record(start_lvl, ep_level)
                    new_min = curriculum.maybe_advance(episode_count)
                    if new_min is not None:
                        print(f"  >> Curriculum advanced: min_level={new_min} "
                              f"(mastered level {new_min - 1})")

                # Self-Imitation Learning: store successful episode trajectories
                if sil_buffer and len(env_ep_obs[i]) > 0:
                    # Compute discounted returns for the episode
                    ep_returns = []
                    G = 0.0
                    for r in reversed(env_ep_rewards[i]):
                        G = r + cfg.gamma * G
                        ep_returns.insert(0, G)
                    added = sil_buffer.maybe_add(
                        env_ep_obs[i], env_ep_actions[i], ep_returns,
                        float(episode_rewards[i]), ep_level)
                    env_ep_obs[i] = []
                    env_ep_actions[i] = []
                    env_ep_rewards[i] = []

                # Extended stats for detailed analysis
                ext = {}
                try:
                    ext = dict(envs.get_stats_ext(i))
                except Exception as e:
                    if episode_count <= 5:
                        print(f"  [warn] get_stats_ext failed: {e}")
                log_file.write(json.dumps({
                    "episode": episode_count,
                    "score": ep_score,
                    "level": ep_level,
                    "start_level": start_lvl,
                    "lives_left": ep_lives,
                    "steps": ep_steps,
                    "enemies_killed": ep_ekills,
                    "kamikazes_killed": ep_kkills,
                    "missiles_shot": ep_mshots,
                    "times_hit": ep_hits,
                    "update": update_count,
                    "shots_fired": ext.get("shots_fired", 0),
                    "shots_hit": ext.get("shots_hit", 0),
                    "hit_rate": round(ext["shots_hit"] / max(ext["shots_fired"], 1), 3) if ext.get("shots_fired", 0) > 0 else 0,
                    "edge_cols": ext.get("edge_cols_eliminated", 0),
                    "bounces": ext.get("bounces", 0),
                    "enemies_left": ext.get("enemies_left", 0),
                    "formation_width": ext.get("formation_width", 0),
                    "monsters_killed": ext.get("monsters_killed", 0),
                    "fire_pct": round(int(env_fire_actions[i]) / max(int(env_total_actions[i]), 1), 3),
                }) + "\n")
                # Reset per-episode action counters
                env_fire_actions[i] = 0
                env_total_actions[i] = 0

                if episode_count % 1000 == 0 or episode_count <= 5:
                    elapsed = time.time() - start_time
                    eps_per_sec = episode_count / elapsed if elapsed > 0 else 0
                    avg_score = np.mean(scores) if scores else 0
                    avg_level = np.mean(levels) if levels else 0
                    print(f"Ep {episode_count:>8,} | "
                          f"Avg Score: {avg_score:>8.0f} | "
                          f"Best: {best_score:>8,} | "
                          f"Avg Lvl: {avg_level:.1f} | "
                          f"Best Lvl: {best_level} | "
                          f"Updates: {update_count} | "
                          f"{eps_per_sec:.0f} ep/s | "
                          f"{elapsed:.0f}s")

                    # Best-average model saving
                    if episode_count > 1000:
                        current_avg = float(np.mean(scores))
                        if current_avg > best_avg_score:
                            best_avg_score = current_avg
                            best_avg_episode = episode_count
                            save_checkpoint(agent, optimizer, obs_rms, update_count,
                                          episode_count, best_avg_score, best_avg_episode, cfg,
                                          os.path.join(save_dir, "model_ppo_best_avg.pt"), curriculum)
                            print(f"  -> New best avg: score={current_avg:.0f}, "
                                  f"level={float(np.mean(levels)):.1f}")

                    # Peak detection
                    if episode_count > 50_000 and not peak_detected:
                        eps_since_best = episode_count - best_avg_episode
                        if eps_since_best >= 30_000:
                            current_avg = float(np.mean(scores))
                            if current_avg >= best_avg_score * 0.95:
                                peak_detected = True
                                print(f"\n{'='*70}")
                                print(f"PEAK DETECTED at episode {episode_count}")
                                print(f"  Best avg: {best_avg_score:.0f} (ep {best_avg_episode})")
                                print(f"  Current avg: {current_avg:.0f}")
                                print(f"{'='*70}\n")
                                if auto_stop:
                                    break

                if episode_count % 100 == 0:
                    log_file.flush()

                # Reset done env (with curriculum or mixed starts)
                sl = None
                if curriculum:
                    sl = curriculum.sample_level()
                elif cfg.mixed_starts:
                    # Progressive curriculum: ramp high-level starts based on avg level
                    # Phase 1 (avg < 4): 10% at L3-5 (gentle exposure)
                    # Phase 2 (avg 4-6): 25% at L4-7
                    # Phase 3 (avg 6+): 40% at L6-8
                    avg_lvl = float(np.mean(levels)) if levels else 1.0
                    if avg_lvl >= 6.0 and random.random() < 0.40:
                        sl = random.choice([6, 7, 7, 7, 8, 8])
                    elif avg_lvl >= 4.0 and random.random() < 0.25:
                        sl = random.choice([4, 5, 5, 6, 6, 7])
                    elif random.random() < 0.10:
                        sl = random.choice([3, 4, 4, 5])
                if sl is not None and sl > 1:
                    raw_state = envs.reset_one_at_level(i, sl)
                    env_start_levels[i] = sl
                else:
                    raw_state = envs.reset_one(i)
                    env_start_levels[i] = 1
                if frame_stack:
                    next_states[i] = frame_stack.reset(i, raw_state)
                else:
                    next_states[i] = raw_state
                # Reset GRU hidden state for this env on episode end
                if use_gru:
                    gru_hx[i] = 0.0

            # Update observation normalization
            if obs_rms:
                obs_rms.update(next_states)
                states = obs_rms.normalize(next_states)
            else:
                states = next_states

            if peak_detected and auto_stop:
                break

        if peak_detected and auto_stop:
            break

        # === Compute GAE ===
        with torch.no_grad():
            states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
            next_value = agent.get_value(states_t, gru_hx).cpu().numpy()

        advantages, returns = compute_gae(
            roll_rewards, roll_values, roll_dones, next_value,
            gamma=cfg.gamma, lam=cfg.gae_lambda)

        # === PPO Update ===
        # Update learning rate
        update_count += 1
        lr = scheduled_lr(update_count, cfg.lr, cfg.lr_min,
                         cfg.lr_warmup_updates, cfg.lr_decay_updates)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Entropy decay: linearly from entropy_coeff → entropy_coeff_end
        if cfg.entropy_coeff_end < cfg.entropy_coeff and cfg.lr_decay_updates > 0:
            progress = min(update_count / cfg.lr_decay_updates, 1.0)
            current_entropy_coeff = cfg.entropy_coeff + progress * (cfg.entropy_coeff_end - cfg.entropy_coeff)
        else:
            current_entropy_coeff = cfg.entropy_coeff

        if use_gru:
            # Chunk-based PPO update for GRU: sequential chunks within each env
            losses = ppo_update_chunked(
                agent, optimizer, roll_obs, roll_actions, roll_log_probs,
                returns, advantages, roll_hx, roll_dones,
                cfg, device, entropy_coeff_override=current_entropy_coeff,
                chunk_length=cfg.chunk_length)
        else:
            # Standard flat PPO update (no GRU)
            n_transitions = cfg.rollout_length * num_envs
            obs_flat = roll_obs.reshape(n_transitions, state_size)
            actions_flat = roll_actions.reshape(n_transitions)
            log_probs_flat = roll_log_probs.reshape(n_transitions)
            returns_flat = returns.reshape(n_transitions).astype(np.float32)
            advantages_flat = advantages.reshape(n_transitions).astype(np.float32)
            losses = ppo_update(agent, optimizer, obs_flat, actions_flat, log_probs_flat,
                               returns_flat, advantages_flat, cfg, device,
                               entropy_coeff_override=current_entropy_coeff,
                               sil_buffer=sil_buffer)

        # Periodic checkpoint
        if update_count % 100 == 0:
            save_checkpoint(agent, optimizer, obs_rms, update_count,
                          episode_count, best_avg_score, best_avg_episode, cfg,
                          os.path.join(save_dir, f"model_ppo_ep{episode_count}.pt"), curriculum)
            cur_min = curriculum.min_level if curriculum else 1
            kl_str = f", kl={losses['kl']:.4f}{'!' if losses['kl_stopped'] else ''}"
            sil_str = f", sil={sil_buffer.stats()}" if sil_buffer else ""
            print(f"  -> Checkpoint: update={update_count}, lr={lr:.2e}, "
                  f"ploss={losses['policy_loss']:.4f}, vloss={losses['value_loss']:.4f}, "
                  f"entropy={losses['entropy']:.4f}{kl_str}{sil_str}")

    # Final save
    save_checkpoint(agent, optimizer, obs_rms, update_count,
                  episode_count, best_avg_score, best_avg_episode, cfg,
                  os.path.join(save_dir, "model_ppo_final.pt"), curriculum)

    # Export JSON weights for browser
    export_ppo_to_json(agent, cfg, os.path.join(save_dir, "model_weights.json"), obs_rms)

    log_file.close()

    elapsed = time.time() - start_time
    avg_score_final = float(np.mean(scores)) if scores else 0
    print("\n" + "=" * 70)
    print(f"TRAINING {'PEAKED' if peak_detected else 'COMPLETE'}")
    print("=" * 70)
    print(f"Episodes:    {episode_count:,}")
    print(f"Updates:     {update_count}")
    print(f"Time:        {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"Best Score:  {best_score:,}")
    print(f"Best Level:  {best_level}")
    print(f"Best Avg:    {best_avg_score:.0f} (ep {best_avg_episode})")
    print(f"Avg Score (last 1k): {avg_score_final:.0f}")
    print(f"\nModel saved to: {save_dir}/model_ppo_final.pt")
    print(f"JSON weights:    {save_dir}/model_weights.json")


def save_checkpoint(agent, optimizer, obs_rms, update_count, episode_count,
                   best_avg_score, best_avg_episode, cfg, path, curriculum=None):
    agent_mod = agent._orig_mod if hasattr(agent, '_orig_mod') else agent
    data = {
        'agent': agent_mod.state_dict(),
        'optimizer': optimizer.state_dict(),
        'update_count': update_count,
        'total_episodes': episode_count,
        'best_avg_score': best_avg_score,
        'best_avg_episode': best_avg_episode,
        'config': {
            'hidden_sizes': cfg.hidden_sizes,
            'head_hidden': cfg.head_hidden,
            'n_frames': cfg.n_frames,
            'gru_hidden': cfg.gru_hidden,
            'lr': cfg.lr,
            'lr_min': cfg.lr_min,
            'lr_decay_updates': cfg.lr_decay_updates,
            'clip_epsilon': cfg.clip_epsilon,
            'entropy_coeff': cfg.entropy_coeff,
            'entropy_coeff_end': cfg.entropy_coeff_end,
            'rollout_length': cfg.rollout_length,
            'gamma': cfg.gamma,
            'gae_lambda': cfg.gae_lambda,
            'value_coeff': cfg.value_coeff,
            'target_kl': cfg.target_kl,
        },
    }
    if obs_rms:
        data['obs_rms'] = obs_rms.state_dict()
    if curriculum:
        data['curriculum'] = curriculum.state_dict()
    torch.save(data, path)


def export_ppo_to_json(agent, cfg, path, obs_rms=None):
    """Export PPO policy network to JSON for browser inference."""
    net = agent._orig_mod if hasattr(agent, '_orig_mod') else agent

    weights = {}
    for name, param in net.named_parameters():
        # Only export backbone + policy_head (skip value_head)
        if 'value_head' in name:
            continue
        weights[name] = param.detach().cpu().numpy().tolist()

    # Architecture: input → backbone layers → policy head → output
    arch = [net.state_size] + net.hidden_sizes + [net.head_hidden, net.action_size]

    data = {
        "architecture": arch,
        "activation": "relu",
        "type": "actor_critic",
        "level_conditioned": True,
        "n_frames": cfg.n_frames,
        "weights": weights,
    }

    # Export observation normalization stats for browser inference
    if obs_rms:
        data["obs_norm"] = {
            "mean": obs_rms.mean.tolist(),
            "std": np.sqrt(obs_rms.var + 1e-8).tolist(),
        }

    with open(path, 'w') as f:
        json.dump(data, f)

    size_kb = os.path.getsize(path) // 1024
    print(f"Exported PPO policy → {path} ({size_kb} KB)")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for eyeBMinvaders")
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--no-auto-scale", action="store_true")
    parser.add_argument("--auto-stop", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--no-obs-norm", action="store_true")
    parser.add_argument("--no-gru", action="store_true",
                        help="Disable GRU side-channel (feedforward only)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable aggressive curriculum learning")
    parser.add_argument("--no-sil", action="store_true",
                        help="Disable self-imitation learning")
    parser.add_argument("--sil-min-level", type=int, default=None,
                        help="Minimum level for SIL episode storage (default: 5)")
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.lr is not None:
        cfg.lr = args.lr
    if args.rollout_length is not None:
        cfg.rollout_length = args.rollout_length
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs
    if args.clip_epsilon is not None:
        cfg.clip_epsilon = args.clip_epsilon
    if args.entropy_coeff is not None:
        cfg.entropy_coeff = args.entropy_coeff
    if args.no_obs_norm:
        cfg.obs_norm = False
    if args.no_gru:
        cfg.gru_hidden = 0
    if args.no_curriculum:
        cfg.use_curriculum = False
    if args.no_sil:
        cfg.sil_weight = 0
    if args.sil_min_level is not None:
        cfg.sil_min_level = args.sil_min_level

    train_ppo(
        episodes=args.episodes,
        resume_path=args.resume,
        save_dir=args.save_dir,
        device_override=args.device,
        num_envs=args.num_envs if args.num_envs is not None else NUM_ENVS,
        config=cfg,
        auto_scale=not args.no_auto_scale,
        auto_stop=args.auto_stop,
    )
