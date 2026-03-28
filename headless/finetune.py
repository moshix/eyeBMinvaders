#!/usr/bin/env python3
"""
Browser-as-Environment Fine-Tuning for eyeBMinvaders
=====================================================
Runs game.js headlessly via Node.js subprocess, collects real game
experiences, and fine-tunes the DQN model against actual JS game physics.
Closes the sim-to-real gap between Rust training and browser inference.

Usage:
    python headless/finetune.py                          # Fine-tune from best model
    python headless/finetune.py --model models/model_ep50000.pt
    python headless/finetune.py --episodes 5000 --lr 1e-5
    python headless/finetune.py --export                 # Export after fine-tuning

Requirements:
    pip install torch numpy
    node (for headless game runner)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

# Ensure we can import from parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    sys.exit(1)


# =============================================================================
# Node.js Headless Game Environment
# =============================================================================
class HeadlessJSEnv:
    """Runs game.js in a Node.js subprocess, communicates via stdin/stdout JSON."""

    def __init__(self, game_root, fps=30):
        self.game_root = game_root
        self.fps = fps
        self.dt_ms = 1000.0 / fps
        self.proc = None
        self.state_size = 50
        self.action_size = 6
        self.n_frames = 4
        self.stacked_size = self.state_size * self.n_frames
        self._start_node()

    def _start_node(self):
        """Start a persistent Node.js process running the game."""
        script = os.path.join(os.path.dirname(__file__), 'env_worker.js')
        self.proc = subprocess.Popen(
            ['node', script, self.game_root, str(self.fps)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Wait for ready signal
        line = self.proc.stdout.readline().strip()
        if not line.startswith('READY'):
            err = self.proc.stderr.read()
            raise RuntimeError(f"Node env failed to start: {line} {err}")

    def reset(self):
        """Reset game, return initial stacked state."""
        self._send('RESET')
        return self._recv_state()

    def step(self, action):
        """Apply action, advance one frame, return (state, reward, done, info)."""
        self._send(f'STEP {action}')
        return self._recv_step()

    def _send(self, msg):
        self.proc.stdin.write(msg + '\n')
        self.proc.stdin.flush()

    def _recv_state(self):
        line = self.proc.stdout.readline().strip()
        data = json.loads(line)
        return np.array(data['state'], dtype=np.float32)

    def _recv_step(self):
        line = self.proc.stdout.readline().strip()
        data = json.loads(line)
        state = np.array(data['state'], dtype=np.float32)
        return state, data['reward'], data['done'], data['info']

    def close(self):
        if self.proc:
            self._send('QUIT')
            self.proc.terminate()
            self.proc = None


# =============================================================================
# Simple Replay Buffer
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size


# =============================================================================
# Fine-tuning Loop
# =============================================================================
def finetune(model_path, episodes=2000, lr=1e-5, batch_size=128,
             gamma=0.99, tau=0.005, buffer_size=100_000, save_dir='models',
             export_after=True, fps=30):

    game_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = 'cpu'  # headless fine-tuning is CPU-bound anyway

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Detect architecture from checkpoint keys
    state_dict = checkpoint['policy_net']
    keys = list(state_dict.keys())
    # Strip _orig_mod. prefix if present
    clean_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('_orig_mod.', '')
        clean_dict[clean_k] = v

    first_weight = clean_dict[list(clean_dict.keys())[0]]
    input_size = first_weight.shape[1]
    print(f"Model input size: {input_size}")

    # Determine architecture
    is_dueling = any('value_hidden' in k for k in clean_dict)
    is_noisy = any('weight_mu' in k for k in clean_dict)

    # Build network matching the checkpoint
    from train import DuelingDQN, DQN, NoisyLinear, TrainingConfig

    cfg = TrainingConfig()
    if is_dueling:
        policy_net = DuelingDQN(input_size, 6, cfg.hidden_sizes,
                                use_noisy=is_noisy).to(device)
        target_net = DuelingDQN(input_size, 6, cfg.hidden_sizes,
                                use_noisy=is_noisy).to(device)
    else:
        policy_net = DQN(input_size, 6, cfg.hidden_sizes).to(device)
        target_net = DQN(input_size, 6, cfg.hidden_sizes).to(device)

    # Load weights
    policy_net.load_state_dict(clean_dict, strict=False)
    target_net.load_state_dict(clean_dict, strict=False)
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, input_size)

    print(f"Architecture: {'Dueling' if is_dueling else 'Standard'}"
          f"{'+NoisyNet' if is_noisy else ''}, input={input_size}")
    print(f"Fine-tuning: {episodes} episodes, lr={lr}, batch={batch_size}")
    print(f"Starting headless JS environment...\n")

    env = HeadlessJSEnv(game_root, fps=fps)
    scores = deque(maxlen=100)
    levels = deque(maxlen=100)
    best_score = 0
    train_steps = 0
    start_time = time.time()

    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_score = 0
        steps = 0

        while True:
            steps += 1

            # Select action (greedy with small epsilon for stability)
            epsilon = max(0.02, 0.05 - ep * 0.00005)
            if np.random.random() < epsilon:
                action = np.random.randint(0, 6)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            episode_score = info.get('score', 0)
            state = next_state

            # Train every 4 steps
            if len(buffer) >= batch_size and steps % 4 == 0:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_t = torch.tensor(states, device=device)
                actions_t = torch.tensor(actions, device=device, dtype=torch.long)
                rewards_t = torch.tensor(rewards, device=device)
                next_states_t = torch.tensor(next_states, device=device)
                dones_t = torch.tensor(dones, device=device)

                # Double DQN
                with torch.no_grad():
                    next_actions = policy_net(next_states_t).argmax(dim=1)
                    next_q = target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_t + gamma * next_q * (1 - dones_t)

                current_q = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                train_steps += 1

                # Soft target update
                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)

            if done or steps > 30000:
                break

        scores.append(episode_score)
        levels.append(info.get('level', 1))

        if episode_score > best_score:
            best_score = episode_score
            # Save best
            save_path = os.path.join(save_dir, 'model_browser_best.pt')
            torch.save({
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'episode': ep,
                'best_score': best_score,
                'epsilon': epsilon,
                'steps': train_steps,
            }, save_path)

        if (ep + 1) % 10 == 0 or ep < 5:
            elapsed = time.time() - start_time
            avg_score = np.mean(scores) if scores else 0
            avg_level = np.mean(levels) if levels else 0
            eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
            print(f"Ep {ep+1:5d} | Avg Score: {avg_score:>7.0f} | Best: {best_score:>7,} | "
                  f"Avg Lvl: {avg_level:.1f} | Train: {train_steps} | "
                  f"{eps_per_sec:.1f} ep/s | {elapsed:.0f}s")

    env.close()

    # Save final model
    final_path = os.path.join(save_dir, 'model_browser_final.pt')
    torch.save({
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'episode': episodes,
        'best_score': best_score,
        'steps': train_steps,
    }, final_path)
    print(f"\nSaved final model: {final_path}")

    # Export to JSON for browser
    if export_after:
        print("Exporting to model_weights.json...")
        from export_model import export_model_to_json
        export_model_to_json(final_path)

    return best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune DQN in headless JS game')
    parser.add_argument('--model', default='models/model_ep50000.pt', help='Model to fine-tune')
    parser.add_argument('--episodes', type=int, default=2000, help='Fine-tuning episodes')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--fps', type=int, default=30, help='Game FPS')
    parser.add_argument('--export', action='store_true', help='Export model after fine-tuning')
    args = parser.parse_args()

    finetune(
        model_path=args.model,
        episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        fps=args.fps,
        export_after=args.export,
    )
