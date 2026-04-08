#!/usr/bin/env python3
"""
Explorer Training System for eyeBMinvaders
===========================================
Plateau-breaking through divergent experimentation.

Unlike meta_train.py which only tweaks hyperparameters, this system tries
fundamentally different STRATEGIES when a plateau is detected:
- Different reward functions (survival vs aggression vs evasion)
- Different network architectures (wide/narrow/deep/bottleneck)
- Different algorithms (DQN vs PPO)
- Different curricula (challenge-only, progressive, reverse)
- Different exploration strategies
- Different replay buffer configurations
- Different starting points (from scratch, perturbed weights, early checkpoints)

Each experiment is evaluated with a fail-fast protocol: bad ideas are killed
in seconds, good ideas are adopted as the new baseline.

Usage:
    python explorer_train.py                              # Default: 500K episodes
    python explorer_train.py --episodes 200000            # Shorter
    python explorer_train.py --experiments-per-plateau 8   # More experiments
    python explorer_train.py --fail-fast-budget 5000       # Shorter experiment budget
    python explorer_train.py --resume                      # Continue from state
    python explorer_train.py --resume-from models/model_best.pt
    python explorer_train.py --dry-run                     # Print specs only
"""

import argparse
import json
import os
import random
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from train import (
    train, TrainingConfig, PlateauDetector, CurriculumScheduler,
    NUM_ENVS, DQNAgent
)


# =============================================================================
# Axis Definitions — the "different angles"
# =============================================================================

REWARD_MODES = {
    "balanced": "Use Rust rewards as-is",
    "survival": "Amplify life-loss penalty, boost survival tick, zero kill bonuses",
    "aggressive": "Amplify kill rewards, reduce life-loss penalty",
    "evasion": "Heavy penalty for getting hit, amplify near-miss signals",
    "level_rush": "Massive level-complete bonus, minimal per-tick reward",
    "sparse": "Only +1 level complete, -1 game over, nothing else",
}

ARCHITECTURES = {
    "standard": [512, 256, 128],
    "narrow_deep": [128, 128, 128, 64],
    "wide_shallow": [1024, 256],
    "bottleneck": [512, 64, 512, 128],
    "small": [128, 64],
    "large": [1024, 512, 256, 128],
}

ALGORITHMS = ["dqn", "ppo"]

CURRICULUM_MODES = {
    "standard": "50/30/20 default split",
    "fundamentals_only": "100% level 1",
    "challenge_only": "100% at plateau level",
    "progressive": "Linear ramp from level 1 to max",
    "reverse": "70% challenge, 20% practice, 10% fundamentals",
    "random_uniform": "Uniform random level 1..max",
}

EXPLORATION_MODES = {
    "noisy_net": "NoisyNet (state-dependent exploration)",
    "epsilon_high": "Epsilon-greedy starting at 0.5, slow decay",
    "epsilon_low": "Epsilon-greedy starting at 0.1, fast decay",
    "epsilon_cyclic": "Epsilon resets to 0.3 every 2K episodes",
}

BUFFER_MODES = {
    "standard_dual": "DualReplayBuffer 70/30 split",
    "uniform_only": "FastReplayBuffer, no prioritization",
    "aggressive_important": "DualReplayBuffer 50/50, lower threshold",
    "fresh_buffer": "Flush buffer at experiment start",
    "small_buffer": "100K capacity for faster turnover",
    "large_buffer": "1M capacity for more diversity",
}

STARTING_POINTS = {
    "from_best": "Resume from best model checkpoint",
    "from_scratch": "Random initialization",
    "from_early": "Resume from an early checkpoint",
    "from_perturbed": "Best model + Gaussian noise on weights",
}


# =============================================================================
# ExperimentSpec
# =============================================================================

@dataclass
class ExperimentSpec:
    """Complete specification for one exploration experiment."""
    experiment_id: str
    reward_mode: str = "balanced"
    hidden_sizes: list = field(default_factory=lambda: [512, 256, 128])
    use_dueling: bool = True
    use_noisy: bool = True
    use_layer_norm: bool = False
    algorithm: str = "dqn"
    curriculum_mode: str = "standard"
    exploration_mode: str = "noisy_net"
    buffer_mode: str = "standard_dual"
    starting_point: str = "from_best"
    lr: float = 1e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_step: int = 5
    train_steps_per_tick: int = 4
    max_episodes: int = 7500
    checkpoint_path_hint: Optional[str] = None
    generation: int = 0
    creation_reason: str = ""

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def axis_vector(self):
        """Return a hashable tuple representing this spec's axis choices."""
        return (
            self.reward_mode,
            tuple(self.hidden_sizes),
            self.use_dueling,
            self.use_noisy,
            self.algorithm,
            self.curriculum_mode,
            self.exploration_mode,
            self.buffer_mode,
            self.starting_point,
        )


@dataclass
class ExperimentResult:
    """Result of running one experiment."""
    spec: dict
    avg_score: float = 0.0
    best_score: int = 0
    avg_level: float = 0.0
    best_level: int = 0
    episodes_completed: int = 0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""
    score_trajectory: list = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    learning_velocity: float = 0.0

    def to_dict(self):
        return asdict(self)


# =============================================================================
# Reward Transformer
# =============================================================================

class RewardTransformer:
    """Post-processes rewards from the Rust sim based on reward_mode."""

    def __init__(self, mode: str):
        self.mode = mode

    def transform(self, rewards: np.ndarray, dones: np.ndarray,
                  envs) -> np.ndarray:
        """Transform reward array. Called after step_all_fast."""
        if self.mode == "balanced":
            return rewards

        rewards = rewards.copy()

        if self.mode == "survival":
            # Amplify life-loss penalties (large negative rewards)
            mask_life_loss = rewards < -2.0
            rewards[mask_life_loss] *= 3.0
            # Zero out kill bonuses (moderate positive rewards)
            mask_kill = (rewards > 0.5) & (rewards < 10.0)
            rewards[mask_kill] = 0.0
            # Boost survival tick for alive agents
            rewards[~dones.astype(bool)] += 0.03

        elif self.mode == "aggressive":
            # Amplify kill rewards
            mask_kill = rewards > 0.5
            rewards[mask_kill] *= 3.0
            # Reduce life-loss penalty
            mask_life_loss = rewards < -2.0
            rewards[mask_life_loss] *= 0.3
            # Small bonus for any positive action
            mask_pos = rewards > 0.0
            rewards[mask_pos] += 0.1

        elif self.mode == "evasion":
            # Heavy penalty for getting hit (large negatives)
            mask_hit = rewards < -1.0
            rewards[mask_hit] *= 4.0
            # Reward for surviving near threats (small negatives = near misses)
            mask_near = (rewards < 0.0) & (rewards >= -1.0)
            rewards[mask_near] = 0.05  # near-miss survival bonus
            # Extra survival tick
            rewards[~dones.astype(bool)] += 0.02

        elif self.mode == "level_rush":
            # Massive bonus for level completion (detected as large positive + done or > 5)
            mask_level = rewards > 4.0
            rewards[mask_level] *= 10.0
            # Minimal per-tick reward
            mask_tick = (rewards > -0.5) & (rewards < 0.5)
            rewards[mask_tick] *= 0.1

        elif self.mode == "sparse":
            # Only +1 for level complete, -1 for game over
            new_rewards = np.zeros_like(rewards)
            new_rewards[rewards > 4.0] = 1.0  # level complete
            new_rewards[dones.astype(bool) & (rewards <= 4.0)] = -1.0  # game over
            rewards = new_rewards

        return rewards


# =============================================================================
# Custom Curriculum Modes
# =============================================================================

class FundamentalsCurriculum:
    """100% level 1 — master the basics."""
    def sample(self, levels_deque, episode_count):
        return 1


class ChallengeCurriculum:
    """100% at the target (plateau) level."""
    def __init__(self, target_level):
        self.target_level = max(1, target_level)

    def sample(self, levels_deque, episode_count):
        return self.target_level


class ProgressiveCurriculum:
    """Linear ramp from level 1 to max_level over total_episodes."""
    def __init__(self, max_level, total_episodes):
        self.max_level = max(1, max_level)
        self.total_episodes = max(1, total_episodes)

    def sample(self, levels_deque, episode_count):
        progress = min(1.0, episode_count / self.total_episodes)
        return max(1, int(progress * self.max_level))


class ReverseCurriculum:
    """70% challenge, 20% practice, 10% fundamentals."""
    def __init__(self, max_level):
        self.max_level = max(1, max_level)

    def sample(self, levels_deque, episode_count):
        r = random.random()
        if r < 0.70:
            return self.max_level
        elif r < 0.90:
            return max(1, self.max_level - 1)
        else:
            return 1


class RandomUniformCurriculum:
    """Uniform random level from 1 to max_level."""
    def __init__(self, max_level):
        self.max_level = max(1, max_level)

    def sample(self, levels_deque, episode_count):
        return random.randint(1, self.max_level)


def build_curriculum(mode: str, plateau_context: dict):
    """Build a curriculum object from mode string and context."""
    max_level = plateau_context.get("best_level", 3)
    max_episodes = plateau_context.get("fail_fast_budget", 7500)

    if mode == "standard":
        return CurriculumScheduler()
    elif mode == "fundamentals_only":
        return FundamentalsCurriculum()
    elif mode == "challenge_only":
        return ChallengeCurriculum(max_level)
    elif mode == "progressive":
        return ProgressiveCurriculum(max_level, max_episodes)
    elif mode == "reverse":
        return ReverseCurriculum(max_level)
    elif mode == "random_uniform":
        return RandomUniformCurriculum(max_level)
    else:
        return CurriculumScheduler()


# =============================================================================
# Diversity Tracker
# =============================================================================

class DiversityTracker:
    """Ensures experiments stay diverse using Hamming distance in strategy space."""

    # All axis values for one-hot encoding
    AXES = {
        "reward_mode": list(REWARD_MODES.keys()),
        "architecture": list(ARCHITECTURES.keys()),
        "algorithm": ALGORITHMS,
        "curriculum_mode": list(CURRICULUM_MODES.keys()),
        "exploration_mode": list(EXPLORATION_MODES.keys()),
        "buffer_mode": list(BUFFER_MODES.keys()),
        "starting_point": list(STARTING_POINTS.keys()),
    }

    def _spec_to_keys(self, spec: ExperimentSpec) -> dict:
        arch_key = None
        for name, sizes in ARCHITECTURES.items():
            if spec.hidden_sizes == sizes:
                arch_key = name
                break
        if arch_key is None:
            arch_key = "custom"
        return {
            "reward_mode": spec.reward_mode,
            "architecture": arch_key,
            "algorithm": spec.algorithm,
            "curriculum_mode": spec.curriculum_mode,
            "exploration_mode": spec.exploration_mode,
            "buffer_mode": spec.buffer_mode,
            "starting_point": spec.starting_point,
        }

    def distance(self, spec_a: ExperimentSpec, spec_b: ExperimentSpec) -> float:
        """Normalized Hamming distance (0=identical axes, 1=all different)."""
        keys_a = self._spec_to_keys(spec_a)
        keys_b = self._spec_to_keys(spec_b)
        diffs = sum(1 for k in keys_a if keys_a[k] != keys_b[k])
        return diffs / len(keys_a)

    def filter_for_diversity(self, candidates: list, min_distance: float = 0.3) -> list:
        """Greedily select candidates maintaining minimum pairwise distance."""
        if not candidates:
            return []
        selected = [candidates[0]]
        for c in candidates[1:]:
            if all(self.distance(c, s) >= min_distance for s in selected):
                selected.append(c)
        return selected

    def coverage_report(self, results: list) -> dict:
        """Which axis values have been tried."""
        coverage = {axis: {} for axis in self.AXES}
        for r in results:
            spec = ExperimentSpec.from_dict(r["spec"]) if isinstance(r, dict) else r.spec
            if isinstance(spec, dict):
                spec = ExperimentSpec.from_dict(spec)
            keys = self._spec_to_keys(spec)
            for axis, val in keys.items():
                coverage[axis][val] = coverage[axis].get(val, 0) + 1
        return coverage


# =============================================================================
# Strategy Generator
# =============================================================================

class StrategyGenerator:
    """Creates diverse experiment specifications.

    Uses three generation strategies:
    1. Single-axis radical change — vary one axis dramatically, keep rest default
    2. Random composition — sample freely from all axes
    3. Informed — exploit axis success rates from history
    """

    def __init__(self, axis_stats: dict = None):
        self.axis_stats = axis_stats or {}
        self.diversity = DiversityTracker()

    def generate_batch(self, n: int, plateau_context: dict,
                       available_checkpoints: list,
                       generation: int = 0) -> list:
        """Generate n diverse experiments for the current plateau."""
        specs = []

        # 1. Always include a radical departure (from scratch + different algo or reward)
        specs.append(self._radical_departure(plateau_context, generation))

        # 2. Single-axis variants (change one thing dramatically)
        n_single = max(1, n // 3)
        specs.extend(self._single_axis_variants(n_single, plateau_context,
                                                 available_checkpoints, generation))

        # 3. Random compositions
        n_random = max(1, n // 3)
        specs.extend(self._random_compositions(n_random, plateau_context,
                                                available_checkpoints, generation))

        # 4. Informed by history (if we have enough data)
        remaining = n - len(specs)
        if remaining > 0 and self.axis_stats:
            specs.extend(self._informed_specs(remaining, plateau_context,
                                              available_checkpoints, generation))
        elif remaining > 0:
            specs.extend(self._random_compositions(remaining, plateau_context,
                                                    available_checkpoints, generation))

        # Deduplicate and ensure diversity
        specs = self.diversity.filter_for_diversity(specs, min_distance=0.25)

        # If diversity filtering removed too many, fill with random
        while len(specs) < n:
            extra = self._random_compositions(1, plateau_context,
                                               available_checkpoints, generation)
            specs.extend(extra)

        return specs[:n]

    def _make_id(self, tag: str) -> str:
        return f"exp_{tag}_{uuid.uuid4().hex[:6]}"

    def _base_spec(self, generation: int) -> dict:
        """Default axis values."""
        return {
            "generation": generation,
            "reward_mode": "balanced",
            "hidden_sizes": [512, 256, 128],
            "use_dueling": True,
            "use_noisy": True,
            "use_layer_norm": False,
            "algorithm": "dqn",
            "curriculum_mode": "standard",
            "exploration_mode": "noisy_net",
            "buffer_mode": "standard_dual",
            "starting_point": "from_best",
            "lr": 1e-4,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "n_step": 5,
            "train_steps_per_tick": 4,
        }

    def _radical_departure(self, ctx: dict, generation: int) -> ExperimentSpec:
        """A from-scratch experiment with completely different approach."""
        algo = random.choice(["dqn", "ppo"])
        reward = random.choice(["aggressive", "survival", "sparse"])
        arch_name = random.choice(["wide_shallow", "narrow_deep", "large"])
        arch = ARCHITECTURES[arch_name]

        return ExperimentSpec(
            experiment_id=self._make_id(f"radical_{algo}_{reward}"),
            reward_mode=reward,
            hidden_sizes=arch,
            use_dueling=random.choice([True, False]),
            use_noisy=False,  # force epsilon for fresh start
            algorithm=algo,
            curriculum_mode=random.choice(["challenge_only", "fundamentals_only"]),
            exploration_mode="epsilon_high",
            buffer_mode="fresh_buffer",
            starting_point="from_scratch",
            lr=random.choice([3e-4, 5e-4, 1e-3]),
            batch_size=random.choice([128, 256, 512]),
            gamma=random.choice([0.95, 0.99]),
            n_step=random.choice([3, 5]),
            generation=generation,
            creation_reason=f"Radical departure: {algo} + {reward} + {arch_name} from scratch",
        )

    def _single_axis_variants(self, n: int, ctx: dict,
                               checkpoints: list, generation: int) -> list:
        """Change one axis dramatically, keep everything else default."""
        axes_to_try = [
            ("reward_mode", ["survival", "aggressive", "evasion", "level_rush", "sparse"]),
            ("architecture", list(ARCHITECTURES.keys())),
            ("curriculum_mode", ["challenge_only", "fundamentals_only", "progressive", "reverse"]),
            ("starting_point", ["from_scratch", "from_perturbed", "from_early"]),
            ("buffer_mode", ["uniform_only", "small_buffer", "fresh_buffer"]),
            ("exploration_mode", ["epsilon_high", "epsilon_cyclic"]),
        ]
        random.shuffle(axes_to_try)
        specs = []

        for axis, values in axes_to_try[:n]:
            val = random.choice(values)
            base = self._base_spec(generation)

            if axis == "architecture":
                base["hidden_sizes"] = ARCHITECTURES[val]
                tag = f"arch_{val}"
            elif axis == "reward_mode":
                base["reward_mode"] = val
                tag = f"reward_{val}"
            elif axis == "curriculum_mode":
                base["curriculum_mode"] = val
                tag = f"curriculum_{val}"
            elif axis == "starting_point":
                base["starting_point"] = val
                if val == "from_early" and checkpoints:
                    base["checkpoint_path_hint"] = checkpoints[0]
                tag = f"start_{val}"
            elif axis == "buffer_mode":
                base["buffer_mode"] = val
                tag = f"buffer_{val}"
            elif axis == "exploration_mode":
                base["exploration_mode"] = val
                if val in ("epsilon_high", "epsilon_cyclic"):
                    base["use_noisy"] = False
                tag = f"explore_{val}"
            else:
                tag = f"single_{axis}_{val}"

            base["creation_reason"] = f"Single-axis change: {axis}={val}"
            base["experiment_id"] = self._make_id(tag)
            specs.append(ExperimentSpec(**base))

        return specs

    def _random_compositions(self, n: int, ctx: dict,
                              checkpoints: list, generation: int) -> list:
        """Randomly sample from all axes independently."""
        specs = []
        for _ in range(n):
            reward = random.choice(list(REWARD_MODES.keys()))
            arch_name = random.choice(list(ARCHITECTURES.keys()))
            arch = ARCHITECTURES[arch_name]
            algo = random.choice(ALGORITHMS)
            curric = random.choice(list(CURRICULUM_MODES.keys()))
            explore = random.choice(list(EXPLORATION_MODES.keys()))
            buf = random.choice(list(BUFFER_MODES.keys()))
            start = random.choice(list(STARTING_POINTS.keys()))

            use_noisy = explore == "noisy_net"

            spec = ExperimentSpec(
                experiment_id=self._make_id(f"rand_{reward[:3]}_{arch_name[:4]}"),
                reward_mode=reward,
                hidden_sizes=arch,
                use_dueling=random.choice([True, False]),
                use_noisy=use_noisy,
                use_layer_norm=random.choice([True, False]),
                algorithm=algo,
                curriculum_mode=curric,
                exploration_mode=explore,
                buffer_mode=buf,
                starting_point=start,
                lr=random.choice([1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3]),
                batch_size=random.choice([128, 256, 512, 1024]),
                gamma=random.choice([0.95, 0.97, 0.99, 0.995]),
                tau=random.choice([0.001, 0.005, 0.01]),
                n_step=random.choice([1, 3, 5, 7]),
                train_steps_per_tick=random.choice([2, 4, 8]),
                generation=generation,
                checkpoint_path_hint=checkpoints[0] if checkpoints and start != "from_scratch" else None,
                creation_reason=f"Random: {reward}/{arch_name}/{algo}/{curric}/{start}",
            )
            specs.append(spec)
        return specs

    def _informed_specs(self, n: int, ctx: dict,
                         checkpoints: list, generation: int) -> list:
        """Generate specs biased toward axis values that worked before."""
        specs = []
        for _ in range(n):
            base = self._base_spec(generation)

            # For each axis, pick the value with highest success rate if available
            for axis_name, stats in self.axis_stats.items():
                if not stats:
                    continue
                # Weighted random by success rate
                values = list(stats.keys())
                weights = [max(0.01, stats[v]) for v in values]
                total = sum(weights)
                weights = [w / total for w in weights]
                chosen = np.random.choice(values, p=weights)

                if axis_name == "reward_mode":
                    base["reward_mode"] = chosen
                elif axis_name == "architecture":
                    if chosen in ARCHITECTURES:
                        base["hidden_sizes"] = ARCHITECTURES[chosen]
                elif axis_name == "algorithm":
                    base["algorithm"] = chosen
                elif axis_name == "curriculum_mode":
                    base["curriculum_mode"] = chosen
                elif axis_name == "exploration_mode":
                    base["exploration_mode"] = chosen
                    if chosen != "noisy_net":
                        base["use_noisy"] = False
                elif axis_name == "buffer_mode":
                    base["buffer_mode"] = chosen
                elif axis_name == "starting_point":
                    base["starting_point"] = chosen

            # Add some random mutation to prevent exact repeats
            if random.random() < 0.5:
                base["lr"] = random.choice([5e-5, 1e-4, 3e-4])
            if random.random() < 0.3:
                base["batch_size"] = random.choice([256, 512, 1024])

            base["experiment_id"] = self._make_id("informed")
            base["creation_reason"] = "Informed by past successes"
            if checkpoints and base["starting_point"] != "from_scratch":
                base["checkpoint_path_hint"] = checkpoints[0]
            specs.append(ExperimentSpec(**base))

        return specs


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs a single experiment with fail-fast evaluation."""

    CHECKPOINT_INTERVAL = 2500

    def run(self, spec: ExperimentSpec, baseline_score: float,
            save_dir: str, device: str, num_envs: int,
            plateau_context: dict) -> ExperimentResult:
        """Run experiment with fail-fast. Returns ExperimentResult."""
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()

        try:
            if spec.algorithm == "ppo":
                return self._run_ppo(spec, baseline_score, save_dir,
                                      device, num_envs, plateau_context, start_time)
            else:
                return self._run_dqn(spec, baseline_score, save_dir,
                                      device, num_envs, plateau_context, start_time)
        except Exception as e:
            return ExperimentResult(
                spec=spec.to_dict(),
                stop_reason=f"error: {e}",
                elapsed_seconds=time.time() - start_time,
            )

    def _run_dqn(self, spec, baseline_score, save_dir, device, num_envs,
                  plateau_context, start_time):
        """Run DQN experiment in chunks with fail-fast."""
        config = self._build_config(spec)
        resume_path = self._resolve_starting_point(spec, save_dir)
        reward_transform = None
        if spec.reward_mode != "balanced":
            transformer = RewardTransformer(spec.reward_mode)
            reward_transform = transformer.transform

        curriculum = build_curriculum(spec.curriculum_mode, plateau_context)

        episodes_run = 0
        score_trajectory = []
        best_score = 0
        best_level = 0
        last_avg = 0.0

        while episodes_run < spec.max_episodes:
            chunk = min(self.CHECKPOINT_INTERVAL, spec.max_episodes - episodes_run)

            result = train(
                episodes=chunk,
                resume_path=resume_path,
                save_dir=save_dir,
                device_override=device,
                num_envs=num_envs,
                config=config,
                reward_transform=reward_transform,
                curriculum_override=curriculum,
                auto_scale=False,
                use_curriculum=True,
            )

            episodes_run += result["episodes_completed"]
            last_avg = result["avg_score"]
            score_trajectory.append(last_avg)
            best_score = max(best_score, result["best_score"])
            best_level = max(best_level, result["best_level"])

            # Resume from where we left off next chunk
            final_path = os.path.join(save_dir, "model_final.pt")
            if os.path.exists(final_path):
                resume_path = final_path

            # Success check
            if last_avg > baseline_score:
                return ExperimentResult(
                    spec=spec.to_dict(),
                    avg_score=last_avg,
                    best_score=best_score,
                    avg_level=result["avg_level"],
                    best_level=best_level,
                    episodes_completed=episodes_run,
                    elapsed_seconds=time.time() - start_time,
                    stop_reason="success",
                    score_trajectory=score_trajectory,
                    checkpoint_path=os.path.join(save_dir, "model_best.pt"),
                    learning_velocity=self._velocity(score_trajectory),
                )

            # Fail-fast check
            if self._should_abort(episodes_run, last_avg, baseline_score,
                                   spec.max_episodes):
                return ExperimentResult(
                    spec=spec.to_dict(),
                    avg_score=last_avg,
                    best_score=best_score,
                    avg_level=result["avg_level"],
                    best_level=best_level,
                    episodes_completed=episodes_run,
                    elapsed_seconds=time.time() - start_time,
                    stop_reason="fail_fast",
                    score_trajectory=score_trajectory,
                    learning_velocity=self._velocity(score_trajectory),
                )

        # Budget exhausted
        checkpoint = os.path.join(save_dir, "model_best.pt")
        return ExperimentResult(
            spec=spec.to_dict(),
            avg_score=last_avg,
            best_score=best_score,
            avg_level=result["avg_level"] if episodes_run > 0 else 0,
            best_level=best_level,
            episodes_completed=episodes_run,
            elapsed_seconds=time.time() - start_time,
            stop_reason="budget_exhausted",
            score_trajectory=score_trajectory,
            checkpoint_path=checkpoint if os.path.exists(checkpoint) else None,
            learning_velocity=self._velocity(score_trajectory),
        )

    def _run_ppo(self, spec, baseline_score, save_dir, device, num_envs,
                  plateau_context, start_time):
        """Run PPO experiment (single chunk, no reward transform support)."""
        try:
            from train_ppo import train_ppo
        except ImportError:
            return ExperimentResult(
                spec=spec.to_dict(),
                stop_reason="error: train_ppo not available",
                elapsed_seconds=time.time() - start_time,
            )

        result = train_ppo(
            episodes=spec.max_episodes,
            save_dir=save_dir,
            device_override=device,
            num_envs=num_envs,
            auto_scale=False,
        )

        # train_ppo returns None — parse results from saved files
        elapsed = time.time() - start_time
        avg_score, best_score_val, best_level_val, eps_completed = \
            self._parse_ppo_results(save_dir, spec.max_episodes)

        checkpoint = os.path.join(save_dir, "model_ppo_best_avg.pt")
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join(save_dir, "model_ppo_final.pt")

        return ExperimentResult(
            spec=spec.to_dict(),
            avg_score=avg_score,
            best_score=best_score_val,
            avg_level=0,
            best_level=best_level_val,
            episodes_completed=eps_completed,
            elapsed_seconds=elapsed,
            stop_reason="success" if avg_score > baseline_score else "budget_exhausted",
            score_trajectory=[avg_score],
            checkpoint_path=checkpoint if os.path.exists(checkpoint) else None,
        )

    def _parse_ppo_results(self, save_dir: str, max_episodes: int):
        """Parse PPO results from training log since train_ppo returns None."""
        # PPO uses training_events_ppo.jsonl, DQN uses training_events.jsonl
        log_path = os.path.join(save_dir, "training_events_ppo.jsonl")
        if not os.path.exists(log_path):
            log_path = os.path.join(save_dir, "training_events.jsonl")
        scores = []
        best_score = 0
        best_level = 0
        eps = 0
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                            s = ev.get("score", 0)
                            scores.append(s)
                            if s > best_score:
                                best_score = s
                            lv = ev.get("level", 0)
                            if lv > best_level:
                                best_level = lv
                            eps = ev.get("episode", eps)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        avg = float(np.mean(scores[-1000:])) if scores else 0
        return avg, best_score, best_level, eps

    def _build_config(self, spec: ExperimentSpec) -> TrainingConfig:
        """Build a TrainingConfig from an ExperimentSpec."""
        epsilon_start = 1.0
        epsilon_decay = 0.99995
        if spec.exploration_mode == "epsilon_high":
            epsilon_start = 0.5
            epsilon_decay = 0.9999
        elif spec.exploration_mode == "epsilon_low":
            epsilon_start = 0.1
            epsilon_decay = 0.999
        elif spec.exploration_mode == "epsilon_cyclic":
            epsilon_start = 0.3
            epsilon_decay = 0.9998

        buffer_capacity = 500_000
        use_dual = True
        important_ratio = 0.30
        important_threshold = 0.5
        if spec.buffer_mode == "uniform_only":
            use_dual = False
        elif spec.buffer_mode == "aggressive_important":
            important_ratio = 0.50
            important_threshold = 0.2
        elif spec.buffer_mode == "small_buffer":
            buffer_capacity = 100_000
        elif spec.buffer_mode == "large_buffer":
            buffer_capacity = 1_000_000

        return TrainingConfig(
            lr=spec.lr,
            batch_size=spec.batch_size,
            gamma=spec.gamma,
            tau=spec.tau,
            n_step=spec.n_step,
            hidden_sizes=list(spec.hidden_sizes),
            train_steps_per_tick=spec.train_steps_per_tick,
            use_dueling=spec.use_dueling,
            use_noisy=spec.use_noisy,
            use_layer_norm=spec.use_layer_norm,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            buffer_capacity=buffer_capacity,
            use_dual_buffer=use_dual,
            important_ratio=important_ratio,
            important_reward_threshold=important_threshold,
        )

    def _resolve_starting_point(self, spec: ExperimentSpec, save_dir: str):
        """Resolve starting_point axis to a file path."""
        if spec.starting_point == "from_scratch":
            return None
        elif spec.starting_point == "from_perturbed":
            if spec.checkpoint_path_hint and os.path.exists(spec.checkpoint_path_hint):
                return self._create_perturbed_checkpoint(
                    spec.checkpoint_path_hint, save_dir)
            return None
        elif spec.starting_point in ("from_best", "from_early"):
            if spec.checkpoint_path_hint and os.path.exists(spec.checkpoint_path_hint):
                return spec.checkpoint_path_hint
            return None
        return None

    def _create_perturbed_checkpoint(self, source_path: str, save_dir: str) -> str:
        """Load checkpoint, add Gaussian noise to weights."""
        if not HAS_TORCH:
            return source_path
        checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
        for key in ['policy_net', 'target_net']:
            if key not in checkpoint:
                continue
            for param_name, param in checkpoint[key].items():
                if param.is_floating_point():
                    noise_scale = 0.02 * (param.std().item() + 1e-8)
                    checkpoint[key][param_name] = param + torch.randn_like(param) * noise_scale
        perturbed_path = os.path.join(save_dir, "model_perturbed.pt")
        torch.save(checkpoint, perturbed_path)
        return perturbed_path

    def _should_abort(self, episodes_run: int, current_avg: float,
                       baseline: float, max_episodes: int) -> bool:
        """Fail-fast logic."""
        if baseline <= 0:
            return False
        progress = episodes_run / max_episodes
        ratio = current_avg / baseline
        if progress >= 0.33 and ratio < 0.50:
            return True
        if progress >= 0.66 and ratio < 0.80:
            return True
        return False

    def _velocity(self, trajectory: list) -> float:
        """Learning velocity = slope of score trajectory."""
        if len(trajectory) < 2:
            return 0.0
        x = np.arange(len(trajectory), dtype=float)
        y = np.array(trajectory, dtype=float)
        if y.std() == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)


# =============================================================================
# Explorer State (persistence)
# =============================================================================

class ExplorerState:
    """Persists all explorer state to disk."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.state_path = os.path.join(save_dir, "explorer_state.json")
        self.log_path = os.path.join(save_dir, "explorer_events.jsonl")
        self.generation = 0
        self.total_episodes = 0
        self.baseline_score = 0.0
        self.baseline_config = TrainingConfig().to_dict()
        self.best_checkpoint = None
        self.experiments = []
        self.axis_stats = {}
        self.experiments_per_plateau = 6

    def load(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                data = json.load(f)
            self.generation = data.get("generation", 0)
            self.total_episodes = data.get("total_episodes", 0)
            self.baseline_score = data.get("baseline_score", 0.0)
            self.baseline_config = data.get("baseline_config", TrainingConfig().to_dict())
            self.best_checkpoint = data.get("best_checkpoint")
            self.experiments = data.get("experiments", [])
            self.axis_stats = data.get("axis_stats", {})
            self.experiments_per_plateau = data.get("experiments_per_plateau", 6)

    def save(self):
        data = {
            "generation": self.generation,
            "total_episodes": self.total_episodes,
            "baseline_score": self.baseline_score,
            "baseline_config": self.baseline_config,
            "best_checkpoint": self.best_checkpoint,
            "experiments": self.experiments,
            "axis_stats": self.axis_stats,
            "experiments_per_plateau": self.experiments_per_plateau,
        }
        with open(self.state_path, 'w') as f:
            json.dump(data, f, indent=2)

    def log_event(self, event: dict):
        event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def log_experiment(self, result: ExperimentResult):
        self.experiments.append(result.to_dict())
        self.log_event({
            "event": "experiment_complete",
            "experiment_id": result.spec.get("experiment_id", "?"),
            "avg_score": result.avg_score,
            "stop_reason": result.stop_reason,
            "episodes": result.episodes_completed,
        })

    def adopt_winner(self, result: ExperimentResult):
        """Adopt winning experiment as new baseline."""
        spec = result.spec
        if isinstance(spec, dict):
            spec_obj = ExperimentSpec.from_dict(spec)
        else:
            spec_obj = spec

        # Update baseline config from winner's spec
        self.baseline_config = TrainingConfig(
            lr=spec_obj.lr,
            batch_size=spec_obj.batch_size,
            gamma=spec_obj.gamma,
            tau=spec_obj.tau,
            n_step=spec_obj.n_step,
            hidden_sizes=list(spec_obj.hidden_sizes),
            train_steps_per_tick=spec_obj.train_steps_per_tick,
            use_dueling=spec_obj.use_dueling,
            use_noisy=spec_obj.use_noisy,
            use_layer_norm=spec_obj.use_layer_norm,
        ).to_dict()

        self.baseline_score = result.avg_score
        if result.checkpoint_path and os.path.exists(result.checkpoint_path):
            self.best_checkpoint = result.checkpoint_path

        self.log_event({
            "event": "winner_adopted",
            "experiment_id": spec_obj.experiment_id,
            "new_baseline_score": result.avg_score,
            "checkpoint": result.checkpoint_path,
        })

    def update_axis_stats(self, results: list):
        """Update per-axis success rates based on experiment results."""
        diversity = DiversityTracker()
        for r in results:
            spec = r.spec if isinstance(r.spec, dict) else r.spec.to_dict()
            spec_obj = ExperimentSpec.from_dict(spec)
            keys = diversity._spec_to_keys(spec_obj)
            is_success = r.stop_reason == "success"
            score = 1.0 if is_success else 0.0

            for axis, value in keys.items():
                if axis not in self.axis_stats:
                    self.axis_stats[axis] = {}
                if value not in self.axis_stats[axis]:
                    self.axis_stats[axis][value] = 0.5  # prior
                # Exponential moving average
                self.axis_stats[axis][value] = (
                    0.7 * self.axis_stats[axis][value] + 0.3 * score
                )

    def find_checkpoints(self) -> list:
        """Find available model checkpoints."""
        checkpoints = []
        # Check common locations
        for d in [self.save_dir, "models"]:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".pt") and "best" in f:
                    checkpoints.append(os.path.join(d, f))
        # Also check generation subdirs
        for entry in sorted(os.listdir(self.save_dir)) if os.path.isdir(self.save_dir) else []:
            subdir = os.path.join(self.save_dir, entry)
            if os.path.isdir(subdir):
                best = os.path.join(subdir, "model_best.pt")
                if os.path.exists(best):
                    checkpoints.append(best)
        return checkpoints


# =============================================================================
# Explorer Loop — main orchestrator
# =============================================================================

class ExplorerLoop:
    """Main orchestrator for the exploration training system."""

    def __init__(self, save_dir="models/explorer", device=None, num_envs=None):
        self.save_dir = save_dir
        self.device = device
        self.num_envs = num_envs or NUM_ENVS
        self.state = ExplorerState(save_dir)
        self.diversity = DiversityTracker()

    def run(self, total_episodes=500_000, experiments_per_plateau=6,
            fail_fast_budget=7500, resume=False, resume_from=None,
            dry_run=False):
        """Main exploration loop."""
        os.makedirs(self.save_dir, exist_ok=True)

        if resume:
            self.state.load()
            print(f"Resumed from generation {self.state.generation}, "
                  f"{self.state.total_episodes:,} episodes trained")

        if resume_from and os.path.exists(resume_from):
            self.state.best_checkpoint = resume_from
            print(f"Starting from checkpoint: {resume_from}")

        self.state.experiments_per_plateau = experiments_per_plateau

        print("=" * 70)
        print("EXPLORER TRAINING SYSTEM")
        print("=" * 70)
        print(f"Total episode budget: {total_episodes:,}")
        print(f"Experiments per plateau: {experiments_per_plateau}")
        print(f"Fail-fast budget per experiment: {fail_fast_budget}")
        print(f"Parallel environments: {self.num_envs}")
        print(f"Save directory: {self.save_dir}")
        print("=" * 70)

        while self.state.total_episodes < total_episodes:
            self.state.generation += 1
            gen = self.state.generation

            print(f"\n{'#' * 70}")
            print(f"# GENERATION {gen}")
            print(f"# Episodes so far: {self.state.total_episodes:,} / {total_episodes:,}")
            print(f"{'#' * 70}")

            # --- Phase 1: Train until plateau ---
            baseline_config = TrainingConfig.from_dict(self.state.baseline_config)
            resume_path = self.state.best_checkpoint

            baseline_dir = os.path.join(self.save_dir, f"gen_{gen:03d}_baseline")

            detector = PlateauDetector(
                window=3000,
                min_episodes=8000,
                cooldown=5000,
                score_threshold=0.03,
            )

            remaining = total_episodes - self.state.total_episodes
            baseline_budget = min(50_000, remaining)

            print(f"\n--- Baseline Phase (up to {baseline_budget:,} episodes) ---")
            print(f"  Config: lr={baseline_config.lr:.2e}, "
                  f"arch={baseline_config.hidden_sizes}, "
                  f"dueling={baseline_config.use_dueling}")
            if resume_path:
                print(f"  Resuming from: {resume_path}")

            if dry_run:
                print("  [DRY RUN] Skipping baseline training")
                # Simulate a plateau for dry-run
                baseline_result = {
                    "episodes_completed": 10000,
                    "avg_score": self.state.baseline_score or 500,
                    "avg_level": 3.0,
                    "best_score": 2000,
                    "best_level": 5,
                    "elapsed": 0,
                    "stop_reason": "plateau",
                }
            else:
                baseline_result = train(
                    episodes=baseline_budget,
                    resume_path=resume_path,
                    save_dir=baseline_dir,
                    device_override=self.device,
                    num_envs=self.num_envs,
                    config=baseline_config,
                    plateau_detector=detector,
                    auto_scale=True,
                    use_curriculum=True,
                )

            self.state.total_episodes += baseline_result["episodes_completed"]
            baseline_score = baseline_result["avg_score"]

            if baseline_score > self.state.baseline_score:
                self.state.baseline_score = baseline_score
                best_path = os.path.join(baseline_dir, "model_best.pt")
                if os.path.exists(best_path):
                    self.state.best_checkpoint = best_path

            print(f"\n  Baseline result: avg_score={baseline_score:.0f}, "
                  f"best_score={baseline_result['best_score']}, "
                  f"stop={baseline_result['stop_reason']}")

            if baseline_result["stop_reason"] == "complete":
                print("  No plateau — baseline still improving. Continuing...")
                self.state.save()
                continue

            # --- Phase 2: Generate experiments ---
            plateau_context = {
                "avg_score": baseline_score,
                "avg_level": baseline_result["avg_level"],
                "best_score": baseline_result["best_score"],
                "best_level": baseline_result["best_level"],
                "episodes_trained": self.state.total_episodes,
                "fail_fast_budget": fail_fast_budget,
            }

            checkpoints = self.state.find_checkpoints()
            generator = StrategyGenerator(axis_stats=self.state.axis_stats)

            n_experiments = self.state.experiments_per_plateau
            specs = generator.generate_batch(
                n=n_experiments,
                plateau_context=plateau_context,
                available_checkpoints=checkpoints,
                generation=gen,
            )

            # Set max_episodes and checkpoint hints
            for spec in specs:
                spec.max_episodes = fail_fast_budget
                if spec.starting_point in ("from_best", "from_early", "from_perturbed"):
                    if not spec.checkpoint_path_hint and checkpoints:
                        spec.checkpoint_path_hint = checkpoints[0]

            print(f"\n--- Exploration Phase ({len(specs)} experiments) ---")
            print(f"  Baseline to beat: {baseline_score:.0f}")
            for i, spec in enumerate(specs):
                print(f"  [{i+1}] {spec.experiment_id}")
                print(f"      {spec.creation_reason}")
                print(f"      reward={spec.reward_mode} arch={spec.hidden_sizes} "
                      f"algo={spec.algorithm} start={spec.starting_point}")

            if dry_run:
                print("\n  [DRY RUN] Skipping experiment execution")
                self.state.save()
                if self.state.total_episodes >= total_episodes:
                    break
                continue

            # --- Phase 3: Run experiments with fail-fast ---
            runner = ExperimentRunner()
            results = []

            for i, spec in enumerate(specs):
                exp_dir = os.path.join(self.save_dir, f"gen_{gen:03d}", spec.experiment_id)

                print(f"\n  Running [{i+1}/{len(specs)}]: {spec.experiment_id}...")

                exp_result = runner.run(
                    spec=spec,
                    baseline_score=baseline_score,
                    save_dir=exp_dir,
                    device=self.device,
                    num_envs=self.num_envs,
                    plateau_context=plateau_context,
                )

                results.append(exp_result)
                self.state.total_episodes += exp_result.episodes_completed
                self.state.log_experiment(exp_result)

                delta_pct = (
                    ((exp_result.avg_score - baseline_score) / max(baseline_score, 1)) * 100
                )
                status = "WIN" if exp_result.stop_reason == "success" else "FAIL"
                print(f"    {status}: avg={exp_result.avg_score:.0f} "
                      f"({delta_pct:+.1f}%) "
                      f"eps={exp_result.episodes_completed} "
                      f"[{exp_result.stop_reason}] "
                      f"vel={exp_result.learning_velocity:.1f} "
                      f"in {exp_result.elapsed_seconds:.0f}s")

                if self.state.total_episodes >= total_episodes:
                    break

            # --- Phase 4: Select winner and integrate ---
            successful = [r for r in results if r.avg_score > baseline_score]

            if successful:
                # Multi-criteria winner selection
                winner = self._select_winner(successful, baseline_score)
                delta = ((winner.avg_score - baseline_score) / max(baseline_score, 1)) * 100
                spec_id = winner.spec.get("experiment_id", "?") if isinstance(winner.spec, dict) else winner.spec.experiment_id

                print(f"\n  WINNER: {spec_id}")
                print(f"    avg={winner.avg_score:.0f} ({delta:+.1f}%) "
                      f"level={winner.avg_level:.1f} vel={winner.learning_velocity:.1f}")

                self.state.adopt_winner(winner)
                # Reset experiments_per_plateau back to default on success
                self.state.experiments_per_plateau = experiments_per_plateau
            else:
                print(f"\n  No experiment beat baseline ({baseline_score:.0f})")

                # Check for diversity adoption (within 95% but very different approach)
                close_enough = [r for r in results
                                if r.avg_score >= baseline_score * 0.95 and r.avg_score > 0]
                if close_enough:
                    closest = max(close_enough, key=lambda r: r.avg_score)
                    spec_obj = ExperimentSpec.from_dict(closest.spec) if isinstance(closest.spec, dict) else closest.spec
                    # Check if it's a genuinely different approach
                    baseline_spec = ExperimentSpec()  # defaults
                    dist = self.diversity.distance(spec_obj, baseline_spec)
                    if dist > 0.4:
                        spec_id = closest.spec.get("experiment_id", "?") if isinstance(closest.spec, dict) else closest.spec.experiment_id
                        print(f"  Adopting {spec_id} for diversity "
                              f"(score={closest.avg_score:.0f}, distance={dist:.2f})")
                        self.state.adopt_winner(closest)

                # Increase exploration next round
                self.state.experiments_per_plateau = min(12,
                    self.state.experiments_per_plateau + 2)
                print(f"  Next round: {self.state.experiments_per_plateau} experiments")

            # Update axis stats
            self.state.update_axis_stats(results)
            self.state.save()

            # Print coverage report
            coverage = self.diversity.coverage_report(self.state.experiments)
            print(f"\n  Axis coverage (all generations):")
            for axis, counts in coverage.items():
                if counts:
                    items = ", ".join(f"{k}:{v}" for k, v in
                                     sorted(counts.items(), key=lambda x: -x[1]))
                    print(f"    {axis}: {items}")

            print(f"\n  Total episodes: {self.state.total_episodes:,} / {total_episodes:,}")

        # Final summary
        self._print_summary()

    def _select_winner(self, candidates: list, baseline_score: float) -> ExperimentResult:
        """Multi-criteria winner selection."""
        if len(candidates) == 1:
            return candidates[0]

        best = None
        best_composite = -float('inf')
        max_level = max(r.avg_level for r in candidates) or 1
        velocities = [r.learning_velocity for r in candidates]
        max_vel = max(velocities) if velocities else 1

        for r in candidates:
            score_ratio = r.avg_score / max(baseline_score, 1)
            level_ratio = r.avg_level / max(max_level, 1)
            vel_ratio = r.learning_velocity / max(max_vel, 0.01) if max_vel > 0 else 0

            composite = (
                0.50 * score_ratio +
                0.25 * level_ratio +
                0.25 * max(0, vel_ratio)
            )

            if composite > best_composite:
                best_composite = composite
                best = r

        return best

    def _print_summary(self):
        """Print final summary of all exploration."""
        print("\n" + "#" * 70)
        print("# EXPLORER TRAINING COMPLETE")
        print("#" * 70)
        print(f"Total generations: {self.state.generation}")
        print(f"Total episodes: {self.state.total_episodes:,}")
        print(f"Final baseline score: {self.state.baseline_score:.0f}")
        print(f"Total experiments run: {len(self.state.experiments)}")

        successes = sum(1 for e in self.state.experiments
                        if e.get("stop_reason") == "success")
        fails = sum(1 for e in self.state.experiments
                    if e.get("stop_reason") == "fail_fast")
        print(f"  Successes: {successes}")
        print(f"  Fail-fast kills: {fails}")
        print(f"  Budget exhausted: {len(self.state.experiments) - successes - fails}")

        if self.state.axis_stats:
            print(f"\nAxis success rates:")
            for axis, stats in sorted(self.state.axis_stats.items()):
                top = sorted(stats.items(), key=lambda x: -x[1])[:3]
                items = ", ".join(f"{k}={v:.2f}" for k, v in top)
                print(f"  {axis}: {items}")

        if self.state.best_checkpoint:
            print(f"\nBest model: {self.state.best_checkpoint}")
        print(f"State saved: {self.state.state_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Explorer Training: plateau-breaking through divergent experimentation")
    parser.add_argument("--episodes", type=int, default=500_000,
                        help="Total episode budget (default: 500,000)")
    parser.add_argument("--experiments-per-plateau", type=int, default=6,
                        help="Number of experiments per plateau (default: 6)")
    parser.add_argument("--fail-fast-budget", type=int, default=7500,
                        help="Max episodes per experiment (default: 7,500)")
    parser.add_argument("--save-dir", type=str, default="models/explorer",
                        help="Directory for explorer state and models")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, cuda, mps)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help=f"Number of parallel environments (default: {NUM_ENVS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing explorer_state.json")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Start from a specific checkpoint file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and print experiment specs without training")
    args = parser.parse_args()

    loop = ExplorerLoop(
        save_dir=args.save_dir,
        device=args.device,
        num_envs=args.num_envs,
    )

    loop.run(
        total_episodes=args.episodes,
        experiments_per_plateau=args.experiments_per_plateau,
        fail_fast_budget=args.fail_fast_budget,
        resume=args.resume,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
