#!/usr/bin/env python3
"""
Meta-Learning Training Loop for eyeBMinvaders
==============================================
Outer loop that detects training plateaus and automatically restarts
from the best model with evolved hyperparameters. Tracks which parameter
mutations helped and adapts the mutation strategy over time.

Usage:
    python meta_train.py                          # Default: 500K episodes
    python meta_train.py --episodes 200000        # Shorter run
    python meta_train.py --resume-meta             # Continue from last meta state
    python meta_train.py --resume models/model_best.pt  # Start from checkpoint

The system uses a multi-armed bandit to select hyperparameter mutations,
boosting mutations that led to improvement and penalizing those that didn't.
"""

import json
import os
import random
import time
import argparse
from copy import deepcopy

import numpy as np

from train import train, TrainingConfig, PlateauDetector, NUM_ENVS


# =============================================================================
# Mutation Definitions
# =============================================================================
DEFAULT_MUTATION_WEIGHTS = {
    "lr_reduce": 0.30,
    "epsilon_bump": 0.30,
    "epsilon_bump_large": 0.05,
    "buffer_flush": 0.10,
    "batch_size_up": 0.05,
    "gamma_increase": 0.05,
    "tau_reduce": 0.05,
}

# Mutations that are always applied (proven effective from learnings.md)
ALWAYS_APPLY = {"lr_reduce", "epsilon_bump"}


def apply_mutation(config: TrainingConfig, mutation: str) -> TrainingConfig:
    """Apply a single mutation to the config, returning a modified copy."""
    cfg = deepcopy(config)
    if mutation == "lr_reduce":
        factor = random.uniform(0.3, 0.5)
        cfg.lr = max(1e-6, cfg.lr * factor)
    elif mutation == "epsilon_bump":
        cfg.epsilon_start = random.uniform(0.10, 0.25)
    elif mutation == "epsilon_bump_large":
        cfg.epsilon_start = random.uniform(0.40, 0.60)
    elif mutation == "buffer_flush":
        pass  # handled via flush_buffer flag in train()
    elif mutation == "batch_size_up":
        cfg.batch_size = min(1024, cfg.batch_size * 2)
    elif mutation == "gamma_increase":
        cfg.gamma = min(0.999, cfg.gamma + 0.005)
    elif mutation == "tau_reduce":
        cfg.tau = max(0.0005, cfg.tau * 0.5)
    return cfg


def select_mutations(weights: dict, n_extra=1) -> list:
    """Select mutations: always apply core ones + sample extras."""
    selected = list(ALWAYS_APPLY)

    # Sample extra mutations from the remaining pool
    extras = {k: v for k, v in weights.items() if k not in ALWAYS_APPLY}
    if extras and n_extra > 0:
        keys = list(extras.keys())
        probs = np.array([extras[k] for k in keys])
        probs = probs / probs.sum()
        n = min(n_extra, len(keys))
        chosen = np.random.choice(keys, size=n, replace=False, p=probs)
        selected.extend(chosen.tolist())

    return selected


def update_mutation_weights(weights: dict, mutations: list, improvement_pct: float) -> dict:
    """Update mutation weights based on cycle results (multi-armed bandit)."""
    weights = dict(weights)
    for m in mutations:
        if m not in weights:
            continue
        if improvement_pct > 5.0:
            weights[m] *= 1.5
        elif improvement_pct > 0.0:
            weights[m] *= 1.1
        elif improvement_pct < -3.0:
            weights[m] *= 0.6
        else:
            weights[m] *= 0.9

    # Re-normalize
    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] /= total

    return weights


# =============================================================================
# Meta State I/O
# =============================================================================
def load_meta_state(save_dir: str) -> dict:
    path = os.path.join(save_dir, "meta_learning.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "version": 1,
        "current_cycle": 0,
        "total_episodes": 0,
        "mutation_weights": dict(DEFAULT_MUTATION_WEIGHTS),
        "cycles": [],
        "best_ever_avg_score": 0,
        "best_ever_model": None,
        "base_config": TrainingConfig().to_dict(),
    }


def save_meta_state(state: dict, save_dir: str):
    path = os.path.join(save_dir, "meta_learning.json")
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def log_meta_event(save_dir: str, event: dict):
    event["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = os.path.join(save_dir, "meta_events.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


# =============================================================================
# Meta Training Loop
# =============================================================================
def meta_train(total_episodes=500_000, save_dir="models", device=None,
               num_envs=NUM_ENVS, resume_path=None, resume_meta=False,
               max_cycles=50):
    os.makedirs(save_dir, exist_ok=True)

    # Load or create meta state
    if resume_meta:
        meta = load_meta_state(save_dir)
        print(f"Resuming meta-learning from cycle {meta['current_cycle']}, "
              f"{meta['total_episodes']} episodes trained")
    else:
        meta = load_meta_state(save_dir) if os.path.exists(
            os.path.join(save_dir, "meta_learning.json")) else {
            "version": 1,
            "current_cycle": 0,
            "total_episodes": 0,
            "mutation_weights": dict(DEFAULT_MUTATION_WEIGHTS),
            "cycles": [],
            "best_ever_avg_score": 0,
            "best_ever_model": resume_path,
            "base_config": TrainingConfig().to_dict(),
        }

    base_config = TrainingConfig.from_dict(meta["base_config"])
    cycle_num = meta["current_cycle"]

    print("=" * 70)
    print("META-LEARNING TRAINING")
    print("=" * 70)
    print(f"Total episodes target: {total_episodes:,}")
    print(f"Starting cycle: {cycle_num}")
    print(f"Mutation weights: {json.dumps(meta['mutation_weights'], indent=2)}")
    print("=" * 70)

    while meta["total_episodes"] < total_episodes and cycle_num < max_cycles:
        cycle_num += 1
        remaining = total_episodes - meta["total_episodes"]
        # Each cycle runs up to this many episodes (or until plateau)
        cycle_episodes = min(50_000, remaining)

        print(f"\n{'#' * 70}")
        print(f"# CYCLE {cycle_num}")
        print(f"{'#' * 70}")

        # Determine config and resume path for this cycle
        if cycle_num == 1 and not meta["cycles"]:
            # First cycle: use base config and optional resume path
            config = deepcopy(base_config)
            cycle_resume = resume_path
            mutations = []
            flush_buffer = False
            print("First cycle — using base configuration")
        else:
            # Subsequent cycles: mutate and restart from best
            mutations = select_mutations(meta["mutation_weights"])
            config = deepcopy(base_config)
            flush_buffer = "buffer_flush" in mutations

            for m in mutations:
                config = apply_mutation(config, m)

            cycle_resume = meta.get("best_ever_model")
            if not cycle_resume or not os.path.exists(cycle_resume):
                cycle_resume = os.path.join(save_dir, "model_best.pt")
            if not os.path.exists(cycle_resume):
                cycle_resume = None

            print(f"Mutations: {mutations}")
            print(f"Config: lr={config.lr:.2e}, eps_start={config.epsilon_start:.2f}, "
                  f"batch={config.batch_size}, gamma={config.gamma}, tau={config.tau}")
            print(f"Resume from: {cycle_resume}")
            print(f"Flush buffer: {flush_buffer}")

        # Log cycle start
        log_meta_event(save_dir, {
            "event": "cycle_start",
            "cycle": cycle_num,
            "mutations": mutations,
            "config": config.to_dict(),
            "resume_from": cycle_resume,
        })

        # Create plateau detector for this cycle
        detector = PlateauDetector(
            window=5000,
            min_episodes=15000,
            cooldown=10000,
            score_threshold=0.03,
        )

        # Run training
        result = train(
            episodes=cycle_episodes,
            resume_path=cycle_resume,
            save_dir=save_dir,
            device_override=device,
            num_envs=num_envs,
            config=config,
            plateau_detector=detector,
            cycle=cycle_num,
            flush_buffer=flush_buffer,
        )

        # Evaluate cycle
        plateau_score = meta.get("best_ever_avg_score", 0)
        cycle_avg = result["avg_score"]
        if plateau_score > 0:
            improvement_pct = (cycle_avg - plateau_score) / plateau_score * 100
        else:
            improvement_pct = 0

        # Update best ever
        if cycle_avg > meta["best_ever_avg_score"]:
            meta["best_ever_avg_score"] = cycle_avg
            best_path = os.path.join(save_dir, "model_best.pt")
            if os.path.exists(best_path):
                meta["best_ever_model"] = best_path

        cycle_record = {
            "cycle": cycle_num,
            "start_episodes": meta["total_episodes"],
            "episodes_this_cycle": result["episodes_completed"],
            "plateau_score": plateau_score,
            "end_avg_score": cycle_avg,
            "best_score": result["best_score"],
            "best_level": result["best_level"],
            "avg_level": result["avg_level"],
            "improvement_pct": round(improvement_pct, 2),
            "mutations": mutations,
            "config": config.to_dict(),
            "stop_reason": result["stop_reason"],
            "elapsed_seconds": round(result["elapsed"], 1),
        }
        meta["cycles"].append(cycle_record)
        meta["total_episodes"] += result["episodes_completed"]
        meta["current_cycle"] = cycle_num

        # Update mutation weights based on results
        if mutations:
            meta["mutation_weights"] = update_mutation_weights(
                meta["mutation_weights"], mutations, improvement_pct)

        # Save meta state
        meta["base_config"] = base_config.to_dict()
        save_meta_state(meta, save_dir)

        # Log cycle end
        log_meta_event(save_dir, {
            "event": "cycle_end",
            "cycle": cycle_num,
            "improvement_pct": round(improvement_pct, 2),
            "avg_score": cycle_avg,
            "stop_reason": result["stop_reason"],
        })

        # Print cycle summary
        arrow = "^" if improvement_pct > 0 else "v" if improvement_pct < 0 else "="
        print(f"\n{'='*70}")
        print(f"CYCLE {cycle_num} SUMMARY")
        print(f"{'='*70}")
        print(f"  Episodes: {result['episodes_completed']:,} | "
              f"Stop: {result['stop_reason']}")
        print(f"  Avg Score: {cycle_avg:.0f} {arrow} ({improvement_pct:+.1f}%)")
        print(f"  Best Score: {result['best_score']:,} | "
              f"Best Level: {result['best_level']}")
        print(f"  Mutations: {mutations}")
        print(f"  Updated weights: {json.dumps(meta['mutation_weights'], indent=4)}")
        print(f"  Total episodes so far: {meta['total_episodes']:,}")
        print(f"{'='*70}\n")

        if result["stop_reason"] == "complete" and not mutations:
            # First cycle completed without plateau — keep going
            pass

    # Final summary
    print("\n" + "#" * 70)
    print("# META-LEARNING COMPLETE")
    print("#" * 70)
    print(f"Total cycles: {cycle_num}")
    print(f"Total episodes: {meta['total_episodes']:,}")
    print(f"Best ever avg score: {meta['best_ever_avg_score']:.0f}")
    print(f"\nCycle history:")
    for c in meta["cycles"]:
        print(f"  Cycle {c['cycle']}: avg={c['end_avg_score']:.0f} "
              f"({c['improvement_pct']:+.1f}%) "
              f"mutations={c['mutations']} "
              f"stop={c['stop_reason']}")
    print(f"\nFinal mutation weights:")
    for k, v in sorted(meta["mutation_weights"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-learning training for eyeBMinvaders")
    parser.add_argument("--episodes", type=int, default=500_000,
                        help="Total episodes across all cycles (default: 500,000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to start from")
    parser.add_argument("--resume-meta", action="store_true",
                        help="Continue from existing meta_learning.json state")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models (default: models)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, cuda, mps)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help=f"Number of parallel environments (default: {NUM_ENVS})")
    parser.add_argument("--max-cycles", type=int, default=50,
                        help="Maximum number of cycles (default: 50)")
    args = parser.parse_args()

    meta_train(
        total_episodes=args.episodes,
        save_dir=args.save_dir,
        device=args.device,
        num_envs=args.num_envs or NUM_ENVS,
        resume_path=args.resume,
        resume_meta=args.resume_meta,
        max_cycles=args.max_cycles,
    )
