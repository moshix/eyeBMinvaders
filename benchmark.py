#!/usr/bin/env python3
"""
eyeBMinvaders Model Benchmark Runner
=====================================
Runs headless episodes with each saved model and the heuristic AI,
then outputs comparative results.

Usage:
    python benchmark.py                    # Run all models, 50 episodes each
    python benchmark.py --episodes 100     # More episodes for accuracy
    python benchmark.py --output results.json  # Save results as JSON

Requirements:
    pip install torch numpy
    Rust game_sim must be compiled: cd game_sim && maturin develop --release
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

try:
    from game_sim import BatchedGames
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("ERROR: Rust game_sim not found. Build with: cd game_sim && maturin develop --release")
    exit(1)

import torch
import torch.nn as nn
from torch.distributions import Categorical

from train import FrameStack
from train_ppo import ActorCritic, RunningMeanStd, PPOConfig


def load_dqn_model(path, device='cpu'):
    """Load a DQN checkpoint and return (net, config)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    arch = checkpoint.get('arch', {})
    hidden = arch.get('hidden_sizes', [512, 256, 128])
    use_dueling = arch.get('use_dueling', True)
    use_noisy = arch.get('use_noisy', True)
    n_frames = arch.get('n_frames', 4)

    # Detect state size from first layer
    policy_sd = checkpoint.get('policy_net', {})
    first_key = None
    for k in policy_sd:
        if 'weight' in k and ('features.0' in k or 'net.0' in k):
            first_key = k
            break

    if first_key is None:
        return None, None

    state_size = policy_sd[first_key].shape[1]
    action_size = 6

    from train import DuelingDQN, DQN
    if use_dueling:
        net = DuelingDQN(state_size, action_size, hidden, use_noisy=use_noisy)
    else:
        net = DQN(state_size, action_size, hidden)

    # Load weights (handle _orig_mod prefix)
    clean_sd = {}
    for k, v in policy_sd.items():
        clean_k = k.replace('_orig_mod.', '')
        clean_sd[clean_k] = v
    try:
        net.load_state_dict(clean_sd, strict=False)
    except Exception:
        pass

    net.eval()
    return net, {'type': 'dqn', 'state_size': state_size, 'n_frames': n_frames,
                 'hidden': hidden, 'dueling': use_dueling, 'noisy': use_noisy}


def load_ppo_model(path, device='cpu'):
    """Load a PPO checkpoint and return (net, obs_rms, config)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    arch_cfg = checkpoint.get('config', {})
    hidden = arch_cfg.get('hidden_sizes', [512, 256, 128])
    head_h = arch_cfg.get('head_hidden', 64)
    gru_h = arch_cfg.get('gru_hidden', 0)
    n_frames = arch_cfg.get('n_frames', 4)

    # Detect state size from weights
    agent_sd = checkpoint.get('agent', {})
    for k, v in agent_sd.items():
        if 'backbone.0.weight' in k:
            state_size = v.shape[1]
            break
    else:
        state_size = 248  # default

    net = ActorCritic(state_size, 6, hidden_sizes=hidden, head_hidden=head_h,
                      gru_hidden=gru_h, n_frames=n_frames)
    net.load_state_dict(agent_sd)
    net.eval()

    obs_rms = None
    if 'obs_rms' in checkpoint:
        obs_rms = RunningMeanStd(state_size)
        obs_rms.load_state_dict(checkpoint['obs_rms'])

    return net, obs_rms, {
        'type': 'ppo', 'state_size': state_size, 'n_frames': n_frames,
        'hidden': hidden, 'head_hidden': head_h, 'gru_hidden': gru_h,
        'episodes': checkpoint.get('total_episodes', 0),
        'best_avg': checkpoint.get('best_avg_score', 0),
    }


def run_benchmark(model_name, select_action_fn, raw_state_size, n_frames,
                  num_episodes=50, num_envs=64):
    """Run headless episodes and return stats."""
    envs = BatchedGames(num_envs, seed=42)
    frame_stack = FrameStack(num_envs, raw_state_size, n_frames) if n_frames > 1 else None
    state_size = raw_state_size * n_frames if n_frames > 1 else raw_state_size

    raw_states = envs.reset_all()
    states = frame_stack.reset_all(raw_states) if frame_stack else raw_states

    scores = []
    levels = []
    steps_list = []
    episode_count = 0
    start_time = time.time()

    while episode_count < num_episodes:
        actions = select_action_fn(states)

        raw_next, rewards, dones = envs.step_all_fast(actions)
        next_states = frame_stack.push(raw_next) if frame_stack else raw_next

        done_idxs = np.where(np.asarray(dones))[0]
        for i in done_idxs:
            i = int(i)
            ep_score, ep_level, _, ep_steps, _, _, _, _ = envs.get_stats(i)
            scores.append(ep_score)
            levels.append(ep_level)
            steps_list.append(ep_steps)
            episode_count += 1

            if episode_count % 10 == 0:
                elapsed = time.time() - start_time
                avg_s = np.mean(scores)
                avg_l = np.mean(levels)
                print(f"  {model_name}: {episode_count}/{num_episodes} eps, "
                      f"avg score={avg_s:.0f}, avg level={avg_l:.1f} "
                      f"({elapsed:.0f}s)")

            if episode_count >= num_episodes:
                break

            raw_state = envs.reset_one(i)
            if frame_stack:
                next_states[i] = frame_stack.reset(i, raw_state)
            else:
                next_states[i] = raw_state

        states = next_states

    elapsed = time.time() - start_time
    return {
        'model': model_name,
        'episodes': len(scores),
        'avg_score': float(np.mean(scores)),
        'median_score': float(np.median(scores)),
        'max_score': int(np.max(scores)),
        'min_score': int(np.min(scores)),
        'std_score': float(np.std(scores)),
        'avg_level': float(np.mean(levels)),
        'max_level': int(np.max(levels)),
        'avg_steps': float(np.mean(steps_list)),
        'elapsed_sec': round(elapsed, 1),
        'eps_per_sec': round(len(scores) / elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark eyeBMinvaders AI models")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per model (default: 50)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file (default: benchmark_results.json)")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory containing model files")
    args = parser.parse_args()

    device = 'cpu'  # headless benchmark, CPU is fine
    raw_state_size = BatchedGames(1, seed=0).state_size  # 62
    results = []

    print("=" * 70)
    print("eyeBMinvaders Model Benchmark")
    print(f"Episodes per model: {args.episodes}")
    print(f"State size: {raw_state_size}")
    print("=" * 70)

    # --- 1. Random Agent ---
    print("\n[Random Agent]")
    def random_action(states):
        return [random.randrange(6) for _ in range(len(states))]
    results.append(run_benchmark("Random Agent", random_action, raw_state_size, 1,
                                 args.episodes))

    # --- 2. Heuristic AI (rule-based threat avoidance) ---
    print("\n[Heuristic AI]")
    def heuristic_action(states):
        """Simple rule-based AI: dodge nearest threat, fire when safe."""
        actions = []
        for s in states:
            # Use last frame's features (indices relative to last frame)
            base = (4 - 1) * raw_state_size  # last frame offset for 4-frame stack
            if len(s) <= raw_state_size:
                base = 0

            player_x = s[base + 0]  # normalized player X
            # Nearest bullet relative X (feature 8)
            bullet_dx = s[base + 8] if abs(s[base + 9]) < 0.5 else 0  # only if bullet nearby
            # Nearest kamikaze relative X (feature 18)
            kamikaze_dx = s[base + 18] if abs(s[base + 19]) < 0.5 else 0
            # Nearest missile relative X (feature 13)
            missile_dx = s[base + 13] if abs(s[base + 14]) < 0.5 else 0

            # Pick the most dangerous threat
            threats = []
            if abs(s[base + 9]) < 0.5:  # bullet nearby
                threats.append(('bullet', bullet_dx, abs(s[base + 9])))
            if abs(s[base + 19]) < 0.5:  # kamikaze nearby
                threats.append(('kamikaze', kamikaze_dx, abs(s[base + 19])))
            if abs(s[base + 14]) < 0.5:  # missile nearby
                threats.append(('missile', missile_dx, abs(s[base + 14])))

            if threats:
                # Sort by distance (closest first)
                threats.sort(key=lambda t: t[2])
                dx = threats[0][1]
                # Dodge: move away from threat, fire while dodging
                if dx < -0.02:
                    actions.append(5)  # FIRE+RIGHT
                elif dx > 0.02:
                    actions.append(4)  # FIRE+LEFT
                else:
                    # Threat directly above — pick a direction
                    actions.append(5 if player_x < 0.5 else 4)
            else:
                # No immediate threat — fire
                actions.append(3)  # FIRE
        return actions

    results.append(run_benchmark("Heuristic AI", heuristic_action, raw_state_size, 1,
                                 args.episodes))

    # --- 3. DQN models ---
    dqn_models = {
        'DQN Dueling+NoisyNet': 'model_best_54.pt',
    }
    for name, filename in dqn_models.items():
        path = os.path.join(args.models_dir, filename)
        if not os.path.exists(path):
            print(f"\n[{name}] — skipped (file not found: {path})")
            continue
        print(f"\n[{name}]")
        net, cfg = load_dqn_model(path, device)
        if net is None:
            print(f"  Failed to load")
            continue
        n_frames = cfg['n_frames']
        state_size = cfg['state_size']
        model_raw = state_size // n_frames if n_frames > 1 else state_size

        def make_dqn_action(net, model_raw, n_frames):
            def select(states):
                # Trim state to match model if needed
                if n_frames > 1 and states.shape[-1] != state_size:
                    trimmed = np.zeros((len(states), state_size), dtype=np.float32)
                    for f in range(n_frames):
                        src_start = f * raw_state_size
                        dst_start = f * model_raw
                        copy_len = min(model_raw, raw_state_size)
                        trimmed[:, dst_start:dst_start+copy_len] = states[:, src_start:src_start+copy_len]
                    states = trimmed
                with torch.no_grad():
                    s = torch.as_tensor(states, dtype=torch.float32)
                    q = net(s)
                    return q.argmax(dim=1).numpy().tolist()
            return select

        action_fn = make_dqn_action(net, model_raw, n_frames)
        r = run_benchmark(name, action_fn, raw_state_size, n_frames, args.episodes)
        r['config'] = cfg
        results.append(r)

    # --- 4. PPO models ---
    ppo_models = {
        'PPO v10 (GRU+SIL)': 'model_ppo_best_avg.pt',
        'PPO Vanilla (62-feat)': 'ppo_vanilla_baseline/model_ppo_best_avg.pt',
    }
    for name, filename in ppo_models.items():
        path = os.path.join(args.models_dir, filename)
        if not os.path.exists(path):
            print(f"\n[{name}] — skipped (file not found: {path})")
            continue
        print(f"\n[{name}]")
        net, obs_rms, cfg = load_ppo_model(path, device)
        n_frames = cfg['n_frames']
        model_state_size = cfg['state_size']
        model_raw = model_state_size // n_frames if n_frames > 1 else model_state_size
        has_gru = cfg.get('gru_hidden', 0) > 0

        # Per-env GRU hidden state
        gru_hx = torch.zeros(64, cfg.get('gru_hidden', 64)) if has_gru else None

        def make_ppo_action(net, obs_rms, model_raw, n_frames, has_gru, gru_hx_ref):
            def select(states):
                nonlocal gru_hx_ref
                # Trim state to match model
                if n_frames > 1 and states.shape[-1] != model_state_size:
                    trimmed = np.zeros((len(states), model_state_size), dtype=np.float32)
                    for f in range(n_frames):
                        src_start = f * raw_state_size
                        dst_start = f * model_raw
                        copy_len = min(model_raw, raw_state_size)
                        trimmed[:, dst_start:dst_start+copy_len] = states[:, src_start:src_start+copy_len]
                    states = trimmed
                if obs_rms:
                    states = obs_rms.normalize(states)
                with torch.no_grad():
                    s = torch.as_tensor(states, dtype=torch.float32)
                    if has_gru:
                        logits, _, new_hx = net.forward(s, gru_hx_ref)
                        gru_hx_ref = new_hx.detach()
                    else:
                        logits, _, _ = net.forward(s)
                    return logits.argmax(dim=1).numpy().tolist()
            return select

        action_fn = make_ppo_action(net, obs_rms, model_raw, n_frames, has_gru, gru_hx)
        r = run_benchmark(name, action_fn, raw_state_size, n_frames, args.episodes)
        r['config'] = cfg
        results.append(r)

    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<25} {'Avg Score':>10} {'Avg Lvl':>8} {'Max Lvl':>8} "
          f"{'Max Score':>10} {'Std':>8} {'ep/s':>6}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x['avg_score'], reverse=True):
        print(f"{r['model']:<25} {r['avg_score']:>10,.0f} {r['avg_level']:>8.1f} "
              f"{r['max_level']:>8} {r['max_score']:>10,} {r['std_score']:>8,.0f} "
              f"{r['eps_per_sec']:>6.1f}")

    # --- Save JSON ---
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'episodes_per_model': args.episodes,
        'state_size': raw_state_size,
        'results': results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
