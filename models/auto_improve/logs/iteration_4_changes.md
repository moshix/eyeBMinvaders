# Iteration 4: Fix PPO SIL crash — every PPO experiment was dying on first episode

## What changed

**File:** `train_ppo.py`, line 952
**Change:** `episode_rewards[i]` → `ep_score`

The Self-Imitation Learning (SIL) buffer code referenced an undefined variable `episode_rewards[i]`. The correct variable is `ep_score` (defined on line 924 from `envs.get_stats(i)`).

## Why this matters

This bug caused **100% of PPO experiments to crash on the very first episode completion**. The NameError fired before any log entries were written (line 964), so the explorer saw 0 episodes completed and 0 scores for every PPO run.

Evidence from explorer state:
- PPO win rate: 0.00 across all 455 experiments
- All PPO from_scratch experiments show avg_score=0, avg_level=0, episodes_completed=0
- PPO budget_exhausted experiments ran for ~190-200 seconds but produced zero scores
- The `'NoneType' object has no attribute 'get'` errors in earlier iterations were likely downstream of this crash

## Expected impact

**HIGH.** PPO+GRU is the only architecture capable of temporal reasoning (DQN has no memory). Previous standalone PPO benchmarks reached level 9. With PPO actually working in the explorer:

- PPO experiments should now complete and report real scores
- The explorer can discover optimal PPO hyperparameters
- PPO+GRU should break through the DQN level 6-8 plateau
- Expected: level 10-12 within 2-3 iterations, level 15 within 5-6 iterations

## How to verify

In the next explorer run:
1. PPO experiments should show non-zero avg_score and avg_level
2. PPO from_scratch experiments should complete 7500 episodes (not 0)
3. At least some PPO experiments should beat the DQN baseline (~36K avg score)
4. Check `axis_stats.algorithm.ppo` — should be >> 0 (currently 0.00)
