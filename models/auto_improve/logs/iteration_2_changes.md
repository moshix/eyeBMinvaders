# Iteration 2: Fix PPO Training (100% of PPO experiments were broken)

## What Changed

Three fixes in two files to make PPO+GRU functional in the explorer:

### Fix 1: `train_ppo.py` line 978 — bracket access crash
**Before:** `ext["shots_hit"]` — crashes with KeyError when `ext` is empty (after `get_stats_ext` fails)
**After:** `ext.get("shots_hit", 0)` — safe access with default

This bug silently crashed PPO episode logging, causing the JSONL file to be empty/partial, which made the explorer report avg_score=0 for every PPO experiment.

### Fix 2: `explorer_train.py` `_checkpoint_compatible()` — state size validation for PPO
**Before:** Only checked `'agent' in checkpoint` — allowed loading checkpoints with old state size (248) into new model (276)
**After:** Also validates that `backbone.0.weight` input dimension matches `STATE_SIZE * n_frames`

This prevented experiments using `from_best`/`from_early`/`from_perturbed` with `model_ppo_best_avg.pt` from crashing instantly with size mismatch errors.

### Fix 3: `explorer_train.py` `_run_ppo()` — compute avg_level from logs
**Before:** `avg_level=0` hardcoded for all PPO results
**After:** Parses actual level data from the JSONL training log

## Expected Impact

- **156 PPO experiments** ran in the explorer — ALL scored 0. This is 38% of total experiments wasted.
- PPO+GRU reached **level 9** in standalone benchmarks (vs DQN's level 6-8 ceiling)
- With PPO now functional, the explorer's 70% PPO bias will actually produce useful experiments
- Expected: PPO experiments should start scoring 5K-20K+ avg and reaching levels 5-9
- The explorer will discover PPO hyperparameter configurations that push beyond DQN's plateau

## How to Verify

In the next explorer run, check:
1. PPO experiments should have `avg_score > 0` and `episodes_completed > 0`
2. PPO experiments should NOT have `stop_reason` starting with `"error:"`
3. `avg_level` for PPO experiments should be > 0
4. At least some PPO experiments should reach `best_level >= 5`
