# Iteration 5: Fix Explorer Experiment Waste (~70% failure rate)

## What Changed

**File:** `explorer_train.py` — 3 critical bug fixes

### Bug 1: DQN experiments loading PPO checkpoints (instant failure)
- **Root cause:** `find_checkpoints()` returned all `.pt` files with "best" in the name, including `model_ppo_best_avg.pt`. DQN experiments got assigned this PPO checkpoint via `checkpoint_path_hint`, then crashed with `KeyError: 'policy_net'` because PPO checkpoints use `'agent'` key.
- **Fix:** Made `find_checkpoints(algorithm)` filter by algorithm type using filename conventions (`ppo` in filename = PPO checkpoint). Updated all callers to pass algorithm. Added fallback: if no compatible checkpoint exists, switch to `from_scratch` instead of crashing.
- **Impact:** ~40% of experiments were failing from this bug alone.

### Bug 2: ALL PPO experiments crash with NoneType error
- **Root cause:** `_run_ppo()` called `train_ppo()` which could fail silently. `_parse_ppo_results()` then tried to call `.get()` on non-dict JSON parse results, crashing with `'NoneType' object has no attribute 'get'`.
- **Fix:** Added try/except around the PPO training call. Made `_parse_ppo_results()` validate that parsed JSON is actually a dict before calling `.get()`. Added `AttributeError` and `TypeError` to caught exceptions.
- **Impact:** ~30% of experiments (all PPO) were failing from this bug.

### Bug 3: Explorer crash at end of generation
- **Root cause:** `ExperimentSpec()` called without required `experiment_id` argument on line 1312, in the diversity adoption logic.
- **Fix:** Pass `experiment_id="baseline_default"`.
- **Impact:** Explorer crashed after every generation that didn't beat baseline, preventing diversity adoption.

## Expected Impact

- **Before:** Only ~30% of experiments actually ran (DQN from_scratch only). All PPO experiments, all DQN-with-checkpoint experiments, and all diversity adoption failed.
- **After:** ~90%+ of experiments should run successfully. This effectively 3x the compute budget per generation.
- **Next iteration should see:** More diverse experiment results, PPO experiments actually producing scores, and better convergence to optimal configs.

## How to Verify

Run the explorer and check:
1. Experiments with `starting_point: from_best/from_early/from_perturbed` should either find compatible checkpoints or gracefully fall back to `from_scratch`
2. PPO experiments should produce non-zero scores (even if lower than DQN)
3. No `'policy_net'` KeyError or `'NoneType' object has no attribute 'get'` errors in output
4. Explorer should not crash with `TypeError: ExperimentSpec.__init__() missing 1 required positional argument`

## Best Results from This Run

| Experiment | Avg Score | Best Level | Config |
|-----------|----------|-----------|--------|
| exp_informed_b1eb03 | 17,090 | 5 | sparse, [128,64], challenge_only, epsilon_high |
| exp_informed_1e43bd | 16,711 | 5 | sparse, [128,64], challenge_only, noisy_net |
| exp_radical_dqn_aggressive_29813d | 15,306 | 6 | aggressive, [128,128,128,64], challenge_only |
| exp_informed_7b0b81 | 12,544 | 6 | level_rush, [1024,256], random_uniform, epsilon_cyclic |
