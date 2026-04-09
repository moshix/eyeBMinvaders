# Iteration 3: Fix PPO Integration in Explorer

## What Changed

**File:** `explorer_train.py` — three fixes in one commit

### Fix 1: ParallelGameEnv → BatchedGames (line 965)
The `_checkpoint_compatible()` method imported `ParallelGameEnv` from `game_sim`, but this class doesn't exist in the Rust module. The correct export is `BatchedGames`. This caused ALL PPO checkpoint compatibility checks to crash, making every PPO resume experiment fail silently and fall back to scratch (where it would hit other issues).

### Fix 2: Build and pass PPOConfig from experiment spec
The `_run_ppo()` method called `train_ppo()` without passing a `config` parameter. This meant PPO always used default `PPOConfig()` regardless of the experiment spec's `hidden_sizes`, `lr`, `gamma`, `batch_size` etc. Now the method constructs a `PPOConfig` from the spec's hyperparameters and passes it to `train_ppo(config=ppo_cfg)`.

### Fix 3: Wrap result parsing in try/except
The result-parsing code after `train_ppo()` (reading JSONL logs, computing avg_level) was outside any exception handler. If parsing failed, the exception escaped to the outer catch-all (line 663) which reported a generic error without the "PPO" prefix, making it indistinguishable from DQN errors. Now result parsing has its own try/except that returns a descriptive error.

## Why This Matters

**100% of PPO experiments were broken.** Out of 430 total experiments:
- 165 were PPO (38%)
- ALL 165 either crashed on checkpoint load (size mismatch 248 vs 276) or produced 0 scores
- This wasted ~50% of the GPU compute budget

The known bottleneck states: "DQN plateaus at level 6-8 regardless of hyperparameters — PPO+GRU is the path to level 15." With PPO broken, the system was forced to use DQN which has a hard ceiling.

## Expected Impact

- PPO experiments will now actually train and produce meaningful results
- The explorer can discover winning PPO configs (PPO+GRU reached level 9 in separate benchmarks)
- PPO's temporal reasoning (GRU) should enable the agent to handle high-level threats (kamikazes, homing missiles) that DQN can't
- Expected: best level should increase from 8 → 10+ within 1-2 iterations

## How to Verify

In the next iteration:
1. Check that PPO experiments no longer show `'NoneType' object has no attribute 'get'` errors
2. Check that PPO experiments no longer show `size mismatch` errors
3. Check that PPO experiments show non-zero avg_score and best_level > 0
4. Check axis_stats: `algorithm.ppo` should have non-zero success rate
