# Iteration 6: Fix checkpoint override to always enforce algorithm-compatibility

## What Changed

**File:** `explorer_train.py`, lines 1262-1275 (checkpoint assignment in main run loop)

**Problem:** The explorer's strategy generators (`_single_axis_changes`, `_random_compositions`, `_informed_specs`) assign `checkpoint_path_hint` from a generic `checkpoints` list that can contain PPO checkpoints. The post-generation fix (line 1267) was supposed to correct this, but it only ran when `checkpoint_path_hint` was empty (`if not spec.checkpoint_path_hint`). Since generators already set a hint, the fix was bypassed.

**Result:** ~50% of DQN experiments crashed instantly with `'policy_net'` KeyError because they received a PPO checkpoint (`models/model_ppo_best_avg.pt`). This wasted enormous compute — out of 273 total experiments, a large proportion failed at 0 episodes in <0.02 seconds.

**Fix:** Changed the override logic to **always** set the correct algorithm-specific checkpoint, regardless of what generators assigned:
- `algo_ckpts = dqn_checkpoints if spec.algorithm == "dqn" else ppo_checkpoints`
- Always override `spec.checkpoint_path_hint = algo_ckpts[0]` (removed `not spec.checkpoint_path_hint` guard)
- Fall back to `from_scratch` if no compatible checkpoint exists
- Also clear checkpoint hint for `from_scratch` experiments to prevent stale hints

## Expected Impact

- **DQN experiments loading from checkpoints will actually run** instead of crashing instantly
- This roughly doubles the number of useful experiments per generation
- More experiments running means faster convergence to better configs
- The "single-axis change" experiments (which always start from_best) will finally produce data instead of all failing
- Combined with the bug fixes from iteration 5, the explorer should now operate at full efficiency

## How to Verify

After the next training run:
1. Check explorer output for `'policy_net'` errors — should be zero
2. Count experiments with `episodes_completed > 0` — should be much higher than before
3. Single-axis experiments should show actual scores instead of all failing
4. Overall best level should improve since more experiment search space is being explored
