# Iteration 9: Extend Successful Experiments (20K continuation)

## What Changed
Modified `explorer_train.py` `_run_dqn()` method: when an experiment beats baseline, instead of immediately stopping with "success", extend its training budget by 20,000 more episodes.

### Before
- Experiment beats baseline at ~2,500 episodes → immediately returns "success"
- Best models only get 2,500-7,500 episodes of training total
- Not enough training time to learn strategies for levels 7+

### After
- Experiment beats baseline → prints message, extends budget by 20K episodes
- Continues training until extended budget exhausted or learning velocity goes negative
- Fail-fast is disabled during extended training (we already beat baseline)
- Budget-exhausted with extension reports "success" so winner selection works correctly

### Specific Changes
1. Added `SUCCESS_EXTENSION = 20000` constant
2. Success check now extends `spec.max_episodes` on first beat instead of returning
3. During extended training, stops early only if last 3 checkpoints show negative velocity
4. Fail-fast check skipped for extended experiments
5. Budget-exhausted stop_reason = "success" if experiment was extended

## Why This Is The Highest-Impact Change
- The explorer already found winning configs (sparse/[128,64]/challenge_only, sparse/[1024,256]/progressive)
- These configs hit baseline at 2,500 episodes and STOP - never getting to train long enough
- With only 2,500-7,500 episodes, models plateau at level 5-6
- With 22,500+ episodes (2,500 initial + 20,000 extension), models have ~9x more training time
- This should push winning configs from level 5-6 to level 8-10+

## Expected Impact
- Winning experiments train for 20K+ episodes instead of 2,500
- Best level should increase from 6 to 8-10
- Best avg score should increase significantly beyond 17,090
- More total episodes spent on promising configs, fewer wasted on exploration

## How to Verify
- Check if any experiment reaches best_level > 6
- Check if experiments show "Beat baseline! Extending training" messages
- Check if extended experiments have episodes_completed > 7,500
- Compare avg_score of extended experiments vs previous best (17,090)
