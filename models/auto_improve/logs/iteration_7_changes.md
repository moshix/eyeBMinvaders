# Iteration 7: Quadratic Level Completion Reward

## What Changed
Modified `game_sim/src/state.rs` — the `calculate_reward` function:

1. **Level completion bonus**: Changed from linear (`5.0 + 3.0 * level`) to quadratic (`5.0 + 1.0 * level * level`)
   - Old: L1=8, L5=20, L10=35, L15=50 (only 2.5x between L5 and L15)
   - New: L1=6, L5=30, L10=105, L15=230 (7.7x between L5 and L15)

2. **Survival bonus**: Changed from linear (`0.01 * level`) to quadratic (`0.01 * level * level`)
   - Old: L1=0.01, L5=0.05, L10=0.10, L15=0.15
   - New: L1=0.01, L5=0.25, L10=1.00, L15=2.25

## Why
The agent plateaus at level 6 because the reward gradient between levels is too flat. A linear bonus gives almost equal reward for clearing level 5 vs level 15. Quadratic scaling creates exponentially stronger incentive to push deeper, making each new level significantly more rewarding than the last.

This directly addresses the known bottleneck: "Reward function: survival bonus only 0.01*level, level-complete bonus linear not exponential."

## Expected Impact
- Agent should push past level 6 plateau due to stronger reward gradient
- Higher levels become "worth" dramatically more, so the agent will invest more in survival strategies
- The per-tick survival bonus at high levels (2.25 at L15 vs 0.15 before) creates continuous pressure to stay alive longer

## How to Verify
- Next training run should show best_level > 6
- Average scores should increase significantly for experiments reaching L5+
- The reward signal for reaching L10+ should be ~3-5x stronger than before
