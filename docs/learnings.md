# Training Pipeline Learnings

## Performance Optimization Journey

### Starting point
- Python game simulation, MPS (Mac), 8 parallel envs
- **4 ep/s, 7% GPU utilization**
- 200K episodes would take ~14 hours

### Step 1: Hyperparameter tuning for GPU (Python sim)
- Increased parallel envs: 8 → 32
- Faster epsilon decay: 0.99998 → 0.9999
- Larger replay buffer: 200K → 500K
- `torch.set_float32_matmul_precision('high')` for TF32 tensor cores
- `torch.compile()` on policy/target networks
- **Result: Still ~4 ep/s** — game sim was the bottleneck, not GPU

### Step 2: Rust game simulation (PyO3)
- Ported entire HeadlessGame (~1000 lines Python → ~800 lines Rust)
- Exposed via PyO3 with `BatchedGames` class for minimal Python↔Rust overhead
- `step_all_fast()` returns only numpy arrays, no Python dict construction
- Used `rand_chacha::ChaCha8Rng` for deterministic, seedable RNG
- **Result: 5 ep/s** — Rust sim was fast but Python training loop was now the bottleneck

### Step 3: Batched neural network inference
- Replaced 32 individual `select_action()` calls with one `select_actions_batch()`
- Single forward pass for all envs instead of 32 serial passes
- Global epsilon check (all explore or all exploit per tick)
- **Result: 5 → 23 ep/s** with step-based training every 4 ticks

### Step 4: Fast replay buffer
- Replaced PER (SumTree + namedtuples + Python loops) with numpy-backed uniform buffer
- `push()`: direct numpy array write vs namedtuple + SumTree O(log N) update
- `sample()`: `np.random.randint` + fancy indexing vs 256-iteration Python loop
- Added `push_batch()` to write all 128 experiences in one numpy operation
- **Trade-off**: Lost prioritized replay (rare events learn slower), gained ~50-100x buffer speed

### Step 5: Loop optimization
- 128 parallel envs (from 32)
- `step_all_fast()` — no Python dict construction per tick
- Vectorized `episode_rewards += rewards`
- Only loop over `done_idxs` (0-3 per tick, not 128)
- Bulk state update: `states[not_done] = next_states[not_done]`
- **Result: 88 ep/s peak, 20-24 ep/s sustained (longer episodes)**

### Final state
- **20-24 ep/s at 78-80% GPU utilization**
- 200K episodes in ~2.5 hours (down from 14+ hours)
- **22x total speedup**

## Training Observations

### Epsilon decay
- `0.9999` decay reaches floor (0.02) around episode 50K
- Agent shows steady improvement until ~ep 80K
- Plateau begins around avg score 65-73K, avg level 3.3-3.7

### Plateau breaking
- Fixed learning rate (1e-4) causes oscillation at plateau
- Reducing LR to 3e-5 and resuming from checkpoint helps fine-tune
- Without PER, the agent learns rare high-level events slower

### Score progression milestones
| Episodes | Avg Score | Avg Level | Best Level | Notes |
|----------|-----------|-----------|------------|-------|
| 1K | 5,000 | 1.0 | 1 | Random play |
| 5K | 8,000 | 1.0 | 2 | Learning to shoot |
| 10K | 21,000 | 1.7 | 4 | Clearing level 1 |
| 20K | 33,600 | 2.1 | 5 | Consistent level 2 |
| 50K | 47,000 | 2.7 | 7 | Epsilon at floor |
| 100K | 65,000 | 3.3 | 7 | Plateau begins |
| 140K | 66,000 | 3.3 | 7 | Plateau continues |

### PER vs Uniform Replay
- PER (old version): Higher sample efficiency, earlier level 2 (ep 1K vs 2K), but 50-100x slower buffer operations
- Uniform (new version): More episodes per hour compensates for lower per-episode efficiency
- At same wall time, uniform + fast pipeline produces better models despite lower per-episode quality

## Architecture Notes

### Rust game sim (`game_sim/`)
- 8 source files: constants, entities, game, movement, collision, spawning, state, lib
- PyO3 bindings expose `Game` (single) and `BatchedGames` (multi-env)
- `get_entities()` method returns full game state for TUI replay visualization
- Entity positions never cross FFI boundary during training — only 24-float state vector
- Borrow checker required index-based iteration in collision detection and monster movement

### DQN Configuration
- Network: 24 → 256 → 256 → 128 → 6
- Double DQN with Polyak soft target updates (tau=0.005)
- Batch size 256, train every 2 ticks
- Uniform replay buffer, 500K capacity
- 6 actions: idle, left, right, fire, fire+left, fire+right

### State representation (24 features)
1. Player X position
2. Player lives
3. Level
4. Enemy count
5-6. Nearest enemy (relative dx, dy)
7-8. Lowest enemy (relative dx, y)
9-11. Nearest enemy bullet (dx, dy, count)
12-14. Nearest missile (dx, dy, count)
15-17. Nearest kamikaze (dx, dy, count)
18-19. Monster position (dx, y)
20. Player invulnerability flag
21. Wall count
22-24. Nearest wall (dx, dy, health)

### Known limitations
- State only tracks "nearest" of each entity type — can't see multiple simultaneous threats
- No temporal information (velocity, acceleration of threats)
- Monster2's 8 movement patterns not directly observable from state
- Network may be too small for higher-level strategies

## Bug Found
Python `train.py` line ~442 had: `self.events.count(EventType.WALL_DESTROYED)` — comparing a list of dicts against a string, always returns 0. Wall destruction penalty never fired. Fixed in Rust port.

## Tools Built
- `visualize.py` — Live TUI training dashboard (rich), reads JSONL log, shows sparklines, system metrics, trend analysis
- `replay.py` — Side-by-side ASCII model comparison TUI, loads two checkpoints, runs games with same seed
- `export_model.py` — Converts .pt checkpoint to JSON weights for browser inference
