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

### Step 6: CUDA + GPU auto-scaling
- Moved training from Mac MPS to NVIDIA CUDA GPU
- Auto-scaling detects GPU memory and scales batch_size and num_envs accordingly
- 16+ GB VRAM: batch_size=2048, num_envs=512
- 8+ GB VRAM: batch_size=1024, num_envs=256
- **Result: 91 ep/s peak on CUDA** (vs 24 ep/s on MPS) with 76% GPU utilization
- VRAM usage still only ~4% with default 256→256→128 network — room for much larger models
- PyTorch `total_memory` vs `total_mem` attribute varies by version — need `getattr` fallback

### Final state
- **91 ep/s peak on CUDA, 20-24 ep/s on MPS**
- 200K episodes in ~2.5 hours on MPS, ~37 minutes on CUDA
- **22x speedup over original Python sim (MPS), ~90x on CUDA**

## Training Observations

### Epsilon decay
- `0.9999` decay reaches floor (0.02) around episode 50K
- Agent shows steady improvement until ~ep 80K
- Plateau begins around avg score 65-73K, avg level 3.3-3.7

### Plateau breaking
- Fixed learning rate (1e-4) causes oscillation at plateau
- Reducing LR to 3e-5 and resuming from checkpoint helps fine-tune
- Without PER, the agent learns rare high-level events slower
- **Critical bug**: resuming from checkpoint with default `epsilon_start=1.0` resets epsilon to 1.0, erasing all learned exploitation — agent plays randomly again. Fix: only override epsilon when config explicitly sets it below 1.0 (i.e., a mutation bumps it to 0.1-0.3, not back to 1.0)

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
- Network: configurable via `--hidden-sizes` (default: 512,256,128; optional `--layer-norm`)
- Double DQN with Polyak soft target updates (tau=0.005)
- Batch size configurable via `--batch-size` (default: 256, auto-scaled by GPU memory)
- Train every 4 ticks, 4 gradient steps per tick (Rust path)
- Dual-buffer PER: main uniform buffer + secondary buffer for important transitions (high |reward| or done)
- Cosine annealing LR schedule (cycles between lr_min=1e-5 and lr=3e-5 every 20K episodes)
- 5-step returns (default, configurable via `--n-step`)
- 6 actions: idle, left, right, fire, fire+left, fire+right

### State representation (45 features)
0. Player X position
1. Player lives
2. Level
3. Enemy count
4-5. Nearest enemy (relative dx, dy)
6-7. Lowest enemy (relative dx, y)
8-12. Nearest enemy bullet (dx, dy, count, velocity_dx, velocity_dy)
13-17. Nearest missile (dx, dy, count, cos(angle), sin(angle))
18-22. Nearest kamikaze (dx, dy, count, cos(angle), sin(angle))
23-24. Monster position (dx, y)
25-28. Monster2 position + velocity (dx, y, vel_dx, vel_dy)
29. Player invulnerability flag
30. Wall count
31-33. Nearest wall (dx, dy, health)
34-36. 2nd nearest bullet (dx, dy, velocity_dy)
37-39. 2nd nearest missile (dx, dy, sin(angle))
40-44. Danger heatmap: 5 columns (weighted threat density, bullets=1, missiles=2, kamikazes=3)

### Reward shaping
- Score-based: (score_delta) * 0.01
- Life penalty: -5.0 per life lost
- Game over penalty: -20.0
- Wall destruction penalty: -2.0 per wall destroyed
- Progressive survival bonus: +0.01 * current_level (scales with difficulty)
- Kamikaze kill bonus: +1.5 (extra on top of score reward)
- Missile shoot-down bonus: +2.0 (hardest threat to deal with)
- Dodging reward: +0.1 per near-miss (threat passes within 80px without hitting)

### Weight transfer across architectures
- When upgrading network size (e.g., 256→256→128 to 512→256→128), weights can be transferred
- Overlapping region is copied exactly, new neurons initialized with Kaiming uniform
- Gives the larger network a head start — plays at small-model level immediately, then improves
- `torch.compile()` adds `_orig_mod.` prefix to state dict keys — must normalize before transfer

### Previous known limitations (now addressed)
- ~~State only tracks "nearest" of each entity type~~ → now tracks top 2 bullets/missiles + danger heatmap
- ~~No temporal information (velocity, acceleration of threats)~~ → velocity features for bullets, missiles, kamikazes
- ~~Monster2's 8 movement patterns not directly observable~~ → Monster2 position + velocity now in state
- ~~Network may be too small~~ → default upgraded to 512,256,128 with optional LayerNorm

## Bugs Found
- Python `train.py` had: `self.events.count(EventType.WALL_DESTROYED)` — comparing a list of dicts against a string, always returns 0. Wall destruction penalty never fired. Fixed in Rust port and Python (now uses `e.get("type") ==` comparison).
- Resuming from checkpoint with default `epsilon_start=1.0` reset epsilon to 1.0, erasing learned exploitation. Fix: only override epsilon when config explicitly sets it below 1.0.

## Meta-Learning System

### Plateau detection (`PlateauDetector`)
- Compares three consecutive 5K-episode windows (old, mid, recent)
- Triggers when score change < 3% across windows, with secondary confirmation (level stagnant or hits stagnant)
- Anti-false-positive: won't trigger if recent best score exceeds all-time best by >10%
- Cooldown: minimum 10K episodes between triggers
- Warmup: minimum 15K episodes before first trigger

### Mutation bandit (`meta_train.py`)
- Multi-armed bandit selects hyperparameter mutations per cycle
- Always applies: `lr_reduce` + `epsilon_bump` (proven effective)
- Additional mutations sampled by weight: buffer_flush, batch_size_up, gamma_increase, tau_reduce
- Weights updated after each cycle: +50% for >5% improvement, -40% for regression
- State persisted in `models/meta_learning.json`

### Lessons learned
- Epsilon must NOT reset to 1.0 on resume — only bump to 0.1-0.3 via explicit mutation
- Fresh Adam optimizer on cycle restart is beneficial (stale momentum from plateau hurts)
- GPU auto-scaling attribute name varies by PyTorch version (`total_memory` vs `total_mem`)

## Tools Built
- `visualize.py` — Live TUI training dashboard (rich), reads JSONL log, shows sparklines, system metrics, trend analysis
- `replay.py` — Side-by-side ASCII model comparison TUI, loads two checkpoints, runs games with same seed
- `export_model.py` — Converts .pt checkpoint to JSON weights for browser inference
- `meta_train.py` — Meta-learning outer loop with plateau detection, mutation bandit, and cycle management
