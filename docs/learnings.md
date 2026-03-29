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

### Score progression milestones (Dueling+NoisyNet+2frames, 50-feat state)
| Episodes | Avg Score | Avg Level | Best Level | Best Score | Notes |
|----------|-----------|-----------|------------|------------|-------|
| 1K | 5,000 | 1.0 | 1 | 13K | Random play |
| 5K | 10,000 | 1.1 | 2 | 27K | Learning to shoot |
| 10K | 18,000 | 1.6 | 3 | 51K | Clearing level 1 |
| 20K | 34,000 | 2.1 | 5 | 84K | Consistent level 2 |
| 50K | 60,000 | 3.2 | 7 | 127K | Epsilon at floor |
| 100K | 65,000 | 3.4 | 7 | 141K | Old plateau zone |
| 200K | 62,000 | 3.7 | 8 | 153K | Level 8 reached |
| 292K | 59,000 | 3.5 | 8 | 153K | Training stopped (CPU) |

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

### State representation (50 features)
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
45. Enemy speed (normalized, critical for high-level play)
46. Enemy direction (-1 left, +1 right, normalized)
47. Fire cooldown (0=can fire now, 1=just fired)
48. Threat urgency (min time-to-impact across bullets + kamikazes, lower=more dangerous)
49. Enemies in bottom half (danger level, enemies close to walls/player)

### Reward shaping
- Score-based: (score_delta) * 0.01
- Life penalty: -5.0 per life lost
- Game over penalty: -20.0
- Wall destruction penalty: -2.0 per wall destroyed
- Progressive survival bonus: +0.01 * current_level (scales with difficulty)
- Kamikaze kill bonus: +1.5 (extra on top of score reward)
- Missile shoot-down bonus: +2.0 (hardest threat to deal with)
- Dodging reward: +0.1 per near-miss (threat passes within 80px without hitting)
- Level completion bonus: +5.0 + 3.0 * level (scales with progression)

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
- ALWAYS_APPLY emptied (lr_reduce was always applied before, killing LR after a few cycles)
- 12 mutations: lr_reduce, lr_boost, epsilon_bump, epsilon_bump_large, buffer_flush, batch_size_up, gamma_increase, tau_reduce, n_step_change, cosine_reset, train_steps_up, num_envs_up
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

## Breaking the Level 7-8 Plateau (March 2025)

### Changes Made (rust branch)

1. **Dueling DQN + NoisyNet + Frame Stacking** — new architecture replacing standard DQN
2. **Rayon parallelization** — `par_iter_mut` for `reset_all`, `step_all`, `step_all_fast`
3. **50-feature state vector** — added enemy speed, fire cooldown, threat urgency, enemy direction, bottom-half enemy count
4. **Scaled level completion bonus** — `+5 + 3*level` instead of flat `+5`
5. **Curriculum learning support** — `reset_at_level()` in Rust sim (available but not recommended without tuning)
6. **Per-env epsilon fix** — critical bug where all envs explored/exploited simultaneously
7. **Meta-train mutation overhaul** — removed always-applied lr_reduce, added lr_boost, cosine_reset, n_step_change, train_steps_up, num_envs_up

### Experiments Run on CPU (32-core, no GPU)

| Run | Config | Best Level | Best Score | Avg Score | Episodes | Wall Clock |
|-----|--------|-----------|-----------|-----------|----------|------------|
| 1 | 45-feat, 256-128-64, 2-frame, original reward | **8** | **153K** | 59K | 292K | ~7 hours |
| 2 | 45-feat, 256-128-64, scaled death penalty | 6 | 114K | 38K | 68K | ~2 hours |
| 3 | 45-feat, 512-256-128, 4-frame | 5 | 112K | 47K | 54K | ~1.5 hours |
| 4 | 50-feat, 256-128-64, original reward | 7 | 135K | 55K | 95K | ~1.5 hours |
| 5 | 50-feat, 256-128-64, curriculum+clipping | 7 | 135K | 33K | 132K | ~1.5 hours |
| 6 | 50-feat, 256-128-64, proven config (still running when stopped) | 8 | 139K | 63K | 40K | ~30 min |

**Key takeaway**: Run 1 (simplest config, longest duration) performed best. Duration matters more than complexity on CPU.

### GPU Training Plan (Level 9-10 Target)

**Recommended config for GPU with 16+ GB VRAM:**
```python
cfg = TrainingConfig(
    train_steps_per_tick=4,       # full gradient steps
    train_every=4,
    batch_size=2048,              # large batch for GPU
    n_frames=4,                   # full 4-frame stacking
    hidden_sizes=[512, 256, 128], # full network
    buffer_capacity=1_000_000,    # 1M buffer
    use_dueling=True,
    use_noisy=True,
    lr=1e-4,
    n_step=5,
    num_envs=2048,                # rayon handles this easily
)
```

**Expected timeline on GPU:**
- Level 8 reliable: ~1-2 hours
- Level 9 occasional: ~3-5 hours
- Level 10: ~10+ hours (500K+ episodes)

**Future improvements to implement on GPU:**
1. True PER (SumTree) — highest impact, 15-30% sample efficiency gain
2. C51 distributional DQN — risk-aware decisions at high levels
3. RND exploration — directed exploration toward novel high-level states
4. Curriculum learning with separate buffers — needs careful tuning

## Sim-to-Real Gap: Closing the Rust ↔ Browser Divide (March 29, 2025)

### Critical Bugs Found and Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| Bullet velocity: `dy:-5` made JS report bullets going UP | Agent didn't dodge | Check `isMonster2Bullet` for directional bullets |
| Threat urgency TTI: 1000x scale error in JS | Every bullet appeared infinitely far | Match Rust formula `speed*1000/60` |
| Frame rate: browser 60Hz vs training 30Hz | Temporal patterns halved | 30Hz throttle via `performance.now()` gate |
| Frame buffer init: first 3 frames all zeros | Garbage Q-values on game start | Call `dqnResetFrameBuffer()` on init |
| Level transition: `setTimeout(createEnemies, 1500)` | Enemies never spawned for level 2+ in headless | Mock setTimeout with mock clock |
| `bIndex`/`bulletIndex` variable mismatch | Crash in Node.js, silent fail in browser | Rename to consistent `bIndex` |
| Bullet hitbox: Rust 3.4x5.9 vs JS 5x10 | Agent's dodge margins 47% too tight | Changed Rust to 5x10 to match JS |
| Bullet X-bounds filter missing in JS | Phantom off-screen bullets | Added `bullet.x > 0 && bullet.x < canvas.width` |
| Monster2 bullets: JS moved all straight down | Spread pattern didn't work | Added `dx/dy` directional movement for Monster2 bullets |
| Monster2 patterns: JS levels 7+ all 'random' | Teleport/chase never used | Added levels 7-9 to MONSTER2_PATTERNS dict |
| FIRE_RATE: JS used 0.190 vs Rust 0.16 | Fire cooldown feature misaligned | Changed JS to 0.16 |

### What DID NOT Work

#### Heuristic Safety Layer on DQN Actions
Added rules to override DQN decisions (wall-fire blocking, empty-space-fire blocking, emergency dodge). Every version caused problems:
- **Full fire masking** (actions 3,4,5): crippled dodging since model uses fire+move as primary movement
- **Emergency dodge override**: pinned agent to edges (always dodging left → stuck at left wall)
- **Center-biased dodge**: better but still interfered with learned Q-value balance
- **Monster dodge override**: same edge-pinning issue
- **Lesson**: Don't override a neural network's actions with heuristics. The network's Q-values encode complex tradeoffs that simple rules can't replicate. Fix the training instead.

#### God Mode Training
Concept: hits penalized but agent never dies. Should learn avoidance from massive hit exposure.
- **Bug**: initial implementation didn't detect hits (lives never decremented in god mode, so `player_lives < old_lives` was never true). Fixed by checking `PlayerHit` events.
- **Too-harsh penalty** (-15 base, escalating): agent learned to hide in corners (89% edge time)
- **Too-mild penalty** (-5): agent learned hits don't matter, played recklessly
- **Episode length**: 15K steps with 2048 envs = episodes never completed (hours per batch). Reduced to 6K.
- **Result**: avg level 2.1 in god mode, avg level 1.4 in normal mode. The fundamental problem: god mode teaches "survive hits" not "avoid hits". The agent learns it can absorb damage, which transfers poorly to normal play where 6 hits = game over.
- **Lesson**: God mode may work with much more careful tuning, but normal training with correct hitbox is simpler and more reliable.

#### Aggressive Reward Shaping (proximity penalties, harsh wall penalty)
- Wall penalty at -3.0 per hit (was -0.5): made agent too conservative, stopped shooting
- Bullet proximity penalty (-0.15 within 80px): made agent timid, reduced score
- Kamikaze/missile proximity penalties: same issue
- Removing near-miss reward (+0.1): removed a useful learning signal
- **Combined effect**: avg level dropped from 5.0 to 1.1
- **Lesson**: The original mild reward function (hit -5, death -20, wall destroy -2, wall hit -0.5, near-miss +0.1, level complete +5+3*level) is the proven winner. Don't over-penalize.

#### JS Headless Fine-Tuning
- At 1e-7 LR: model slowly forgets (avg drops over 90 episodes)
- At 5e-8 LR: model stable but doesn't improve (too slow to learn)
- At 3e-6 LR: catastrophic forgetting in 20 episodes
- 0.3 ep/s is too slow for meaningful training (~5K episodes/hour)
- Single-env experience is too correlated for gradient updates
- **Lesson**: Fine-tuning a 2048-env trained model with single-env JS experience doesn't work well. The experience distribution is too different.

### Best Model: Run 6 ep180K

The best performing model across all experiments:
- **Training**: Normal mode, old 3.4x5.9 hitbox, mild rewards, RTX 5090
- **Rust sim**: avg 84K score, avg level 3.9, best level 7
- **JS headless**: avg level 4.1, best level 5, edge time 16%
- **Browser**: reaches level 3-4 visually

### Remaining Sim-to-Real Mismatches (to fix in Rust sim)

| Issue | Description | Priority |
|-------|-------------|----------|
| Monster2 velocity [27-28] | Always [0,0] for 6/7 movement patterns (spiral, zigzag, figure8, wave, chase, teleport) | HIGH |
| Monster2 bounce randomness | JS has random 30% direction flips every 1.5-3s, Rust is deterministic | MEDIUM |
| Monster2 chase prediction | JS leads player by ±100px based on movement keys, Rust uses raw position | MEDIUM |
| Monster2 teleport | JS instant jump every 2s, Rust smooth slide every 1s | MEDIUM |
| Monster2 zigzag oscillation | JS has extra `sin(phase*2)*dt*15` vertical bob, Rust doesn't | LOW |
| Monster missile size | Verify JS monster-spawned missiles match Rust (both should be 44x44) | LOW |

### Key Architectural Insight

The **sim-to-real gap is the #1 bottleneck**, not the RL algorithm. The model trains well in Rust (avg level 7-8) but loses 2-4 levels in the browser. Every constant mismatch, physics difference, and feature computation error compounds. The most impactful work is aligning the Rust sim to exactly match game.js, not improving the DQN architecture.

### Tools Built

| Tool | Purpose |
|------|---------|
| `headless/run.js` | Run game.js in Node.js at 30Hz with mock clock, test DQN models |
| `headless/browser-shim.js` | Canvas/DOM/Image/Audio/timing mocks for Node.js |
| `headless/env_worker.js` | JSON stdin/stdout game worker for Python fine-tuning |
| `headless/finetune.py` | DQN fine-tuning against real JS game (with god mode support) |
| `browser_validate.py` | Playwright browser validation — runs AI in real Chrome |
