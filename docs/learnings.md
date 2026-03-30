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
| `wasm_game.js` | Loads WasmGame, bridges WASM tick() to JS globals for rendering |
| `wasm_bridge.js` | PPO observation, turbo training dashboard, gameplay recording |
| `serve.py` | HTTP server + POST /api/gameplay for recording transitions |

## WASM Physics Replacement: Eliminating the Sim-to-Real Gap (March 29, 2025)

### The Problem We Solved

The #1 bottleneck was never the RL algorithm — it was the **sim-to-real gap**. Three independent implementations of the same game physics (Rust training, JS browser, Python fallback) accumulated dozens of subtle divergences:
- Bullet hitbox 3.4x5.9 vs 5x10
- Monster2 patterns different between Rust and JS
- Score values mismatched (40 vs 10 per enemy)
- Bullet velocity reported as wrong direction
- Threat urgency 1000x scale error
- Frame rate mismatch (60Hz browser vs 30Hz training)
- And many more discovered over 2 days of debugging

### The Solution

Compile `game_sim_core` (Rust) to **both** native (for GPU training) and WASM (for browser). One source of truth, zero gap.

```
game_sim_core (Rust)
  ├── wasm32-unknown-unknown → WasmGame (browser physics)
  └── native + PyO3 → game_sim (GPU training)
```

### Architecture

- `game_sim_core/` — Pure Rust game simulation, no FFI. ~800 lines.
- `wasm_agent/` — WASM wrapper exposing WasmGame + PPO agent. 260KB binary.
- `wasm_game.js` — Loads WASM, calls `wasmGame.tick(dt, action)` each frame, syncs entity state to JS globals for rendering.
- `game.js` — WASM bypass at top of gameLoop(). If WASM available, skips all JS physics. Falls back to legacy JS if WASM fails.

### Key Design Decisions

1. **Fixed-timestep accumulator**: WASM tick() runs at 33.333ms intervals regardless of browser frame rate. Time accumulates between frames. Prevents 2x speed at 60fps.

2. **Event-driven sound/animation**: WASM emits `RenderEvent`s (EnemyKilled{x,y}, PlayerHit, WallHit{wall_index,x,y}, etc.) during collision detection. JS processes events array for sounds and explosion visuals.

3. **Legacy fallback**: All JS physics code stays in game.js. If WASM fails to load (missing pkg, old browser), the game works exactly as before. Zero risk deployment.

4. **Entity image mapping**: WASM returns position/state data, JS adds image references from pre-loaded sprites. Each entity type maps to the correct SVG via helpers (_wasmGetEnemyImage for row-based colors, missileImage, monsterImage, etc.)

### Integration Issues Found and Fixed

| Issue | Cause | Fix |
|-------|-------|-----|
| Game ran at 2x speed | tick() called every frame (60fps × 33ms = 2x) | Fixed-timestep accumulator |
| Lives never updated | Read from state.player.lives (wrong path) | Changed to state.lives |
| No sounds played | 5 event name mismatches between Rust and JS | Aligned all strings |
| Explosions at (0,0) | EventType→RenderEvent mapped without positions | Emit RenderEvents directly from collision code |
| R key didn't reset WASM | restartGame() only reset JS globals | Added wasmPhysics.reset() |
| No wall damage | WallHit events not emitted from collision code | Added to all 3 collision points |
| Missing entity images | WASM sync didn't include sprite references | Added image mapping for all entity types |
| drawImage TypeError | Explosions pushed with wrong field format | Use createExplosion() from game.js |

### Training Results with Zero Gap

The first training run on the aligned sim showed the fastest learning ever:

| Episode | Avg Score | Avg Level | Best Level |
|---------|-----------|-----------|------------|
| 15K | 44K | 2.7 | 6 |
| 20K | 58K | 3.2 | 6 |
| 25K | 63K | 3.9 | 7 |
| 40K (peak) | **74K** | **4.6** | **8** |

Previous best (with sim-to-real gap): avg 84K in Rust, but only avg level 2-3 in browser.
With zero gap: avg 74K in Rust AND in browser — model transfers perfectly.

### What We Learned

1. **The sim-to-real gap was the real enemy, not the algorithm.** We tried: reward shaping, god mode, safety layers, proximity penalties, edge penalties, curriculum learning, fine-tuning. None of these moved the needle more than fixing the physics mismatches.

2. **One source of truth > two aligned sources.** We spent days finding and fixing individual mismatches. The crate split + dual compilation eliminated the entire class of bugs in one architectural change.

3. **WASM performance is sufficient.** 260KB binary, <1ms per tick, runs at 60fps with time left over. No need for Web Workers or SharedArrayBuffer for the game itself.

4. **Browser fine-tuning needs zero gap to work.** Earlier attempts at JS headless fine-tuning degraded the model because the JS game was subtly different from training. With WASM physics, the PPO trains on identical code.

5. **Fixed-timestep matters.** Variable deltaTime causes training/inference divergence. The accumulator pattern (only tick at 33ms intervals) was essential.

## PPO Training: Breaking the DQN Plateau (March 2025)

### Why PPO > DQN for This Game

DQN plateaued at avg level 4.4-4.6 despite 500K+ episodes. The fundamental limitation: DQN is off-policy — the replay buffer mixes stale transitions from easy early levels with fresh high-level experience, diluting gradient signal. Curriculum learning made it worse because the replay buffer still contained mismatched experiences.

PPO is on-policy — every gradient update uses only fresh experience from the current policy. This naturally solves the stale-experience problem for difficulty-scaling games.

### PPO Evolution

| Version | Avg Level | Avg Score | Best Level | Best Score | Key Changes |
|---------|-----------|-----------|------------|------------|-------------|
| DQN best | 4.6 | 85K | 8 | 155K | Dueling+NoisyNet+4frames, 500K eps |
| PPO v1 | 5.3 | 107K | 8 | 166K | Basic actor-critic [256,128], obs norm, 90K eps |
| PPO v2 (curriculum) | 3.3 | 34K | 5 | 85K | Aggressive curriculum — too fast advancement |
| PPO v3 (tuned curriculum) | 3.3 | 39K | 5 | 92K | 90% threshold, 500-ep window — still hurt |
| PPO v4 (no curriculum) | 4.0 | 80K | 8 | 139K | Level-conditioned heads, no curriculum |
| **PPO v5** | **5.8** | **116K** | **9** | **170K** | Bigger backbone [512,256,128], continuous level conditioning, entropy decay |

### Key Learnings

1. **PPO throughput is 4-5x DQN** on the same hardware (80+ ep/s vs 18 ep/s) due to no replay buffer overhead and simpler update logic.

2. **Curriculum learning failed again** — even with PPO. The AggressiveCurriculum (advance when 80-90% clear rate) caused the agent to be thrown into levels it wasn't ready for. Plain training from level 1 every episode consistently outperformed curriculum variants. **The learnings doc was right: "duration matters more than complexity."**

3. **Bigger backbone matters at high levels.** [256,128] plateaued at avg 5.3; [512,256,128] pushed to 5.8 and reached level 9. The extra capacity handles the complexity of simultaneous fast enemies, kamikazes, missiles, and Monster2 chase patterns at levels 7+.

4. **Entropy decay is critical for PPO.** Starting at 0.02 and decaying to 0.005 over training gives exploration early on and decisive action selection (tight policy) at high levels where split-second dodging matters. Without decay, the policy stays too stochastic.

5. **Continuous level conditioning > bucket one-hot.** A single continuous level/10 value appended to backbone features before the policy/value heads scales to any level without bucket boundary artifacts. The 3-bucket approach [1-3, 4-6, 7+] made levels 7-10 indistinguishable.

6. **Observation normalization (RunningMeanStd) is essential for PPO.** Without it, different feature scales cause unstable training. Welford's online algorithm with [-10, 10] clamping works well.

### Remaining Failure Modes (Level 8+)

1. **Enemies reaching the bottom** — at high levels enemies move fast (1.33^N speed multiplier) and step down frequently. The agent doesn't prioritize clearing enemies urgently enough. Fixed with enemy descent pressure reward (v6).

2. **Dense threat death** — too many bullets + kamikazes converging simultaneously overwhelm the agent. Fixed with dense threat survival bonus (v6).

3. **Red diagonal bullets (Monster2)** — Monster2 chase pattern at level 8+ fires directional bullets that are harder to dodge than straight-down fire. The state vector tracks these but the agent may need more training time at these levels.

### PPO Architecture (v5)

```
Input: 216 features (4 frames × 54 features)

Shared Backbone:
  Linear(216 → 512) + ReLU
  Linear(512 → 256) + ReLU
  Linear(256 → 128) + ReLU

[concat continuous level value → 129 features]

Policy Head:
  Linear(129 → 64) + ReLU
  Linear(64 → 6) → action logits

Value Head:
  Linear(129 → 64) + ReLU
  Linear(64 → 1) → state value
```

### PPO Hyperparameters (v5)

| Parameter | Value |
|-----------|-------|
| lr | 3e-4 → 1e-5 (cosine decay over 5000 updates) |
| clip_epsilon | 0.2 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| entropy_coeff | 0.02 → 0.005 (linear decay) |
| value_coeff | 0.5 |
| n_epochs | 4 |
| minibatch_size | 4096 (GPU auto-scaled) |
| rollout_length | 128 steps × 2048 envs = 262K transitions/update |
| n_frames | 4 |
| obs_norm | ON (RunningMeanStd, clamp [-10, 10]) |

### Reward Shaping (v6 additions)

| Signal | Value | Purpose |
|--------|-------|---------|
| Enemy descent pressure | -0.03 * danger (0 to 1 scale) | Gradient warning as enemies approach walls |
| Dense threat survival | +0.02 * (threats - 3), capped at +0.10 | Reward surviving under 4+ simultaneous threats |
