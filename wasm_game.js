/**
 * wasm_game.js — WASM Physics Engine bridge for eyeBMinvaders.
 *
 * Loads the WasmGame module and provides a physics-tick interface that
 * game.js can call instead of running its own JS physics.
 *
 * Exports on window.wasmPhysics:
 *   .ready       — true once WASM loaded and WasmGame instantiated
 *   .game        — the WasmGame instance
 *   .tick(dt, action) — step physics, return render state
 *   .getState()  — 50-feature Float32Array for DQN AI
 *   .reset()     — reset for a new game
 *   .processEvents(events) — trigger sounds/animations from WASM events
 */

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let _wasmModule = null;   // The loaded wasm-bindgen JS module
let _wasmGame = null;     // WasmGame instance
let _wasmReady = false;   // True once init succeeds

// ---------------------------------------------------------------------------
// Color mapping for enemy rows (mirrors createEnemies in game.js)
// ---------------------------------------------------------------------------

const ENEMY_ROW_COLORS = ['red', 'orange', 'yellow', 'green', 'blue'];
const _enemyImageCache = {};

function _getEnemyImage(row) {
  const color = ENEMY_ROW_COLORS[row % ENEMY_ROW_COLORS.length];
  if (!_enemyImageCache[color]) {
    const img = new Image();
    img.src = `enemy-ship-${color}.svg`;
    _enemyImageCache[color] = img;
  }
  return _enemyImageCache[color];
}

// ---------------------------------------------------------------------------
// Sound helpers — reference the global sound objects from game.js
// ---------------------------------------------------------------------------

function _playSound(sound) {
  if (typeof isMuted !== 'undefined' && isMuted) return;
  if (!sound) return;
  try {
    sound.currentTime = 0;
    sound.play().catch(() => {});
  } catch (_e) {
    // Ignore sound errors
  }
}

function _playSoundFactory(createFn) {
  if (typeof isMuted !== 'undefined' && isMuted) return;
  if (typeof playSoundWithCleanup === 'function') {
    playSoundWithCleanup(createFn);
  }
}

// ---------------------------------------------------------------------------
// Event processing — maps WASM event types to game.js sounds/effects
// ---------------------------------------------------------------------------

function _processEvents(events) {
  if (!events || !Array.isArray(events)) return;

  for (const evt of events) {
    switch (evt.type) {
      case 'enemy_hit':
        // First hit sound (enemy not yet dead)
        if (typeof createExplosionSound === 'function') {
          _playSoundFactory(createExplosionSound);
        }
        break;

      case 'enemy_killed':
        // Use game.js createExplosion which handles sound + correct image format
        if (typeof createExplosion === 'function') {
          createExplosion(evt.x || 0, evt.y || 0);
        }
        break;

      case 'wall_hit':
        // Add damage hole to wallHits array for visual rendering
        if (typeof wallHits !== 'undefined' && evt.wallIndex !== undefined) {
          if (!wallHits[evt.wallIndex]) wallHits[evt.wallIndex] = [];
          wallHits[evt.wallIndex].push({
            x: Math.random() * 40 - 10,
            y: Math.random() * 30 - 5,
            rotation: Math.random() * Math.PI * 2,
            fromEnemy: !evt.fromPlayer,
          });
        }
        break;

      case 'player_hit':
        _playSound(typeof playerExplosionSound !== 'undefined' ? playerExplosionSound : null);
        break;

      case 'player_killed':
        _playSound(typeof playerExplosionSound !== 'undefined' ? playerExplosionSound : null);
        break;

      case 'game_over':
        _playSound(typeof gameOverSound !== 'undefined' ? gameOverSound : null);
        break;

      case 'level_complete':
        _playSound(typeof clearLevelSound !== 'undefined' ? clearLevelSound : null);
        break;

      case 'missile_destroyed':
        _playSound(typeof missileBoomSound !== 'undefined' ? missileBoomSound : null);
        if (typeof createExplosion === 'function') {
          createExplosion(evt.x || 0, evt.y || 0);
        }
        break;

      case 'missile_bonus':
        _playSound(typeof bonusSound !== 'undefined' ? bonusSound : null);
        // Trigger bonus animation
        if (typeof showBonusAnimation !== 'undefined') {
          showBonusAnimation = true;
          bonusAnimationStart = Date.now();
        }
        break;

      case 'bonus_life':
        _playSound(typeof newLifeSound !== 'undefined' ? newLifeSound : null);
        // Trigger life grant animation (same as JS physics path)
        if (typeof lifeGrant !== 'undefined') {
          lifeGrant = true;
          if (typeof animations !== 'undefined') {
            animations.lifeGrant = {
              startTime: Date.now(),
              startX: 1024 / 2,
              startY: 576 - 100,
            };
          }
        }
        break;

      case 'monster_hit':
        _playSound(typeof monsterDeadSound !== 'undefined' ? monsterDeadSound : null);
        if (typeof createExplosion === 'function') {
          createExplosion(evt.x || 0, evt.y || 0);
        }
        break;

      case 'kamikaze_spawned':
        _playSound(typeof kamikazeLaunchSound !== 'undefined' ? kamikazeLaunchSound : null);
        break;

      case 'kamikaze_killed':
        _playSound(typeof kamikazeExplosionSound !== 'undefined' ? kamikazeExplosionSound : null);
        if (typeof createExplosion === 'function') {
          createExplosion(evt.x || 0, evt.y || 0);
        }
        break;

      case 'wall_destroyed':
        if (typeof createWallGoneSound === 'function') {
          _playSoundFactory(createWallGoneSound);
        }
        break;

      case 'player_fired':
        _playSound(typeof playerShotSound !== 'undefined' ? playerShotSound : null);
        break;

      default:
        // Unknown event type — ignore
        break;
    }
  }
}

// ---------------------------------------------------------------------------
// Initialization — loads WASM module and creates WasmGame
// ---------------------------------------------------------------------------

async function _initWasmGame() {
  try {
    let wasmMod = null;
    const paths = [
      './wasm_agent/pkg/wasm_agent.js',
      './wasm_agent_pkg/wasm_agent.js',
    ];

    for (const path of paths) {
      try {
        wasmMod = await import(path);
        break;
      } catch (_e) {
        // Try next path
      }
    }

    if (!wasmMod) {
      console.warn('[WASM Physics] Could not load WASM module from any known path.');
      return false;
    }

    // wasm-bindgen --target web requires calling the default init()
    if (typeof wasmMod.default === 'function') {
      await wasmMod.default();
    }

    _wasmModule = wasmMod;

    // WasmGame constructor takes no arguments
    if (typeof wasmMod.WasmGame !== 'function') {
      console.warn('[WASM Physics] WasmGame class not found in module.');
      return false;
    }

    _wasmGame = new wasmMod.WasmGame();
    _wasmReady = true;

    console.log('[WASM Physics] WasmGame initialized successfully.');
    return true;

  } catch (err) {
    console.error('[WASM Physics] Initialization failed:', err);
    return false;
  }
}

// ---------------------------------------------------------------------------
// Public API — exposed on window.wasmPhysics
// ---------------------------------------------------------------------------

window.wasmPhysics = {
  /** True once WASM is loaded and WasmGame is ready */
  get ready() {
    return _wasmReady;
  },

  /** The raw WasmGame instance (for advanced use) */
  get game() {
    return _wasmGame;
  },

  /**
   * Step the WASM physics engine.
   * @param {number} dt — delta time in seconds
   * @param {number} action — action code 0-5:
   *   0=idle, 1=left, 2=right, 3=fire, 4=fire+left, 5=fire+right
   * @returns {object|null} — render state, or null if WASM not ready
   */
  tick(dt, action) {
    if (!_wasmReady || !_wasmGame) return null;
    try {
      return _wasmGame.tick(dt, action);
    } catch (err) {
      console.error('[WASM Physics] tick() error:', err);
      return null;
    }
  },

  /**
   * Get the 50-feature state vector for DQN AI inference.
   * @returns {Float32Array|null}
   */
  getState() {
    if (!_wasmReady || !_wasmGame) return null;
    try {
      return _wasmGame.get_state();
    } catch (err) {
      console.error('[WASM Physics] get_state() error:', err);
      return null;
    }
  },

  /**
   * Reset for a new game.
   */
  reset() {
    if (!_wasmReady || !_wasmGame) return;
    try {
      _wasmGame.reset();
    } catch (err) {
      console.error('[WASM Physics] reset() error:', err);
    }
  },

  /**
   * Reset to a specific level (for curriculum training).
   * @param {number} level
   */
  resetAtLevel(level) {
    if (!_wasmReady || !_wasmGame) return;
    try {
      _wasmGame.reset_at_level(level);
    } catch (err) {
      console.error('[WASM Physics] reset_at_level() error:', err);
    }
  },

  /**
   * Process WASM events to trigger sounds and animations.
   * @param {Array} events — array of {type, x, y, ...} from WASM tick
   */
  processEvents(events) {
    _processEvents(events);
  },

  /**
   * Lookahead: simulate top actions forward n_steps, return scores.
   * @param {number[]} actions — action codes to evaluate
   * @param {number} nSteps — ticks to simulate forward
   * @returns {Float32Array|null} — score per action
   */
  evaluate_actions(actions, nSteps) {
    if (!_wasmReady || !_wasmGame) return null;
    try {
      return _wasmGame.evaluate_actions(actions, nSteps);
    } catch (err) {
      console.error('[WASM Physics] evaluate_actions() error:', err);
      return null;
    }
  },

  // --- Online DQN Learning ---

  /** The WasmOnlineDQN instance (null until initialized) */
  get onlineDQN() { return _onlineDQN; },

  /** True if online learning is active */
  get learningActive() { return _onlineLearning; },

  /** Get online learning stats */
  get learningStats() { return _onlineStats; },

  /** Toggle online learning on/off */
  toggleLearning() {
    if (!_onlineDQN) {
      _initOnlineDQN();
    }
    _onlineLearning = !_onlineLearning;
    if (_onlineLearning) {
      _startOnlineTraining();
      console.log('[Online DQN] Learning enabled');
    } else {
      _stopOnlineTraining();
      console.log('[Online DQN] Learning paused');
    }
  },
};

// ---------------------------------------------------------------------------
// Online DQN — learns while playing
// ---------------------------------------------------------------------------

let _onlineDQN = null;
let _onlineLearning = false;
let _onlineStats = { episodes: 0, avgScore: 0, bestScore: 0, updates: 0, bufferSize: 0, loss: 0 };
let _onlineTrainId = null;

function _initOnlineDQN() {
  if (!_wasmModule || _onlineDQN) return;
  try {
    _onlineDQN = new _wasmModule.WasmOnlineDQN();
    // Load pretrained weights
    fetch('models/model_weights.json')
      .then(r => r.text())
      .then(json => {
        if (_onlineDQN.load_weights(json)) {
          console.log('[Online DQN] Loaded pretrained weights');
        }
      })
      .catch(() => console.log('[Online DQN] No pretrained weights, starting fresh'));
  } catch (e) {
    console.error('[Online DQN] Init failed:', e);
    _onlineDQN = null;
  }
}

function _onlineTrainCallback(deadline) {
  if (!_onlineLearning || !_onlineDQN) return;

  // Run DQN steps during idle time (max 4ms to avoid stutter)
  const start = performance.now();
  while (performance.now() - start < 4 && deadline.timeRemaining() > 1) {
    try {
      const stats = _onlineDQN.step();
      if (stats) {
        _onlineStats.episodes = stats.episodes || 0;
        _onlineStats.avgScore = stats.avgScore || 0;
        _onlineStats.bestScore = stats.bestScore || 0;
        _onlineStats.updates = stats.updates || 0;
        _onlineStats.bufferSize = stats.bufferSize || 0;
        _onlineStats.loss = stats.lastLoss || 0;
        _onlineStats.currentLevel = stats.currentLevel || 0;
        _onlineStats.totalSteps = stats.totalSteps || 0;
      }
    } catch (_e) { break; }
  }

  _onlineTrainId = requestIdleCallback(_onlineTrainCallback, { timeout: 50 });
}

function _startOnlineTraining() {
  if (_onlineTrainId !== null) return;
  _onlineTrainId = requestIdleCallback(_onlineTrainCallback, { timeout: 50 });
}

function _stopOnlineTraining() {
  if (_onlineTrainId !== null) {
    cancelIdleCallback(_onlineTrainId);
    _onlineTrainId = null;
  }
}

// ---------------------------------------------------------------------------
// Auto-init on page load
// ---------------------------------------------------------------------------

_initWasmGame().then((ok) => {
  if (ok) {
    console.log('[WASM Physics] Ready — wasmPhysics.ready =', window.wasmPhysics.ready);
    // Auto-init online DQN (loads in background)
    _initOnlineDQN();
  } else {
    console.log('[WASM Physics] Not available — falling back to JS physics.');
  }
});
