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
      case 'enemy_killed':
        // Use game.js createExplosion which handles sound + correct image format
        if (typeof createExplosion === 'function') {
          createExplosion(evt.x || 0, evt.y || 0);
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

      case 'missile_launched':
        if (typeof createMissileLaunchSound === 'function') {
          _playSoundFactory(createMissileLaunchSound);
        }
        break;

      case 'missile_destroyed':
        _playSound(typeof missileBoomSound !== 'undefined' ? missileBoomSound : null);
        break;

      case 'bonus':
        _playSound(typeof bonusSound !== 'undefined' ? bonusSound : null);
        break;

      case 'life_granted':
        _playSound(typeof newLifeSound !== 'undefined' ? newLifeSound : null);
        break;

      case 'monster_killed':
        _playSound(typeof monsterDeadSound !== 'undefined' ? monsterDeadSound : null);
        break;

      case 'kamikaze_launched':
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

      case 'player_shot':
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
};

// ---------------------------------------------------------------------------
// Auto-init on page load
// ---------------------------------------------------------------------------

_initWasmGame().then((ok) => {
  if (ok) {
    console.log('[WASM Physics] Ready — wasmPhysics.ready =', window.wasmPhysics.ready);
  } else {
    console.log('[WASM Physics] Not available — falling back to JS physics.');
  }
});
