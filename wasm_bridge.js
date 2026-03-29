/**
 * wasm_bridge.js — JavaScript bridge for the WASM PPO agent in eyeBMinvaders.
 *
 * Dynamically loads the wasm_agent WASM module, initializes a WasmAgent,
 * and provides functions that the game loop can call to let the PPO agent
 * drive gameplay and train in the browser.
 *
 * Key bindings:
 *   F2  — Toggle WASM PPO agent on/off
 *   F3  — Toggle turbo training mode
 *   F4  — Export trained weights to JSON download
 *
 * Does NOT modify game.js or index.html. All integration is additive.
 */

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

let wasmModule = null;       // The loaded wasm-bindgen JS module
let wasmAgent = null;        // WasmAgent instance
let wasmReady = false;       // True once init succeeds
let wasmActive = false;      // True while the WASM PPO agent is driving the game
let turboMode = false;       // Turbo training via requestIdleCallback
let turboMultiplier = 10;    // How many extra train_steps per idle callback
let learningEnabled = true;  // Whether the agent is learning (updating weights)

// Stats from the agent, refreshed each frame
let agentStats = {
  episode: 0,
  avgReward: 0.0,
  policyLoss: 0.0,
  entropy: 0.0,
  totalSteps: 0,
  bestReward: 0.0,
};

// HUD DOM reference
let hudElement = null;

// ---------------------------------------------------------------------------
// Default agent configuration
// ---------------------------------------------------------------------------

const DEFAULT_AGENT_CONFIG = {
  // Game dimensions
  game_width: 1024,
  game_height: 576,

  // PPO hyperparameters
  learning_rate: 3e-4,
  gamma: 0.99,
  gae_lambda: 0.95,
  clip_epsilon: 0.2,
  entropy_coeff: 0.01,
  value_coeff: 0.5,
  max_grad_norm: 0.5,

  // Network architecture
  hidden_sizes: [256, 256, 128],

  // Training
  batch_size: 64,
  n_epochs: 4,
  rollout_length: 2048,

  // Actions: idle, left, right, fire, fire+left, fire+right
  n_actions: 6,
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/**
 * Loads the WASM module and creates a WasmAgent instance.
 * Call this once, e.g. on page load or on first F2 press.
 * Returns true on success, false on failure.
 */
async function initWasmAgent(configOverrides) {
  if (wasmReady) return true;

  try {
    // Try the pkg output directory first, then the copied location
    let mod = null;
    const paths = [
      './wasm_agent/pkg/wasm_agent.js',
      './wasm_agent_pkg/wasm_agent.js',
    ];

    for (const path of paths) {
      try {
        mod = await import(path);
        break;
      } catch (_e) {
        // Try next path
      }
    }

    if (!mod) {
      console.warn('[WASM Bridge] Could not load WASM module from any known path.');
      _showHudMessage('WASM module not found. Run wasm_agent/build.sh first.', 'error');
      return false;
    }

    // wasm-bindgen --target web requires calling the default init()
    if (typeof mod.default === 'function') {
      await mod.default();
    }

    wasmModule = mod;

    // Merge config
    const config = Object.assign({}, DEFAULT_AGENT_CONFIG, configOverrides || {});
    const configJson = JSON.stringify(config);

    wasmAgent = new mod.WasmAgent(configJson);
    wasmReady = true;

    console.log('[WASM Bridge] Agent initialized successfully.');
    _createHud();
    _updateHud();
    return true;

  } catch (err) {
    console.error('[WASM Bridge] Initialization failed:', err);
    _showHudMessage('WASM init failed: ' + err.message, 'error');
    return false;
  }
}

// ---------------------------------------------------------------------------
// Per-frame update — called from the game loop
// ---------------------------------------------------------------------------

/**
 * Runs one PPO agent step. Returns an action index (0-5) that the caller
 * can translate into player input, or -1 if the agent is inactive.
 *
 * The caller is responsible for building the observation and passing it in.
 * If the WASM agent manages its own internal sim, call with no args and it
 * returns a full state object from step().
 */
function updateWasmAgent() {
  if (!wasmReady || !wasmActive || !wasmAgent) return null;

  try {
    // step() runs one tick inside the WASM sim and returns a JS object
    // with all entity data plus the chosen action.
    const result = wasmAgent.step();

    // Parse if it came back as a string
    const state = (typeof result === 'string') ? JSON.parse(result) : result;

    // Apply the WASM simulation state to the JS game globals so the
    // existing rendering code draws everything correctly.
    applyWasmStateToGame(state);

    // Refresh stats periodically (not every frame for perf)
    if (state.step_count % 30 === 0) {
      _refreshStats();
      _updateHud();
    }

    return state;

  } catch (err) {
    console.error('[WASM Bridge] step() error:', err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// State mapping — WASM result -> JS game globals
// ---------------------------------------------------------------------------

/**
 * Maps the entity data returned by WasmAgent.step() back onto the global
 * variables that game.js rendering functions read.
 *
 * This is intentionally defensive: missing fields are silently skipped
 * so partial state updates still work.
 */
function applyWasmStateToGame(state) {
  if (!state) return;

  // --- Player ---
  if (state.player) {
    if (typeof window.player !== 'undefined') {
      window.player.x = state.player.x;
      window.player.y = state.player.y;
      window.player.width = 48;
      window.player.height = 48;
      if (state.player.lives !== undefined) {
        window.player.lives = state.player.lives;
      }
      // Keep the existing image reference — the renderer owns it
      if (typeof playerNormalImage !== 'undefined' && window.player.image !== playerNormalImage) {
        window.player.image = playerNormalImage;
      }
    }
  }

  // --- Enemies ---
  if (Array.isArray(state.enemies)) {
    // Rebuild the enemies array. The renderer iterates over it each frame.
    const mapped = state.enemies.map((e) => {
      const enemy = {
        x: e.x,
        y: e.y,
        width: e.width || 43,
        height: e.height || 43,
        hits: e.hits || 0,
        image: new Image(),
      };
      // Pick a color SVG based on the row hint, or default to red
      const colors = ['red', 'orange', 'yellow', 'green', 'blue'];
      const color = colors[(e.row || 0) % colors.length];
      enemy.image.src = 'enemy-ship-' + color + '.svg';
      return enemy;
    });
    // Replace the global array in-place so any external references stay valid
    if (typeof enemies !== 'undefined') {
      enemies.length = 0;
      enemies.push(...mapped);
    }
  }

  // --- Bullets ---
  if (Array.isArray(state.bullets)) {
    const mapped = state.bullets.map((b) => ({
      x: b.x,
      y: b.y,
      isEnemyBullet: !!b.is_enemy_bullet,
      dx: b.dx || 0,
      dy: b.dy || 0,
      isMonster2Bullet: !!b.is_monster2_bullet,
    }));
    if (typeof bullets !== 'undefined') {
      bullets.length = 0;
      bullets.push(...mapped);
    }
  }

  // --- Kamikaze enemies ---
  if (Array.isArray(state.kamikaze_enemies)) {
    const mapped = state.kamikaze_enemies.map((k) => ({
      x: k.x,
      y: k.y,
      width: k.width || 43,
      height: k.height || 43,
      angle: k.angle || 0,
    }));
    if (typeof kamikazeEnemies !== 'undefined') {
      kamikazeEnemies.length = 0;
      kamikazeEnemies.push(...mapped);
    }
  }

  // --- Homing missiles ---
  if (Array.isArray(state.homing_missiles)) {
    const mapped = state.homing_missiles.map((m) => ({
      x: m.x,
      y: m.y,
      angle: m.angle || 0,
      width: 57,
      height: 57,
      time: m.time || 0,
    }));
    if (typeof homingMissiles !== 'undefined') {
      homingMissiles.length = 0;
      homingMissiles.push(...mapped);
    }
  }

  // --- Walls ---
  if (Array.isArray(state.walls)) {
    const mapped = state.walls.map((w) => ({
      x: w.x,
      y: w.y,
      width: w.width || 58,
      height: w.height || 23,
      hitCount: w.hit_count || 0,
      missileHits: w.missile_hits || 0,
      image: (typeof wallImage !== 'undefined') ? wallImage : new Image(),
    }));
    if (typeof walls !== 'undefined') {
      walls.length = 0;
      walls.push(...mapped);
    }
  }

  // --- Monster (boss 1) ---
  if (state.monster !== undefined) {
    if (state.monster === null) {
      if (typeof window !== 'undefined') window.monster = null;
    } else {
      const m = state.monster;
      if (typeof window !== 'undefined') {
        window.monster = {
          x: m.x,
          y: m.y,
          width: m.width || 56,
          height: m.height || 56,
          hit: !!m.hit,
          direction: m.direction || 1,
          speed: m.speed || 3,
        };
      }
    }
  }

  // --- Monster2 (boss 2) ---
  if (state.monster2 !== undefined) {
    if (state.monster2 === null) {
      if (typeof window !== 'undefined') window.monster2 = null;
    } else {
      const m2 = state.monster2;
      if (typeof window !== 'undefined') {
        window.monster2 = {
          x: m2.x,
          y: m2.y,
          width: m2.width || 56,
          height: m2.height || 56,
          spiralAngle: m2.spiral_angle || 0,
          pattern: m2.pattern || 'bounce',
          isDisappeared: !!m2.is_disappeared,
        };
      }
    }
  }

  // --- Scalar game state ---
  if (state.score !== undefined && typeof score !== 'undefined') {
    // Update via window to be safe with var/let scoping
    window.score = state.score;
    // Also try direct assignment for let-scoped globals in game.js
    try { score = state.score; } catch (_e) { /* let-scoped, window fallback */ }
  }

  if (state.current_level !== undefined) {
    try { currentLevel = state.current_level; } catch (_e) { /* fallback */ }
  }

  if (state.game_over !== undefined) {
    try { gameOverFlag = state.game_over; } catch (_e) { /* fallback */ }
  }
}

// ---------------------------------------------------------------------------
// Turbo training — extra train_steps during idle time
// ---------------------------------------------------------------------------

let _turboCallbackId = null;

function _turboTrainCallback(deadline) {
  if (!turboMode || !wasmReady || !wasmAgent) return;

  // Use remaining idle time to squeeze in training steps
  while (deadline.timeRemaining() > 2) {
    try {
      wasmAgent.train_steps(turboMultiplier);
    } catch (err) {
      console.error('[WASM Bridge] turbo train error:', err);
      turboMode = false;
      _updateHud();
      return;
    }
  }

  // Schedule next idle callback
  _turboCallbackId = requestIdleCallback(_turboTrainCallback, { timeout: 100 });
}

function _startTurbo() {
  if (_turboCallbackId !== null) return;
  _turboCallbackId = requestIdleCallback(_turboTrainCallback, { timeout: 100 });
}

function _stopTurbo() {
  if (_turboCallbackId !== null) {
    cancelIdleCallback(_turboCallbackId);
    _turboCallbackId = null;
  }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

function _refreshStats() {
  if (!wasmReady || !wasmAgent) return;
  try {
    const raw = wasmAgent.get_stats();
    const s = (typeof raw === 'string') ? JSON.parse(raw) : raw;
    if (s) {
      agentStats.episode = s.episode || 0;
      agentStats.avgReward = s.avg_reward || 0;
      agentStats.policyLoss = s.policy_loss || 0;
      agentStats.entropy = s.entropy || 0;
      agentStats.totalSteps = s.total_steps || 0;
      agentStats.bestReward = s.best_reward || 0;
    }
  } catch (_e) {
    // Stats are best-effort
  }
}

// ---------------------------------------------------------------------------
// Weight import / export
// ---------------------------------------------------------------------------

function exportWasmWeights() {
  if (!wasmReady || !wasmAgent) {
    console.warn('[WASM Bridge] Agent not ready, cannot export.');
    return;
  }
  try {
    const weightsJson = wasmAgent.export_weights();
    const blob = new Blob([weightsJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ppo_weights_ep' + agentStats.episode + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    _showHudMessage('Weights exported.', 'success');
  } catch (err) {
    console.error('[WASM Bridge] Export failed:', err);
    _showHudMessage('Export failed: ' + err.message, 'error');
  }
}

function importWasmWeights(jsonString) {
  if (!wasmReady || !wasmAgent) {
    console.warn('[WASM Bridge] Agent not ready, cannot import.');
    return false;
  }
  try {
    wasmAgent.load_weights(jsonString);
    _showHudMessage('Weights loaded.', 'success');
    return true;
  } catch (err) {
    console.error('[WASM Bridge] Import failed:', err);
    _showHudMessage('Import failed: ' + err.message, 'error');
    return false;
  }
}

// ---------------------------------------------------------------------------
// HUD overlay
// ---------------------------------------------------------------------------

function _createHud() {
  if (hudElement) return;

  hudElement = document.createElement('div');
  hudElement.id = 'wasm-ppo-hud';
  hudElement.style.cssText = [
    'position: fixed',
    'bottom: 10px',
    'left: 50%',
    'transform: translateX(-50%)',
    'background: rgba(0, 0, 0, 0.85)',
    'color: #39FF14',
    'font-family: "Courier New", monospace',
    'font-size: 13px',
    'padding: 8px 16px',
    'border-radius: 8px',
    'border: 1px solid #39FF14',
    'z-index: 9999',
    'pointer-events: auto',
    'user-select: none',
    'white-space: nowrap',
    'display: none',
    'box-shadow: 0 0 12px rgba(57, 255, 20, 0.3)',
  ].join('; ');

  hudElement.innerHTML = _buildHudHtml();
  document.body.appendChild(hudElement);

  // Wire up HUD buttons
  const btnTurbo = hudElement.querySelector('#wasm-hud-turbo');
  const btnPause = hudElement.querySelector('#wasm-hud-pause');
  const btnExport = hudElement.querySelector('#wasm-hud-export');

  if (btnTurbo) {
    btnTurbo.addEventListener('click', (e) => {
      e.stopPropagation();
      _toggleTurbo();
    });
  }
  if (btnPause) {
    btnPause.addEventListener('click', (e) => {
      e.stopPropagation();
      _toggleLearning();
    });
  }
  if (btnExport) {
    btnExport.addEventListener('click', (e) => {
      e.stopPropagation();
      exportWasmWeights();
    });
  }
}

function _buildHudHtml() {
  return [
    '<div style="display:flex; align-items:center; gap:12px; flex-wrap:nowrap;">',
    '  <span style="font-weight:bold;">PPO</span>',
    '  <span id="wasm-hud-stats">Ep: 0 | Avg: 0.0 | Loss: 0.000 | Ent: 0.00</span>',
    '  <span id="wasm-hud-progress" style="display:inline-block; width:120px; height:10px;',
    '    background:#222; border-radius:5px; overflow:hidden; vertical-align:middle;">',
    '    <span id="wasm-hud-bar" style="display:block; height:100%; width:0%;',
    '      background:#39FF14; border-radius:5px; transition:width 0.3s;"></span>',
    '  </span>',
    '  <button id="wasm-hud-turbo" style="',
    '    background:transparent; color:#39FF14; border:1px solid #39FF14;',
    '    border-radius:4px; padding:2px 8px; cursor:pointer; font-size:12px;',
    '    font-family:inherit;">Turbo</button>',
    '  <button id="wasm-hud-pause" style="',
    '    background:transparent; color:#39FF14; border:1px solid #39FF14;',
    '    border-radius:4px; padding:2px 8px; cursor:pointer; font-size:12px;',
    '    font-family:inherit;">Pause</button>',
    '  <button id="wasm-hud-export" style="',
    '    background:transparent; color:#39FF14; border:1px solid #39FF14;',
    '    border-radius:4px; padding:2px 8px; cursor:pointer; font-size:12px;',
    '    font-family:inherit;">Export</button>',
    '</div>',
  ].join('\n');
}

function _updateHud() {
  if (!hudElement) return;

  // Show/hide based on active state
  hudElement.style.display = wasmActive ? 'block' : 'none';

  const statsEl = hudElement.querySelector('#wasm-hud-stats');
  if (statsEl) {
    statsEl.textContent = [
      'Ep: ' + agentStats.episode,
      'Avg: ' + agentStats.avgReward.toFixed(1),
      'Loss: ' + agentStats.policyLoss.toFixed(3),
      'Ent: ' + agentStats.entropy.toFixed(2),
    ].join(' | ');
  }

  // Progress bar — show rollout buffer fill percentage
  const barEl = hudElement.querySelector('#wasm-hud-bar');
  if (barEl) {
    const rolloutLen = DEFAULT_AGENT_CONFIG.rollout_length || 2048;
    const pct = Math.min(100, ((agentStats.totalSteps % rolloutLen) / rolloutLen) * 100);
    barEl.style.width = pct.toFixed(0) + '%';
  }

  // Turbo button state
  const btnTurbo = hudElement.querySelector('#wasm-hud-turbo');
  if (btnTurbo) {
    btnTurbo.textContent = turboMode ? ('Turbo x' + turboMultiplier) : 'Turbo';
    btnTurbo.style.background = turboMode ? '#39FF14' : 'transparent';
    btnTurbo.style.color = turboMode ? '#000' : '#39FF14';
  }

  // Pause/Learn button state
  const btnPause = hudElement.querySelector('#wasm-hud-pause');
  if (btnPause) {
    btnPause.textContent = learningEnabled ? 'Pause' : 'Learn';
    btnPause.style.background = learningEnabled ? 'transparent' : '#FF6600';
    btnPause.style.color = learningEnabled ? '#39FF14' : '#000';
  }
}

function _showHudMessage(msg, level) {
  const color = level === 'error' ? '#FF4444' : level === 'success' ? '#39FF14' : '#FFAA00';
  console.log('[WASM Bridge] ' + msg);

  // Brief on-screen toast
  const toast = document.createElement('div');
  toast.style.cssText = [
    'position: fixed',
    'top: 60px',
    'left: 50%',
    'transform: translateX(-50%)',
    'background: rgba(0,0,0,0.9)',
    'color: ' + color,
    'font-family: "Courier New", monospace',
    'font-size: 14px',
    'padding: 8px 20px',
    'border-radius: 6px',
    'border: 1px solid ' + color,
    'z-index: 10000',
    'pointer-events: none',
    'transition: opacity 0.5s',
  ].join('; ');
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; }, 2000);
  setTimeout(() => { toast.remove(); }, 2600);
}

// ---------------------------------------------------------------------------
// Toggle helpers
// ---------------------------------------------------------------------------

async function _toggleWasmAgent() {
  if (!wasmReady) {
    const ok = await initWasmAgent();
    if (!ok) return;
  }

  wasmActive = !wasmActive;

  if (wasmActive) {
    // Disable the heuristic/DQN AI so they don't fight
    if (typeof autoPlayEnabled !== 'undefined') {
      try { autoPlayEnabled = false; } catch (_e) { /* ignore */ }
    }
    wasmAgent.reset();
    _showHudMessage('WASM PPO agent enabled.', 'success');
  } else {
    _stopTurbo();
    turboMode = false;
    _showHudMessage('WASM PPO agent disabled.', 'info');
  }

  _updateHud();
}

function _toggleTurbo() {
  if (!wasmActive) return;
  turboMode = !turboMode;

  if (turboMode) {
    _startTurbo();
  } else {
    _stopTurbo();
  }

  _updateHud();
}

function _toggleLearning() {
  if (!wasmReady || !wasmAgent) return;
  learningEnabled = !learningEnabled;
  try {
    wasmAgent.set_learning(learningEnabled);
  } catch (_e) {
    // Method may not exist yet
  }
  _updateHud();
}

// ---------------------------------------------------------------------------
// Key bindings — F2, F3, F4
// ---------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
  if (e.code === 'F2') {
    e.preventDefault();
    _toggleWasmAgent();
  }
  if (e.code === 'F3') {
    e.preventDefault();
    _toggleTurbo();
  }
  if (e.code === 'F4') {
    e.preventDefault();
    exportWasmWeights();
  }
});

// ---------------------------------------------------------------------------
// Public API — exposed on window for integration from index.html / test pages
// ---------------------------------------------------------------------------

window.wasmBridge = {
  init: initWasmAgent,
  update: updateWasmAgent,
  applyState: applyWasmStateToGame,
  exportWeights: exportWasmWeights,
  importWeights: importWasmWeights,
  get active() { return wasmActive; },
  get ready() { return wasmReady; },
  get stats() { return Object.assign({}, agentStats); },
  get turbo() { return turboMode; },
  set turboMultiplier(n) { turboMultiplier = Math.max(1, Math.min(100, n)); },
};
