/**
 * wasm_bridge.js — JavaScript bridge for the WASM PPO agent in eyeBMinvaders.
 *
 * Dynamically loads the wasm_agent WASM module, initializes a WasmAgent,
 * and provides functions that the game loop can call to let the PPO agent
 * drive gameplay and train in the browser.
 *
 * Key bindings:
 *   W  — Toggle WASM PPO agent on/off
 *   T  — Toggle turbo training mode
 *   E  — Export trained weights to JSON download
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
      console.warn('[WASM Bridge] Could not load WASM module from any known path.');
      _showHudMessage('WASM module not found. Run wasm_agent/build.sh first.', 'error');
      return false;
    }

    // wasm-bindgen --target web requires calling the default init()
    if (typeof wasmMod.default === 'function') {
      await wasmMod.default();
    }

    wasmModule = wasmMod;

    // Merge config
    const config = Object.assign({}, DEFAULT_AGENT_CONFIG, configOverrides || {});
    const configJson = JSON.stringify(config);

    wasmAgent = new wasmMod.WasmAgent(configJson);
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
 * Lightweight observer — just updates the HUD. Does NOT call wasmAgent.step()
 * during gameplay to avoid blocking. The heavy PPO training runs only during
 * idle time (requestIdleCallback) or turbo mode.
 */
let _ppoFrameCount = 0;
let _ppoEpisodes = 0;
let _ppoTotalReward = 0;
let _ppoRewardHistory = [];

let _prevScore = 0;
let _prevLives = 0;
let _prevLevel = 0;

function updateWasmAgent() {
  if (!wasmReady || !wasmActive) return null;

  _ppoFrameCount++;

  // Track game stats for the HUD (zero-cost — just reading globals)
  const newScore = (typeof score !== 'undefined') ? score : 0;
  const newLives = (typeof player !== 'undefined') ? player.lives : 0;
  const newLevel = (typeof currentLevel !== 'undefined') ? currentLevel : 1;
  const isGameOver = (typeof gameOverFlag !== 'undefined') ? gameOverFlag : false;

  // Accumulate reward
  let reward = (newScore - _prevScore) * 0.01;
  if (newLives < _prevLives) reward -= 5.0;
  if (newLevel > _prevLevel) reward += 5.0 + 3.0 * newLevel;
  _ppoTotalReward += reward;

  _prevScore = newScore;
  _prevLives = newLives;
  _prevLevel = newLevel;

  // Episode ended
  if (isGameOver) {
    _ppoEpisodes++;
    _ppoRewardHistory.push(_ppoTotalReward);
    if (_ppoRewardHistory.length > 100) _ppoRewardHistory.shift();
    _ppoTotalReward = 0;
    _prevScore = 0;
    _prevLives = 6;
    _prevLevel = 1;
  }

  // Update HUD stats (cheap — just setting local object)
  agentStats.episode = _ppoEpisodes;
  agentStats.avgReward = _ppoRewardHistory.length > 0
    ? _ppoRewardHistory.reduce((a, b) => a + b, 0) / _ppoRewardHistory.length
    : 0;
  agentStats.totalSteps = _ppoFrameCount;

  // Refresh HUD every second
  if (_ppoFrameCount % 60 === 0) {
    // Also run WASM training in background if agent exists
    if (wasmAgent) {
      try { _refreshStats(); } catch (_e) {}
    }
    _updateHud();
  }

  return true;
}

/** Read the current player action from keyboard/AI state */
function _getCurrentAction() {
  if (typeof keys === 'undefined') return 0;
  const left = keys.ArrowLeft || keys.KeyA;
  const right = keys.ArrowRight || keys.KeyD;
  const fire = keys.Space;
  if (fire && left) return 4;   // fire+left
  if (fire && right) return 5;  // fire+right
  if (fire) return 3;           // fire
  if (left) return 1;           // left
  if (right) return 2;          // right
  return 0;                     // idle
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

  // --- Player (Rust sends flat: playerX, playerY, playerLives, isPlayerHit) ---
  if (state.playerX !== undefined) {
    if (typeof window.player !== 'undefined') {
      window.player.x = state.playerX;
      window.player.y = state.playerY;
      window.player.width = 48;
      window.player.height = 48;
      if (state.playerLives !== undefined) {
        window.player.lives = state.playerLives;
      }
      // Keep the existing image reference — the renderer owns it
      if (typeof playerNormalImage !== 'undefined' && window.player.image !== playerNormalImage) {
        window.player.image = playerNormalImage;
      }
    }
  }

  // --- Enemies (Rust sends: x, y, w, h, hits) ---
  if (Array.isArray(state.enemies)) {
    const mapped = state.enemies.map((e) => {
      const enemy = {
        x: e.x,
        y: e.y,
        width: e.w || 43,
        height: e.h || 43,
        hits: e.hits || 0,
        image: new Image(),
      };
      // Pick a color SVG based on the row hint, or default to red
      const colors = ['red', 'orange', 'yellow', 'green', 'blue'];
      const color = colors[(e.row || 0) % colors.length];
      enemy.image.src = 'enemy-ship-' + color + '.svg';
      return enemy;
    });
    if (typeof enemies !== 'undefined') {
      enemies.length = 0;
      enemies.push(...mapped);
    }
  }

  // --- Bullets (Rust sends: x, y, isEnemy) ---
  if (Array.isArray(state.bullets)) {
    const mapped = state.bullets.map((b) => ({
      x: b.x,
      y: b.y,
      isEnemyBullet: !!b.isEnemy,
      dx: b.dx || 0,
      dy: b.dy || 0,
    }));
    if (typeof bullets !== 'undefined') {
      bullets.length = 0;
      bullets.push(...mapped);
    }
  }

  // --- Kamikaze enemies (Rust sends as "kamikazes": x, y, w, h, angle) ---
  if (Array.isArray(state.kamikazes)) {
    const mapped = state.kamikazes.map((k) => ({
      x: k.x,
      y: k.y,
      width: k.w || 43,
      height: k.h || 43,
      angle: k.angle || 0,
    }));
    if (typeof kamikazeEnemies !== 'undefined') {
      kamikazeEnemies.length = 0;
      kamikazeEnemies.push(...mapped);
    }
  }

  // --- Homing missiles (Rust sends as "missiles": x, y, angle, w, h) ---
  if (Array.isArray(state.missiles)) {
    const mapped = state.missiles.map((m) => ({
      x: m.x,
      y: m.y,
      angle: m.angle || 0,
      width: m.w || 57,
      height: m.h || 57,
      time: m.time || 0,
    }));
    if (typeof homingMissiles !== 'undefined') {
      homingMissiles.length = 0;
      homingMissiles.push(...mapped);
    }
  }

  // --- Walls (Rust sends: x, y, w, h, hitCount) ---
  if (Array.isArray(state.walls)) {
    const mapped = state.walls.map((w) => ({
      x: w.x,
      y: w.y,
      width: w.w || 58,
      height: w.h || 23,
      hitCount: w.hitCount || 0,
      missileHits: w.missileHits || 0,
      image: (typeof wallImage !== 'undefined') ? wallImage : new Image(),
    }));
    if (typeof walls !== 'undefined') {
      walls.length = 0;
      walls.push(...mapped);
    }
  }

  // --- Monster (Rust sends: x, y, w, h — or absent if none/hit) ---
  if (state.monster !== undefined) {
    if (state.monster === null) {
      if (typeof window !== 'undefined') window.monster = null;
    } else {
      const m = state.monster;
      if (typeof window !== 'undefined') {
        window.monster = {
          x: m.x,
          y: m.y,
          width: m.w || 56,
          height: m.h || 56,
          hit: false,
          direction: 1,
          speed: 3,
        };
      }
    }
  } else {
    // Rust omits monster entirely when there is none
    if (typeof window !== 'undefined') window.monster = null;
  }

  // --- Monster2 (Rust sends: x, y, w, h — or absent if none/hit/disappeared) ---
  if (state.monster2 !== undefined) {
    if (state.monster2 === null) {
      if (typeof window !== 'undefined') window.monster2 = null;
    } else {
      const m2 = state.monster2;
      if (typeof window !== 'undefined') {
        window.monster2 = {
          x: m2.x,
          y: m2.y,
          width: m2.w || 56,
          height: m2.h || 56,
          spiralAngle: 0,
          pattern: 'bounce',
          isDisappeared: false,
        };
      }
    }
  } else {
    if (typeof window !== 'undefined') window.monster2 = null;
  }

  // --- Scalar game state (Rust sends: score, level, gameOver) ---
  if (state.score !== undefined && typeof score !== 'undefined') {
    window.score = state.score;
    try { score = state.score; } catch (_e) { /* let-scoped, window fallback */ }
  }

  if (state.level !== undefined) {
    try { currentLevel = state.level; } catch (_e) { /* fallback */ }
  }

  if (state.gameOver !== undefined) {
    try { gameOverFlag = state.gameOver; } catch (_e) { /* fallback */ }
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

  // Refresh stats and HUD after the turbo batch
  _refreshStats();
  _updateHud();

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
// Background training — gentle idle-time PPO steps (won't block gameplay)
// ---------------------------------------------------------------------------

let _bgTrainId = null;

function _bgTrainCallback(deadline) {
  if (!wasmActive || !wasmReady || !wasmAgent) return;

  // Only use leftover idle time — never block a frame
  while (deadline.timeRemaining() > 2) {
    try {
      wasmAgent.step(); // one internal sim step
    } catch (_e) { break; }
  }

  _bgTrainId = requestIdleCallback(_bgTrainCallback, { timeout: 200 });
}

function _startBackgroundTraining() {
  if (_bgTrainId !== null) return;
  _bgTrainId = requestIdleCallback(_bgTrainCallback, { timeout: 200 });
}

function _stopBackgroundTraining() {
  if (_bgTrainId !== null) {
    cancelIdleCallback(_bgTrainId);
    _bgTrainId = null;
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
      agentStats.episode = s.episodes || 0;
      agentStats.avgReward = s.avgReward || 0;
      agentStats.totalSteps = s.totalSteps || 0;
      agentStats.bestReward = s.bestScore || 0;
      // Policy loss and entropy live inside the nested lastUpdate object
      if (s.lastUpdate) {
        agentStats.policyLoss = s.lastUpdate.policyLoss || 0;
        agentStats.entropy = s.lastUpdate.entropy || 0;
      }
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
    // Don't disable DQN/human — PPO observes, doesn't control
    _prevScore = (typeof score !== 'undefined') ? score : 0;
    _prevLives = (typeof player !== 'undefined') ? player.lives : 6;
    _prevLevel = (typeof currentLevel !== 'undefined') ? currentLevel : 1;
    _ppoTotalReward = 0;
    // Start gentle background PPO training via idle callbacks
    _startBackgroundTraining();
    _showHudMessage('PPO learning — play normally, AI watches & learns', 'success');
  } else {
    _stopTurbo();
    _stopBackgroundTraining();
    turboMode = false;
    _showHudMessage('PPO learning stopped', 'info');
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
  if (e.code === 'KeyW') {          // W — toggle WASM PPO agent
    _toggleWasmAgent();
  }
  if (e.code === 'KeyT') {          // T — toggle turbo training
    _toggleTurbo();
  }
  if (e.code === 'KeyE') {          // E — export trained weights
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
