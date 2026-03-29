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
  lr: 3e-4,
  gamma: 0.99,
  gae_lambda: 0.95,
  clip_epsilon: 0.2,
  entropy_coeff: 0.02,
  value_coeff: 0.5,
  max_grad_norm: 1.0,
  minibatch_size: 64,
  n_epochs: 4,
  rollout_length: 2048,
  n_frames: 4,
  seed: 42,
  god_mode: false,
  lr_warmup_updates: 10,
  lr_decay_updates: 500,
  lr_min: 1e-5,
  curriculum_enabled: true,
  curriculum_start_level: 1,
  curriculum_advance_threshold: 15.0,
  curriculum_window: 50,
  obs_norm_enabled: true,
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
let _recordedTransitions = []; // stored gameplay observations
const _MAX_RECORDED = 50000;   // cap at 50K transitions (~20MB)

let _prevScore = 0;
let _prevLives = 0;
let _prevLevel = 0;
let _prevState = null;
let _restartPending = false;

function updateWasmAgent() {
  if (!wasmActive) return null;

  _ppoFrameCount++;
  // Throttle to 30Hz
  if (_ppoFrameCount % 2 !== 0) return true;

  // Turbo mode: train PPO internally, update HUD stats only
  // Does NOT touch the visible game — training runs in idle callbacks
  if (turboMode && _ppoFrameCount % 60 === 0) {
    _refreshStats();
    _updateHud();
    return true;
  }

  // W without T: just observe and record (no game control)
  const newScore = (typeof score !== 'undefined') ? score : 0;
  const newLives = (typeof player !== 'undefined') ? player.lives : 0;
  const newLevel = (typeof currentLevel !== 'undefined') ? currentLevel : 1;
  const isGameOver = (typeof gameOverFlag !== 'undefined') ? gameOverFlag : false;

  // Compute reward
  let reward = (newScore - _prevScore) * 0.01;
  if (newLives < _prevLives) reward -= 5.0;
  if (newLevel > _prevLevel) reward += 5.0 + 3.0 * newLevel;
  reward += 0.01 * newLevel;
  _ppoTotalReward += reward;

  // Get current state and action
  const curState = (typeof buildDQNState === 'function') ? buildDQNState() : null;
  const action = _getCurrentAction();

  // Record transition
  if (_prevState && _recordedTransitions.length < _MAX_RECORDED) {
    _recordedTransitions.push({
      s: Array.from(_prevState),
      a: action,
      r: reward,
      d: isGameOver ? 1 : 0,
    });
  }

  _prevState = curState;
  _prevScore = newScore;
  _prevLives = newLives;
  _prevLevel = newLevel;

  if (isGameOver) {
    _ppoEpisodes++;
    _ppoRewardHistory.push(_ppoTotalReward);
    if (_ppoRewardHistory.length > 100) _ppoRewardHistory.shift();
    _ppoTotalReward = 0;
    _prevScore = 0;
    _prevLives = 6;
    _prevLevel = 1;
    _prevState = null;
  }

  // Update HUD stats
  agentStats.episode = _ppoEpisodes;
  agentStats.avgReward = _ppoRewardHistory.length > 0
    ? _ppoRewardHistory.reduce((a, b) => a + b, 0) / _ppoRewardHistory.length
    : 0;
  agentStats.totalSteps = _recordedTransitions.length + _totalFlushed;

  // Auto-flush to backend when buffer is full
  if (_recordedTransitions.length >= _AUTO_FLUSH_INTERVAL) {
    _autoFlushTransitions();
  }

  // Refresh HUD every second
  if (_ppoFrameCount % 60 === 0) {
    _updateHud();
  }

  return true;
}

/** Auto-flush recorded transitions to backend every N steps */
const _AUTO_FLUSH_INTERVAL = 5000; // flush every 5000 transitions
let _totalFlushed = 0;

async function _autoFlushTransitions() {
  if (_recordedTransitions.length < _AUTO_FLUSH_INTERVAL) return;

  const batch = _recordedTransitions.splice(0, _AUTO_FLUSH_INTERVAL);
  try {
    const resp = await fetch('/api/gameplay', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transitions: batch,
        episodes: _ppoEpisodes,
        timestamp: new Date().toISOString(),
      }),
    });
    if (resp.ok) {
      _totalFlushed += batch.length;
      _showHudMessage(`Saved ${batch.length} steps (total: ${_totalFlushed})`, 'success');
    } else {
      // Backend not available — push back and try file download
      _recordedTransitions.unshift(...batch);
      _downloadTransitions();
    }
  } catch (_e) {
    // No backend — push back
    _recordedTransitions.unshift(...batch);
    _downloadTransitions();
  }
}

/** Fallback: download as JSON file */
function _downloadTransitions() {
  if (_recordedTransitions.length === 0) return;
  const data = JSON.stringify({
    transitions: _recordedTransitions,
    episodes: _ppoEpisodes,
    timestamp: new Date().toISOString(),
  });
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `gameplay_${_recordedTransitions.length}steps.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Export: tries backend first, falls back to download */
function exportRecordedGameplay() {
  if (_recordedTransitions.length === 0 && _totalFlushed === 0) {
    _showHudMessage('No recorded gameplay yet', 'error');
    return;
  }
  if (_recordedTransitions.length > 0) {
    _autoFlushTransitions();
  } else {
    _showHudMessage(`All ${_totalFlushed} steps already saved`, 'success');
  }
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
// Background training — runs WASM PPO steps during idle time
// Only active when Turbo (T) is enabled
// ---------------------------------------------------------------------------

let _bgTrainId = null;

function _bgTrainCallback(deadline) {
  if (!turboMode || !wasmReady || !wasmAgent) return;

  // Run WASM steps only during idle time, max 5ms per callback
  const start = performance.now();
  let steps = 0;
  while (performance.now() - start < 5 && deadline.timeRemaining() > 1) {
    try {
      wasmAgent.step();
      steps++;
    } catch (_e) { break; }
  }

  _bgTrainId = requestIdleCallback(_bgTrainCallback, { timeout: 50 });
}

function _startBackgroundTraining() {
  if (_bgTrainId !== null) return;
  _bgTrainId = requestIdleCallback(_bgTrainCallback, { timeout: 50 });
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
      agentStats.bestScore = s.bestScore || 0;
      agentStats.bestLevel = s.bestLevel || 0;
      agentStats.currentScore = s.currentScore || 0;
      agentStats.currentLevel = s.currentLevel || 0;
      agentStats.currentLives = s.currentLives || 0;
      agentStats.enemiesKilled = s.enemiesKilled || 0;
      agentStats.kamikazesKilled = s.kamikazesKilled || 0;
      agentStats.missilesShot = s.missilesShot || 0;
      agentStats.monstersKilled = s.monstersKilled || 0;
      agentStats.monsters2Killed = s.monsters2Killed || 0;
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
    '  <span id="wasm-hud-mode" style="font-weight:bold; color:#FFAA00;">PPO</span>',
    '  <span id="wasm-hud-stats">Watching...</span>',
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

  // Mode indicator
  const modeEl = hudElement.querySelector('#wasm-hud-mode');
  if (modeEl) {
    if (turboMode) {
      modeEl.textContent = 'PPO TRAINING';
      modeEl.style.color = '#39FF14';
    } else {
      modeEl.textContent = 'PPO watching';
      modeEl.style.color = '#FFAA00';
    }
  }

  const statsEl = hudElement.querySelector('#wasm-hud-stats');
  if (statsEl) {
    if (turboMode) {
      // Show training stats from WASM
      statsEl.textContent = [
        'Ep: ' + agentStats.episode,
        'Avg: ' + agentStats.avgReward.toFixed(1),
        'Loss: ' + agentStats.policyLoss.toFixed(3),
        'Ent: ' + agentStats.entropy.toFixed(2),
      ].join(' | ');
    } else {
      // Show observation stats from gameplay
      const avgR = _ppoRewardHistory.length > 0
        ? (_ppoRewardHistory.reduce((a, b) => a + b, 0) / _ppoRewardHistory.length).toFixed(1)
        : '-';
      statsEl.textContent = [
        'Games: ' + _ppoEpisodes,
        'Avg: ' + avgR,
        'Saved: ' + (_recordedTransitions.length + _totalFlushed),
        'T=train E=export',
      ].join(' | ');
    }
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
// Training Dashboard — replaces game view during turbo training
// ---------------------------------------------------------------------------

let _dashboardEl = null;
let _dashboardInterval = null;
let _rewardHistory = [];   // for sparkline
let _maxReward = 1;

function _showTrainingDashboard() {
  // Hide game canvas
  const canvas = document.getElementById('gameCanvas');
  if (canvas) canvas.style.display = 'none';

  if (!_dashboardEl) {
    _dashboardEl = document.createElement('div');
    _dashboardEl.id = 'ppo-dashboard';
    _dashboardEl.style.cssText = [
      'position:fixed; top:0; left:0; width:100vw; height:100vh',
      'background:#0a0a0a; color:#e0e0e0; font-family:"Courier New",monospace',
      'display:flex; flex-direction:column; align-items:center; justify-content:center',
      'z-index:10000; gap:20px',
    ].join(';');
    document.body.appendChild(_dashboardEl);
  }
  _dashboardEl.style.display = 'flex';
  _rewardHistory = [];

  // Update dashboard at 4Hz
  _dashboardInterval = setInterval(_updateDashboard, 250);
  _updateDashboard();
}

function _hideTrainingDashboard() {
  if (_dashboardEl) _dashboardEl.style.display = 'none';
  if (_dashboardInterval) { clearInterval(_dashboardInterval); _dashboardInterval = null; }
  // Restore game canvas
  const canvas = document.getElementById('gameCanvas');
  if (canvas) canvas.style.display = 'block';
}

function _updateDashboard() {
  if (!_dashboardEl) return;

  // Fetch latest stats from WASM
  _refreshStats();
  const s = agentStats;

  // Track reward history for sparkline
  _rewardHistory.push(s.avgReward || 0);
  if (_rewardHistory.length > 80) _rewardHistory.shift();
  _maxReward = Math.max(1, ..._rewardHistory.map(Math.abs));

  // Build sparkline
  const spark = _rewardHistory.map(r => {
    const h = Math.max(1, Math.round((r / _maxReward) * 30));
    return `<span style="display:inline-block;width:4px;height:${h}px;background:#39FF14;vertical-align:bottom;margin:0 1px;border-radius:1px;"></span>`;
  }).join('');

  const elapsed = (s.totalSteps / 30).toFixed(0); // approx seconds at 30Hz

  _dashboardEl.innerHTML = `
    <div style="font-size:28px;color:#39FF14;font-weight:bold;letter-spacing:2px;">
      PPO TRAINING
    </div>
    <div style="font-size:14px;color:#888;margin-top:-10px;">
      Press T to stop and return to game
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px 40px;margin:20px 0;font-size:16px;">
      <div style="text-align:right;color:#888;">Episodes</div>
      <div style="color:#fff;font-size:20px;font-weight:bold;">${s.episode}</div>
      <div style="text-align:right;color:#888;">Avg Reward</div>
      <div style="color:${s.avgReward > 0 ? '#39FF14' : '#FF4444'};font-size:20px;font-weight:bold;">${s.avgReward.toFixed(1)}</div>
      <div style="text-align:right;color:#888;">Best Reward</div>
      <div style="color:#0ff;font-size:20px;font-weight:bold;">${(s.bestReward || 0).toFixed(1)}</div>
      <div style="text-align:right;color:#888;">Policy Loss</div>
      <div style="color:#fff;">${s.policyLoss.toFixed(4)}</div>
      <div style="text-align:right;color:#888;">Entropy</div>
      <div style="color:#fff;">${s.entropy.toFixed(3)}</div>
      <div style="text-align:right;color:#888;">Best Score</div>
      <div style="color:#0ff;font-size:20px;font-weight:bold;">${(s.bestScore || 0).toLocaleString()}</div>
      <div style="text-align:right;color:#888;">Best Level</div>
      <div style="color:#0ff;font-size:20px;font-weight:bold;">${s.bestLevel || 0}</div>
      <div style="text-align:right;color:#888;">Current Score</div>
      <div style="color:#fff;">${(s.currentScore || 0).toLocaleString()}</div>
      <div style="text-align:right;color:#888;">Current Level</div>
      <div style="color:#fff;">${s.currentLevel || 0}</div>
      <div style="text-align:right;color:#888;">Lives</div>
      <div style="color:#fff;">${s.currentLives || 0}</div>
      <div style="text-align:right;color:#888;">Steps</div>
      <div style="color:#fff;">${s.totalSteps.toLocaleString()}</div>
      <div style="text-align:right;color:#888;">Time</div>
      <div style="color:#fff;">${elapsed}s</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:10px 0;text-align:center;">
      <div style="background:#111;border:1px solid #333;border-radius:6px;padding:8px;">
        <div style="color:#888;font-size:10px;">Enemies</div>
        <div style="color:#FF4444;font-size:18px;font-weight:bold;">${s.enemiesKilled || 0}</div>
      </div>
      <div style="background:#111;border:1px solid #333;border-radius:6px;padding:8px;">
        <div style="color:#888;font-size:10px;">Kamikazes</div>
        <div style="color:#FF8800;font-size:18px;font-weight:bold;">${s.kamikazesKilled || 0}</div>
      </div>
      <div style="background:#111;border:1px solid #333;border-radius:6px;padding:8px;">
        <div style="color:#888;font-size:10px;">Missiles</div>
        <div style="color:#FFDD00;font-size:18px;font-weight:bold;">${s.missilesShot || 0}</div>
      </div>
      <div style="background:#111;border:1px solid #333;border-radius:6px;padding:8px;">
        <div style="color:#888;font-size:10px;">Monster</div>
        <div style="color:#AA44FF;font-size:18px;font-weight:bold;">${s.monstersKilled || 0}</div>
      </div>
      <div style="background:#111;border:1px solid #333;border-radius:6px;padding:8px;">
        <div style="color:#888;font-size:10px;">Monster2</div>
        <div style="color:#FF44FF;font-size:18px;font-weight:bold;">${s.monsters2Killed || 0}</div>
      </div>
    </div>
    <div style="background:#111;border:1px solid #333;border-radius:8px;padding:15px 20px;min-width:350px;">
      <div style="color:#888;font-size:12px;margin-bottom:8px;">Reward Trend</div>
      <div style="height:35px;display:flex;align-items:flex-end;">${spark}</div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Toggle helpers
// ---------------------------------------------------------------------------

async function _toggleWasmAgent() {
  // W key: toggle recording/observation mode. WASM PPO is optional.
  // If WASM fails to load, recording still works (pure JS).
  if (!wasmReady) {
    try {
      await initWasmAgent();
    } catch (_e) {
      // WASM PPO failed — that's OK, recording works without it
      console.log('[WASM Bridge] PPO not available — recording-only mode');
    }
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
    _showHudMessage('Recording gameplay — play normally', 'success');
  } else {
    _stopTurbo();
    _stopBackgroundTraining();
    turboMode = false;
    _showHudMessage('PPO learning stopped', 'info');
  }

  _updateHud();
}

async function _toggleTurbo() {
  // Auto-activate WASM agent if not already on
  if (!wasmActive) {
    await _toggleWasmAgent();
    if (!wasmActive) return; // init failed
  }
  turboMode = !turboMode;

  if (turboMode) {
    _startTurbo();
    _startBackgroundTraining();
    _showTrainingDashboard();
  } else {
    _stopTurbo();
    _stopBackgroundTraining();
    _hideTrainingDashboard();
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
  if (e.code === 'KeyE') {          // E — export recorded gameplay or weights
    if (wasmActive && _recordedTransitions.length > 0) {
      exportRecordedGameplay();
    } else {
      exportWasmWeights();
    }
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
  exportGameplay: exportRecordedGameplay,
  get recordedSteps() { return _recordedTransitions.length; },
  importWeights: importWasmWeights,
  get active() { return wasmActive; },
  get ready() { return wasmReady; },
  get stats() { return Object.assign({}, agentStats); },
  get turbo() { return turboMode; },
  set turboMultiplier(n) { turboMultiplier = Math.max(1, Math.min(100, n)); },
};
