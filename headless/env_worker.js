#!/usr/bin/env node
/**
 * Headless game environment worker for Python fine-tuning.
 * Communicates via stdin/stdout JSON lines.
 *
 * Protocol:
 *   RESET           -> {"state": [200 floats]}
 *   STEP <action>   -> {"state": [...], "reward": float, "done": bool, "info": {...}}
 *   QUIT            -> exits
 */

const path = require('path');
const fs = require('fs');
const readline = require('readline');
const { installShims, stepFrame, advanceMockTime, setMockTime } = require('./browser-shim');

const GAME_ROOT = process.argv[2] || path.resolve(__dirname, '..');
const FPS = parseInt(process.argv[3] || '30');
const DT_MS = 1000.0 / FPS;
const STATE_SIZE = 50;

// Install shims and load game.js
installShims();

const gameCode = fs.readFileSync(path.join(GAME_ROOT, 'game.js'), 'utf-8');
const patchedCode = gameCode
  .replace(/^(let |const )/gm, 'var ')
  .replace(/^class /gm, 'var _cls = class ');

const vm = require('vm');
try {
  vm.runInThisContext(patchedCode, { filename: 'game.js' });
} catch (e) {
  // Ignore init errors from missing DOM elements
}

// Clear auto-reload intervals
const highId = setInterval(() => {}, 100000);
for (let i = 0; i <= highId; i++) clearInterval(i);

// Load model synchronously
const modelPath = path.join(GAME_ROOT, 'models', 'model_weights.json');
if (fs.existsSync(modelPath)) {
  try {
    dqnModel = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
  } catch (e) {}
}
// Prevent reloads
global.loadDQNModel = async function() {};

// Frame stacking buffer
const N_FRAMES = (dqnModel && dqnModel.n_frames) || 4;
let frameBuffer = new Array(N_FRAMES).fill(null).map(() => new Float32Array(STATE_SIZE));

function resetFrameBuffer(state) {
  for (let i = 0; i < N_FRAMES; i++) {
    frameBuffer[i] = new Float32Array(state);
  }
}

function pushFrame(state) {
  for (let i = 0; i < N_FRAMES - 1; i++) {
    frameBuffer[i] = frameBuffer[i + 1];
  }
  frameBuffer[N_FRAMES - 1] = new Float32Array(state);
}

function getStackedState() {
  const out = new Array(N_FRAMES * STATE_SIZE);
  for (let f = 0; f < N_FRAMES; f++) {
    for (let j = 0; j < STATE_SIZE; j++) {
      out[f * STATE_SIZE + j] = frameBuffer[f][j];
    }
  }
  return out;
}

// Reward calculation matching Rust sim
function calculateReward(oldScore, oldLives, oldLevel) {
  let reward = 0;
  const newScore = score || 0;
  const newLives = player ? player.lives : 0;
  const newLevel = currentLevel || 1;

  // Score delta
  reward += (newScore - oldScore) * 0.01;

  // Life loss
  if (newLives < oldLives) reward -= 5.0;

  // Game over
  if (gameOverFlag) reward -= 20.0;

  // Survival bonus
  reward += 0.01 * newLevel;

  // Level completion
  if (newLevel > oldLevel) {
    reward += 5.0 + 3.0 * newLevel;
  }

  // Wall-shooting penalty (approximate — count bullets that disappeared near walls)
  // This is harder to track in JS without modifying game.js, so skip for now

  return reward;
}

function resetGame() {
  setMockTime(1000);
  global._rafCallbacks = [];

  restartGame();
  autoPlayEnabled = false;  // we control actions manually
  dqnLastDecisionTime = 0;
  gamePaused = false;
  gameOverFlag = false;
  enemySpeed = 0.54;
  enemyDirection = 1;
  if (typeof currentEnemyFireRate !== 'undefined') currentEnemyFireRate = 0.85;
  lastEnemyFireTime = 0;
  lastFireTime = 0;
  monster2 = null;
  if (typeof explosions !== 'undefined') explosions = [];
  isPlayerHit = false;
  whilePlayerHit = false;
  lastTime = 1000;

  global._rafCallbacks = [gameLoop];

  // Prime one frame
  advanceMockTime(DT_MS);
  stepFrame(Date.now());

  // Build initial state and reset frame buffer
  const rawState = buildDQNState();
  resetFrameBuffer(rawState);
  return getStackedState();
}

function stepGame(action) {
  const oldScore = score || 0;
  const oldLives = player ? player.lives : 0;
  const oldLevel = currentLevel || 1;

  // Apply action
  applyDQNAction(action);

  // Advance game by one frame
  advanceMockTime(DT_MS);
  stepFrame(Date.now());

  // Get new state
  const rawState = buildDQNState();
  pushFrame(rawState);
  const stackedState = getStackedState();

  // Compute reward
  const reward = calculateReward(oldScore, oldLives, oldLevel);
  const done = !!gameOverFlag;

  const info = {
    score: score || 0,
    level: currentLevel || 1,
    lives: player ? player.lives : 0,
  };

  return { state: stackedState, reward, done, info };
}

// Initialize
if (typeof createEnemies === 'function') createEnemies();

// Signal ready
process.stdout.write('READY\n');

// Read commands from stdin
const rl = readline.createInterface({ input: process.stdin });

rl.on('line', (line) => {
  const cmd = line.trim();

  if (cmd === 'RESET') {
    const state = resetGame();
    process.stdout.write(JSON.stringify({ state: Array.from(state) }) + '\n');
  }
  else if (cmd.startsWith('STEP ')) {
    const action = parseInt(cmd.split(' ')[1]);
    const result = stepGame(action);
    process.stdout.write(JSON.stringify({
      state: Array.from(result.state),
      reward: result.reward,
      done: result.done,
      info: result.info,
    }) + '\n');
  }
  else if (cmd === 'QUIT') {
    process.exit(0);
  }
});
