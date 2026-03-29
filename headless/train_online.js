#!/usr/bin/env node
/**
 * Headless Online DQN Training
 * =============================
 * Runs game.js at full speed and trains the DQN model.
 * Same physics as the browser game — zero gap.
 *
 * Usage:
 *   node headless/train_online.js                          # 10K episodes
 *   node headless/train_online.js --episodes 50000         # custom
 *   node headless/train_online.js --lr 1e-5 --export-every 500
 *   node headless/train_online.js --epsilon 0.1 --batch-size 64
 */

'use strict';

const path = require('path');
const fs = require('fs');
const { installShims, stepFrame, advanceMockTime, setMockTime } = require('./browser-shim');
const { DuelingDQN, ReplayBuffer } = require('./dqn_trainer');

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const idx = args.indexOf(name);
  if (idx >= 0 && idx + 1 < args.length) return args[idx + 1];
  return defaultVal;
}

const NUM_EPISODES   = parseInt(getArg('--episodes', '10000'));
const LEARNING_RATE  = parseFloat(getArg('--lr', '1e-5'));
const EXPORT_EVERY   = parseInt(getArg('--export-every', '1000'));
const EPSILON        = parseFloat(getArg('--epsilon', '0.05'));
const MAX_STEPS      = parseInt(getArg('--max-steps', '10000'));
const BATCH_SIZE     = parseInt(getArg('--batch-size', '32'));
const GAMMA          = parseFloat(getArg('--gamma', '0.99'));
const BUFFER_CAP     = parseInt(getArg('--buffer-size', '50000'));
const TAU            = parseFloat(getArg('--tau', '0.005'));
const TRAIN_EVERY    = parseInt(getArg('--train-every', '4'));
const TARGET_EVERY   = parseInt(getArg('--target-every', '1000'));
const LOG_EVERY      = parseInt(getArg('--log-every', '100'));

const DT_MS = 33.333;  // 30Hz to match training sim (Rust dt)
const N_FRAMES = 4;
const RAW_STATE_SIZE = 54;
const STACKED_SIZE = RAW_STATE_SIZE * N_FRAMES;
const N_ACTIONS = 6;

// ---------------------------------------------------------------------------
// Install browser shims and load game.js (same pattern as headless/run.js)
// ---------------------------------------------------------------------------
installShims();

console.log('Loading game.js...');
const gameCode = fs.readFileSync(path.resolve(__dirname, '..', 'game.js'), 'utf-8');

// Convert top-level let/const to var so they become global
const patchedCode = gameCode
  .replace(/^(let |const )/gm, 'var ')
  .replace(/^class /gm, 'var _cls = class ');

try {
  const vm = require('vm');
  vm.runInThisContext(patchedCode, { filename: 'game.js' });
} catch (e) {
  if (e.message && !e.message.includes('Cannot read properties of null')) {
    console.error('Error loading game.js:', e.message);
  }
}

// Clear all intervals created by game.js (model auto-reload etc.)
const highId = setInterval(() => {}, 100000);
for (let i = 0; i <= highId; i++) clearInterval(i);

// Verify critical functions
const requiredFns = ['buildDQNState', 'applyDQNAction', 'gameLoop', 'restartGame', 'createEnemies'];
const missing = requiredFns.filter(fn => typeof global[fn] !== 'function');
if (missing.length > 0) {
  console.error(`Missing functions: ${missing.join(', ')}`);
  process.exit(1);
}
console.log('game.js loaded successfully.');

// Disable the game's own model loading to prevent interference
global.loadDQNModel = async function() {};

// ---------------------------------------------------------------------------
// Initialize DQN networks
// ---------------------------------------------------------------------------
const modelPath = path.resolve(__dirname, '..', 'models', 'model_weights.json');

const policyNet = new DuelingDQN(STACKED_SIZE);
const targetNet = new DuelingDQN(STACKED_SIZE);

if (fs.existsSync(modelPath)) {
  const modelData = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
  policyNet.loadWeightsJSON(modelData);
  console.log(`Loaded existing model: ${modelData.type}, arch=${JSON.stringify(modelData.architecture)}, n_frames=${modelData.n_frames}`);
} else {
  console.log('No existing model found — training from scratch.');
}

targetNet.copyFrom(policyNet);

// Also set the game's dqnModel so autoplay/inference works during simulation
// (the game reads dqnModel for frame buffer initialization etc.)
global.dqnModel = policyNet.exportWeightsJSON();

const replayBuffer = new ReplayBuffer(BUFFER_CAP);

// ---------------------------------------------------------------------------
// Frame stacking utilities (local, independent of game's frame buffer)
// ---------------------------------------------------------------------------
function createFrameBuffer() {
  return {
    frames: new Array(N_FRAMES).fill(null).map(() => new Float64Array(RAW_STATE_SIZE)),
  };
}

function resetFrameBuffer(fb, rawState) {
  for (let i = 0; i < N_FRAMES; i++) {
    for (let j = 0; j < RAW_STATE_SIZE; j++) {
      fb.frames[i][j] = rawState[j];
    }
  }
}

function pushFrame(fb, rawState) {
  // Shift left
  const tmp = fb.frames[0];
  for (let i = 0; i < N_FRAMES - 1; i++) {
    fb.frames[i] = fb.frames[i + 1];
  }
  fb.frames[N_FRAMES - 1] = tmp;
  for (let j = 0; j < RAW_STATE_SIZE; j++) {
    fb.frames[N_FRAMES - 1][j] = rawState[j];
  }
}

function getStackedState(fb) {
  const out = new Float64Array(STACKED_SIZE);
  for (let f = 0; f < N_FRAMES; f++) {
    for (let j = 0; j < RAW_STATE_SIZE; j++) {
      out[f * RAW_STATE_SIZE + j] = fb.frames[f][j];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Epsilon-greedy action selection
// ---------------------------------------------------------------------------
function selectAction(state, epsilon) {
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * N_ACTIONS);
  }
  const q = policyNet.forward(state);
  let best = 0, bestQ = q[0];
  for (let i = 1; i < N_ACTIONS; i++) {
    if (q[i] > bestQ) { bestQ = q[i]; best = i; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Train one batch (Double DQN with Huber loss)
// ---------------------------------------------------------------------------
let globalStep = 0;

function trainBatch() {
  if (replayBuffer.length < BATCH_SIZE) return 0;

  const batch = replayBuffer.sample(BATCH_SIZE);
  globalStep++;
  policyNet.zeroGrad();

  let totalLoss = 0;

  for (let i = 0; i < batch.length; i++) {
    const { state, action, reward, nextState, done } = batch[i];

    // Double DQN: use policy net to select action, target net to evaluate
    let target;
    if (done) {
      target = reward;
    } else {
      // Policy net picks the best action for next state
      const qNext = policyNet.forward(nextState);
      let bestA = 0, bestQ = qNext[0];
      for (let a = 1; a < N_ACTIONS; a++) {
        if (qNext[a] > bestQ) { bestQ = qNext[a]; bestA = a; }
      }
      // Target net evaluates that action
      const qTarget = targetNet.forward(nextState);
      target = reward + GAMMA * qTarget[bestA];
    }

    // Forward through policy (with caching) then backward
    policyNet.forwardTrain(state);
    totalLoss += policyNet.backward(action, target);
  }

  // Average gradients over batch
  for (const layer of policyNet._layers()) {
    for (let i = 0; i < layer.outSize; i++) {
      for (let j = 0; j < layer.inSize; j++) {
        layer.gradWeight[i][j] /= BATCH_SIZE;
      }
      layer.gradBias[i] /= BATCH_SIZE;
    }
  }

  policyNet.update(LEARNING_RATE, globalStep);

  return totalLoss / BATCH_SIZE;
}

// ---------------------------------------------------------------------------
// Reset the game for a new episode (same approach as headless/run.js)
// ---------------------------------------------------------------------------
function resetGame() {
  setMockTime(1000);
  global._rafCallbacks = [];
  global._pendingTimers = [];

  global.restartGame();

  // Manual state fixes that restartGame() misses
  global.autoPlayEnabled = true;
  global.dqnLastDecisionTime = 0;
  global.gamePaused = false;
  global.gameOverFlag = false;
  global.enemySpeed = 0.54;
  global.enemyDirection = 1;
  if (typeof global.currentEnemyFireRate !== 'undefined') global.currentEnemyFireRate = 0.85;
  global.lastEnemyFireTime = 0;
  global.lastFireTime = 0;
  global.monster2 = null;
  if (typeof global.explosions !== 'undefined') global.explosions = [];
  global.isPlayerHit = false;
  global.whilePlayerHit = false;
  if (global.dqnFrameBuffer) global.dqnFrameBuffer = null;

  global.lastTime = 1000;

  // Ensure exactly one gameLoop is queued
  global._rafCallbacks = [global.gameLoop];

  // Prime with one initial frame
  advanceMockTime(DT_MS);
  stepFrame(Date.now());
}

// ---------------------------------------------------------------------------
// Export model weights to disk
// ---------------------------------------------------------------------------
function exportModel(episode) {
  const modelData = policyNet.exportWeightsJSON();
  const outPath = path.resolve(__dirname, '..', 'models', 'model_weights.json');

  // Ensure models directory exists
  const modelsDir = path.dirname(outPath);
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
  }

  fs.writeFileSync(outPath, JSON.stringify(modelData));
  console.log(`  >> Model exported to ${outPath} (ep ${episode})`);

  // Also save a checkpoint
  const ckptPath = path.resolve(modelsDir, `model_ep${episode}.json`);
  fs.writeFileSync(ckptPath, JSON.stringify(modelData));
}

// ---------------------------------------------------------------------------
// Main training loop
// ---------------------------------------------------------------------------
function main() {
  console.log('\n=== Headless DQN Online Training ===');
  console.log(`Episodes: ${NUM_EPISODES}, LR: ${LEARNING_RATE}, Epsilon: ${EPSILON}`);
  console.log(`Batch: ${BATCH_SIZE}, Buffer: ${BUFFER_CAP}, Gamma: ${GAMMA}`);
  console.log(`Train every ${TRAIN_EVERY} steps, Target update every ${TARGET_EVERY} steps (tau=${TAU})`);
  console.log(`Export every ${EXPORT_EVERY} episodes, Max steps: ${MAX_STEPS}`);
  console.log(`Frame stacking: ${N_FRAMES} frames x ${RAW_STATE_SIZE} features = ${STACKED_SIZE} input\n`);

  // Initialize game
  if (typeof global.createEnemies === 'function') {
    global.createEnemies();
  }

  // Running statistics
  const recentScores = [];
  const recentLevels = [];
  let bestScore = 0;
  let totalTrainSteps = 0;
  let recentLoss = 0;
  let recentLossCount = 0;
  const startTime = Date.now();
  // Save real Date.now for wall-clock timing (mock overrides it)
  const { performance: nodePerf } = require('perf_hooks');
  const wallStart = nodePerf.now();

  for (let ep = 0; ep < NUM_EPISODES; ep++) {
    resetGame();

    const fb = createFrameBuffer();

    // Get initial state
    const rawState0 = global.buildDQNState();
    resetFrameBuffer(fb, rawState0);
    let state = getStackedState(fb);

    let oldScore = global.score || 0;
    let oldLives = global.player ? global.player.lives : 6;
    let oldLevel = global.currentLevel || 1;
    let episodeReward = 0;
    let episodeSteps = 0;
    let stepsSinceLastTrain = 0;
    let stepsSinceLastTarget = 0;

    for (let step = 0; step < MAX_STEPS; step++) {
      // Select and apply action
      const action = selectAction(state, EPSILON);
      global.applyDQNAction(action);

      // Advance game by one frame
      advanceMockTime(DT_MS);
      stepFrame(Date.now());

      // Read new state
      const newScore = global.score || 0;
      const newLives = global.player ? global.player.lives : 0;
      const newLevel = global.currentLevel || 1;
      const gameOver = global.gameOverFlag;

      // Compute reward (matching Rust sim)
      let reward = (newScore - oldScore) * 0.01;
      if (newLives < oldLives) reward -= 5.0;
      if (gameOver) reward -= 20.0;
      reward += 0.01 * newLevel;
      if (newLevel > oldLevel) reward += 5.0 + 3.0 * newLevel;

      episodeReward += reward;
      oldScore = newScore;
      oldLives = newLives;
      oldLevel = newLevel;

      // Get next stacked state
      const rawNext = global.buildDQNState();
      pushFrame(fb, rawNext);
      const nextState = getStackedState(fb);

      // Store transition
      replayBuffer.push(
        Float64Array.from(state),
        action,
        reward,
        Float64Array.from(nextState),
        gameOver
      );

      state = nextState;
      episodeSteps++;
      stepsSinceLastTrain++;
      stepsSinceLastTarget++;
      totalTrainSteps++;

      // Train every TRAIN_EVERY steps
      if (stepsSinceLastTrain >= TRAIN_EVERY && replayBuffer.length >= BATCH_SIZE) {
        const loss = trainBatch();
        recentLoss += loss;
        recentLossCount++;
        stepsSinceLastTrain = 0;
      }

      // Soft update target network
      if (stepsSinceLastTarget >= TARGET_EVERY) {
        targetNet.softUpdate(policyNet, TAU);
        stepsSinceLastTarget = 0;
      }

      if (gameOver) break;
    }

    // Track statistics
    const finalScore = global.score || 0;
    const finalLevel = global.currentLevel || 1;
    recentScores.push(finalScore);
    recentLevels.push(finalLevel);
    if (finalScore > bestScore) bestScore = finalScore;

    // Keep only last LOG_EVERY scores for averaging
    if (recentScores.length > LOG_EVERY) recentScores.shift();
    if (recentLevels.length > LOG_EVERY) recentLevels.shift();

    // Log every LOG_EVERY episodes
    if ((ep + 1) % LOG_EVERY === 0) {
      const avgScore = recentScores.reduce((a, b) => a + b, 0) / recentScores.length;
      const avgLevel = recentLevels.reduce((a, b) => a + b, 0) / recentLevels.length;
      const avgLoss = recentLossCount > 0 ? recentLoss / recentLossCount : 0;
      const elapsed = (nodePerf.now() - wallStart) / 1000;
      const epPerSec = ((ep + 1) / elapsed).toFixed(0);

      console.log(
        `Ep ${(ep + 1).toString().padStart(6)} | ` +
        `Avg Score: ${avgScore.toFixed(0).padStart(6)} | ` +
        `Best: ${bestScore.toString().padStart(6)} | ` +
        `Avg Lvl: ${avgLevel.toFixed(1)} | ` +
        `Loss: ${avgLoss.toFixed(4)} | ` +
        `Buf: ${replayBuffer.length.toString().padStart(6)} | ` +
        `${epPerSec} ep/s`
      );

      recentLoss = 0;
      recentLossCount = 0;
    }

    // Export model periodically
    if ((ep + 1) % EXPORT_EVERY === 0) {
      exportModel(ep + 1);
    }
  }

  // Final export
  exportModel(NUM_EPISODES);

  const totalElapsed = (nodePerf.now() - wallStart) / 1000;
  console.log(`\n=== Training complete ===`);
  console.log(`Total time: ${totalElapsed.toFixed(1)}s, Best score: ${bestScore}`);
  console.log(`Total training steps: ${globalStep}, Buffer size: ${replayBuffer.length}`);
}

main();
