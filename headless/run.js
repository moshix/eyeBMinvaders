#!/usr/bin/env node
/**
 * Headless Node.js runner for eyeBMinvaders
 * ==========================================
 * Runs the actual game.js in Node.js with browser API shims.
 * Tests the DQN AI model against the real JS game logic (no Rust sim).
 *
 * Usage:
 *   node headless/run.js                    # Run 5 episodes
 *   node headless/run.js --episodes 20      # Run 20 episodes
 *   node headless/run.js --fps 60           # Run at 60Hz (default 30)
 *   node headless/run.js --diagnose         # Log per-frame Q-values
 *   node headless/run.js --max-steps 50000  # Limit steps per episode
 */

const path = require('path');
const fs = require('fs');
const { installShims, dispatchKeyEvent, stepFrame, advanceMockTime, setMockTime } = require('./browser-shim');

// Parse args
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
  const idx = args.indexOf(name);
  if (idx >= 0 && idx + 1 < args.length) return args[idx + 1];
  return defaultVal;
}
const NUM_EPISODES = parseInt(getArg('--episodes', '5'));
const FPS = parseInt(getArg('--fps', '30'));
const MAX_STEPS = parseInt(getArg('--max-steps', '30000'));
const DIAGNOSE = args.includes('--diagnose');
const DT_MS = 1000 / FPS;

// Install browser shims BEFORE loading game.js
installShims();

// Load game.js into the global scope using runInThisContext
// so let/const declarations become accessible
console.log('Loading game.js...');
const gameCode = fs.readFileSync(path.resolve(__dirname, '..', 'game.js'), 'utf-8');

// game.js uses 'let'/'const' at top level — wrap in a Function to make them
// accessible, or use eval. We convert top-level let/const to var so they
// land on globalThis.
const patchedCode = gameCode
  .replace(/^(let |const )/gm, 'var ')  // top-level let/const -> var (global)
  .replace(/^class /gm, 'var _cls = class '); // rare top-level class

try {
  const vm = require('vm');
  vm.runInThisContext(patchedCode, { filename: 'game.js' });
} catch (e) {
  if (e.message && !e.message.includes('Cannot read properties of null')) {
    console.error('Error loading game.js:', e.message);
  }
}

// Clear all intervals created by game.js (model auto-reload every 30s)
const highId = setInterval(() => {}, 100000);
for (let i = 0; i <= highId; i++) clearInterval(i);

// Verify critical functions exist
const requiredFns = ['buildDQNState', 'applyDQNAction', 'gameLoop', 'restartGame', 'createEnemies'];
const missing = requiredFns.filter(fn => typeof global[fn] !== 'function');
if (missing.length > 0) {
  console.error(`Missing functions: ${missing.join(', ')}`);
  console.error('game.js may not have loaded correctly.');
  process.exit(1);
}
console.log('game.js loaded successfully.');

// Load DQN model
async function loadModel() {
  const modelPath = path.resolve(__dirname, '..', 'models', 'model_weights.json');
  if (!fs.existsSync(modelPath)) {
    console.error('No model_weights.json found. Run without AI.');
    return false;
  }
  // Call the game's own model loader
  if (typeof global.loadDQNModel === 'function') {
    await global.loadDQNModel();
    // After first load, replace with no-op to prevent mid-episode reloads
    global.loadDQNModel = async function() {};
    return global.dqnModel !== null;
  }
  return false;
}

// Run one episode, return stats
function runEpisode(episodeNum) {
  // Reset mock time and clear rAF queue before restart
  setMockTime(1000);
  global._rafCallbacks = [];  // prevent duplicate gameLoop instances

  global.restartGame();

  // restartGame() misses several state resets — fix them here
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

  // Fix lastTime so first deltaTime isn't huge
  global.lastTime = 1000; // match mock time

  // Ensure exactly one gameLoop is queued
  global._rafCallbacks = [global.gameLoop];

  // Prime with one initial frame
  advanceMockTime(DT_MS);
  stepFrame(Date.now());

  const stats = {
    episode: episodeNum,
    score: 0,
    level: 1,
    lives: 6,
    steps: 0,
    maxLevel: 1,
    timeAtEdge: 0,
    bulletHits: 0,
    wallShots: 0,
  };

  for (let step = 0; step < MAX_STEPS; step++) {
    // Advance mock clock by one frame interval
    advanceMockTime(DT_MS);
    const simTime = Date.now(); // reads mock time

    // Step the game loop
    stepFrame(simTime);

    // Read game state
    const gameOver = global.gameOverFlag;
    const score = global.score || 0;
    const level = global.currentLevel || 1;
    const lives = global.player ? global.player.lives : 0;
    const playerX = global.player ? global.player.x / 1024 : 0.5;

    stats.score = score;
    stats.level = level;
    stats.lives = lives;
    stats.steps = step + 1;
    if (level > stats.maxLevel) stats.maxLevel = level;

    // Track edge time
    if (playerX < 0.08 || playerX > 0.92) {
      stats.timeAtEdge++;
    }

    // Diagnose mode: log Q-values periodically
    if (DIAGNOSE && step % 500 === 0 && step > 0 && !gameOver) {
      const actions = ['idle', 'left', 'right', 'fire', 'fire+L', 'fire+R'];
      if (typeof global.buildDQNState === 'function' && global.dqnModel) {
        const rawState = global.buildDQNState();
        let qStr = 'no model';
        if (typeof global.dqnForward === 'function') {
          const nFrames = global.dqnModel.n_frames || 1;
          let state = rawState;
          if (nFrames > 1 && global.dqnFrameBuffer) {
            state = global.dqnPushFrame(rawState);
          }
          const q = global.dqnForward(state);
          if (q) {
            const best = q.indexOf(Math.max(...q));
            qStr = actions.map((a, i) => `${a}:${q[i].toFixed(1)}`).join(' ');
            qStr += ` -> ${actions[best]}`;
          }
        }
        const enemies = global.enemies ? global.enemies.length : '?';
        const bullets = global.bullets ? global.bullets.filter(b => b.isEnemyBullet).length : '?';
        console.log(`    step ${step}: x=${playerX.toFixed(2)} lvl=${level} enemies=${enemies} bullets=${bullets} Q=[${qStr}]`);
      }
    }

    if (gameOver) break;
  }

  return stats;
}

// Main
async function main() {
  console.log(`\nHeadless eyeBMinvaders runner`);
  console.log(`Episodes: ${NUM_EPISODES}, FPS: ${FPS}, Max steps: ${MAX_STEPS}`);
  console.log(`Decision interval: ${DT_MS.toFixed(1)}ms\n`);

  // Load AI model
  const hasModel = await loadModel();
  if (hasModel) {
    console.log(`DQN model loaded: ${global.dqnModel.type}, n_frames=${global.dqnModel.n_frames}`);
    console.log(`Architecture: ${JSON.stringify(global.dqnModel.architecture)}\n`);
  } else {
    console.log('Running with heuristic AI (no DQN model)\n');
  }

  // Initialize game
  if (typeof global.createEnemies === 'function') {
    global.createEnemies();
  }

  const allStats = [];
  const actions = ['idle', 'left', 'right', 'fire', 'fire+L', 'fire+R'];

  for (let ep = 0; ep < NUM_EPISODES; ep++) {
    const stats = runEpisode(ep);
    allStats.push(stats);

    const edgePct = (stats.timeAtEdge / Math.max(stats.steps, 1) * 100).toFixed(0);
    console.log(
      `Ep ${(ep + 1).toString().padStart(3)} | ` +
      `Score: ${stats.score.toString().padStart(7)} | ` +
      `Level: ${stats.maxLevel} | ` +
      `Steps: ${stats.steps.toString().padStart(6)} | ` +
      `Lives: ${stats.lives} | ` +
      `Edge: ${edgePct}%`
    );
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  const avgScore = allStats.reduce((s, st) => s + st.score, 0) / allStats.length;
  const avgLevel = allStats.reduce((s, st) => s + st.maxLevel, 0) / allStats.length;
  const bestScore = Math.max(...allStats.map(s => s.score));
  const bestLevel = Math.max(...allStats.map(s => s.maxLevel));
  const avgEdge = allStats.reduce((s, st) => s + st.timeAtEdge / Math.max(st.steps, 1), 0) / allStats.length * 100;
  console.log(`Summary (${NUM_EPISODES} episodes):`);
  console.log(`  Avg Score: ${avgScore.toFixed(0)} | Best: ${bestScore}`);
  console.log(`  Avg Level: ${avgLevel.toFixed(1)} | Best: ${bestLevel}`);
  console.log(`  Avg Edge Time: ${avgEdge.toFixed(0)}%`);
  console.log('='.repeat(60));
}

main().catch(e => {
  console.error('Fatal error:', e);
  process.exit(1);
});
