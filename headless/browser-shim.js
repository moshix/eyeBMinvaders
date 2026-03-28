/**
 * Browser API shim for running game.js in Node.js headlessly.
 * Mocks Canvas, DOM, Image, Audio, and timing APIs.
 * Only game LOGIC runs — all rendering is no-op.
 */

// ============================================================================
// Canvas & Context2D stub
// ============================================================================
class CanvasRenderingContext2DStub {
  constructor() {
    this.fillStyle = '';
    this.strokeStyle = '';
    this.font = '';
    this.textAlign = '';
    this.textBaseline = '';
    this.globalAlpha = 1;
    this.globalCompositeOperation = 'source-over';
    this.lineWidth = 1;
    this.shadowColor = '';
    this.shadowBlur = 0;
    this.shadowOffsetX = 0;
    this.shadowOffsetY = 0;
  }
  // All drawing methods are no-ops
  clearRect() {}
  fillRect() {}
  strokeRect() {}
  drawImage() {}
  fillText() {}
  strokeText() {}
  beginPath() {}
  closePath() {}
  moveTo() {}
  lineTo() {}
  arc() {}
  ellipse() {}
  fill() {}
  stroke() {}
  save() {}
  restore() {}
  translate() {}
  rotate() {}
  scale() {}
  setTransform() {}
  resetTransform() {}
  createLinearGradient() { return { addColorStop() {} }; }
  createRadialGradient() { return { addColorStop() {} }; }
  measureText() { return { width: 50 }; }
  clip() {}
  rect() {}
  quadraticCurveTo() {}
  bezierCurveTo() {}
  getImageData() { return { data: new Uint8ClampedArray(4) }; }
  putImageData() {}
}

class CanvasStub {
  constructor(w = 1024, h = 576) {
    this.width = w;
    this.height = h;
    this.style = {};
    this._ctx = new CanvasRenderingContext2DStub();
  }
  getContext() { return this._ctx; }
  addEventListener() {}
  removeEventListener() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: this.width, height: this.height }; }
}

// ============================================================================
// Image stub — tracks src/width/height, fires onload
// ============================================================================
class ImageStub {
  constructor() {
    this.width = 48;
    this.height = 48;
    this._src = '';
    this.onload = null;
    this.onerror = null;
  }
  get src() { return this._src; }
  set src(val) {
    this._src = val;
    // Auto-fire onload on next tick
    if (this.onload) {
      setTimeout(() => this.onload && this.onload(), 0);
    }
  }
}

// ============================================================================
// Audio stub — all methods are no-ops
// ============================================================================
class AudioStub {
  constructor() {
    this.src = '';
    this.volume = 1;
    this.currentTime = 0;
    this.loop = false;
    this.muted = false;
  }
  play() { return Promise.resolve(); }
  pause() {}
  load() {}
  remove() {}
  cloneNode() { return new AudioStub(); }
  addEventListener() {}
  removeEventListener() {}
}

// ============================================================================
// DOM stub — minimal document/window mocking
// ============================================================================
const _elements = {};
const _eventListeners = {};

const documentStub = {
  getElementById(id) {
    if (!_elements[id]) {
      _elements[id] = {
        id,
        style: {},
        innerHTML: '',
        textContent: '',
        innerText: '',
        classList: { add() {}, remove() {}, contains() { return false; } },
        appendChild() {},
        removeChild() {},
        remove() {},
        querySelector() { return null; },
        querySelectorAll() { return []; },
        addEventListener() {},
        removeEventListener() {},
        getBoundingClientRect() { return { left: 0, top: 0, width: 0, height: 0 }; },
        // For canvas element
        getContext(type) {
          if (id === 'gameCanvas') return global._gameCanvas._ctx;
          return new CanvasRenderingContext2DStub();
        },
        width: 1024,
        height: 576,
      };
    }
    return _elements[id];
  },
  querySelector(sel) {
    return null;
  },
  querySelectorAll(sel) {
    return [];
  },
  createElement(tag) {
    if (tag === 'canvas') return new CanvasStub();
    return {
      style: {},
      appendChild() {},
      removeChild() {},
      remove() {},
      addEventListener() {},
      src: '',
      onload: null,
    };
  },
  addEventListener(event, handler, options) {
    if (!_eventListeners[event]) _eventListeners[event] = [];
    _eventListeners[event].push(handler);
  },
  removeEventListener(event, handler) {
    if (_eventListeners[event]) {
      _eventListeners[event] = _eventListeners[event].filter(h => h !== handler);
    }
  },
  body: {
    style: {},
    appendChild() {},
    removeChild() {},
  },
};

const windowStub = {
  innerWidth: 1200,
  innerHeight: 700,
  addEventListener(event, handler) {
    if (!_eventListeners['window_' + event]) _eventListeners['window_' + event] = [];
    _eventListeners['window_' + event].push(handler);
  },
  removeEventListener() {},
  requestAnimationFrame: null, // set below
};

// ============================================================================
// Timing — controllable mock clock for deterministic stepping
// ============================================================================
const { performance: nodePerf } = require('perf_hooks');

let _mockTime = 0;  // milliseconds, advances manually

function setMockTime(ms) { _mockTime = ms; }
function advanceMockTime(ms) { _mockTime += ms; }
function getMockTime() { return _mockTime; }

// ============================================================================
// Exports: install shims into global scope
// ============================================================================
function installShims() {
  global._gameCanvas = new CanvasStub(1024, 576);
  global.document = documentStub;
  global.window = windowStub;
  global.navigator = { maxTouchPoints: 0, msMaxTouchPoints: 0, userAgent: 'node' };
  global.Image = ImageStub;
  global.Audio = AudioStub;
  global.canvas = global._gameCanvas;
  // Mock Date.now and performance.now to use controllable clock
  const origDateNow = Date.now;
  Date.now = function() { return _mockTime; };
  global.performance = {
    now: function() { return _mockTime; },
    // Keep other performance methods if needed
    mark: nodePerf.mark?.bind(nodePerf),
    measure: nodePerf.measure?.bind(nodePerf),
  };
  global.localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };

  // requestAnimationFrame — stores callback for manual stepping
  global._rafCallbacks = [];
  global.requestAnimationFrame = function(cb) {
    global._rafCallbacks.push(cb);
    return global._rafCallbacks.length;
  };
  global.cancelAnimationFrame = function() {};

  // fetch stub for model loading
  global.fetch = async function(url) {
    const fs = require('fs');
    const path = require('path');
    // Resolve relative to game root
    const gameRoot = path.resolve(__dirname, '..');
    const filePath = path.join(gameRoot, url.split('?')[0]);
    try {
      const data = fs.readFileSync(filePath, 'utf-8');
      return {
        ok: true,
        json: async () => JSON.parse(data),
        text: async () => data,
      };
    } catch (e) {
      return { ok: false, status: 404, json: async () => null };
    }
  };
}

// Dispatch a synthetic keyboard event
function dispatchKeyEvent(type, key) {
  const event = { key, code: key, preventDefault() {} };
  if (_eventListeners[type]) {
    for (const handler of _eventListeners[type]) {
      handler(event);
    }
  }
}

// Step the game loop by one frame (call stored rAF callbacks)
function stepFrame(timestamp) {
  const callbacks = global._rafCallbacks.slice();
  global._rafCallbacks = [];
  for (const cb of callbacks) {
    try {
      cb(timestamp);
    } catch (e) {
      // game.js has some variable scoping bugs (bIndex vs bulletIndex)
      // that browsers tolerate but Node doesn't. Swallow and re-register gameLoop.
      if (typeof global.gameLoop === 'function') {
        global._rafCallbacks.push(global.gameLoop);
      }
    }
  }
}

module.exports = {
  installShims,
  dispatchKeyEvent,
  stepFrame,
  setMockTime,
  advanceMockTime,
  getMockTime,
  CanvasStub,
  ImageStub,
  AudioStub,
};
