/**
 * DQN Training Math Module — Pure JS, no game dependencies
 * =========================================================
 * Provides Linear layers with forward/backward, Adam optimizer,
 * Dueling DQN network, and replay buffer for online training.
 */

'use strict';

// ---------------------------------------------------------------------------
// He (Kaiming) initialization for ReLU layers
// ---------------------------------------------------------------------------
function heInit(fanIn) {
  return Math.sqrt(2.0 / fanIn);
}

function randn() {
  // Box-Muller transform
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1 || 1e-12)) * Math.cos(2.0 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Linear Layer with forward, backward, and Adam optimizer
// ---------------------------------------------------------------------------
class Linear {
  /**
   * @param {number} inSize  - input dimension
   * @param {number} outSize - output dimension
   */
  constructor(inSize, outSize) {
    this.inSize = inSize;
    this.outSize = outSize;

    // Weights: [outSize][inSize], biases: [outSize]
    const scale = heInit(inSize);
    this.weight = [];
    this.bias = [];
    for (let i = 0; i < outSize; i++) {
      const row = new Float64Array(inSize);
      for (let j = 0; j < inSize; j++) row[j] = randn() * scale;
      this.weight.push(row);
      this.bias.push(0.0);
    }

    // Gradient accumulators
    this.gradWeight = [];
    this.gradBias = new Float64Array(outSize);
    for (let i = 0; i < outSize; i++) {
      this.gradWeight.push(new Float64Array(inSize));
    }

    // Adam moment estimates
    this.mW = [];
    this.vW = [];
    for (let i = 0; i < outSize; i++) {
      this.mW.push(new Float64Array(inSize));
      this.vW.push(new Float64Array(inSize));
    }
    this.mB = new Float64Array(outSize);
    this.vB = new Float64Array(outSize);
  }

  /**
   * Forward pass: y = Wx + b, optionally apply ReLU.
   * @param {number[]|Float64Array} x - input vector [inSize]
   * @param {boolean} relu - apply ReLU activation
   * @returns {Float64Array} output [outSize]
   */
  forward(x, relu) {
    const out = new Float64Array(this.outSize);
    for (let i = 0; i < this.outSize; i++) {
      let sum = this.bias[i];
      const row = this.weight[i];
      for (let j = 0; j < this.inSize; j++) sum += row[j] * x[j];
      out[i] = relu ? Math.max(0.0, sum) : sum;
    }
    return out;
  }

  /**
   * Backward pass: accumulate gradients, return gradInput.
   * @param {number[]|Float64Array} input   - the input that was passed to forward
   * @param {Float64Array} gradOutput        - gradient from upstream [outSize]
   * @param {Float64Array} output            - output of forward (for ReLU mask)
   * @param {boolean} relu                   - was ReLU applied in forward?
   * @returns {Float64Array} gradInput [inSize]
   */
  backward(input, gradOutput, output, relu) {
    const gradIn = new Float64Array(this.inSize);

    for (let i = 0; i < this.outSize; i++) {
      // Apply ReLU derivative: if output <= 0, gradient is 0
      const g = (relu && output[i] <= 0) ? 0.0 : gradOutput[i];
      if (g === 0.0) continue;

      this.gradBias[i] += g;
      const gw = this.gradWeight[i];
      const w = this.weight[i];
      for (let j = 0; j < this.inSize; j++) {
        gw[j] += g * input[j];
        gradIn[j] += w[j] * g;
      }
    }

    return gradIn;
  }

  /**
   * Adam optimizer step.
   * @param {number} lr    - learning rate
   * @param {number} t     - global step (1-indexed for bias correction)
   * @param {number} beta1 - first moment decay (default 0.9)
   * @param {number} beta2 - second moment decay (default 0.999)
   * @param {number} eps   - numerical stability (default 1e-8)
   */
  update(lr, t, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    const bc1 = 1.0 / (1.0 - Math.pow(beta1, t));
    const bc2 = 1.0 / (1.0 - Math.pow(beta2, t));

    for (let i = 0; i < this.outSize; i++) {
      const gw = this.gradWeight[i];
      const mw = this.mW[i];
      const vw = this.vW[i];
      const w = this.weight[i];
      for (let j = 0; j < this.inSize; j++) {
        mw[j] = beta1 * mw[j] + (1 - beta1) * gw[j];
        vw[j] = beta2 * vw[j] + (1 - beta2) * gw[j] * gw[j];
        const mHat = mw[j] * bc1;
        const vHat = vw[j] * bc2;
        w[j] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }

      // Bias
      this.mB[i] = beta1 * this.mB[i] + (1 - beta1) * this.gradBias[i];
      this.vB[i] = beta2 * this.vB[i] + (1 - beta2) * this.gradBias[i] * this.gradBias[i];
      const mBHat = this.mB[i] * bc1;
      const vBHat = this.vB[i] * bc2;
      this.bias[i] -= lr * mBHat / (Math.sqrt(vBHat) + eps);
    }
  }

  /** Reset gradient accumulators to zero. */
  zeroGrad() {
    for (let i = 0; i < this.outSize; i++) {
      this.gradWeight[i].fill(0.0);
    }
    this.gradBias.fill(0.0);
  }
}

// ---------------------------------------------------------------------------
// Dueling DQN Network
// ---------------------------------------------------------------------------
class DuelingDQN {
  /**
   * @param {number} inputSize - state dimension (e.g. 216 for 54*4 frames)
   */
  constructor(inputSize) {
    this.inputSize = inputSize;
    // Shared feature extractor (matches model: features.0 and features.2)
    this.shared1 = new Linear(inputSize, 512);
    this.shared2 = new Linear(512, 256);
    // Value stream
    this.value1 = new Linear(256, 128);
    this.value2 = new Linear(128, 1);
    // Advantage stream
    this.adv1 = new Linear(256, 128);
    this.adv2 = new Linear(128, 6);

    // Cache for backward pass
    this._cache = null;
  }

  /** All layers in order (for iteration). */
  _layers() {
    return [this.shared1, this.shared2, this.value1, this.value2, this.adv1, this.adv2];
  }

  /**
   * Forward pass (inference only, no cache).
   * @param {number[]|Float64Array} state
   * @returns {Float64Array} Q-values [6]
   */
  forward(state) {
    const h1 = this.shared1.forward(state, true);
    const h2 = this.shared2.forward(h1, true);
    let v = this.value1.forward(h2, true);
    v = this.value2.forward(v, false);
    let a = this.adv1.forward(h2, true);
    a = this.adv2.forward(a, false);
    // Q = V + A - mean(A)
    let aMean = 0;
    for (let i = 0; i < 6; i++) aMean += a[i];
    aMean /= 6.0;
    const q = new Float64Array(6);
    for (let i = 0; i < 6; i++) q[i] = v[0] + a[i] - aMean;
    return q;
  }

  /**
   * Forward pass with activation caching for backward.
   * @param {number[]|Float64Array} state
   * @returns {Float64Array} Q-values [6]
   */
  forwardTrain(state) {
    const x0 = state instanceof Float64Array ? state : Float64Array.from(state);
    const h1 = this.shared1.forward(x0, true);
    const h2 = this.shared2.forward(h1, true);

    const vh1 = this.value1.forward(h2, true);
    const vh2 = this.value2.forward(vh1, false);

    const ah1 = this.adv1.forward(h2, true);
    const ah2 = this.adv2.forward(ah1, false);

    let aMean = 0;
    for (let i = 0; i < 6; i++) aMean += ah2[i];
    aMean /= 6.0;

    const q = new Float64Array(6);
    for (let i = 0; i < 6; i++) q[i] = vh2[0] + ah2[i] - aMean;

    this._cache = { x0, h1, h2, vh1, vh2, ah1, ah2, aMean, q };
    return q;
  }

  /**
   * Backward pass: compute gradients for Huber loss on a single (action, target) pair.
   * Huber loss: L = 0.5*d^2 if |d|<=1, else |d|-0.5, where d = Q[action] - target.
   * Accumulates into layer .gradWeight/.gradBias.
   *
   * @param {number} action - action index (0-5)
   * @param {number} target - target Q-value for this action
   * @returns {number} loss value
   */
  backward(action, target) {
    const c = this._cache;
    const d = c.q[action] - target;

    // Huber loss and its derivative
    let loss, dLdQ;
    if (Math.abs(d) <= 1.0) {
      loss = 0.5 * d * d;
      dLdQ = d;
    } else {
      loss = Math.abs(d) - 0.5;
      dLdQ = d > 0 ? 1.0 : -1.0;
    }

    // dL/dQ is non-zero only for the chosen action
    // Dueling decomposition: Q[i] = V + A[i] - mean(A)
    // dL/dA[i] = dL/dQ[i] - (1/6)*sum_j(dL/dQ[j])
    // dL/dV   = sum_j(dL/dQ[j])
    // Since only dL/dQ[action] is non-zero:
    //   dL/dV = dLdQ
    //   dL/dA[action] = dLdQ * (1 - 1/6) = dLdQ * 5/6
    //   dL/dA[other]  = dLdQ * (-1/6)

    const gradA = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      gradA[i] = (i === action) ? dLdQ * (5.0 / 6.0) : dLdQ * (-1.0 / 6.0);
    }
    const gradV = new Float64Array(1);
    gradV[0] = dLdQ;

    // Backprop through advantage stream: adv2 -> adv1
    const gradAh1 = this.adv2.backward(c.ah1, gradA, c.ah2, false);
    const gradH2_a = this.adv1.backward(c.h2, gradAh1, c.ah1, true);

    // Backprop through value stream: value2 -> value1
    const gradVh1 = this.value2.backward(c.vh1, gradV, c.vh2, false);
    const gradH2_v = this.value1.backward(c.h2, gradVh1, c.vh1, true);

    // Merge gradients from both streams at shared output
    const gradH2 = new Float64Array(256);
    for (let i = 0; i < 256; i++) gradH2[i] = gradH2_a[i] + gradH2_v[i];

    // Backprop through shared layers
    const gradH1 = this.shared2.backward(c.h1, gradH2, c.h2, true);
    this.shared1.backward(c.x0, gradH1, c.h1, true);

    return loss;
  }

  /**
   * Run Adam update on all layers.
   * @param {number} lr - learning rate
   * @param {number} t  - global step count (1-indexed)
   */
  update(lr, t) {
    for (const layer of this._layers()) {
      layer.update(lr, t);
    }
  }

  /** Zero all gradient accumulators. */
  zeroGrad() {
    for (const layer of this._layers()) {
      layer.zeroGrad();
    }
  }

  /**
   * Copy all weights from another DuelingDQN (hard copy).
   * @param {DuelingDQN} other
   */
  copyFrom(other) {
    const src = other._layers();
    const dst = this._layers();
    for (let l = 0; l < dst.length; l++) {
      for (let i = 0; i < dst[l].outSize; i++) {
        dst[l].weight[i].set(src[l].weight[i]);
        dst[l].bias[i] = src[l].bias[i];
      }
    }
  }

  /**
   * Polyak (soft) averaging: this = (1-tau)*this + tau*other.
   * @param {DuelingDQN} other - source network
   * @param {number} tau       - interpolation factor (e.g. 0.005)
   */
  softUpdate(other, tau) {
    const src = other._layers();
    const dst = this._layers();
    for (let l = 0; l < dst.length; l++) {
      for (let i = 0; i < dst[l].outSize; i++) {
        const dw = dst[l].weight[i];
        const sw = src[l].weight[i];
        for (let j = 0; j < dst[l].inSize; j++) {
          dw[j] = (1 - tau) * dw[j] + tau * sw[j];
        }
        dst[l].bias[i] = (1 - tau) * dst[l].bias[i] + tau * src[l].bias[i];
      }
    }
  }

  /**
   * Load weights from model_weights.json format (as used by game.js).
   * @param {object} model - parsed JSON with .weights, .type, .architecture, .n_frames
   */
  loadWeightsJSON(model) {
    const w = model.weights;

    function loadLayer(layer, wKey, bKey) {
      const wData = w[wKey];
      const bData = w[bKey];
      if (!wData || !bData) throw new Error(`Missing weight key: ${wKey} or ${bKey}`);
      for (let i = 0; i < layer.outSize; i++) {
        for (let j = 0; j < layer.inSize; j++) {
          layer.weight[i][j] = wData[i][j];
        }
        layer.bias[i] = bData[i];
      }
    }

    loadLayer(this.shared1, 'features.0.weight', 'features.0.bias');
    loadLayer(this.shared2, 'features.2.weight', 'features.2.bias');
    loadLayer(this.value1, 'value_hidden.weight', 'value_hidden.bias');
    loadLayer(this.value2, 'value_out.weight', 'value_out.bias');
    loadLayer(this.adv1, 'adv_hidden.weight', 'adv_hidden.bias');
    loadLayer(this.adv2, 'adv_out.weight', 'adv_out.bias');
  }

  /**
   * Export weights to model_weights.json format (readable by game.js).
   * @returns {object} JSON-serializable model object
   */
  exportWeightsJSON() {
    function exportLayer(layer) {
      const w = [];
      for (let i = 0; i < layer.outSize; i++) {
        w.push(Array.from(layer.weight[i]));
      }
      return { weight: w, bias: Array.from(layer.bias instanceof Float64Array ? layer.bias : layer.bias) };
    }

    const s1 = exportLayer(this.shared1);
    const s2 = exportLayer(this.shared2);
    const v1 = exportLayer(this.value1);
    const v2 = exportLayer(this.value2);
    const a1 = exportLayer(this.adv1);
    const a2 = exportLayer(this.adv2);

    return {
      type: 'dueling',
      architecture: [this.inputSize, 512, 256, 128, 6],
      n_frames: 4,
      weights: {
        'features.0.weight': s1.weight,
        'features.0.bias': s1.bias,
        'features.2.weight': s2.weight,
        'features.2.bias': s2.bias,
        'value_hidden.weight': v1.weight,
        'value_hidden.bias': v1.bias,
        'value_out.weight': v2.weight,
        'value_out.bias': v2.bias,
        'adv_hidden.weight': a1.weight,
        'adv_hidden.bias': a1.bias,
        'adv_out.weight': a2.weight,
        'adv_out.bias': a2.bias,
      },
    };
  }
}

// ---------------------------------------------------------------------------
// Replay Buffer — circular buffer with uniform random sampling
// ---------------------------------------------------------------------------
class ReplayBuffer {
  /**
   * @param {number} capacity - maximum number of transitions
   */
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = [];
    this.pos = 0;
  }

  /**
   * Store a transition.
   * @param {Float64Array} state
   * @param {number} action
   * @param {number} reward
   * @param {Float64Array} nextState
   * @param {boolean} done
   */
  push(state, action, reward, nextState, done) {
    const transition = { state, action, reward, nextState, done };
    if (this.buffer.length < this.capacity) {
      this.buffer.push(transition);
    } else {
      this.buffer[this.pos] = transition;
    }
    this.pos = (this.pos + 1) % this.capacity;
  }

  /**
   * Sample a random batch.
   * @param {number} batchSize
   * @returns {Array<{state, action, reward, nextState, done}>}
   */
  sample(batchSize) {
    const batch = [];
    const len = this.buffer.length;
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * len);
      batch.push(this.buffer[idx]);
    }
    return batch;
  }

  get length() {
    return this.buffer.length;
  }
}

module.exports = { Linear, DuelingDQN, ReplayBuffer };
