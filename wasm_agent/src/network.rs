use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::math::{self, Linear};

/// Stored activations from a forward pass, needed for backpropagation.
pub struct ForwardResult {
    // Inputs to each layer (pre-activation values fed into that layer)
    pub input: Vec<f32>,           // original input (200)
    pub shared1_out: Vec<f32>,     // after shared layer 1 + relu (256)
    pub shared2_out: Vec<f32>,     // after shared layer 2 + relu (128)
    pub policy1_out: Vec<f32>,     // after policy layer 1 + relu (64)
    pub value1_out: Vec<f32>,      // after value layer 1 + relu (64)
    // Outputs
    pub policy_logits: Vec<f32>,   // raw logits before softmax (6)
    pub policy_probs: Vec<f32>,    // after softmax (6)
    pub value: f32,                // scalar value estimate
}

/// Actor-Critic network for PPO.
///
/// Architecture:
///   SharedFeatures: Linear(200, 256) -> ReLU -> Linear(256, 128) -> ReLU
///   PolicyHead:     Linear(128, 64) -> ReLU -> Linear(64, 6) -> Softmax
///   ValueHead:      Linear(128, 64) -> ReLU -> Linear(64, 1)
pub struct ActorCriticNet {
    // Shared feature layers
    pub shared1: Linear,  // 200 -> 256
    pub shared2: Linear,  // 256 -> 128
    // Policy head
    pub policy1: Linear,  // 128 -> 64
    pub policy2: Linear,  // 64 -> 6
    // Value head
    pub value1: Linear,   // 128 -> 64
    pub value2: Linear,   // 64 -> 1
}

impl ActorCriticNet {
    /// Create a new network with random Kaiming-uniform weights.
    pub fn new_random(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Self {
            shared1: Linear::new(200, 256, &mut rng),
            shared2: Linear::new(256, 128, &mut rng),
            policy1: Linear::new(128, 64, &mut rng),
            policy2: Linear::new(64, 6, &mut rng),
            value1: Linear::new(128, 64, &mut rng),
            value2: Linear::new(64, 1, &mut rng),
        }
    }

    /// Forward pass returning (action_probs, value).
    pub fn forward(&self, state: &[f32]) -> (Vec<f32>, f32) {
        // Shared
        let h1 = self.shared1.forward(state, true);
        let h2 = self.shared2.forward(&h1, true);
        // Policy
        let p1 = self.policy1.forward(&h2, true);
        let logits = self.policy2.forward(&p1, false);
        let probs = math::softmax(&logits);
        // Value
        let v1 = self.value1.forward(&h2, true);
        let val_out = self.value2.forward(&v1, false);
        (probs, val_out[0])
    }

    /// Forward pass that stores all intermediate activations for backpropagation.
    pub fn forward_with_grads(&self, state: &[f32]) -> ForwardResult {
        let input = state.to_vec();
        let shared1_out = self.shared1.forward(state, true);
        let shared2_out = self.shared2.forward(&shared1_out, true);
        let policy1_out = self.policy1.forward(&shared2_out, true);
        let policy_logits = self.policy2.forward(&policy1_out, false);
        let policy_probs = math::softmax(&policy_logits);
        let value1_out = self.value1.forward(&shared2_out, true);
        let value_vec = self.value2.forward(&value1_out, false);

        ForwardResult {
            input,
            shared1_out,
            shared2_out,
            policy1_out,
            value1_out,
            policy_logits,
            policy_probs,
            value: value_vec[0],
        }
    }

    /// Backward pass given policy gradient (dL/d_logits) and value gradient (dL/d_value).
    pub fn backward(&mut self, fwd: &ForwardResult, policy_grad: &[f32], value_grad: f32) {
        // --- Policy head backward ---
        // policy2: input=policy1_out, output=policy_logits (no relu)
        let grad_p2_input = self.policy2.backward(
            &fwd.policy1_out, policy_grad, &fwd.policy_logits, false
        );
        // policy1: input=shared2_out, output=policy1_out (relu)
        let grad_p1_input = self.policy1.backward(
            &fwd.shared2_out, &grad_p2_input, &fwd.policy1_out, true
        );

        // --- Value head backward ---
        let value_grad_vec = vec![value_grad];
        let value_out_vec = vec![fwd.value];
        let grad_v2_input = self.value2.backward(
            &fwd.value1_out, &value_grad_vec, &value_out_vec, false
        );
        let grad_v1_input = self.value1.backward(
            &fwd.shared2_out, &grad_v2_input, &fwd.value1_out, true
        );

        // --- Shared layers backward (sum gradients from both heads) ---
        let grad_shared2_out: Vec<f32> = grad_p1_input.iter()
            .zip(grad_v1_input.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let grad_s2_input = self.shared2.backward(
            &fwd.shared1_out, &grad_shared2_out, &fwd.shared2_out, true
        );
        let _grad_s1_input = self.shared1.backward(
            &fwd.input, &grad_s2_input, &fwd.shared1_out, true
        );
    }

    /// Apply gradient clipping (max norm) and Adam update to all layers.
    pub fn update(&mut self, lr: f32, t: u32, max_grad_norm: f32) {
        // Compute total gradient norm
        let mut total_norm_sq = 0.0f32;
        let layers = self.layers_mut_refs();
        for layer in &layers {
            total_norm_sq += layer_grad_norm_sq(*layer);
        }
        let total_norm = total_norm_sq.sqrt();

        // Clip if needed
        if total_norm > max_grad_norm {
            let scale = max_grad_norm / (total_norm + 1e-8);
            for layer in &layers {
                layer_scale_grads(*layer, scale);
            }
        }

        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let epsilon = 1e-8f32;

        // We need mutable access to each layer for update
        self.shared1.update(lr, beta1, beta2, epsilon, t);
        self.shared2.update(lr, beta1, beta2, epsilon, t);
        self.policy1.update(lr, beta1, beta2, epsilon, t);
        self.policy2.update(lr, beta1, beta2, epsilon, t);
        self.value1.update(lr, beta1, beta2, epsilon, t);
        self.value2.update(lr, beta1, beta2, epsilon, t);
    }

    /// Zero gradients on all layers.
    pub fn zero_grad(&mut self) {
        self.shared1.zero_grad();
        self.shared2.zero_grad();
        self.policy1.zero_grad();
        self.policy2.zero_grad();
        self.value1.zero_grad();
        self.value2.zero_grad();
    }

    /// Helper to get raw pointers for gradient norm/clipping (avoids borrow issues).
    fn layers_mut_refs(&mut self) -> [*mut Linear; 6] {
        [
            &mut self.shared1 as *mut Linear,
            &mut self.shared2 as *mut Linear,
            &mut self.policy1 as *mut Linear,
            &mut self.policy2 as *mut Linear,
            &mut self.value1 as *mut Linear,
            &mut self.value2 as *mut Linear,
        ]
    }

    /// Load weights from the DQN model_weights.json format.
    /// The DQN has a different architecture, so this does a best-effort partial load
    /// of the shared feature layers only. Returns true on success.
    pub fn load_weights_json(&mut self, json: &str) -> bool {
        let parsed: Result<ModelWeightsJson, _> = serde_json::from_str(json);
        match parsed {
            Ok(mw) => {
                self.load_from_model_weights(&mw);
                true
            }
            Err(_) => false,
        }
    }

    /// Export all weights as JSON for saving/loading PPO checkpoints.
    pub fn export_weights_json(&self) -> String {
        let export = PpoWeightsJson {
            shared1_w: self.shared1.weights_flat(),
            shared1_b: self.shared1.bias.clone(),
            shared2_w: self.shared2.weights_flat(),
            shared2_b: self.shared2.bias.clone(),
            policy1_w: self.policy1.weights_flat(),
            policy1_b: self.policy1.bias.clone(),
            policy2_w: self.policy2.weights_flat(),
            policy2_b: self.policy2.bias.clone(),
            value1_w: self.value1.weights_flat(),
            value1_b: self.value1.bias.clone(),
            value2_w: self.value2.weights_flat(),
            value2_b: self.value2.bias.clone(),
        };
        serde_json::to_string(&export).unwrap_or_default()
    }

    /// Load PPO weights from our own export format.
    pub fn load_ppo_weights_json(&mut self, json: &str) -> bool {
        let parsed: Result<PpoWeightsJson, _> = serde_json::from_str(json);
        match parsed {
            Ok(w) => {
                self.shared1.load_weights_flat(&w.shared1_w);
                self.shared1.load_bias(&w.shared1_b);
                self.shared2.load_weights_flat(&w.shared2_w);
                self.shared2.load_bias(&w.shared2_b);
                self.policy1.load_weights_flat(&w.policy1_w);
                self.policy1.load_bias(&w.policy1_b);
                self.policy2.load_weights_flat(&w.policy2_w);
                self.policy2.load_bias(&w.policy2_b);
                self.value1.load_weights_flat(&w.value1_w);
                self.value1.load_bias(&w.value1_b);
                self.value2.load_weights_flat(&w.value2_w);
                self.value2.load_bias(&w.value2_b);
                true
            }
            Err(_) => false,
        }
    }

    /// Best-effort load from the DQN model_weights.json format.
    /// DQN architecture: 200->256->256->128->6
    /// PPO shared: 200->256->128, so we load the first DQN layer into shared1.
    fn load_from_model_weights(&mut self, mw: &ModelWeightsJson) {
        // DQN layer 0 (200->256) maps to shared1 (200->256)
        if mw.layers.len() >= 1 {
            let layer = &mw.layers[0];
            if layer.weights.len() == 256 && layer.weights[0].len() == 200 {
                for (i, row) in layer.weights.iter().enumerate() {
                    self.shared1.weight[i].copy_from_slice(row);
                }
                self.shared1.bias.copy_from_slice(&layer.biases);
            }
        }
        // DQN layer 1 (256->256) is skipped (PPO shared2 is 256->128, different shape)
        // Remaining layers are incompatible, left as random initialization.
    }
}

// Helpers that work through raw pointers to avoid simultaneous borrows.
fn layer_grad_norm_sq(ptr: *mut Linear) -> f32 {
    unsafe { (*ptr).grad_norm_sq() }
}

fn layer_scale_grads(ptr: *mut Linear, factor: f32) {
    unsafe { (*ptr).scale_grads(factor) }
}

/// JSON format for the existing DQN model_weights.json
#[derive(Deserialize)]
struct ModelWeightsJson {
    layers: Vec<DqnLayer>,
}

#[derive(Deserialize)]
struct DqnLayer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

/// JSON format for PPO weight checkpoints.
#[derive(Serialize, Deserialize)]
struct PpoWeightsJson {
    shared1_w: Vec<f32>,
    shared1_b: Vec<f32>,
    shared2_w: Vec<f32>,
    shared2_b: Vec<f32>,
    policy1_w: Vec<f32>,
    policy1_b: Vec<f32>,
    policy2_w: Vec<f32>,
    policy2_b: Vec<f32>,
    value1_w: Vec<f32>,
    value1_b: Vec<f32>,
    value2_w: Vec<f32>,
    value2_b: Vec<f32>,
}
