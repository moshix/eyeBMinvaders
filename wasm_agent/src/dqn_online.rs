use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::math::Linear;

// ---------------------------------------------------------------------------
// Replay Buffer
// ---------------------------------------------------------------------------

pub struct ReplayBuffer {
    states: Vec<Vec<f32>>,
    actions: Vec<u8>,
    rewards: Vec<f32>,
    next_states: Vec<Vec<f32>>,
    dones: Vec<bool>,
    capacity: usize,
    pos: usize,
    size: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            capacity,
            pos: 0,
            size: 0,
        }
    }

    pub fn push(
        &mut self,
        state: Vec<f32>,
        action: u8,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        if self.size < self.capacity {
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.next_states.push(next_state);
            self.dones.push(done);
            self.size += 1;
        } else {
            self.states[self.pos] = state;
            self.actions[self.pos] = action;
            self.rewards[self.pos] = reward;
            self.next_states[self.pos] = next_state;
            self.dones[self.pos] = done;
        }
        self.pos = (self.pos + 1) % self.capacity;
    }

    pub fn sample(
        &self,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> (Vec<Vec<f32>>, Vec<u8>, Vec<f32>, Vec<Vec<f32>>, Vec<bool>) {
        let mut states = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.gen_range(0..self.size);
            states.push(self.states[idx].clone());
            actions.push(self.actions[idx]);
            rewards.push(self.rewards[idx]);
            next_states.push(self.next_states[idx].clone());
            dones.push(self.dones[idx]);
        }

        (states, actions, rewards, next_states, dones)
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

// ---------------------------------------------------------------------------
// Dueling DQN Network
// ---------------------------------------------------------------------------

/// Dueling DQN network: shared features -> value stream + advantage stream.
///
/// Architecture:
///   shared1: input(216) -> 512, ReLU
///   shared2: 512 -> 256, ReLU
///   value1:  256 -> 128, ReLU
///   value2:  128 -> 1
///   adv1:    256 -> 128, ReLU
///   adv2:    128 -> 6
///
///   Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
pub struct DuelingNetwork {
    pub shared1: Linear,
    pub shared2: Linear,
    pub value1: Linear,
    pub value2: Linear,
    pub adv1: Linear,
    pub adv2: Linear,
}

/// Stored activations from a forward pass, needed for backpropagation.
struct DuelingForward {
    input: Vec<f32>,
    h1: Vec<f32>,
    h2: Vec<f32>,
    v1: Vec<f32>,
    v2: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
}

impl DuelingNetwork {
    /// Create with random Kaiming-uniform initialization.
    pub fn new(input_size: usize, rng: &mut ChaCha8Rng) -> Self {
        Self {
            shared1: Linear::new(input_size, 512, rng),
            shared2: Linear::new(512, 256, rng),
            value1: Linear::new(256, 128, rng),
            value2: Linear::new(128, 1, rng),
            adv1: Linear::new(256, 128, rng),
            adv2: Linear::new(128, 6, rng),
        }
    }

    /// Forward pass returning 6 Q-values.
    pub fn forward(&self, state: &[f32]) -> Vec<f32> {
        let h1 = self.shared1.forward(state, true);
        let h2 = self.shared2.forward(&h1, true);
        let v1 = self.value1.forward(&h2, true);
        let v = self.value2.forward(&v1, false);
        let a1 = self.adv1.forward(&h2, true);
        let a = self.adv2.forward(&a1, false);

        let a_mean = a.iter().sum::<f32>() / a.len() as f32;
        a.iter().map(|ai| v[0] + ai - a_mean).collect()
    }

    /// Forward pass storing all intermediate activations.
    fn forward_store(&self, state: &[f32]) -> (Vec<f32>, DuelingForward) {
        let input = state.to_vec();
        let h1 = self.shared1.forward(state, true);
        let h2 = self.shared2.forward(&h1, true);
        let v1 = self.value1.forward(&h2, true);
        let v2 = self.value2.forward(&v1, false);
        let a1 = self.adv1.forward(&h2, true);
        let a2 = self.adv2.forward(&a1, false);

        let n = a2.len() as f32;
        let a_mean = a2.iter().sum::<f32>() / n;
        let q: Vec<f32> = a2.iter().map(|ai| v2[0] + ai - a_mean).collect();

        let fwd = DuelingForward { input, h1, h2, v1, v2, a1, a2 };
        (q, fwd)
    }

    /// Compute Huber loss for a batch and accumulate gradients.
    /// Returns the mean loss.
    pub fn forward_batch_loss(
        &mut self,
        states: &[Vec<f32>],
        actions: &[u8],
        targets: &[f32],
    ) -> f32 {
        let batch_size = states.len();
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let (q, fwd) = self.forward_store(&states[i]);
            let action = actions[i] as usize;
            let predicted = q[action];
            let error = predicted - targets[i];

            // Huber loss
            let loss = if error.abs() < 1.0 {
                0.5 * error * error
            } else {
                error.abs() - 0.5
            };
            total_loss += loss;

            // Gradient of Huber loss w.r.t. predicted
            let grad = if error.abs() < 1.0 { error } else { error.signum() };

            // Scale gradient by 1/batch_size for mean loss
            let grad = grad / batch_size as f32;

            // Build gradient of Q-values: only the selected action gets gradient
            let mut q_grad = vec![0.0; 6];
            q_grad[action] = grad;

            // Backprop through dueling architecture:
            // Q[i] = V + A[i] - mean(A)
            // dL/dA[i] = dL/dQ[i] - (1/n) * sum_j(dL/dQ[j])
            // dL/dV   = sum_j(dL/dQ[j])
            let n_actions = 6.0;
            let q_grad_sum: f32 = q_grad.iter().sum();

            // Gradient w.r.t. advantage output (before mean subtraction)
            let adv_grad: Vec<f32> = q_grad
                .iter()
                .map(|&g| g - q_grad_sum / n_actions)
                .collect();

            // Gradient w.r.t. value output
            let val_grad = vec![q_grad_sum];

            // Backprop advantage stream
            let grad_a1 = self.adv2.backward(&fwd.a1, &adv_grad, &fwd.a2, false);
            let grad_adv_h2 = self.adv1.backward(&fwd.h2, &grad_a1, &fwd.a1, true);

            // Backprop value stream
            let grad_v1 = self.value2.backward(&fwd.v1, &val_grad, &fwd.v2, false);
            let grad_val_h2 = self.value1.backward(&fwd.h2, &grad_v1, &fwd.v1, true);

            // Merge gradients at shared2 output
            let grad_h2: Vec<f32> = grad_adv_h2
                .iter()
                .zip(grad_val_h2.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            // Backprop shared layers
            let grad_h1 =
                self.shared2.backward(&fwd.h1, &grad_h2, &fwd.h2, true);
            let _grad_input =
                self.shared1.backward(&fwd.input, &grad_h1, &fwd.h1, true);
        }

        total_loss / batch_size as f32
    }

    /// Copy all weights from another network.
    pub fn copy_from(&mut self, other: &DuelingNetwork) {
        copy_layer(&mut self.shared1, &other.shared1);
        copy_layer(&mut self.shared2, &other.shared2);
        copy_layer(&mut self.value1, &other.value1);
        copy_layer(&mut self.value2, &other.value2);
        copy_layer(&mut self.adv1, &other.adv1);
        copy_layer(&mut self.adv2, &other.adv2);
    }

    /// Polyak averaging: self = tau * other + (1 - tau) * self.
    pub fn soft_update(&mut self, other: &DuelingNetwork, tau: f32) {
        soft_update_layer(&mut self.shared1, &other.shared1, tau);
        soft_update_layer(&mut self.shared2, &other.shared2, tau);
        soft_update_layer(&mut self.value1, &other.value1, tau);
        soft_update_layer(&mut self.value2, &other.value2, tau);
        soft_update_layer(&mut self.adv1, &other.adv1, tau);
        soft_update_layer(&mut self.adv2, &other.adv2, tau);
    }

    /// Adam step on all layers.
    pub fn update(&mut self, lr: f32, t: u32) {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        self.shared1.update(lr, beta1, beta2, eps, t);
        self.shared2.update(lr, beta1, beta2, eps, t);
        self.value1.update(lr, beta1, beta2, eps, t);
        self.value2.update(lr, beta1, beta2, eps, t);
        self.adv1.update(lr, beta1, beta2, eps, t);
        self.adv2.update(lr, beta1, beta2, eps, t);
    }

    /// Zero all accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.shared1.zero_grad();
        self.shared2.zero_grad();
        self.value1.zero_grad();
        self.value2.zero_grad();
        self.adv1.zero_grad();
        self.adv2.zero_grad();
    }
}

// ---------------------------------------------------------------------------
// Weight I/O
// ---------------------------------------------------------------------------

impl DuelingNetwork {
    /// Load weights from the model_weights.json format.
    /// Supports both standard and NoisyNet key variants.
    /// Returns true on success.
    pub fn load_weights_json(&mut self, json: &str) -> bool {
        // The JSON has dot-separated keys like "features.0.weight" which serde
        // cannot handle via struct field aliases. Parse as a generic Value first.
        let parsed: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(_) => return false,
        };

        let weights = match parsed.get("weights") {
            Some(w) => w,
            None => return false,
        };

        // Helper: extract a 2D weight matrix [out][in]
        let get_w = |key: &str| -> Option<Vec<Vec<f32>>> {
            weights.get(key).and_then(|v| {
                serde_json::from_value(v.clone()).ok()
            })
        };
        // Helper: extract a 1D bias vector
        let get_b = |key: &str| -> Option<Vec<f32>> {
            weights.get(key).and_then(|v| {
                serde_json::from_value(v.clone()).ok()
            })
        };

        // shared1 = features.0
        if let (Some(w), Some(b)) =
            (get_w("features.0.weight"), get_b("features.0.bias"))
        {
            load_layer_2d(&mut self.shared1, &w, &b);
        } else {
            return false;
        }

        // shared2 = features.2
        if let (Some(w), Some(b)) =
            (get_w("features.2.weight"), get_b("features.2.bias"))
        {
            load_layer_2d(&mut self.shared2, &w, &b);
        } else {
            return false;
        }

        // value_hidden (try standard then NoisyNet mu)
        let vh_w = get_w("value_hidden.weight")
            .or_else(|| get_w("value_hidden.weight_mu"));
        let vh_b = get_b("value_hidden.bias")
            .or_else(|| get_b("value_hidden.bias_mu"));
        if let (Some(w), Some(b)) = (vh_w, vh_b) {
            load_layer_2d(&mut self.value1, &w, &b);
        } else {
            return false;
        }

        // value_out
        let vo_w =
            get_w("value_out.weight").or_else(|| get_w("value_out.weight_mu"));
        let vo_b =
            get_b("value_out.bias").or_else(|| get_b("value_out.bias_mu"));
        if let (Some(w), Some(b)) = (vo_w, vo_b) {
            load_layer_2d(&mut self.value2, &w, &b);
        } else {
            return false;
        }

        // adv_hidden
        let ah_w = get_w("adv_hidden.weight")
            .or_else(|| get_w("adv_hidden.weight_mu"));
        let ah_b = get_b("adv_hidden.bias")
            .or_else(|| get_b("adv_hidden.bias_mu"));
        if let (Some(w), Some(b)) = (ah_w, ah_b) {
            load_layer_2d(&mut self.adv1, &w, &b);
        } else {
            return false;
        }

        // adv_out
        let ao_w =
            get_w("adv_out.weight").or_else(|| get_w("adv_out.weight_mu"));
        let ao_b =
            get_b("adv_out.bias").or_else(|| get_b("adv_out.bias_mu"));
        if let (Some(w), Some(b)) = (ao_w, ao_b) {
            load_layer_2d(&mut self.adv2, &w, &b);
        } else {
            return false;
        }

        true
    }

    /// Export weights in the same JSON format as model_weights.json.
    pub fn export_weights_json(&self) -> String {
        // Build JSON manually to get dot-separated keys
        let mut s = String::from("{\n");
        s.push_str("  \"type\": \"dueling\",\n");
        s.push_str("  \"n_frames\": 4,\n");
        s.push_str("  \"architecture\": [216, 512, 256, 128, 6],\n");
        s.push_str("  \"activation\": \"relu\",\n");
        s.push_str("  \"weights\": {\n");

        fn write_layer(
            s: &mut String,
            prefix: &str,
            layer: &Linear,
            trailing_comma: bool,
        ) {
            // weight: [out][in]
            s.push_str(&format!("    \"{}.weight\": [", prefix));
            for (i, row) in layer.weight.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push('[');
                for (j, &v) in row.iter().enumerate() {
                    if j > 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&format!("{}", v));
                }
                s.push(']');
            }
            s.push_str("],\n");

            // bias
            s.push_str(&format!("    \"{}.bias\": [", prefix));
            for (i, &v) in layer.bias.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&format!("{}", v));
            }
            s.push(']');
            if trailing_comma {
                s.push(',');
            }
            s.push('\n');
        }

        write_layer(&mut s, "features.0", &self.shared1, true);
        write_layer(&mut s, "features.2", &self.shared2, true);
        write_layer(&mut s, "value_hidden", &self.value1, true);
        write_layer(&mut s, "value_out", &self.value2, true);
        write_layer(&mut s, "adv_hidden", &self.adv1, true);
        write_layer(&mut s, "adv_out", &self.adv2, false);

        s.push_str("  }\n}");
        s
    }
}

// ---------------------------------------------------------------------------
// Online DQN Agent
// ---------------------------------------------------------------------------

pub struct OnlineDQNConfig {
    pub lr: f32,
    pub gamma: f32,
    pub batch_size: usize,
    pub buffer_capacity: usize,
    pub train_every: usize,
    pub target_update_every: usize,
    pub tau: f32,
    pub min_buffer: usize,
    pub epsilon: f32,
}

impl Default for OnlineDQNConfig {
    fn default() -> Self {
        Self {
            lr: 1e-5,
            gamma: 0.99,
            batch_size: 32,
            buffer_capacity: 50_000,
            train_every: 4,
            target_update_every: 1000,
            tau: 0.005,
            min_buffer: 1000,
            epsilon: 0.05,
        }
    }
}

pub struct OnlineDQN {
    pub policy_net: DuelingNetwork,
    pub target_net: DuelingNetwork,
    pub buffer: ReplayBuffer,
    pub config: OnlineDQNConfig,
    pub steps: u64,
    pub updates: u64,
    pub last_loss: f32,
    adam_t: u32,
}

impl OnlineDQN {
    pub fn new(input_size: usize, config: OnlineDQNConfig, rng: &mut ChaCha8Rng) -> Self {
        let policy_net = DuelingNetwork::new(input_size, rng);
        let mut target_net = DuelingNetwork::new(input_size, rng);
        target_net.copy_from(&policy_net);

        let buffer = ReplayBuffer::new(config.buffer_capacity);

        Self {
            policy_net,
            target_net,
            buffer,
            config,
            steps: 0,
            updates: 0,
            last_loss: 0.0,
            adam_t: 0,
        }
    }

    /// Select an action using epsilon-greedy policy.
    pub fn select_action(&self, state: &[f32], rng: &mut impl Rng) -> u8 {
        if rng.gen::<f32>() < self.config.epsilon {
            rng.gen_range(0..6) as u8
        } else {
            let q = self.policy_net.forward(state);
            q.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u8
        }
    }

    /// Store a transition in the replay buffer.
    pub fn store(
        &mut self,
        state: Vec<f32>,
        action: u8,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        self.buffer.push(state, action, reward, next_state, done);
        self.steps += 1;
    }

    /// Train one batch if conditions are met. Returns true if training occurred.
    pub fn maybe_train(&mut self, rng: &mut impl Rng) -> bool {
        if self.buffer.len() < self.config.min_buffer {
            return false;
        }
        if self.steps % self.config.train_every as u64 != 0 {
            return false;
        }

        let (states, actions, rewards, next_states, dones) =
            self.buffer.sample(self.config.batch_size, rng);

        // Double DQN targets: policy net picks action, target net evaluates
        let mut targets = Vec::with_capacity(self.config.batch_size);
        for i in 0..self.config.batch_size {
            if dones[i] {
                targets.push(rewards[i]);
            } else {
                let policy_q = self.policy_net.forward(&next_states[i]);
                let best_action = policy_q
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let target_q = self.target_net.forward(&next_states[i]);
                targets.push(
                    rewards[i] + self.config.gamma * target_q[best_action],
                );
            }
        }

        // Train policy network
        self.policy_net.zero_grad();
        self.last_loss =
            self.policy_net
                .forward_batch_loss(&states, &actions, &targets);
        self.adam_t += 1;
        self.policy_net.update(self.config.lr, self.adam_t);
        self.updates += 1;

        // Soft update target network periodically
        if self.steps % self.config.target_update_every as u64 == 0 {
            self.target_net
                .soft_update(&self.policy_net, self.config.tau);
        }

        true
    }

    /// Load pretrained weights from model_weights.json format.
    /// Initializes both policy and target networks.
    pub fn load_weights_json(&mut self, json: &str) -> bool {
        let ok = self.policy_net.load_weights_json(json);
        if ok {
            self.target_net.copy_from(&self.policy_net);
        }
        ok
    }

    /// Export current policy network weights as JSON.
    pub fn export_weights_json(&self) -> String {
        self.policy_net.export_weights_json()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy weights from src to dst layer.
fn copy_layer(dst: &mut Linear, src: &Linear) {
    for (d_row, s_row) in dst.weight.iter_mut().zip(src.weight.iter()) {
        d_row.copy_from_slice(s_row);
    }
    dst.bias.copy_from_slice(&src.bias);
}

/// Polyak averaging: dst = tau * src + (1 - tau) * dst.
fn soft_update_layer(dst: &mut Linear, src: &Linear, tau: f32) {
    for (d_row, s_row) in dst.weight.iter_mut().zip(src.weight.iter()) {
        for (d, s) in d_row.iter_mut().zip(s_row.iter()) {
            *d = tau * s + (1.0 - tau) * *d;
        }
    }
    for (d, s) in dst.bias.iter_mut().zip(src.bias.iter()) {
        *d = tau * s + (1.0 - tau) * *d;
    }
}

/// Load a 2D weight matrix [out_features][in_features] and bias into a Linear layer.
fn load_layer_2d(layer: &mut Linear, weight: &[Vec<f32>], bias: &[f32]) {
    for (i, row) in weight.iter().enumerate() {
        layer.weight[i].copy_from_slice(row);
    }
    layer.bias.copy_from_slice(bias);
}
