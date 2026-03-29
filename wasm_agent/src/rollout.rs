/// A single transition stored in the rollout buffer.
struct Transition {
    state: Vec<f32>,
    action: u8,
    log_prob: f32,
    reward: f32,
    value: f32,
    done: bool,
}

/// A minibatch of indices into the rollout buffer, with pre-computed advantages and returns.
pub struct Minibatch {
    pub indices: Vec<usize>,
}

/// PPO rollout buffer that collects trajectories for on-policy learning.
pub struct RolloutBuffer {
    transitions: Vec<Transition>,
    pub capacity: usize,
}

impl RolloutBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_full(&self) -> bool {
        self.transitions.len() >= self.capacity
    }

    /// Store a single transition.
    pub fn push(&mut self, state: Vec<f32>, action: u8, log_prob: f32, reward: f32, value: f32, done: bool) {
        self.transitions.push(Transition {
            state, action, log_prob, reward, value, done,
        });
    }

    /// Compute GAE advantages and discounted returns.
    /// Returns (advantages, returns) each of length len().
    pub fn compute_advantages(&self, gamma: f32, gae_lambda: f32, last_value: f32) -> (Vec<f32>, Vec<f32>) {
        let n = self.transitions.len();
        let mut advantages = vec![0.0f32; n];
        let mut returns = vec![0.0f32; n];

        let mut gae = 0.0f32;
        for t in (0..n).rev() {
            let next_value = if t + 1 < n {
                self.transitions[t + 1].value
            } else {
                last_value
            };
            let next_non_terminal = if self.transitions[t].done { 0.0 } else { 1.0 };
            let delta = self.transitions[t].reward
                + gamma * next_value * next_non_terminal
                - self.transitions[t].value;
            gae = delta + gamma * gae_lambda * next_non_terminal * gae;
            advantages[t] = gae;
            returns[t] = gae + self.transitions[t].value;
        }

        // Normalize advantages
        let mean = advantages.iter().sum::<f32>() / n as f32;
        let var = advantages.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / n as f32;
        let std = (var + 1e-8).sqrt();
        for a in advantages.iter_mut() {
            *a = (*a - mean) / std;
        }

        (advantages, returns)
    }

    /// Generate shuffled minibatches of the given size.
    pub fn get_minibatches(&self, batch_size: usize, rng: &mut impl rand::Rng) -> Vec<Minibatch> {
        let n = self.transitions.len();
        let mut indices: Vec<usize> = (0..n).collect();

        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        indices
            .chunks(batch_size)
            .map(|chunk| Minibatch {
                indices: chunk.to_vec(),
            })
            .collect()
    }

    /// Access a transition's state by index.
    pub fn state(&self, idx: usize) -> &[f32] {
        &self.transitions[idx].state
    }

    /// Access a transition's action by index.
    pub fn action(&self, idx: usize) -> u8 {
        self.transitions[idx].action
    }

    /// Access a transition's old log_prob by index.
    pub fn log_prob(&self, idx: usize) -> f32 {
        self.transitions[idx].log_prob
    }

    /// Clear all stored transitions.
    pub fn clear(&mut self) {
        self.transitions.clear();
    }
}
