use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::math;
use crate::network::ActorCriticNet;
use crate::rollout::RolloutBuffer;

/// PPO hyperparameters.
#[derive(Clone, Serialize, Deserialize)]
pub struct PPOConfig {
    pub clip_epsilon: f32,
    pub gamma: f32,
    pub gae_lambda: f32,
    pub entropy_coeff: f32,
    pub value_coeff: f32,
    #[serde(alias = "learning_rate")]
    pub lr: f32,
    pub n_epochs: u32,
    #[serde(alias = "batch_size")]
    pub minibatch_size: usize,
    pub rollout_length: usize,
    pub n_frames: usize,
    pub max_grad_norm: f32,
    pub seed: u64,
    pub god_mode: bool,
    // LR scheduling
    #[serde(default = "default_lr_warmup_updates")]
    pub lr_warmup_updates: u32,
    #[serde(default = "default_lr_decay_updates")]
    pub lr_decay_updates: u32,
    #[serde(default = "default_lr_min")]
    pub lr_min: f32,
    // Curriculum
    #[serde(default = "default_curriculum_enabled")]
    pub curriculum_enabled: bool,
    #[serde(default = "default_curriculum_start_level")]
    pub curriculum_start_level: i32,
    #[serde(default = "default_curriculum_advance_threshold")]
    pub curriculum_advance_threshold: f32,
    #[serde(default = "default_curriculum_window")]
    pub curriculum_window: u32,
    // Observation normalization
    #[serde(default = "default_obs_norm_enabled")]
    pub obs_norm_enabled: bool,
}

fn default_lr_warmup_updates() -> u32 { 10 }
fn default_lr_decay_updates() -> u32 { 500 }
fn default_lr_min() -> f32 { 1e-5 }
fn default_curriculum_enabled() -> bool { true }
fn default_curriculum_start_level() -> i32 { 1 }
fn default_curriculum_advance_threshold() -> f32 { 15.0 }
fn default_curriculum_window() -> u32 { 50 }
fn default_obs_norm_enabled() -> bool { true }

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            clip_epsilon: 0.2,
            gamma: 0.99,
            gae_lambda: 0.95,
            entropy_coeff: 0.02,
            value_coeff: 0.5,
            lr: 3e-4,
            n_epochs: 4,
            minibatch_size: 64,
            rollout_length: 2048,
            n_frames: 4,
            max_grad_norm: 1.0,
            seed: 42,
            god_mode: true,
            lr_warmup_updates: 10,
            lr_decay_updates: 500,
            lr_min: 1e-5,
            curriculum_enabled: true,
            curriculum_start_level: 1,
            curriculum_advance_threshold: 15.0,
            curriculum_window: 50,
            obs_norm_enabled: true,
        }
    }
}

/// Statistics returned after a PPO update.
#[derive(Clone, Serialize, Deserialize)]
pub struct UpdateStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub total_loss: f32,
    pub approx_kl: f32,
}

/// Running statistics for tracking agent performance.
#[derive(Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub total_steps: u64,
    pub episodes: u32,
    pub best_score: i32,
    pub best_level: i32,
    pub current_score: i32,
    pub current_level: i32,
    pub current_lives: i32,
    pub episode_reward: f32,
    pub avg_reward: f32,
    pub updates: u32,
    pub last_update: Option<UpdateStats>,
    pub learning_enabled: bool,
}

impl Default for AgentStats {
    fn default() -> Self {
        Self {
            total_steps: 0,
            episodes: 0,
            best_score: 0,
            best_level: 1,
            current_score: 0,
            current_level: 1,
            current_lives: 6,
            episode_reward: 0.0,
            avg_reward: 0.0,
            updates: 0,
            last_update: None,
            learning_enabled: true,
        }
    }
}

/// Running statistics using Welford's online algorithm for observation normalization.
pub struct RunningStats {
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub count: f32,
    size: usize,
}

impl RunningStats {
    pub fn new(size: usize) -> Self {
        Self {
            mean: vec![0.0; size],
            var: vec![1.0; size],
            count: 0.0,
            size,
        }
    }

    pub fn update(&mut self, x: &[f32]) {
        self.count += 1.0;
        for i in 0..self.size.min(x.len()) {
            let delta = x[i] - self.mean[i];
            self.mean[i] += delta / self.count;
            let delta2 = x[i] - self.mean[i];
            self.var[i] += delta * delta2;
        }
    }

    pub fn normalize(&self, x: &[f32]) -> Vec<f32> {
        let n = self.size.min(x.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let std = if self.count > 1.0 {
                (self.var[i] / self.count).sqrt().max(1e-8)
            } else {
                1.0
            };
            let val = (x[i] - self.mean[i]) / std;
            out.push(val.clamp(-10.0, 10.0));
        }
        out
    }
}

/// Reward normalizer — divides by running std without subtracting mean.
pub struct RewardNormalizer {
    mean: f64,
    var: f64,
    count: f64,
}

impl RewardNormalizer {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0.0,
        }
    }

    pub fn normalize(&mut self, reward: f32) -> f32 {
        let r = reward as f64;
        self.count += 1.0;
        let delta = r - self.mean;
        self.mean += delta / self.count;
        let delta2 = r - self.mean;
        self.var += delta * delta2;
        let std = if self.count > 1.0 {
            (self.var / self.count).sqrt().max(1e-8)
        } else {
            1.0
        };
        (r / std) as f32
    }
}

/// PPO Agent with actor-critic network, rollout buffer, and frame stacking.
pub struct PPOAgent {
    pub network: ActorCriticNet,
    pub rollout: RolloutBuffer,
    pub config: PPOConfig,
    pub stats: AgentStats,
    rng: ChaCha8Rng,
    // Frame stacking
    frame_buffer: Vec<Vec<f32>>,
    adam_step: u32,
    // Observation normalization
    obs_stats: RunningStats,
    // Reward normalization
    pub reward_norm: RewardNormalizer,
    // Curriculum learning
    pub curriculum_level: i32,
    pub level_rewards: Vec<f32>,
}

impl PPOAgent {
    pub fn new(config: PPOConfig) -> Self {
        let network = ActorCriticNet::new_random(config.seed);
        let rollout = RolloutBuffer::new(config.rollout_length);
        let rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(1));
        let n_frames = config.n_frames;
        let obs_size = n_frames * 50;
        let curriculum_level = config.curriculum_start_level;

        Self {
            network,
            rollout,
            config,
            stats: AgentStats::default(),
            rng,
            frame_buffer: vec![vec![0.0f32; 50]; n_frames],
            adam_step: 0,
            obs_stats: RunningStats::new(obs_size),
            reward_norm: RewardNormalizer::new(),
            curriculum_level,
            level_rewards: Vec::new(),
        }
    }

    /// Push a new frame into the frame buffer (FIFO).
    pub fn push_frame(&mut self, state: &[f32]) {
        self.frame_buffer.remove(0);
        self.frame_buffer.push(state.to_vec());
    }

    /// Reset the frame buffer (fill with the given state).
    pub fn reset_frames(&mut self, state: &[f32]) {
        for frame in self.frame_buffer.iter_mut() {
            frame.copy_from_slice(state);
        }
    }

    /// Get the stacked state (n_frames * 50 = 200 features), optionally normalized.
    pub fn stacked_state(&mut self) -> Vec<f32> {
        let mut s = Vec::with_capacity(self.config.n_frames * 50);
        for frame in &self.frame_buffer {
            s.extend_from_slice(frame);
        }
        if self.config.obs_norm_enabled {
            self.obs_stats.update(&s);
            self.obs_stats.normalize(&s)
        } else {
            s
        }
    }

    /// Select an action by sampling from the policy distribution.
    /// Returns (action, log_prob).
    pub fn select_action(&mut self, stacked_state: &[f32]) -> (u8, f32, f32) {
        let (probs, value) = self.network.forward(stacked_state);

        if !self.stats.learning_enabled {
            // Greedy in eval mode
            let action = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0) as u8;
            let log_prob = (probs[action as usize] + 1e-8).ln();
            return (action, log_prob, value);
        }

        // Sample from categorical distribution
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0f32;
        let mut action = 0u8;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                action = i as u8;
                break;
            }
            if i == probs.len() - 1 {
                action = i as u8;
            }
        }

        let log_prob = (probs[action as usize] + 1e-8).ln();
        (action, log_prob, value)
    }

    /// Store a transition in the rollout buffer.
    pub fn store_transition(&mut self, state: Vec<f32>, action: u8, log_prob: f32, reward: f32, value: f32, done: bool) {
        self.rollout.push(state, action, log_prob, reward, value, done);
    }

    /// Compute the learning rate with warmup and cosine decay schedule.
    pub fn scheduled_lr(&self) -> f32 {
        let t = self.adam_step;
        let warmup = self.config.lr_warmup_updates;
        let decay_end = self.config.lr_decay_updates;
        let lr_max = self.config.lr;
        let lr_min = self.config.lr_min;
        if t < warmup {
            let progress = (t + 1) as f32 / warmup as f32;
            lr_min + progress * (lr_max - lr_min)
        } else if t < decay_end {
            let progress = (t - warmup) as f32 / (decay_end - warmup) as f32;
            let cosine = (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0;
            lr_min + cosine * (lr_max - lr_min)
        } else {
            lr_min
        }
    }

    /// Check if the curriculum level should advance based on recent episode rewards.
    /// Returns Some(new_level) if advanced, None otherwise.
    pub fn check_curriculum_advance(&mut self, episode_reward: f32) -> Option<i32> {
        if !self.config.curriculum_enabled {
            return None;
        }
        self.level_rewards.push(episode_reward);
        let window = self.config.curriculum_window as usize;
        if self.level_rewards.len() > window {
            self.level_rewards.drain(..self.level_rewards.len() - window);
        }
        if self.level_rewards.len() >= window {
            let avg = self.level_rewards.iter().sum::<f32>() / self.level_rewards.len() as f32;
            if avg >= self.config.curriculum_advance_threshold {
                self.curriculum_level += 1;
                self.level_rewards.clear();
                return Some(self.curriculum_level);
            }
        }
        None
    }

    /// Run PPO update when the rollout buffer is full. Returns update stats.
    pub fn update(&mut self, last_value: f32) -> UpdateStats {
        self.adam_step += 1;
        let (advantages, returns) = self.rollout.compute_advantages(
            self.config.gamma, self.config.gae_lambda, last_value,
        );

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy = 0.0f32;
        let mut total_approx_kl = 0.0f32;
        let mut n_updates = 0u32;

        for _epoch in 0..self.config.n_epochs {
            let minibatches = self.rollout.get_minibatches(self.config.minibatch_size, &mut self.rng);

            for mb in &minibatches {
                self.network.zero_grad();

                let mut batch_policy_loss = 0.0f32;
                let mut batch_value_loss = 0.0f32;
                let mut batch_entropy = 0.0f32;
                let mut batch_kl = 0.0f32;
                let batch_size = mb.indices.len() as f32;

                for &idx in &mb.indices {
                    let state = self.rollout.state(idx);
                    let old_action = self.rollout.action(idx);
                    let old_log_prob = self.rollout.log_prob(idx);
                    let advantage = advantages[idx];
                    let ret = returns[idx];

                    // Forward pass with activations stored
                    let fwd = self.network.forward_with_grads(state);

                    // Compute log probs and entropy
                    let log_probs = math::log_softmax(&fwd.policy_logits);
                    let new_log_prob = log_probs[old_action as usize];
                    let ratio = (new_log_prob - old_log_prob).exp();

                    // Clipped surrogate objective (we want to maximize, so negate for loss)
                    let surr1 = ratio * advantage;
                    let surr2 = ratio.clamp(
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * advantage;
                    let policy_loss = -surr1.min(surr2);

                    // Value loss (MSE)
                    let value_loss = (fwd.value - ret).powi(2);

                    // Entropy bonus (negative for loss since we want to maximize entropy)
                    let entropy: f32 = -fwd.policy_probs.iter()
                        .zip(log_probs.iter())
                        .map(|(&p, &lp)| p * lp)
                        .sum::<f32>();

                    // Approximate KL divergence for monitoring
                    let approx_kl = old_log_prob - new_log_prob;

                    batch_policy_loss += policy_loss;
                    batch_value_loss += value_loss;
                    batch_entropy += entropy;
                    batch_kl += approx_kl;

                    // Compute gradients for this sample
                    // dL/d_logits for policy loss + entropy
                    let mut policy_grad = vec![0.0f32; 6];
                    for a in 0..6 {
                        // Gradient of -min(surr1, surr2) w.r.t. logits
                        // Through log_softmax: d_log_prob[a]/d_logit[j] = 1{a==j} - prob[j]
                        let d_log_prob_d_logit = |j: usize| -> f32 {
                            if j == old_action as usize { 1.0 - fwd.policy_probs[j] }
                            else { -fwd.policy_probs[j] }
                        };

                        // d_policy_loss / d_logit[a]
                        // = d(-min(surr1, surr2)) / d_new_log_prob * d_new_log_prob / d_logit[a]
                        let use_clipped = surr2 < surr1;
                        let d_surr = if use_clipped {
                            // d(-surr2)/d_log_prob = -clamp(ratio)*advantage * ratio
                            // Actually: d(ratio)/d(log_prob) = ratio
                            // d(-clamp(ratio)*adv)/d(log_prob) = if ratio in [1-eps, 1+eps] then -ratio*adv else 0
                            if ratio >= 1.0 - self.config.clip_epsilon && ratio <= 1.0 + self.config.clip_epsilon {
                                -ratio * advantage
                            } else {
                                0.0
                            }
                        } else {
                            // d(-surr1)/d_log_prob = -ratio * advantage
                            -ratio * advantage
                        };

                        let d_from_policy = d_surr * d_log_prob_d_logit(a);

                        // Entropy gradient: d(-entropy_coeff * entropy)/d_logit[a]
                        // entropy = -sum_j p_j * log_p_j
                        // d_entropy/d_logit[a] = -sum_j (d_p_j/d_logit[a] * log_p_j + p_j * d_log_p_j/d_logit[a])
                        // = -sum_j (p_j*(1{a==j} - p_a) * log_p_j + p_j * (1{a==j} - p_a))
                        // = -sum_j p_j*(1{a==j} - p_a)*(log_p_j + 1)
                        // Simplified: d_entropy/d_logit[a] = p_a * (log_p_a + 1) - p_a * sum_j p_j*(log_p_j+1)
                        // Which equals: p_a * ((log_p_a + 1) - sum_j p_j*(log_p_j + 1))
                        // We want to maximize entropy, so gradient of -entropy_coeff*(-entropy) = entropy_coeff*entropy
                        // Loss includes -entropy_coeff * entropy, so grad is -entropy_coeff * d_entropy/d_logit
                        let h_bar: f32 = fwd.policy_probs.iter()
                            .zip(log_probs.iter())
                            .map(|(&p, &lp)| p * (lp + 1.0))
                            .sum();
                        let d_entropy = fwd.policy_probs[a] * ((log_probs[a] + 1.0) - h_bar);
                        let d_from_entropy = -self.config.entropy_coeff * d_entropy;

                        policy_grad[a] = (d_from_policy + d_from_entropy) / batch_size;
                    }

                    // Value gradient: d(value_coeff * (value - return)^2) / d_value
                    let value_grad = self.config.value_coeff * 2.0 * (fwd.value - ret) / batch_size;

                    self.network.backward(&fwd, &policy_grad, value_grad);
                }

                // Apply gradient clipping and Adam update with scheduled LR
                self.network.update(self.scheduled_lr(), self.adam_step, self.config.max_grad_norm);

                total_policy_loss += batch_policy_loss / batch_size;
                total_value_loss += batch_value_loss / batch_size;
                total_entropy += batch_entropy / batch_size;
                total_approx_kl += batch_kl / batch_size;
                n_updates += 1;
            }
        }

        self.rollout.clear();

        let stats = UpdateStats {
            policy_loss: total_policy_loss / n_updates as f32,
            value_loss: total_value_loss / n_updates as f32,
            entropy: total_entropy / n_updates as f32,
            total_loss: (total_policy_loss + self.config.value_coeff * total_value_loss
                - self.config.entropy_coeff * total_entropy) / n_updates as f32,
            approx_kl: total_approx_kl / n_updates as f32,
        };

        self.stats.updates += 1;
        self.stats.last_update = Some(stats.clone());
        stats
    }

    /// Load weights into the network (PPO format).
    pub fn load_weights(&mut self, json: &str) -> bool {
        self.network.load_ppo_weights_json(json)
    }

    /// Load weights from DQN model_weights.json format (partial, shared layers only).
    pub fn load_dqn_weights(&mut self, json: &str) -> bool {
        self.network.load_weights_json(json)
    }

    /// Export weights as JSON.
    pub fn export_weights(&self) -> String {
        self.network.export_weights_json()
    }
}
