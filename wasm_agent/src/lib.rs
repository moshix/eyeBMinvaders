pub mod math;
pub mod network;
pub mod rollout;
pub mod agent;
pub mod bridge;
pub mod dqn_online;

use wasm_bindgen::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use game_sim_core::game::HeadlessGame;
use game_sim_core::state;
use game_sim_core::constants::STATE_SIZE;

use agent::{PPOAgent, PPOConfig};
use bridge::{game_state_to_js, render_state_to_js, stats_to_js};
use dqn_online::{OnlineDQN, OnlineDQNConfig};

/// WASM-exported PPO agent that wraps the game simulation and neural network.
#[wasm_bindgen]
pub struct WasmAgent {
    game: HeadlessGame,
    agent: PPOAgent,
    episode_reward: f32,
    reward_history: Vec<f32>,
}

#[wasm_bindgen]
impl WasmAgent {
    /// Create a new WasmAgent. Pass a JSON config string or empty/null for defaults.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> WasmAgent {
        let config: PPOConfig = if config_json.is_empty() {
            PPOConfig::default()
        } else {
            serde_json::from_str(config_json).unwrap_or_default()
        };

        let mut game = HeadlessGame::new(config.seed);
        game.god_mode = config.god_mode;
        let initial_state = state::get_state(&game);

        let mut agent = PPOAgent::new(config);
        agent.reset_frames(&initial_state);

        WasmAgent {
            game,
            agent,
            episode_reward: 0.0,
            reward_history: Vec::new(),
        }
    }

    /// Main loop step: get state, select action, step game, store transition, return render data.
    pub fn step(&mut self) -> JsValue {
        let stacked = self.agent.stacked_state();
        let (action, log_prob, value) = self.agent.select_action(&stacked);

        let result = self.game.step(action);

        self.episode_reward += result.reward;
        self.agent.stats.total_steps += 1;
        self.agent.stats.current_score = result.score;
        self.agent.stats.current_level = result.level;
        self.agent.stats.current_lives = result.lives;

        if result.score > self.agent.stats.best_score {
            self.agent.stats.best_score = result.score;
        }
        if result.level > self.agent.stats.best_level {
            self.agent.stats.best_level = result.level;
        }
        self.count_events();

        // Store transition for training
        if self.agent.stats.learning_enabled {
            self.agent.store_transition(
                stacked, action, log_prob, result.reward, value, result.done,
            );

            // Run PPO update when buffer is full
            if self.agent.rollout.is_full() {
                // Bootstrap value for the last state
                let next_stacked = self.agent.stacked_state();
                let (_, _, last_value) = self.agent.select_action(&next_stacked);
                self.agent.update(last_value);
            }
        }

        // Push the new frame
        self.agent.push_frame(&result.state);

        // Handle episode end
        if result.done {
            self.handle_episode_end();
        }

        self.agent.stats.episode_reward = self.episode_reward;
        game_state_to_js(&self.game, action, &self.agent.stats)
    }

    /// Turbo mode: run N steps without building render data. Returns stats.
    pub fn train_steps(&mut self, n: u32) -> JsValue {
        for _ in 0..n {
            let stacked = self.agent.stacked_state();
            let (action, log_prob, value) = self.agent.select_action(&stacked);

            let result = self.game.step(action);

            self.episode_reward += result.reward;
            self.agent.stats.total_steps += 1;
            self.agent.stats.current_score = result.score;
            self.agent.stats.current_level = result.level;
            self.agent.stats.current_lives = result.lives;

            if result.score > self.agent.stats.best_score {
                self.agent.stats.best_score = result.score;
            }
            if result.level > self.agent.stats.best_level {
                self.agent.stats.best_level = result.level;
            }
            self.count_events();

            if self.agent.stats.learning_enabled {
                self.agent.store_transition(
                    stacked, action, log_prob, result.reward, value, result.done,
                );

                if self.agent.rollout.is_full() {
                    let next_stacked = self.agent.stacked_state();
                    let (_, _, last_value) = self.agent.select_action(&next_stacked);
                    self.agent.update(last_value);
                }
            }

            self.agent.push_frame(&result.state);

            if result.done {
                self.handle_episode_end();
            }
        }

        self.agent.stats.episode_reward = self.episode_reward;
        stats_to_js(&self.agent.stats)
    }

    /// Get current agent stats.
    pub fn get_stats(&self) -> JsValue {
        stats_to_js(&self.agent.stats)
    }

    /// Enable or disable learning (training vs. inference-only mode).
    pub fn set_learning(&mut self, enabled: bool) {
        self.agent.stats.learning_enabled = enabled;
    }

    /// Reset game and agent state for a new episode.
    pub fn reset(&mut self) {
        let initial_state = self.game.reset();
        self.agent.reset_frames(&initial_state);
        self.episode_reward = 0.0;
    }

    /// Load PPO weights from JSON string.
    pub fn load_weights(&mut self, json: &str) -> bool {
        self.agent.load_weights(json)
    }

    /// Load DQN weights (partial, shared layers only) from the model_weights.json format.
    pub fn load_dqn_weights(&mut self, json: &str) -> bool {
        self.agent.load_dqn_weights(json)
    }

    /// Export current PPO weights as JSON.
    pub fn export_weights(&self) -> String {
        self.agent.export_weights()
    }
}

// ---------------------------------------------------------------------------
// WasmGame — standalone game engine for browser physics replacement
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct WasmGame {
    game: HeadlessGame,
}

#[wasm_bindgen]
impl WasmGame {
    /// Create a new WasmGame instance with a default seed.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmGame {
        WasmGame {
            game: HeadlessGame::new(42),
        }
    }

    /// Tick the game with an action code (0-5). Returns full render state as a JS object.
    pub fn tick(&mut self, dt: f64, action: u8) -> JsValue {
        let state = self.game.tick(dt, action);
        render_state_to_js(&state)
    }

    /// Tick with raw keyboard input booleans. Returns full render state as a JS object.
    pub fn tick_input(&mut self, _dt: f64, left: bool, right: bool, fire: bool) -> JsValue {
        let state = self.game.step_with_input(left, right, fire);
        render_state_to_js(&state)
    }

    /// Get the 50-feature state vector for AI inference.
    pub fn get_state(&self) -> Vec<f32> {
        state::get_state(&self.game).to_vec()
    }

    /// Reset for a new game. Returns the initial state vector.
    pub fn reset(&mut self) {
        self.game.reset();
    }

    /// Reset to a specific level (for curriculum training).
    pub fn reset_at_level(&mut self, level: i32) {
        self.game.reset_at_level(level);
    }
}

// ---------------------------------------------------------------------------
// WasmOnlineDQN — online DQN learner that fine-tunes from its own gameplay
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct WasmOnlineDQN {
    game: HeadlessGame,
    dqn: OnlineDQN,
    frame_buffer: Vec<Vec<f32>>,
    n_frames: usize,
    state_size: usize,
    rng: ChaCha8Rng,
    episode_count: u32,
    episode_reward: f32,
    best_score: i32,
    avg_score: f32,
    score_history: Vec<i32>,
}

#[wasm_bindgen]
impl WasmOnlineDQN {
    /// Create a new online DQN learner with default config.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let state_size = STATE_SIZE;
        let n_frames = 4;
        let input_size = state_size * n_frames;
        let config = OnlineDQNConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let dqn = OnlineDQN::new(input_size, config, &mut rng);
        let game = HeadlessGame::new(42);
        let initial_state = state::get_state(&game).to_vec();

        let frame_buffer = vec![initial_state; n_frames];

        Self {
            game,
            dqn,
            frame_buffer,
            n_frames,
            state_size,
            rng,
            episode_count: 0,
            episode_reward: 0.0,
            best_score: 0,
            avg_score: 0.0,
            score_history: Vec::new(),
        }
    }

    /// Load pretrained weights from JSON (model_weights.json format).
    /// Initializes both policy and target networks.
    pub fn load_weights(&mut self, json: &str) -> bool {
        self.dqn.load_weights_json(json)
    }

    /// Run one game step: observe, act, store transition, maybe train.
    pub fn step(&mut self) -> JsValue {
        let stacked = self.get_stacked_state();
        let action = self.dqn.select_action(&stacked, &mut self.rng);

        let result = self.game.step(action);
        self.episode_reward += result.reward;

        // Push new frame into buffer
        let new_state = state::get_state(&self.game).to_vec();
        self.push_frame(&new_state);
        let next_stacked = self.get_stacked_state();

        // Store transition and maybe train
        self.dqn
            .store(stacked, action, result.reward, next_stacked, result.done);
        self.dqn.maybe_train(&mut self.rng);

        // Handle episode end
        if result.done {
            self.episode_count += 1;
            self.score_history.push(result.score);
            if self.score_history.len() > 100 {
                self.score_history.remove(0);
            }
            self.avg_score = self.score_history.iter().sum::<i32>() as f32
                / self.score_history.len() as f32;
            if result.score > self.best_score {
                self.best_score = result.score;
            }
            self.episode_reward = 0.0;
            let s = self.game.reset();
            self.reset_frames(&s.to_vec());
        }

        self.build_stats_js()
    }

    /// Run N steps in turbo mode (no per-step JS object overhead).
    pub fn train_steps(&mut self, n: u32) -> JsValue {
        for _ in 0..n {
            let stacked = self.get_stacked_state();
            let action = self.dqn.select_action(&stacked, &mut self.rng);

            let result = self.game.step(action);
            self.episode_reward += result.reward;

            let new_state = state::get_state(&self.game).to_vec();
            self.push_frame(&new_state);
            let next_stacked = self.get_stacked_state();

            self.dqn
                .store(stacked, action, result.reward, next_stacked, result.done);
            self.dqn.maybe_train(&mut self.rng);

            if result.done {
                self.episode_count += 1;
                self.score_history.push(result.score);
                if self.score_history.len() > 100 {
                    self.score_history.remove(0);
                }
                self.avg_score = self.score_history.iter().sum::<i32>() as f32
                    / self.score_history.len() as f32;
                if result.score > self.best_score {
                    self.best_score = result.score;
                }
                self.episode_reward = 0.0;
                let s = self.game.reset();
                self.reset_frames(&s.to_vec());
            }
        }
        self.build_stats_js()
    }

    /// Get current training statistics.
    pub fn get_stats(&self) -> JsValue {
        self.build_stats_js()
    }

    /// Export current policy network weights as JSON.
    pub fn export_weights(&self) -> String {
        self.dqn.export_weights_json()
    }

    /// Set the exploration epsilon.
    pub fn set_epsilon(&mut self, eps: f32) {
        self.dqn.config.epsilon = eps;
    }

    /// Set the learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.dqn.config.lr = lr;
    }
}

impl WasmOnlineDQN {
    fn push_frame(&mut self, state: &[f32]) {
        self.frame_buffer.remove(0);
        self.frame_buffer.push(state.to_vec());
    }

    fn reset_frames(&mut self, state: &[f32]) {
        for f in &mut self.frame_buffer {
            f.copy_from_slice(state);
        }
    }

    fn get_stacked_state(&self) -> Vec<f32> {
        let mut s = Vec::with_capacity(self.n_frames * self.state_size);
        for frame in &self.frame_buffer {
            s.extend_from_slice(frame);
        }
        s
    }

    fn build_stats_js(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let set_i32 = |k: &str, v: i32| {
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str(k),
                &JsValue::from_f64(v as f64),
            )
            .ok();
        };
        let set_f64 = |k: &str, v: f64| {
            js_sys::Reflect::set(
                &obj,
                &JsValue::from_str(k),
                &JsValue::from_f64(v),
            )
            .ok();
        };
        set_i32("episodes", self.episode_count as i32);
        set_f64("avgScore", self.avg_score as f64);
        set_i32("bestScore", self.best_score);
        set_i32("currentScore", self.game.score);
        set_i32("currentLevel", self.game.current_level);
        set_i32("currentLives", self.game.player_lives);
        set_f64("lastLoss", self.dqn.last_loss as f64);
        set_i32("updates", self.dqn.updates as i32);
        set_i32("bufferSize", self.dqn.buffer.len() as i32);
        set_i32("totalSteps", self.dqn.steps as i32);
        set_f64("epsilon", self.dqn.config.epsilon as f64);
        obj.into()
    }
}

impl WasmAgent {
    fn count_events(&mut self) {
        use game_sim_core::entities::EventType;
        for ev in &self.game.events {
            match ev.event_type {
                EventType::EnemyKilled => self.agent.stats.enemies_killed += 1,
                EventType::KamikazeKilled => self.agent.stats.kamikazes_killed += 1,
                EventType::MissileShotDown => self.agent.stats.missiles_shot += 1,
                EventType::MonsterKilled => self.agent.stats.monsters_killed += 1,
                EventType::Monster2Killed => self.agent.stats.monsters2_killed += 1,
                _ => {}
            }
        }
    }

    fn handle_episode_end(&mut self) {
        self.agent.stats.episodes += 1;
        self.reward_history.push(self.episode_reward);

        // Cap reward_history at 100 entries to prevent unbounded growth
        if self.reward_history.len() > 100 {
            self.reward_history.drain(..self.reward_history.len() - 100);
        }

        // Running average over last 100 episodes
        let window = &self.reward_history;
        self.agent.stats.avg_reward = window.iter().sum::<f32>() / window.len() as f32;

        // Check curriculum advancement
        if let Some(new_level) = self.agent.check_curriculum_advance(self.episode_reward) {
            // Reset at the new curriculum level
            self.episode_reward = 0.0;
            let initial_state = self.game.reset_at_level(new_level);
            self.agent.reset_frames(&initial_state);
            return;
        }

        // Reset for next episode using current curriculum level
        self.episode_reward = 0.0;
        let level = self.agent.curriculum_level;
        let initial_state = self.game.reset_at_level(level);
        self.agent.reset_frames(&initial_state);
    }
}
