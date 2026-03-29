pub mod math;
pub mod network;
pub mod rollout;
pub mod agent;
pub mod bridge;

use wasm_bindgen::prelude::*;
use game_sim_core::game::HeadlessGame;
use game_sim_core::state;

use agent::{PPOAgent, PPOConfig};
use bridge::{game_state_to_js, stats_to_js};

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

impl WasmAgent {
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

        // Reset for next episode
        self.episode_reward = 0.0;
        let initial_state = self.game.reset();
        self.agent.reset_frames(&initial_state);
    }
}
