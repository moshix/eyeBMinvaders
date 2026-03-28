mod constants;
mod entities;
mod game;
mod movement;
mod collision;
mod spawning;
mod state;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2};
use rayon::prelude::*;
use game::HeadlessGame;

#[pyclass]
struct Game {
    inner: HeadlessGame,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        Self { inner: HeadlessGame::new(seed.unwrap_or(42)) }
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let state = self.inner.reset();
        PyArray1::from_slice_bound(py, &state)
    }

    fn step<'py>(&mut self, py: Python<'py>, action: u8) -> PyResult<(Bound<'py, PyArray1<f32>>, f32, bool, PyObject)> {
        let result = self.inner.step(action);
        let state = PyArray1::from_slice_bound(py, &result.state);
        let info = make_info(py, &result)?;
        Ok((state, result.reward, result.done, info))
    }

    #[getter]
    fn state_size(&self) -> usize { constants::STATE_SIZE }

    #[getter]
    fn action_size(&self) -> usize { constants::ACTION_SIZE }

    #[getter]
    fn score(&self) -> i32 { self.inner.score }

    #[getter]
    fn current_level(&self) -> i32 { self.inner.current_level }

    #[getter]
    fn player_lives(&self) -> i32 { self.inner.player_lives }

    #[getter]
    fn enemies_killed(&self) -> u32 { self.inner.enemies_killed }

    #[getter]
    fn kamikazes_killed(&self) -> u32 { self.inner.kamikazes_killed }

    #[getter]
    fn missiles_shot(&self) -> u32 { self.inner.missiles_shot }

    #[getter]
    fn times_hit(&self) -> u32 { self.inner.times_hit }

    #[getter]
    fn total_steps(&self) -> u64 { self.inner.total_steps }

    fn get_entities<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let g = &self.inner;
        let dict = PyDict::new_bound(py);
        dict.set_item("player_x", g.player_x)?;
        dict.set_item("player_y", g.player_y)?;
        dict.set_item("is_hit", g.is_player_hit)?;
        dict.set_item("game_over", g.game_over)?;
        dict.set_item("score", g.score)?;
        dict.set_item("level", g.current_level)?;
        dict.set_item("lives", g.player_lives)?;

        let enemies: Vec<(f64, f64, i32)> = g.enemies.iter()
            .map(|e| (e.x, e.y, e.hits)).collect();
        dict.set_item("enemies", enemies)?;

        let bullets: Vec<(f64, f64, bool)> = g.bullets.iter()
            .map(|b| (b.x, b.y, b.is_enemy)).collect();
        dict.set_item("bullets", bullets)?;

        let kamikazes: Vec<(f64, f64)> = g.kamikazes.iter()
            .map(|k| (k.x, k.y)).collect();
        dict.set_item("kamikazes", kamikazes)?;

        let missiles: Vec<(f64, f64)> = g.missiles.iter()
            .map(|m| (m.x, m.y)).collect();
        dict.set_item("missiles", missiles)?;

        match &g.monster {
            Some(m) if !m.hit => dict.set_item("monster", (m.x, m.y))?,
            _ => dict.set_item("monster", py.None())?,
        }
        match &g.monster2 {
            Some(m) if !m.is_disappeared && !m.hit => dict.set_item("monster2", (m.x, m.y))?,
            _ => dict.set_item("monster2", py.None())?,
        }

        let walls: Vec<(f64, i32, i32)> = g.walls.iter()
            .map(|w| (w.x, w.hit_count, w.missile_hits)).collect();
        dict.set_item("walls", walls)?;

        Ok(dict.into())
    }
}

#[pyclass]
struct BatchedGames {
    games: Vec<HeadlessGame>,
}

#[pymethods]
impl BatchedGames {
    #[new]
    #[pyo3(signature = (num_envs, seed=42))]
    fn new(num_envs: usize, seed: u64) -> Self {
        let games = (0..num_envs)
            .map(|i| HeadlessGame::new(seed + i as u64))
            .collect();
        Self { games }
    }

    fn reset_all<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let states: Vec<[f32; constants::STATE_SIZE]> = py.allow_threads(|| {
            self.games.par_iter_mut()
                .map(|game| game.reset())
                .collect()
        });
        let data: Vec<Vec<f32>> = states.iter().map(|s| s.to_vec()).collect();
        PyArray2::from_vec2_bound(py, &data).unwrap()
    }

    fn reset_one<'py>(&mut self, py: Python<'py>, idx: usize) -> Bound<'py, PyArray1<f32>> {
        let state = self.games[idx].reset();
        PyArray1::from_slice_bound(py, &state)
    }

    /// Reset a specific env to start at a given level (curriculum learning).
    fn reset_one_at_level<'py>(&mut self, py: Python<'py>, idx: usize, level: i32) -> Bound<'py, PyArray1<f32>> {
        let state = self.games[idx].reset_at_level(level);
        PyArray1::from_slice_bound(py, &state)
    }

    /// Enable/disable god mode for all envs (hits penalized but no life loss)
    fn set_god_mode(&mut self, enabled: bool) {
        for game in &mut self.games {
            game.god_mode = enabled;
        }
    }

    /// Step all environments.
    /// Returns (states [num_envs, 45], rewards [num_envs], dones [num_envs], infos list[dict])
    fn step_all<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>, Vec<PyObject>)> {
        let n = self.games.len();

        // Parallel step with GIL release
        let results: Vec<game::StepResult> = py.allow_threads(|| {
            self.games.par_iter_mut()
                .enumerate()
                .map(|(i, game)| {
                    let action = actions.get(i).copied().unwrap_or(0);
                    game.step(action)
                })
                .collect()
        });

        // Sequential: pack into numpy arrays + info dicts (needs GIL)
        let mut states = vec![0.0f32; n * constants::STATE_SIZE];
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];
        let mut infos = Vec::with_capacity(n);

        for (i, result) in results.iter().enumerate() {
            states[i * constants::STATE_SIZE..(i + 1) * constants::STATE_SIZE]
                .copy_from_slice(&result.state);
            rewards[i] = result.reward;
            dones[i] = result.done;
            infos.push(make_info(py, result)?);
        }

        let states_arr = PyArray2::from_vec2_bound(
            py,
            &states.chunks(constants::STATE_SIZE).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).unwrap();
        let rewards_arr = PyArray1::from_slice_bound(py, &rewards);
        let dones_arr = PyArray1::from_slice_bound(py, &dones);

        Ok((states_arr, rewards_arr, dones_arr, infos))
    }

    /// Fast step: returns only (states, rewards, dones) — no Python dict overhead
    fn step_all_fast<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>)> {
        let n = self.games.len();

        // Parallel step with GIL release
        let results: Vec<game::StepResult> = py.allow_threads(|| {
            self.games.par_iter_mut()
                .enumerate()
                .map(|(i, game)| {
                    let action = actions.get(i).copied().unwrap_or(0);
                    game.step(action)
                })
                .collect()
        });

        // Sequential: pack into numpy arrays (needs GIL)
        let mut states = vec![0.0f32; n * constants::STATE_SIZE];
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];

        for (i, result) in results.iter().enumerate() {
            states[i * constants::STATE_SIZE..(i + 1) * constants::STATE_SIZE]
                .copy_from_slice(&result.state);
            rewards[i] = result.reward;
            dones[i] = result.done;
        }

        let states_arr = PyArray2::from_vec2_bound(
            py,
            &states.chunks(constants::STATE_SIZE).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).unwrap();
        let rewards_arr = PyArray1::from_slice_bound(py, &rewards);
        let dones_arr = PyArray1::from_slice_bound(py, &dones);

        Ok((states_arr, rewards_arr, dones_arr))
    }

    /// Get game stats for a specific environment (for logging)
    fn get_stats(&self, idx: usize) -> PyResult<(i32, i32, i32, u64, u32, u32, u32, u32)> {
        let g = &self.games[idx];
        Ok((g.score, g.current_level, g.player_lives, g.total_steps,
            g.enemies_killed, g.kamikazes_killed, g.missiles_shot, g.times_hit))
    }

    #[getter]
    fn state_size(&self) -> usize { constants::STATE_SIZE }

    #[getter]
    fn action_size(&self) -> usize { constants::ACTION_SIZE }

    #[getter]
    fn num_envs(&self) -> usize { self.games.len() }
}

fn make_info(py: Python<'_>, result: &game::StepResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("score", result.score)?;
    dict.set_item("level", result.level)?;
    dict.set_item("lives", result.lives)?;
    dict.set_item("enemies_left", result.enemies_left)?;
    dict.set_item("steps", result.steps)?;
    let events: Vec<&str> = result.events.iter()
        .map(|e| e.event_type.as_str())
        .collect();
    dict.set_item("events", events)?;
    Ok(dict.into())
}

#[pymodule]
fn game_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<BatchedGames>()?;
    Ok(())
}
