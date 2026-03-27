use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::constants::*;
use crate::entities::*;
use crate::movement;
use crate::collision;
use crate::spawning;
use crate::state;

pub struct HeadlessGame {
    pub rng: ChaCha8Rng,
    pub game_time: f64,
    pub score: i32,
    pub current_level: i32,
    pub game_over: bool,

    // Player
    pub player_x: f64,
    pub player_y: f64,
    pub player_lives: i32,
    pub is_player_hit: bool,
    pub while_player_hit: bool,
    pub player_hit_timer: f64,

    // Enemies
    pub enemies: Vec<Enemy>,
    pub enemy_speed: f64,
    pub enemy_direction: i32,
    pub current_enemy_fire_rate: f64,
    pub last_enemy_fire_time: f64,

    // Bullets
    pub bullets: Vec<Bullet>,
    pub last_fire_time: f64,

    // Missiles
    pub missiles: Vec<Missile>,
    pub next_missile_time: f64,
    pub homing_missile_hits: i32,
    pub bonus_grants: i32,

    // Kamikazes
    pub kamikazes: Vec<Kamikaze>,
    pub next_kamikaze_time: f64,

    // Monsters
    pub monster: Option<Monster>,
    pub last_monster_time: f64,
    pub monster2: Option<Monster2>,
    pub last_monster2_time: f64,

    // Walls
    pub walls: Vec<Wall>,

    // Events
    pub events: Vec<GameEvent>,

    // Stats
    pub total_steps: u64,
    pub enemies_killed: u32,
    pub kamikazes_killed: u32,
    pub missiles_shot: u32,
    pub times_hit: u32,
    pub near_misses: i32,
}

pub struct StepResult {
    pub state: [f32; 45],
    pub reward: f32,
    pub done: bool,
    pub score: i32,
    pub level: i32,
    pub lives: i32,
    pub enemies_left: usize,
    pub steps: u64,
    pub events: Vec<GameEvent>,
    pub enemies_killed: u32,
    pub kamikazes_killed: u32,
    pub missiles_shot: u32,
    pub times_hit: u32,
}

impl HeadlessGame {
    pub fn new(seed: u64) -> Self {
        let mut game = Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            game_time: 0.0,
            score: 0,
            current_level: 1,
            game_over: false,
            player_x: GAME_WIDTH / 2.0 - 37.0,
            player_y: GAME_HEIGHT - 30.0,
            player_lives: PLAYER_LIVES,
            is_player_hit: false,
            while_player_hit: false,
            player_hit_timer: 0.0,
            enemies: Vec::new(),
            enemy_speed: 0.54,
            enemy_direction: 1,
            current_enemy_fire_rate: BASE_ENEMY_FIRE_RATE,
            last_enemy_fire_time: 0.0,
            bullets: Vec::new(),
            last_fire_time: 0.0,
            missiles: Vec::new(),
            next_missile_time: 0.0,
            homing_missile_hits: 0,
            bonus_grants: 0,
            kamikazes: Vec::new(),
            next_kamikaze_time: 0.0,
            monster: None,
            last_monster_time: 0.0,
            monster2: None,
            last_monster2_time: 0.0,
            walls: Vec::new(),
            events: Vec::new(),
            total_steps: 0,
            enemies_killed: 0,
            kamikazes_killed: 0,
            missiles_shot: 0,
            times_hit: 0,
            near_misses: 0,
        };
        game.next_kamikaze_time = game.random_kamikaze_time();
        game.restore_walls();
        game.create_enemies();
        game
    }

    pub fn reset(&mut self) -> [f32; 45] {
        self.game_time = 0.0;
        self.score = 0;
        self.current_level = 1;
        self.game_over = false;
        self.player_x = GAME_WIDTH / 2.0 - 37.0;
        self.player_y = GAME_HEIGHT - 30.0;
        self.player_lives = PLAYER_LIVES;
        self.is_player_hit = false;
        self.while_player_hit = false;
        self.player_hit_timer = 0.0;
        self.enemies.clear();
        self.enemy_speed = 0.54;
        self.enemy_direction = 1;
        self.current_enemy_fire_rate = BASE_ENEMY_FIRE_RATE;
        self.last_enemy_fire_time = 0.0;
        self.bullets.clear();
        self.last_fire_time = 0.0;
        self.missiles.clear();
        self.next_missile_time = 0.0;
        self.homing_missile_hits = 0;
        self.bonus_grants = 0;
        self.kamikazes.clear();
        self.next_kamikaze_time = self.random_kamikaze_time();
        self.monster = None;
        self.last_monster_time = 0.0;
        self.monster2 = None;
        self.last_monster2_time = 0.0;
        self.restore_walls();
        self.events.clear();
        self.total_steps = 0;
        self.enemies_killed = 0;
        self.kamikazes_killed = 0;
        self.missiles_shot = 0;
        self.times_hit = 0;
        self.near_misses = 0;
        self.create_enemies();
        state::get_state(self)
    }

    pub fn step(&mut self, action: u8) -> StepResult {
        self.events.clear();
        self.near_misses = 0;
        let dt_ms: f64 = 33.333;
        let dt = dt_ms / 1000.0;
        self.game_time += dt_ms;
        self.total_steps += 1;
        let old_score = self.score;
        let old_lives = self.player_lives;

        if self.game_over {
            let st = state::get_state(self);
            return StepResult {
                state: st, reward: 0.0, done: true,
                score: self.score, level: self.current_level,
                lives: self.player_lives, enemies_left: self.enemies.len(),
                steps: self.total_steps, events: self.events.clone(),
                enemies_killed: self.enemies_killed, kamikazes_killed: self.kamikazes_killed,
                missiles_shot: self.missiles_shot, times_hit: self.times_hit,
            };
        }

        // Parse action
        let move_left = action == 1 || action == 4;
        let move_right = action == 2 || action == 5;
        let shoot = action == 3 || action == 4 || action == 5;

        // Player hit animation
        if self.is_player_hit {
            self.while_player_hit = true;
            if self.game_time - self.player_hit_timer > PLAYER_HIT_ANIMATION_DURATION {
                self.is_player_hit = false;
                self.while_player_hit = false;
            }
        } else {
            self.while_player_hit = false;
        }

        // Player movement
        if !self.is_player_hit {
            if move_left && self.player_x > 0.0 {
                self.player_x -= PLAYER_SPEED * dt;
                if self.player_x < 0.0 { self.player_x = 0.0; }
            }
            if move_right && self.player_x < GAME_WIDTH - PLAYER_WIDTH {
                self.player_x += PLAYER_SPEED * dt;
                if self.player_x > GAME_WIDTH - PLAYER_WIDTH {
                    self.player_x = GAME_WIDTH - PLAYER_WIDTH;
                }
            }
        }

        // Player shooting
        if shoot && !self.while_player_hit {
            if self.game_time - self.last_fire_time >= FIRE_RATE * 1000.0 {
                let bx = self.player_x + PLAYER_WIDTH / 2.0 - BULLET_W / 2.0;
                let by = self.player_y;
                self.bullets.push(Bullet::new(bx, by, false));
                self.last_fire_time = self.game_time;
                self.emit(EventType::PlayerShot);
            }
        }

        // Spawning
        spawning::handle_monster_creation(self);
        spawning::handle_monster2_creation(self);
        spawning::handle_kamikaze_creation(self);

        // Movement
        movement::move_bullets(self, dt);
        movement::move_enemies(self, dt);
        movement::move_kamikazes(self, dt);
        movement::move_missiles(self, dt);
        movement::move_monster(self, dt);
        movement::move_monster2(self, dt);

        // Enemy actions
        spawning::handle_enemy_shooting(self);
        spawning::handle_missile_launching(self);

        // Collision detection
        collision::detect_collisions(self);

        // Victory check
        if self.enemies.is_empty() && !self.game_over {
            self.victory();
        }

        // Reward calculation — count events for shaping
        let wall_destroyed_count = self.events.iter()
            .filter(|e| matches!(e.event_type, EventType::WallDestroyed))
            .count() as i32;
        let kamikazes_killed_this_step = self.events.iter()
            .filter(|e| matches!(e.event_type, EventType::KamikazeKilled))
            .count() as i32;
        let missiles_shot_this_step = self.events.iter()
            .filter(|e| matches!(e.event_type, EventType::MissileShotDown))
            .count() as i32;
        let level_completed = self.events.iter()
            .any(|e| matches!(e.event_type, EventType::LevelComplete));
        let reward = state::calculate_reward(
            self, old_score, old_lives, wall_destroyed_count,
            kamikazes_killed_this_step, missiles_shot_this_step, self.near_misses,
            level_completed);

        let st = state::get_state(self);
        StepResult {
            state: st, reward, done: self.game_over,
            score: self.score, level: self.current_level,
            lives: self.player_lives, enemies_left: self.enemies.len(),
            steps: self.total_steps, events: self.events.clone(),
            enemies_killed: self.enemies_killed, kamikazes_killed: self.kamikazes_killed,
            missiles_shot: self.missiles_shot, times_hit: self.times_hit,
        }
    }

    pub fn random_kamikaze_time(&mut self) -> f64 {
        self.game_time + self.rng.gen_range(KAMIKAZE_MIN_TIME..KAMIKAZE_MAX_TIME)
    }

    pub fn create_enemies(&mut self) {
        let cols = ((GAME_WIDTH - 60.0) / (ENEMY_WIDTH + ENEMY_PADDING)) as i32;
        let cols = cols.max(4).min(12);
        let max_offset_top: f64 = 35.0;
        let wall_y = WALL_Y;
        let ideal_gap = GAME_HEIGHT * 0.3;
        let bottom_row_y = wall_y - ideal_gap;
        let total_height = ENEMY_ROWS as f64 * (ENEMY_HEIGHT + ENEMY_PADDING);
        let offset_top = max_offset_top.max(bottom_row_y - total_height);
        let offset_left = (GAME_WIDTH - cols as f64 * (ENEMY_WIDTH + ENEMY_PADDING)) / 2.0;

        for r in 0..ENEMY_ROWS {
            for c in 0..cols {
                self.enemies.push(Enemy::new(
                    c as f64 * (ENEMY_WIDTH + ENEMY_PADDING) + offset_left,
                    r as f64 * (ENEMY_HEIGHT + ENEMY_PADDING) + offset_top,
                ));
            }
        }
    }

    pub fn restore_walls(&mut self) {
        self.walls.clear();
        for wx in initial_wall_xs().iter() {
            self.walls.push(Wall::new(*wx));
        }
    }

    pub fn emit(&mut self, event_type: EventType) {
        self.events.push(GameEvent {
            event_type,
            time: self.game_time,
            score: self.score,
        });
    }

    pub fn handle_player_hit(&mut self) {
        self.player_lives -= 1;
        self.is_player_hit = true;
        self.player_hit_timer = self.game_time;
        self.times_hit += 1;
        self.emit(EventType::PlayerHit);

        // Clear threats
        for b in self.bullets.iter_mut() {
            if b.is_enemy { b.removed = true; }
        }
        for m in self.missiles.iter_mut() {
            m.removed = true;
        }
        for k in self.kamikazes.iter_mut() {
            k.removed = true;
        }

        if self.player_lives <= 0 {
            self.game_over = true;
            self.emit(EventType::GameOver);
        }
    }

    fn victory(&mut self) {
        self.current_level += 1;
        self.enemy_speed *= 1.33;
        self.current_enemy_fire_rate = BASE_ENEMY_FIRE_RATE
            / (1.0 + ENEMY_FIRE_RATE_INCREASE * (self.current_level - 1) as f64);
        self.score += 2500;
        self.emit(EventType::LevelComplete);

        // Clear everything
        self.enemies.clear();
        self.bullets.clear();
        self.missiles.clear();
        self.kamikazes.clear();
        self.monster = None;
        self.monster2 = None;
        self.restore_walls();
        self.next_kamikaze_time = self.random_kamikaze_time();
        self.create_enemies();
    }
}
