// Game constants (from game.js / train.py)

pub const GAME_WIDTH: f64 = 1024.0;
pub const GAME_HEIGHT: f64 = 576.0;

pub const PLAYER_SPEED: f64 = 300.0;
pub const PLAYER_WIDTH: f64 = 48.0;
pub const PLAYER_HEIGHT: f64 = 48.0;
pub const PLAYER_LIVES: i32 = 6;
pub const FIRE_RATE: f64 = 0.16; // seconds
pub const BULLET_SPEED: f64 = 300.0;
pub const BULLET_W: f64 = 5.0;   // matches JS hardcoded hitbox (was 3.4)
pub const BULLET_H: f64 = 10.0;  // matches JS hardcoded hitbox (was 5.9)

pub const ENEMY_SPEED: f64 = 50.0;
pub const ENEMY_WIDTH: f64 = 43.0;
pub const ENEMY_HEIGHT: f64 = 43.0;
pub const ENEMY_PADDING: f64 = 16.0;
pub const ENEMY_HITS_TO_DESTROY: i32 = 2;
pub const ENEMY_ROWS: i32 = 5;
pub const ENEMY_BULLET_SPEED: f64 = BULLET_SPEED / 3.0;

pub const BASE_ENEMY_FIRE_RATE: f64 = 0.85;
pub const ENEMY_FIRE_RATE_INCREASE: f64 = 0.10;

pub const MISSILE_SPEED: f64 = 170.0;
pub const MISSILE_WIDTH: f64 = 57.0;
pub const MISSILE_HEIGHT: f64 = 57.0;
pub const MIN_MISSILE_INTERVAL: f64 = 3200.0;
pub const MAX_MISSILE_INTERVAL: f64 = 7200.0;

pub const KAMIKAZE_SPEED: f64 = 170.0;
pub const KAMIKAZE_FIRE_RATE: f64 = 900.0;
pub const KAMIKAZE_MIN_TIME: f64 = 6000.0;
pub const KAMIKAZE_MAX_TIME: f64 = 11000.0;
pub const KAMIKAZE_AGGRESSIVE_TIME: f64 = 4000.0;
pub const KAMIKAZE_VERY_AGGRESSIVE_TIME: f64 = 2200.0;
pub const KAMIKAZE_AGGRESSIVE_THRESHOLD: usize = 26;
pub const KAMIKAZE_VERY_AGGRESSIVE_THRESHOLD: usize = 11;
pub const KAMIKAZE_HITS_TO_DESTROY: i32 = 2;

pub const MONSTER_SPEED: f64 = 175.0;
pub const MONSTER_WIDTH: f64 = 56.0;
pub const MONSTER_HEIGHT: f64 = 56.0;
pub const MONSTER_INTERVAL: f64 = 6000.0;
pub const MONSTER_HIT_DURATION: f64 = 700.0;
pub const MONSTER_SLALOM_AMPLITUDE: f64 = 350.0;
pub const MONSTER_VERTICAL_SPEED: f64 = 60.0;
pub const MONSTER_SLALOM_FIRE_RATE: f64 = 1800.0;
pub const MONSTER_SLALOM_THRESHOLD: usize = 19;

pub const MONSTER2_WIDTH: f64 = 56.0;
pub const MONSTER2_HEIGHT: f64 = 56.0;
pub const MONSTER2_SPEED: f64 = 220.0;
pub const MONSTER2_INTERVAL: f64 = 10000.0;
pub const MONSTER2_VERTICAL_SPEED: f64 = 40.0;
pub const MONSTER2_SPIRAL_RADIUS: f64 = 100.0;
pub const MONSTER2_SPIRAL_SPEED: f64 = 3.0;

pub const WALL_HITS_FROM_BELOW: i32 = 3;
pub const WALL_MAX_HITS_TOTAL: i32 = 11;
pub const WALL_MAX_MISSILE_HITS: i32 = 4;

pub const PLAYER_HIT_ANIMATION_DURATION: f64 = 750.0;

pub const BONUS2LIVES: i32 = 5;

pub const WALL_Y: f64 = GAME_HEIGHT - 75.0;
pub const WALL_WIDTH: f64 = 58.0;
pub const WALL_HEIGHT: f64 = 23.0;

pub fn initial_wall_xs() -> [f64; 4] {
    [
        GAME_WIDTH * 1.0 / 5.0 - 29.0,
        GAME_WIDTH * 2.0 / 5.0 - 29.0,
        GAME_WIDTH * 3.0 / 5.0 - 29.0,
        GAME_WIDTH * 4.0 / 5.0 - 29.0,
    ]
}

pub const STATE_SIZE: usize = 62;
pub const ACTION_SIZE: usize = 6;
