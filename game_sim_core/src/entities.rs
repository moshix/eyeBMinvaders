use crate::constants::*;

#[derive(Clone)]
pub struct Bullet {
    pub x: f64,
    pub y: f64,
    pub is_enemy: bool,
    pub dx: f64,
    pub dy: f64,
    pub has_direction: bool,
    pub removed: bool,
}

impl Bullet {
    pub fn new(x: f64, y: f64, is_enemy: bool) -> Self {
        Self { x, y, is_enemy, dx: 0.0, dy: 0.0, has_direction: false, removed: false }
    }

    pub fn with_direction(x: f64, y: f64, is_enemy: bool, dx: f64, dy: f64) -> Self {
        Self { x, y, is_enemy, dx, dy, has_direction: true, removed: false }
    }
}

#[derive(Clone)]
pub struct Enemy {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub hits: i32,
}

impl Enemy {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y, width: ENEMY_WIDTH, height: ENEMY_HEIGHT, hits: 0 }
    }
}

#[derive(Clone)]
pub struct Kamikaze {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub angle: f64,
    pub time: f64,
    pub hits: i32,
    pub last_fire_time: f64,
    pub removed: bool,
}

impl Kamikaze {
    pub fn new(x: f64, y: f64, width: f64, height: f64, last_fire_time: f64) -> Self {
        Self {
            x, y, width, height,
            angle: 0.0, time: 0.0, hits: 0,
            last_fire_time, removed: false,
        }
    }
}

#[derive(Clone)]
pub struct Missile {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
    pub width: f64,
    pub height: f64,
    pub time: f64,
    pub from_monster: bool,
    pub removed: bool,
}

impl Missile {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x, y, angle: 0.0,
            width: MISSILE_WIDTH, height: MISSILE_HEIGHT,
            time: 0.0, from_monster: false, removed: false,
        }
    }

    pub fn from_monster(x: f64, y: f64, angle: f64, width: f64, height: f64) -> Self {
        Self {
            x, y, angle, width, height,
            time: 0.0, from_monster: true, removed: false,
        }
    }
}

#[derive(Clone)]
pub struct Monster {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub hit: bool,
    pub hit_time: f64,
    pub has_shot: bool,
    pub slalom_time: f64,
    pub is_slaloming: bool,
    pub last_fire_time: f64,
    pub direction: i32,
}

impl Monster {
    pub fn new(x: f64, y: f64, is_slaloming: bool, last_fire_time: f64, direction: i32) -> Self {
        Self {
            x, y,
            width: MONSTER_WIDTH, height: MONSTER_HEIGHT,
            hit: false, hit_time: 0.0, has_shot: false,
            slalom_time: 0.0, is_slaloming, last_fire_time, direction,
        }
    }
}

#[derive(Clone)]
pub struct Monster2 {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub spiral_angle: f64,
    pub center_x: f64,
    pub hit: bool,
    pub hit_time: f64,
    pub is_disappeared: bool,
    pub disappear_time: f64,
    pub return_delay: f64,
    pub last_fire_time: f64,
    pub zigzag_dir: i32,
    pub zigzag_amplitude: f64,
    pub zigzag_phase: f64,
    pub dx_val: f64,
    pub dy_val: f64,
    pub last_direction_change: f64,
    pub direction_change_interval: f64,
    pub wave_start_x: f64,
    pub next_teleport_time: f64,
    pub target_x: f64,
    pub target_y: f64,
    pub next_move_time: f64,
}

impl Monster2 {
    pub fn new(x: f64, y: f64, last_fire_time: f64) -> Self {
        Self {
            x, y,
            width: MONSTER2_WIDTH, height: MONSTER2_HEIGHT,
            spiral_angle: 0.0, center_x: GAME_WIDTH / 2.0,
            hit: false, hit_time: 0.0,
            is_disappeared: false, disappear_time: 0.0, return_delay: 0.0,
            last_fire_time,
            zigzag_dir: 0, zigzag_amplitude: 0.0, zigzag_phase: 0.0,
            dx_val: 0.0, dy_val: 0.0,
            last_direction_change: 0.0, direction_change_interval: 0.0,
            wave_start_x: 0.0, next_teleport_time: 0.0,
            target_x: 0.0, target_y: 0.0, next_move_time: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct Wall {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub hit_count: i32,
    pub missile_hits: i32,
}

impl Wall {
    pub fn new(x: f64) -> Self {
        Self { x, y: WALL_Y, width: WALL_WIDTH, height: WALL_HEIGHT, hit_count: 0, missile_hits: 0 }
    }
}

#[derive(Clone, Debug)]
pub enum EventType {
    EnemyKilled,
    KamikazeKilled,
    MissileShotDown,
    MonsterKilled,
    Monster2Killed,
    PlayerHit,
    LevelComplete,
    GameOver,
    BonusEarned,
    LifeGranted,
    WallDestroyed,
    PlayerShot,
    KamikazeSpawned,
    MonsterSpawned,
    MissileLaunched,
}

impl EventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventType::EnemyKilled => "enemy_killed",
            EventType::KamikazeKilled => "kamikaze_killed",
            EventType::MissileShotDown => "missile_shot_down",
            EventType::MonsterKilled => "monster_killed",
            EventType::Monster2Killed => "monster2_killed",
            EventType::PlayerHit => "player_hit",
            EventType::LevelComplete => "level_complete",
            EventType::GameOver => "game_over",
            EventType::BonusEarned => "bonus_earned",
            EventType::LifeGranted => "life_granted",
            EventType::WallDestroyed => "wall_destroyed",
            EventType::PlayerShot => "player_shot",
            EventType::KamikazeSpawned => "kamikaze_spawned",
            EventType::MonsterSpawned => "monster_spawned",
            EventType::MissileLaunched => "missile_launched",
        }
    }
}

#[derive(Clone)]
pub struct GameEvent {
    pub event_type: EventType,
    pub time: f64,
    pub score: i32,
}

// ---------------------------------------------------------------------------
// Render-friendly entity snapshots for JS/WASM rendering
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct PlayerRender {
    pub x: f64, pub y: f64, pub width: f64, pub height: f64, pub is_hit: bool,
}

#[derive(Clone)]
pub struct EnemyRender {
    pub x: f64, pub y: f64, pub width: f64, pub height: f64, pub hits: i32, pub row: i32,
}

#[derive(Clone)]
pub struct BulletRender {
    pub x: f64, pub y: f64, pub is_enemy: bool, pub dx: f64, pub dy: f64, pub is_monster2: bool,
}

#[derive(Clone)]
pub struct KamikazeRender {
    pub x: f64, pub y: f64, pub width: f64, pub height: f64, pub angle: f64,
}

#[derive(Clone)]
pub struct MissileRender {
    pub x: f64, pub y: f64, pub width: f64, pub height: f64, pub angle: f64,
}

#[derive(Clone)]
pub struct WallRender {
    pub x: f64, pub y: f64, pub width: f64, pub height: f64,
    pub hit_count: i32, pub missile_hits: i32,
}

#[derive(Clone)]
pub struct MonsterRender {
    pub x: f64, pub y: f64, pub width: f64, pub height: f64,
    pub is_hit: bool, pub is_slaloming: bool,
}

#[derive(Clone)]
pub struct Monster2Render {
    pub x: f64, pub y: f64, pub width: f64, pub height: f64,
    pub dx: f64, pub dy: f64, pub is_disappeared: bool,
}

// ---------------------------------------------------------------------------
// Detailed render events for browser sound/VFX triggers
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum RenderEvent {
    EnemyHit { x: f64, y: f64 },
    EnemyKilled { x: f64, y: f64 },
    PlayerHit,
    PlayerFired { x: f64, y: f64 },
    MissileDestroyed { x: f64, y: f64 },
    MissileBonus,
    WallHit { wall_index: usize, x: f64, y: f64, from_player: bool },
    WallDestroyed { wall_index: usize },
    KamikazeSpawned { x: f64, y: f64 },
    KamikazeKilled { x: f64, y: f64 },
    MonsterSpawned,
    MonsterHit { x: f64, y: f64 },
    Monster2Spawned,
    Monster2Disappeared,
    Monster2Reappeared { x: f64, y: f64 },
    LevelComplete { level: i32 },
    GameOver,
    BonusLife,
    ScoreChange { delta: i32 },
}
