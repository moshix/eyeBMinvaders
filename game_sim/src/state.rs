use crate::constants::*;
use crate::entities::*;
use crate::game::HeadlessGame;

pub fn get_state(game: &HeadlessGame) -> [f32; 24] {
    let player_cx = game.player_x + PLAYER_WIDTH / 2.0;
    let player_cy = game.player_y + PLAYER_HEIGHT / 2.0;

    let nx = |v: f64| -> f32 { (v / GAME_WIDTH) as f32 };
    let ny = |v: f64| -> f32 { (v / GAME_HEIGHT) as f32 };

    let mut features = [0.0f32; 24];
    let mut i = 0;

    // 1. Player position (normalized)
    features[i] = nx(player_cx);
    i += 1;

    // 2. Player lives (normalized)
    features[i] = game.player_lives as f32 / PLAYER_LIVES as f32;
    i += 1;

    // 3. Level (normalized, cap at 10)
    features[i] = (game.current_level.min(10) as f64 / 10.0) as f32;
    i += 1;

    // 4. Number of enemies (normalized)
    features[i] = ((game.enemies.len().min(60) as f64) / 60.0) as f32;
    i += 1;

    // 5-6. Nearest enemy relative position
    if !game.enemies.is_empty() {
        let nearest = game
            .enemies
            .iter()
            .min_by(|a, b| {
                let da = (a.x + a.width / 2.0 - player_cx).abs();
                let db = (b.x + b.width / 2.0 - player_cx).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        features[i] = nx(nearest.x + nearest.width / 2.0 - player_cx);
        i += 1;
        features[i] = ny(nearest.y + nearest.height / 2.0 - player_cy);
        i += 1;
    } else {
        features[i] = 0.0;
        i += 1;
        features[i] = -1.0;
        i += 1;
    }

    // 7-8. Lowest enemy position (danger indicator)
    if !game.enemies.is_empty() {
        let lowest = game
            .enemies
            .iter()
            .max_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
            .unwrap();
        features[i] = nx(lowest.x + lowest.width / 2.0 - player_cx);
        i += 1;
        features[i] = ny(lowest.y);
        i += 1;
    } else {
        features[i] = 0.0;
        i += 1;
        features[i] = 0.0;
        i += 1;
    }

    // 9-11. Nearest enemy bullet
    let enemy_bullets: Vec<&Bullet> = game.bullets.iter().filter(|b| b.is_enemy).collect();
    if !enemy_bullets.is_empty() {
        let nearest_b = enemy_bullets
            .iter()
            .min_by(|a, b| {
                let da = (a.x - player_cx).powi(2) + (a.y - player_cy).powi(2);
                let db = (b.x - player_cx).powi(2) + (b.y - player_cy).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        features[i] = nx(nearest_b.x - player_cx);
        i += 1;
        features[i] = ny(nearest_b.y - player_cy);
        i += 1;
        features[i] = enemy_bullets.len() as f32 / 10.0;
        i += 1;
    } else {
        features[i] = 0.0;
        i += 1;
        features[i] = -1.0;
        i += 1;
        features[i] = 0.0;
        i += 1;
    }

    // 12-14. Nearest missile
    if !game.missiles.is_empty() {
        let nearest_m = game
            .missiles
            .iter()
            .min_by(|a, b| {
                let da = (a.x - player_cx).powi(2) + (a.y - player_cy).powi(2);
                let db = (b.x - player_cx).powi(2) + (b.y - player_cy).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        features[i] = nx(nearest_m.x - player_cx);
        i += 1;
        features[i] = ny(nearest_m.y - player_cy);
        i += 1;
        features[i] = game.missiles.len() as f32 / 5.0;
        i += 1;
    } else {
        features[i] = 0.0;
        i += 1;
        features[i] = -1.0;
        i += 1;
        features[i] = 0.0;
        i += 1;
    }

    // 15-17. Nearest kamikaze
    if !game.kamikazes.is_empty() {
        let nearest_k = game
            .kamikazes
            .iter()
            .min_by(|a, b| {
                let da =
                    (a.x + a.width / 2.0 - player_cx).powi(2) + (a.y + a.height / 2.0 - player_cy).powi(2);
                let db =
                    (b.x + b.width / 2.0 - player_cx).powi(2) + (b.y + b.height / 2.0 - player_cy).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        features[i] = nx(nearest_k.x + nearest_k.width / 2.0 - player_cx);
        i += 1;
        features[i] = ny(nearest_k.y + nearest_k.height / 2.0 - player_cy);
        i += 1;
        features[i] = game.kamikazes.len() as f32 / 5.0;
        i += 1;
    } else {
        features[i] = 0.0;
        i += 1;
        features[i] = -1.0;
        i += 1;
        features[i] = 0.0;
        i += 1;
    }

    // 18-19. Monster info
    if let Some(ref monster) = game.monster {
        if !monster.hit {
            features[i] = nx(monster.x + monster.width / 2.0 - player_cx);
            i += 1;
            features[i] = ny(monster.y);
            i += 1;
        } else {
            features[i] = 0.0;
            i += 1;
            features[i] = -1.0;
            i += 1;
        }
    } else {
        features[i] = 0.0;
        i += 1;
        features[i] = -1.0;
        i += 1;
    }

    // 20. Is player currently invulnerable
    features[i] = if game.is_player_hit { 1.0 } else { 0.0 };
    i += 1;

    // 21. Number of walls remaining
    features[i] = game.walls.len() as f32 / 4.0;
    i += 1;

    // 22-24. Nearest wall
    if !game.walls.is_empty() {
        let nearest_w = game
            .walls
            .iter()
            .min_by(|a, b| {
                let da = (a.x + a.width / 2.0 - player_cx).abs();
                let db = (b.x + b.width / 2.0 - player_cx).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        features[i] = nx(nearest_w.x + nearest_w.width / 2.0 - player_cx);
        i += 1;
        features[i] = ny(nearest_w.y - player_cy);
        i += 1;
        features[i] = 1.0 - nearest_w.hit_count as f32 / WALL_MAX_HITS_TOTAL as f32;
    } else {
        features[i] = 0.0;
        i += 1;
        features[i] = 0.0;
        i += 1;
        features[i] = 0.0;
    }

    features
}

pub fn calculate_reward(
    game: &HeadlessGame,
    old_score: i32,
    old_lives: i32,
    wall_destroyed_count: i32,
) -> f32 {
    let mut reward: f32 = 0.0;

    reward += (game.score - old_score) as f32 * 0.01;

    if game.player_lives < old_lives {
        reward -= 5.0;
    }

    if game.game_over {
        reward -= 20.0;
    }

    if wall_destroyed_count > 0 {
        reward -= 2.0 * wall_destroyed_count as f32;
    }

    reward += 0.01; // survival bonus

    reward
}
