use crate::constants::*;
use crate::entities::*;
use crate::game::HeadlessGame;

/// 45-feature state vector for DQN training.
///
/// Features 0-23: original 24 features (with velocity additions interleaved)
/// Features 24-44: new features (monster2, 2nd nearest threats, danger heatmap)
pub fn get_state(game: &HeadlessGame) -> [f32; STATE_SIZE] {
    let player_cx = game.player_x + PLAYER_WIDTH / 2.0;
    let player_cy = game.player_y + PLAYER_HEIGHT / 2.0;

    let nx = |v: f64| -> f32 { (v / GAME_WIDTH) as f32 };
    let ny = |v: f64| -> f32 { (v / GAME_HEIGHT) as f32 };

    let mut f = [0.0f32; STATE_SIZE];

    // [0] Player position (normalized)
    f[0] = nx(player_cx);

    // [1] Player lives (normalized)
    f[1] = game.player_lives as f32 / PLAYER_LIVES as f32;

    // [2] Level (normalized, cap at 10)
    f[2] = (game.current_level.min(10) as f64 / 10.0) as f32;

    // [3] Number of enemies (normalized)
    f[3] = ((game.enemies.len().min(60) as f64) / 60.0) as f32;

    // [4-5] Nearest enemy relative position
    if !game.enemies.is_empty() {
        let nearest = game.enemies.iter()
            .min_by(|a, b| {
                let da = (a.x + a.width / 2.0 - player_cx).abs();
                let db = (b.x + b.width / 2.0 - player_cx).abs();
                da.partial_cmp(&db).unwrap()
            }).unwrap();
        f[4] = nx(nearest.x + nearest.width / 2.0 - player_cx);
        f[5] = ny(nearest.y + nearest.height / 2.0 - player_cy);
    } else {
        f[4] = 0.0;
        f[5] = -1.0;
    }

    // [6-7] Lowest enemy position (danger indicator)
    if !game.enemies.is_empty() {
        let lowest = game.enemies.iter()
            .max_by(|a, b| a.y.partial_cmp(&b.y).unwrap()).unwrap();
        f[6] = nx(lowest.x + lowest.width / 2.0 - player_cx);
        f[7] = ny(lowest.y);
    }

    // --- Enemy bullets: nearest + velocity ---
    let enemy_bullets: Vec<&Bullet> = game.bullets.iter().filter(|b| b.is_enemy).collect();
    let mut sorted_bullets = enemy_bullets.clone();
    sorted_bullets.sort_by(|a, b| {
        let da = (a.x - player_cx).powi(2) + (a.y - player_cy).powi(2);
        let db = (b.x - player_cx).powi(2) + (b.y - player_cy).powi(2);
        da.partial_cmp(&db).unwrap()
    });

    // [8-10] Nearest enemy bullet (rel x, rel y, count)
    if let Some(b) = sorted_bullets.first() {
        f[8] = nx(b.x - player_cx);
        f[9] = ny(b.y - player_cy);
        f[10] = enemy_bullets.len() as f32 / 10.0;
        // [11-12] Nearest bullet velocity (dx, dy normalized)
        if b.has_direction {
            f[11] = (b.dx / ENEMY_BULLET_SPEED) as f32;
            f[12] = (b.dy / ENEMY_BULLET_SPEED) as f32;
        } else {
            f[11] = 0.0;  // straight down
            f[12] = 1.0;
        }
    } else {
        f[8] = 0.0;
        f[9] = -1.0;
        f[10] = 0.0;
        f[11] = 0.0;
        f[12] = 0.0;
    }

    // --- Missiles: nearest + velocity ---
    let mut sorted_missiles: Vec<&Missile> = game.missiles.iter().collect();
    sorted_missiles.sort_by(|a, b| {
        let da = (a.x - player_cx).powi(2) + (a.y - player_cy).powi(2);
        let db = (b.x - player_cx).powi(2) + (b.y - player_cy).powi(2);
        da.partial_cmp(&db).unwrap()
    });

    // [13-15] Nearest missile (rel x, rel y, count)
    if let Some(m) = sorted_missiles.first() {
        f[13] = nx(m.x - player_cx);
        f[14] = ny(m.y - player_cy);
        f[15] = game.missiles.len() as f32 / 5.0;
        // [16-17] Nearest missile velocity (cos/sin of angle)
        f[16] = m.angle.cos() as f32;
        f[17] = m.angle.sin() as f32;
    } else {
        f[13] = 0.0;
        f[14] = -1.0;
        f[15] = 0.0;
        f[16] = 0.0;
        f[17] = 0.0;
    }

    // --- Kamikazes: nearest + velocity ---
    let mut sorted_kamikazes: Vec<&Kamikaze> = game.kamikazes.iter().collect();
    sorted_kamikazes.sort_by(|a, b| {
        let da = (a.x + a.width / 2.0 - player_cx).powi(2) + (a.y + a.height / 2.0 - player_cy).powi(2);
        let db = (b.x + b.width / 2.0 - player_cx).powi(2) + (b.y + b.height / 2.0 - player_cy).powi(2);
        da.partial_cmp(&db).unwrap()
    });

    // [18-20] Nearest kamikaze (rel x, rel y, count)
    if let Some(k) = sorted_kamikazes.first() {
        f[18] = nx(k.x + k.width / 2.0 - player_cx);
        f[19] = ny(k.y + k.height / 2.0 - player_cy);
        f[20] = game.kamikazes.len() as f32 / 5.0;
        // [21-22] Nearest kamikaze velocity (cos/sin of angle)
        f[21] = k.angle.cos() as f32;
        f[22] = k.angle.sin() as f32;
    } else {
        f[18] = 0.0;
        f[19] = -1.0;
        f[20] = 0.0;
        f[21] = 0.0;
        f[22] = 0.0;
    }

    // [23-24] Monster info
    if let Some(ref monster) = game.monster {
        if !monster.hit {
            f[23] = nx(monster.x + monster.width / 2.0 - player_cx);
            f[24] = ny(monster.y);
        } else {
            f[23] = 0.0;
            f[24] = -1.0;
        }
    } else {
        f[23] = 0.0;
        f[24] = -1.0;
    }

    // [25-28] Monster2 info (position + velocity) — NEW
    if let Some(ref m2) = game.monster2 {
        if !m2.hit && !m2.is_disappeared {
            f[25] = nx(m2.x + m2.width / 2.0 - player_cx);
            f[26] = ny(m2.y);
            f[27] = (m2.dx_val / MONSTER2_SPEED) as f32;
            f[28] = (m2.dy_val / MONSTER2_SPEED) as f32;
        } else {
            f[25] = 0.0;
            f[26] = -1.0;
            f[27] = 0.0;
            f[28] = 0.0;
        }
    } else {
        f[25] = 0.0;
        f[26] = -1.0;
        f[27] = 0.0;
        f[28] = 0.0;
    }

    // [29] Is player currently invulnerable
    f[29] = if game.is_player_hit { 1.0 } else { 0.0 };

    // [30] Number of walls remaining
    f[30] = game.walls.len() as f32 / 4.0;

    // [31-33] Nearest wall
    if !game.walls.is_empty() {
        let nearest_w = game.walls.iter()
            .min_by(|a, b| {
                let da = (a.x + a.width / 2.0 - player_cx).abs();
                let db = (b.x + b.width / 2.0 - player_cx).abs();
                da.partial_cmp(&db).unwrap()
            }).unwrap();
        f[31] = nx(nearest_w.x + nearest_w.width / 2.0 - player_cx);
        f[32] = ny(nearest_w.y - player_cy);
        f[33] = 1.0 - nearest_w.hit_count as f32 / WALL_MAX_HITS_TOTAL as f32;
    }

    // [34-36] 2nd nearest enemy bullet (rel x, rel y, dy) — NEW
    if sorted_bullets.len() >= 2 {
        let b2 = sorted_bullets[1];
        f[34] = nx(b2.x - player_cx);
        f[35] = ny(b2.y - player_cy);
        f[36] = if b2.has_direction { (b2.dy / ENEMY_BULLET_SPEED) as f32 } else { 1.0 };
    } else {
        f[34] = 0.0;
        f[35] = -1.0;
        f[36] = 0.0;
    }

    // [37-39] 2nd nearest missile (rel x, rel y, angle_sin) — NEW
    if sorted_missiles.len() >= 2 {
        let m2 = sorted_missiles[1];
        f[37] = nx(m2.x - player_cx);
        f[38] = ny(m2.y - player_cy);
        f[39] = m2.angle.sin() as f32;
    } else {
        f[37] = 0.0;
        f[38] = -1.0;
        f[39] = 0.0;
    }

    // [40-49] Danger heatmap: 10 columns (102px each) — higher resolution for dodging
    // Weighted threat density per column: bullets=1.0, missiles=2.0, kamikazes=3.0
    let col_width = GAME_WIDTH / 10.0;
    for b in enemy_bullets.iter() {
        let col = ((b.x / col_width) as usize).min(9);
        f[40 + col] += 1.0;
    }
    for m in game.missiles.iter() {
        let col = ((m.x / col_width) as usize).min(9);
        f[40 + col] += 2.0;
    }
    for k in game.kamikazes.iter() {
        let col = (((k.x + k.width / 2.0) / col_width) as usize).min(9);
        f[40 + col] += 3.0;
    }
    for j in 40..50 {
        f[j] = (f[j] / 10.0).min(1.0);
    }

    // [50] Enemy speed (normalized)
    f[50] = (game.enemy_speed / 10.0).min(1.0) as f32;

    // [51] Enemy direction (-1 left, +1 right) -> normalized to [0, 1]
    f[51] = (game.enemy_direction as f32 + 1.0) / 2.0;

    // [52] Fire cooldown (0 = can fire now, 1 = just fired)
    let fire_elapsed = game.game_time - game.last_fire_time;
    let fire_ready = (fire_elapsed / (FIRE_RATE * 1000.0)).min(1.0) as f32;
    f[52] = fire_ready;

    // [53] Threat urgency: closest threat time-to-impact (lower = more dangerous)
    let player_y = game.player_y + PLAYER_HEIGHT / 2.0;
    let mut min_tti: f64 = 1.0;
    for b in game.bullets.iter().filter(|b| b.is_enemy) {
        if b.y < player_y {
            let dy = player_y - b.y;
            let tti = dy / (ENEMY_BULLET_SPEED * 1000.0 / 60.0);
            let tti_norm = (tti / 60.0).min(1.0);
            if tti_norm < min_tti { min_tti = tti_norm; }
        }
    }
    for k in game.kamikazes.iter() {
        let ky = k.y + k.height / 2.0;
        if ky < player_y {
            let dist = ((k.x + k.width/2.0 - player_cx).powi(2) + (ky - player_y).powi(2)).sqrt();
            let tti = dist / (KAMIKAZE_SPEED * 1000.0 / 60.0);
            let tti_norm = (tti / 60.0).min(1.0);
            if tti_norm < min_tti { min_tti = tti_norm; }
        }
    }
    f[53] = min_tti as f32;

    // [54] Lowest enemy proximity to wall (0=far, 1=at wall=game over imminent)
    if !game.enemies.is_empty() {
        let lowest_y = game.enemies.iter()
            .map(|e| e.y + e.height)
            .fold(0.0f64, f64::max);
        f[54] = (lowest_y / WALL_Y).min(1.0) as f32;
    }

    // [55-56] Predictive lateral threat: project bullets 0.3s ahead, check ±60px
    let lookahead_frames: f64 = 9.0; // 0.3s at 30Hz
    let mut left_future: f32 = 0.0;
    let mut right_future: f32 = 0.0;
    for b in game.bullets.iter().filter(|b| b.is_enemy && !b.removed) {
        let future_x = if b.has_direction { b.x + b.dx * lookahead_frames } else { b.x };
        let future_y = if b.has_direction { b.y + b.dy * lookahead_frames } else { b.y + ENEMY_BULLET_SPEED * 1000.0 / 60.0 * lookahead_frames };
        let dx = future_x - player_cx;
        if future_y > player_y - 60.0 && future_y < player_y + 30.0 && dx.abs() < 60.0 {
            if dx < 0.0 { left_future += 1.0; } else { right_future += 1.0; }
        }
    }
    for k in game.kamikazes.iter().filter(|k| !k.removed) {
        let kx = k.x + k.width / 2.0;
        let ky = k.y + k.height / 2.0;
        let future_x = kx + k.angle.cos() * KAMIKAZE_SPEED * 1000.0 / 60.0 * lookahead_frames;
        let future_y = ky + k.angle.sin() * KAMIKAZE_SPEED * 1000.0 / 60.0 * lookahead_frames;
        let dx = future_x - player_cx;
        if future_y > player_y - 80.0 && future_y < player_y + 30.0 && dx.abs() < 60.0 {
            if dx < 0.0 { left_future += 2.0; } else { right_future += 2.0; }
        }
    }
    f[55] = (left_future / 5.0).min(1.0);
    f[56] = (right_future / 5.0).min(1.0);

    // [57] Bullet convergence: count of enemy bullets arriving within ±50px of player in next 0.5s
    let lookahead_05s: f64 = 15.0; // 0.5s at 30Hz
    let mut convergence_count: f32 = 0.0;
    for b in game.bullets.iter().filter(|b| b.is_enemy && !b.removed) {
        let future_x = if b.has_direction { b.x + b.dx * lookahead_05s } else { b.x };
        let future_y = if b.has_direction { b.y + b.dy * lookahead_05s } else { b.y + ENEMY_BULLET_SPEED * 1000.0 / 60.0 * lookahead_05s };
        if (future_x - player_cx).abs() < 50.0 && future_y >= player_y - 30.0 && future_y <= player_y + 30.0 {
            convergence_count += 1.0;
        }
    }
    f[57] = (convergence_count / 8.0).min(1.0);

    // [58] Monster2 predicted X at player Y (where will Monster2 cross player's row?)
    if let Some(ref m2) = game.monster2 {
        if !m2.hit && !m2.is_disappeared && m2.dy_val.abs() > 0.01 {
            let frames_to_player = (player_y - m2.y) / (m2.dy_val * 1000.0 / 60.0);
            if frames_to_player > 0.0 && frames_to_player < 60.0 {
                let predicted_x = m2.x + m2.width / 2.0 + m2.dx_val * 1000.0 / 60.0 * frames_to_player;
                f[58] = ((predicted_x - player_cx) / GAME_WIDTH).max(-0.5).min(0.5) as f32 + 0.5;
            } else {
                f[58] = 0.5; // neutral — not heading toward player row
            }
        } else {
            f[58] = 0.5;
        }
    }

    // [59] Player position bias: -1 at left edge, 0 at center, +1 at right edge
    f[59] = ((player_cx / GAME_WIDTH) as f32 - 0.5) * 2.0;

    // [60] Fire line clear: 1.0 if no wall blocks vertical shot, 0.0 if wall above
    let fire_x = game.player_x + PLAYER_WIDTH / 2.0;
    let fire_blocked = game.walls.iter().any(|w| {
        fire_x >= w.x && fire_x <= w.x + w.width && w.hit_count < WALL_MAX_HITS_TOTAL
    });
    f[60] = if fire_blocked { 0.0 } else { 1.0 };

    // [61] Total active threats (bullets + kamikazes + missiles, normalized)
    let total_threats = enemy_bullets.len() + game.kamikazes.len() + game.missiles.len();
    f[61] = (total_threats as f32 / 15.0).min(1.0);

    f
}

pub fn calculate_reward(
    game: &HeadlessGame,
    old_score: i32,
    old_lives: i32,
    wall_destroyed_count: i32,
    kamikazes_killed_this_step: i32,
    missiles_shot_this_step: i32,
    near_misses: i32,
    level_completed: bool,
    player_wall_hits: i32,
    enemies_killed_this_step: i32,
    monster_killed_this_step: bool,
) -> f32 {
    let mut reward: f32 = 0.0;

    // Score-based reward
    reward += (game.score - old_score) as f32 * 0.01;

    // Life loss penalty (reduced from 5.0 — was causing excessive dodging over offense)
    if game.player_lives < old_lives {
        reward -= 3.0;
    }

    if game.game_over {
        reward -= 20.0;
    }

    if wall_destroyed_count > 0 {
        reward -= 2.0 * wall_destroyed_count as f32;
    }

    // Penalty for shooting own walls
    if player_wall_hits > 0 {
        reward -= 0.5 * player_wall_hits as f32;
    }

    // Progressive survival bonus: scales with level
    reward += 0.01 * game.current_level as f32;

    // Enemy kill bonus — direct reward so killing enemies isn't dwarfed by missiles
    reward += enemies_killed_this_step as f32 * 1.0;

    // Extra reward for killing kamikazes
    reward += kamikazes_killed_this_step as f32 * 1.5;

    // Missile interception (reduced from 2.0 — score already gives +5.0 via 500pts)
    reward += missiles_shot_this_step as f32 * 0.5;

    // Monster kill bonus — restoring walls is strategically critical
    if monster_killed_this_step {
        reward += 3.0;
    }

    // Dodging reward: threats passed close but missed
    reward += near_misses as f32 * 0.15;

    // Level completion bonus — increased and scaling to push past level 7
    if level_completed {
        let level = game.current_level as f32;
        reward += 5.0 + 3.0 * level;
    }

    // Progressive penalty for enemies approaching wall — gives gradient BEFORE game over
    // Scales with level: higher levels have faster enemies, need stronger urgency
    // Base: quadratic ramp 0 at 70% height → -15 at wall
    // Level multiplier: 1.0 at level 1, up to 2.5 at level 8+
    if !game.enemies.is_empty() {
        let lowest_y = game.enemies.iter()
            .map(|e| e.y + e.height)
            .fold(0.0f64, f64::max);
        let proximity = (lowest_y / WALL_Y).min(1.0) as f32;
        if proximity > 0.7 {
            let danger = (proximity - 0.7) / 0.3;
            let level_mult = 1.0 + 0.2 * (game.current_level.min(9) - 1) as f32;
            reward -= 15.0 * level_mult * danger * danger;
        }
    }

    reward
}
