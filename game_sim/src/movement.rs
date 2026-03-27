use crate::constants::*;
use crate::entities::*;
use crate::game::HeadlessGame;
use rand::Rng;

pub fn move_bullets(game: &mut HeadlessGame, dt: f64) {
    if game.while_player_hit {
        game.bullets.retain(|b| !b.is_enemy);
        return;
    }
    for b in game.bullets.iter_mut() {
        if b.is_enemy {
            if b.has_direction {
                b.x += b.dx * dt;
                b.y += b.dy * dt;
            } else {
                b.y += ENEMY_BULLET_SPEED * dt;
            }
        } else {
            b.y -= BULLET_SPEED * dt;
        }
    }
    game.bullets.retain(|b| {
        b.y > 0.0 && b.y < GAME_HEIGHT && b.x > 0.0 && b.x < GAME_WIDTH
    });
}

pub fn move_enemies(game: &mut HeadlessGame, dt: f64) {
    if game.enemies.is_empty() {
        return;
    }
    let dt_capped = dt.min(0.1);
    let speed = ENEMY_SPEED * game.enemy_speed * dt_capped;
    let mut needs_down = false;

    for e in game.enemies.iter_mut() {
        if e.hits >= ENEMY_HITS_TO_DESTROY {
            continue;
        }
        e.x += speed * game.enemy_direction as f64;
        if e.x + e.width > GAME_WIDTH || e.x < 0.0 {
            needs_down = true;
            e.x = e.x.max(0.0).min(GAME_WIDTH - e.width);
        }
    }

    if needs_down {
        game.enemy_direction *= -1;
        let wall_y = if !game.walls.is_empty() {
            game.walls[0].y - 20.0
        } else {
            GAME_HEIGHT * 0.9
        };
        for e in game.enemies.iter_mut() {
            if e.hits < ENEMY_HITS_TO_DESTROY {
                e.y += 20.0;
                if e.y + e.height >= wall_y {
                    game.game_over = true;
                    game.emit(EventType::GameOver);
                    return;
                }
            }
        }
    }

    game.enemies.retain(|e| e.hits < ENEMY_HITS_TO_DESTROY);
}

pub fn move_kamikazes(game: &mut HeadlessGame, dt: f64) {
    let player_cx = game.player_x + PLAYER_WIDTH / 2.0;
    let player_cy = game.player_y + PLAYER_HEIGHT / 2.0;
    let game_time = game.game_time;
    let player_y = game.player_y;

    let mut new_bullets: Vec<Bullet> = Vec::new();

    for ki in 0..game.kamikazes.len() {
        if game.kamikazes[ki].removed {
            continue;
        }
        game.kamikazes[ki].time += dt;

        // Shooting
        if game_time - game.kamikazes[ki].last_fire_time >= KAMIKAZE_FIRE_RATE {
            new_bullets.push(Bullet::new(
                game.kamikazes[ki].x + game.kamikazes[ki].width / 2.0,
                game.kamikazes[ki].y + game.kamikazes[ki].height,
                true,
            ));
            game.kamikazes[ki].last_fire_time = game_time;
        }

        // Check if past player
        if game.kamikazes[ki].y >= player_y {
            game.kamikazes[ki].removed = true;
            continue;
        }

        // Homing movement
        let target_dx = player_cx - game.kamikazes[ki].x;
        let target_dy = player_cy - game.kamikazes[ki].y;
        game.kamikazes[ki].angle = target_dy.atan2(target_dx);
        let angle = game.kamikazes[ki].angle;
        let curve = (game.kamikazes[ki].time * 2.0).sin() * 100.0;
        game.kamikazes[ki].x += angle.cos() * KAMIKAZE_SPEED * dt;
        game.kamikazes[ki].y += angle.sin() * KAMIKAZE_SPEED * dt;
        game.kamikazes[ki].x += (angle + std::f64::consts::FRAC_PI_2).cos() * curve * dt;

        // Wall collision
        for wi in 0..game.walls.len() {
            if game.walls[wi].hit_count < WALL_MAX_HITS_TOTAL
                && game.walls[wi].missile_hits < WALL_MAX_MISSILE_HITS
            {
                let k = &game.kamikazes[ki];
                let w = &game.walls[wi];
                if k.x + k.width > w.x
                    && k.x < w.x + w.width
                    && k.y + k.height > w.y
                    && k.y < w.y + w.height
                {
                    game.walls[wi].missile_hits += 1;
                    game.score += 30;
                    game.kamikazes[ki].removed = true;
                    break;
                }
            }
        }

        // Off-screen check
        if !game.kamikazes[ki].removed
            && (game.kamikazes[ki].x <= 0.0 || game.kamikazes[ki].x >= GAME_WIDTH)
        {
            game.kamikazes[ki].removed = true;
        }
    }

    game.bullets.extend(new_bullets);
    game.kamikazes.retain(|k| !k.removed);
}

pub fn move_missiles(game: &mut HeadlessGame, dt: f64) {
    let player_cx = game.player_x + PLAYER_WIDTH / 2.0;
    let player_cy = game.player_y + PLAYER_HEIGHT / 2.0;
    let wall_row_y = if !game.walls.is_empty() {
        game.walls[0].y - 50.0
    } else {
        GAME_HEIGHT * 0.85
    };

    for m in game.missiles.iter_mut() {
        m.time += dt;
        if m.y < wall_row_y {
            let dx = player_cx - m.x;
            let dy = player_cy - m.y;
            m.angle = dy.atan2(dx);
        }
        let curve = (m.time * 2.0).sin() * 100.0;
        m.x += m.angle.cos() * MISSILE_SPEED * dt;
        m.y += m.angle.sin() * MISSILE_SPEED * dt;
        m.x += (m.angle + std::f64::consts::FRAC_PI_2).cos() * curve * dt;
    }

    game.missiles.retain(|m| {
        m.y > 0.0 && m.y < GAME_HEIGHT && m.x > 0.0 && m.x < GAME_WIDTH
    });
}

pub fn move_monster(game: &mut HeadlessGame, dt: f64) {
    if game.monster.is_none() {
        return;
    }

    // Check hit animation removal
    {
        let m = game.monster.as_ref().unwrap();
        if m.hit {
            if game.game_time - m.hit_time > MONSTER_HIT_DURATION {
                game.monster = None;
                game.last_monster_time = game.game_time;
            }
            return;
        }
    }

    let is_slaloming = game.monster.as_ref().unwrap().is_slaloming;

    if is_slaloming {
        // Update movement
        {
            let m = game.monster.as_mut().unwrap();
            m.slalom_time += dt;
            let center_x = GAME_WIDTH / 2.0;
            m.x = center_x + (m.slalom_time * 1.2).sin() * MONSTER_SLALOM_AMPLITUDE;
            m.y += MONSTER_VERTICAL_SPEED * dt;
        }

        // Fire missiles
        let m = game.monster.as_ref().unwrap();
        if game.game_time - m.last_fire_time >= MONSTER_SLALOM_FIRE_RATE {
            let mx = m.x + m.width / 2.0;
            let my = m.y + m.height;
            game.missiles.push(Missile::from_monster(
                mx, my, std::f64::consts::FRAC_PI_2, 44.0, 44.0,
            ));
            game.monster.as_mut().unwrap().last_fire_time = game.game_time;
            game.emit(EventType::MissileLaunched);
        }

        // Check if past walls
        let wall_y = if !game.walls.is_empty() { game.walls[0].y } else { GAME_HEIGHT * 0.85 };
        let m = game.monster.as_ref().unwrap();
        if m.y >= wall_y - m.height - 20.0 {
            game.monster = None;
            game.last_monster_time = game.game_time;
        }
    } else {
        // Linear movement
        {
            let m = game.monster.as_mut().unwrap();
            m.x += MONSTER_SPEED * m.direction as f64 * dt;
        }

        let m = game.monster.as_ref().unwrap();
        let is_on_screen = m.x >= 0.0 && m.x + m.width <= GAME_WIDTH;

        if !m.has_shot && is_on_screen {
            let mx = m.x + m.width / 2.0;
            let my = m.y + m.height;
            let mw = m.width;
            for offset in &[-mw / 4.0, mw / 4.0] {
                game.missiles.push(Missile::from_monster(
                    mx + offset, my, std::f64::consts::FRAC_PI_2, 44.0, 44.0,
                ));
            }
            game.monster.as_mut().unwrap().has_shot = true;
            game.emit(EventType::MissileLaunched);
        }

        let m = game.monster.as_ref().unwrap();
        let dir = m.direction;
        let mx = m.x;
        if (dir == 1 && mx > GAME_WIDTH + MONSTER_WIDTH)
            || (dir == -1 && mx < -MONSTER_WIDTH)
        {
            game.monster = None;
            game.last_monster_time = game.game_time;
        }
    }
}

pub fn get_monster2_pattern(level: i32) -> &'static str {
    match level {
        2 => "spiral",
        3 => "zigzag",
        4 => "figure8",
        5 => "bounce",
        6 => "wave",
        7 => "teleport",
        8 => "chase",
        _ => "random",
    }
}

pub fn move_monster2(game: &mut HeadlessGame, dt: f64) {
    if game.monster2.is_none() {
        return;
    }

    // Handle hit state
    {
        let m2 = game.monster2.as_ref().unwrap();
        if m2.hit {
            if game.game_time - m2.hit_time > MONSTER_HIT_DURATION {
                let return_delay = game.rng.gen_range(5000.0..9000.0);
                let m2 = game.monster2.as_mut().unwrap();
                m2.is_disappeared = true;
                m2.disappear_time = game.game_time;
                m2.return_delay = return_delay;
                m2.hit = false;
            }
            return;
        }
    }

    // Handle disappeared state
    {
        let m2 = game.monster2.as_ref().unwrap();
        if m2.is_disappeared {
            if game.game_time - m2.disappear_time > m2.return_delay {
                let m2 = game.monster2.as_mut().unwrap();
                m2.is_disappeared = false;
                m2.x = GAME_WIDTH / 2.0;
                m2.y = -MONSTER2_HEIGHT;
                m2.spiral_angle = 0.0;
                m2.center_x = GAME_WIDTH / 2.0;
            }
            return;
        }
    }

    // Pattern movement
    let pattern = get_monster2_pattern(game.current_level);
    let player_x = game.player_x;
    let player_y = game.player_y;
    let game_time = game.game_time;

    // Generate random values needed by patterns before borrowing monster2
    let bounce_sign_x: f64 = if game.rng.gen::<bool>() { 1.0 } else { -1.0 };
    let bounce_sign_y: f64 = if game.rng.gen::<bool>() { 1.0 } else { -1.0 };
    let rand1 = game.rng.gen::<f64>();
    let rand2 = game.rng.gen::<f64>();

    {
        let m2 = game.monster2.as_mut().unwrap();
        m2.y += MONSTER2_VERTICAL_SPEED * dt;

        match pattern {
            "spiral" => {
                m2.spiral_angle += MONSTER2_SPIRAL_SPEED * dt;
                let radius = MONSTER2_SPIRAL_RADIUS * (m2.y / 200.0).min(1.0);
                m2.x = m2.center_x + m2.spiral_angle.cos() * radius;
            }
            "zigzag" => {
                if m2.zigzag_dir == 0 {
                    m2.zigzag_dir = 1;
                    m2.zigzag_amplitude = GAME_WIDTH * 0.4;
                    m2.zigzag_phase = 0.0;
                }
                m2.zigzag_phase += dt * 1.5;
                m2.x = GAME_WIDTH / 2.0 + m2.zigzag_phase.sin() * m2.zigzag_amplitude;
                m2.y += MONSTER2_VERTICAL_SPEED * dt * 0.33;
                m2.x = m2.x.max(0.0).min(GAME_WIDTH - m2.width);
            }
            "figure8" => {
                m2.spiral_angle += MONSTER2_SPIRAL_SPEED * dt;
                m2.x = m2.center_x + m2.spiral_angle.cos() * MONSTER2_SPIRAL_RADIUS;
                m2.y += (2.0 * m2.spiral_angle).sin() * dt * 30.0;
            }
            "bounce" => {
                if m2.dx_val == 0.0 && m2.dy_val == 0.0 {
                    m2.dx_val = bounce_sign_x * MONSTER2_SPEED;
                    m2.dy_val = bounce_sign_y * MONSTER2_SPEED * 0.7;
                }
                m2.x += m2.dx_val * dt;
                m2.y += m2.dy_val * dt;
                if m2.x < 0.0 || m2.x > GAME_WIDTH - m2.width {
                    m2.dx_val = -m2.dx_val;
                    m2.x = m2.x.max(0.0).min(GAME_WIDTH - m2.width);
                }
                if m2.y < 0.0 || m2.y > GAME_HEIGHT * 0.7 {
                    m2.dy_val = -m2.dy_val;
                    m2.y = m2.y.max(0.0).min(GAME_HEIGHT * 0.7);
                }
            }
            "wave" => {
                if m2.wave_start_x == 0.0 {
                    m2.wave_start_x = m2.x;
                }
                m2.x = m2.wave_start_x + (m2.y / 50.0).sin() * (GAME_WIDTH / 4.0);
            }
            "chase" => {
                let chase_dx = player_x - m2.x;
                let chase_dy = player_y - m2.y - 200.0;
                let dist = (chase_dx * chase_dx + chase_dy * chase_dy).sqrt();
                if dist > 1.0 {
                    m2.x += (chase_dx / dist) * MONSTER2_SPEED * 1.2 * dt;
                    m2.y += (chase_dy / dist) * MONSTER2_SPEED * 0.7 * dt;
                }
            }
            _ => {
                // teleport / random
                if m2.next_move_time == 0.0 || game_time > m2.next_move_time {
                    m2.target_x = rand1 * (GAME_WIDTH - m2.width);
                    m2.target_y = (rand2 * GAME_HEIGHT * 0.5).min(m2.y + 100.0);
                    m2.next_move_time = game_time + 1000.0;
                }
                let tdx = m2.target_x - m2.x;
                let tdy = m2.target_y - m2.y;
                let tdist = (tdx * tdx + tdy * tdy).sqrt();
                if tdist > 1.0 {
                    m2.x += (tdx / tdist) * MONSTER2_SPEED * dt;
                    m2.y += (tdy / tdist) * MONSTER2_SPEED * dt;
                }
            }
        }
    }

    // Spread bullets every 2.8s
    let m2 = game.monster2.as_ref().unwrap();
    if game_time - m2.last_fire_time >= 2800.0 {
        let bx = m2.x + m2.width / 2.0;
        let by = m2.y + m2.height;
        let half_pi = std::f64::consts::FRAC_PI_2;
        let eighth_pi = std::f64::consts::FRAC_PI_8;
        for i in -1..=1i32 {
            let spread_angle = half_pi + i as f64 * eighth_pi;
            game.bullets.push(Bullet::with_direction(
                bx, by, true,
                spread_angle.cos() * ENEMY_BULLET_SPEED * 1.2,
                spread_angle.sin() * ENEMY_BULLET_SPEED * 1.2,
            ));
        }
        game.monster2.as_mut().unwrap().last_fire_time = game_time;
    }

    // Off-screen removal
    let m2_y = game.monster2.as_ref().unwrap().y;
    if m2_y > GAME_HEIGHT + MONSTER2_HEIGHT {
        game.monster2 = None;
        game.last_monster2_time = game_time;
    }
}
