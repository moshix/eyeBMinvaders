use crate::constants::*;
use crate::entities::*;
use crate::entities::RenderEvent;
use crate::game::HeadlessGame;
use rand::Rng;
use std::collections::HashMap;

pub fn handle_monster_creation(game: &mut HeadlessGame) {
    if game.monster.is_some() {
        return;
    }
    if game.game_time - game.last_monster_time <= MONSTER_INTERVAL {
        return;
    }

    let direction: i32 = if game.rng.gen::<f64>() < 0.5 { 1 } else { -1 };
    let start_x = if direction == 1 {
        -MONSTER_WIDTH
    } else {
        GAME_WIDTH + MONSTER_WIDTH
    };
    let should_slalom = game.enemies.len() < MONSTER_SLALOM_THRESHOLD;

    let top_row = if !game.enemies.is_empty() {
        game.enemies
            .iter()
            .map(|e| e.y)
            .fold(f64::INFINITY, f64::min)
            - 50.0
    } else {
        MONSTER_HEIGHT
    };

    let y = if should_slalom {
        0.0
    } else {
        top_row.max(MONSTER_HEIGHT) - 45.0
    };

    game.monster = Some(Monster::new(
        start_x,
        y,
        should_slalom,
        game.game_time,
        direction,
    ));
    game.last_monster_time = game.game_time;
    game.emit(EventType::MonsterSpawned);
    game.render_events.push(RenderEvent::MonsterSpawned);
}

pub fn handle_monster2_creation(game: &mut HeadlessGame) {
    if game.current_level < 2 {
        return;
    }

    if let Some(ref monster) = game.monster {
        if monster.is_slaloming {
            game.last_monster2_time = game.game_time;
            return;
        }
    }

    if game.monster.is_none()
        && (game.game_time - game.last_monster_time < MONSTER2_INTERVAL / 2.0)
    {
        return;
    }

    if game.monster2.is_some() {
        return;
    }
    if game.game_time - game.last_monster2_time <= MONSTER2_INTERVAL {
        return;
    }

    game.monster2 = Some(Monster2::new(
        GAME_WIDTH / 2.0,
        -MONSTER2_HEIGHT,
        game.game_time,
    ));
    game.last_monster2_time = game.game_time;
}

pub fn handle_kamikaze_creation(game: &mut HeadlessGame) {
    if game.game_time < game.next_kamikaze_time || game.enemies.is_empty() {
        return;
    }

    let idx = game.rng.gen_range(0..game.enemies.len());
    let enemy = game.enemies.remove(idx);

    game.kamikazes.push(Kamikaze::new(
        enemy.x,
        enemy.y,
        enemy.width,
        enemy.height,
        game.game_time,
    ));
    game.emit(EventType::KamikazeSpawned);
    let kx = game.kamikazes.last().map(|k| k.x).unwrap_or(0.0);
    let ky = game.kamikazes.last().map(|k| k.y).unwrap_or(0.0);
    game.render_events.push(RenderEvent::KamikazeSpawned { x: kx, y: ky });

    let n = game.enemies.len();
    if n < KAMIKAZE_VERY_AGGRESSIVE_THRESHOLD {
        game.next_kamikaze_time = game.game_time + KAMIKAZE_VERY_AGGRESSIVE_TIME;
    } else if n < KAMIKAZE_AGGRESSIVE_THRESHOLD {
        game.next_kamikaze_time = game.game_time + KAMIKAZE_AGGRESSIVE_TIME;
    } else {
        game.next_kamikaze_time = game.random_kamikaze_time();
    }
}

pub fn handle_enemy_shooting(game: &mut HeadlessGame) {
    if game.game_time - game.last_enemy_fire_time < game.current_enemy_fire_rate * 1000.0 {
        return;
    }
    if game.enemies.is_empty() {
        return;
    }

    // Build columns: map column index -> index of lowest enemy in that column
    let mut columns: HashMap<i32, usize> = HashMap::new();
    for (i, e) in game.enemies.iter().enumerate() {
        let col = (e.x / (e.width + 20.0)) as i32;
        match columns.get(&col) {
            Some(&existing_idx) => {
                if e.y > game.enemies[existing_idx].y {
                    columns.insert(col, i);
                }
            }
            None => {
                columns.insert(col, i);
            }
        }
    }

    if columns.is_empty() {
        return;
    }

    // Find the bottom-row enemy closest to the player
    let closest_idx = *columns
        .values()
        .min_by(|&&a, &&b| {
            let dist_a = (game.enemies[a].x - game.player_x).abs();
            let dist_b = (game.enemies[b].x - game.player_x).abs();
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .unwrap();

    let e = &game.enemies[closest_idx];
    let bx = e.x + e.width / 2.0;
    let by = e.y + e.height;

    game.bullets.push(Bullet::new(bx, by, true));
    game.last_enemy_fire_time = game.game_time;
}

pub fn handle_missile_launching(game: &mut HeadlessGame) {
    if game.game_time < game.next_missile_time || game.enemies.is_empty() {
        return;
    }

    let min_y = game
        .enemies
        .iter()
        .map(|e| e.y)
        .fold(f64::INFINITY, f64::min);

    let top_row: Vec<usize> = game
        .enemies
        .iter()
        .enumerate()
        .filter(|(_, e)| (e.y - min_y).abs() < f64::EPSILON)
        .map(|(i, _)| i)
        .collect();

    if !top_row.is_empty() {
        let shooter_idx = top_row[game.rng.gen_range(0..top_row.len())];
        let shooter = &game.enemies[shooter_idx];
        let mx = shooter.x + shooter.width / 2.0;
        let my = shooter.y + shooter.height;

        game.missiles.push(Missile::new(mx, my));
        game.emit(EventType::MissileLaunched);
    }

    let interval = game.rng.gen_range(MIN_MISSILE_INTERVAL..MAX_MISSILE_INTERVAL);
    game.next_missile_time = game.game_time + interval;
}
