use crate::constants::*;
use crate::entities::*;
use crate::game::HeadlessGame;

pub fn detect_collisions(game: &mut HeadlessGame) {
    check_bullet_kamikaze(game);
    check_bullet_wall(game);
    check_bullet_enemy(game);
    check_bullet_missile(game);
    check_bullet_monster(game);
    check_bullet_monster2(game);
    check_enemy_bullet_player(game);
    check_kamikaze_player(game);
    check_missile_player(game);
    check_missile_wall(game);
    check_enemy_bullet_wall(game);
    remove_destroyed_walls(game);
    compact_entities(game);
}

fn compact_entities(game: &mut HeadlessGame) {
    game.bullets.retain(|b| !b.removed);
    game.kamikazes.retain(|k| !k.removed);
    game.missiles.retain(|m| !m.removed);
    game.enemies.retain(|e| e.hits < ENEMY_HITS_TO_DESTROY);
}

fn check_bullet_kamikaze(game: &mut HeadlessGame) {
    for bi in 0..game.bullets.len() {
        if game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        for ki in 0..game.kamikazes.len() {
            if game.kamikazes[ki].removed {
                continue;
            }
            let b = &game.bullets[bi];
            let k = &game.kamikazes[ki];
            if b.x < k.x + k.width
                && b.x + BULLET_W > k.x
                && b.y < k.y + k.height
                && b.y + BULLET_H > k.y
            {
                game.bullets[bi].removed = true;
                game.kamikazes[ki].hits += 1;
                if game.kamikazes[ki].hits >= KAMIKAZE_HITS_TO_DESTROY {
                    game.kamikazes[ki].removed = true;
                    game.score += 300;
                    game.kamikazes_killed += 1;
                    game.emit(EventType::KamikazeKilled);
                }
                break;
            }
        }
    }
}

fn check_bullet_wall(game: &mut HeadlessGame) {
    let wall_y_top = WALL_Y - BULLET_H;
    for bi in 0..game.bullets.len() {
        if game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        if game.bullets[bi].y < wall_y_top {
            continue;
        }
        for wi in 0..game.walls.len() {
            let b = &game.bullets[bi];
            let w = &game.walls[wi];
            if b.x < w.x + w.width
                && b.x + BULLET_W > w.x
                && b.y < w.y + w.height
                && b.y + BULLET_H > w.y
            {
                game.bullets[bi].removed = true;
                game.walls[wi].hit_count += 1;
                break;
            }
        }
    }
}

fn check_bullet_enemy(game: &mut HeadlessGame) {
    for bi in 0..game.bullets.len() {
        if game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        for ei in 0..game.enemies.len() {
            if game.enemies[ei].hits >= ENEMY_HITS_TO_DESTROY {
                continue;
            }
            let b = &game.bullets[bi];
            let e = &game.enemies[ei];
            if b.x < e.x + e.width
                && b.x + BULLET_W > e.x
                && b.y < e.y + e.height
                && b.y + BULLET_H > e.y
            {
                game.enemies[ei].hits += 1;
                game.bullets[bi].removed = true;
                if game.enemies[ei].hits >= ENEMY_HITS_TO_DESTROY {
                    game.score += 10 + 30; // kill + explosion
                    game.enemies_killed += 1;
                    game.emit(EventType::EnemyKilled);
                }
                break;
            }
        }
    }
}

fn check_bullet_missile(game: &mut HeadlessGame) {
    for bi in 0..game.bullets.len() {
        if game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        for mi in 0..game.missiles.len() {
            if game.missiles[mi].removed {
                continue;
            }
            let b = &game.bullets[bi];
            let m = &game.missiles[mi];
            let dx = b.x - m.x;
            let dy = b.y - m.y;
            let radius = m.width / 2.0 + 5.0;
            if dx * dx + dy * dy < radius * radius {
                game.bullets[bi].removed = true;
                game.missiles[mi].removed = true;
                game.homing_missile_hits += 1;
                game.score += 500;
                game.missiles_shot += 1;
                game.emit(EventType::MissileShotDown);
                if game.homing_missile_hits % 4 == 0 {
                    game.score += 500;
                    game.bonus_grants += 1;
                    game.emit(EventType::BonusEarned);
                    if game.bonus_grants >= BONUS2LIVES {
                        game.player_lives = (game.player_lives + 1).min(PLAYER_LIVES);
                        game.bonus_grants = 0;
                        game.emit(EventType::LifeGranted);
                    }
                }
                break;
            }
        }
    }
}

fn check_bullet_monster(game: &mut HeadlessGame) {
    let monster = match &game.monster {
        Some(m) if !m.hit => m,
        _ => return,
    };
    let monster_x = monster.x;
    let monster_y = monster.y;
    let monster_width = monster.width;
    let monster_height = monster.height;

    for bi in 0..game.bullets.len() {
        if game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        let b = &game.bullets[bi];
        let has_enemy_in_path = game.enemies.iter().any(|e| {
            b.x >= e.x && b.x <= e.x + e.width && b.y > e.y && b.y < monster_y + monster_height
        });
        if has_enemy_in_path {
            continue;
        }
        let b = &game.bullets[bi];
        if b.x < monster_x + monster_width
            && b.x + BULLET_W > monster_x
            && b.y < monster_y + monster_height
            && b.y + BULLET_H > monster_y
        {
            game.bullets[bi].removed = true;
            if let Some(ref mut m) = game.monster {
                m.hit = true;
                m.hit_time = game.game_time;
            }
            game.score += 500;
            game.restore_walls();
            game.emit(EventType::MonsterKilled);
            break;
        }
    }
}

fn check_bullet_monster2(game: &mut HeadlessGame) {
    let m2 = match &game.monster2 {
        Some(m) if !m.is_disappeared && !m.hit => m,
        _ => return,
    };
    let m2_x = m2.x;
    let m2_y = m2.y;
    let m2_width = m2.width;
    let m2_height = m2.height;

    for bi in 0..game.bullets.len() {
        if game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        let b = &game.bullets[bi];
        if b.x < m2_x + m2_width
            && b.x + BULLET_W > m2_x
            && b.y < m2_y + m2_height
            && b.y + BULLET_H > m2_y
        {
            game.bullets[bi].removed = true;
            if let Some(ref mut m) = game.monster2 {
                m.hit = true;
                m.hit_time = game.game_time;
            }
            game.score += 1500;
            game.restore_walls();
            game.emit(EventType::Monster2Killed);
            break;
        }
    }
}

fn check_enemy_bullet_player(game: &mut HeadlessGame) {
    if game.is_player_hit {
        return;
    }
    for bi in 0..game.bullets.len() {
        if !game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        let b = &game.bullets[bi];
        if b.x < game.player_x + PLAYER_WIDTH
            && b.x + BULLET_W > game.player_x
            && b.y < game.player_y + PLAYER_HEIGHT
            && b.y + BULLET_H > game.player_y
        {
            game.handle_player_hit();
            for bb in game.bullets.iter_mut() {
                if bb.is_enemy {
                    bb.removed = true;
                }
            }
            return;
        }
    }
}

fn check_kamikaze_player(game: &mut HeadlessGame) {
    if game.is_player_hit {
        return;
    }
    for ki in 0..game.kamikazes.len() {
        if game.kamikazes[ki].removed {
            continue;
        }
        // Check wall collision first
        let mut hit_wall = false;
        for wi in 0..game.walls.len() {
            let k = &game.kamikazes[ki];
            let w = &game.walls[wi];
            if k.x < w.x + w.width
                && k.x + k.width > w.x
                && k.y < w.y + w.height
                && k.y + k.height > w.y
            {
                game.score += 30;
                game.kamikazes[ki].removed = true;
                hit_wall = true;
                break;
            }
        }
        if hit_wall {
            continue;
        }
        let k = &game.kamikazes[ki];
        if k.x < game.player_x + PLAYER_WIDTH
            && k.x + k.width > game.player_x
            && k.y < game.player_y + PLAYER_HEIGHT
            && k.y + k.height > game.player_y
        {
            game.kamikazes[ki].removed = true;
            game.handle_player_hit();
        }
    }
}

fn check_missile_player(game: &mut HeadlessGame) {
    if game.is_player_hit {
        return;
    }
    let player_cx = game.player_x + PLAYER_WIDTH / 2.0;
    let player_cy = game.player_y + PLAYER_HEIGHT / 2.0;
    let hit_radius = PLAYER_WIDTH / 2.0 + MISSILE_WIDTH / 4.0;
    let hit_radius_sq = hit_radius * hit_radius;

    for mi in 0..game.missiles.len() {
        if game.missiles[mi].removed {
            continue;
        }
        let m = &game.missiles[mi];
        let dx = m.x - player_cx;
        let dy = m.y - player_cy;
        if dx * dx + dy * dy < hit_radius_sq {
            game.handle_player_hit();
            for mm in game.missiles.iter_mut() {
                mm.removed = true;
            }
            return;
        }
    }
}

fn check_missile_wall(game: &mut HeadlessGame) {
    for mi in 0..game.missiles.len() {
        if game.missiles[mi].removed {
            continue;
        }
        for wi in 0..game.walls.len() {
            let m = &game.missiles[mi];
            let w = &game.walls[wi];
            if m.x >= w.x
                && m.x <= w.x + w.width
                && m.y >= w.y
                && m.y <= w.y + w.height
            {
                game.missiles[mi].removed = true;
                game.walls[wi].missile_hits += 1;
                break;
            }
        }
    }
}

fn check_enemy_bullet_wall(game: &mut HeadlessGame) {
    let wall_y_top = WALL_Y;
    for bi in 0..game.bullets.len() {
        if !game.bullets[bi].is_enemy || game.bullets[bi].removed {
            continue;
        }
        if game.bullets[bi].y < wall_y_top {
            continue;
        }
        for wi in 0..game.walls.len() {
            let b = &game.bullets[bi];
            let w = &game.walls[wi];
            if b.x >= w.x
                && b.x <= w.x + w.width
                && b.y >= w.y
                && b.y <= w.y + w.height
            {
                game.bullets[bi].removed = true;
                game.walls[wi].hit_count += 1;
                break;
            }
        }
    }
}

fn remove_destroyed_walls(game: &mut HeadlessGame) {
    let before = game.walls.len();
    game.walls.retain(|w| {
        w.hit_count < WALL_MAX_HITS_TOTAL && w.missile_hits < WALL_MAX_MISSILE_HITS
    });
    let removed = before - game.walls.len();
    for _ in 0..removed {
        game.emit(EventType::WallDestroyed);
    }
}
