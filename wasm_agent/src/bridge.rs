use wasm_bindgen::prelude::*;
use game_sim_core::game::{HeadlessGame, RenderState};
use game_sim_core::entities::RenderEvent;

use crate::agent::AgentStats;

/// Convert game state + agent info into a JsValue object for the browser renderer.
pub fn game_state_to_js(game: &HeadlessGame, action: u8, stats: &AgentStats) -> JsValue {
    let obj = js_sys::Object::new();

    // Player
    set_f64(&obj, "playerX", game.player_x);
    set_f64(&obj, "playerY", game.player_y);
    set_i32(&obj, "playerLives", game.player_lives);
    set_bool(&obj, "isPlayerHit", game.is_player_hit);

    // Game state
    set_i32(&obj, "score", game.score);
    set_i32(&obj, "level", game.current_level);
    set_bool(&obj, "gameOver", game.game_over);
    set_i32(&obj, "action", action as i32);

    // Counts
    set_i32(&obj, "enemyCount", game.enemies.len() as i32);
    set_i32(&obj, "bulletCount", game.bullets.len() as i32);
    set_i32(&obj, "missileCount", game.missiles.len() as i32);
    set_i32(&obj, "kamikazeCount", game.kamikazes.len() as i32);
    set_i32(&obj, "wallCount", game.walls.len() as i32);
    set_bool(&obj, "hasMonster", game.monster.is_some());
    set_bool(&obj, "hasMonster2", game.monster2.is_some());

    // Enemies array (positions for rendering)
    let enemies_arr = js_sys::Array::new();
    for e in &game.enemies {
        let eobj = js_sys::Object::new();
        set_f64(&eobj, "x", e.x);
        set_f64(&eobj, "y", e.y);
        set_f64(&eobj, "w", e.width);
        set_f64(&eobj, "h", e.height);
        set_i32(&eobj, "hits", e.hits);
        enemies_arr.push(&eobj);
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("enemies"), &enemies_arr).ok();

    // Bullets array
    let bullets_arr = js_sys::Array::new();
    for b in &game.bullets {
        if b.removed { continue; }
        let bobj = js_sys::Object::new();
        set_f64(&bobj, "x", b.x);
        set_f64(&bobj, "y", b.y);
        set_bool(&bobj, "isEnemy", b.is_enemy);
        bullets_arr.push(&bobj);
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("bullets"), &bullets_arr).ok();

    // Missiles array
    let missiles_arr = js_sys::Array::new();
    for m in &game.missiles {
        if m.removed { continue; }
        let mobj = js_sys::Object::new();
        set_f64(&mobj, "x", m.x);
        set_f64(&mobj, "y", m.y);
        set_f64(&mobj, "angle", m.angle);
        set_f64(&mobj, "w", m.width);
        set_f64(&mobj, "h", m.height);
        missiles_arr.push(&mobj);
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("missiles"), &missiles_arr).ok();

    // Kamikazes array
    let kamikazes_arr = js_sys::Array::new();
    for k in &game.kamikazes {
        if k.removed { continue; }
        let kobj = js_sys::Object::new();
        set_f64(&kobj, "x", k.x);
        set_f64(&kobj, "y", k.y);
        set_f64(&kobj, "w", k.width);
        set_f64(&kobj, "h", k.height);
        set_f64(&kobj, "angle", k.angle);
        kamikazes_arr.push(&kobj);
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("kamikazes"), &kamikazes_arr).ok();

    // Walls array
    let walls_arr = js_sys::Array::new();
    for w in &game.walls {
        let wobj = js_sys::Object::new();
        set_f64(&wobj, "x", w.x);
        set_f64(&wobj, "y", w.y);
        set_f64(&wobj, "w", w.width);
        set_f64(&wobj, "h", w.height);
        set_i32(&wobj, "hitCount", w.hit_count);
        walls_arr.push(&wobj);
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("walls"), &walls_arr).ok();

    // Monster
    if let Some(ref monster) = game.monster {
        if !monster.hit {
            let mobj = js_sys::Object::new();
            set_f64(&mobj, "x", monster.x);
            set_f64(&mobj, "y", monster.y);
            set_f64(&mobj, "w", monster.width);
            set_f64(&mobj, "h", monster.height);
            js_sys::Reflect::set(&obj, &JsValue::from_str("monster"), &mobj).ok();
        }
    }

    // Monster2
    if let Some(ref m2) = game.monster2 {
        if !m2.hit && !m2.is_disappeared {
            let mobj = js_sys::Object::new();
            set_f64(&mobj, "x", m2.x);
            set_f64(&mobj, "y", m2.y);
            set_f64(&mobj, "w", m2.width);
            set_f64(&mobj, "h", m2.height);
            js_sys::Reflect::set(&obj, &JsValue::from_str("monster2"), &mobj).ok();
        }
    }

    // Events
    let events_arr = js_sys::Array::new();
    for ev in &game.events {
        events_arr.push(&JsValue::from_str(ev.event_type.as_str()));
    }
    js_sys::Reflect::set(&obj, &JsValue::from_str("events"), &events_arr).ok();

    // Agent stats
    set_i32(&obj, "totalSteps", stats.total_steps as i32);
    set_i32(&obj, "episodes", stats.episodes as i32);
    set_i32(&obj, "bestScore", stats.best_score);
    set_i32(&obj, "bestLevel", stats.best_level);
    set_f64(&obj, "episodeReward", stats.episode_reward as f64);
    set_f64(&obj, "avgReward", stats.avg_reward as f64);
    set_i32(&obj, "updates", stats.updates as i32);
    set_bool(&obj, "learningEnabled", stats.learning_enabled);

    obj.into()
}

/// Convert agent stats to a standalone JsValue.
pub fn stats_to_js(stats: &AgentStats) -> JsValue {
    let obj = js_sys::Object::new();
    set_i32(&obj, "totalSteps", stats.total_steps as i32);
    set_i32(&obj, "episodes", stats.episodes as i32);
    set_i32(&obj, "bestScore", stats.best_score);
    set_i32(&obj, "bestLevel", stats.best_level);
    set_i32(&obj, "currentScore", stats.current_score);
    set_i32(&obj, "currentLevel", stats.current_level);
    set_i32(&obj, "currentLives", stats.current_lives);
    set_f64(&obj, "episodeReward", stats.episode_reward as f64);
    set_f64(&obj, "avgReward", stats.avg_reward as f64);
    set_i32(&obj, "updates", stats.updates as i32);
    set_bool(&obj, "learningEnabled", stats.learning_enabled);
    set_i32(&obj, "enemiesKilled", stats.enemies_killed as i32);
    set_i32(&obj, "kamikazesKilled", stats.kamikazes_killed as i32);
    set_i32(&obj, "missilesShot", stats.missiles_shot as i32);
    set_i32(&obj, "monstersKilled", stats.monsters_killed as i32);
    set_i32(&obj, "monsters2Killed", stats.monsters2_killed as i32);

    if let Some(ref upd) = stats.last_update {
        let uobj = js_sys::Object::new();
        set_f64(&uobj, "policyLoss", upd.policy_loss as f64);
        set_f64(&uobj, "valueLoss", upd.value_loss as f64);
        set_f64(&uobj, "entropy", upd.entropy as f64);
        set_f64(&uobj, "totalLoss", upd.total_loss as f64);
        set_f64(&uobj, "approxKl", upd.approx_kl as f64);
        js_sys::Reflect::set(&obj, &JsValue::from_str("lastUpdate"), &uobj).ok();
    }

    obj.into()
}

// Helpers for setting JS object properties
fn set_f64(obj: &js_sys::Object, key: &str, val: f64) {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(val)).ok();
}

fn set_i32(obj: &js_sys::Object, key: &str, val: i32) {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(val as f64)).ok();
}

fn set_bool(obj: &js_sys::Object, key: &str, val: bool) {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_bool(val)).ok();
}

fn set_str(obj: &js_sys::Object, key: &str, val: &str) {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_str(val)).ok();
}

// ---------------------------------------------------------------------------
// render_state_to_js — serializes a full RenderState for the browser renderer
// ---------------------------------------------------------------------------

pub fn render_state_to_js(state: &RenderState) -> JsValue {
    let obj = js_sys::Object::new();

    // Player
    let p = js_sys::Object::new();
    set_f64(&p, "x", state.player.x);
    set_f64(&p, "y", state.player.y);
    set_f64(&p, "width", state.player.width);
    set_f64(&p, "height", state.player.height);
    set_bool(&p, "isHit", state.player.is_hit);
    js_sys::Reflect::set(&obj, &"player".into(), &p).ok();

    // Enemies array
    let enemies_arr = js_sys::Array::new();
    for e in &state.enemies {
        let eo = js_sys::Object::new();
        set_f64(&eo, "x", e.x);
        set_f64(&eo, "y", e.y);
        set_f64(&eo, "width", e.width);
        set_f64(&eo, "height", e.height);
        set_i32(&eo, "hits", e.hits);
        set_i32(&eo, "row", e.row);
        enemies_arr.push(&eo);
    }
    js_sys::Reflect::set(&obj, &"enemies".into(), &enemies_arr).ok();

    // Bullets array
    let bullets_arr = js_sys::Array::new();
    for b in &state.bullets {
        let bo = js_sys::Object::new();
        set_f64(&bo, "x", b.x);
        set_f64(&bo, "y", b.y);
        set_bool(&bo, "isEnemy", b.is_enemy);
        set_f64(&bo, "dx", b.dx);
        set_f64(&bo, "dy", b.dy);
        set_bool(&bo, "isMonster2Bullet", b.is_monster2);
        bullets_arr.push(&bo);
    }
    js_sys::Reflect::set(&obj, &"bullets".into(), &bullets_arr).ok();

    // Kamikazes array
    let kamikazes_arr = js_sys::Array::new();
    for k in &state.kamikazes {
        let ko = js_sys::Object::new();
        set_f64(&ko, "x", k.x);
        set_f64(&ko, "y", k.y);
        set_f64(&ko, "width", k.width);
        set_f64(&ko, "height", k.height);
        set_f64(&ko, "angle", k.angle);
        kamikazes_arr.push(&ko);
    }
    js_sys::Reflect::set(&obj, &"kamikazes".into(), &kamikazes_arr).ok();

    // Missiles array
    let missiles_arr = js_sys::Array::new();
    for m in &state.missiles {
        let mo = js_sys::Object::new();
        set_f64(&mo, "x", m.x);
        set_f64(&mo, "y", m.y);
        set_f64(&mo, "width", m.width);
        set_f64(&mo, "height", m.height);
        set_f64(&mo, "angle", m.angle);
        missiles_arr.push(&mo);
    }
    js_sys::Reflect::set(&obj, &"missiles".into(), &missiles_arr).ok();

    // Walls array
    let walls_arr = js_sys::Array::new();
    for w in &state.walls {
        let wo = js_sys::Object::new();
        set_f64(&wo, "x", w.x);
        set_f64(&wo, "y", w.y);
        set_f64(&wo, "width", w.width);
        set_f64(&wo, "height", w.height);
        set_i32(&wo, "hitCount", w.hit_count);
        set_i32(&wo, "missileHits", w.missile_hits);
        walls_arr.push(&wo);
    }
    js_sys::Reflect::set(&obj, &"walls".into(), &walls_arr).ok();

    // Monster (optional)
    if let Some(ref m) = state.monster {
        let mo = js_sys::Object::new();
        set_f64(&mo, "x", m.x);
        set_f64(&mo, "y", m.y);
        set_f64(&mo, "width", m.width);
        set_f64(&mo, "height", m.height);
        set_bool(&mo, "isHit", m.is_hit);
        set_bool(&mo, "isSlaloming", m.is_slaloming);
        js_sys::Reflect::set(&obj, &"monster".into(), &mo).ok();
    }

    // Monster2 (optional)
    if let Some(ref m) = state.monster2 {
        let mo = js_sys::Object::new();
        set_f64(&mo, "x", m.x);
        set_f64(&mo, "y", m.y);
        set_f64(&mo, "width", m.width);
        set_f64(&mo, "height", m.height);
        set_f64(&mo, "dx", m.dx);
        set_f64(&mo, "dy", m.dy);
        set_bool(&mo, "isDisappeared", m.is_disappeared);
        js_sys::Reflect::set(&obj, &"monster2".into(), &mo).ok();
    }

    // Scalars
    set_i32(&obj, "score", state.score);
    set_i32(&obj, "level", state.level);
    set_i32(&obj, "lives", state.lives);
    set_bool(&obj, "gameOver", state.game_over);
    set_bool(&obj, "done", state.done);
    set_f64(&obj, "reward", state.reward as f64);

    // Events array
    let events_arr = js_sys::Array::new();
    for ev in &state.events {
        let evo = js_sys::Object::new();
        match ev {
            RenderEvent::EnemyHit { x, y } => {
                set_str(&evo, "type", "enemy_hit");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::EnemyKilled { x, y } => {
                set_str(&evo, "type", "enemy_killed");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::PlayerHit => {
                set_str(&evo, "type", "player_hit");
            }
            RenderEvent::PlayerFired { x, y } => {
                set_str(&evo, "type", "player_fired");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::MissileDestroyed { x, y } => {
                set_str(&evo, "type", "missile_destroyed");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::MissileBonus => {
                set_str(&evo, "type", "missile_bonus");
            }
            RenderEvent::WallHit { wall_index, x, y, from_player } => {
                set_str(&evo, "type", "wall_hit");
                set_i32(&evo, "wallIndex", *wall_index as i32);
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
                set_bool(&evo, "fromPlayer", *from_player);
            }
            RenderEvent::WallDestroyed { wall_index } => {
                set_str(&evo, "type", "wall_destroyed");
                set_i32(&evo, "wallIndex", *wall_index as i32);
            }
            RenderEvent::KamikazeSpawned { x, y } => {
                set_str(&evo, "type", "kamikaze_spawned");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::KamikazeKilled { x, y } => {
                set_str(&evo, "type", "kamikaze_killed");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::MonsterSpawned => {
                set_str(&evo, "type", "monster_spawned");
            }
            RenderEvent::MonsterHit { x, y } => {
                set_str(&evo, "type", "monster_hit");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::Monster2Spawned => {
                set_str(&evo, "type", "monster2_spawned");
            }
            RenderEvent::Monster2Disappeared => {
                set_str(&evo, "type", "monster2_disappeared");
            }
            RenderEvent::Monster2Reappeared { x, y } => {
                set_str(&evo, "type", "monster2_reappeared");
                set_f64(&evo, "x", *x);
                set_f64(&evo, "y", *y);
            }
            RenderEvent::LevelComplete { level } => {
                set_str(&evo, "type", "level_complete");
                set_i32(&evo, "level", *level);
            }
            RenderEvent::GameOver => {
                set_str(&evo, "type", "game_over");
            }
            RenderEvent::BonusLife => {
                set_str(&evo, "type", "bonus_life");
            }
            RenderEvent::ScoreChange { delta } => {
                set_str(&evo, "type", "score_change");
                set_i32(&evo, "delta", *delta);
            }
        }
        events_arr.push(&evo);
    }
    js_sys::Reflect::set(&obj, &"events".into(), &events_arr).ok();

    // Features (Float32Array for AI)
    let features = js_sys::Float32Array::from(&state.features[..]);
    js_sys::Reflect::set(&obj, &"features".into(), &features).ok();

    obj.into()
}
