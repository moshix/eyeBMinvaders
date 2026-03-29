use wasm_bindgen::prelude::*;
use game_sim_core::game::HeadlessGame;

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
