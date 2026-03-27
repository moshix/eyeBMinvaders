#!/usr/bin/env python3
"""
eyeBMinvaders Headless Training Script
=======================================
Runs the game headlessly for N iterations using Deep Q-Learning (DQN)
to train a neural network that learns optimal play strategy.

All game mechanics are faithfully ported from game.js.

Usage:
    python train.py                    # Train for 1,000,000 games
    python train.py --episodes 10000   # Train for 10,000 games
    python train.py --resume model.pt  # Resume from checkpoint

Requirements:
    pip install torch numpy
"""

import math
import random
import time
import json
import argparse
import os
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np

# Try to import Rust game simulation
try:
    from game_sim import BatchedGames as RustBatchedGames
    HAS_RUST_SIM = True
except ImportError:
    HAS_RUST_SIM = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
    torch.set_float32_matmul_precision('high')
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found. Install with: pip install torch")
    print("Running in random-agent mode for testing.\n")


# =============================================================================
# Game Constants (from game.js)
# =============================================================================
GAME_WIDTH = 1024
GAME_HEIGHT = 576

PLAYER_SPEED = 300          # pixels/sec
PLAYER_WIDTH = 48
PLAYER_HEIGHT = 48
PLAYER_LIVES = 6
FIRE_RATE = 0.16            # seconds between player shots
BULLET_SPEED = 300          # pixels/sec
BULLET_W = 3.4
BULLET_H = 5.9

ENEMY_SPEED = 50            # base pixels/sec
ENEMY_WIDTH = 43
ENEMY_HEIGHT = 43
ENEMY_PADDING = 16
ENEMY_HITS_TO_DESTROY = 2
ENEMY_ROWS = 5
ENEMY_BULLET_SPEED = BULLET_SPEED / 3

BASE_ENEMY_FIRE_RATE = 0.85   # seconds
ENEMY_FIRE_RATE_INCREASE = 0.10

MISSILE_SPEED = 170
MISSILE_WIDTH = 57
MISSILE_HEIGHT = 57
MIN_MISSILE_INTERVAL = 3200   # ms
MAX_MISSILE_INTERVAL = 7200   # ms

KAMIKAZE_SPEED = 170
KAMIKAZE_FIRE_RATE = 900      # ms
KAMIKAZE_MIN_TIME = 6000      # ms
KAMIKAZE_MAX_TIME = 11000     # ms
KAMIKAZE_AGGRESSIVE_TIME = 4000
KAMIKAZE_VERY_AGGRESSIVE_TIME = 2200
KAMIKAZE_AGGRESSIVE_THRESHOLD = 26
KAMIKAZE_VERY_AGGRESSIVE_THRESHOLD = 11
KAMIKAZE_HITS_TO_DESTROY = 2

MONSTER_SPEED = 175
MONSTER_WIDTH = 56
MONSTER_HEIGHT = 56
MONSTER_INTERVAL = 6000       # ms
MONSTER_HIT_DURATION = 700    # ms
MONSTER_SLALOM_AMPLITUDE = 350
MONSTER_VERTICAL_SPEED = 60
MONSTER_SLALOM_FIRE_RATE = 1800  # ms
MONSTER_SLALOM_THRESHOLD = 19    # KAMIKAZE_AGGRESSIVE_THRESHOLD - 7

MONSTER2_WIDTH = 56
MONSTER2_HEIGHT = 56
MONSTER2_SPEED = 220
MONSTER2_INTERVAL = 10000     # ms
MONSTER2_VERTICAL_SPEED = 40
MONSTER2_SPIRAL_RADIUS = 100
MONSTER2_SPIRAL_SPEED = 3

WALL_HITS_FROM_BELOW = 3
WALL_MAX_HITS_TOTAL = 11
WALL_MAX_MISSILE_HITS = 4

PLAYER_HIT_ANIMATION_DURATION = 750   # ms

BONUS2LIVES = 5  # every 5 bonuses, player gets one life

# Wall positions
WALL_Y = GAME_HEIGHT - 75
WALL_WIDTH = 58
WALL_HEIGHT = 23
INITIAL_WALL_XS = [
    GAME_WIDTH * 1 / 5 - 29,
    GAME_WIDTH * 2 / 5 - 29,
    GAME_WIDTH * 3 / 5 - 29,
    GAME_WIDTH * 4 / 5 - 29,
]


# =============================================================================
# Game Event Types
# =============================================================================
class EventType:
    ENEMY_KILLED = "enemy_killed"
    KAMIKAZE_KILLED = "kamikaze_killed"
    MISSILE_SHOT_DOWN = "missile_shot_down"
    MONSTER_KILLED = "monster_killed"
    MONSTER2_KILLED = "monster2_killed"
    PLAYER_HIT = "player_hit"
    LEVEL_COMPLETE = "level_complete"
    GAME_OVER = "game_over"
    BONUS_EARNED = "bonus_earned"
    LIFE_GRANTED = "life_granted"
    WALL_DESTROYED = "wall_destroyed"
    PLAYER_SHOT = "player_shot"
    KAMIKAZE_SPAWNED = "kamikaze_spawned"
    MONSTER_SPAWNED = "monster_spawned"
    MISSILE_LAUNCHED = "missile_launched"


# =============================================================================
# Data classes for game objects
# =============================================================================
@dataclass
class Bullet:
    x: float
    y: float
    is_enemy: bool
    dx: float = 0.0  # for monster2 spread bullets
    dy: float = 0.0
    has_direction: bool = False  # True for monster2 bullets with dx/dy
    removed: bool = False


@dataclass
class Enemy:
    x: float
    y: float
    width: float = ENEMY_WIDTH
    height: float = ENEMY_HEIGHT
    hits: int = 0


@dataclass
class Kamikaze:
    x: float
    y: float
    width: float = ENEMY_WIDTH
    height: float = ENEMY_HEIGHT
    angle: float = 0.0
    time: float = 0.0
    hits: int = 0
    last_fire_time: float = 0.0
    removed: bool = False


@dataclass
class Missile:
    x: float
    y: float
    angle: float = 0.0
    width: float = MISSILE_WIDTH
    height: float = MISSILE_HEIGHT
    time: float = 0.0
    from_monster: bool = False
    removed: bool = False


@dataclass
class Monster:
    x: float
    y: float
    width: float = MONSTER_WIDTH
    height: float = MONSTER_HEIGHT
    hit: bool = False
    hit_time: float = 0.0
    has_shot: bool = False
    slalom_time: float = 0.0
    is_slaloming: bool = False
    last_fire_time: float = 0.0
    direction: int = 1  # 1 right, -1 left


@dataclass
class Monster2:
    x: float
    y: float
    width: float = MONSTER2_WIDTH
    height: float = MONSTER2_HEIGHT
    spiral_angle: float = 0.0
    center_x: float = GAME_WIDTH / 2
    hit: bool = False
    hit_time: float = 0.0
    is_disappeared: bool = False
    disappear_time: float = 0.0
    return_delay: float = 0.0
    last_fire_time: float = 0.0
    # Pattern-specific state
    zigzag_dir: int = 0
    zigzag_amplitude: float = 0.0
    zigzag_phase: float = 0.0
    dx_val: float = 0.0
    dy_val: float = 0.0
    last_direction_change: float = 0.0
    direction_change_interval: float = 0.0
    wave_start_x: float = 0.0
    next_teleport_time: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    next_move_time: float = 0.0


@dataclass
class Wall:
    x: float
    y: float = WALL_Y
    width: float = WALL_WIDTH
    height: float = WALL_HEIGHT
    hit_count: int = 0
    missile_hits: int = 0


# =============================================================================
# Headless Game Engine
# =============================================================================
class HeadlessGame:
    """Faithful port of game.js mechanics for headless simulation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to initial game state."""
        self.game_time = 0.0  # ms
        self.score = 0
        self.current_level = 1
        self.game_over = False
        self.game_paused = False

        # Player
        self.player_x = GAME_WIDTH / 2 - 37
        self.player_y = GAME_HEIGHT - 30
        self.player_lives = PLAYER_LIVES
        self.is_player_hit = False
        self.while_player_hit = False
        self.player_hit_timer = 0.0

        # Enemies
        self.enemies: List[Enemy] = []
        self.enemy_speed = 0.54
        self.enemy_direction = 1
        self.current_enemy_fire_rate = BASE_ENEMY_FIRE_RATE
        self.last_enemy_fire_time = 0.0

        # Bullets
        self.bullets: List[Bullet] = []
        self.last_fire_time = 0.0

        # Missiles
        self.missiles: List[Missile] = []
        self.next_missile_time = 0.0
        self.homing_missile_hits = 0
        self.bonus_grants = 0

        # Kamikazes
        self.kamikazes: List[Kamikaze] = []
        self.next_kamikaze_time = self._random_kamikaze_time()

        # Monsters
        self.monster: Optional[Monster] = None
        self.last_monster_time = 0.0
        self.monster2: Optional[Monster2] = None
        self.last_monster2_time = 0.0

        # Walls
        self.walls: List[Wall] = [Wall(x=wx) for wx in INITIAL_WALL_XS]

        # Events log for this step
        self.events: List[dict] = []

        # Stats
        self.total_steps = 0
        self.enemies_killed = 0
        self.kamikazes_killed = 0
        self.missiles_shot = 0
        self.times_hit = 0

        # Create initial enemies
        self._create_enemies()

        return self._get_state()

    def _random_kamikaze_time(self):
        return self.game_time + random.uniform(KAMIKAZE_MIN_TIME, KAMIKAZE_MAX_TIME)

    def _create_enemies(self):
        """Create enemy grid (matching game.js createEnemies)."""
        cols = min(12, max(4, (GAME_WIDTH - 60) // (ENEMY_WIDTH + ENEMY_PADDING)))
        max_offset_top = 35
        wall_y = WALL_Y
        ideal_gap = GAME_HEIGHT * 0.3
        bottom_row_y = wall_y - ideal_gap
        total_height = ENEMY_ROWS * (ENEMY_HEIGHT + ENEMY_PADDING)
        offset_top = max(max_offset_top, bottom_row_y - total_height)
        offset_left = (GAME_WIDTH - cols * (ENEMY_WIDTH + ENEMY_PADDING)) / 2

        for r in range(ENEMY_ROWS):
            for c in range(cols):
                self.enemies.append(Enemy(
                    x=c * (ENEMY_WIDTH + ENEMY_PADDING) + offset_left,
                    y=r * (ENEMY_HEIGHT + ENEMY_PADDING) + offset_top,
                ))

    def _restore_walls(self):
        self.walls = [Wall(x=wx) for wx in INITIAL_WALL_XS]

    def _emit(self, event_type, **kwargs):
        event = {"type": event_type, "time": self.game_time, "score": self.score, **kwargs}
        self.events.append(event)

    # =========================================================================
    # Step: one game tick
    # =========================================================================
    def step(self, action):
        """
        Execute one game tick (~16ms, i.e. 60fps equivalent).

        Actions:
            0 = stay
            1 = left
            2 = right
            3 = shoot
            4 = left + shoot
            5 = right + shoot

        Returns: (state, reward, done, info)
        """
        self.events = []
        dt_ms = 33.333  # ~30fps (2x speedup, still accurate enough)
        dt = dt_ms / 1000.0  # seconds
        self.game_time += dt_ms
        self.total_steps += 1
        old_score = self.score
        old_lives = self.player_lives

        if self.game_over:
            return self._get_state(), 0.0, True, {"events": self.events}

        # Parse action
        move_left = action in (1, 4)
        move_right = action in (2, 5)
        shoot = action in (3, 4, 5)

        # --- Player hit animation ---
        if self.is_player_hit:
            self.while_player_hit = True
            if self.game_time - self.player_hit_timer > PLAYER_HIT_ANIMATION_DURATION:
                self.is_player_hit = False
                self.while_player_hit = False
        else:
            self.while_player_hit = False

        # --- Player movement ---
        if not self.is_player_hit:
            if move_left and self.player_x > 0:
                self.player_x -= PLAYER_SPEED * dt
                self.player_x = max(0, self.player_x)
            if move_right and self.player_x < GAME_WIDTH - PLAYER_WIDTH:
                self.player_x += PLAYER_SPEED * dt
                self.player_x = min(GAME_WIDTH - PLAYER_WIDTH, self.player_x)

        # --- Player shooting ---
        if shoot and not self.while_player_hit:
            if self.game_time - self.last_fire_time >= FIRE_RATE * 1000:
                self.bullets.append(Bullet(
                    x=self.player_x + PLAYER_WIDTH / 2 - BULLET_W / 2,
                    y=self.player_y,
                    is_enemy=False,
                ))
                self.last_fire_time = self.game_time
                self._emit(EventType.PLAYER_SHOT)

        # --- Game logic (spawning) ---
        self._handle_monster_creation()
        self._handle_monster2_creation()
        self._handle_kamikaze_creation()

        # --- Movement ---
        self._move_bullets(dt)
        self._move_enemies(dt)
        self._move_kamikazes(dt)
        self._move_missiles(dt)
        self._move_monster(dt)
        self._move_monster2(dt)

        # --- Enemy shooting ---
        self._handle_enemy_shooting()
        self._handle_missile_launching()

        # --- Collision detection ---
        self._detect_collisions()

        # --- Check victory ---
        if len(self.enemies) == 0 and not self.game_over:
            self._victory()

        # --- Reward calculation ---
        reward = 0.0
        # Score-based reward
        reward += (self.score - old_score) * 0.01
        # Penalty for losing a life
        if self.player_lives < old_lives:
            reward -= 5.0
        # Big penalty for game over
        if self.game_over:
            reward -= 20.0
        # Penalty for destroying a wall
        wall_destroyed_count = sum(1 for e in self.events if e.get("type") == EventType.WALL_DESTROYED)
        if wall_destroyed_count > 0:
            reward -= 2.0 * wall_destroyed_count
        # Progressive survival bonus: scales with level
        reward += 0.01 * self.current_level
        # Extra reward for killing kamikazes and shooting down missiles
        kamikazes_killed = sum(1 for e in self.events if e.get("type") == EventType.KAMIKAZE_KILLED)
        missiles_shot = sum(1 for e in self.events if e.get("type") == EventType.MISSILE_SHOT_DOWN)
        reward += kamikazes_killed * 1.5
        reward += missiles_shot * 2.0
        # Level completion bonus
        if len(self.enemies) == 0 and not self.game_over:
            reward += 5.0

        state = self._get_state()
        info = {
            "events": self.events,
            "score": self.score,
            "level": self.current_level,
            "lives": self.player_lives,
            "enemies_left": len(self.enemies),
            "steps": self.total_steps,
        }

        return state, reward, self.game_over, info

    # =========================================================================
    # State representation for the neural network
    # =========================================================================
    def _get_state(self):
        """
        Returns a 45-element feature vector representing game state.
        """
        player_cx = self.player_x + PLAYER_WIDTH / 2
        player_cy = self.player_y + PLAYER_HEIGHT / 2

        def nx(v): return v / GAME_WIDTH
        def ny(v): return v / GAME_HEIGHT

        f = [0.0] * 45

        # [0] Player position
        f[0] = nx(player_cx)
        # [1] Player lives
        f[1] = self.player_lives / PLAYER_LIVES
        # [2] Level
        f[2] = min(self.current_level, 10) / 10.0
        # [3] Enemy count
        f[3] = min(len(self.enemies), 60) / 60.0

        # [4-5] Nearest enemy
        if self.enemies:
            nearest = min(self.enemies, key=lambda e: abs(e.x + e.width / 2 - player_cx))
            f[4] = nx(nearest.x + nearest.width / 2 - player_cx)
            f[5] = ny(nearest.y + nearest.height / 2 - player_cy)
        else:
            f[5] = -1.0

        # [6-7] Lowest enemy
        if self.enemies:
            lowest = max(self.enemies, key=lambda e: e.y)
            f[6] = nx(lowest.x + lowest.width / 2 - player_cx)
            f[7] = ny(lowest.y)

        # Enemy bullets — sort by distance
        enemy_bullets = [b for b in self.bullets if b.is_enemy]
        sorted_bullets = sorted(enemy_bullets,
            key=lambda b: (b.x - player_cx) ** 2 + (b.y - player_cy) ** 2)

        # [8-12] Nearest bullet + velocity
        if sorted_bullets:
            b = sorted_bullets[0]
            f[8] = nx(b.x - player_cx)
            f[9] = ny(b.y - player_cy)
            f[10] = len(enemy_bullets) / 10.0
            if hasattr(b, 'has_direction') and b.has_direction:
                f[11] = b.dx / ENEMY_BULLET_SPEED
                f[12] = b.dy / ENEMY_BULLET_SPEED
            else:
                f[11] = 0.0
                f[12] = 1.0
        else:
            f[9] = -1.0

        # Missiles — sort by distance
        sorted_missiles = sorted(self.missiles,
            key=lambda m: (m.x - player_cx) ** 2 + (m.y - player_cy) ** 2)

        # [13-17] Nearest missile + velocity
        if sorted_missiles:
            m = sorted_missiles[0]
            f[13] = nx(m.x - player_cx)
            f[14] = ny(m.y - player_cy)
            f[15] = len(self.missiles) / 5.0
            f[16] = math.cos(m.angle)
            f[17] = math.sin(m.angle)
        else:
            f[14] = -1.0

        # Kamikazes — sort by distance
        sorted_kamikazes = sorted(self.kamikazes,
            key=lambda k: (k.x + k.width / 2 - player_cx) ** 2 + (k.y + k.height / 2 - player_cy) ** 2)

        # [18-22] Nearest kamikaze + velocity
        if sorted_kamikazes:
            k = sorted_kamikazes[0]
            f[18] = nx(k.x + k.width / 2 - player_cx)
            f[19] = ny(k.y + k.height / 2 - player_cy)
            f[20] = len(self.kamikazes) / 5.0
            f[21] = math.cos(k.angle)
            f[22] = math.sin(k.angle)
        else:
            f[19] = -1.0

        # [23-24] Monster info
        if self.monster and not self.monster.hit:
            f[23] = nx(self.monster.x + self.monster.width / 2 - player_cx)
            f[24] = ny(self.monster.y)
        else:
            f[24] = -1.0

        # [25-28] Monster2 info
        if hasattr(self, 'monster2') and self.monster2 and not self.monster2.hit and not getattr(self.monster2, 'is_disappeared', False):
            f[25] = nx(self.monster2.x + self.monster2.width / 2 - player_cx)
            f[26] = ny(self.monster2.y)
            f[27] = getattr(self.monster2, 'dx_val', 0.0) / MONSTER2_SPEED if hasattr(self.monster2, 'dx_val') else 0.0
            f[28] = getattr(self.monster2, 'dy_val', 0.0) / MONSTER2_SPEED if hasattr(self.monster2, 'dy_val') else 0.0
        else:
            f[26] = -1.0

        # [29] Invulnerability
        f[29] = 1.0 if self.is_player_hit else 0.0
        # [30] Walls remaining
        f[30] = len(self.walls) / 4.0

        # [31-33] Nearest wall
        if self.walls:
            nearest_w = min(self.walls, key=lambda w: abs(w.x + w.width / 2 - player_cx))
            f[31] = nx(nearest_w.x + nearest_w.width / 2 - player_cx)
            f[32] = ny(nearest_w.y - player_cy)
            f[33] = 1.0 - nearest_w.hit_count / WALL_MAX_HITS_TOTAL

        # [34-36] 2nd nearest bullet
        if len(sorted_bullets) >= 2:
            b2 = sorted_bullets[1]
            f[34] = nx(b2.x - player_cx)
            f[35] = ny(b2.y - player_cy)
            f[36] = (b2.dy / ENEMY_BULLET_SPEED) if hasattr(b2, 'has_direction') and b2.has_direction else 1.0
        else:
            f[35] = -1.0

        # [37-39] 2nd nearest missile
        if len(sorted_missiles) >= 2:
            m2 = sorted_missiles[1]
            f[37] = nx(m2.x - player_cx)
            f[38] = ny(m2.y - player_cy)
            f[39] = math.sin(m2.angle)
        else:
            f[38] = -1.0

        # [40-44] Danger heatmap: 5 columns
        col_width = GAME_WIDTH / 5.0
        for b in enemy_bullets:
            col = min(int(b.x / col_width), 4)
            f[40 + col] += 1.0
        for m in self.missiles:
            col = min(int(m.x / col_width), 4)
            f[40 + col] += 2.0
        for k in self.kamikazes:
            col = min(int((k.x + k.width / 2) / col_width), 4)
            f[40 + col] += 3.0
        for j in range(40, 45):
            f[j] = min(f[j] / 10.0, 1.0)

        return np.array(f, dtype=np.float32)

    @property
    def state_size(self):
        return 45

    @property
    def action_size(self):
        return 6  # stay, left, right, shoot, left+shoot, right+shoot

    # =========================================================================
    # Movement functions
    # =========================================================================
    def _move_bullets(self, dt):
        if self.while_player_hit:
            # Remove enemy bullets when player is hit
            self.bullets = [b for b in self.bullets if not b.is_enemy]
            return

        for b in self.bullets:
            if b.is_enemy:
                if b.has_direction:
                    b.x += b.dx * dt
                    b.y += b.dy * dt
                else:
                    b.y += ENEMY_BULLET_SPEED * dt
            else:
                b.y -= BULLET_SPEED * dt

        self.bullets = [b for b in self.bullets
                        if 0 < b.y < GAME_HEIGHT and 0 < b.x < GAME_WIDTH]

    def _move_enemies(self, dt):
        if not self.enemies:
            return

        dt_capped = min(dt, 0.1)
        speed = ENEMY_SPEED * self.enemy_speed * dt_capped
        needs_down = False

        for e in self.enemies:
            if e.hits >= ENEMY_HITS_TO_DESTROY:
                continue
            e.x += speed * self.enemy_direction
            if e.x + e.width > GAME_WIDTH or e.x < 0:
                needs_down = True
                e.x = max(0, min(GAME_WIDTH - e.width, e.x))

        if needs_down:
            self.enemy_direction *= -1
            wall_y = self.walls[0].y - 20 if self.walls else GAME_HEIGHT * 0.9
            for e in self.enemies:
                if e.hits < ENEMY_HITS_TO_DESTROY:
                    e.y += 20
                    if e.y + e.height >= wall_y:
                        self.game_over = True
                        self._emit(EventType.GAME_OVER, reason="enemies_reached_walls")
                        return

        self.enemies = [e for e in self.enemies if e.hits < ENEMY_HITS_TO_DESTROY]

    def _move_kamikazes(self, dt):
        player_cx = self.player_x + PLAYER_WIDTH / 2
        player_cy = self.player_y + PLAYER_HEIGHT / 2

        for k in self.kamikazes:
            if k.removed:
                continue
            k.time += dt

            # Shooting
            if self.game_time - k.last_fire_time >= KAMIKAZE_FIRE_RATE:
                self.bullets.append(Bullet(
                    x=k.x + k.width / 2,
                    y=k.y + k.height,
                    is_enemy=True,
                ))
                k.last_fire_time = self.game_time

            # Check if past player
            if k.y >= self.player_y:
                k.removed = True
                continue

            # Homing movement
            target_dx = player_cx - k.x
            target_dy = player_cy - k.y
            k.angle = math.atan2(target_dy, target_dx)

            curve = math.sin(k.time * 2) * 100
            k.x += math.cos(k.angle) * KAMIKAZE_SPEED * dt
            k.y += math.sin(k.angle) * KAMIKAZE_SPEED * dt
            k.x += math.cos(k.angle + math.pi / 2) * curve * dt

            # Wall collision
            for w in self.walls:
                if (w.hit_count < WALL_MAX_HITS_TOTAL and
                        w.missile_hits < WALL_MAX_MISSILE_HITS):
                    if (k.x + k.width > w.x and k.x < w.x + w.width and
                            k.y + k.height > w.y and k.y < w.y + w.height):
                        w.missile_hits += 1
                        self.score += 30
                        k.removed = True
                        break

            # Off-screen check
            if not k.removed and (k.x <= 0 or k.x >= GAME_WIDTH):
                k.removed = True

        self.kamikazes = [k for k in self.kamikazes if not k.removed]

    def _move_missiles(self, dt):
        player_cx = self.player_x + PLAYER_WIDTH / 2
        player_cy = self.player_y + PLAYER_HEIGHT / 2
        wall_row_y = self.walls[0].y - 50 if self.walls else GAME_HEIGHT * 0.85

        for m in self.missiles:
            m.time += dt
            if m.y < wall_row_y:
                dx = player_cx - m.x
                dy = player_cy - m.y
                m.angle = math.atan2(dy, dx)

            curve = math.sin(m.time * 2) * 100
            m.x += math.cos(m.angle) * MISSILE_SPEED * dt
            m.y += math.sin(m.angle) * MISSILE_SPEED * dt
            m.x += math.cos(m.angle + math.pi / 2) * curve * dt

        self.missiles = [m for m in self.missiles
                         if 0 < m.y < GAME_HEIGHT and 0 < m.x < GAME_WIDTH]

    def _move_monster(self, dt):
        if not self.monster:
            return
        m = self.monster
        if m.hit:
            if self.game_time - m.hit_time > MONSTER_HIT_DURATION:
                self.monster = None
                self.last_monster_time = self.game_time
            return

        if m.is_slaloming:
            m.slalom_time += dt
            center_x = GAME_WIDTH / 2
            m.x = center_x + math.sin(m.slalom_time * 1.2) * MONSTER_SLALOM_AMPLITUDE
            m.y += MONSTER_VERTICAL_SPEED * dt

            # Fire missiles
            if self.game_time - m.last_fire_time >= MONSTER_SLALOM_FIRE_RATE:
                self.missiles.append(Missile(
                    x=m.x + m.width / 2,
                    y=m.y + m.height,
                    angle=math.pi / 2,
                    width=44, height=44,
                    from_monster=True,
                ))
                m.last_fire_time = self.game_time
                self._emit(EventType.MISSILE_LAUNCHED, source="monster_slalom")

            wall_y = self.walls[0].y if self.walls else GAME_HEIGHT * 0.85
            if m.y >= wall_y - m.height - 20:
                self.monster = None
                self.last_monster_time = self.game_time
        else:
            m.x += MONSTER_SPEED * m.direction * dt
            is_on_screen = 0 <= m.x and m.x + m.width <= GAME_WIDTH
            if not m.has_shot and is_on_screen:
                for offset in [-m.width / 4, m.width / 4]:
                    self.missiles.append(Missile(
                        x=m.x + m.width / 2 + offset,
                        y=m.y + m.height,
                        angle=math.pi / 2,
                        width=44, height=44,
                        from_monster=True,
                    ))
                m.has_shot = True
                self._emit(EventType.MISSILE_LAUNCHED, source="monster")

            if ((m.direction == 1 and m.x > GAME_WIDTH + MONSTER_WIDTH) or
                    (m.direction == -1 and m.x < -MONSTER_WIDTH)):
                self.monster = None
                self.last_monster_time = self.game_time

    def _move_monster2(self, dt):
        m2 = self.monster2
        if not m2:
            return

        if m2.hit:
            if self.game_time - m2.hit_time > MONSTER_HIT_DURATION:
                m2.is_disappeared = True
                m2.disappear_time = self.game_time
                m2.return_delay = random.uniform(5000, 9000)
                m2.hit = False
            return

        if m2.is_disappeared:
            if self.game_time - m2.disappear_time > m2.return_delay:
                m2.is_disappeared = False
                m2.x = GAME_WIDTH / 2
                m2.y = -MONSTER2_HEIGHT
                m2.spiral_angle = 0
                m2.center_x = GAME_WIDTH / 2
            else:
                return

        # Pattern movement
        pattern = self._get_monster2_pattern()
        m2.y += MONSTER2_VERTICAL_SPEED * dt

        if pattern == 'spiral':
            m2.spiral_angle += MONSTER2_SPIRAL_SPEED * dt
            radius = MONSTER2_SPIRAL_RADIUS * min(1, m2.y / 200)
            m2.x = m2.center_x + math.cos(m2.spiral_angle) * radius
        elif pattern == 'zigzag':
            if m2.zigzag_dir == 0:
                m2.zigzag_dir = 1
                m2.zigzag_amplitude = GAME_WIDTH * 0.4
                m2.zigzag_phase = 0
            m2.zigzag_phase += dt * 1.5
            m2.x = GAME_WIDTH / 2 + math.sin(m2.zigzag_phase) * m2.zigzag_amplitude
            m2.y += MONSTER2_VERTICAL_SPEED * dt * 0.33
            m2.x = max(0, min(GAME_WIDTH - m2.width, m2.x))
        elif pattern == 'figure8':
            m2.spiral_angle += MONSTER2_SPIRAL_SPEED * dt
            m2.x = m2.center_x + math.cos(m2.spiral_angle) * MONSTER2_SPIRAL_RADIUS
            m2.y += math.sin(2 * m2.spiral_angle) * dt * 30
        elif pattern == 'bounce':
            if m2.dx_val == 0 and m2.dy_val == 0:
                m2.dx_val = (1 if random.random() > 0.5 else -1) * MONSTER2_SPEED
                m2.dy_val = (1 if random.random() > 0.5 else -1) * MONSTER2_SPEED * 0.7
                m2.last_direction_change = self.game_time
                m2.direction_change_interval = random.uniform(1500, 3000)
            m2.x += m2.dx_val * dt
            m2.y += m2.dy_val * dt
            if m2.x <= 0:
                m2.x = 0; m2.dx_val = abs(m2.dx_val)
            elif m2.x >= GAME_WIDTH - m2.width:
                m2.x = GAME_WIDTH - m2.width; m2.dx_val = -abs(m2.dx_val)
            if m2.y <= 0:
                m2.y = 0; m2.dy_val = abs(m2.dy_val)
            elif m2.y >= GAME_HEIGHT * 0.7:
                m2.y = GAME_HEIGHT * 0.7; m2.dy_val = -abs(m2.dy_val)
        elif pattern == 'wave':
            if m2.wave_start_x == 0:
                m2.wave_start_x = m2.x
            m2.x = m2.wave_start_x + math.sin(m2.y / 50) * (GAME_WIDTH / 4)
        elif pattern == 'chase':
            predicted_x = self.player_x  # simplified - no key prediction in headless
            chase_dx = predicted_x - m2.x
            chase_dy = self.player_y - m2.y - 200
            dist = math.sqrt(chase_dx ** 2 + chase_dy ** 2)
            if dist > 1:
                m2.x += (chase_dx / dist) * MONSTER2_SPEED * 1.2 * dt
                m2.y += (chase_dy / dist) * MONSTER2_SPEED * 0.7 * dt
        else:  # random / teleport
            if m2.next_move_time == 0 or self.game_time > m2.next_move_time:
                m2.target_x = random.random() * (GAME_WIDTH - m2.width)
                m2.target_y = min(random.random() * GAME_HEIGHT * 0.5, m2.y + 100)
                m2.next_move_time = self.game_time + 1000
            rdx = m2.target_x - m2.x
            rdy = m2.target_y - m2.y
            rdist = math.sqrt(rdx ** 2 + rdy ** 2)
            if rdist > 1:
                m2.x += (rdx / rdist) * MONSTER2_SPEED * dt
                m2.y += (rdy / rdist) * MONSTER2_SPEED * dt

        # Monster2 shoots spread bullets every 2.8s
        if self.game_time - m2.last_fire_time >= 2800:
            for i in range(-1, 2):
                spread_angle = math.pi / 2 + i * math.pi / 8
                self.bullets.append(Bullet(
                    x=m2.x + m2.width / 2,
                    y=m2.y + m2.height,
                    is_enemy=True,
                    dx=math.cos(spread_angle) * ENEMY_BULLET_SPEED * 1.2,
                    dy=math.sin(spread_angle) * ENEMY_BULLET_SPEED * 1.2,
                    has_direction=True,
                ))
            m2.last_fire_time = self.game_time

        if m2.y > GAME_HEIGHT + MONSTER2_HEIGHT:
            self.monster2 = None
            self.last_monster2_time = self.game_time

    def _get_monster2_pattern(self):
        patterns = {2: 'spiral', 3: 'zigzag', 4: 'figure8',
                    5: 'bounce', 6: 'wave', 7: 'teleport',
                    8: 'chase', 9: 'random'}
        return patterns.get(self.current_level, 'random')

    # =========================================================================
    # Spawning
    # =========================================================================
    def _handle_monster_creation(self):
        if self.monster is None and self.game_time - self.last_monster_time > MONSTER_INTERVAL:
            direction = 1 if random.random() < 0.5 else -1
            start_x = -MONSTER_WIDTH if direction == 1 else GAME_WIDTH + MONSTER_WIDTH

            should_slalom = len(self.enemies) < MONSTER_SLALOM_THRESHOLD
            if self.enemies:
                top_row = min(e.y for e in self.enemies) - 50
            else:
                top_row = MONSTER_HEIGHT

            self.monster = Monster(
                x=start_x,
                y=0 if should_slalom else max(top_row, MONSTER_HEIGHT) - 45,
                is_slaloming=should_slalom,
                last_fire_time=self.game_time,
                direction=direction,
            )
            self.last_monster_time = self.game_time
            self._emit(EventType.MONSTER_SPAWNED, slaloming=should_slalom)

    def _handle_monster2_creation(self):
        if self.current_level < 2:
            return
        if self.monster and self.monster.is_slaloming:
            self.last_monster2_time = self.game_time
            return
        if not self.monster and (self.game_time - self.last_monster_time < MONSTER2_INTERVAL / 2):
            return
        if self.monster2 is None and self.game_time - self.last_monster2_time > MONSTER2_INTERVAL:
            self.monster2 = Monster2(
                x=GAME_WIDTH / 2,
                y=-MONSTER2_HEIGHT,
                last_fire_time=self.game_time,
            )
            self.last_monster2_time = self.game_time

    def _handle_kamikaze_creation(self):
        if self.game_time >= self.next_kamikaze_time and self.enemies:
            idx = random.randint(0, len(self.enemies) - 1)
            enemy = self.enemies.pop(idx)
            self.kamikazes.append(Kamikaze(
                x=enemy.x, y=enemy.y,
                width=enemy.width, height=enemy.height,
                last_fire_time=self.game_time,
            ))
            self._emit(EventType.KAMIKAZE_SPAWNED)

            n = len(self.enemies)
            if n < KAMIKAZE_VERY_AGGRESSIVE_THRESHOLD:
                self.next_kamikaze_time = self.game_time + KAMIKAZE_VERY_AGGRESSIVE_TIME
            elif n < KAMIKAZE_AGGRESSIVE_THRESHOLD:
                self.next_kamikaze_time = self.game_time + KAMIKAZE_AGGRESSIVE_TIME
            else:
                self.next_kamikaze_time = self._random_kamikaze_time()

    # =========================================================================
    # Enemy actions
    # =========================================================================
    def _handle_enemy_shooting(self):
        if self.game_time - self.last_enemy_fire_time < self.current_enemy_fire_rate * 1000:
            return
        if not self.enemies:
            return

        # Find lowest enemy in each column, then pick closest to player
        columns = {}
        for e in self.enemies:
            col = int(e.x / (e.width + 20))
            if col not in columns or e.y > columns[col].y:
                columns[col] = e

        if columns:
            closest = min(columns.values(), key=lambda e: abs(e.x - self.player_x))
            self.bullets.append(Bullet(
                x=closest.x + closest.width / 2,
                y=closest.y + closest.height,
                is_enemy=True,
            ))
            self.last_enemy_fire_time = self.game_time

    def _handle_missile_launching(self):
        if self.game_time >= self.next_missile_time and self.enemies:
            # Pick random top-row enemy
            min_y = min(e.y for e in self.enemies)
            top_row = [e for e in self.enemies if e.y == min_y]
            if top_row:
                shooter = random.choice(top_row)
                self.missiles.append(Missile(
                    x=shooter.x + shooter.width / 2,
                    y=shooter.y + shooter.height,
                ))
                self._emit(EventType.MISSILE_LAUNCHED, source="enemy")

            self.next_missile_time = (self.game_time +
                                      random.uniform(MIN_MISSILE_INTERVAL, MAX_MISSILE_INTERVAL))

    # =========================================================================
    # Collision detection
    # =========================================================================
    def _detect_collisions(self):
        # Use removed flags to avoid rebuilding lists multiple times
        self._check_bullet_kamikaze()
        self._check_bullet_wall()
        self._check_bullet_enemy()
        self._check_bullet_missile()
        self._check_bullet_monster()
        self._check_bullet_monster2()
        self._check_enemy_bullet_player()
        self._check_kamikaze_player()
        self._check_missile_player()
        self._check_missile_wall()
        self._check_enemy_bullet_wall()
        self._remove_destroyed_walls()
        # Single compaction pass for all entity lists
        self._compact_entities()

    def _compact_entities(self):
        """Single-pass removal of all flagged entities."""
        self.bullets = [b for b in self.bullets if not b.removed]
        self.kamikazes = [k for k in self.kamikazes if not k.removed]
        self.missiles = [m for m in self.missiles if not m.removed]
        self.enemies = [e for e in self.enemies if e.hits < ENEMY_HITS_TO_DESTROY]

    def _check_bullet_kamikaze(self):
        for b in self.bullets:
            if b.is_enemy or b.removed:
                continue
            for k in self.kamikazes:
                if k.removed:
                    continue
                if (b.x < k.x + k.width and b.x + BULLET_W > k.x and
                        b.y < k.y + k.height and b.y + BULLET_H > k.y):
                    b.removed = True
                    k.hits += 1
                    if k.hits >= KAMIKAZE_HITS_TO_DESTROY:
                        k.removed = True
                        self.score += 300
                        self.kamikazes_killed += 1
                        self._emit(EventType.KAMIKAZE_KILLED)
                    break

    def _check_bullet_wall(self):
        wall_y_top = WALL_Y - BULLET_H  # Y-prefilter threshold
        for b in self.bullets:
            if b.is_enemy or b.removed:
                continue
            if b.y < wall_y_top:  # bullet is far above walls, skip
                continue
            for w in self.walls:
                if (b.x < w.x + w.width and b.x + BULLET_W > w.x and
                        b.y < w.y + w.height and b.y + BULLET_H > w.y):
                    b.removed = True
                    w.hit_count += 1
                    break

    def _check_bullet_enemy(self):
        for b in self.bullets:
            if b.is_enemy or b.removed:
                continue
            for e in self.enemies:
                if e.hits >= ENEMY_HITS_TO_DESTROY:
                    continue
                if (b.x < e.x + e.width and b.x + BULLET_W > e.x and
                        b.y < e.y + e.height and b.y + BULLET_H > e.y):
                    e.hits += 1
                    b.removed = True
                    if e.hits >= ENEMY_HITS_TO_DESTROY:
                        self.score += 10 + 30  # kill + explosion
                        self.enemies_killed += 1
                        self._emit(EventType.ENEMY_KILLED)
                    break

    def _check_bullet_missile(self):
        for b in self.bullets:
            if b.is_enemy or b.removed:
                continue
            for m in self.missiles:
                if m.removed:
                    continue
                dx = b.x - m.x
                dy = b.y - m.y
                if dx * dx + dy * dy < (m.width / 2 + 5) ** 2:  # squared distance
                    b.removed = True
                    m.removed = True
                    self.homing_missile_hits += 1
                    self.score += 500
                    self.missiles_shot += 1
                    self._emit(EventType.MISSILE_SHOT_DOWN)

                    if self.homing_missile_hits % 4 == 0:
                        self.score += 500
                        self.bonus_grants += 1
                        self._emit(EventType.BONUS_EARNED)
                        if self.bonus_grants >= BONUS2LIVES:
                            self.player_lives = min(self.player_lives + 1, PLAYER_LIVES)
                            self.bonus_grants = 0
                            self._emit(EventType.LIFE_GRANTED)
                    break

    def _check_bullet_monster(self):
        if not self.monster or self.monster.hit:
            return
        m = self.monster
        for b in self.bullets:
            if b.is_enemy or b.removed:
                continue
            has_enemy_in_path = any(
                b.x >= e.x and b.x <= e.x + e.width and
                b.y > e.y and b.y < m.y + m.height
                for e in self.enemies
            )
            if has_enemy_in_path:
                continue
            if (b.x < m.x + m.width and b.x + BULLET_W > m.x and
                    b.y < m.y + m.height and b.y + BULLET_H > m.y):
                b.removed = True
                m.hit = True
                m.hit_time = self.game_time
                self.score += 500
                self._restore_walls()
                self._emit(EventType.MONSTER_KILLED)
                break

    def _check_bullet_monster2(self):
        m2 = self.monster2
        if not m2 or m2.is_disappeared or m2.hit:
            return
        for b in self.bullets:
            if b.is_enemy or b.removed:
                continue
            if (b.x < m2.x + m2.width and b.x + BULLET_W > m2.x and
                    b.y < m2.y + m2.height and b.y + BULLET_H > m2.y):
                b.removed = True
                m2.hit = True
                m2.hit_time = self.game_time
                self.score += 1500
                self._restore_walls()
                self._emit(EventType.MONSTER2_KILLED)
                break

    def _check_enemy_bullet_player(self):
        if self.is_player_hit:
            return
        for b in self.bullets:
            if not b.is_enemy or b.removed:
                continue
            if (b.x < self.player_x + PLAYER_WIDTH and b.x + BULLET_W > self.player_x and
                    b.y < self.player_y + PLAYER_HEIGHT and b.y + BULLET_H > self.player_y):
                self._handle_player_hit()
                # Clear all enemy bullets
                for bb in self.bullets:
                    if bb.is_enemy:
                        bb.removed = True
                return

    def _check_kamikaze_player(self):
        if self.is_player_hit:
            return
        for k in self.kamikazes:
            if k.removed:
                continue
            # Wall collision first
            hit_wall = False
            for w in self.walls:
                if (k.x < w.x + w.width and k.x + k.width > w.x and
                        k.y < w.y + w.height and k.y + k.height > w.y):
                    self.score += 30
                    k.removed = True
                    hit_wall = True
                    break
            if hit_wall:
                continue
            # Player collision
            if (k.x < self.player_x + PLAYER_WIDTH and
                    k.x + k.width > self.player_x and
                    k.y < self.player_y + PLAYER_HEIGHT and
                    k.y + k.height > self.player_y):
                k.removed = True
                self._handle_player_hit()

    def _check_missile_player(self):
        if self.is_player_hit:
            return
        player_cx = self.player_x + PLAYER_WIDTH / 2
        player_cy = self.player_y + PLAYER_HEIGHT / 2
        hit_radius_sq = (PLAYER_WIDTH / 2 + MISSILE_WIDTH / 4) ** 2
        for m in self.missiles:
            if m.removed:
                continue
            dx = m.x - player_cx
            dy = m.y - player_cy
            if dx * dx + dy * dy < hit_radius_sq:
                self._handle_player_hit()
                # Clear all missiles on hit
                for mm in self.missiles:
                    mm.removed = True
                return

    def _check_missile_wall(self):
        for m in self.missiles:
            if m.removed:
                continue
            for w in self.walls:
                if (m.x >= w.x and m.x <= w.x + w.width and
                        m.y >= w.y and m.y <= w.y + w.height):
                    m.removed = True
                    w.missile_hits += 1
                    break

    def _check_enemy_bullet_wall(self):
        wall_y_top = WALL_Y  # Y-prefilter: only check bullets near wall row
        for b in self.bullets:
            if not b.is_enemy or b.removed:
                continue
            if b.y < wall_y_top:  # bullet hasn't reached wall row yet
                continue
            for w in self.walls:
                if (b.x >= w.x and b.x <= w.x + w.width and
                        b.y >= w.y and b.y <= w.y + w.height):
                    b.removed = True
                    w.hit_count += 1
                    break

    def _remove_destroyed_walls(self):
        before = len(self.walls)
        self.walls = [w for w in self.walls
                      if w.hit_count < WALL_MAX_HITS_TOTAL and
                      w.missile_hits < WALL_MAX_MISSILE_HITS]
        removed = before - len(self.walls)
        for _ in range(removed):
            self._emit(EventType.WALL_DESTROYED)

    def _handle_player_hit(self):
        self.player_lives -= 1
        self.is_player_hit = True
        self.player_hit_timer = self.game_time
        self.times_hit += 1
        self._emit(EventType.PLAYER_HIT, lives=self.player_lives)

        # Clear threats
        for b in self.bullets:
            if b.is_enemy:
                b.removed = True
        for m in self.missiles:
            m.removed = True
        for k in self.kamikazes:
            k.removed = True

        if self.player_lives <= 0:
            self.game_over = True
            self._emit(EventType.GAME_OVER, reason="no_lives")

    def _victory(self):
        self.current_level += 1
        self.enemy_speed *= 1.33
        self.current_enemy_fire_rate = (BASE_ENEMY_FIRE_RATE /
                                        (1 + ENEMY_FIRE_RATE_INCREASE * (self.current_level - 1)))
        self.score += 2500
        self._emit(EventType.LEVEL_COMPLETE, level=self.current_level)

        # Clear everything
        self.enemies = []
        self.bullets = []
        self.missiles = []
        self.kamikazes = []
        self.monster = None
        self.monster2 = None
        self._restore_walls()

        # Reset kamikaze timer
        self.next_kamikaze_time = self._random_kamikaze_time()

        # Create new enemies
        self._create_enemies()


# =============================================================================
# DQN Neural Network
# =============================================================================
if HAS_TORCH:
    class DQN(nn.Module):
        def __init__(self, state_size, action_size, hidden_sizes=None, use_layer_norm=False):
            super().__init__()
            sizes = hidden_sizes or [256, 256, 128]
            layers = []
            prev = state_size
            for h in sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h))
                prev = h
            layers.append(nn.Linear(prev, action_size))
            self.net = nn.Sequential(*layers)
            self.hidden_sizes = sizes

        def forward(self, x):
            return self.net(x)


    class NoisyLinear(nn.Module):
        """Factorized NoisyNet linear layer for state-dependent exploration."""
        def __init__(self, in_features, out_features, sigma_init=0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
            self.sigma_init = sigma_init
            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self):
            mu_range = 1 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

        def _scale_noise(self, size):
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign().mul(x.abs().sqrt())

        def reset_noise(self):
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def forward(self, x):
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(x, weight, bias)


    class DuelingDQN(nn.Module):
        """Dueling DQN with optional NoisyNet layers."""
        def __init__(self, state_size, action_size, hidden_sizes=None,
                     use_noisy=False, use_layer_norm=False):
            super().__init__()
            sizes = hidden_sizes or [512, 256, 128]
            self.action_size = action_size
            self.use_noisy = use_noisy
            self.hidden_sizes = sizes

            # Shared feature extractor (all layers except last)
            layers = []
            prev = state_size
            for h in sizes[:-1]:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h))
                prev = h
            self.features = nn.Sequential(*layers)

            last_h = sizes[-1]
            Linear = NoisyLinear if use_noisy else nn.Linear

            # Value stream: state -> V(s)
            self.value_hidden = Linear(prev, last_h)
            self.value_out = Linear(last_h, 1)

            # Advantage stream: state -> A(s, a)
            self.adv_hidden = Linear(prev, last_h)
            self.adv_out = Linear(last_h, action_size)

        def forward(self, x):
            features = self.features(x)
            v = F.relu(self.value_hidden(features))
            v = self.value_out(v)
            a = F.relu(self.adv_hidden(features))
            a = self.adv_out(a)
            q = v + a - a.mean(dim=1, keepdim=True)
            return q

        def reset_noise(self):
            if self.use_noisy:
                self.value_hidden.reset_noise()
                self.value_out.reset_noise()
                self.adv_hidden.reset_noise()
                self.adv_out.reset_noise()


# =============================================================================
# Replay Buffer (fast numpy-backed uniform sampling)
# =============================================================================
class FastReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        i = self.idx % self.capacity
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones):
        n = len(actions)
        idxs = np.arange(self.idx, self.idx + n) % self.capacity
        self.states[idxs] = states
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.next_states[idxs] = next_states
        self.dones[idxs] = dones
        self.idx += n
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_states[idxs], self.dones[idxs])

    def __len__(self):
        return self.size


class DualReplayBuffer:
    """Dual-buffer replay: main uniform buffer + secondary buffer for important transitions.
    Mixes a fraction of each batch from the important buffer (high |reward| or done=True)."""

    def __init__(self, capacity, state_size, important_capacity=50_000,
                 important_ratio=0.25, reward_threshold=2.0):
        self.main = FastReplayBuffer(capacity, state_size)
        self.important = FastReplayBuffer(important_capacity, state_size)
        self.important_ratio = important_ratio
        self.reward_threshold = reward_threshold

    def push(self, state, action, reward, next_state, done):
        self.main.push(state, action, reward, next_state, done)
        if abs(reward) > self.reward_threshold or done:
            self.important.push(state, action, reward, next_state, done)

    def push_batch(self, states, actions, rewards, next_states, dones):
        self.main.push_batch(states, actions, rewards, next_states, dones)
        mask = (np.abs(rewards) > self.reward_threshold) | (dones > 0.5)
        if mask.any():
            idx = np.where(mask)[0]
            self.important.push_batch(
                states[idx], actions[idx], rewards[idx],
                next_states[idx], dones[idx])

    def sample(self, batch_size):
        if len(self.important) < 32:
            return self.main.sample(batch_size)
        n_important = int(batch_size * self.important_ratio)
        n_main = batch_size - n_important
        main_batch = self.main.sample(n_main)
        imp_batch = self.important.sample(n_important)
        return tuple(np.concatenate([m, i], axis=0) for m, i in zip(main_batch, imp_batch))

    def __len__(self):
        return len(self.main)


class NStepBuffer:
    """Accumulates N-step transitions before pushing to the main replay buffer.
    Computes discounted N-step return: R = r_0 + gamma*r_1 + gamma^2*r_2 + ...
    Then stores (s_0, a_0, R, s_n, done_n) — the reward signal propagates faster."""

    def __init__(self, n_step, gamma, num_envs, state_size):
        self.n = n_step
        self.gamma = gamma
        self.num_envs = num_envs
        # Per-env circular buffers
        self.states = np.zeros((num_envs, n_step, state_size), dtype=np.float32)
        self.actions = np.zeros((num_envs, n_step), dtype=np.int64)
        self.rewards = np.zeros((num_envs, n_step), dtype=np.float32)
        self.next_states = np.zeros((num_envs, state_size), dtype=np.float32)
        self.count = np.zeros(num_envs, dtype=np.int32)  # how many steps buffered
        # Precompute gamma powers
        self.gamma_powers = np.array([gamma ** i for i in range(n_step)], dtype=np.float32)

    def add(self, env_idx, state, action, reward, next_state, done, replay_buffer):
        """Add a transition. When N steps accumulated (or done), push to replay buffer."""
        i = env_idx
        pos = self.count[i]

        if pos < self.n:
            self.states[i, pos] = state
            self.actions[i, pos] = action
            self.rewards[i, pos] = reward
            self.next_states[i] = next_state
            self.count[i] = pos + 1

        if done:
            # Flush all accumulated steps as N-step transitions
            c = self.count[i]
            for start in range(c):
                remaining = c - start
                n_step_reward = np.dot(self.rewards[i, start:c], self.gamma_powers[:remaining])
                replay_buffer.push(
                    self.states[i, start], self.actions[i, start],
                    n_step_reward, next_state, True)
            self.count[i] = 0
        elif self.count[i] >= self.n:
            # Full N-step transition ready
            n_step_reward = np.dot(self.rewards[i], self.gamma_powers)
            replay_buffer.push(
                self.states[i, 0], self.actions[i, 0],
                n_step_reward, next_state, False)
            # Shift buffer left by 1
            self.states[i, :-1] = self.states[i, 1:]
            self.actions[i, :-1] = self.actions[i, 1:]
            self.rewards[i, :-1] = self.rewards[i, 1:]
            self.count[i] = self.n - 1

    def add_batch(self, states, actions, rewards, next_states, dones, replay_buffer):
        """Add transitions for all envs. Vectorized where possible."""
        for i in range(self.num_envs):
            self.add(i, states[i], actions[i], rewards[i], next_states[i], dones[i], replay_buffer)

    def reset_env(self, env_idx):
        """Reset buffer for a specific env (called on episode end)."""
        self.count[env_idx] = 0


# =============================================================================
# Frame Stacking
# =============================================================================
class FrameStack:
    """Maintains a rolling buffer of the last N state vectors per environment."""
    def __init__(self, num_envs, state_size, n_frames=4):
        self.n_frames = n_frames
        self.state_size = state_size
        self.stacked_size = n_frames * state_size
        self.buffer = np.zeros((num_envs, n_frames, state_size), dtype=np.float32)

    def reset(self, env_idx, state):
        for i in range(self.n_frames):
            self.buffer[env_idx, i] = state
        return self.buffer[env_idx].reshape(-1)

    def reset_all(self, states):
        n = states.shape[0]
        for i in range(self.n_frames):
            self.buffer[:n, i] = states
        return self.buffer[:n].reshape(n, -1)

    def push(self, states, dones=None):
        n = states.shape[0]
        self.buffer[:n, :-1] = self.buffer[:n, 1:]
        self.buffer[:n, -1] = states
        return self.buffer[:n].reshape(n, -1)

    def get_all(self):
        return self.buffer.reshape(self.buffer.shape[0], -1)


# =============================================================================
# Training Configuration
# =============================================================================
@dataclass
class TrainingConfig:
    """All tunable hyperparameters for DQN training."""
    lr: float = 1e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99995
    buffer_capacity: int = 500_000
    train_every: int = 4
    hidden_sizes: list = field(default_factory=lambda: [512, 256, 128])
    train_steps_per_tick: int = 4  # gradient steps per training tick
    n_step: int = 5  # N-step returns (1 = standard TD, 3-5 = faster learning)
    use_layer_norm: bool = False
    use_dual_buffer: bool = True  # dual-buffer PER
    important_buffer_capacity: int = 50_000
    important_ratio: float = 0.30
    important_reward_threshold: float = 0.5
    use_cosine_lr: bool = True  # cosine annealing LR schedule
    cosine_cycle_episodes: int = 50_000  # episodes per LR cycle
    lr_min: float = 1e-5
    use_dueling: bool = True    # dueling architecture (V + A streams)
    use_noisy: bool = True      # NoisyNet (replaces epsilon-greedy)
    n_frames: int = 4           # frame stacking (1 = no stacking)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Plateau Detector
# =============================================================================
class PlateauDetector:
    """Detects training plateaus using multi-signal analysis."""

    def __init__(self, window=5000, min_episodes=15000, cooldown=10000,
                 score_threshold=0.03):
        self.window = window
        self.min_episodes = min_episodes
        self.cooldown = cooldown
        self.score_threshold = score_threshold
        self.scores = []
        self.levels = []
        self.hits = []
        self.best_score_ever = 0
        self.last_trigger_episode = 0
        self.total_episodes = 0

    def add(self, score, level, times_hit=0):
        self.scores.append(score)
        self.levels.append(level)
        self.hits.append(times_hit)
        self.best_score_ever = max(self.best_score_ever, score)
        self.total_episodes += 1

    def check(self) -> bool:
        """Returns True if plateau detected."""
        ep = self.total_episodes
        if ep < self.min_episodes:
            return False
        if ep - self.last_trigger_episode < self.cooldown:
            return False
        if len(self.scores) < self.window * 3:
            return False

        w = self.window
        w_old = self.scores[-3 * w:-2 * w]
        w_mid = self.scores[-2 * w:-w]
        w_recent = self.scores[-w:]

        mean_mid = np.mean(w_mid)
        mean_recent = np.mean(w_recent)
        mean_old = np.mean(w_old)

        # Primary: score stagnation over two windows
        if mean_mid > 0:
            recent_vs_mid = abs(mean_recent - mean_mid) / mean_mid
            if recent_vs_mid >= self.score_threshold:
                return False

        # Longer horizon check
        if mean_old > 0:
            recent_vs_old = abs(mean_recent - mean_old) / mean_old
            if recent_vs_old >= 0.05:
                return False

        # Anti-false-positive: recent breakthrough
        recent_best = max(w_recent)
        if recent_best > self.best_score_ever * 1.10:
            return False

        # Secondary confirmation: at least one must hold
        level_stagnant = abs(np.mean(self.levels[-w:]) - np.mean(self.levels[-2 * w:-w])) < 0.15
        hits_stagnant = len(self.hits) >= 2 * w and np.mean(self.hits[-w:]) >= np.mean(self.hits[-2 * w:-w]) * 0.95

        if not (level_stagnant or hits_stagnant):
            return False

        self.last_trigger_episode = ep
        return True

    def get_stats(self):
        """Return current stats for logging."""
        w = min(self.window, len(self.scores))
        return {
            "avg_score": float(np.mean(self.scores[-w:])) if self.scores else 0,
            "avg_level": float(np.mean(self.levels[-w:])) if self.levels else 0,
            "best_score": self.best_score_ever,
            "total_episodes": self.total_episodes,
        }


# =============================================================================
# DQN Agent
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu', config=None):
        cfg = config or TrainingConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.config = cfg

        self.use_noisy = cfg.use_noisy

        if cfg.use_dueling:
            self.policy_net = DuelingDQN(state_size, action_size, cfg.hidden_sizes,
                                         use_noisy=cfg.use_noisy, use_layer_norm=cfg.use_layer_norm).to(self.device)
            self.target_net = DuelingDQN(state_size, action_size, cfg.hidden_sizes,
                                         use_noisy=cfg.use_noisy, use_layer_norm=cfg.use_layer_norm).to(self.device)
        else:
            self.policy_net = DQN(state_size, action_size, cfg.hidden_sizes, cfg.use_layer_norm).to(self.device)
            self.target_net = DQN(state_size, action_size, cfg.hidden_sizes, cfg.use_layer_norm).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # torch.compile() for PyTorch 2.x+ speedup
        try:
            self.policy_net = torch.compile(self.policy_net)
            self.target_net = torch.compile(self.target_net)
        except Exception:
            pass  # Not available on older PyTorch versions

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

        if cfg.use_dual_buffer:
            self.memory = DualReplayBuffer(
                cfg.buffer_capacity, state_size,
                important_capacity=cfg.important_buffer_capacity,
                important_ratio=cfg.important_ratio,
                reward_threshold=cfg.important_reward_threshold)
        else:
            self.memory = FastReplayBuffer(cfg.buffer_capacity, state_size)

        # Cosine annealing LR scheduler
        self.scheduler = None
        if cfg.use_cosine_lr:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cfg.cosine_cycle_episodes, T_mult=1, eta_min=cfg.lr_min)

        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        self.n_step = cfg.n_step
        self.gamma_n = cfg.gamma ** cfg.n_step  # discount for N-step target
        self.epsilon = cfg.epsilon_start
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.tau = cfg.tau
        self.train_every = cfg.train_every
        self.train_steps_per_tick = cfg.train_steps_per_tick
        self.steps = 0
        self.env_steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def select_actions_batch(self, states_batch):
        """Select actions for all environments in one forward pass."""
        n = len(states_batch) if isinstance(states_batch, list) else states_batch.shape[0]

        if self.use_noisy:
            # NoisyNet: reset noise for diverse exploration
            policy_mod = self.policy_net._orig_mod if hasattr(self.policy_net, '_orig_mod') else self.policy_net
            if hasattr(policy_mod, 'reset_noise'):
                policy_mod.reset_noise()
            with torch.no_grad():
                states_t = torch.as_tensor(
                    np.array(states_batch) if isinstance(states_batch, list) else states_batch,
                    dtype=torch.float32, device=self.device)
                q_values = self.policy_net(states_t)
                actions = q_values.argmax(dim=1).cpu().numpy()
            return actions.tolist()

        # Per-env epsilon-greedy (each env independently explores)
        explore_mask = np.random.random(n) < self.epsilon
        n_explore = int(explore_mask.sum())

        with torch.no_grad():
            states_t = torch.as_tensor(
                np.array(states_batch) if isinstance(states_batch, list) else states_batch,
                dtype=torch.float32, device=self.device)
            q_values = self.policy_net(states_t)
            actions = q_values.argmax(dim=1).cpu().numpy()

        if n_explore > 0:
            actions[explore_mask] = np.random.randint(0, self.action_size, size=n_explore)

        return actions.tolist()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        if self.use_noisy:
            for net in [self.policy_net, self.target_net]:
                mod = net._orig_mod if hasattr(net, '_orig_mod') else net
                if hasattr(mod, 'reset_noise'):
                    mod.reset_noise()

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN with N-step returns)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma_n * next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft target network update (Polyak averaging)
        self.steps += 1
        with torch.no_grad():
            for p_target, p_policy in zip(self.target_net.parameters(),
                                          self.policy_net.parameters()):
                p_target.data.mul_(1.0 - self.tau).add_(p_policy.data * self.tau)

        return loss.item()

    def maybe_train(self):
        """Step-based training: multiple gradient steps every N environment steps."""
        self.env_steps += 1
        if self.env_steps % self.train_every == 0:
            total_loss = 0.0
            for _ in range(self.train_steps_per_tick):
                total_loss += self.train_step()
            return total_loss / self.train_steps_per_tick if self.train_steps_per_tick > 0 else 0.0
        return 0.0

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.scheduler:
            self.scheduler.step()

    def reset_epsilon(self, value=0.15):
        """Reset epsilon for re-exploration (e.g., after plateau detection)."""
        self.epsilon = value

    def save(self, path):
        data = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'env_steps': self.env_steps,
            'arch': {
                'use_dueling': self.config.use_dueling,
                'use_noisy': self.config.use_noisy,
                'n_frames': self.config.n_frames,
                'hidden_sizes': self.config.hidden_sizes,
            },
        }
        if self.scheduler:
            data['scheduler'] = self.scheduler.state_dict()
        torch.save(data, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        # Get the actual module (unwrap torch.compile if needed)
        policy_mod = self.policy_net._orig_mod if hasattr(self.policy_net, '_orig_mod') else self.policy_net
        target_mod = self.target_net._orig_mod if hasattr(self.target_net, '_orig_mod') else self.target_net

        # Normalize checkpoint keys (strip _orig_mod. prefix if present)
        def normalize_keys(state_dict):
            new_sd = {}
            for k, v in state_dict.items():
                new_key = k.replace('_orig_mod.', '')
                new_sd[new_key] = v
            return new_sd

        saved_policy = normalize_keys(checkpoint['policy_net'])
        saved_target = normalize_keys(checkpoint['target_net'])

        # Check if architectures match
        current_shapes = {k: v.shape for k, v in policy_mod.state_dict().items()}
        saved_shapes = {k: v.shape for k, v in saved_policy.items()}

        if current_shapes == saved_shapes:
            # Exact match — direct load
            policy_mod.load_state_dict(saved_policy)
            target_mod.load_state_dict(saved_target)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except (ValueError, KeyError, RuntimeError):
                pass
            print(f"  Loaded weights (exact match)")
        else:
            # Architecture mismatch — transfer what fits, init the rest
            self._transfer_weights(policy_mod, saved_policy)
            self._transfer_weights(target_mod, saved_target)
            # Fresh optimizer for new architecture
            print(f"  Transferred weights (architecture changed: {list(saved_shapes.values())} -> {list(current_shapes.values())})")

        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.env_steps = checkpoint.get('env_steps', 0)
        if self.scheduler and 'scheduler' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            except (ValueError, KeyError, RuntimeError):
                pass

    def _transfer_weights(self, module, saved_state_dict):
        """Transfer weights from a differently-sized checkpoint.
        Copies the overlapping region and initializes new neurons with small random values."""
        current_sd = module.state_dict()
        for name, param in current_sd.items():
            if name not in saved_state_dict:
                continue
            saved = saved_state_dict[name]
            if param.shape == saved.shape:
                param.copy_(saved)
            elif len(param.shape) == 2:
                # Linear weight: [out_features, in_features]
                min_out = min(param.shape[0], saved.shape[0])
                min_in = min(param.shape[1], saved.shape[1])
                # Initialize new neurons with small random values
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                param[:min_out, :min_in] = saved[:min_out, :min_in]
            elif len(param.shape) == 1:
                # Bias: [features]
                min_f = min(param.shape[0], saved.shape[0])
                param.zero_()
                param[:min_f] = saved[:min_f]
        module.load_state_dict(current_sd)


# =============================================================================
# GPU Auto-Scaling
# =============================================================================
def auto_scale_for_gpu(config: 'TrainingConfig', device: str, num_envs: int) -> tuple:
    """Auto-scale batch size and num_envs based on available GPU memory."""
    if not HAS_TORCH or device == 'cpu':
        return config, num_envs

    try:
        if device == 'cuda' and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
            gpu_name = props.name
        elif device == 'mps':
            # MPS doesn't expose memory easily, assume conservative
            gpu_mem_gb = 8
            gpu_name = "Apple MPS"
        else:
            return config, num_envs

        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

        # Scale based on GPU memory
        # Network is tiny (~105K-408K params), so we can push batch size very high
        if gpu_mem_gb >= 24:
            config.batch_size = max(config.batch_size, 4096)
            config.train_steps_per_tick = max(config.train_steps_per_tick, 2)
            num_envs = max(num_envs, 2048)
            config.buffer_capacity = max(config.buffer_capacity, 2_000_000)
        elif gpu_mem_gb >= 16:
            config.batch_size = max(config.batch_size, 2048)
            config.train_steps_per_tick = max(config.train_steps_per_tick, 2)
            num_envs = max(num_envs, 1024)
            config.buffer_capacity = max(config.buffer_capacity, 1_000_000)
        elif gpu_mem_gb >= 8:
            config.batch_size = max(config.batch_size, 1024)
            num_envs = max(num_envs, 512)
        elif gpu_mem_gb >= 4:
            config.batch_size = max(config.batch_size, 512)
            num_envs = max(num_envs, 256)

        print(f"Auto-scaled: batch_size={config.batch_size}, num_envs={num_envs}, buffer={config.buffer_capacity:,}")

    except Exception as e:
        print(f"Auto-scale skipped: {e}")

    return config, num_envs


# =============================================================================
# Training Loop
# =============================================================================
NUM_ENVS = 256  # Number of parallel game instances


def train(episodes=1_000_000, resume_path=None, save_dir="models", device_override=None,
          num_envs=NUM_ENVS, config=None, plateau_detector=None, cycle=0,
          flush_buffer=False, cosine_reset=False, auto_scale=True):
    os.makedirs(save_dir, exist_ok=True)

    if device_override:
        device = device_override
    else:
        device = 'cpu'
        if HAS_TORCH and torch.cuda.is_available():
            device = 'cuda'
        elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'

    cfg = config or TrainingConfig()

    # Auto-scale batch size and num_envs based on GPU memory
    if auto_scale:
        cfg, num_envs = auto_scale_for_gpu(cfg, device, num_envs)

    use_rust = HAS_RUST_SIM
    if use_rust:
        envs = RustBatchedGames(num_envs, seed=42)
        raw_state_size = envs.state_size
        action_size = envs.action_size
        print("Using Rust game simulation (fast mode)")
    else:
        games = [HeadlessGame() for _ in range(num_envs)]
        raw_state_size = games[0].state_size
        action_size = games[0].action_size
        print("Using Python game simulation (slow mode)")

    # Frame stacking
    frame_stack = None
    if cfg.n_frames > 1:
        frame_stack = FrameStack(num_envs, raw_state_size, cfg.n_frames)
        state_size = frame_stack.stacked_size
    else:
        state_size = raw_state_size

    if HAS_TORCH:
        agent = DQNAgent(state_size, action_size, device=device, config=cfg)
        if resume_path:
            agent.load(resume_path)
            # Apply config overrides after load (for meta-learning)
            agent.optimizer = optim.Adam(agent.policy_net.parameters(), lr=cfg.lr)
            agent.batch_size = cfg.batch_size
            agent.gamma = cfg.gamma
            agent.tau = cfg.tau
            agent.train_every = cfg.train_every
            # Only bump epsilon if config explicitly set it below 1.0 (mutation)
            # Default 1.0 means "keep checkpoint epsilon"
            if cfg.epsilon_start < 1.0:
                agent.epsilon = cfg.epsilon_start
            agent.epsilon_min = cfg.epsilon_min
            agent.epsilon_decay = cfg.epsilon_decay
            if flush_buffer:
                if cfg.use_dual_buffer:
                    agent.memory = DualReplayBuffer(
                        cfg.buffer_capacity, state_size,
                        important_capacity=cfg.important_buffer_capacity,
                        important_ratio=cfg.important_ratio,
                        reward_threshold=cfg.important_reward_threshold)
                else:
                    agent.memory = FastReplayBuffer(cfg.buffer_capacity, state_size)
                print("  Replay buffer flushed (fresh start)")
            if cosine_reset and cfg.use_cosine_lr:
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                agent.scheduler = CosineAnnealingWarmRestarts(
                    agent.optimizer, T_0=cfg.cosine_cycle_episodes, T_mult=1, eta_min=cfg.lr_min)
                print("  Cosine LR scheduler reset")
            agent.n_step = cfg.n_step
            agent.gamma_n = cfg.gamma ** cfg.n_step
            agent.train_steps_per_tick = cfg.train_steps_per_tick
            print(f"Resumed from {resume_path} (epsilon={agent.epsilon:.4f}, lr={cfg.lr}, steps={agent.steps})")
    else:
        agent = None

    print(f"Training on device: {device}")
    print(f"Parallel environments: {num_envs}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Episodes: {episodes:,}")
    if cycle > 0:
        print(f"Meta-learning cycle: {cycle}")
    buffer_type = "dual-buffer PER" if cfg.use_dual_buffer else "uniform"
    lr_type = f"cosine({cfg.lr_min:.0e}-{cfg.lr:.0e})" if cfg.use_cosine_lr else f"fixed({cfg.lr})"
    print(f"Training: {cfg.train_steps_per_tick} gradient steps every {cfg.train_every} ticks, "
          f"{cfg.n_step}-step returns, batch_size={cfg.batch_size}")
    arch_desc = "Dueling" if cfg.use_dueling else "Standard"
    if cfg.use_noisy:
        arch_desc += "+NoisyNet"
    if cfg.n_frames > 1:
        arch_desc += f"+{cfg.n_frames}frames"
    print(f"Buffer: {buffer_type} | LR: {lr_type} | Network: {cfg.hidden_sizes} ({arch_desc})"
          f"{' +LayerNorm' if cfg.use_layer_norm else ''}")
    print("-" * 70)

    # Stats tracking
    scores = deque(maxlen=1000)
    levels = deque(maxlen=1000)
    best_score = 0
    best_level = 0
    total_events = {t: 0 for t in dir(EventType) if not t.startswith('_')}
    start_time = time.time()

    # Event log file
    log_path = os.path.join(save_dir, "training_events.jsonl")
    log_file = open(log_path, "a")

    # N-step buffer for accumulating multi-step returns
    n_step_buffer = None
    if agent and cfg.n_step > 1:
        n_step_buffer = NStepBuffer(cfg.n_step, cfg.gamma, num_envs, state_size)

    # Initialize all environments
    if use_rust:
        raw_states = envs.reset_all()
        states = frame_stack.reset_all(raw_states) if frame_stack else raw_states
    else:
        states = [game.reset() for game in games]
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_count = 0
    tick_count = 0
    plateau_detected = False

    while episode_count < episodes and not plateau_detected:
        if use_rust:
            # Batch action selection: ONE forward pass for all envs
            if agent:
                actions = agent.select_actions_batch(states)
            else:
                actions = [random.randrange(action_size) for _ in range(num_envs)]

            # Step all environments — fast path, no dict overhead
            raw_next_states, rewards, dones = envs.step_all_fast(actions)
            next_states = frame_stack.push(raw_next_states) if frame_stack else raw_next_states
            tick_count += 1

            # Push experiences (N-step or direct)
            rewards_np = np.asarray(rewards, dtype=np.float32)
            dones_np = np.asarray(dones, dtype=np.float32)
            if agent:
                if n_step_buffer:
                    n_step_buffer.add_batch(
                        states, np.array(actions, dtype=np.int64),
                        rewards_np, next_states, dones_np, agent.memory)
                else:
                    agent.memory.push_batch(
                        states, np.array(actions, dtype=np.int64),
                        rewards_np, next_states, dones_np)

            episode_rewards += rewards_np

            # Handle done envs (typically 0-3 per tick out of 128)
            done_mask = np.asarray(dones)
            done_idxs = np.where(done_mask)[0]
            for i in done_idxs:
                i = int(i)
                episode_count += 1

                ep_score, ep_level, ep_lives, ep_steps, ep_ekills, ep_kkills, ep_mshots, ep_hits = envs.get_stats(i)
                scores.append(ep_score)
                levels.append(ep_level)
                if ep_score > best_score:
                    best_score = ep_score
                if ep_level > best_level:
                    best_level = ep_level

                episode_summary = {
                    "episode": episode_count,
                    "score": ep_score,
                    "level": ep_level,
                    "lives_left": ep_lives,
                    "steps": ep_steps,
                    "enemies_killed": ep_ekills,
                    "kamikazes_killed": ep_kkills,
                    "missiles_shot": ep_mshots,
                    "times_hit": ep_hits,
                    "epsilon": agent.epsilon if agent else 0,
                    "reward": round(float(episode_rewards[i]), 2),
                }
                if cycle > 0:
                    episode_summary["cycle"] = cycle
                log_file.write(json.dumps(episode_summary) + "\n")

                # Feed plateau detector
                if plateau_detector:
                    plateau_detector.add(ep_score, ep_level, ep_hits)

                if agent:
                    agent.decay_epsilon()

                if episode_count % 1000 == 0 or episode_count <= 5:
                    elapsed = time.time() - start_time
                    eps_per_sec = episode_count / elapsed if elapsed > 0 else 0
                    avg_score = np.mean(scores) if scores else 0
                    avg_level = np.mean(levels) if levels else 0
                    eps = agent.epsilon if agent else 0
                    print(f"Ep {episode_count:>8,} | "
                          f"Avg Score: {avg_score:>8.0f} | "
                          f"Best: {best_score:>8,} | "
                          f"Avg Lvl: {avg_level:.1f} | "
                          f"Best Lvl: {best_level} | "
                          f"Eps: {eps:.4f} | "
                          f"{eps_per_sec:.0f} ep/s | "
                          f"{elapsed:.0f}s")

                    # Check for plateau every 1000 episodes
                    if plateau_detector and plateau_detector.check():
                        print(f"\n{'='*70}")
                        print(f"PLATEAU DETECTED at episode {episode_count}")
                        stats = plateau_detector.get_stats()
                        print(f"  Avg Score: {stats['avg_score']:.0f} | Avg Level: {stats['avg_level']:.1f}")
                        print(f"{'='*70}\n")
                        plateau_detected = True
                        break

                if agent and episode_count % 10_000 == 0:
                    path = os.path.join(save_dir, f"model_ep{episode_count}.pt")
                    agent.save(path)
                    print(f"  -> Saved checkpoint: {path}")

                if agent and ep_score >= best_score and episode_count > 100:
                    agent.save(os.path.join(save_dir, "model_best.pt"))

                if episode_count % 100 == 0:
                    log_file.flush()

                raw_state = envs.reset_one(i)
                states[i] = frame_stack.reset(i, raw_state) if frame_stack else raw_state
                episode_rewards[i] = 0.0

                if episode_count >= episodes:
                    break

            # Update non-done states in bulk
            not_done = ~done_mask
            states[not_done] = next_states[not_done]

            # Train multiple gradient steps periodically
            if agent and tick_count % agent.train_every == 0 and len(agent.memory) >= agent.batch_size:
                for _ in range(agent.train_steps_per_tick):
                    agent.train_step()
        else:
            # Python fallback path
            for i, game in enumerate(games):
                if agent:
                    action = agent.select_action(states[i])
                else:
                    action = random.randrange(action_size)

                next_state, reward, done, info = game.step(action)
                episode_rewards[i] += reward

                if agent:
                    agent.memory.push(states[i], action, reward, next_state, done)

                for event in info.get("events", []):
                    etype = event["type"].upper()
                    if etype in total_events:
                        total_events[etype] += 1

                if done:
                    episode_count += 1

                    if agent and len(agent.memory) >= agent.batch_size:
                        for _ in range(32):
                            agent.train_step()

                    scores.append(game.score)
                    levels.append(game.current_level)
                    if game.score > best_score:
                        best_score = game.score
                    if game.current_level > best_level:
                        best_level = game.current_level

                    episode_summary = {
                        "episode": episode_count,
                        "score": game.score,
                        "level": game.current_level,
                        "lives_left": game.player_lives,
                        "steps": game.total_steps,
                        "enemies_killed": game.enemies_killed,
                        "kamikazes_killed": game.kamikazes_killed,
                        "missiles_shot": game.missiles_shot,
                        "times_hit": game.times_hit,
                        "epsilon": agent.epsilon if agent else 0,
                        "reward": round(float(episode_rewards[i]), 2),
                    }
                    if cycle > 0:
                        episode_summary["cycle"] = cycle
                    log_file.write(json.dumps(episode_summary) + "\n")

                    if plateau_detector:
                        plateau_detector.add(game.score, game.current_level, game.times_hit)

                    if agent:
                        agent.decay_epsilon()

                    if episode_count % 1000 == 0 or episode_count <= 5:
                        elapsed = time.time() - start_time
                        eps_per_sec = episode_count / elapsed if elapsed > 0 else 0
                        avg_score = np.mean(scores) if scores else 0
                        avg_level = np.mean(levels) if levels else 0
                        eps = agent.epsilon if agent else 0
                        print(f"Ep {episode_count:>8,} | "
                              f"Avg Score: {avg_score:>8.0f} | "
                              f"Best: {best_score:>8,} | "
                              f"Avg Lvl: {avg_level:.1f} | "
                              f"Best Lvl: {best_level} | "
                              f"Eps: {eps:.4f} | "
                              f"{eps_per_sec:.0f} ep/s | "
                              f"{elapsed:.0f}s")

                        if plateau_detector and plateau_detector.check():
                            print(f"\n{'='*70}")
                            print(f"PLATEAU DETECTED at episode {episode_count}")
                            stats = plateau_detector.get_stats()
                            print(f"  Avg Score: {stats['avg_score']:.0f} | Avg Level: {stats['avg_level']:.1f}")
                            print(f"{'='*70}\n")
                            plateau_detected = True
                            break

                    if agent and episode_count % 10_000 == 0:
                        path = os.path.join(save_dir, f"model_ep{episode_count}.pt")
                        agent.save(path)
                        print(f"  -> Saved checkpoint: {path}")

                    if agent and game.score >= best_score and episode_count > 100:
                        agent.save(os.path.join(save_dir, "model_best.pt"))

                    if episode_count % 100 == 0:
                        log_file.flush()

                    states[i] = game.reset()
                    episode_rewards[i] = 0.0

                    if episode_count >= episodes:
                        break
                else:
                    states[i] = next_state

    # Final save
    if agent:
        agent.save(os.path.join(save_dir, "model_final.pt"))

    log_file.close()

    # Print final stats
    elapsed = time.time() - start_time
    avg_score_final = float(np.mean(scores)) if scores else 0
    stop_reason = "plateau" if plateau_detected else "complete"
    print("\n" + "=" * 70)
    print(f"TRAINING {'PAUSED (plateau)' if plateau_detected else 'COMPLETE'}")
    print("=" * 70)
    print(f"Episodes:    {episode_count:,}")
    print(f"Time:        {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"Best Score:  {best_score:,}")
    print(f"Best Level:  {best_level}")
    print(f"Avg Score (last 1k): {avg_score_final:.0f}")
    print(f"\nEvent totals:")
    for k, v in sorted(total_events.items()):
        if v > 0:
            print(f"  {k}: {v:,}")
    print(f"\nModel saved to: {save_dir}/model_final.pt")
    print(f"Event log:       {log_path}")

    # Export model weights as JSON for potential JS integration
    if agent:
        export_path = os.path.join(save_dir, "model_weights.json")
        export_model_to_json(agent, export_path)
        print(f"JSON weights:    {export_path}")

    return {
        "episodes_completed": episode_count,
        "best_score": best_score,
        "best_level": best_level,
        "avg_score": avg_score_final,
        "avg_level": float(np.mean(levels)) if levels else 0,
        "elapsed": elapsed,
        "stop_reason": stop_reason,
    }


def export_model_to_json(agent, path):
    """Export model weights to JSON for loading in JavaScript."""
    net = agent.policy_net
    if hasattr(net, '_orig_mod'):
        net = net._orig_mod
    weights = {}
    is_dueling = isinstance(net, DuelingDQN)
    for name, param in net.named_parameters():
        # For NoisyNet, export weight_mu/bias_mu as standard weight/bias
        export_name = name.replace('.weight_mu', '.weight').replace('.bias_mu', '.bias')
        # Skip sigma parameters (browser doesn't need noise)
        if '.weight_sigma' in name or '.bias_sigma' in name:
            continue
        weights[export_name] = param.detach().cpu().numpy().tolist()

    if is_dueling:
        # Infer input size from first feature layer
        first_layer = net.features[0]
        in_size = first_layer.in_features
        arch = [in_size] + net.hidden_sizes + [net.action_size]
        model_type = "dueling"
    else:
        arch = [net.net[0].in_features] + net.hidden_sizes + [agent.action_size]
        model_type = "standard"

    n_frames = agent.config.n_frames if hasattr(agent.config, 'n_frames') else 1

    with open(path, 'w') as f:
        json.dump({
            "architecture": arch,
            "activation": "relu",
            "type": model_type,
            "n_frames": n_frames,
            "weights": weights,
        }, f)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI for eyeBMinvaders")
    parser.add_argument("--episodes", type=int, default=1_000_000,
                        help="Number of episodes to train (default: 1,000,000)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models (default: models)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, cuda, mps)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help=f"Number of parallel environments (default: {NUM_ENVS})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training (default: auto-scaled by GPU)")
    parser.add_argument("--hidden-sizes", type=str, default=None,
                        help="Network hidden layer sizes, comma-separated (default: 256,256,128)")
    parser.add_argument("--n-step", type=int, default=None,
                        help="N-step returns (default: 3, use 1 for standard TD)")
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Gradient steps per training tick (default: 4, auto-scaled by GPU)")
    parser.add_argument("--no-auto-scale", action="store_true",
                        help="Disable GPU auto-scaling of batch size and num_envs")
    parser.add_argument("--layer-norm", action="store_true",
                        help="Enable LayerNorm in network")
    parser.add_argument("--no-dual-buffer", action="store_true",
                        help="Disable dual-buffer PER (use uniform sampling)")
    parser.add_argument("--no-cosine-lr", action="store_true",
                        help="Disable cosine annealing LR schedule")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--epsilon-min", type=float, default=None,
                        help="Minimum epsilon for exploration (default: 0.05)")
    parser.add_argument("--epsilon-decay", type=float, default=None,
                        help="Epsilon decay per episode (default: 0.99995)")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.hidden_sizes:
        cfg.hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    if args.n_step:
        cfg.n_step = args.n_step
    if args.train_steps:
        cfg.train_steps_per_tick = args.train_steps
    if args.layer_norm:
        cfg.use_layer_norm = True
    if args.no_dual_buffer:
        cfg.use_dual_buffer = False
    if args.no_cosine_lr:
        cfg.use_cosine_lr = False
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epsilon_min is not None:
        cfg.epsilon_min = args.epsilon_min
    if args.epsilon_decay is not None:
        cfg.epsilon_decay = args.epsilon_decay

    train(episodes=args.episodes, resume_path=args.resume, save_dir=args.save_dir,
          device_override=args.device,
          num_envs=args.num_envs if args.num_envs is not None else NUM_ENVS,
          config=cfg, auto_scale=not args.no_auto_scale)
