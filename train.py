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
from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
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


@dataclass
class Missile:
    x: float
    y: float
    angle: float = 0.0
    width: float = MISSILE_WIDTH
    height: float = MISSILE_HEIGHT
    time: float = 0.0
    from_monster: bool = False


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
        # Penalty for destroying a wall (shooting through your own cover)
        wall_destroyed_count = self.events.count(EventType.WALL_DESTROYED)
        if wall_destroyed_count > 0:
            reward -= 2.0 * wall_destroyed_count
        # Small survival reward
        reward += 0.01

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
        Returns a fixed-size feature vector representing game state.
        23 features total.
        """
        player_cx = self.player_x + PLAYER_WIDTH / 2
        player_cy = self.player_y + PLAYER_HEIGHT / 2

        # Normalize helper
        def nx(v): return v / GAME_WIDTH
        def ny(v): return v / GAME_HEIGHT

        features = []

        # 1. Player position (normalized)
        features.append(nx(player_cx))

        # 2. Player lives (normalized)
        features.append(self.player_lives / PLAYER_LIVES)

        # 3. Level (normalized, cap at 10)
        features.append(min(self.current_level, 10) / 10.0)

        # 4. Number of enemies (normalized)
        features.append(min(len(self.enemies), 60) / 60.0)

        # 5-6. Nearest enemy relative position
        if self.enemies:
            nearest = min(self.enemies,
                          key=lambda e: abs(e.x + e.width / 2 - player_cx))
            features.append(nx(nearest.x + nearest.width / 2 - player_cx))
            features.append(ny(nearest.y + nearest.height / 2 - player_cy))
        else:
            features.extend([0.0, -1.0])

        # 7-8. Lowest enemy position (danger indicator)
        if self.enemies:
            lowest = max(self.enemies, key=lambda e: e.y)
            features.append(nx(lowest.x + lowest.width / 2 - player_cx))
            features.append(ny(lowest.y))
        else:
            features.extend([0.0, 0.0])

        # 9-11. Nearest enemy bullet
        enemy_bullets = [b for b in self.bullets if b.is_enemy]
        if enemy_bullets:
            nearest_b = min(enemy_bullets,
                            key=lambda b: (b.x - player_cx) ** 2 + (b.y - player_cy) ** 2)
            features.append(nx(nearest_b.x - player_cx))
            features.append(ny(nearest_b.y - player_cy))
            features.append(len(enemy_bullets) / 10.0)
        else:
            features.extend([0.0, -1.0, 0.0])

        # 12-14. Nearest missile
        if self.missiles:
            nearest_m = min(self.missiles,
                            key=lambda m: (m.x - player_cx) ** 2 + (m.y - player_cy) ** 2)
            features.append(nx(nearest_m.x - player_cx))
            features.append(ny(nearest_m.y - player_cy))
            features.append(len(self.missiles) / 5.0)
        else:
            features.extend([0.0, -1.0, 0.0])

        # 15-17. Nearest kamikaze
        if self.kamikazes:
            nearest_k = min(self.kamikazes,
                            key=lambda k: (k.x - player_cx) ** 2 + (k.y - player_cy) ** 2)
            features.append(nx(nearest_k.x + nearest_k.width / 2 - player_cx))
            features.append(ny(nearest_k.y + nearest_k.height / 2 - player_cy))
            features.append(len(self.kamikazes) / 5.0)
        else:
            features.extend([0.0, -1.0, 0.0])

        # 18-19. Monster info
        if self.monster and not self.monster.hit:
            features.append(nx(self.monster.x + self.monster.width / 2 - player_cx))
            features.append(ny(self.monster.y))
        else:
            features.extend([0.0, -1.0])

        # 20. Is player currently invulnerable (hit animation)
        features.append(1.0 if self.is_player_hit else 0.0)

        # 21. Number of walls remaining
        features.append(len(self.walls) / 4.0)

        # 22-23. Nearest wall relative position (for cover)
        # 24. Nearest wall health (1.0 = full, 0.0 = destroyed/none)
        if self.walls:
            nearest_w = min(self.walls,
                            key=lambda w: abs(w.x + w.width / 2 - player_cx))
            features.append(nx(nearest_w.x + nearest_w.width / 2 - player_cx))
            features.append(ny(nearest_w.y - player_cy))
            features.append(1.0 - nearest_w.hit_count / WALL_MAX_HITS_TOTAL)
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    @property
    def state_size(self):
        return 24

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
        to_remove = set()
        player_cx = self.player_x + PLAYER_WIDTH / 2
        player_cy = self.player_y + PLAYER_HEIGHT / 2

        for i, k in enumerate(self.kamikazes):
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
                to_remove.add(i)
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
                        to_remove.add(i)
                        break

        self.kamikazes = [k for i, k in enumerate(self.kamikazes) if i not in to_remove]
        # Remove off-screen
        self.kamikazes = [k for k in self.kamikazes
                          if 0 < k.x < GAME_WIDTH]

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

    def _check_bullet_kamikaze(self):
        bullets_to_remove = set()
        kamikazes_to_remove = set()
        for bi, b in enumerate(self.bullets):
            if b.is_enemy:
                continue
            for ki, k in enumerate(self.kamikazes):
                if ki in kamikazes_to_remove:
                    continue
                if (b.x < k.x + k.width and b.x + BULLET_W > k.x and
                        b.y < k.y + k.height and b.y + BULLET_H > k.y):
                    bullets_to_remove.add(bi)
                    k.hits += 1
                    if k.hits >= KAMIKAZE_HITS_TO_DESTROY:
                        kamikazes_to_remove.add(ki)
                        self.score += 300
                        self.kamikazes_killed += 1
                        self._emit(EventType.KAMIKAZE_KILLED)
                    break

        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        self.kamikazes = [k for i, k in enumerate(self.kamikazes) if i not in kamikazes_to_remove]

    def _check_bullet_wall(self):
        bullets_to_remove = set()
        for bi, b in enumerate(self.bullets):
            if b.is_enemy:
                continue
            for w in self.walls:
                if (b.x < w.x + w.width and b.x + BULLET_W > w.x and
                        b.y < w.y + w.height and b.y + BULLET_H > w.y):
                    bullets_to_remove.add(bi)
                    w.hit_count += 1
                    break
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

    def _check_bullet_enemy(self):
        bullets_to_remove = set()
        enemies_to_remove = set()
        for bi, b in enumerate(self.bullets):
            if b.is_enemy or bi in bullets_to_remove:
                continue
            for ei, e in enumerate(self.enemies):
                if ei in enemies_to_remove:
                    continue
                if (b.x < e.x + e.width and b.x + BULLET_W > e.x and
                        b.y < e.y + e.height and b.y + BULLET_H > e.y):
                    e.hits += 1
                    bullets_to_remove.add(bi)
                    if e.hits >= ENEMY_HITS_TO_DESTROY:
                        enemies_to_remove.add(ei)
                        self.score += 10 + 30  # kill + explosion
                        self.enemies_killed += 1
                        self._emit(EventType.ENEMY_KILLED)
                    break

        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]

    def _check_bullet_missile(self):
        bullets_to_remove = set()
        missiles_to_remove = set()
        for bi in range(len(self.bullets) - 1, -1, -1):
            b = self.bullets[bi]
            if b.is_enemy:
                continue
            for mi in range(len(self.missiles) - 1, -1, -1):
                if mi in missiles_to_remove:
                    continue
                m = self.missiles[mi]
                dx = b.x - m.x
                dy = b.y - m.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < (m.width / 2 + 5):
                    bullets_to_remove.add(bi)
                    missiles_to_remove.add(mi)
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

        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        self.missiles = [m for i, m in enumerate(self.missiles) if i not in missiles_to_remove]

    def _check_bullet_monster(self):
        if not self.monster or self.monster.hit:
            return
        bullets_to_remove = set()
        m = self.monster
        for bi, b in enumerate(self.bullets):
            if b.is_enemy:
                continue
            # Check if enemy is in the bullet's path
            has_enemy_in_path = any(
                b.x >= e.x and b.x <= e.x + e.width and
                b.y > e.y and b.y < m.y + m.height
                for e in self.enemies
            )
            if has_enemy_in_path:
                continue
            if (b.x < m.x + m.width and b.x + BULLET_W > m.x and
                    b.y < m.y + m.height and b.y + BULLET_H > m.y):
                bullets_to_remove.add(bi)
                m.hit = True
                m.hit_time = self.game_time
                self.score += 500
                self._restore_walls()
                self._emit(EventType.MONSTER_KILLED)
                break

        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

    def _check_bullet_monster2(self):
        m2 = self.monster2
        if not m2 or m2.is_disappeared or m2.hit:
            return
        bullets_to_remove = set()
        for bi, b in enumerate(self.bullets):
            if b.is_enemy:
                continue
            if (b.x < m2.x + m2.width and b.x + BULLET_W > m2.x and
                    b.y < m2.y + m2.height and b.y + BULLET_H > m2.y):
                bullets_to_remove.add(bi)
                m2.hit = True
                m2.hit_time = self.game_time
                self.score += 1500
                self._restore_walls()
                self._emit(EventType.MONSTER2_KILLED)
                break
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

    def _check_enemy_bullet_player(self):
        if self.is_player_hit:
            return
        bullets_to_remove = set()
        for bi, b in enumerate(self.bullets):
            if not b.is_enemy:
                continue
            if (b.x < self.player_x + PLAYER_WIDTH and b.x + BULLET_W > self.player_x and
                    b.y < self.player_y + PLAYER_HEIGHT and b.y + BULLET_H > self.player_y):
                bullets_to_remove.add(bi)
                self._handle_player_hit()
                # Clear all enemy bullets
                self.bullets = [bb for bb in self.bullets if not bb.is_enemy]
                return
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

    def _check_kamikaze_player(self):
        if self.is_player_hit:
            return
        to_remove = set()
        for ki, k in enumerate(self.kamikazes):
            # Wall collision first
            hit_wall = False
            for w in self.walls:
                if (k.x < w.x + w.width and k.x + k.width > w.x and
                        k.y < w.y + w.height and k.y + k.height > w.y):
                    self.score += 30
                    to_remove.add(ki)
                    hit_wall = True
                    break
            if hit_wall:
                continue
            # Player collision
            if (k.x < self.player_x + PLAYER_WIDTH and
                    k.x + k.width > self.player_x and
                    k.y < self.player_y + PLAYER_HEIGHT and
                    k.y + k.height > self.player_y):
                to_remove.add(ki)
                self._handle_player_hit()

        self.kamikazes = [k for i, k in enumerate(self.kamikazes) if i not in to_remove]

    def _check_missile_player(self):
        if self.is_player_hit:
            return
        player_cx = self.player_x + PLAYER_WIDTH / 2
        player_cy = self.player_y + PLAYER_HEIGHT / 2
        for mi, m in enumerate(self.missiles):
            dx = m.x - player_cx
            dy = m.y - player_cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < (PLAYER_WIDTH / 2 + m.width / 4):
                self._handle_player_hit()
                self.missiles = []  # Clear all missiles on hit
                return

    def _check_missile_wall(self):
        missiles_to_remove = set()
        for mi, m in enumerate(self.missiles):
            for w in self.walls:
                if (m.x >= w.x and m.x <= w.x + w.width and
                        m.y >= w.y and m.y <= w.y + w.height):
                    missiles_to_remove.add(mi)
                    w.missile_hits += 1
                    break
        self.missiles = [m for i, m in enumerate(self.missiles) if i not in missiles_to_remove]

    def _check_enemy_bullet_wall(self):
        bullets_to_remove = set()
        for bi, b in enumerate(self.bullets):
            if not b.is_enemy:
                continue
            for w in self.walls:
                if (b.x >= w.x and b.x <= w.x + w.width and
                        b.y >= w.y and b.y <= w.y + w.height):
                    bullets_to_remove.add(bi)
                    w.hit_count += 1
                    break
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

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
        self.bullets = [b for b in self.bullets if not b.is_enemy]
        self.missiles = []
        self.kamikazes = []

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
        def __init__(self, state_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size),
            )

        def forward(self, x):
            return self.net(x)


# =============================================================================
# Replay Buffer
# =============================================================================
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN Agent
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(200_000)

        self.batch_size = 256
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99998  # reaches ~0.02 around ep 200k
        self.target_update = 1000
        self.train_every = 8  # Only train every N steps (huge speedup)
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


# =============================================================================
# Training Loop
# =============================================================================
def train(episodes=1_000_000, resume_path=None, save_dir="models", device_override=None):
    os.makedirs(save_dir, exist_ok=True)

    game = HeadlessGame()
    if device_override:
        device = device_override
    else:
        device = 'cpu'
        if HAS_TORCH and torch.cuda.is_available():
            device = 'cuda'
        elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'

    if HAS_TORCH:
        agent = DQNAgent(game.state_size, game.action_size, device=device)
        if resume_path:
            agent.load(resume_path)
            print(f"Resumed from {resume_path} (epsilon={agent.epsilon:.4f}, steps={agent.steps})")
    else:
        agent = None

    print(f"Training on device: {device}")
    print(f"State size: {game.state_size}, Action size: {game.action_size}")
    print(f"Episodes: {episodes:,}")
    print("-" * 70)

    # Stats tracking
    scores = deque(maxlen=1000)
    levels = deque(maxlen=1000)
    best_score = 0
    best_level = 0
    total_events = {t: 0 for t in dir(EventType) if not t.startswith('_')}
    start_time = time.time()
    train_steps = 0
    pending_experiences = []  # Batch experiences before training

    # Event log file
    log_path = os.path.join(save_dir, "training_events.jsonl")
    log_file = open(log_path, "a")

    # Training schedule: train N batches after every K episodes
    TRAIN_INTERVAL = 10       # Train every N episodes
    TRAIN_BATCHES = 32        # Number of gradient steps per training round

    for episode in range(1, episodes + 1):
        state = game.reset()
        episode_reward = 0.0

        while not game.game_over:
            # Select action
            if agent:
                action = agent.select_action(state)
            else:
                action = random.randrange(game.action_size)

            next_state, reward, done, info = game.step(action)
            episode_reward += reward

            # Store experience (defer training)
            if agent:
                agent.memory.push(state, action, reward, next_state, done)
                train_steps += 1

            # Collect events
            for event in info.get("events", []):
                etype = event["type"].upper()
                if etype in total_events:
                    total_events[etype] += 1

            state = next_state
            if done:
                break

        # Batch training: do multiple gradient steps periodically
        if agent:
            agent.decay_epsilon()
            if episode % TRAIN_INTERVAL == 0 and len(agent.memory) >= agent.batch_size:
                for _ in range(TRAIN_BATCHES):
                    agent.train_step()

        # Track stats
        scores.append(game.score)
        levels.append(game.current_level)
        if game.score > best_score:
            best_score = game.score
        if game.current_level > best_level:
            best_level = game.current_level

        # Log episode summary
        episode_summary = {
            "episode": episode,
            "score": game.score,
            "level": game.current_level,
            "lives_left": game.player_lives,
            "steps": game.total_steps,
            "enemies_killed": game.enemies_killed,
            "kamikazes_killed": game.kamikazes_killed,
            "missiles_shot": game.missiles_shot,
            "times_hit": game.times_hit,
            "epsilon": agent.epsilon if agent else 0,
            "reward": round(episode_reward, 2),
        }
        log_file.write(json.dumps(episode_summary) + "\n")

        # Print progress
        if episode % 1000 == 0 or episode <= 5:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            avg_score = np.mean(scores) if scores else 0
            avg_level = np.mean(levels) if levels else 0
            eps = agent.epsilon if agent else 0

            print(f"Ep {episode:>8,} | "
                  f"Avg Score: {avg_score:>8.0f} | "
                  f"Best: {best_score:>8,} | "
                  f"Avg Lvl: {avg_level:.1f} | "
                  f"Best Lvl: {best_level} | "
                  f"Eps: {eps:.4f} | "
                  f"{eps_per_sec:.0f} ep/s | "
                  f"{elapsed:.0f}s")

        # Save checkpoints
        if agent and episode % 10_000 == 0:
            path = os.path.join(save_dir, f"model_ep{episode}.pt")
            agent.save(path)
            print(f"  -> Saved checkpoint: {path}")

        # Save best model
        if agent and game.score >= best_score and episode > 100:
            agent.save(os.path.join(save_dir, "model_best.pt"))

        # Periodic flush
        if episode % 100 == 0:
            log_file.flush()

    # Final save
    if agent:
        agent.save(os.path.join(save_dir, "model_final.pt"))

    log_file.close()

    # Print final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Episodes:    {episodes:,}")
    print(f"Time:        {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"Best Score:  {best_score:,}")
    print(f"Best Level:  {best_level}")
    print(f"Avg Score (last 1k): {np.mean(scores):.0f}")
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


def export_model_to_json(agent, path):
    """Export model weights to JSON for loading in JavaScript."""
    weights = {}
    for name, param in agent.policy_net.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    with open(path, 'w') as f:
        json.dump({
            "architecture": [agent.policy_net.net[0].in_features, 256, 256, 128, 6],
            "activation": "relu",
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
    args = parser.parse_args()

    train(episodes=args.episodes, resume_path=args.resume, save_dir=args.save_dir,
          device_override=args.device)
