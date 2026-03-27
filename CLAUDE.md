# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

eyeBMinvaders is a browser-based Space Invaders game inspired by the original Namco arcade game. It's a single-page HTML5 canvas game with no build system or framework — just vanilla HTML, CSS, and JavaScript.

## Running the Game

Serve with any static HTTP server:
```
python3 -m http.server
```
Then open http://localhost:8000 in Chrome (recommended for JS performance over Safari).

## Architecture

The entire game is three files:

- **`index.html`** — Entry point. Contains all CSS inline, the canvas element (`1024x576`), the opening legend/instructions screen, and loads `game.js`.
- **`game.js`** (~3600 lines) — The entire game engine in one file. Contains all game state, constants, rendering, collision detection, enemy AI, player controls, sound management, and the neural network inference engine. Version is tracked via the `VERSION` constant at the top.
- **`firebirds.html`** — Separate opening/splash screen page.

### AI Mode (F1 key toggles)

Two AI systems play the game automatically:

1. **DQN Neural Network** (primary) — A feed-forward neural network (architecture: input→256→256→128→6 actions) runs inference in the browser using weights from `models/model_weights.json`. The inference engine is implemented directly in `game.js` (no ML libraries). 6 actions: idle, move-left, move-right, fire, fire+move-left, fire+move-right.
2. **Heuristic AI** (fallback) — Rule-based system in `updateAutoPlay()` that uses threat trajectory forecasting for bullets, missiles, and kamikazes. Activates if the neural network model fails to load.

### Training Pipeline (Python)

- **`train.py`** — Headless DQN training script that ports game mechanics from `game.js` into Python. Uses PyTorch. Trains via Deep Q-Learning with experience replay. Requires `torch` and `numpy`.
- **`export_model.py`** — Converts PyTorch `.pt` checkpoints to `models/model_weights.json` for browser inference.

```
python train.py --episodes 10000        # train
python export_model.py models/model_best.pt  # export to JSON
```

### Game Entities

- **Enemies** — Grid formation, move horizontally and step down, require 2 hits to destroy. Speed increases ~9% per level. From level 4, bottom row is removed.
- **Kamikazes** — Enemies that break formation and dive toward the player. Frequency increases as enemy count drops.
- **Homing Missiles** — Curved sinusoidal trajectory that re-targets the player above wall row, locks angle below it. Every 5th missile shot down grants a bonus.
- **Monster** — Appears periodically above enemies. Killing it restores walls.
- **Monster2** — Second boss type with level-dependent movement patterns (spiral, zigzag, figure8, bounce, wave, teleport, chase, random).
- **Walls** — 4 destructible barriers. Take damage from both enemy and player shots. Have separate hit counters for bullets vs missiles.

### Controls

Arrow keys / A,D for movement, Space to fire, P to pause, R to restart, M to mute, F1 for AI mode.

## Key Constants

Game balance constants are defined at the top of `game.js` (lines ~98-310) and mirrored in `train.py` (lines ~46-80). When changing game mechanics, both files must stay in sync.

## Assets

All assets (SVG sprites, MP3/WAV sounds, background image) are in the repository root — no asset pipeline or bundling.
