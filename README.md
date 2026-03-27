
eyeBMinvaders
=============

A Namco-like space invaders game for the mainframe, and any other computers out there. Features an AI mode powered by a neural network trained via Deep Q-Learning.

Plays best on Chrome browsers because Safari sucks in JS performance.

Game features several innovations over the original Space Invaders. Play to find out more!

The original code from NAMCO is [here](https://computerarcheology.com/Arcade/SpaceInvaders/Code.html)

<img src="screenshot.png">

## Quick Start

```bash
python3 -m http.server
```
Open http://localhost:8000 in Chrome. Press **F1** to toggle AI mode.

### Controls

| Key | Action |
|-----|--------|
| Arrow keys / A, D | Move |
| Space | Fire |
| F1 | Toggle AI mode |
| P | Pause |
| R | Restart |
| M | Mute |

## AI Training

### Requirements

```bash
pip install torch numpy
```

### Train a model

```bash
python3 train.py --episodes 50000
```

Or use the meta-learning trainer that automatically detects plateaus and restarts with improved hyperparameters:

```bash
python3 meta_train.py --episodes 200000
```

### Resume from checkpoint

```bash
python3 train.py --resume models/model_best.pt
python3 meta_train.py --resume models/model_best.pt
```

### Export model for browser play

```bash
python3 export_model.py models/model_best.pt
```

This writes `models/model_weights.json`, loaded automatically when AI mode is toggled.

### Monitor training

```bash
python3 visualize.py
```

Live TUI dashboard showing score trends, level progression, and training metrics.

### Training options

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 1,000,000 | Number of episodes to train |
| `--resume PATH` | — | Resume from checkpoint |
| `--save-dir DIR` | `models/` | Output directory |
| `--device DEVICE` | auto | Force `cpu`, `cuda`, or `mps` |
| `--num-envs N` | 128 | Parallel game environments |

### Meta-learning options (meta_train.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 500,000 | Total episodes across all cycles |
| `--resume PATH` | — | Start from checkpoint |
| `--resume-meta` | — | Continue from existing meta state |
| `--max-cycles N` | 50 | Maximum depot cycles |

## Architecture

- **`game.js`** — Complete game engine + neural network inference
- **`train.py`** — DQN training with Rust game simulation backend
- **`meta_train.py`** — Meta-learning outer loop with plateau detection
- **`game_sim/`** — Rust game simulation (PyO3) for fast training
- **`visualize.py`** — Live training dashboard (rich TUI)
- **`export_model.py`** — PyTorch to JSON model converter

March, 2025

