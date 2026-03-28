
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

- **`game.js`** — Complete game engine + neural network inference (supports standard DQN, Dueling DQN, NoisyNet, frame stacking)
- **`train.py`** — DQN training with Rust game simulation backend. Supports Dueling DQN, NoisyNet exploration, N-step returns, frame stacking, dual-buffer PER, cosine LR annealing
- **`meta_train.py`** — Meta-learning outer loop with plateau detection and multi-armed bandit hyperparameter mutation
- **`game_sim/`** — Rust game simulation (PyO3 + Rayon) for parallel training. 50-feature state vector, ~5.7M env-steps/s on GPU-class hardware
- **`visualize.py`** — Live training dashboard (rich TUI)
- **`export_model.py`** — PyTorch to JSON model converter (handles Dueling, NoisyNet, frame stacking architectures)

### DQN Architecture

| Component | Details |
|-----------|---------|
| Network | Dueling DQN (shared features → separate Value + Advantage streams) |
| Exploration | NoisyNet (factorized, sigma=0.5) + epsilon-greedy fallback |
| State | 50 hand-crafted features × 2-4 frame stack (enemy speed, fire cooldown, threat urgency, danger heatmap) |
| Returns | N-step (n=5) with gamma=0.99 |
| Replay | Dual-buffer: uniform (300K) + important transitions (100K, 35% sampling) |
| Parallelism | Rayon-parallelized Rust sim, 128-2048 envs |

### Training Results

| Metric | Standard DQN | Dueling+NoisyNet+Frames |
|--------|-------------|------------------------|
| Best Level | 7 | 8 |
| Best Score | ~100K | 153K |
| Avg Score | ~45K plateau | 59-65K |
| Training Speed | ~700K steps/s | ~5.7M steps/s (GPU), ~700K (CPU) |

### GPU Recommendations

Training benefits enormously from GPU acceleration. CPU training is functional but ~10x slower for neural network operations. Recommended: NVIDIA GPU with 8GB+ VRAM.

```bash
# GPU auto-scales batch size and parallel envs based on VRAM
python3 train.py --episodes 500000
# CPU explicit config
python3 train.py --episodes 500000 --num-envs 128
```

March, 2025

