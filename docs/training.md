# Training the AI

The game includes a Deep Q-Learning (DQN) training pipeline that trains a neural network to play autonomously.

## Requirements

```bash
pip install torch numpy
```

## Quick Start

```bash
python3 train.py
```

This trains for 1,000,000 episodes using 8 parallel game environments on the best available device (CUDA > Apple Metal > CPU).

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes N` | 1,000,000 | Number of episodes to train |
| `--resume PATH` | — | Resume training from a checkpoint (.pt file) |
| `--save-dir DIR` | `models/` | Directory for checkpoints and logs |
| `--device DEVICE` | auto | Force device: `cpu`, `cuda`, or `mps` |
| `--num-envs N` | 8 | Number of parallel game environments |

## Examples

```bash
# Short test run
python3 train.py --episodes 10000

# Use Apple Metal GPU
python3 train.py --device mps

# More parallel environments for faster experience collection
python3 train.py --num-envs 16

# Resume from a checkpoint
python3 train.py --resume models/model_best.pt

# Full example
python3 train.py --episodes 500000 --num-envs 12 --device mps --save-dir models/run2
```

## Exporting the Model for Browser Play

After training, export the PyTorch checkpoint to JSON so the browser can load it:

```bash
# Export the best model
python3 export_model.py models/model_best.pt

# Or export a specific checkpoint
python3 export_model.py models/model_ep50000.pt
```

This writes `models/model_weights.json`, which the game loads automatically when you press **F1** to toggle AI mode.

## Testing in the Browser

```bash
python3 -m http.server
```

Open http://localhost:8000 in Chrome, then press **F1** to activate AI mode. The status indicator shows whether the neural network or heuristic fallback is active.

## Training Output

During training, the console shows periodic progress:

```
Ep    1,000 | Avg Score:    5489 | Best:   10,740 | Avg Lvl: 1.0 | Best Lvl: 1 | Eps: 0.9802 | 18 ep/s | 55s
```

- **Avg Score / Best**: rolling average and all-time best game score
- **Avg Lvl / Best Lvl**: how many levels the agent clears
- **Eps**: exploration rate (starts at 1.0, decays to 0.02)
- **ep/s**: training throughput

### Saved Files

| File | Description |
|------|-------------|
| `models/model_best.pt` | Best-scoring checkpoint (auto-updated) |
| `models/model_ep{N}.pt` | Periodic checkpoint every 10,000 episodes |
| `models/model_final.pt` | Final model after training completes |
| `models/model_weights.json` | JSON weights for browser inference |
| `models/training_events.jsonl` | Per-episode log (score, level, events) |

## Architecture

- **Model**: 4-layer MLP — 24 inputs, 256, 256, 128 hidden units, 6 action outputs
- **Algorithm**: Double DQN with Prioritized Experience Replay (PER)
- **Target updates**: Soft (Polyak averaging, tau=0.005)
- **Training schedule**: 32 gradient steps per completed episode
- **State**: 24 normalized features (player position, threats, walls, etc.)
- **Actions**: idle, left, right, fire, fire+left, fire+right
