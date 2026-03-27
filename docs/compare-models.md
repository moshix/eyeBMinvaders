# Comparing Trained Models

## Side-by-Side Live Replay

Watch two AI models play simultaneously in an ASCII TUI. Both games use the same random seed so differences are purely from model decisions.

### Setup

```bash
# Build the Rust game sim (if not already)
cd game_sim && maturin develop --release && cd ..

# Install dependencies
pip install rich torch
```

### Usage

```bash
# Compare two model checkpoints
python3 replay.py models/model_a.pt models/model_b.pt

# Adjust speed (default: 10 ticks/sec)
python3 replay.py models/model_a.pt models/model_b.pt --speed 20

# Use GPU for inference
python3 replay.py models/model_a.pt models/model_b.pt --device cuda

# Custom field size
python3 replay.py models/model_a.pt models/model_b.pt --width 50 --height 20
```

### Controls

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| + / - | Increase / Decrease speed |
| R | Restart both games (new shared seed) |
| Q | Quit |

### Display

```
╭─ model_a.pt ─────────────────────╮ ╭─ model_b.pt ─────────────────────╮
│  Score:   45,230  Level: 3  ♥♥♥♥♥│ │  Score:   38,910  Level: 2  ♥♥♥♥│
│                                   │ │                                   │
│  ··· ▪▪▪▪▪▪▪▪▪▪ ···              │ │  ··· ▪▪▪▪▪▪▪▪▪▪▪▪ ···          │
│      ▪▪▪▪▪▪▪▪▪▪                  │ │       ▪▪▪▪▪▪▪▪▪▪               │
│          M                        │ │                  K              │
│                                   │ │           ◆                     │
│   ═══  ═══  ═══  ═══             │ │   ═══  ═══  ═══  ═══           │
│            ▲                      │ │         ▲                       │
│                                   │ │                                 │
│  Action: fire+right  Step: 1,234  │ │  Action: move-left   Step: 1,234│
╰───────────────────────────────────╯ ╰───────────────────────────────────╯
  Speed: 10 tps   Wins: A=3  B=2  (rounds: 5)
```

### Entity Legend

| Symbol | Entity |
|--------|--------|
| ▲ | Player |
| ✕ | Player (hit) |
| ▪ | Enemy (full health) |
| ▫ | Enemy (damaged) |
| \| | Player bullet |
| · | Enemy bullet |
| K | Kamikaze |
| ◆ | Homing missile |
| M | Monster |
| W | Monster2 |
| ═ | Wall (healthy) |
| ─ | Wall (damaged) |

### How It Works

- Both games start with identical random seeds for fair comparison
- When both games end, wins are tallied by final score and a new round auto-starts
- The model architecture (24→256→256→128→6) is loaded from `.pt` checkpoint files
- Game simulation runs via the Rust `game_sim` module for speed

## Training Dashboard

Monitor training progress in real-time from a second terminal:

```bash
python3 visualize.py
python3 visualize.py --log models/training_events.jsonl --episodes 200000
```

Shows sparkline charts for score, level, combat stats, and progress bars for epsilon decay and episode completion. Refreshes every 2 seconds.

## Exporting Models for Browser

After training, export the best model for use in the browser game (F1 = AI mode):

```bash
python3 export_model.py models/model_best.pt
```

This writes `models/model_weights.json` which `game.js` loads for in-browser neural network inference.
