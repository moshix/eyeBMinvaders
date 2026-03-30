#!/usr/bin/env python3
"""
Behavioral Cloning from Recorded Gameplay
==========================================
Trains the DQN model to imitate recorded gameplay from models/gameplay_data.jsonl.
The gameplay is recorded by pressing W in the browser while AI (0) plays.

This closes the loop: AI plays → browser records → train on recordings → better AI.

Usage:
    python train_bc.py                                    # Train from recordings
    python train_bc.py --model models/model_best.pt      # Fine-tune existing model
    python train_bc.py --epochs 50 --lr 1e-4             # Custom params
    python train_bc.py --export                           # Export to JSON after training
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("ERROR: pip install torch numpy")
    sys.exit(1)

GAMEPLAY_FILE = os.path.join(os.path.dirname(__file__), 'models', 'gameplay_data.jsonl')


def load_gameplay_data(path, max_transitions=500_000):
    """Load recorded transitions from JSONL file."""
    states, actions = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_transitions:
                break
            t = json.loads(line.strip())
            states.append(t['s'])
            actions.append(t['a'])
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


def train_bc(model_path=None, epochs=30, lr=1e-4, batch_size=256, export_after=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load gameplay data
    if not os.path.exists(GAMEPLAY_FILE):
        print(f"No gameplay data at {GAMEPLAY_FILE}")
        print("Press W in browser while AI plays to record, then run this.")
        return

    print(f"Loading gameplay data from {GAMEPLAY_FILE}...")
    states, actions = load_gameplay_data(GAMEPLAY_FILE)
    n = len(states)
    state_size = states.shape[1]
    print(f"Loaded {n:,} transitions, state_size={state_size}")

    if n < 1000:
        print("Not enough data. Play more games with W active.")
        return

    # Load or create model
    from train import DuelingDQN, DQN, TrainingConfig, NoisyLinear

    cfg = TrainingConfig()
    action_size = 6

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        sd = checkpoint['policy_net']
        # Detect architecture
        clean = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        first_w = clean[list(clean.keys())[0]]
        input_size = first_w.shape[1]
        is_dueling = any('value_hidden' in k for k in clean)
        is_noisy = any('weight_mu' in k for k in clean)

        if is_dueling:
            model = DuelingDQN(input_size, action_size, cfg.hidden_sizes,
                               use_noisy=is_noisy).to(device)
        else:
            model = DQN(input_size, action_size, cfg.hidden_sizes).to(device)
        model.load_state_dict(clean, strict=False)
        print(f"Loaded: input={input_size}, dueling={is_dueling}, noisy={is_noisy}")

        # Handle state size mismatch (recorded data may have different features)
        if state_size != input_size:
            print(f"WARNING: recorded state_size={state_size} != model input={input_size}")
            if state_size < input_size:
                # Pad recorded states with zeros
                pad = np.zeros((n, input_size - state_size), dtype=np.float32)
                states = np.concatenate([states, pad], axis=1)
                print(f"  Padded states to {input_size}")
            else:
                # Truncate
                states = states[:, :input_size]
                print(f"  Truncated states to {input_size}")
    else:
        # Create new model matching recorded state size
        input_size = state_size
        model = DuelingDQN(input_size, action_size, cfg.hidden_sizes,
                           use_noisy=False).to(device)
        print(f"New model: input={input_size}, dueling, no-noisy")

    # Behavioral cloning: supervised learning (state → action)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Split train/val
    perm = np.random.permutation(n)
    val_size = min(5000, n // 10)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    states_t = torch.tensor(states, device=device)
    actions_t = torch.tensor(actions, device=device)

    print(f"\nTraining: {len(train_idx):,} train, {val_size:,} val, {epochs} epochs, lr={lr}")
    print(f"Device: {device}\n")

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        # Shuffle training data
        np.random.shuffle(train_idx)

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(train_idx), batch_size):
            idx = train_idx[i:i + batch_size]
            s = states_t[idx]
            a = actions_t[idx]

            q_values = model(s)
            loss = F.cross_entropy(q_values, a)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (q_values.argmax(dim=1) == a).sum().item()
            total += len(idx)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_s = states_t[val_idx]
            val_a = actions_t[val_idx]
            val_q = model(val_s)
            val_loss = F.cross_entropy(val_q, val_a).item()
            val_acc = (val_q.argmax(dim=1) == val_a).float().mean().item()

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join('models', 'model_bc_best.pt')
            torch.save({
                'policy_net': model.state_dict(),
                'target_net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'steps': 0,
                'epsilon': 0.05,
                'arch': {
                    'use_dueling': is_dueling,
                    'use_noisy': is_noisy,
                    'n_frames': input_size // state_size if state_size > 0 else 4,
                    'hidden_sizes': cfg.hidden_sizes,
                },
            }, save_path)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved: models/model_bc_best.pt")

    if export_after:
        from export_model import export_model_to_json
        export_model_to_json('models/model_bc_best.pt')
        print("Exported to models/model_weights.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral cloning from gameplay recordings')
    parser.add_argument('--model', default=None, help='Model to fine-tune (default: new)')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--export', action='store_true', help='Export to JSON after training')
    args = parser.parse_args()

    train_bc(
        model_path=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        export_after=args.export,
    )
