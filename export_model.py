#!/usr/bin/env python3
"""Export the latest .pt model to model_weights.json for the browser game.
Run anytime while train.py is still going — it reads the saved .pt file, not memory.

Usage:
    python3 export_model.py                  # exports models/model_best.pt
    python3 export_model.py models/model_ep50000.pt   # exports a specific checkpoint
"""

import json, sys, os

import torch

# Inline the DQN class so we don't import train.py (which pulls in numpy etc.)
import torch.nn as nn

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


def export(pt_path, json_path):
    checkpoint = torch.load(pt_path, map_location="cpu")
    # Detect state size from saved weights (works for both 23 and 24 feature models)
    state_size = checkpoint["policy_net"]["net.0.weight"].shape[1]
    model = DQN(state_size, 6)
    model.load_state_dict(checkpoint["policy_net"])

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()

    with open(json_path, "w") as f:
        json.dump(
            {"architecture": [state_size, 256, 256, 128, 6], "activation": "relu", "weights": weights},
            f,
        )

    eps = checkpoint.get("epsilon", "?")
    steps = checkpoint.get("steps", "?")
    size_kb = os.path.getsize(json_path) // 1024
    print(f"Exported {pt_path} -> {json_path}  (epsilon={eps}, steps={steps}, {size_kb} KB)")


if __name__ == "__main__":
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "models/model_best.pt"
    json_path = os.path.join(os.path.dirname(pt_path) or "models", "model_weights.json")
    export(pt_path, json_path)
