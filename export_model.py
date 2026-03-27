#!/usr/bin/env python3
"""Export the latest .pt model to model_weights.json for the browser game.
Run anytime while train.py is still going — it reads the saved .pt file, not memory.

Usage:
    python3 export_model.py                  # exports models/model_best.pt
    python3 export_model.py models/model_ep50000.pt   # exports a specific checkpoint
"""

import json, sys, os

import torch
import torch.nn as nn


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


def infer_architecture(state_dict):
    """Infer network architecture from checkpoint weight shapes."""
    # Normalize keys (strip _orig_mod. prefix if present)
    sd = {}
    for k, v in state_dict.items():
        sd[k.replace('_orig_mod.', '')] = v

    # Find all Linear weight layers (shape [out, in])
    weight_keys = sorted([k for k in sd if 'weight' in k and len(sd[k].shape) == 2])
    if not weight_keys:
        return None, None, None, False

    state_size = sd[weight_keys[0]].shape[1]
    action_size = sd[weight_keys[-1]].shape[0]

    hidden_sizes = []
    for k in weight_keys[:-1]:  # all but last (output) layer
        hidden_sizes.append(sd[weight_keys[weight_keys.index(k)]].shape[0])

    # Detect LayerNorm
    has_layer_norm = any('LayerNorm' in k or 'layernorm' in k.lower() for k in sd)

    return state_size, action_size, hidden_sizes, has_layer_norm


def export(pt_path, json_path):
    checkpoint = torch.load(pt_path, map_location="cpu")
    policy_sd = checkpoint["policy_net"]

    state_size, action_size, hidden_sizes, has_layer_norm = infer_architecture(policy_sd)
    if state_size is None:
        print("Error: could not infer architecture from checkpoint")
        sys.exit(1)

    print(f"Detected architecture: {state_size} -> {hidden_sizes} -> {action_size}"
          f"{' +LayerNorm' if has_layer_norm else ''}")

    model = DQN(state_size, action_size, hidden_sizes, has_layer_norm)

    # Normalize checkpoint keys
    normalized_sd = {}
    for k, v in policy_sd.items():
        normalized_sd[k.replace('_orig_mod.', '')] = v
    model.load_state_dict(normalized_sd)

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()

    arch = [state_size] + hidden_sizes + [action_size]
    with open(json_path, "w") as f:
        json.dump(
            {"architecture": arch, "activation": "relu", "weights": weights},
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
