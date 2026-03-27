#!/usr/bin/env python3
"""Export the latest .pt model to model_weights.json for the browser game.
Run anytime while train.py is still going — it reads the saved .pt file, not memory.

Detects architecture from checkpoint keys:
- value_hidden in keys -> Dueling DQN
- weight_mu in keys -> NoisyNet (exports mu weights as standard weight/bias)

Usage:
    python3 export_model.py                  # exports models/model_best.pt
    python3 export_model.py models/model_ep50000.pt   # exports a specific checkpoint
"""

import json, sys, os, math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized NoisyNet linear layer."""
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

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def forward(self, x):
        return F.linear(x, self.weight_mu, self.bias_mu)


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


class DuelingDQN(nn.Module):
    """Dueling DQN with optional NoisyNet layers."""
    def __init__(self, state_size, action_size, hidden_sizes=None,
                 use_noisy=False, use_layer_norm=False):
        super().__init__()
        sizes = hidden_sizes or [512, 256, 128]
        self.action_size = action_size
        self.use_noisy = use_noisy
        self.hidden_sizes = sizes

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

        self.value_hidden = Linear(prev, last_h)
        self.value_out = Linear(last_h, 1)
        self.adv_hidden = Linear(prev, last_h)
        self.adv_out = Linear(last_h, action_size)

    def forward(self, x):
        features = self.features(x)
        v = F.relu(self.value_hidden(features))
        v = self.value_out(v)
        a = F.relu(self.adv_hidden(features))
        a = self.adv_out(a)
        return v + a - a.mean(dim=1, keepdim=True)


def normalize_keys(state_dict):
    """Strip _orig_mod. prefix from torch.compile() wrapped models."""
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}


def detect_architecture(sd):
    """Detect model architecture from state dict keys."""
    is_dueling = any('value_hidden' in k for k in sd)
    is_noisy = any('weight_mu' in k for k in sd)
    return is_dueling, is_noisy


def infer_sizes(sd, is_dueling):
    """Infer state_size, action_size, hidden_sizes from weight shapes."""
    if is_dueling:
        # Find feature layers: features.0.weight, features.2.weight, ...
        feature_keys = sorted([k for k in sd if k.startswith('features.') and 'weight' in k and len(sd[k].shape) == 2])
        state_size = sd[feature_keys[0]].shape[1]
        hidden_sizes = [sd[k].shape[0] for k in feature_keys]

        # Last hidden from value/adv streams
        adv_out_key = [k for k in sd if 'adv_out' in k and 'weight' in k and len(sd[k].shape) == 2][0]
        val_hidden_key = [k for k in sd if 'value_hidden' in k and 'weight' in k and len(sd[k].shape) == 2][0]
        last_h = sd[val_hidden_key].shape[0]
        hidden_sizes.append(last_h)
        action_size = sd[adv_out_key].shape[0]
    else:
        weight_keys = sorted([k for k in sd if 'weight' in k and len(sd[k].shape) == 2])
        state_size = sd[weight_keys[0]].shape[1]
        action_size = sd[weight_keys[-1]].shape[0]
        hidden_sizes = [sd[k].shape[0] for k in weight_keys[:-1]]

    return state_size, action_size, hidden_sizes


def export(pt_path, json_path):
    checkpoint = torch.load(pt_path, map_location="cpu")
    policy_sd = normalize_keys(checkpoint["policy_net"])

    is_dueling, is_noisy = detect_architecture(policy_sd)
    state_size, action_size, hidden_sizes = infer_sizes(policy_sd, is_dueling)

    # Get arch metadata from checkpoint if available
    arch_meta = checkpoint.get('arch', {})
    n_frames = arch_meta.get('n_frames', 1)

    arch_desc = f"{state_size} -> {hidden_sizes} -> {action_size}"
    if is_dueling:
        arch_desc += " (Dueling)"
    if is_noisy:
        arch_desc += " (NoisyNet)"
    if n_frames > 1:
        arch_desc += f" ({n_frames} frames)"
    has_layer_norm = any('LayerNorm' in k or 'layernorm' in k.lower() for k in policy_sd)
    if has_layer_norm:
        arch_desc += " (+LayerNorm)"
    print(f"Detected architecture: {arch_desc}")

    # Build the model
    if is_dueling:
        model = DuelingDQN(state_size, action_size, hidden_sizes,
                           use_noisy=is_noisy, use_layer_norm=has_layer_norm)
    else:
        model = DQN(state_size, action_size, hidden_sizes, has_layer_norm)

    model.load_state_dict(policy_sd)

    # Export weights — for NoisyNet, export weight_mu/bias_mu as weight/bias
    weights = {}
    for name, param in model.named_parameters():
        export_name = name.replace('.weight_mu', '.weight').replace('.bias_mu', '.bias')
        if '.weight_sigma' in name or '.bias_sigma' in name:
            continue
        weights[export_name] = param.detach().cpu().numpy().tolist()

    arch = [state_size] + hidden_sizes + [action_size]
    model_type = "dueling" if is_dueling else "standard"

    with open(json_path, "w") as f:
        json.dump({
            "architecture": arch,
            "activation": "relu",
            "type": model_type,
            "n_frames": n_frames,
            "weights": weights,
        }, f)

    eps = checkpoint.get("epsilon", "?")
    steps = checkpoint.get("steps", "?")
    size_kb = os.path.getsize(json_path) // 1024
    print(f"Exported {pt_path} -> {json_path}  (type={model_type}, n_frames={n_frames}, "
          f"epsilon={eps}, steps={steps}, {size_kb} KB)")


if __name__ == "__main__":
    pt_path = sys.argv[1] if len(sys.argv) > 1 else "models/model_best.pt"
    json_path = os.path.join(os.path.dirname(pt_path) or "models", "model_weights.json")
    export(pt_path, json_path)
