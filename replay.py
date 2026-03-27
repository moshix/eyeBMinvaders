#!/usr/bin/env python3
"""
Side-by-side model comparison TUI for eyeBMinvaders.
Runs two AI models simultaneously and renders ASCII game fields.

Usage:
    python3 replay.py models/model_a.pt models/model_b.pt
    python3 replay.py models/model_a.pt models/model_b.pt --speed 20
"""

import argparse
import os
import sys
import time
import threading
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich.console import Console
except ImportError:
    print("Rich required: pip install rich")
    sys.exit(1)

try:
    from game_sim import Game
except ImportError:
    print("Rust game_sim not found. Build with: cd game_sim && maturin develop --release")
    sys.exit(1)


# =============================================================================
# DQN Network (must match training architecture)
# =============================================================================
class DQN(nn.Module):
    def __init__(self, state_size=24, action_size=6):
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


def load_model(path, device='cpu'):
    model = DQN().to(device)
    checkpoint = torch.load(path, map_location=device)
    if 'policy_net' in checkpoint:
        model.load_state_dict(checkpoint['policy_net'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


ACTION_NAMES = ['idle', 'left', 'right', 'fire', 'fire+left', 'fire+right']

GAME_W = 1024
GAME_H = 576


def select_action(model, state, device='cpu'):
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state_t)
        return q_values.argmax(dim=1).item()


# =============================================================================
# ASCII Renderer
# =============================================================================
def render_game(entities, width=40, height=16):
    """Render game state as ASCII grid."""
    grid = [[' '] * width for _ in range(height)]

    def sx(x):
        return max(0, min(width - 1, int(x / GAME_W * (width - 1))))

    def sy(y):
        return max(0, min(height - 1, int(y / GAME_H * (height - 1))))

    # Walls
    for wx, hc, mh in entities.get('walls', []):
        wy = sy(501)
        x1 = sx(wx)
        x2 = sx(wx + 58)
        for x in range(x1, min(x2 + 1, width)):
            if hc < 6:
                grid[wy][x] = '═'
            else:
                grid[wy][x] = '─'  # damaged

    # Enemies
    for ex, ey, hits in entities.get('enemies', []):
        gx, gy = sx(ex + 21), sy(ey + 21)
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = '▪' if hits == 0 else '▫'

    # Enemy bullets
    for bx, by, is_enemy in entities.get('bullets', []):
        gx, gy = sx(bx), sy(by)
        if 0 <= gy < height and 0 <= gx < width:
            if is_enemy:
                grid[gy][gx] = '·'
            else:
                grid[gy][gx] = '|'

    # Kamikazes
    for kx, ky in entities.get('kamikazes', []):
        gx, gy = sx(kx + 21), sy(ky + 21)
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = 'K'

    # Missiles
    for mx, my in entities.get('missiles', []):
        gx, gy = sx(mx), sy(my)
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = '◆'

    # Monster
    m = entities.get('monster')
    if m is not None:
        gx, gy = sx(m[0] + 28), sy(m[1] + 28)
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = 'M'

    # Monster2
    m2 = entities.get('monster2')
    if m2 is not None:
        gx, gy = sx(m2[0] + 28), sy(m2[1] + 28)
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = 'W'

    # Player (drawn last, on top)
    px = entities.get('player_x', 512)
    py_val = entities.get('player_y', 546)
    gx, gy = sx(px + 24), sy(py_val + 24)
    if entities.get('is_hit'):
        ch = '✕'
    else:
        ch = '▲'
    if 0 <= gy < height and 0 <= gx < width:
        grid[gy][gx] = ch

    return '\n'.join(''.join(row) for row in grid)


def build_panel(name, entities, action_id, step, field_w, field_h):
    """Build a rich Panel for one game."""
    score = entities.get('score', 0)
    level = entities.get('level', 1)
    lives = entities.get('lives', 0)
    game_over = entities.get('game_over', False)

    hearts = '♥' * max(0, lives) + '♡' * max(0, 6 - lives)
    action_name = ACTION_NAMES[action_id] if 0 <= action_id < 6 else '?'

    field = render_game(entities, field_w, field_h)

    status = f"  Score: [bold yellow]{score:>8,}[/]  Level: [bold cyan]{level}[/]  {hearts}"
    if game_over:
        status += "  [bold red]GAME OVER[/]"

    footer = f"  Action: [bold]{action_name:<12}[/] Step: {step:,}"

    content = f"{status}\n\n{field}\n\n{footer}"

    return Panel(
        Text.from_markup(content),
        title=f"[bold white]{name}[/]",
        border_style="blue" if not game_over else "red",
        width=field_w + 6,
    )


# =============================================================================
# Keyboard input (non-blocking)
# =============================================================================
class KeyReader:
    def __init__(self):
        self.last_key = None
        self._stop = False
        self._thread = None

    def start(self):
        import tty
        import termios
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop:
            try:
                ch = sys.stdin.read(1)
                if ch:
                    self.last_key = ch
            except Exception:
                break

    def get(self):
        k = self.last_key
        self.last_key = None
        return k

    def stop(self):
        import termios
        self._stop = True
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Side-by-side model replay")
    parser.add_argument("model_a", help="Path to first model checkpoint")
    parser.add_argument("model_b", help="Path to second model checkpoint")
    parser.add_argument("--speed", type=int, default=10, help="Ticks per second (default: 10)")
    parser.add_argument("--device", default="cpu", help="Device for inference (default: cpu)")
    parser.add_argument("--width", type=int, default=40, help="Field width in chars")
    parser.add_argument("--height", type=int, default=16, help="Field height in chars")
    args = parser.parse_args()

    device = args.device
    name_a = os.path.basename(args.model_a)
    name_b = os.path.basename(args.model_b)

    print(f"Loading {name_a}...")
    model_a = load_model(args.model_a, device)
    print(f"Loading {name_b}...")
    model_b = load_model(args.model_b, device)

    game_a = Game(seed=100)
    game_b = Game(seed=100)  # same seed for fair comparison

    state_a = np.array(game_a.reset())
    state_b = np.array(game_b.reset())

    speed = args.speed
    paused = False
    step = 0
    wins_a = 0
    wins_b = 0
    rounds = 0
    action_a = 0
    action_b = 0

    console = Console()
    keys = KeyReader()
    keys.start()

    try:
        with Live(console=console, refresh_per_second=30, screen=True) as live:
            while True:
                key = keys.get()
                if key == 'q' or key == 'Q':
                    break
                elif key == ' ':
                    paused = not paused
                elif key == '+' or key == '=':
                    speed = min(200, speed + 5)
                elif key == '-' or key == '_':
                    speed = max(1, speed - 5)
                elif key == 'r' or key == 'R':
                    seed = int(time.time()) % 100000
                    game_a = Game(seed=seed)
                    game_b = Game(seed=seed)
                    state_a = np.array(game_a.reset())
                    state_b = np.array(game_b.reset())
                    step = 0
                    action_a = 0
                    action_b = 0

                if not paused:
                    # Select actions
                    action_a = select_action(model_a, state_a, device)
                    action_b = select_action(model_b, state_b, device)

                    # Step both games
                    result_a = game_a.step(action_a)
                    result_b = game_b.step(action_b)
                    state_a = np.array(result_a[0])
                    state_b = np.array(result_b[0])
                    step += 1

                    # Check for game over — auto restart
                    done_a = result_a[2]
                    done_b = result_b[2]
                    if done_a and done_b:
                        sa = game_a.score
                        sb = game_b.score
                        if sa > sb:
                            wins_a += 1
                        elif sb > sa:
                            wins_b += 1
                        rounds += 1
                        # Auto restart after brief pause
                        time.sleep(0.5)
                        seed = int(time.time()) % 100000
                        game_a = Game(seed=seed)
                        game_b = Game(seed=seed)
                        state_a = np.array(game_a.reset())
                        state_b = np.array(game_b.reset())
                        step = 0
                    elif done_a and not done_b:
                        pass  # let B keep playing
                    elif done_b and not done_a:
                        pass  # let A keep playing

                # Render
                ent_a = game_a.get_entities()
                ent_b = game_b.get_entities()

                panel_a = build_panel(name_a, ent_a, action_a, step, args.width, args.height)
                panel_b = build_panel(name_b, ent_b, action_b, step, args.width, args.height)

                status_line = (
                    f"  Speed: {speed} tps   "
                    f"[Space] pause   [+/-] speed   [R] restart   [Q] quit   "
                    f"Wins: A={wins_a}  B={wins_b}  (rounds: {rounds})"
                )
                if paused:
                    status_line = "  [bold yellow]PAUSED[/]   " + status_line

                layout = Text.from_markup(
                    "\n" + status_line + "\n"
                )

                columns = Columns([panel_a, panel_b], padding=1)
                from rich.console import Group
                group = Group(columns, layout)
                live.update(group)

                if not paused:
                    time.sleep(1.0 / speed)
                else:
                    time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        print("\nReplay ended.")
        if rounds > 0:
            print(f"Results: {name_a} won {wins_a}, {name_b} won {wins_b} ({rounds} rounds)")


if __name__ == "__main__":
    main()
