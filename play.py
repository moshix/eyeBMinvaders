#!/usr/bin/env python3
"""
Terminal UI for eyeBMinvaders — play or watch the AI.

Usage:
    python3 play.py                           # Play with keyboard
    python3 play.py --ai models/model_best.pt # Watch AI play
    python3 play.py --ai models/model_best.pt --speed 30

Controls:
    Arrow keys / A,D  Move
    Space             Fire
    F1                Toggle AI mode (if model loaded)
    +/-               Speed up/down
    P                 Pause
    R                 Restart
    Q                 Quit
"""

import argparse
import os
import sys
import time
import threading
import numpy as np

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console, Group
    from rich.table import Table
except ImportError:
    print("Rich required: pip install rich")
    sys.exit(1)

try:
    from game_sim import Game
except ImportError:
    print("Rust game_sim not found. Build with: cd game_sim && maturin develop --release")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

GAME_W = 1024
GAME_H = 576
ACTION_NAMES = ['idle', 'left', 'right', 'fire', 'fire+left', 'fire+right']

# Entity sprites: (character, style)
SPRITES = {
    'player': ('A', 'bold white on green'),
    'player_hit': ('X', 'bold white on red'),
    'enemy_full': ('W', 'bold red'),
    'enemy_hit': ('w', 'bold yellow'),
    'bullet_player': ('|', 'bold cyan'),
    'bullet_enemy': ('.', 'bold red'),
    'kamikaze': ('K', 'bold white on magenta'),
    'missile': ('*', 'bold white on red'),
    'monster': ('M', 'bold black on yellow'),
    'monster2': ('W', 'bold white on cyan'),
    'wall_ok': ('#', 'bold green'),
    'wall_dmg': ('#', 'bold yellow'),
    'wall_crit': ('#', 'bold red'),
}


# =============================================================================
# DQN (matches training architecture — variable hidden sizes)
# =============================================================================
if HAS_TORCH:
    class DQN(nn.Module):
        def __init__(self, state_size=45, action_size=6, hidden_sizes=None):
            super().__init__()
            sizes = hidden_sizes or [256, 256, 128]
            layers = []
            prev = state_size
            for h in sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, action_size))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    def load_model(path, device='cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if 'policy_net' in checkpoint:
            state_dict = checkpoint['policy_net']
        else:
            state_dict = checkpoint
        # Normalize keys (strip _orig_mod prefix)
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Detect architecture from weights — find all Linear weight layers
        layer_sizes = []
        for key in sorted(state_dict.keys()):
            if key.startswith('net.') and key.endswith('.weight') and len(state_dict[key].shape) == 2:
                layer_sizes.append(state_dict[key].shape)

        if layer_sizes:
            input_size = layer_sizes[0][1]
            hidden = [s[0] for s in layer_sizes[:-1]]
            output_size = layer_sizes[-1][0]
        else:
            input_size, hidden, output_size = 45, [512, 256, 128], 6

        # Detect LayerNorm
        has_ln = any('LayerNorm' in k or 'layernorm' in k.lower() for k in state_dict)
        model = DQN(input_size, output_size, hidden).to(device)
        # If checkpoint has LayerNorm but our DQN doesn't, rebuild with it
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Try with LayerNorm
            sizes = hidden or [512, 256, 128]
            layers = []
            prev = input_size
            for h in sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                if has_ln:
                    layers.append(nn.LayerNorm(h))
                prev = h
            layers.append(nn.Linear(prev, output_size))
            model = nn.Sequential(*layers).to(device)
            # Wrap in a namespace-compatible object
            class Wrapper(nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net
                def forward(self, x):
                    return self.net(x)
            model = Wrapper(model).to(device)
            model.load_state_dict(state_dict)
        model.eval()
        return model


# =============================================================================
# ASCII Renderer (larger, colored)
# =============================================================================
def render_game(entities, width=60, height=24, return_grid=False):
    """Render game state as a rich.text.Text object with proper alignment."""
    grid = [[None] * width for _ in range(height)]

    def sx(x):
        return max(0, min(width - 1, int(x / GAME_W * (width - 1))))

    def sy(y):
        return max(0, min(height - 1, int(y / GAME_H * (height - 1))))

    def place(gx, gy, sprite_key):
        if 0 <= gy < height and 0 <= gx < width:
            grid[gy][gx] = sprite_key

    # Walls
    for wx, hc, mh in entities.get('walls', []):
        wy = sy(501)
        x1 = sx(wx)
        x2 = sx(wx + 58)
        for x in range(x1, min(x2 + 1, width)):
            if hc < 4:
                place(x, wy, 'wall_ok')
            elif hc < 8:
                place(x, wy, 'wall_dmg')
            else:
                place(x, wy, 'wall_crit')

    def place_rect(x_game, y_game, w_game, h_game, sprite_key):
        """Place entity using full width but only 1 row height."""
        x1 = sx(x_game)
        x2 = sx(x_game + w_game)
        # Use vertical CENTER for y — single row only
        gy = sy(y_game + h_game / 2)
        for gx in range(x1, x2 + 1):
            if 0 <= gy < height and 0 <= gx < width:
                grid[gy][gx] = sprite_key

    # Enemies (43x43)
    for ex, ey, hits in entities.get('enemies', []):
        place_rect(ex, ey, 43, 43, 'enemy_full' if hits == 0 else 'enemy_hit')

    # Bullets (small, single pixel is fine)
    for bx, by, is_enemy in entities.get('bullets', []):
        gx, gy = sx(bx), sy(by)
        place(gx, gy, 'bullet_enemy' if is_enemy else 'bullet_player')

    # Kamikazes (43x43)
    for kx, ky in entities.get('kamikazes', []):
        place_rect(kx, ky, 43, 43, 'kamikaze')

    # Missiles (57x57)
    for mx, my in entities.get('missiles', []):
        place_rect(mx, my, 57, 57, 'missile')

    # Monster (56x56)
    m = entities.get('monster')
    if m is not None:
        place_rect(m[0], m[1], 56, 56, 'monster')

    # Monster2 (56x56)
    m2 = entities.get('monster2')
    if m2 is not None:
        place_rect(m2[0], m2[1], 56, 56, 'monster2')

    # Player (48x48, drawn last)
    px = entities.get('player_x', 512)
    py_val = entities.get('player_y', 546)
    skey = 'player_hit' if entities.get('is_hit') else 'player'
    place_rect(px, py_val, 48, 48, skey)

    # Build rich Text object — each char is exactly 1 column wide
    result = Text()
    for y in range(height):
        for x in range(width):
            key = grid[y][x]
            if key:
                ch, style = SPRITES[key]
                result.append(ch, style=style)
            else:
                result.append(' ')
        if y < height - 1:
            result.append('\n')

    if return_grid:
        # Convert grid to compact ASCII for debug logging
        ascii_grid = []
        for y in range(height):
            row = ''
            for x in range(width):
                key = grid[y][x]
                row += SPRITES[key][0] if key else '.'
            ascii_grid.append(row)
        return result, ascii_grid
    return result


# =============================================================================
# Keyboard input (non-blocking, handles arrow keys)
# =============================================================================
class KeyReader:
    def __init__(self):
        self.keys_pressed = set()
        self.last_special = None
        self._stop = False
        self._thread = None
        self._old_settings = None

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
                if not ch:
                    continue
                if ch == '\x1b':
                    # Escape sequence (arrow keys)
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A':
                            self.keys_pressed.add('up')
                        elif ch3 == 'B':
                            self.keys_pressed.add('down')
                        elif ch3 == 'C':
                            self.keys_pressed.add('right')
                        elif ch3 == 'D':
                            self.keys_pressed.add('left')
                    elif ch2 == 'O':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'P':  # F1
                            self.last_special = 'f1'
                    else:
                        self.last_special = 'esc'
                else:
                    self.keys_pressed.add(ch)
            except Exception:
                break

    def get_action(self):
        """Convert held keys to game action.
        Returns: action_id (0-5) based on current keys."""
        keys = self.keys_pressed.copy()
        self.keys_pressed.clear()

        fire = ' ' in keys
        left = 'a' in keys or 'left' in keys
        right = 'd' in keys or 'right' in keys

        if fire and left:
            return 4  # fire+left
        elif fire and right:
            return 5  # fire+right
        elif fire:
            return 3  # fire
        elif left:
            return 1  # left
        elif right:
            return 2  # right
        return 0  # idle

    def get_special(self):
        s = self.last_special
        self.last_special = None
        return s

    def has_key(self, k):
        if k in self.keys_pressed:
            self.keys_pressed.discard(k)
            return True
        return False

    def stop(self):
        import termios
        self._stop = True
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Play eyeBMinvaders in the terminal")
    parser.add_argument("--ai", type=str, default=None,
                        help="Path to model checkpoint for AI mode")
    parser.add_argument("--speed", type=int, default=15,
                        help="Game speed in ticks per second (default: 15)")
    parser.add_argument("--device", default="cpu",
                        help="Device for AI inference (default: cpu)")
    parser.add_argument("--width", type=int, default=0,
                        help="Field width in chars (default: auto-fit terminal)")
    parser.add_argument("--height", type=int, default=0,
                        help="Field height in chars (default: auto-fit terminal)")
    parser.add_argument("--debug", type=str, nargs='?', const='play_debug.log', default=None,
                        help="Log entity data + grid to file each second (default: play_debug.log)")
    args = parser.parse_args()

    model = None
    ai_mode = False
    if args.ai and HAS_TORCH:
        print(f"Loading model: {args.ai}")
        model = load_model(args.ai, args.device)
        ai_mode = True
        print("AI mode active. Press F1 to toggle.")
    elif args.ai and not HAS_TORCH:
        print("PyTorch not found — playing in manual mode")

    debug_file = None
    if args.debug:
        debug_file = open(args.debug, 'w')
        print(f"Debug logging to: {args.debug}")

    game = Game(seed=int(time.time()) % 100000)
    state = np.array(game.reset())
    speed = args.speed
    paused = False
    step = 0
    games_played = 0
    high_score = 0

    console = Console()
    # Auto-fit terminal size (leave room for panel border + header/footer)
    term_w, term_h = os.get_terminal_size()
    field_w = args.width if args.width > 0 else max(40, term_w - 4)
    field_h = args.height if args.height > 0 else max(16, term_h - 14)
    keys = KeyReader()
    keys.start()

    try:
        with Live(console=console, refresh_per_second=30, screen=True) as live:
            while True:
                # Handle special keys
                special = keys.get_special()
                if special == 'f1' and model:
                    ai_mode = not ai_mode
                elif special == 'esc':
                    break

                if keys.has_key('q') or keys.has_key('Q'):
                    break
                if keys.has_key('p') or keys.has_key('P'):
                    paused = not paused
                if keys.has_key('+') or keys.has_key('='):
                    speed = min(200, speed + 5)
                if keys.has_key('-') or keys.has_key('_'):
                    speed = max(1, speed - 5)
                if keys.has_key('r') or keys.has_key('R'):
                    game = Game(seed=int(time.time()) % 100000)
                    state = np.array(game.reset())
                    step = 0

                if not paused:
                    # Get action
                    if ai_mode and model:
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(args.device)
                            q_values = model(state_t)
                            action = q_values.argmax(dim=1).item()
                    else:
                        action = keys.get_action()

                    # Step game
                    result = game.step(action)
                    state = np.array(result[0])
                    done = result[2]
                    step += 1

                    if done:
                        end_entities = game.get_entities()
                        end_score = end_entities.get('score', 0)
                        if end_score > high_score:
                            high_score = end_score
                        games_played += 1
                        time.sleep(0.5)
                        game = Game(seed=int(time.time()) % 100000)
                        state = np.array(game.reset())
                        step = 0

                # Render
                entities = game.get_entities()
                if debug_file and step % max(1, speed) == 0:
                    field, debug_grid = render_game(entities, field_w, field_h, return_grid=True)
                    import json as _json
                    bullets = entities.get('bullets', [])
                    debug_entry = {
                        "frame": step,
                        "term": [term_w, term_h],
                        "field": [field_w, field_h],
                        "score": entities.get('score', 0),
                        "level": entities.get('level', 1),
                        "counts": {
                            "enemies": len(entities.get('enemies', [])),
                            "e_bullets": len([b for b in bullets if b[2]]),
                            "p_bullets": len([b for b in bullets if not b[2]]),
                            "kamikazes": len(entities.get('kamikazes', [])),
                            "missiles": len(entities.get('missiles', [])),
                            "monster": entities.get('monster') is not None,
                            "monster2": entities.get('monster2') is not None,
                        },
                        "player": [round(entities.get('player_x', 0), 1),
                                   round(entities.get('player_y', 0), 1)],
                        "enemies_sample": [[round(e[0], 1), round(e[1], 1)]
                                           for e in entities.get('enemies', [])[:3]],
                        "action": ACTION_NAMES[action] if not paused else '-',
                        "ai": ai_mode,
                        "grid": debug_grid,
                    }
                    debug_file.write(_json.dumps(debug_entry) + '\n')
                    debug_file.flush()
                else:
                    field = render_game(entities, field_w, field_h)

                score = entities.get('score', 0)
                level = entities.get('level', 1)
                lives = entities.get('lives', 0)
                game_over = entities.get('game_over', False)

                mode_str = 'AI' if ai_mode else 'HUMAN'
                action_name = ACTION_NAMES[action] if not paused else '-'

                # Build header and footer as Text objects
                header = Text()
                header.append(f"  Score: ")
                header.append(f"{score:>8,}", style="bold yellow")
                header.append(f"  Level: ")
                header.append(f"{level}", style="bold cyan")
                header.append("  ")
                for i in range(6):
                    if i < lives:
                        header.append("♥", style="bold red")
                    else:
                        header.append("♡", style="dim")
                header.append(f"  Mode: ")
                header.append(mode_str, style="bold cyan" if ai_mode else "bold green")
                if game_over:
                    header.append("  GAME OVER", style="bold red")

                footer = Text()
                footer.append(f"  Action: ")
                footer.append(f"{action_name:<12}", style="bold")
                footer.append(f"Step: {step:,}  Speed: {speed} tps  High: ")
                footer.append(f"{high_score:,}", style="yellow")
                footer.append(f"  Games: {games_played}")
                n_enemies = len(entities.get('enemies', []))
                n_bullets = len([b for b in entities.get('bullets', []) if b[2]])  # enemy bullets
                footer.append(f"  Enemies: {n_enemies}  EBullets: {n_bullets}")

                # Combine into panel content
                content = Text()
                content.append_text(header)
                content.append("\n\n")
                content.append_text(field)
                content.append("\n\n")
                content.append_text(footer)

                panel = Panel(
                    content,
                    title="[bold white]eyeBMinvaders[/]",
                    subtitle="[dim]Arrow/AD:move  Space:fire  F1:AI  P:pause  +/-:speed  R:restart  Q:quit[/]",
                    border_style="cyan" if not game_over else "red",
                    width=field_w + 4,
                    height=term_h,
                )

                if paused:
                    pause_text = Text("\n  PAUSED\n", style="bold yellow")
                    live.update(Group(panel, pause_text))
                else:
                    live.update(panel)

                if not paused:
                    time.sleep(1.0 / speed)
                else:
                    time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        if debug_file:
            debug_file.close()
            print(f"\nDebug log written to: {args.debug}")
        print(f"\nGame over! High score: {high_score:,} across {games_played} games.")


if __name__ == "__main__":
    main()
