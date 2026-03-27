#!/usr/bin/env python3
"""
Live TUI Training Dashboard for eyeBMinvaders
Run in a second terminal while training is in progress.

Usage:
    python3 visualize.py
    python3 visualize.py --log models/training_events.jsonl --episodes 200000
"""

import argparse
import json
import os
import time
import sys

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.console import Group
    from rich import box
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)


SPARK = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_EMPTY = "░"


def sparkline(values, width=40):
    if not values or len(values) < 2:
        return "─" * width
    # Bucket values into `width` bins
    n = len(values)
    if n <= width:
        buckets = values
    else:
        step = n / width
        buckets = []
        for i in range(width):
            start = int(i * step)
            end = int((i + 1) * step)
            buckets.append(sum(values[start:end]) / max(1, end - start))
    mn = min(buckets)
    mx = max(buckets)
    rng = mx - mn
    if rng == 0:
        return SPARK[4] * len(buckets)
    return "".join(SPARK[min(7, int((v - mn) / rng * 7.99))] for v in buckets)


def progress_bar(fraction, width=20):
    filled = int(fraction * width)
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def read_log(path, max_lines=None):
    """Read JSONL log file, return list of dicts."""
    if not os.path.exists(path):
        return []
    entries = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except IOError:
        pass
    if max_lines and len(entries) > max_lines:
        return entries[-max_lines:]
    return entries


def rolling_avg(values, window):
    if not values:
        return 0
    tail = values[-window:]
    return sum(tail) / len(tail)


def build_dashboard(entries, total_episodes, window):
    if not entries:
        return Panel(
            Text("Waiting for training data...\n\nMake sure training is running and logging to the JSONL file.",
                 justify="center"),
            title="eyeBMinvaders Training Dashboard",
            border_style="blue",
        )

    # Extract fields
    episodes = [e.get("episode", 0) for e in entries]
    all_scores = [e.get("score", 0) for e in entries]
    all_levels = [e.get("level", 1) for e in entries]
    all_ekills = [e.get("enemies_killed", 0) for e in entries]
    all_kkills = [e.get("kamikazes_killed", 0) for e in entries]
    all_mshots = [e.get("missiles_shot", 0) for e in entries]
    all_hits = [e.get("times_hit", 0) for e in entries]
    all_epsilon = [e.get("epsilon", 1.0) for e in entries]
    all_rewards = [e.get("reward", 0) for e in entries]
    all_steps = [e.get("steps", 0) for e in entries]
    all_lives = [e.get("lives_left", 0) for e in entries]

    current_ep = episodes[-1] if episodes else 0
    current_eps = all_epsilon[-1] if all_epsilon else 1.0

    # Compute ep/s from recent entries
    recent = entries[-500:] if len(entries) > 500 else entries
    if len(recent) > 1:
        ep_span = recent[-1].get("episode", 0) - recent[0].get("episode", 0)
        # Use file-based timing: approximate from entry count and known refresh
        ep_per_sec = ep_span / max(1, len(recent)) * 50  # rough estimate
    else:
        ep_per_sec = 0

    # Rolling averages
    avg_score = rolling_avg(all_scores, window)
    avg_level = rolling_avg(all_levels, window)
    best_score = max(all_scores) if all_scores else 0
    best_level = max(all_levels) if all_levels else 1
    avg_ekills = rolling_avg(all_ekills, window)
    avg_kkills = rolling_avg(all_kkills, window)
    avg_mshots = rolling_avg(all_mshots, window)
    avg_hits = rolling_avg(all_hits, window)
    avg_reward = rolling_avg(all_rewards, window)
    avg_steps = rolling_avg(all_steps, window)
    avg_lives = rolling_avg(all_lives, window)

    # Sparklines — use wide charts
    sw = 60  # sparkline width
    sw2 = 50  # combat sparkline width
    spark_scores = sparkline(all_scores, sw)
    spark_levels = sparkline(all_levels, sw)
    spark_ekills = sparkline(all_ekills, sw2)
    spark_kkills = sparkline(all_kkills, sw2)
    spark_mshots = sparkline(all_mshots, sw2)
    spark_hits = sparkline(all_hits, sw2)
    spark_reward = sparkline(all_rewards, sw)
    spark_steps = sparkline(all_steps, sw)

    # Progress
    ep_frac = min(1.0, current_ep / total_episodes) if total_episodes > 0 else 0
    eps_frac = 1.0 - current_eps

    # ETA
    elapsed_entries = len(entries)
    if elapsed_entries > 100 and current_ep > 0:
        # Estimate from log growth rate
        remaining = total_episodes - current_ep
        # Use a simple heuristic: we know roughly when entries started
        eta_str = f"~{remaining / max(1, current_ep) * (time.time() % 86400) / 3600:.0f}h" if remaining > 0 else "done"
    else:
        eta_str = "calculating..."

    # Build layout
    lines = []

    # Header stats
    lines.append("")
    lines.append(f"  [bold cyan]Episode:[/] {current_ep:>8,}    "
                 f"[bold cyan]Epsilon:[/] {current_eps:.4f}    "
                 f"[bold cyan]Avg Reward:[/] {avg_reward:.2f}")
    lines.append("")

    # Score & Level sparklines
    lines.append(f"  [bold green]Score[/]  {spark_scores}  "
                 f"Avg: [bold]{avg_score:,.0f}[/]  Best: [bold yellow]{best_score:,}[/]")
    lines.append(f"  [bold green]Level[/]  {spark_levels}  "
                 f"Avg: [bold]{avg_level:.1f}[/]     Best: [bold yellow]{best_level}[/]")
    lines.append(f"  [bold green]Steps[/]  {spark_steps}  "
                 f"Avg: [bold]{avg_steps:.0f}[/]")
    lines.append(f"  [bold green]Reward[/] {spark_reward}  "
                 f"Avg: [bold]{avg_reward:.2f}[/]")
    lines.append("")

    # Combat stats
    lines.append(f"  [bold magenta]Combat Stats[/] (rolling {window})")
    lines.append(f"    Enemies killed:   {avg_ekills:>5.1f}/ep  {spark_ekills}")
    lines.append(f"    Kamikazes killed: {avg_kkills:>5.1f}/ep  {spark_kkills}")
    lines.append(f"    Missiles shot:    {avg_mshots:>5.1f}/ep  {spark_mshots}")
    lines.append(f"    Times hit:        {avg_hits:>5.1f}/ep  {spark_hits}  [dim](lower=better)[/]")
    lines.append(f"    Lives left:       {avg_lives:>5.1f}/ep")
    lines.append("")

    # Learning progress
    ep_bar = progress_bar(ep_frac, 50)
    eps_bar = progress_bar(eps_frac, 50)
    lines.append(f"  [bold blue]Learning Progress[/]")
    lines.append(f"    Exploration: {eps_bar}  {eps_frac*100:.0f}% exploiting")
    lines.append(f"    Episodes:    {ep_bar}  {ep_frac*100:.1f}%  ({current_ep:,}/{total_episodes:,})")
    lines.append("")

    content = "\n".join(lines)

    # Pad to fill terminal height
    try:
        term_h = os.get_terminal_size().lines
    except OSError:
        term_h = 40
    content_lines = len(lines)
    pad_needed = max(0, term_h - content_lines - 4)  # -4 for panel border + margins
    pad_top = pad_needed // 3
    pad_bottom = pad_needed - pad_top
    content = "\n" * pad_top + content + "\n" * pad_bottom

    return Panel(
        Text.from_markup(content),
        title="[bold white]eyeBMinvaders Training Dashboard[/]",
        border_style="blue",
        padding=(0, 2),
        expand=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--log", default="models/training_events.jsonl",
                        help="Path to JSONL log file")
    parser.add_argument("--episodes", type=int, default=200_000,
                        help="Total episodes for progress bar")
    parser.add_argument("--window", type=int, default=1000,
                        help="Rolling average window size")
    parser.add_argument("--refresh", type=float, default=2.0,
                        help="Refresh interval in seconds")
    args = parser.parse_args()

    print(f"Watching: {args.log}")
    print(f"Target episodes: {args.episodes:,}")
    print("Press Ctrl+C to exit\n")

    try:
        with Live(build_dashboard([], args.episodes, args.window),
                  refresh_per_second=0.5, screen=True) as live:
            while True:
                entries = read_log(args.log)
                dashboard = build_dashboard(entries, args.episodes, args.window)
                live.update(dashboard)
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nDashboard closed.")


if __name__ == "__main__":
    main()
