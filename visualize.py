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
import subprocess
import time
import sys

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich import box
except ImportError:
    print("Install rich: pip install rich")
    sys.exit(1)


# =============================================================================
# Helpers
# =============================================================================
SPARK = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_EMPTY = "░"


def sparkline(values, width=30):
    if not values or len(values) < 2:
        return "[dim]" + "─" * width + "[/]"
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
        return "[green]" + SPARK[4] * len(buckets) + "[/]"
    return "[green]" + "".join(
        SPARK[min(7, int((v - mn) / rng * 7.99))] for v in buckets
    ) + "[/]"


def progress_bar(fraction, width=30, color="cyan"):
    fraction = max(0, min(1, fraction))
    filled = int(fraction * width)
    return f"[{color}]{BAR_FULL * filled}[/{color}][dim]{BAR_EMPTY * (width - filled)}[/]"


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


def trend_arrow(current, previous):
    if previous == 0:
        return "", "white"
    pct = (current - previous) / abs(previous) * 100
    if pct > 3:
        return f"↑ +{pct:.1f}%", "green"
    elif pct < -3:
        return f"↓ {pct:.1f}%", "red"
    else:
        return f"→ {pct:+.1f}%", "yellow"


def rolling_avg(values, window):
    if not values:
        return 0
    tail = values[-window:]
    return sum(tail) / len(tail)


# =============================================================================
# System metrics
# =============================================================================
_last_gpu_check = 0
_cached_gpu = (None, None)
_last_cpu_idle = None
_last_cpu_total = None


def get_gpu_stats():
    global _last_gpu_check, _cached_gpu
    now = time.time()
    if now - _last_gpu_check < 3:
        return _cached_gpu
    _last_gpu_check = now
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=2, stderr=subprocess.DEVNULL
        ).decode().strip()
        parts = [float(x.strip()) for x in out.split(',')]
        gpu_util = parts[0]
        vram_pct = parts[1] / parts[2] * 100 if parts[2] > 0 else 0
        _cached_gpu = (gpu_util, vram_pct)
    except Exception:
        _cached_gpu = (None, None)
    return _cached_gpu


def get_cpu_ram():
    # CPU from /proc/stat
    cpu_pct = None
    ram_pct = None
    global _last_cpu_idle, _last_cpu_total
    try:
        with open('/proc/stat', 'r') as f:
            line = f.readline()
        parts = [int(x) for x in line.split()[1:]]
        idle = parts[3]
        total = sum(parts)
        if _last_cpu_total is not None:
            d_total = total - _last_cpu_total
            d_idle = idle - _last_cpu_idle
            cpu_pct = (1 - d_idle / max(1, d_total)) * 100
        _last_cpu_idle = idle
        _last_cpu_total = total
    except Exception:
        pass
    # RAM from /proc/meminfo
    try:
        info = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(':')] = int(parts[1])
        total = info.get('MemTotal', 1)
        avail = info.get('MemAvailable', total)
        ram_pct = (1 - avail / total) * 100
    except Exception:
        pass
    return cpu_pct, ram_pct


# =============================================================================
# Read log
# =============================================================================
def read_log(path):
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
    return entries


# =============================================================================
# Dashboard builder
# =============================================================================
def build_dashboard(entries, total_episodes, window, start_time):
    term_w = 100
    term_h = 30
    try:
        sz = os.get_terminal_size()
        term_w = sz.columns
        term_h = sz.lines
    except OSError:
        pass

    if not entries:
        return Panel(
            Text("Waiting for training data...\n\nMake sure training is running.",
                 justify="center"),
            title="[bold white]eyeBMinvaders Training Dashboard[/]",
            border_style="blue", expand=True,
        )

    # Extract all fields
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

    n_eps = len(entries)
    current_ep = entries[-1].get("episode", n_eps)
    current_eps = all_epsilon[-1] if all_epsilon else 1.0
    elapsed = time.time() - start_time

    # Speed calculation from recent episodes
    ep_per_sec = current_ep / max(1, elapsed)
    remaining = total_episodes - current_ep
    eta = remaining / max(0.1, ep_per_sec)

    # Rolling averages
    avg_score = rolling_avg(all_scores, window)
    avg_level = rolling_avg(all_levels, window)
    best_score = max(all_scores)
    best_level = max(all_levels)
    avg_ekills = rolling_avg(all_ekills, window)
    avg_kkills = rolling_avg(all_kkills, window)
    avg_mshots = rolling_avg(all_mshots, window)
    avg_hits = rolling_avg(all_hits, window)
    avg_reward = rolling_avg(all_rewards, window)
    avg_steps = rolling_avg(all_steps, window)
    avg_lives = rolling_avg(all_lives, window)

    # Trend analysis (compare last 10K vs previous 10K)
    t_window = min(10000, n_eps // 2)
    if n_eps > t_window * 2:
        prev_scores = all_scores[-(t_window * 2):-t_window]
        curr_scores = all_scores[-t_window:]
        prev_levels = all_levels[-(t_window * 2):-t_window]
        curr_levels = all_levels[-t_window:]
        prev_hits = all_hits[-(t_window * 2):-t_window]
        curr_hits = all_hits[-t_window:]

        score_trend, score_color = trend_arrow(
            sum(curr_scores) / len(curr_scores),
            sum(prev_scores) / len(prev_scores))
        level_trend, level_color = trend_arrow(
            sum(curr_levels) / len(curr_levels),
            sum(prev_levels) / len(prev_levels))
        hits_trend, hits_color = trend_arrow(
            sum(curr_hits) / len(curr_hits),
            sum(prev_hits) / len(prev_hits))
        # For hits, lower is better — invert colors
        if "↓" in hits_trend:
            hits_color = "green"
            hits_trend += " (better)"
        elif "↑" in hits_trend:
            hits_color = "red"
            hits_trend += " (worse)"
    else:
        score_trend, score_color = "collecting...", "dim"
        level_trend, level_color = "collecting...", "dim"
        hits_trend, hits_color = "collecting...", "dim"

    # Auto status
    if n_eps > 20000:
        score_pct = 0
        if n_eps > t_window * 2:
            prev_avg = sum(all_scores[-(t_window * 2):-t_window]) / t_window
            curr_avg = sum(all_scores[-t_window:]) / t_window
            score_pct = (curr_avg - prev_avg) / max(1, abs(prev_avg)) * 100
        if abs(score_pct) < 3:
            status = "[yellow]Plateau[/] — try lower LR or larger network"
        elif score_pct > 0:
            status = "[green]Improving[/] — training is productive"
        else:
            status = "[red]Declining[/] — possible overfitting"
    elif n_eps > 5000:
        status = "[green]Learning[/] — agent exploring strategies"
    else:
        status = "[cyan]Warmup[/] — filling replay buffer"

    # Sparkline widths based on terminal
    sw = max(15, (term_w - 50) // 3)
    sw2 = max(12, sw - 5)

    # K/D ratio
    total_kills = sum(all_ekills) + sum(all_kkills)
    total_hits_sum = sum(all_hits)
    kd_ratio = total_kills / max(1, total_hits_sum)
    avg_survival = avg_steps * 0.0333  # seconds

    # System stats
    gpu_util, vram_pct = get_gpu_stats()
    cpu_pct, ram_pct = get_cpu_ram()

    # Bottleneck detection
    if gpu_util is not None and cpu_pct is not None:
        if gpu_util > 70 and cpu_pct < 50:
            bottleneck = "[yellow]GPU[/] (training)"
        elif cpu_pct > 70 and (gpu_util is None or gpu_util < 50):
            bottleneck = "[yellow]CPU[/] (game sim)"
        elif gpu_util > 60 and cpu_pct > 60:
            bottleneck = "[green]Balanced[/]"
        else:
            bottleneck = "[dim]Low utilization[/]"
    else:
        bottleneck = "[dim]N/A[/]"

    # === BUILD LAYOUT ===
    lines = []
    lines.append("")

    # Progress header
    ep_frac = min(1.0, current_ep / total_episodes)
    pb = progress_bar(ep_frac, max(20, term_w - 55), "cyan")
    lines.append(f"  [bold]Episode {current_ep:,}[/] / {total_episodes:,}  {pb}  {ep_frac*100:.1f}%")
    lines.append(f"  Elapsed: [bold]{fmt_time(elapsed)}[/]    "
                 f"ETA: [bold]{fmt_time(eta)}[/]    "
                 f"Speed: [bold]{ep_per_sec:.0f}[/] ep/s")
    lines.append("")

    # Two-column section: Performance (left) | System + Learning (right)
    # Build as aligned text since rich Columns inside Panel is tricky
    left_w = max(40, (term_w - 10) // 2)
    right_w = max(35, term_w - left_w - 10)

    # Performance section
    perf_lines = []
    perf_lines.append("  [bold green]Performance[/]")
    perf_lines.append("")
    perf_lines.append(f"    Score   {sparkline(all_scores, sw)}")
    perf_lines.append(f"    Avg: [bold]{avg_score:>8,.0f}[/]    Best: [bold yellow]{best_score:>9,}[/]")
    perf_lines.append("")
    perf_lines.append(f"    Level   {sparkline(all_levels, sw)}")
    perf_lines.append(f"    Avg: [bold]{avg_level:>8.1f}[/]    Best: [bold yellow]{best_level:>9}[/]")
    perf_lines.append("")
    perf_lines.append(f"    Reward  {sparkline(all_rewards, sw)}")
    perf_lines.append(f"    Avg: [bold]{avg_reward:>8.2f}[/]")
    perf_lines.append("")
    perf_lines.append(f"    Length  {sparkline(all_steps, sw)}")
    perf_lines.append(f"    Avg: [bold]{avg_steps:>8,.0f}[/] steps  ({avg_survival:.0f}s)")
    perf_lines.append("")

    # Combat section
    perf_lines.append("  [bold magenta]Combat[/] (rolling {})".format(window))
    perf_lines.append("")
    perf_lines.append(f"    Enemies killed   {avg_ekills:>5.1f}/ep  {sparkline(all_ekills, sw2)}")
    perf_lines.append(f"    Kamikazes killed {avg_kkills:>5.1f}/ep  {sparkline(all_kkills, sw2)}")
    perf_lines.append(f"    Missiles shot    {avg_mshots:>5.1f}/ep  {sparkline(all_mshots, sw2)}")
    perf_lines.append(f"    Times hit        {avg_hits:>5.1f}/ep  {sparkline(all_hits, sw2)}")
    perf_lines.append("")
    perf_lines.append(f"    K/D Ratio: [bold]{kd_ratio:.1f}[/]    "
                      f"Survival: [bold]{avg_survival:.0f}s[/] avg    "
                      f"Lives left: [bold]{avg_lives:.1f}[/]")

    # System section
    sys_lines = []
    sys_lines.append("  [bold blue]System[/]")
    sys_lines.append("")
    if gpu_util is not None:
        sys_lines.append(f"    GPU:  {progress_bar(gpu_util / 100, 20, 'green' if gpu_util < 60 else 'yellow' if gpu_util < 85 else 'red')}  {gpu_util:.0f}%")
    else:
        sys_lines.append("    GPU:  [dim]N/A[/]")
    if vram_pct is not None:
        sys_lines.append(f"    VRAM: {progress_bar(vram_pct / 100, 20, 'cyan')}  {vram_pct:.0f}%")
    else:
        sys_lines.append("    VRAM: [dim]N/A[/]")
    if cpu_pct is not None:
        sys_lines.append(f"    CPU:  {progress_bar(cpu_pct / 100, 20, 'green' if cpu_pct < 60 else 'yellow' if cpu_pct < 85 else 'red')}  {cpu_pct:.0f}%")
    else:
        sys_lines.append("    CPU:  [dim]N/A[/]")
    if ram_pct is not None:
        sys_lines.append(f"    RAM:  {progress_bar(ram_pct / 100, 20, 'cyan')}  {ram_pct:.0f}%")
    else:
        sys_lines.append("    RAM:  [dim]N/A[/]")
    sys_lines.append("")
    sys_lines.append(f"    Bottleneck: {bottleneck}")
    sys_lines.append("")

    # Learning section
    sys_lines.append("  [bold blue]Learning[/]")
    sys_lines.append("")
    eps_frac = 1.0 - current_eps
    eps_label = " (floor)" if current_eps <= 0.021 else ""
    sys_lines.append(f"    Epsilon: [bold]{current_eps:.4f}[/]{eps_label}")
    sys_lines.append(f"    {progress_bar(eps_frac, 25, 'green')}")
    sys_lines.append(f"    Explore {current_eps * 100:.0f}% ──── Exploit {eps_frac * 100:.0f}%")
    sys_lines.append("")
    sys_lines.append("    [bold]Trend[/] (last {} episodes):".format(t_window))
    sys_lines.append(f"    Score: [{score_color}]{score_trend}[/{score_color}]")
    sys_lines.append(f"    Level: [{level_color}]{level_trend}[/{level_color}]")
    sys_lines.append(f"    Hits:  [{hits_color}]{hits_trend}[/{hits_color}]")
    sys_lines.append("")
    sys_lines.append(f"    Status: {status}")

    # Merge columns — interleave left and right
    max_rows = max(len(perf_lines), len(sys_lines))
    while len(perf_lines) < max_rows:
        perf_lines.append("")
    while len(sys_lines) < max_rows:
        sys_lines.append("")

    for i in range(max_rows):
        lines.append(perf_lines[i])

    lines.append("")
    lines.append("  " + "─" * (term_w - 8))
    lines.append("")

    for i in range(len(sys_lines)):
        lines.append(sys_lines[i])

    lines.append("")

    # Pad to fill terminal height
    content = "\n".join(lines)
    content_h = len(lines)
    pad_needed = max(0, term_h - content_h - 4)
    content = content + "\n" * pad_needed

    return Panel(
        Text.from_markup(content),
        title="[bold white]eyeBMinvaders Training Dashboard[/]",
        border_style="blue",
        expand=True,
    )


# =============================================================================
# Main
# =============================================================================
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

    console = Console()
    console.print(f"[bold]Watching:[/] {args.log}")
    console.print(f"[bold]Target:[/] {args.episodes:,} episodes")
    console.print("Press Ctrl+C to exit\n")

    start_time = time.time()

    # Try to estimate start time from first entry
    entries = read_log(args.log)
    if entries:
        first_ep = entries[0].get("episode", 0)
        last_ep = entries[-1].get("episode", 0)
        if last_ep > first_ep:
            # Approximate: assume log has been growing at current rate
            log_mtime = os.path.getmtime(args.log)
            start_time = log_mtime - (last_ep / max(1, last_ep - first_ep + 1))

    try:
        with Live(build_dashboard([], args.episodes, args.window, start_time),
                  console=console, refresh_per_second=1, screen=True) as live:
            while True:
                entries = read_log(args.log)
                dashboard = build_dashboard(entries, args.episodes, args.window, start_time)
                live.update(dashboard)
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        console.print("\n[bold]Dashboard closed.[/]")


if __name__ == "__main__":
    main()
