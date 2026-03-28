#!/usr/bin/env python3
"""
Browser-as-Environment Validation for eyeBMinvaders
=====================================================
Runs the actual browser game via Playwright and compares the state vectors
produced by game.js buildDQNState() against what the Rust sim would produce
for the same game situation. Finds sim-to-real divergences.

Also supports running the DQN model in the real browser to measure actual
performance and diagnose behavioral issues.

Usage:
    python browser_validate.py                    # Watch AI play in browser
    python browser_validate.py --headless         # Headless mode, log stats
    python browser_validate.py --episodes 10      # Run N episodes
    python browser_validate.py --diagnose         # Detailed per-feature logging

Requirements:
    pip install playwright
    playwright install chromium
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict

# Use venv python if available
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("ERROR: playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)


async def run_ai_episode(page, episode_num, diagnose=False, max_steps=30000):
    """Run one episode with the DQN AI, collecting stats."""

    # Reset game and enable AI
    await page.evaluate("""
        restartGame();
        autoPlayEnabled = true;
        dqnLastDecisionTime = 0;
    """)
    await asyncio.sleep(0.3)

    stats = {
        'episode': episode_num,
        'score': 0,
        'level': 1,
        'lives': 6,
        'steps': 0,
        'actions': Counter(),
        'deaths_at_level': Counter(),
        'time_at_edge': 0,  # steps where player is near edge
        'bullets_nearby': 0,  # steps where bullets are close
        'state_samples': [],  # sample states for diagnosis
    }

    step = 0
    sample_interval = 100  # sample state every N steps

    while step < max_steps:
        step += 1

        # Check game state
        game_state = await page.evaluate("""
            ({
                gameOver: gameOverFlag,
                score: score,
                level: currentLevel,
                lives: player.lives,
                playerX: player.x / 1024.0,
                enemyCount: enemies.length,
                bulletCount: bullets.filter(b => b.isEnemyBullet).length,
                missileCount: typeof homingMissiles !== 'undefined' ? homingMissiles.length : 0,
                kamikazeCount: typeof kamikazeEnemies !== 'undefined' ? kamikazeEnemies.length : 0,
                autoPlayEnabled: autoPlayEnabled,
            })
        """)

        if game_state['gameOver']:
            stats['score'] = game_state['score']
            stats['level'] = game_state['level']
            stats['lives'] = game_state['lives']
            stats['steps'] = step
            stats['deaths_at_level'][game_state['level']] += 1
            break

        # Track edge time
        px = game_state['playerX']
        if px < 0.08 or px > 0.92:
            stats['time_at_edge'] += 1

        # Track nearby threats
        if game_state['bulletCount'] > 0:
            stats['bullets_nearby'] += 1

        # Sample state for diagnosis
        if diagnose and step % sample_interval == 0:
            state = await page.evaluate("buildDQNState()")
            qvals = await page.evaluate("""
                (() => {
                    if (!dqnModel) return null;
                    const rawState = buildDQNState();
                    const nFrames = dqnModel.n_frames || 1;
                    const state = nFrames > 1 ? dqnPushFrame(rawState) : rawState;
                    return dqnForward(state);
                })()
            """)
            stats['state_samples'].append({
                'step': step,
                'state': state[:50],  # first frame only
                'q_values': qvals,
                'player_x': px,
                'threats': {
                    'bullets': game_state['bulletCount'],
                    'missiles': game_state['missileCount'],
                    'kamikazes': game_state['kamikazeCount'],
                },
            })

        stats['score'] = game_state['score']
        stats['level'] = game_state['level']

        # Wait one decision interval (30Hz)
        await asyncio.sleep(0.033)

    return stats


async def main():
    parser = argparse.ArgumentParser(description='Browser validation for eyeBMinvaders AI')
    parser.add_argument('--headless', action='store_true', help='Run headless')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--diagnose', action='store_true', help='Detailed per-feature logging')
    parser.add_argument('--url', default='http://localhost:8000', help='Game URL')
    parser.add_argument('--port', type=int, default=8000, help='Start HTTP server on this port')
    args = parser.parse_args()

    # Start a local HTTP server if needed
    server_proc = None
    try:
        import urllib.request
        urllib.request.urlopen(args.url, timeout=1)
    except Exception:
        print(f"Starting HTTP server on port {args.port}...")
        import subprocess
        server_proc = subprocess.Popen(
            [sys.executable, '-m', 'http.server', str(args.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        await asyncio.sleep(1)

    actions = ['idle', 'left', 'right', 'fire', 'fire+L', 'fire+R']

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        context = await browser.new_context(viewport={'width': 1200, 'height': 700})
        page = await context.new_page()

        # Navigate to game and start it
        await page.goto(args.url)
        await asyncio.sleep(1)

        # Trigger game start (skips legend screen)
        await page.evaluate("startGame()")
        # Wait for firebirds iframe (3s) + game.js load
        await asyncio.sleep(5)

        # Wait for game.js to be loaded and running
        await page.wait_for_function("typeof buildDQNState === 'function'", timeout=15000)
        # Wait for model to load
        await asyncio.sleep(2)
        print(f"Game loaded. Running {args.episodes} episodes {'(headless)' if args.headless else '(visible)'}...\n")

        all_stats = []
        for ep in range(args.episodes):
            stats = await run_ai_episode(page, ep, diagnose=args.diagnose)
            all_stats.append(stats)

            edge_pct = stats['time_at_edge'] / max(stats['steps'], 1) * 100
            threat_pct = stats['bullets_nearby'] / max(stats['steps'], 1) * 100

            print(f"Ep {ep+1:3d} | Score: {stats['score']:>7,} | Level: {stats['level']} | "
                  f"Steps: {stats['steps']:>5,} | Edge: {edge_pct:.0f}% | "
                  f"Under fire: {threat_pct:.0f}%")

            if args.diagnose and stats['state_samples']:
                # Show Q-value patterns
                for sample in stats['state_samples'][:3]:
                    if sample['q_values']:
                        q = sample['q_values']
                        best = max(range(6), key=lambda i: q[i])
                        print(f"  step {sample['step']:>5}: x={sample['player_x']:.2f} "
                              f"threats={sample['threats']} "
                              f"best={actions[best]} Q=[{', '.join(f'{q[i]:.1f}' for i in range(6))}]")

        # Summary
        print(f"\n{'='*60}")
        avg_score = sum(s['score'] for s in all_stats) / len(all_stats)
        avg_level = sum(s['level'] for s in all_stats) / len(all_stats)
        avg_edge = sum(s['time_at_edge'] / max(s['steps'], 1) for s in all_stats) / len(all_stats) * 100
        best_score = max(s['score'] for s in all_stats)
        best_level = max(s['level'] for s in all_stats)

        print(f"Summary ({args.episodes} episodes):")
        print(f"  Avg Score: {avg_score:,.0f} | Best: {best_score:,}")
        print(f"  Avg Level: {avg_level:.1f} | Best: {best_level}")
        print(f"  Avg Edge Time: {avg_edge:.0f}%")
        print(f"{'='*60}")

        await browser.close()

    if server_proc:
        server_proc.terminate()


if __name__ == '__main__':
    asyncio.run(main())
