#!/bin/bash
# =============================================================================
# Continuous Training Loop: GPU ↔ Browser Feedback
# =============================================================================
#
# This script creates a continuous improvement cycle:
#
# 1. GPU trains DQN model (fast, ~30 ep/s)
# 2. Export best model → model_weights.json
# 3. Browser picks up new model (DQN AI plays with it)
# 4. Browser records gameplay (W key) → gameplay_data.jsonl
# 5. Behavioral cloning refines model from recordings
# 6. Repeat
#
# Usage:
#   ./train_loop.sh                  # Run on GPU machine
#   ./train_loop.sh --cycles 10      # Run 10 cycles
#   ./train_loop.sh --bc-only        # Skip GPU training, just do BC
#
# Prerequisites:
#   - serve.py running on the browser machine
#   - Browser open with 0 + W active
#   - SSH access to GPU machine (if running remotely)

set -euo pipefail

CYCLES=${1:-5}
GPU_EPISODES=100000
BC_EPOCHS=20
EXPORT_EVERY=20000
MODEL_DIR="models"
GAMEPLAY_FILE="$MODEL_DIR/gameplay_data.jsonl"

echo "========================================"
echo "  Continuous Training Loop"
echo "  Cycles: $CYCLES"
echo "  GPU episodes per cycle: $GPU_EPISODES"
echo "  BC epochs per cycle: $BC_EPOCHS"
echo "========================================"

for cycle in $(seq 1 $CYCLES); do
    echo ""
    echo "=== CYCLE $cycle / $CYCLES ==="
    echo ""

    # Step 1: GPU DQN Training
    echo "[1/4] GPU training ($GPU_EPISODES episodes)..."
    if [ -f "$MODEL_DIR/model_best.pt" ]; then
        python3 -u train.py --episodes $GPU_EPISODES --resume "$MODEL_DIR/model_best.pt" 2>&1 | tail -5
    else
        python3 -u train.py --episodes $GPU_EPISODES 2>&1 | tail -5
    fi

    # Step 2: Export model for browser
    echo "[2/4] Exporting model..."
    python3 export_model.py "$MODEL_DIR/model_best.pt"
    echo "  → model_weights.json updated (browser will auto-reload)"

    # Step 3: Wait for gameplay recordings
    # (The browser with 0+W active continuously records DQN gameplay)
    echo "[3/4] Checking for gameplay recordings..."
    if [ -f "$GAMEPLAY_FILE" ]; then
        LINES=$(wc -l < "$GAMEPLAY_FILE")
        echo "  Found $LINES recorded transitions"

        if [ "$LINES" -gt 5000 ]; then
            # Step 4: Behavioral cloning from recordings
            echo "[4/4] Behavioral cloning ($BC_EPOCHS epochs)..."
            python3 train_bc.py \
                --model "$MODEL_DIR/model_best.pt" \
                --epochs $BC_EPOCHS \
                --lr 5e-5 \
                --export 2>&1 | tail -5

            # Archive old recordings
            mv "$GAMEPLAY_FILE" "$MODEL_DIR/gameplay_cycle${cycle}.jsonl"
            echo "  Archived recordings to gameplay_cycle${cycle}.jsonl"
        else
            echo "  Not enough data ($LINES < 5000). Skipping BC."
            echo "  Make sure browser has 0+W active!"
        fi
    else
        echo "  No gameplay data. Skipping BC."
        echo "  Start the browser with 0+W to record."
    fi

    echo ""
    echo "--- Cycle $cycle complete ---"
    echo "  Best model: $MODEL_DIR/model_best.pt"
    echo "  Browser JSON: $MODEL_DIR/model_weights.json"
done

echo ""
echo "========================================"
echo "  Training loop complete ($CYCLES cycles)"
echo "========================================"
