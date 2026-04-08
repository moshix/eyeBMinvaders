#!/usr/bin/env bash
# =============================================================================
# Auto-Improve Loop for eyeBMinvaders
# =============================================================================
# Autonomous training loop that:
#   1. Runs explorer training on remote GPU
#   2. Analyzes results with claude -p
#   3. Implements code improvements (state features, rewards, architecture)
#   4. Recompiles Rust sim if changed
#   5. Commits, pushes, pulls on remote
#   6. Loops back to training
#
# Usage:
#   ./scripts/auto_improve.sh                    # Default settings
#   ./scripts/auto_improve.sh --episodes 100000  # Override episodes per cycle
#   ./scripts/auto_improve.sh --max-iterations 5 # Limit iterations
#   ./scripts/auto_improve.sh --dry-run          # Analyze only, no training
#
# Requirements:
#   - claude CLI (local)
#   - SSH access to training server
#   - Rust toolchain on remote (for recompilation)
# =============================================================================

set -euo pipefail

# --- Configuration ---
REMOTE_HOST="dennis@192.168.0.223"
REMOTE_DIR="/home/dennis/eyeBMinvaders"
REMOTE_VENV="/home/dennis/.venv/bin/python3"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${LOCAL_DIR}/models/auto_improve"
STATE_FILE="${STATE_DIR}/iteration_state.json"
LOG_DIR="${STATE_DIR}/logs"

EPISODES_PER_CYCLE=200000
EXPERIMENTS_PER_PLATEAU=6
FAIL_FAST_BUDGET=7500
MAX_ITERATIONS=20
TARGET_LEVEL=15
DRY_RUN=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes) EPISODES_PER_CYCLE="$2"; shift 2;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2;;
        --target-level) TARGET_LEVEL="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# --- Setup ---
mkdir -p "${STATE_DIR}" "${LOG_DIR}"

# Initialize state if not exists
if [[ ! -f "${STATE_FILE}" ]]; then
    cat > "${STATE_FILE}" << 'INITJSON'
{
  "iteration": 0,
  "target_level": 15,
  "best_level_reached": 0,
  "best_avg_score": 0,
  "best_model_path": null,
  "history": [],
  "changes_made": [],
  "failed_ideas": [],
  "successful_ideas": []
}
INITJSON
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/auto_improve.log"
}

# --- Remote helpers ---
remote_exec() {
    ssh -o ConnectTimeout=10 -o ServerAliveInterval=30 "${REMOTE_HOST}" "$@"
}

remote_training_running() {
    # Use pgrep -x with python3 and check /proc for the actual explorer script
    remote_exec "pgrep -f 'python3.*explorer_train\.py' > /dev/null 2>&1" && return 0 || return 1
}

wait_for_training() {
    log "Waiting for training to complete on ${REMOTE_HOST}..."
    local elapsed=0
    local check_interval=60

    while remote_training_running; do
        sleep ${check_interval}
        elapsed=$((elapsed + check_interval))

        # Progress check every 5 minutes
        if (( elapsed % 300 == 0 )); then
            local last_line
            last_line=$(remote_exec "tail -1 ${REMOTE_DIR}/training_explorer_output.log 2>/dev/null" || echo "?")
            log "  Still training (${elapsed}s elapsed): ${last_line}"
        fi
    done

    log "Training finished after ${elapsed}s"
}

# --- Main Loop ---
log "=========================================="
log "AUTO-IMPROVE LOOP STARTING"
log "=========================================="
log "Target: level ${TARGET_LEVEL}"
log "Episodes per cycle: ${EPISODES_PER_CYCLE}"
log "Max iterations: ${MAX_ITERATIONS}"
log "Remote: ${REMOTE_HOST}:${REMOTE_DIR}"
log "Local: ${LOCAL_DIR}"
log "=========================================="

ITERATION=$(jq -r '.iteration' "${STATE_FILE}")

while (( ITERATION < MAX_ITERATIONS )); do
    ITERATION=$((ITERATION + 1))
    CYCLE_LOG="${LOG_DIR}/iteration_${ITERATION}.log"
    CYCLE_ANALYSIS="${LOG_DIR}/iteration_${ITERATION}_analysis.md"
    CYCLE_CHANGES="${LOG_DIR}/iteration_${ITERATION}_changes.md"

    log ""
    log "############################################"
    log "# ITERATION ${ITERATION} / ${MAX_ITERATIONS}"
    log "############################################"

    # Update iteration in state
    jq ".iteration = ${ITERATION}" "${STATE_FILE}" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "${STATE_FILE}"

    # =================================================================
    # PHASE 1: TRAIN (on remote GPU)
    # =================================================================
    if [[ "${DRY_RUN}" == "false" ]]; then
        log "Phase 1: Starting training on remote..."

        # Ensure remote is up to date
        remote_exec "cd ${REMOTE_DIR} && git pull" 2>&1 | tee -a "${CYCLE_LOG}"

        # Rebuild Rust sim if source changed
        log "  Rebuilding Rust sim..."
        remote_exec "cd ${REMOTE_DIR}/game_sim && maturin develop --release 2>&1 | tail -5" 2>&1 | tee -a "${CYCLE_LOG}" || {
            log "  WARNING: Rust build failed, continuing with existing sim"
        }

        # Reset explorer state so training starts fresh (keeps best model but resets episode count)
        log "  Resetting explorer episode count for fresh cycle..."
        remote_exec "cd ${REMOTE_DIR} && ${REMOTE_VENV} -c \"
import json, os
state_path = 'models/explorer/explorer_state.json'
if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)
    state['total_episodes'] = 0
    state['generation'] = 0
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    print(f'Reset explorer state (kept baseline_score={state.get(\\\"baseline_score\\\", 0)})')
else:
    print('No explorer state to reset')
\"" 2>&1 | tee -a "${CYCLE_LOG}"

        # Start training
        remote_exec "cd ${REMOTE_DIR} && nohup ${REMOTE_VENV} -u explorer_train.py \
            --episodes ${EPISODES_PER_CYCLE} \
            --experiments-per-plateau ${EXPERIMENTS_PER_PLATEAU} \
            --fail-fast-budget ${FAIL_FAST_BUDGET} \
            --device cuda \
            --resume \
            > training_explorer_output.log 2>&1 &"

        sleep 15  # Let it start

        if remote_training_running; then
            log "  Training started successfully"
            wait_for_training
        else
            # Check if it already finished (very fast completion)
            if grep -q "EXPLORER TRAINING COMPLETE" <(remote_exec "tail -5 ${REMOTE_DIR}/training_explorer_output.log" 2>/dev/null); then
                log "  Training completed immediately"
            else
                log "  ERROR: Training failed to start"
                remote_exec "tail -30 ${REMOTE_DIR}/training_explorer_output.log" 2>&1 | tee -a "${CYCLE_LOG}"
            fi
        fi

        # Fetch training results
        log "  Fetching training logs..."
        remote_exec "tail -100 ${REMOTE_DIR}/training_explorer_output.log" > "${CYCLE_LOG}" 2>&1
        remote_exec "cat ${REMOTE_DIR}/models/explorer/explorer_state.json" > "${STATE_DIR}/latest_explorer_state.json" 2>&1 || true
    else
        log "Phase 1: [DRY RUN] Skipping training"
    fi

    # =================================================================
    # PHASE 2: ANALYZE + IMPLEMENT (with claude -p)
    # =================================================================
    log "Phase 2: Analyzing results and implementing improvements..."

    # Build the context for claude -p
    TRAINING_OUTPUT=""
    if [[ -f "${CYCLE_LOG}" ]]; then
        TRAINING_OUTPUT=$(tail -100 "${CYCLE_LOG}" 2>/dev/null || echo "No training log available")
    fi

    EXPLORER_STATE=""
    if [[ -f "${STATE_DIR}/latest_explorer_state.json" ]]; then
        EXPLORER_STATE=$(cat "${STATE_DIR}/latest_explorer_state.json" 2>/dev/null || echo "{}")
    fi

    ITERATION_STATE=$(cat "${STATE_FILE}")

    # The big prompt for claude -p
    IMPROVE_PROMPT=$(cat << PROMPTEOF
You are an autonomous AI training improvement system for the eyeBMinvaders Space Invaders game.
Your goal: get the AI agent to consistently reach level ${TARGET_LEVEL}.

## Current State (Iteration ${ITERATION}/${MAX_ITERATIONS})

### Iteration History
\`\`\`json
${ITERATION_STATE}
\`\`\`

### Latest Training Output
\`\`\`
${TRAINING_OUTPUT}
\`\`\`

### Explorer State
\`\`\`json
${EXPLORER_STATE}
\`\`\`

## Your Task

You are working in ${LOCAL_DIR}. Analyze the training results and implement ONE focused improvement to push the agent toward level ${TARGET_LEVEL}.

### Rules
1. **ONE change per iteration** — do not try to fix everything at once. Pick the single highest-impact change.
2. **Fail fast philosophy** — if a previous change didn't help (check history), try a completely different angle.
3. **Changes you can make:**
   - Modify reward function in \`game_sim/src/state.rs\` (the \`calculate_reward\` function)
   - Add state features in \`game_sim/src/state.rs\` (the \`get_state\` function) and update STATE_SIZE in \`game_sim/src/constants.rs\`
   - Modify training hyperparameters/architecture in \`explorer_train.py\`
   - Modify curriculum strategy in \`explorer_train.py\`
   - Modify \`train.py\` or \`train_ppo.py\` training loops
   - Add new experiment types to the explorer's strategy generator
4. **After making changes:**
   - If you modified Rust code (\`game_sim/\`), verify it compiles: \`cd game_sim && cargo check 2>&1\`
   - Run a quick sanity test if possible
   - Commit your changes with a descriptive message
   - Push to origin

### Key Files
- \`game_sim/src/state.rs\` — reward function (\`calculate_reward\`) and state features (\`get_state\`)
- \`game_sim/src/constants.rs\` — STATE_SIZE, game constants
- \`game_sim/src/game.rs\` — game logic, difficulty scaling
- \`explorer_train.py\` — explorer training system (strategy generator, experiment runner)
- \`train.py\` — DQN training loop
- \`train_ppo.py\` — PPO training loop

### Analysis Framework
1. **What level did we reach?** Compare to target (${TARGET_LEVEL}) and previous iterations.
2. **What's the bottleneck?** Is it perception (state), motivation (rewards), capacity (architecture), or strategy (curriculum)?
3. **What was tried before?** Check \`changes_made\` and \`failed_ideas\` in the iteration state. Do NOT repeat failed approaches.
4. **What's the single highest-impact change?** Implement it.
5. **Update the iteration state** by writing to ${STATE_FILE}

### Output Format
After implementing your change, write a summary to ${CYCLE_CHANGES} with:
- What you changed and why
- What you expect the impact to be
- How to verify it worked in the next iteration

Then update ${STATE_FILE} — add your change to \`changes_made\`, update \`best_level_reached\` and \`best_avg_score\` from the training results.

IMPORTANT: Make exactly ONE focused change, commit it, and push. Do not over-engineer.
PROMPTEOF
)

    # Run claude -p with the improvement prompt
    log "  Running claude -p for analysis and implementation..."
    echo "${IMPROVE_PROMPT}" | claude -p \
        --permission-mode auto \
        --max-turns 40 \
        2>&1 | tee "${CYCLE_ANALYSIS}"

    # Check if changes were committed
    LATEST_COMMIT=$(git -C "${LOCAL_DIR}" log --oneline -1)
    log "  Latest commit: ${LATEST_COMMIT}"

    # =================================================================
    # PHASE 3: PUSH TO REMOTE
    # =================================================================
    log "Phase 3: Syncing changes to remote..."

    # Push if there are unpushed commits
    if git -C "${LOCAL_DIR}" status | grep -q "ahead"; then
        git -C "${LOCAL_DIR}" push 2>&1 | tee -a "${CYCLE_LOG}"
        log "  Pushed to origin"
    fi

    # Pull on remote
    remote_exec "cd ${REMOTE_DIR} && git pull" 2>&1 | tee -a "${CYCLE_LOG}"

    # =================================================================
    # PHASE 4: CHECK TARGET
    # =================================================================
    BEST_LEVEL=$(jq -r '.best_level_reached // 0' "${STATE_FILE}")
    log "Best level so far: ${BEST_LEVEL} / target: ${TARGET_LEVEL}"

    if (( BEST_LEVEL >= TARGET_LEVEL )); then
        log ""
        log "=========================================="
        log "TARGET REACHED! Level ${BEST_LEVEL} >= ${TARGET_LEVEL}"
        log "=========================================="
        break
    fi

    log "Iteration ${ITERATION} complete. Looping..."
    log ""
done

# --- Final Summary ---
log ""
log "=========================================="
log "AUTO-IMPROVE LOOP FINISHED"
log "=========================================="
log "Iterations completed: ${ITERATION}"
log "Best level reached: $(jq -r '.best_level_reached // 0' "${STATE_FILE}")"
log "Best avg score: $(jq -r '.best_avg_score // 0' "${STATE_FILE}")"
log "State: ${STATE_FILE}"
log "Logs: ${LOG_DIR}"
log "=========================================="
