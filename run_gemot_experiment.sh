#!/bin/bash
# run_gemot_experiment.sh — Orchestrates a Diplomacy game with gemot briefings.
#
# Flow per year (or per season with --per-season):
#   1. Run lm_game.py for one year/season
#   2. Run gemot diplomacy analysis on messages so far
#   3. Inject briefings into per-power system prompts
#   4. Resume the game for the next year/season
#
# Usage:
#   ./run_gemot_experiment.sh --name gemot_v13 --max-year 1910
#   ./run_gemot_experiment.sh --name control_v13 --max-year 1910 --no-gemot
#   ./run_gemot_experiment.sh --name gemot_v13_seasonal --max-year 1910 --per-season
#
# Prerequisites:
#   - gemot server running: cd ~/Documents/gemot && ./gemot http --addr :8080
#   - AI_Diplomacy venv with dependencies installed
#   - .env files in both gemot/ and AI_Diplomacy/ with API keys
#
# Known issues:
#   - Alliance scopes with 2 members must use "negotiation" template, not "consensus"
#     (consensus requires min 3 participants). The diplomacy script handles this
#     automatically, but if you see "quorum not met" errors, check the template.
#   - The deliberation state file tracks deliberation IDs across years. If a scope's
#     template needs to change (e.g., alliance grows from 2 to 3 members), the script
#     re-sets the template on reuse. If you still hit issues, delete the stale entry
#     from the state JSON and rerun that year.
#   - Long-running analysis (year 5+) can cause SSE connection drops. The script
#     reconnects automatically (up to 10 times). If analysis still fails, consider
#     running that year's analysis separately.
#
set -euo pipefail

# --- Defaults ---
EXPERIMENT_NAME=""
MODEL="claude-sonnet-4-6"
MAX_YEAR=1910
START_YEAR=1901
NUM_NEGOTIATION_ROUNDS=2
GEMOT_ENABLED=true
PER_SEASON=false
GEMOT_DIR="$HOME/Documents/gemot"
AI_DIPLOMACY_DIR="$HOME/Documents/AI_Diplomacy"
GEMOT_URL="http://localhost:8080/mcp"
RESULTS_DIR=""
PROMPTS_TEMPLATE_DIR=""

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-year) MAX_YEAR="$2"; shift 2 ;;
    --start-year) START_YEAR="$2"; shift 2 ;;
    --rounds) NUM_NEGOTIATION_ROUNDS="$2"; shift 2 ;;
    --no-gemot) GEMOT_ENABLED=false; shift ;;
    --per-season) PER_SEASON=true; shift ;;
    --gemot-url) GEMOT_URL="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --prompts) PROMPTS_TEMPLATE_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "Usage: $0 --name <experiment_name> [--model <model>] [--max-year <year>] [--no-gemot]"
  exit 1
fi

# --- Setup ---
RESULTS_DIR="${RESULTS_DIR:-$AI_DIPLOMACY_DIR/results/$EXPERIMENT_NAME}"
PROMPTS_TEMPLATE_DIR="${PROMPTS_TEMPLATE_DIR:-$AI_DIPLOMACY_DIR/prompts_per_power_t3c_v2}"
RUN_DIR="$RESULTS_DIR/game"
STATE_FILE="$RESULTS_DIR/deliberation_state.json"

mkdir -p "$RESULTS_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $MODEL"
echo "Max year: $MAX_YEAR"
echo "Gemot: $GEMOT_ENABLED"
echo "Results: $RESULTS_DIR"
echo "---"

# Load env vars
if [[ -f "$AI_DIPLOMACY_DIR/.env" ]]; then
  export $(grep -v '^#' "$AI_DIPLOMACY_DIR/.env" | xargs)
fi
if [[ -f "$GEMOT_DIR/.env" ]]; then
  export $(grep -v '^#' "$GEMOT_DIR/.env" | xargs)
fi
export GEMOT_LIVE_URL="$GEMOT_URL"

# Activate AI_Diplomacy venv
if [[ -f "$AI_DIPLOMACY_DIR/.venv/bin/activate" ]]; then
  source "$AI_DIPLOMACY_DIR/.venv/bin/activate"
fi

# Build models string (same model for all 7 powers)
MODELS_STR="$MODEL,$MODEL,$MODEL,$MODEL,$MODEL,$MODEL,$MODEL"

# --- Build phase list ---
# Per-season: analyze after each movement phase (S and F)
# Per-year: analyze after each full year (after W adjustment)
build_phases() {
  for YEAR_INT in $(seq "$START_YEAR" "$MAX_YEAR"); do
    if [[ "$PER_SEASON" == "true" ]]; then
      echo "S${YEAR_INT}M"
      echo "F${YEAR_INT}M"
    else
      echo "Y${YEAR_INT}"  # sentinel for year-based mode
    fi
  done
}

# --- Helper: run one analysis + injection cycle ---
run_analysis_cycle() {
  local LABEL="$1"       # e.g., "year1_spring" or "year1"
  local YEAR_NUM="$2"    # game year number (1-based)
  local PHASE_NAME="$3"  # e.g., "S1901M" for per-season, empty for per-year

  if [[ "$GEMOT_ENABLED" == "false" ]]; then
    echo "[control] Skipping gemot analysis for $LABEL"
    return
  fi

  local CYCLE_OUTPUT="$RESULTS_DIR/${LABEL}/briefings"
  local CYCLE_PROMPTS="$RESULTS_DIR/${LABEL}/prompts"
  mkdir -p "$CYCLE_OUTPUT" "$CYCLE_PROMPTS"

  echo "[step 2] Running gemot v13 analysis for $LABEL..."
  cd "$GEMOT_DIR"

  ANALYSIS_ARGS=(
    --game "$RUN_DIR/lmvsgame.json"
    --year "$YEAR_NUM"
    --output "$CYCLE_OUTPUT"
    --state "$STATE_FILE"
    --experiment "$EXPERIMENT_NAME"
  )

  # Per-season: only collect messages from the target phase
  if [[ -n "$PHASE_NAME" && "$PHASE_NAME" != Y* ]]; then
    ANALYSIS_ARGS+=(--through-phase "$PHASE_NAME")
  fi

  go run ./scripts/diplomacy/ "${ANALYSIS_ARGS[@]}" \
    2>&1 | tee "$RESULTS_DIR/${LABEL}_analysis.log"

  # Inject briefings into per-power system prompts
  echo "[step 3] Injecting briefings into prompts..."
  cp "$PROMPTS_TEMPLATE_DIR"/*.txt "$CYCLE_PROMPTS/" 2>/dev/null || true
  if [[ -d "$PROMPTS_TEMPLATE_DIR/formatting" ]]; then
    cp -r "$PROMPTS_TEMPLATE_DIR/formatting" "$CYCLE_PROMPTS/"
  fi
  if [[ -d "$PROMPTS_TEMPLATE_DIR/flavor_system_prompts" ]]; then
    cp -r "$PROMPTS_TEMPLATE_DIR/flavor_system_prompts" "$CYCLE_PROMPTS/"
  fi

  # For each power, prepend the base system prompt + append the briefing
  for POWER in austria england france germany italy russia turkey; do
    POWER_UPPER=$(echo "$POWER" | tr '[:lower:]' '[:upper:]')
    BRIEFING_FILE="$CYCLE_OUTPUT/${POWER}_briefing.txt"
    PROMPT_FILE="$CYCLE_PROMPTS/${POWER}_system_prompt.txt"
    BASE_PROMPT="$PROMPTS_TEMPLATE_DIR/${POWER}_system_prompt.txt"

    if [[ -f "$BRIEFING_FILE" ]]; then
      # Strip any existing briefing from the base prompt
      if grep -q "=== YOUR PRIVATE DIPLOMATIC INTELLIGENCE BRIEFING" "$BASE_PROMPT" 2>/dev/null; then
        sed '/=== YOUR PRIVATE DIPLOMATIC INTELLIGENCE BRIEFING/,$d' "$BASE_PROMPT" > "$PROMPT_FILE"
      else
        cp "$BASE_PROMPT" "$PROMPT_FILE"
      fi

      cat >> "$PROMPT_FILE" << BRIEFING_EOF

=== YOUR PRIVATE DIPLOMATIC INTELLIGENCE BRIEFING ($LABEL) ===
The following analysis is based on YOUR diplomatic communications only.
Other powers have their own intelligence based on their own communications.

IMPORTANT: When making strategic decisions, explicitly note when you are using
insights from this briefing. For example:
- "Based on my intelligence showing strong alignment with Russia, I will..."
- "The briefing identifies Germany as a swing power, so I should..."
- "Given the high controversy score on Mediterranean strategies, I will..."

This helps track how diplomatic intelligence influences your strategy.

$(cat "$BRIEFING_FILE")

=== END BRIEFING ===
BRIEFING_EOF

      echo "  $POWER_UPPER: briefing injected ($(wc -l < "$BRIEFING_FILE") lines)"
    else
      echo "  $POWER_UPPER: no briefing available, using base prompt"
    fi
  done
}

# --- Get latest prompts dir (most recent analysis cycle's prompts) ---
latest_prompts() {
  # Find most recent prompts directory
  local latest=""
  for d in "$RESULTS_DIR"/*/prompts; do
    if [[ -d "$d" ]]; then
      latest="$d"
    fi
  done
  echo "${latest:-$PROMPTS_TEMPLATE_DIR}"
}

# --- Main game loop ---
STEP_NUM=0
for YEAR_INT in $(seq "$START_YEAR" "$MAX_YEAR"); do
  YEAR_NUM=$((YEAR_INT - 1900))

  if [[ "$PER_SEASON" == "true" ]]; then
    SEASONS=("spring" "fall")
    PHASES=("S${YEAR_INT}M" "F${YEAR_INT}M")
  else
    SEASONS=("full")
    PHASES=("Y${YEAR_INT}")
  fi

  for IDX in "${!SEASONS[@]}"; do
    SEASON="${SEASONS[$IDX]}"
    PHASE="${PHASES[$IDX]}"

    if [[ "$PER_SEASON" == "true" ]]; then
      LABEL="year${YEAR_NUM}_${SEASON}"
    else
      LABEL="year${YEAR_NUM}"
    fi

    STEP_NUM=$((STEP_NUM + 1))
    echo ""
    echo "============================================"
    echo "  Step $STEP_NUM: $LABEL ($PHASE)"
    echo "  $(date)"
    echo "============================================"

    # Step 1: Run the game
    echo "[step 1] Running game through $PHASE..."
    GAME_ARGS=(
      --run_dir "$RUN_DIR"
      --models "$MODELS_STR"
      --max_year "$MAX_YEAR"
      --num_negotiation_rounds "$NUM_NEGOTIATION_ROUNDS"
      --simple_prompts false
    )

    # Determine how to pause
    if [[ "$PER_SEASON" == "true" ]]; then
      GAME_ARGS+=(--end_at_phase "$PHASE")
    else
      GAME_ARGS+=(--pause_after_year "$YEAR_INT")
    fi

    # Use latest available prompts
    CURRENT_PROMPTS=$(latest_prompts)
    GAME_ARGS+=(--prompts_dir "$CURRENT_PROMPTS")

    cd "$AI_DIPLOMACY_DIR"
    python lm_game.py "${GAME_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/${LABEL}_game.log"

    # Check if game file exists
    if [[ ! -f "$RUN_DIR/lmvsgame.json" ]]; then
      echo "No game file found — stopping."
      break 2
    fi

    # Check if game ended (victory)
    GAME_DONE=$(python3 -c "
import json
g = json.load(open('$RUN_DIR/lmvsgame.json'))
done = any(len(p.get('state',{}).get('centers',{}).get(pw,[])) >= 18
           for p in g['phases'] for pw in g['phases'][-1].get('state',{}).get('centers',{}))
print('true' if done else 'false')
" 2>/dev/null || echo "false")
    if [[ "$GAME_DONE" == "true" ]]; then
      echo "Victory detected — game over!"
      break 2
    fi

    # Step 2+3: Run analysis + inject briefings
    run_analysis_cycle "$LABEL" "$YEAR_NUM" "$PHASE"

    echo "[$LABEL complete]"
  done
done

echo ""
echo "============================================"
echo "  EXPERIMENT COMPLETE: $EXPERIMENT_NAME"
echo "  $(date)"
echo "============================================"

# Print final game state
if [[ -f "$RUN_DIR/lmvsgame.json" ]]; then
  python3 -c "
import json
g = json.load(open('$RUN_DIR/lmvsgame.json'))
for p in reversed(g['phases']):
    if 'state' in p and 'centers' in p['state'] and p['state']['centers']:
        centers = p['state']['centers']
        print('Final SC counts:')
        for power in sorted(centers.keys()):
            print(f'  {power}: {len(centers[power])} SCs')
        alive = sum(1 for v in centers.values() if len(v) > 0)
        print(f'Survival: {alive}/7')
        break
" 2>/dev/null || true
fi
