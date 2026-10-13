#!/usr/bin/env bash
# UAD local inference: no cloud LLM/API embedding; public weights download allowed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/DATA}"
UAD_ROOT="${UAD_ROOT:-$SCRIPT_DIR/unsup-affordance}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results_affordance_UAD}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/prompts_affordance.tsv}"
AFFORDANCE_COLOR_FILE="${AFFORDANCE_COLOR_FILE:-$SCRIPT_DIR/affordance_colors.tsv}"
UAD_CONFIG="${UAD_CONFIG:-configs/st_emb.yaml}"
UAD_CHECKPOINT="${UAD_CHECKPOINT:-checkpoints/st_emb.pth}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_SIZE="${MAX_SIZE:-336}"

# Allow downloads of public model weights; refuse cloud-LLM credentials.
export WANDB_MODE=disabled WANDB_DISABLED=true
unset OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY || true

args=(
  "$SCRIPT_DIR/test_IoU_affordance_UAD.py"
  --data_root "$DATA_ROOT"
  --uad_root "$UAD_ROOT"
  --output_root "$RESULT_ROOT"
  --prompt_file "$PROMPT_FILE"
  --affordance_colors "$AFFORDANCE_COLOR_FILE"
  --config "$UAD_CONFIG"
  --checkpoint "$UAD_CHECKPOINT"
  --threshold "$THRESHOLD"
  --max_size "$MAX_SIZE"
  "$@"
)
exec python "${args[@]}"
