#!/usr/bin/env bash
# Usage: scripts/run_pdm.sh <agent_config_name> <experiment_name> [split=navtest] [workers=6] [extra hydra overrides...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
AGENT="$1"; NAME="$2"; SPLIT="${3:-navtest}"; WORKERS="${4:-6}"
shift $(( $# < 4 ? $# : 4 ))
CACHE="$NAVSIM_EXP_ROOT/metric_cache_${SPLIT}"
cd "$NAVSIM_DEVKIT_ROOT"
"$NAVSIM_PYTHON" navsim/planning/script/run_pdm_score.py \
  train_test_split="$SPLIT" \
  agent="$AGENT" experiment_name="${NAME}_${SPLIT}" \
  worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers="$WORKERS" \
  metric_cache_path="$CACHE" "$@" \
  2>&1 | tee -a "$NAVSIM_EXP_ROOT/pdm_${NAME}_${SPLIT}.log"
