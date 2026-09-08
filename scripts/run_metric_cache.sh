#!/usr/bin/env bash
# Usage: scripts/run_metric_cache.sh navtest [workers]
set -euo pipefail
source "$(dirname "$0")/env.sh"
SPLIT="${1:-navtest}"; WORKERS="${2:-6}"
CACHE="$NAVSIM_EXP_ROOT/metric_cache_${SPLIT}"
mkdir -p "$CACHE"
cd "$NAVSIM_DEVKIT_ROOT"
"$NAVSIM_PYTHON" navsim/planning/script/run_metric_caching.py \
  train_test_split="$SPLIT" \
  worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers="$WORKERS" \
  cache.cache_path="$CACHE" \
  2>&1 | tee -a "$NAVSIM_EXP_ROOT/metric_cache_${SPLIT}.log"
