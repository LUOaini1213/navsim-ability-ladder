#!/usr/bin/env bash
# Score every trained cell under exp/ladder on navtest.
#   pose_* runs are scored twice: speed_mode=learned (learned path + learned speed)
#                                 speed_mode=rule    (learned path + rule speed)
#   ds_*   runs once (rule path + learned speed).  Experiments are named l_<tag>[_rs].
# Usage: scripts/run_ladder_learned.sh [split=navtest] [workers=6] [pattern="*"]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/env.sh"
SPLIT="${1:-navtest}"; WORKERS="${2:-6}"; PAT="${3:-*}"
for d in "$NAVSIM_EXP_ROOT"/ladder/$PAT/; do
  tag="$(basename "$d")"; [ -f "$d/model.pt" ] || continue
  case "$tag" in _trial*) continue;; esac
  modes="learned"; case "$tag" in pose_*|reg_*) modes="learned rule";; esac
  for m in $modes; do
    name="l_${tag}"; [ "$m" = "rule" ] && name="${name}_rs"
    if ls "$NAVSIM_EXP_ROOT/${name}_${SPLIT}"/*/*.csv >/dev/null 2>&1; then echo "skip $name"; continue; fi
    echo "==== $(date +%T) PDM ladder_mlp_agent[$tag, speed=$m] -> ${name}_${SPLIT} ===="
    "$HERE/run_pdm.sh" ladder_mlp_agent "$name" "$SPLIT" "$WORKERS" "agent.run_dir=$NAVSIM_EXP_ROOT/ladder/$tag" "agent.speed_mode=$m" 2>&1 | grep --line-buffered -E "Finished|successful|failed|Final average|Error|error" | head -8
  done
done
echo "==== $(date +%T) learned ladder done ===="
