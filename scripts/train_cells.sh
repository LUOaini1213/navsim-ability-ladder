#!/usr/bin/env bash
# Train every learned cell of the ladder tables from frozen navtrain features.
# Usage: scripts/train_cells.sh [seeds="0 1 2"] [epochs=40]
#   pose_*  : learned trajectory (8 poses); 2x2 factorial over {map} x {agents}
#   ds_*    : rule centerline path + learned speed (8 arc-length offsets)
#   reg_*   : regularisation sweep on pose_map (open-loop L1 vs PDMS), seed 0 only
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/env.sh"
SEEDS="${1:-0 1 2}"; EPOCHS="${2:-40}"
F="$NAVSIM_EXP_ROOT/features/features_navtrain.npz"
[ -f "$F" ] || { echo "missing $F"; exit 1; }
run() { # tag, args...
  local tag="$1"; shift
  if [ -f "$NAVSIM_EXP_ROOT/ladder/$tag/model.pt" ]; then echo "skip $tag"; return; fi
  echo "==== $(date +%T) train $tag ===="
  "$NAVSIM_PYTHON" "$HERE/../analysis/train_ladder.py" --train "$F" --epochs "$EPOCHS" --tag "$tag" "$@" 2>&1 | grep --line-buffered -E "^\[|epoch +(1|10|20|30|40) " 
}
for s in $SEEDS; do
  run "pose_kin_s$s"        --target pose --seed "$s"
  run "pose_map_s$s"        --target pose --seed "$s" --see-map
  run "pose_agents_s$s"     --target pose --seed "$s" --see-agents
  run "pose_map_agents_s$s" --target pose --seed "$s" --see-map --see-agents
  run "ds_kin_s$s"          --target ds   --seed "$s"
  run "ds_map_s$s"          --target ds   --seed "$s" --see-map
  run "ds_map_agents_s$s"   --target ds   --seed "$s" --see-map --see-agents
done
# regularisation sweep (seed 0, map-conditioned learned trajectory): weight decay x dropout
for wd in 1e-4 1e-3 1e-2; do run "reg_pose_map_wd${wd}_s0" --target pose --seed 0 --see-map --wd "$wd"; done
run "reg_pose_map_do0.2_s0"        --target pose --seed 0 --see-map --dropout 0.2
run "reg_pose_map_do0.2_wd1e-3_s0" --target pose --seed 0 --see-map --dropout 0.2 --wd 1e-3
run "reg_pose_map_h128_s0"         --target pose --seed 0 --see-map --hidden 128
echo "==== $(date +%T) all cells trained ===="
