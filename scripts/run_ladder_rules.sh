#!/usr/bin/env bash
# Score every hand-written / privileged rung + the official EgoStatusMLP seeds on one split, sequentially.
# Usage: scripts/run_ladder_rules.sh [split=navtest] [workers=6]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/env.sh"
SPLIT="${1:-navtest}"; WORKERS="${2:-6}"
CK="$NAVSIM_EXP_ROOT/checkpoints/ego_status_mlp"   # official baselines, huggingface.co/autonomousvision/navsim_baselines
# "agent config | experiment name | extra overrides"
RUNGS=(
  "constant_velocity_agent|cv|"
  "kinematic_agent|kin|"
  "privileged_brake_agent|privbrake|"
  "privileged_centerline_agent|privmap_idm|"
  "privileged_centerline_kin|privmapkin|"
  "privileged_centerline_gtspeed|privmap_gtspd|"
  "privileged_gtpath_kin|privgtpath_kin|"
  "human_agent|human|"
  "ego_status_mlp_agent|egomlp_s0|agent.checkpoint_path=$CK/ego_status_mlp_seed_0.ckpt"
  "ego_status_mlp_agent|egomlp_s1|agent.checkpoint_path=$CK/ego_status_mlp_seed_1.ckpt"
  "ego_status_mlp_agent|egomlp_s2|agent.checkpoint_path=$CK/ego_status_mlp_seed_2.ckpt"
)
for r in "${RUNGS[@]}"; do
  IFS='|' read -r agent name extra <<< "$r"
  if ls "$NAVSIM_EXP_ROOT/${name}_${SPLIT}"/*/*.csv >/dev/null 2>&1; then
    echo "==== skip ${name} (csv exists) ===="; continue
  fi
  echo "==== $(date +%H:%M:%S) PDM ${agent} -> ${name}_${SPLIT} ${extra} ===="
  if [ -n "$extra" ]; then
    "$HERE/run_pdm.sh" "$agent" "$name" "$SPLIT" "$WORKERS" $extra 2>&1 | grep --line-buffered -E "Finished|successful|failed|Final average|Error|error" | head -8
  else
    "$HERE/run_pdm.sh" "$agent" "$name" "$SPLIT" "$WORKERS" 2>&1 | grep --line-buffered -E "Finished|successful|failed|Final average|Error|error" | head -8
  fi
done
echo "==== $(date +%H:%M:%S) ladder rules done ===="
