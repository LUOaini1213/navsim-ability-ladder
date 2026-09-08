# NAVSIM workspace environment. Point NAVSIM_WORKSPACE at your checkout, then
# `source scripts/env.sh` before any script in this directory.
#
#   <workspace>/navsim        NAVSIM v1.1 devkit (agents/ and configs/ copied in)
#   <workspace>/dataset       maps/ + navsim_logs/{test,trainval}
#   <workspace>/exp           metric caches, per-run CSVs, trained cells
#   <workspace>/.venv         Python 3.9 environment

export NAVSIM_WORKSPACE="${NAVSIM_WORKSPACE:-$HOME/navsim_workspace}"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_WORKSPACE/navsim"
export OPENSCENE_DATA_ROOT="$NAVSIM_WORKSPACE/dataset"
export NUPLAN_MAPS_ROOT="$NAVSIM_WORKSPACE/dataset/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="$NAVSIM_WORKSPACE/exp"
# Windows: .venv/Scripts/python.exe   POSIX: .venv/bin/python
if [ -x "$NAVSIM_WORKSPACE/.venv/Scripts/python.exe" ]; then
  export NAVSIM_PYTHON="$NAVSIM_WORKSPACE/.venv/Scripts/python.exe"
else
  export NAVSIM_PYTHON="${NAVSIM_PYTHON:-$NAVSIM_WORKSPACE/.venv/bin/python}"
fi
export PYTHONUTF8=1           # see docs/WINDOWS.md
export PYTHONIOENCODING=utf-8
export HYDRA_FULL_ERROR=1
