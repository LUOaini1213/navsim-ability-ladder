# NAVSIM workspace environment. Set NAVSIM_WORKSPACE first, then dot-source:
#   $env:NAVSIM_WORKSPACE = "E:\navsim_workspace"; . .\scripts\env.ps1
if (-not $env:NAVSIM_WORKSPACE) { $env:NAVSIM_WORKSPACE = "$env:USERPROFILE\navsim_workspace" }
$env:NAVSIM_DEVKIT_ROOT   = "$env:NAVSIM_WORKSPACE\navsim"
$env:OPENSCENE_DATA_ROOT  = "$env:NAVSIM_WORKSPACE\dataset"
$env:NUPLAN_MAPS_ROOT     = "$env:NAVSIM_WORKSPACE\dataset\maps"
$env:NUPLAN_MAP_VERSION   = "nuplan-maps-v1.0"
$env:NAVSIM_EXP_ROOT      = "$env:NAVSIM_WORKSPACE\exp"
$env:NAVSIM_PYTHON        = "$env:NAVSIM_WORKSPACE\.venv\Scripts\python.exe"
$env:PYTHONUTF8           = "1"     # see docs/WINDOWS.md
$env:PYTHONIOENCODING     = "utf-8"
$env:HYDRA_FULL_ERROR     = "1"
