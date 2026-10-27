"""Use canonical repository agents without modifying the user's devkit checkout."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(workspace=None):
    workspace = Path(workspace or os.environ.get("NAVSIM_WORKSPACE", "E:/navsim_workspace"))
    defaults = {
        "NAVSIM_WORKSPACE": workspace,
        "NAVSIM_DEVKIT_ROOT": workspace / "navsim",
        "OPENSCENE_DATA_ROOT": workspace / "dataset",
        "NUPLAN_MAPS_ROOT": workspace / "dataset/maps",
        "NAVSIM_EXP_ROOT": workspace / "exp",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    sys.path.insert(0, os.environ["NAVSIM_DEVKIT_ROOT"])
    import navsim.agents

    source = str(ROOT / "agents")
    if source not in navsim.agents.__path__:
        navsim.agents.__path__.insert(0, source)
    return workspace
