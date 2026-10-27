"""Verify new real-input model adapters against cached-feature predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mechanism_data import load_features, transform
from runtime import ROOT, configure


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--scenes", type=int, default=16)
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2/mechanism_live.json")
    args = ap.parse_args()
    workspace = configure()
    from build_features import scene_filter_for
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.agents.mechanism_agent import MechanismAgent
    torch.set_num_threads(2)
    data = load_features(workspace / "exp/features/features_navtest.npz")
    order = np.argsort(data["token"])
    chosen = order[np.unique(np.linspace(0, len(order)-1, args.scenes).astype(int))]
    loaded = []
    for run in args.runs:
        agent = MechanismAgent(str(run))
        agent.initialize()
        norm = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
        context, objects = transform(data, chosen, norm)
        with torch.no_grad():
            cached = agent.model(torch.from_numpy(context), torch.from_numpy(objects)).numpy().reshape(-1, 8, 3)
        loaded.append((agent, cached))
    ledger = []
    for log in sorted(set(data["log_name"][chosen])):
        positions = [j for j, i in enumerate(chosen) if data["log_name"][i] == log]
        tokens = [str(data["token"][chosen[j]]) for j in positions]
        sf = scene_filter_for("navtest", [str(log)])
        sf.tokens = tokens
        loader = SceneLoader(workspace / "dataset/navsim_logs/test", None, sf, SensorConfig.build_no_sensors())
        if set(loader.tokens) != set(tokens):
            raise ValueError("Missing scenes")
        for position, token in zip(positions, tokens):
            scene = loader.get_scene_from_token(token)
            history = SimpleNamespace(scene_metadata=scene.scene_metadata, map_api=scene.map_api,
                                      frames=scene.frames[:scene.scene_metadata.num_history_frames])
            agent_input = loader.get_agent_input_from_token(token)
            for agent, cached in loaded:
                actual = agent.compute_trajectory(agent_input, history).poses
                error = float(np.max(np.abs(actual-cached[position])))
                ledger.append({"run": agent.run_dir.name, "token": token, "max_error": error,
                               "passed": error < 1e-4, "diagnostic": agent.last_diagnostic})
        print("new live parity", log, len(tokens), "scenes x", len(loaded), "models", flush=True)
    report = {"passed": len(ledger) > 0 and all(row["passed"] for row in ledger),
              "scope": "Deterministic navtest samples, live adapter with history-only Scene versus batch cached inference",
              "ledger": ledger}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)
    print("NEW LIVE PARITY PASS", len(ledger), flush=True)


if __name__ == "__main__":
    main()
