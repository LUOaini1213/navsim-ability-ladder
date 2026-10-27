"""Audit archived features/checkpoints and real-scene train/inference parity.

Works log-by-log to bound memory. Real-scene sampling is deterministic and
explicitly reported; it is not a claim of full-dataset map availability.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from runtime import ROOT, configure


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace")
    ap.add_argument("--logs-per-split", type=int, default=12)
    ap.add_argument("--scenes-per-log", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2")
    ap.add_argument("--reuse-real-audit", type=Path,
                    help="Reuse an existing real-scene ledger only if all feature/source hashes and sampling agree")
    args = ap.parse_args()
    workspace = configure(args.workspace)
    import numpy as np
    import torch
    from build_features import scene_filter_for, SPLIT_TO_DATA
    from navsim.agents.ladder_features import scene_blocks, assemble
    from navsim.agents.ladder_mlp_agent import load_run, LadderMLPAgent
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.agents.centerline_util import to_global

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema_version": 2, "python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__, "workspace": str(workspace),
              "agent_source": str(ROOT / "agents"), "features": {}, "checkpoints": [],
              "sampling": {"logs_per_split": args.logs_per_split, "scenes_per_log": args.scenes_per_log},
              "limitations": ["Real-scene/map parity is sampled, not exhaustive.",
                              "Archived scores have no historical fallback ledger; it cannot be reconstructed as fact.",
                              "No-future access is tested by supplying only history frames to feature extraction.",
                              "Velocity finite differences are a diagnostic, not an exact annotation identity."]}
    sources = [ROOT / "agents/ladder_features.py", ROOT / "agents/ladder_mlp_agent.py",
               ROOT / "agents/centerline_util.py",
               workspace / "navsim/navsim/common/dataclasses.py",
               workspace / "navsim/navsim/planning/scenario_builder/navsim_scenario_utils.py"]
    report["source_sha256"] = {str(p): sha256(p) for p in sources}
    report["devkit_revision"] = subprocess.check_output(
        ["git", "-C", str(workspace / "navsim"), "rev-parse", "HEAD"], text=True).strip()
    datasets = {}
    for split in ("navtrain", "navtest"):
        path = workspace / "exp/features" / ("features_" + split + ".npz")
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        datasets[split] = data
        agents = data["agents"]
        padded = agents[:, :, 10] == 0
        numeric = [key for key, arr in data.items() if np.issubdtype(arr.dtype, np.number)]
        info = {"path": str(path), "sha256": sha256(path), "scenes": len(data["token"]),
                "logs": len(np.unique(data["log_name"])),
                "unique_tokens": len(np.unique(data["token"])),
                "nonfinite": {key: int((~np.isfinite(data[key])).sum()) for key in numeric},
                "bad_masks": int((~np.isin(agents[:, :, 10], [0, 1])).sum()),
                "nonzero_padding": int(np.count_nonzero(agents[:, :, :10][padded])),
                "loss_units": "Historical pose L1 mixes metres (x/y) and radians (heading); ds is metres."}
        report["features"][split] = info
        print(split, info["scenes"], "scenes", info["logs"], "logs", flush=True)
    train, test = datasets["navtrain"], datasets["navtest"]
    report["token_overlap"] = len(set(train["token"]) & set(test["token"]))
    report["log_overlap"] = len(set(train["log_name"]) & set(test["log_name"]))
    with open(ROOT / "results/navtest/cells.csv", encoding="utf-8") as stream:
        archived_cells = {row["cell"] for row in csv.DictReader(stream)}
    report["checkpoint_scope"] = "Exactly the cells archived in results/navtest/cells.csv; excludes trial checkpoints trained on different files."
    for run in sorted((workspace / "exp/ladder").iterdir()):
        if run.name not in archived_cells:
            continue
        if not (run / "model.pt").exists():
            continue
        model, mu, sd, norm = load_run(str(run))
        keep = ~np.isin(train["log_name"], norm["val_logs"])
        if norm["target"] == "ds":
            keep &= train["ds_valid"] > .5
        def matrix(data):
            parts = [data["ego"]]
            if norm["see_map"]:
                parts.append(data["cl"])
            if norm["see_agents"]:
                parts.extend([data["agents"].reshape(-1, 88), data["lead"]])
            return np.concatenate(parts, axis=1).astype(np.float32)
        xtr = matrix(train)[keep]
        mu2, sd2 = xtr.mean(0), xtr.std(0)
        sd2 = np.where(sd2 < 1e-3, 1., sd2)
        xt = (matrix(test) - mu) / sd
        bad = 0
        with torch.no_grad():
            for offset in range(0, len(xt), 512):
                pred = model(torch.from_numpy(xt[offset:offset+512])).numpy()
                bad += int((~np.isfinite(pred)).sum())
        item = {"run": run.name, "checkpoint_sha256": sha256(run / "model.pt"),
                "normalization_train_only_max_mu_error": float(np.max(np.abs(mu-mu2))),
                "normalization_train_only_max_sd_error": float(np.max(np.abs(sd-sd2))),
                "nonfinite_outputs_on_cached_navtest": bad,
                "validation_logs": len(norm["val_logs"]), "select": norm["select"]}
        report["checkpoints"].append(item)
        print("checkpoint", run.name, "norm errors", item["normalization_train_only_max_mu_error"],
              item["normalization_train_only_max_sd_error"], "bad outputs", bad, flush=True)
        del model, xtr, xt
        gc.collect()
    ledger = []
    previous = None
    if args.reuse_real_audit:
        previous = json.loads((args.reuse_real_audit / "audit.json").read_text(encoding="utf-8"))
        if (previous["source_sha256"] != report["source_sha256"] or previous["sampling"] != report["sampling"]
            or any(previous["features"][s]["sha256"] != report["features"][s]["sha256"] for s in datasets)):
            raise ValueError("Cannot reuse real audit: source/features/sampling changed")
        with open(args.reuse_real_audit / "feature_ledger.jsonl", encoding="utf-8") as stream:
            ledger = [json.loads(line) for line in stream]
    velocity_errors = []
    for split, data in ({} if previous else datasets).items():
        logs = np.unique(data["log_name"])
        picked = logs[np.unique(np.linspace(0, len(logs)-1, min(args.logs_per_split, len(logs))).astype(int))]
        index = {str(token): i for i, token in enumerate(data["token"])}
        for log in picked:
            sf = scene_filter_for(split, [str(log)])
            loader = SceneLoader(workspace / "dataset/navsim_logs" / SPLIT_TO_DATA[split], None, sf,
                                 SensorConfig.build_no_sensors())
            available = sorted(set(loader.tokens) & set(index))
            positions = np.unique(np.linspace(0, len(available)-1, min(args.scenes_per_log, len(available))).astype(int))
            for pos in positions:
                token = available[pos]
                row = {"split": split, "log_name": str(log), "token": token}
                try:
                    scene = loader.get_scene_from_token(token)
                    agent_input = loader.get_agent_input_from_token(token)
                    h = scene.scene_metadata.num_history_frames
                    history_scene = SimpleNamespace(scene_metadata=scene.scene_metadata, map_api=scene.map_api,
                                                    frames=scene.frames[:h])
                    blocks = scene_blocks(history_scene, agent_input=agent_input)
                    if blocks is None:
                        row.update(reason="missing_route", passed=False)
                    else:
                        i = index[token]
                        errors = {key: float(np.max(np.abs(data[key][i] - blocks[key])))
                                  for key in ("ego", "cl", "agents", "lead")}
                        target = scene.get_future_trajectory(num_trajectory_frames=8).poses.reshape(-1)
                        errors["y_pose"] = float(np.max(np.abs(target - data["y_pose"][i])))
                        row.update(reason="feature_parity", errors=errors,
                                   passed=max(errors.values()) <= 1e-5)
                        # Compare new live execution with archived cached prediction in four seed-0 cells.
                        row["agents"] = {}
                        for tag in ("pose_kin_s0", "pose_agents_s0", "pose_map_s0", "pose_map_agents_s0"):
                            agent = LadderMLPAgent(run_dir=str(workspace / "exp/ladder" / tag), strict=True)
                            agent.initialize()
                            predicted = agent.compute_trajectory(agent_input, history_scene).poses
                            x = assemble(data["ego"][i], data["cl"][i], data["agents"][i], data["lead"][i],
                                         agent._see_map, agent._see_agents)
                            with torch.no_grad():
                                cached = agent._model(torch.from_numpy((x-agent._mu)/agent._sd)[None]).numpy().reshape(8, 3)
                            difference = float(np.max(np.abs(predicted-cached)))
                            row["agents"][tag] = dict(agent.last_diagnostic, max_cached_prediction_error=difference)
                            row["passed"] &= difference <= 1e-4
                        # Absolute object velocity in the ego basis; compare with global track displacement.
                        current, future = scene.frames[h-1:h+1]
                        dt = (future.timestamp-current.timestamp) / 1e6
                        if .1 < dt < 2:
                            lookup = {t: j for j, t in enumerate(future.annotations.track_tokens)}
                            for j, track in enumerate(current.annotations.track_tokens):
                                if track not in lookup:
                                    continue
                                k = lookup[track]
                                cxy = current.annotations.boxes[j, :2]
                                fxy = future.annotations.boxes[k, :2]
                                cg = to_global(np.array([[*cxy, 0.]]), current.ego_status.ego_pose)[0, :2]
                                fg = to_global(np.array([[*fxy, 0.]]), future.ego_status.ego_pose)[0, :2]
                                yaw = current.ego_status.ego_pose[2]
                                rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                                vg = rotation @ current.annotations.velocity_3d[j, :2]
                                if np.all(np.isfinite(vg)):
                                    velocity_errors.append(float(np.linalg.norm((fg-cg)/dt - vg)))
                except Exception as error:
                    row.update(reason="exception", passed=False, exception_type=type(error).__name__, exception=str(error))
                ledger.append(row)
            print("parity", split, str(log), "checked", len(positions), flush=True)
    report["real_scene_audit"] = {"checked": len(ledger), "passed": sum(r["passed"] for r in ledger),
                                  "reasons": dict(Counter(r["reason"] for r in ledger))}
    report["velocity_convention"] = {"source": "navsim_scenario_utils.annotations_to_detection_tracks rotates stored velocity into global coordinates without adding ego velocity",
        "interpretation": "absolute object velocity, expressed in the ego coordinate basis, not relative velocity",
        "diagnostic_track_pairs": len(velocity_errors),
        "median_global_finite_difference_error_mps": float(np.median(velocity_errors)) if velocity_errors else None,
        "p95_global_finite_difference_error_mps": float(np.percentile(velocity_errors, 95)) if velocity_errors else None}
    if previous:
        report["velocity_convention"] = previous["velocity_convention"]
        report["real_scene_audit"]["reuse"] = "Reused ledger after verifying exact feature/source hashes and sampling parameters."
    report["elapsed_seconds"] = time.monotonic()-started
    report["passed"] = (report["token_overlap"] == report["log_overlap"] == 0
        and all(v["unique_tokens"] == v["scenes"] and not any(v["nonfinite"].values())
                and v["bad_masks"] == v["nonzero_padding"] == 0 for v in report["features"].values())
        and all(c["normalization_train_only_max_mu_error"] < 1e-4
                and c["normalization_train_only_max_sd_error"] < 1e-4
                and c["nonfinite_outputs_on_cached_navtest"] == 0 for c in report["checkpoints"])
        and len(ledger) > 0 and all(r["passed"] for r in ledger))
    with open(args.out / "feature_ledger.jsonl", "w", encoding="utf-8") as stream:
        for row in ledger:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("AUDIT", "PASS" if report["passed"] else "FAIL", "->", args.out, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
