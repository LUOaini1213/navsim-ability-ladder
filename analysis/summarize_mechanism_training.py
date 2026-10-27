"""Summarize verified training diagnostics; these are not driving-score findings."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from verify_mechanism_core import CONFIGS, ROOT, digest, verify

LOSSES = ("ade_m", "fde_m", "xy_l1_m", "heading_l1_rad", "mixed_l1")
DIAGNOSTICS = ("cpu_single_scene_ms", "same_scene_reverse_order_max_pose_difference", "elapsed_seconds")


def distribution(values):
    if len(values) != 3 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Diagnostics require three finite, nonnegative values")
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "seeds": {str(seed): value for seed, value in enumerate(values)}}


def summarize(core, model_root):
    rows = {row["tag"]: row for row in core["runs"]}
    summary, histories = {}, {}
    for config in CONFIGS:
        runs = [rows[f"{config}_f1_steps_s{seed}"] for seed in range(3)]
        if len({row["parameters"] for row in runs}) != 1:
            raise ValueError("Parameter counts differ across seeds")
        if {row["inference_threads"] for row in runs} != {2}:
            raise ValueError("Timing table requires the registered two-thread execution")
        summary[config] = {"parameters": runs[0]["parameters"],
                           "losses": {key: distribution([row["final_validation"][key] for row in runs]) for key in LOSSES},
                           "diagnostics": {key: distribution([row[key] for row in runs]) for key in DIAGNOSTICS}}
        if config in ("M0", "M3", "M4", "M5") and summary[config]["diagnostics"][DIAGNOSTICS[1]]["max"] > 1e-5:
            raise ValueError("Order-invariant branch failed the registered reversal diagnostic: "+config)
        for row in runs:
            path = model_root / row["tag"] / "history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            if (not history or history[-1]["step"] != row["optimizer_steps"]
                or history[-1]["validation"] != row["final_validation"]
                or any(a["step"] >= b["step"] for a, b in zip(history, history[1:]))):
                raise ValueError("Convergence history mismatch: "+row["tag"])
            histories[row["tag"]] = digest(path)
    return {"scope": "Full-budget last-checkpoint training diagnostics only; no PDMS result or deployment claim",
            "protocol_sha256": core["protocol_sha256"], "donors_sha256": core["donors_sha256"],
            "normalization_sha256": next(iter(rows.values()))["normalization_sha256"],
            "n_runs": len(rows), "optimizer_steps_per_run": core["optimizer_steps_per_run"],
            "parameter_ratio": core["parameter_ratio"], "summary": summary, "history_sha256": histories,
            "checkpoint_sha256": {tag: row["checkpoint_sha256"] for tag, row in rows.items()},
            "limitations": ["Validation imitation losses are not official driving metrics; checkpoint selection was not based on these losses",
                            "Min/max describe three fitted seeds, not confidence intervals or population training uncertainty",
                            "CPU timing uses one cached tensor scene, 10 warm-ups and 200 forwards, two threads, on a shared machine; extraction, map routing, sensors, transfer and scoring are excluded",
                            "Reversal compares up to the first 512 fixed validation inputs; concat sensitivity is a diagnostic, not a shuffled-training comparison",
                            "Pose-difference maximum mixes coordinate metres and raw-heading radians; it is not a geometric distance"]}


def markdown(report):
    lines = ["# Verified core training diagnostics", "", report["scope"], "",
             f"All {report['n_runs']} runs: {report['optimizer_steps_per_run']:,} optimizer updates; last checkpoint; parameters max/min = {report['parameter_ratio']:.4f}.", "",
             "Entries are three-seed mean [min, max], not confidence intervals.", "",
             "| condition | parameters | ADE (m) | coordinate L1 (m) | heading L1 (rad) | cached forward (ms) | reversal max (m/rad) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def cell(row):
        return f"{row['mean']:.4f} [{row['min']:.4f}, {row['max']:.4f}]"
    for config, row in report["summary"].items():
        values = [config, f"{row['parameters']:,}", cell(row["losses"]["ade_m"]), cell(row["losses"]["xy_l1_m"]),
                  cell(row["losses"]["heading_l1_rad"]), cell(row["diagnostics"]["cpu_single_scene_ms"]),
                  cell(row["diagnostics"]["same_scene_reverse_order_max_pose_difference"])]
        lines.append("| "+" | ".join(values)+" |")
    lines.extend(["", "## Interpretation limits", ""]+["- "+value for value in report["limitations"]])
    lines.extend(["", "The JSON retains individual seed values, all five losses, timing and history/checkpoint hashes.", ""])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--runs", type=Path, default=Path("E:/navsim_workspace/exp/mechanism_v2/models"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/mechanism_v2/training_diagnostics")
    args = ap.parse_args()
    report = summarize(verify(args.runs, args.protocol), args.runs)
    report["summary_source_sha256"] = digest(Path(__file__))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (args.out / "summary.md").write_text(markdown(report), encoding="utf-8")
    print("VERIFIED TRAINING DIAGNOSTICS ONLY", args.out, flush=True)


if __name__ == "__main__":
    main()
