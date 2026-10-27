"""Audit the registered 24 follow-ups plus six reused anchors and paired curves."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_mechanism import METRICS, cluster_interval, interval_text
from mechanism_data import session_id, split_indices
from run_mechanism_curves import matrix, verified_gate, verify_run
from verify_mechanism_core import ROOT, digest, verify


def verify_training(models, protocol_path, policy_path):
    """Confirmation cannot see an incomplete small-data matrix."""
    core = verify(models, protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    jobs = matrix(protocol, policy)
    trained = {row["tag"]: verify_run(models, row, protocol, protocol_path) for row in jobs}
    for fraction in (.1, .3):
        normalizers = {trained[row["tag"]]["normalization_sha256"] for row in jobs if row["fraction"] == fraction}
        if len(normalizers) != 1:
            raise ValueError("Subset-only normalizers differ within a registered fraction")
    trained.update({row["tag"]: row for row in core["runs"] if row["tag"].split("_")[0] in ("M1", "M4")})
    if len(trained) != 30:
        raise ValueError("Exactly 24 small-data models plus six full-data anchors required")
    return core, trained


def read_archives(scores, core, trained, tokens, logs, source_hash):
    complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
    settings = json.loads((scores / "settings.json").read_text(encoding="utf-8"))
    if (not complete["eligible_for_paper"] or settings["limit"] is not None
        or complete.get("final_invalid_scores", complete.get("failures", 0)) != 0
        or complete["settings_sha256"] != digest(scores / "settings.json")
        or set(complete["runs"]) != set(trained) or set(settings["runs"]) != set(trained)):
        raise ValueError("Incomplete, changed or smoke curve score archive")
    expected_wrappers = {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")}
    if settings["wrapper_source_sha256"] != expected_wrappers:
        raise ValueError("Curve scorer source changed")
    wanted = dict(zip(map(str, tokens), map(str, logs)))
    if not wanted or len(wanted) != len(tokens):
        raise ValueError("Empty/duplicate expected curve coverage")
    arrays, provenance = {}, {}
    for tag, model in trained.items():
        meta = settings["runs"][tag]["meta"]
        if (not meta["eligible_for_paper"] or meta["fallbacks"] != 0 or meta["run"] != tag
            or meta["split"] != complete["split"] or meta["scenes"] != len(tokens)
            or meta["protocol_sha256"] != core["protocol_sha256"] or meta["features_sha256"] != source_hash
            or meta["donors_sha256"] != core["donors_sha256"]
            or {key.replace("\\", "/"): value for key, value in meta["source_sha256"].items()} != core["source_sha256"]):
            raise ValueError("Curve prediction provenance mismatch: " + tag)
        for key in ("checkpoint_sha256", "normalization_sha256", "training_settings_sha256"):
            if meta[key] != model["settings_sha256" if key == "training_settings_sha256" else key]:
                raise ValueError("Curve trained artifact mismatch: " + tag)
        path = scores / (tag + ".csv.gz")
        if digest(path) != complete["runs"][tag]["csv_sha256"]:
            raise ValueError("Curve CSV changed: " + tag)
        rows = {}
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                token = row["token"]
                if token in rows or row["valid"].lower() not in ("true", "1"):
                    raise ValueError("Invalid/duplicate curve row")
                rows[token] = row
        if (set(rows) != set(wanted) or complete["runs"][tag]["n"] != len(wanted)
            or any(rows[token]["log_name"] != log for token, log in wanted.items())):
            raise ValueError("Curve token/log coverage mismatch")
        values = np.array([[float(rows[str(token)][metric]) for metric in METRICS] for token in tokens])
        if not np.isfinite(values).all() or np.any(values < -1e-8) or np.any(values > 1 + 1e-8):
            raise ValueError("Nonfinite/out-of-range curve metric")
        if abs(values[:, 0].mean() - complete["runs"][tag]["mean_score"]) > 1e-10:
            raise ValueError("Curve score mean mismatch")
        arrays[tag] = values
        provenance[tag] = dict(meta, predictions_sha256=settings["runs"][tag]["predictions_sha256"], csv_sha256=digest(path))
    return arrays, provenance, complete


def tag_for(config, fraction, budget, seed):
    # Full-data equal-epoch and fixed-update training are the same 13,000-update
    # run. Reuse the anchor, never present it as an independent replication.
    return f"{config}_f{fraction:g}_{'steps' if fraction == 1. else budget}_s{seed}"


def summarize(arrays, logs, protocol, draws=10000):
    sessions = np.array([session_id(log) for log in logs])
    seeds = protocol["training"]["seeds"]
    if seeds != [0, 1, 2]:
        raise ValueError("Registered three-seed curve summary required")
    cells, differences = {}, {}
    for budget in ("steps", "epochs"):
        for fraction in (.1, .3, 1.):
            key = f"{budget}/f{fraction:g}"
            n = protocol["fractions"][str(fraction)]["n_scenes"]
            cells[key] = {"n_train": n, "fraction": fraction, "budget": budget, "reused_full_data_anchor": fraction == 1., "models": {}}
            values = {}
            for config in ("M1", "M4"):
                values[config] = np.stack([arrays[tag_for(config, fraction, budget, seed)] for seed in seeds])
                seed_means = values[config].mean(1)
                cells[key]["models"][config] = {
                    "metric_means": dict(zip(METRICS, seed_means.mean(0).tolist())),
                    "seed_metrics": {str(seed): dict(zip(METRICS, mean.tolist())) for seed, mean in zip(seeds, seed_means)},
                    "score_interval": cluster_interval(values[config][:, :, 0].mean(0), sessions, draws)}
            delta = values["M4"] - values["M1"]
            differences[key] = delta
            cells[key]["M4-M1"] = {
                "metric_effects": dict(zip(METRICS, delta.mean((0, 1)).tolist())),
                "seed_effects": {str(seed): float(row[:, 0].mean()) for seed, row in zip(seeds, delta)},
                "session_interval": cluster_interval(delta[:, :, 0].mean(0), sessions, draws)}
    interactions = {}
    for budget in ("steps", "epochs"):
        for fraction in (.1, .3):
            key = f"{budget}/f{fraction:g}"
            delta = differences[key] - differences["steps/f1"]
            interactions[key + "-full"] = {
                "metric_effects": dict(zip(METRICS, delta.mean((0, 1)).tolist())),
                "seed_effects": {str(seed): float(row[:, 0].mean()) for seed, row in zip(seeds, delta)},
                "session_interval": cluster_interval(delta[:, :, 0].mean(0), sessions, draws),
                "four_contrast_interval": cluster_interval(delta[:, :, 0].mean(0), sessions, draws, .05/4)}
    return {"cells": cells, "small_minus_full_encoder_effect": interactions,
            "interval_scope": "Whole-original-session bootstrap conditional on three fitted seeds; no new-training-run uncertainty or causal data effect",
            "multiplicity_scope": "Four curve-interaction sensitivity contrasts only; not an unqualified global adaptive-study family-wise guarantee"}


def render(report):
    lines = ["# Registered data-budget sensitivity: " + report["split"], "",
             "24 verified small-data models plus six reused full-data anchors; exact shared evaluation coverage.",
             "All seeds are averaged as scores, never ensembled as trajectories. Fixed-update and equal-epoch curves are separate.",
             "The 100% point is reused in both curves, not two independent experiments.",
             report["summary"]["interval_scope"], "",
             "| budget | fraction | training scenes | M1 PDMS | M4 PDMS | paired M4-M1 | 95% session interval |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for key, cell in report["summary"]["cells"].items():
        effect = cell["M4-M1"]
        lines.append(f"| {cell['budget']} | {cell['fraction']:.0%} | {cell['n_train']} | "
                     f"{cell['models']['M1']['metric_means']['score']:.4f} | {cell['models']['M4']['metric_means']['score']:.4f} | "
                     f"{effect['metric_effects']['score']:+.4f} | {interval_text(effect['session_interval'])} |")
    lines.extend(["", "## Small-data minus full-data encoder effect", "",
                  "Paired difference of differences: (M4-M1) at a smaller fraction minus (M4-M1) at 100%.",
                  report["summary"]["multiplicity_scope"], "",
                  "| contrast | delta | 95% session interval | four-contrast adjusted interval |",
                  "|---|---:|---|---|"])
    for name, row in report["summary"]["small_minus_full_encoder_effect"].items():
        lines.append(f"| {name} | {row['metric_effects']['score']:+.4f} | {interval_text(row['session_interval'])} | "
                     f"{interval_text(row['four_contrast_interval'])} |")
    lines.extend(["", "Full per-seed/submetric results, actual update budgets and provenance are in report.json.",
                  "Outcome-gated sensitivity analysis; confirmation history and privileged-input limitations remain applicable.", ""])
    return "\n".join(lines)


def build_report(scores, models, protocol_path, draws=10000, workspace=None):
    """Recompute audited statistics for both report writing and figure preflight."""
    args = SimpleNamespace(scores=Path(scores), models=Path(models), protocol=Path(protocol_path), draws=draws)
    workspace = Path(workspace or os.environ.get("NAVSIM_WORKSPACE", "E:/navsim_workspace"))
    policy_path = args.protocol.parent / "followup_policy.json"
    core, trained = verify_training(args.models, args.protocol, policy_path)
    results = ROOT / "results/mechanism_v2"
    gate = verified_gate(results / "analysis/validation/report.json", results / "scores/validation",
                         results / "gate_validation.json", policy_path, core)
    if not gate["learning_curve_required"]:
        raise ValueError("Curve analysis requires the registered positive execution gate")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    split = json.loads((args.scores / "completed.json").read_text(encoding="utf-8"))["split"]
    if split not in ("validation", "confirmation", "navtest"):
        raise ValueError("Unknown curve split")
    source = protocol["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen curve feature source changed")
    with np.load(source["path"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("token", "log_name")}
    indices = split_indices(data, protocol, split)
    tokens, logs = data["token"][indices], data["log_name"][indices]
    arrays, provenance, complete = read_archives(args.scores, core, trained, tokens, logs, source["sha256"])
    settings = json.loads((args.scores / "settings.json").read_text(encoding="utf-8"))
    if settings["scoring_config_sha256"] != digest(workspace / "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"):
        raise ValueError("Curve score configuration changed")
    report = {"split": split, "protocol_sha256": digest(args.protocol), "policy_sha256": digest(policy_path),
              "analysis_source_sha256": {name: digest(ROOT / "analysis" / name) for name in
                                         ("analyze_mechanism_curves.py", "analyze_mechanism.py", "run_mechanism_curves.py",
                                          "verify_mechanism_core.py", "mechanism_data.py")},
              "gate_sha256": digest(results / "gate_validation.json"), "score_settings_sha256": digest(args.scores / "settings.json"),
              "attempt_failures_recorded": complete.get("attempt_failures_recorded", 0),
              "n_scenes": len(tokens), "n_logs": len(set(logs)), "n_sessions": len({session_id(log) for log in logs}),
              "provenance": provenance, "training": trained, "summary": summarize(arrays, logs, protocol, args.draws)}
    return report


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=Path("E:/navsim_workspace/exp/mechanism_v2/models"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--workspace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()
    report = build_report(args.scores, args.models, args.protocol, args.draws, args.workspace)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(render(report), encoding="utf-8")
    print(render(report), flush=True)


if __name__ == "__main__":
    main()
