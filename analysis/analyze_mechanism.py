"""Audit and summarize complete controlled evaluations; never promote smoke data."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from mechanism_data import CONFIGS, digest, session_id, split_indices, subgroups
from runtime import ROOT

METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
           "time_to_collision_within_bound", "ego_progress", "comfort", "driving_direction_compliance")
METRIC_LABELS = ("PDMS", "NC", "DAC", "TTC", "EP", "C", "DDC")
SECONDARY_CONTRASTS = ("M1-M2", "M3-M0", "M4-M0", "M5-M2")


def cluster_interval(values, groups, draws=10000, alpha=.05, seed=12345):
    """Paired ratio bootstrap of complete clusters, keeping scene weighting.

    With fewer than two clusters there is no population interval; returning a
    zero-width interval would give misleading certainty.
    """
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups)
    if values.ndim != 1 or len(values) == 0 or len(values) != len(groups):
        raise ValueError("Nonempty, aligned one-dimensional values/groups required")
    if not np.isfinite(values).all() or draws < 100 or not 0 < alpha < 1:
        raise ValueError("Finite values, >=100 draws and valid alpha required")
    unique, inverse = np.unique(groups, return_inverse=True)
    result = {"mean": float(values.mean()), "n_scenes": len(values), "n_clusters": len(unique),
              "alpha": alpha, "draws": draws, "seed": seed, "low": None, "high": None}
    if len(unique) < 2:
        result["status"] = "insufficient_clusters"
        return result
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.RandomState(seed)
    samples = []
    for offset in range(0, draws, 1000):
        chosen = rng.randint(len(unique), size=(min(1000, draws-offset), len(unique)))
        samples.append(sums[chosen].sum(1)/counts[chosen].sum(1))
    low, high = np.quantile(np.concatenate(samples), [alpha/2, 1-alpha/2])
    result.update(low=float(low), high=float(high), status="ok")
    return result


def read_score_archive(folder, protocol, protocol_hash, tokens, logs):
    """Require provenance, all 18 cells, exact token/log coverage and finite metrics."""
    complete = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    if not complete["eligible_for_paper"] or settings["limit"] is not None:
        raise ValueError("Smoke/subset evaluations cannot enter the paper report")
    if complete["settings_sha256"] != digest(folder / "settings.json"):
        raise ValueError("Scoring settings hash mismatch")
    if complete.get("final_invalid_scores", complete.get("failures", 0)) != 0:
        raise ValueError("Incomplete/invalid scores")
    expected = {f"{config}_f1_steps_s{seed}" for config in CONFIGS for seed in protocol["training"]["seeds"]}
    if set(complete["runs"]) != expected or set(settings["runs"]) != expected:
        raise ValueError("Exactly the preregistered 18 core runs are required")
    wanted = {str(token): str(log) for token, log in zip(tokens, logs)}
    if len(wanted) != len(tokens):
        raise ValueError("Duplicate source tokens")
    arrays, provenance = {}, {}
    expected_sources = {name: digest(ROOT / name) for name in ("agents/mechanism_model.py", "analysis/mechanism_data.py", "analysis/train_mechanism.py")}
    normalizers, donor_hashes = set(), set()
    for name in sorted(expected):
        meta = settings["runs"][name]["meta"]
        if (not meta["eligible_for_paper"] or meta["protocol_sha256"] != protocol_hash or meta["run"] != name
            or meta["split"] != complete["split"] or meta["fallbacks"] != 0
            or meta["scenes"] != len(wanted)):
            raise ValueError("Prediction metadata mismatch: " + name)
        source = protocol["sources"]["navtest" if complete["split"] == "navtest" else "navtrain"]
        if meta["features_sha256"] != source["sha256"]:
            raise ValueError("Prediction feature provenance mismatch: " + name)
        trained_sources = {key.replace("\\", "/"): value for key, value in meta["source_sha256"].items()}
        if trained_sources != expected_sources or not meta["training_settings_sha256"]:
            raise ValueError("Prediction source/settings provenance mismatch: "+name)
        normalizers.add(meta["normalization_sha256"])
        donor_hashes.add(meta["donors_sha256"])
        path = folder / (name+".csv.gz")
        if digest(path) != complete["runs"][name]["csv_sha256"]:
            raise ValueError("CSV hash mismatch: " + name)
        rows = {}
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                token = row["token"]
                if token in rows or row["valid"].lower() not in ("true", "1"):
                    raise ValueError("Duplicate/invalid scene in " + name)
                rows[token] = row
        if set(rows) != set(wanted) or len(rows) != complete["runs"][name]["n"]:
            raise ValueError("Token coverage mismatch: " + name)
        if any(rows[token]["log_name"] != log for token, log in wanted.items()):
            raise ValueError("Token/log misalignment: " + name)
        values = np.array([[float(rows[str(token)][metric]) for metric in METRICS] for token in tokens])
        if not np.isfinite(values).all() or np.any(values < -1e-8) or np.any(values > 1+1e-8):
            raise ValueError("Nonfinite/out-of-range metrics: " + name)
        if abs(values[:, 0].mean()-complete["runs"][name]["mean_score"]) > 1e-10:
            raise ValueError("Summary mean mismatch: " + name)
        arrays[name] = values
        provenance[name] = {"csv_sha256": digest(path), "checkpoint_sha256": meta["checkpoint_sha256"],
                            "normalization_sha256": meta["normalization_sha256"], "training_settings_sha256": meta["training_settings_sha256"],
                            "donors_sha256": meta["donors_sha256"], "predictions_sha256": settings["runs"][name]["predictions_sha256"]}
    if len(normalizers) != 1 or len(donor_hashes) != 1:
        raise ValueError("Full-data core normalizers or donors differ")
    return arrays, provenance, complete["split"]


def summarize(arrays, logs, masks, primary, seeds, draws):
    logs = np.asarray(logs)
    sessions = np.array([session_id(log) for log in logs])
    cells = {config: np.stack([arrays[f"{config}_f1_steps_s{seed}"] for seed in seeds]) for config in CONFIGS}
    result = {"cells": {}, "contrasts": {}, "subgroups": {}}
    for config, values in cells.items():
        seed_means = values.mean(1)
        result["cells"][config] = {
            "mean": dict(zip(METRICS, seed_means.mean(0).tolist())),
            "seed_metrics": {str(seed): dict(zip(METRICS, value.tolist())) for seed, value in zip(seeds, seed_means)},
            "seed_score_range": [float(seed_means[:, 0].min()), float(seed_means[:, 0].max())],
            "score_interval": cluster_interval(values[:, :, 0].mean(0), sessions, draws)}
    comparisons = list(primary) + [name for name in SECONDARY_CONTRASTS if name not in primary]
    for label in comparisons:
        left, right = label.split("-")
        delta = cells[left]-cells[right]
        row = {"primary": label in primary,
               "seed_effects": {str(seed): float(values[:, 0].mean()) for seed, values in zip(seeds, delta)},
               "metric_effects": dict(zip(METRICS, delta.mean((0, 1)).tolist())),
               "session_interval": cluster_interval(delta[:, :, 0].mean(0), sessions, draws),
               "log_interval": cluster_interval(delta[:, :, 0].mean(0), logs, draws)}
        if label in primary:
            row["multiplicity_interval"] = cluster_interval(delta[:, :, 0].mean(0), sessions, draws, .05/len(primary))
        result["contrasts"][label] = row
    for group, mask in masks.items():
        mask = np.asarray(mask, bool)
        if len(mask) != len(logs):
            raise ValueError("Subgroup mask misalignment")
        count = int(mask.sum())
        row = {"n_scenes": count, "n_logs": len(set(logs[mask])), "n_sessions": len(set(sessions[mask])),
               "cells": {}, "contrasts": {}}
        if count:
            for config, values in cells.items():
                row["cells"][config] = dict(zip(METRICS, values[:, mask].mean((0, 1)).tolist()))
            for label in primary:
                left, right = label.split("-")
                delta = (cells[left]-cells[right])[:, mask].mean(0)
                row["contrasts"][label] = {
                    "metric_effects": dict(zip(METRICS, delta.mean(0).tolist())),
                    "session_interval": cluster_interval(delta[:, 0], sessions[mask], draws)}
        result["subgroups"][group] = row
    return result


def interval_text(interval):
    if interval["low"] is None:
        return "not estimable (<2 clusters)"
    return f"[{interval['low']:+.4f}, {interval['high']:+.4f}]"


def render(report):
    lines = ["# Controlled dynamic-object study: "+report["split"], "",
             "All 18 preregistered full-budget core runs passed provenance and exact scene-coverage checks.",
             "The estimand is the scene-weighted average of per-seed scores, not the score of an ensemble trajectory.",
             "Intervals resample whole original driving sessions and condition on three fitted training seeds.",
             "Seed-specific effects are descriptive; these intervals do not quantify new-training-run uncertainty.",
             "Three primary contrasts use Bonferroni-adjusted percentile intervals (approximate family-wise coverage).",
             "Subgroup and secondary comparisons are exploratory, not multiplicity-adjusted.", "",
             f"Scenes: {report['n_scenes']}; logs: {report['n_logs']}; original sessions: {report['n_sessions']}.", "",
             "## Cell metrics", "", "| cell | "+" | ".join(METRIC_LABELS)+" | PDMS seed range |", "|---|"+"---:|"*8]
    for config, row in report["summary"]["cells"].items():
        lines.append("| "+config+" | "+" | ".join(f"{row['mean'][metric]:.4f}" for metric in METRICS)
                     +f" | [{row['seed_score_range'][0]:.4f}, {row['seed_score_range'][1]:.4f}] |")
    lines.extend(["", "## Paired contrasts", "", "| comparison | delta PDMS | seed deltas | 95% session CI | adjusted primary CI |",
                  "|---|---:|---|---|---|"])
    for label, row in report["summary"]["contrasts"].items():
        seed_text = ", ".join(f"{seed}: {value:+.4f}" for seed, value in row["seed_effects"].items())
        lines.append(f"| {label} | {row['metric_effects']['score']:+.4f} | {seed_text} | "
                     +interval_text(row["session_interval"])+" | "
                     +(interval_text(row["multiplicity_interval"]) if row["primary"] else "exploratory")+" |")
    lines.extend(["", "## Frozen-input subgroups", "",
                  "Groups overlap; they are not causal interaction labels. The crossing group is a straight-corridor proxy.",
                  "Small-cluster groups should not support population claims. Complete submetrics are in report.json.", "",
                  "| group | scenes | logs | sessions | comparison | delta PDMS | 95% session CI |", "|---|---:|---:|---:|---|---:|---|"])
    for group, row in report["summary"]["subgroups"].items():
        if not row["n_scenes"]:
            lines.append(f"| {group} | 0 | 0 | 0 | none | - | not estimable |")
        for label, contrast in row["contrasts"].items():
            lines.append(f"| {group} | {row['n_scenes']} | {row['n_logs']} | {row['n_sessions']} | {label} | "
                         f"{contrast['metric_effects']['score']:+.4f} | "+interval_text(contrast["session_interval"])+" |")
    lines.extend(["", "## Boundaries", "", report["disclosure"],
                  "The v2 experiment label denotes this study's second implementation, not NAVSIM benchmark v2.",
                  "These results use NAVSIM v1, privileged current boxes and map centerlines; no sensor-SOTA or deployment-safety claim is made.", ""])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    complete = json.loads((args.scores / "completed.json").read_text(encoding="utf-8"))
    split = complete["split"]
    if split not in ("validation", "confirmation", "navtest"):
        raise ValueError("Unknown split")
    source = protocol["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen feature source changed")
    with np.load(source["path"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("token", "log_name", "map_name", "agents", "speed", "cmd", "lead")}
    idx = split_indices(data, protocol, split)
    tokens, logs = data["token"][idx], data["log_name"][idx]
    arrays, provenance, split = read_score_archive(args.scores, protocol, digest(args.protocol), tokens, logs)
    masks = subgroups(data, idx, protocol["thresholds"])
    for city in sorted(set(data["map_name"][idx])):
        masks["city:"+str(city)] = data["map_name"][idx] == city
    report = {"schema": protocol["schema"], "split": split, "n_scenes": len(tokens),
              "n_logs": len(set(logs)), "n_sessions": len({session_id(log) for log in logs}),
              "protocol_sha256": digest(args.protocol), "score_settings_sha256": digest(args.scores / "settings.json"),
              "provenance": provenance, "disclosure": protocol["disclosure"],
              "summary": summarize(arrays, logs, masks, protocol["primary_contrasts"], protocol["training"]["seeds"], args.draws)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (args.out / "report.md").write_text(render(report), encoding="utf-8")
    print(render(report).split("## Frozen-input subgroups")[0], flush=True)


if __name__ == "__main__":
    main()
