"""Audit the distinct 12-cell same-city follow-up on identical eligible coverage."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_mechanism import METRICS, cluster_interval, interval_text
from city_models import CONDITIONS, verify_all, verify_anchor_metadata
from city_followup import verify_gate
from mechanism_data import session_id, split_indices, subgroups
from predict_city import select_anchor
from verify_mechanism_core import ROOT, digest

CONTRASTS = {"C1-C2": {"C1": 1, "C2": -1}, "C4-C5": {"C4": 1, "C5": -1},
             "C4-C5-C1+C2": {"C4": 1, "C5": -1, "C1": -1, "C2": 1}}


def tags():
    return {f"{condition}_f1_steps_s{seed}" for condition in CONDITIONS for seed in (0, 1, 2)}


def read_scores(scores, predictions, original_predictions, base_path, city_path, audit_path,
                core, city, auxiliary, tokens, logs, base_tokens, base_logs):
    complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
    settings = json.loads((scores / "settings.json").read_text(encoding="utf-8"))
    split = complete["split"]
    if (split not in ("validation", "confirmation", "navtest") or settings["split"] != split
        or not complete["eligible_for_paper"] or settings["limit"] is not None
        or complete.get("final_invalid_scores") != 0
        or complete["settings_sha256"] != digest(scores / "settings.json")
        or settings["wrapper_source_sha256"] != {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")}
        or set(complete["runs"]) != tags() or set(settings["runs"]) != tags()):
        raise ValueError("Incomplete/smoke/wrong-matrix city scores")
    wanted = {str(token): str(log) for token, log in zip(tokens, logs)}
    if not wanted or len(wanted) != len(tokens) or len(tokens) != len(logs):
        raise ValueError("Invalid registered city coverage")
    source = city["sources"]["navtest" if split == "navtest" else "navtrain"]
    arrays, provenance = {}, {}
    for tag in sorted(tags()):
        condition, seed = tag.split("_")[0], int(tag.rsplit("s", 1)[1])
        original_tag = f"{CONDITIONS[condition]}_f1_steps_s{seed}"
        anchor = condition in ("C1", "C4")
        model = next(row for row in core["runs"] if row["tag"] == original_tag) if anchor else auxiliary[condition + "_s" + str(seed)]
        meta = settings["runs"][tag]["meta"]
        expected = {"schema": "same-city-auxiliary-predictions-v1", "run": tag, "original_run": original_tag,
                    "condition": condition, "seed": seed, "split": split, "scenes": len(tokens), "eligible_for_paper": True, "fallbacks": 0,
                    "protocol_sha256": digest(city_path), "base_protocol_sha256": digest(base_path),
                    "donors_sha256": digest(city_path.parent / "donors.npz"), "checkpoint_sha256": model["checkpoint_sha256"],
                    "normalization_sha256": model["normalization_sha256"], "training_settings_sha256": model["settings_sha256"],
                    "features_sha256": source["sha256"], "source_sha256": core["source_sha256"],
                    "city_input_audit_sha256": digest(audit_path), "exporter_sha256": digest(ROOT / "analysis/predict_city.py")}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError("City model/protocol/source metadata mismatch: " + tag)
        path = predictions / (tag + ".npz")
        receipt = json.loads(path.with_suffix(".completed.json").read_text(encoding="utf-8"))
        checksum = digest(path)
        if receipt["meta"] != meta or receipt["predictions_sha256"] != checksum or settings["runs"][tag]["predictions_sha256"] != checksum:
            raise ValueError("City prediction receipt/score lineage mismatch: " + tag)
        with np.load(path, allow_pickle=False) as archive:
            poses = archive["poses"]
            if (json.loads(str(archive["meta"])) != meta or not np.array_equal(archive["token"], tokens)
                or not np.array_equal(archive["log_name"], logs) or poses.shape != (len(tokens), 8, 3) or not np.isfinite(poses).all()):
                raise ValueError("City prediction coverage/content mismatch: " + tag)
        if anchor:
            original = original_predictions / (original_tag + ".npz")
            receipt_path = original.with_suffix(".completed.json")
            original_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (meta["anchor_original_predictions_sha256"] != digest(original)
                or original_receipt["predictions_sha256"] != digest(original)
                or meta["anchor_original_receipt_sha256"] != digest(receipt_path)):
                raise ValueError("Original anchor archive/receipt changed: " + tag)
            with np.load(original, allow_pickle=False) as archive:
                original_meta = json.loads(str(archive["meta"]))
                verify_anchor_metadata(original_meta, model, core, base_path, source, len(base_tokens), original_tag, split)
                if (original_receipt["meta"] != original_meta or meta["anchor_original_metadata"] != original_meta
                    or not np.array_equal(archive["token"], base_tokens) or not np.array_equal(archive["log_name"], base_logs)
                    or not np.array_equal(select_anchor(archive["token"], archive["log_name"], archive["poses"], tokens, logs), poses)):
                    raise ValueError("Real-anchor common-cohort filter/poses mismatch: " + tag)
        path = scores / (tag + ".csv.gz")
        if digest(path) != complete["runs"][tag]["csv_sha256"]:
            raise ValueError("City CSV hash changed: " + tag)
        rows = {}
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["token"] in rows or row["valid"].lower() not in ("true", "1"):
                    raise ValueError("Duplicate/invalid city score: " + tag)
                rows[row["token"]] = row
        if (set(rows) != set(wanted) or len(rows) != complete["runs"][tag]["n"]
            or any(rows[token]["log_name"] != log for token, log in wanted.items())):
            raise ValueError("Unpaired/incomplete city token-log coverage: " + tag)
        values = np.array([[float(rows[str(token)][metric]) for metric in METRICS] for token in tokens])
        if (not np.isfinite(values).all() or (values < -1e-8).any() or (values > 1 + 1e-8).any()
            or abs(float(values[:, 0].mean()) - complete["runs"][tag]["mean_score"]) > 1e-10):
            raise ValueError("Invalid city metric/summary: " + tag)
        arrays[tag] = values
        provenance[tag] = dict(meta, csv_sha256=digest(path), predictions_sha256=checksum)
    return arrays, provenance, split


def summarize(arrays, logs, masks, draws):
    if set(arrays) != tags():
        raise ValueError("Exactly 12 fitted city comparison cells required")
    logs = np.asarray(logs)
    sessions = np.array([session_id(log) for log in logs])
    cells = {condition: np.stack([arrays[f"{condition}_f1_steps_s{seed}"] for seed in (0, 1, 2)]) for condition in CONDITIONS}
    if not len(logs) or any(values.shape != (3, len(logs), len(METRICS)) or not np.isfinite(values).all() for values in cells.values()):
        raise ValueError("City arrays must have the same nonempty scene/metric shape")
    deltas = {label: sum(weight * cells[name] for name, weight in weights.items()) for label, weights in CONTRASTS.items()}
    result = {"cells": {}, "contrasts": {}, "subgroups": {}}
    for name, values in cells.items():
        means = values.mean(1)
        result["cells"][name] = {"mean": dict(zip(METRICS, means.mean(0).tolist())),
                                 "seed_metrics": {str(seed): dict(zip(METRICS, row.tolist())) for seed, row in enumerate(means)},
                                 "session_interval": cluster_interval(values[:, :, 0].mean(0), sessions, draws)}
    for label, values in deltas.items():
        paired = values.mean(0)
        result["contrasts"][label] = {"seed_effects": {str(seed): float(row[:, 0].mean()) for seed, row in enumerate(values)},
                                     "metric_effects": dict(zip(METRICS, paired.mean(0).tolist())),
                                     "session_interval": cluster_interval(paired[:, 0], sessions, draws),
                                     "log_interval": cluster_interval(paired[:, 0], logs, draws),
                                     "auxiliary_family_interval": cluster_interval(paired[:, 0], sessions, draws, .05 / 3)}
    for name, mask in masks.items():
        mask = np.asarray(mask, bool)
        if mask.shape != logs.shape:
            raise ValueError("City subgroup mask misalignment")
        row = {"n_scenes": int(mask.sum()), "n_logs": len(set(logs[mask])), "n_sessions": len(set(sessions[mask])), "contrasts": {}}
        if row["n_scenes"]:
            for label, values in deltas.items():
                paired = values[:, mask].mean(0)
                row["contrasts"][label] = {"metric_effects": dict(zip(METRICS, paired.mean(0).tolist())),
                                           "session_interval": cluster_interval(paired[:, 0], sessions[mask], draws)}
        result["subgroups"][name] = row
    return result


def render(report):
    lines = ["# Same-city cross-session follow-up: " + report["split"], "",
             f"Exactly 12 cells; {report['n_scenes']} common eligible scenes, {report['n_sessions']} original sessions.",
             "All 18 base and six auxiliary models are audited; real anchors are filtered by frozen inputs only.",
             "Conditional exploratory follow-up, not a new globally confirmatory family.",
             "Intervals preserve scene weighting and resample whole sessions conditional on the three fitted seeds.",
             "They do not quantify uncertainty over newly trained seeds. Subgroups overlap and are exploratory.", "",
             "| contrast | delta PDMS | per-seed deltas | session CI | three-contrast auxiliary CI |",
             "|---|---:|---|---|---|"]
    for label, row in report["summary"]["contrasts"].items():
        seeds = ", ".join(f"{seed}: {value:+.4f}" for seed, value in row["seed_effects"].items())
        lines.append(f"| {label} | {row['metric_effects']['score']:+.4f} | {seeds} | "
                     + interval_text(row["session_interval"]) + " | " + interval_text(row["auxiliary_family_interval"]) + " |")
    lines.extend(["", "## Coverage and limits", "", json.dumps(report["coverage"], ensure_ascii=False), "",
                  report["limitations"], "NAVSIM v1 PDMS only; no EPDMS, physical-counterfactual or deployment-safety claim.", ""])
    return "\n".join(lines)


def build_report(scores, predictions, workspace, draws=10000):
    """Rebuild from actual model/gate/prediction/score artifacts, never trust JSON alone."""
    args = SimpleNamespace(scores=Path(scores), predictions=Path(predictions), workspace=Path(workspace), draws=draws)
    results = ROOT / "results/mechanism_v2"
    base_path, city_path = results / "protocol/protocol.json", results / "city_protocol/protocol.json"
    audit_path = ROOT / "results/audit_v2/city_inputs.json"
    root = args.workspace / "exp/mechanism_v2"
    core, city, auxiliary = verify_all(root / "models", root / "city_models", base_path, city_path, audit_path)
    gate = verify_gate(city_path.parent / "execution_policy.json", base_path, city_path, results, core)
    if not gate["required"] or json.loads((results / "gate_city.json").read_text(encoding="utf-8")) != gate:
        raise ValueError("City report requires the actual frozen execution trigger")
    split = json.loads((args.scores / "completed.json").read_text(encoding="utf-8"))["split"]
    if split not in ("validation", "confirmation", "navtest"):
        raise ValueError("Unregistered city split")
    settings = json.loads((args.scores / "settings.json").read_text(encoding="utf-8"))
    if settings["scoring_config_sha256"] != digest(args.workspace / "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"):
        raise ValueError("City score configuration changed")
    source = city["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen city feature source changed")
    with np.load(source["path"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("token", "log_name", "map_name", "agents", "speed", "cmd", "lead")}
    indices = split_indices(data, city, split)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base_indices = split_indices(data, base, split)
    tokens, logs = data["token"][indices], data["log_name"][indices]
    arrays, provenance, split = read_scores(args.scores, args.predictions, root / "predictions" / split,
                                          base_path, city_path, audit_path, core, city, auxiliary, tokens, logs,
                                          data["token"][base_indices], data["log_name"][base_indices])
    masks = subgroups(data, indices, city["thresholds"])
    for name in sorted(set(data["map_name"][indices])):
        masks["city:" + str(name)] = data["map_name"][indices] == name
    report = {"schema": "same-city-follow-up-report-v1", "split": split, "n_scenes": len(tokens),
              "n_logs": len(set(logs)), "n_sessions": len({session_id(log) for log in logs}),
              "protocol_sha256": digest(city_path), "base_protocol_sha256": digest(base_path),
              "execution_policy_sha256": digest(city_path.parent / "execution_policy.json"), "gate_sha256": digest(results / "gate_city.json"),
              "analysis_source_sha256": {name: digest(ROOT / "analysis" / name) for name in
                                         ("analyze_city.py", "city_models.py", "city_followup.py", "predict_city.py", "analyze_mechanism.py")},
              "score_settings_sha256": digest(args.scores / "settings.json"), "provenance": provenance,
              "coverage": city["extension"]["coverage"][split], "limitations": city["extension"]["limitations"],
              "summary": summarize(arrays, logs, masks, args.draws)}
    return report


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    args = ap.parse_args()
    report = build_report(args.scores, args.predictions, args.workspace, args.draws)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, text in (("report.json", json.dumps(report, indent=2) + "\n"), ("report.md", render(report))):
        path = args.out / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    print(render(report), flush=True)


if __name__ == "__main__":
    main()
