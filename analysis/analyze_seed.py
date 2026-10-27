"""Audit unchanged core scores plus twelve added runs; summarize five paired seeds."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from predict_seed import metadata
from seed_models import verify_all
from verify_mechanism_core import CONFIGS, ROOT, digest

METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
           "time_to_collision_within_bound", "ego_progress", "comfort", "driving_direction_compliance")


def matches_meta(actual, wanted):
    for key, value in wanted.items():
        present = actual.get(key)
        if key == "source_sha256":
            if not isinstance(present, dict):
                return False
            present = {name.replace("\\", "/"): checksum for name, checksum in present.items()}
            value = {name.replace("\\", "/"): checksum for name, checksum in value.items()}
        if present != value:
            return False
    return True


def inspect_prediction(path, wanted_meta, tokens, logs):
    """Lazy array dependency; metadata, receipt, bytes and exact cohort all matter."""
    import numpy as np
    receipt = json.loads(path.with_suffix(".completed.json").read_text(encoding="utf-8"))
    if receipt["predictions_sha256"] != digest(path):
        raise ValueError("Prediction receipt/bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        actual_meta = json.loads(str(archive["meta"]))
        if (actual_meta != receipt["meta"]
                or not matches_meta(actual_meta, wanted_meta)
                or not np.array_equal(archive["token"], tokens)
                or not np.array_equal(archive["log_name"], logs)
                or archive["poses"].shape != (len(tokens), 8, 3)
                or not np.isfinite(archive["poses"]).all()):
            raise ValueError("Prediction metadata/cohort/output mismatch")
    return actual_meta


def audit_archive(folder, predictions, expected_meta, tokens, logs, split, workspace,
                  inspect=inspect_prediction):
    """Pure CSV audit apart from the injected/default strict prediction inspector."""
    wanted = dict(zip(map(str, tokens), map(str, logs)))
    if not wanted or len(wanted) != len(tokens) or len(tokens) != len(logs):
        raise ValueError("Nonempty unique aligned frozen cohort required")
    complete = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    settings_path = folder / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    expected = set(expected_meta)
    if len({meta["protocol_sha256"] for meta in expected_meta.values()}) != 1:
        raise ValueError("Mixed training protocols must be audited as separate score archives")
    cache = workspace / "exp" / ("metric_cache_navtest" if split == "navtest" else "mechanism_v2/metric_cache")
    wrapper = {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")}
    scoring_config = workspace / "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
    if (complete.get("eligible_for_paper") is not True or complete["split"] != split
            or settings["split"] != split or settings["limit"] is not None
            or complete.get("final_invalid_scores", complete.get("failures", 0)) != 0
            or complete["settings_sha256"] != digest(settings_path)
            or set(complete["runs"]) != expected or set(settings["runs"]) != expected
            or settings["wrapper_source_sha256"] != wrapper
            or settings["scoring_config_sha256"] != digest(scoring_config)
            or Path(settings["cache_root"]).resolve() != cache.resolve()):
        raise ValueError("Incomplete/smoke/mismatched official score archive")
    arrays, provenance = {}, {}
    for tag in sorted(expected):
        meta = settings["runs"][tag]["meta"]
        wanted_meta = expected_meta[tag]
        if not matches_meta(meta, wanted_meta):
            raise ValueError("Prediction/model/training-protocol identity mismatch")
        prediction = predictions / (tag + ".npz")
        if digest(prediction) != settings["runs"][tag]["predictions_sha256"]:
            raise ValueError("Scored prediction bytes changed")
        if inspect(prediction, wanted_meta, tokens, logs) != meta:
            raise ValueError("Scored prediction metadata differs from archive")
        path = folder / (tag + ".csv.gz")
        csv_hash = digest(path)
        if csv_hash != complete["runs"][tag]["csv_sha256"]:
            raise ValueError("Official score CSV changed")
        rows = {}
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                token = row["token"]
                values = [float(row[metric]) for metric in METRICS]
                if (token in rows or row["valid"].lower() not in ("true", "1")
                        or not all(math.isfinite(value) and -1e-8 <= value <= 1 + 1e-8 for value in values)):
                    raise ValueError("Duplicate/invalid/nonfinite official score")
                rows[token] = row["log_name"], values
        if (len(rows) != complete["runs"][tag]["n"] or set(rows) != set(wanted)
                or any(rows[token][0] != log for token, log in wanted.items())):
            raise ValueError("Exact token/log score coverage mismatch")
        values = [rows[str(token)][1] for token in tokens]
        mean = complete["runs"][tag]["mean_score"]
        if (not isinstance(mean, (int, float)) or isinstance(mean, bool) or not math.isfinite(mean)
                or abs(math.fsum(row[0] for row in values) / len(values) - mean) > 1e-10):
            raise ValueError("Official score summary mean mismatch")
        arrays[tag] = values
        provenance[tag] = {"csv_sha256": csv_hash, "predictions_sha256": digest(prediction),
                           "prediction_receipt_sha256": digest(prediction.with_suffix(".completed.json")),
                           "training_protocol_sha256": meta["protocol_sha256"],
                           **{key: meta[key] for key in ("checkpoint_sha256", "normalization_sha256", "training_settings_sha256", "donors_sha256")}}
    return arrays, provenance, {"settings_sha256": digest(settings_path),
                                "completed_sha256": digest(folder / "completed.json"),
                                "attempt_failures_recorded": complete.get("attempt_failures_recorded")}


def expected_metadata(split, count, core, added, base_path, protocol_path, registration_path):
    base = json.loads(base_path.read_text(encoding="utf-8"))
    source = base["sources"]["navtest" if split == "navtest" else "navtrain"]
    original = {}
    for model in core["runs"]:
        original[model["tag"]] = {"schema": base["schema"], "run": model["tag"], "split": split, "scenes": count,
             "protocol_sha256": digest(base_path), "checkpoint_sha256": model["checkpoint_sha256"],
             "normalization_sha256": model["normalization_sha256"],
             "training_settings_sha256": model["settings_sha256"], "donors_sha256": core["donors_sha256"],
             "source_sha256": core["source_sha256"],
             "features_sha256": source["sha256"], "eligible_for_paper": True, "fallbacks": 0}
    auxiliary = {f"{config}_f1_steps_s{seed}": metadata(config, seed, split, count, core, added,
                                                     base_path, protocol_path, registration_path)
                 for seed in (3, 4) for config in CONFIGS}
    return original, auxiliary


def build_report(workspace, results, split, draws=10000):
    if split not in ("validation", "confirmation", "navtest") or draws != 10000:
        raise ValueError("Registered split and 10,000 fixed bootstrap draws required")
    base_path = results / "protocol/protocol.json"
    registration_path = results / "seed_protocol/registration.json"
    protocol_path = registration_path.parent / "active/protocol.json"
    work = workspace / "exp/mechanism_v2"
    core, protocol, added, gate = verify_all(work / "models", work / "seed_models", base_path,
                                            base_path.parent / "followup_policy.json", registration_path, results)
    import numpy as np
    from analyze_mechanism import METRICS as CORE_METRICS, summarize
    from mechanism_data import session_id, split_indices, subgroups
    if METRICS != CORE_METRICS:
        raise ValueError("Canonical metric order changed")
    source = protocol["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen features changed before seed analysis")
    with np.load(source["path"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("token", "log_name", "map_name", "agents", "speed", "cmd", "lead")}
    indices = split_indices(data, protocol, split)
    tokens, logs = data["token"][indices], data["log_name"][indices]
    original_meta, added_meta = expected_metadata(split, len(indices), core, added, base_path, protocol_path, registration_path)
    originals, original_provenance, original_receipt = audit_archive(results / "scores" / split,
                       work / "predictions" / split, original_meta, tokens, logs, split, workspace)
    additions, added_provenance, added_receipt = audit_archive(results / "seed_scores" / split,
                       work / "seed_predictions" / split, added_meta, tokens, logs, split, workspace)
    arrays = {tag: np.asarray(values, dtype=np.float64) for tag, values in {**originals, **additions}.items()}
    if len(arrays) != 30 or len({row["normalization_sha256"] for row in {**original_provenance, **added_provenance}.values()}) != 1:
        raise ValueError("All thirty aligned runs and one full-data normalizer required")
    masks = subgroups(data, indices, protocol["thresholds"])
    for city in sorted(set(data["map_name"][indices])):
        masks["city:" + str(city)] = data["map_name"][indices] == city
    summary = summarize(arrays, logs, masks, protocol["primary_contrasts"], [0, 1, 2, 3, 4], draws)
    for row in summary["contrasts"].values():
        effects = row["seed_effects"]
        row["seed_extension_diagnostic"] = {
            "original_three_mean": math.fsum(effects[str(seed)] for seed in (0, 1, 2)) / 3,
            "added_two_mean": math.fsum(effects[str(seed)] for seed in (3, 4)) / 2,
            "five_seed_min": min(effects.values()), "five_seed_max": max(effects.values()),
            "discordance_at_registered_threshold": min(effects.values()) <= -.01 and max(effects.values()) >= .01}
    return {"schema": "gated-five-seed-analysis-v1", "split": split, "n_scenes": len(tokens),
            "n_logs": len(set(logs)), "n_sessions": len({session_id(log) for log in logs}),
            "base_protocol_sha256": digest(base_path), "extension_protocol_sha256": digest(protocol_path),
            "registration_sha256": digest(registration_path), "activation_receipt_sha256": digest(protocol_path.parent / "completed.json"),
            "gate": gate, "paired_seeds": [0, 1, 2, 3, 4], "reused_core_runs": 18, "added_runs": 12,
            "score_archives": {"original_core": original_receipt, "added_seeds": added_receipt},
            "provenance": {**original_provenance, **added_provenance},
            "analysis_source_sha256": {name: digest(ROOT / "analysis" / name) for name in
                                       ("analyze_seed.py", "analyze_mechanism.py", "predict_seed.py", "seed_models.py", "seed_followup.py")},
            "summary": summary,
            "disclosure": "Outcome-gated five-seed sensitivity study, not fresh independent confirmation. Session intervals condition on these five fitted seeds, not population training uncertainty; three adjusted primary intervals do not provide unconditional confirmatory guarantees after outcome-based gating. Original NAVSIM v1 score bytes and training-protocol identities are reused unchanged. Subgroups/secondary contrasts and navtest are exploratory; DDC is diagnostic. No sensor-SOTA, causal interaction or deployment-safety claim. " + protocol["disclosure"]}


def render(report):
    from analyze_mechanism import interval_text
    lines = ["# Five-seed outcome-gated sensitivity: " + report["split"], "",
             "Thirty aligned full-budget runs: eighteen original fits plus twelve added fits. No favorable seed or checkpoint selection.",
             report["disclosure"], "",
             f"Scenes: {report['n_scenes']}; logs: {report['n_logs']}; original sessions: {report['n_sessions']}.", "",
             "## All seven metrics", "", "| cell | " + " | ".join(METRICS) + " |", "|---|" + "---:|" * 7]
    for config, row in report["summary"]["cells"].items():
        lines.append("| " + config + " | " + " | ".join(f"{row['mean'][metric]:.4f}" for metric in METRICS) + " |")
    lines.extend(["", "## Paired effects for all five seeds", "",
                  "| contrast | mean PDMS effect | seeds 0/1/2/3/4 | 95% session CI | adjusted primary CI |",
                  "|---|---:|---|---|---|"])
    for label, row in report["summary"]["contrasts"].items():
        seeds = ", ".join(f"{seed}: {row['seed_effects'][str(seed)]:+.4f}" for seed in range(5))
        adjusted = interval_text(row["multiplicity_interval"]) if row["primary"] else "exploratory"
        lines.append(f"| {label} | {row['metric_effects']['score']:+.4f} | {seeds} | " + interval_text(row["session_interval"]) + " | " + adjusted + " |")
    lines.extend(["", "Complete per-seed submetrics, overlapping input strata, counts, log/session intervals and original-three versus added-two diagnostics are retained in report.json.",
                  "Scientific figure rendering, visual QA, interpretation and final project audit remain separate.", ""])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--split", choices=("validation", "confirmation", "navtest"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = build_report(args.workspace, ROOT / "results/mechanism_v2", args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    # Only source-rebuilt reports are eligible; different prior bytes are preserved.
    values = {"report.json": json.dumps(report, indent=2) + "\n", "report.md": render(report)}
    for name, value in values.items():
        path = args.out / name
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise ValueError("Existing five-seed report differs; preserve it and use a fresh output version")
    for name, value in values.items():
        path = args.out / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(value)
    print("FIVE-SEED SOURCE-REBUILT ANALYSIS COMPLETE; figures and final project audit remain", flush=True)


if __name__ == "__main__":
    main()
