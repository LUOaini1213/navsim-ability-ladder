"""Register a prospective five-seed extension, without fitting or scoring models.

The original three-seed protocol and fitted artifacts are never rewritten.
Activation requires complete, paired, hash-bound validation AND confirmation
score archives. A registration or a passing fixture is not experimental evidence.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from mechanism_gates import decide
from verify_mechanism_core import CONFIGS, ROOT, digest

PRIMARY = ("M1-M0", "M4-M1", "M4-M5")
SPLITS = ("validation", "confirmation")


def matrix(protocol):
    if (protocol["schema"] != "dynamic-objects-v2"
            or protocol["training"]["seeds"] != [0, 1, 2]
            or protocol["primary_contrasts"] != list(PRIMARY)):
        raise ValueError("Original six-condition/three-seed design required")
    n_train = protocol["split_counts"]["train"]["scenes"]
    hp = protocol["training"]
    steps = math.ceil(n_train / hp["batch_size"]) * hp["full_data_epochs_equivalent"]
    if n_train <= 0 or steps <= 0:
        raise ValueError("Positive full-data training budget required")
    return [{"tag": f"{config}_f1_steps_s{seed}", "config": config,
             "seed": seed, "fraction": 1., "budget": "steps",
             "n_train": n_train, "optimizer_steps": steps}
            for seed in (3, 4) for config in CONFIGS]


def signature(base_path, policy_path):
    base = json.loads(base_path.read_text(encoding="utf-8"))
    original = json.loads(policy_path.read_text(encoding="utf-8"))
    if (original["protocol_sha256"] != digest(base_path)
            or original["seed_discordance"]["negative_threshold"] != -.01
            or original["seed_discordance"]["positive_threshold"] != .01):
        raise ValueError("Original seed-discordance policy changed")
    return {"kind": "prospective-five-seed-extension-v1",
            "base_protocol_sha256": digest(base_path),
            "base_donors_sha256": digest(base_path.parent / "donors.npz"),
            "original_followup_policy_sha256": digest(policy_path),
            "registration_source_sha256": digest(Path(__file__)),
            "training_source_sha256": {name: digest(ROOT / name) for name in
                ("agents/mechanism_model.py", "analysis/mechanism_data.py", "analysis/train_mechanism.py")},
            "primary_contrasts": list(PRIMARY), "source_splits": list(SPLITS),
            "negative_threshold": -.01, "positive_threshold": .01,
            "trigger": "Any primary paired contrast has min(seed effect) <= -0.01 and max(seed effect) >= 0.01 in validation or confirmation",
            "matrix": matrix(base), "reused_seeds": [0, 1, 2], "additional_seeds": [3, 4],
            "all_conditions_rule": "Run all six conditions for both added seeds; never select the favorable seed, contrast or checkpoint",
            "unchanged_inputs": "Same complete training/validation/confirmation cohorts, donors, normalizer fitting policy, architecture, optimizer and last-checkpoint budget",
            "confirmation_access": "Verify all twelve full-budget added artifacts and normalizer parity with the original core before reading their confirmation outcomes",
            "analysis_rule": "Report every original and added paired seed and all seven metrics; session intervals condition on the five fitted seeds, not population training uncertainty",
            "scope": "Execution design bound after core fitting and earlier historical/smoke evidence, before complete controlled comparisons; not an original-plan numeric preregistration",
            "claim_boundary": "Outcome-gated sensitivity extension, not fresh independent confirmation, novelty, deployment safety or significance-seeking stopping",
            "if_unavailable": "Preserve failure evidence and withhold a key claim relying on the discordant contrast"}


def freeze(path, base_path, policy_path, results):
    wanted = signature(base_path, policy_path)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in wanted.items()):
            raise ValueError("Five-seed registration changed after freezing")
        if previous.get("complete_core_reports_present_at_binding") != []:
            raise ValueError("Invalid prospective binding disclosure")
        return previous
    reports = [results / "analysis" / split / "report.json"
               for split in (*SPLITS, "navtest")]
    if any(report.exists() for report in reports):
        raise ValueError("Cannot freeze a pre-comparison design after complete reports exist")
    wanted.update(captured_utc=datetime.now(timezone.utc).isoformat(),
                  complete_core_reports_present_at_binding=[])
    # Exclusive creation: concurrent attempts cannot replace an earlier binding.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(wanted, indent=2) + "\n")
    return wanted


def verified_effects(report_path, scores, core, protocol, split):
    """Recompute gate-driving paired effects from all eighteen actual CSVs."""
    if split not in SPLITS or core.get("passed") is not True or core["n_runs"] != 18:
        raise ValueError("Verified eighteen-model core and registered split required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    settings_path = scores / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
    expected = {row["tag"]: row for row in core["runs"]}
    if (len(core["runs"]) != 18 or set(expected) != {
            f"{config}_f1_steps_s{seed}" for seed in (0, 1, 2) for config in CONFIGS}):
        raise ValueError("Exact six-condition/three-seed core identities required")
    count = protocol["split_counts"][split]["scenes"]
    if (report["split"] != split or report["protocol_sha256"] != core["protocol_sha256"]
            or report["n_scenes"] != count or set(report["provenance"]) != set(expected)
            or complete.get("eligible_for_paper") is not True or complete["split"] != split
            or complete.get("final_invalid_scores", complete.get("failures", 0)) != 0
            or settings["limit"] is not None
            or complete["settings_sha256"] != digest(settings_path)
            or report["score_settings_sha256"] != digest(settings_path)
            or set(complete["runs"]) != set(expected) or set(settings["runs"]) != set(expected)):
        raise ValueError("Incomplete/smoke/mismatched core score evidence")
    means, reference = {}, None
    for tag, trained in expected.items():
        meta = settings["runs"][tag]["meta"]
        provenance = report["provenance"][tag]
        if (meta.get("eligible_for_paper") is not True or meta["run"] != tag
                or meta["split"] != split or meta["scenes"] != count or meta["fallbacks"] != 0
                or meta["protocol_sha256"] != core["protocol_sha256"]
                or {key.replace("\\", "/"): value for key, value in meta["source_sha256"].items()} != core["source_sha256"]):
            raise ValueError("Core prediction provenance mismatch")
        for key, original_key in (("checkpoint_sha256", "checkpoint_sha256"),
                                  ("normalization_sha256", "normalization_sha256"),
                                  ("training_settings_sha256", "settings_sha256"),
                                  ("donors_sha256", "donors_sha256")):
            wanted = core[original_key] if key == "donors_sha256" else trained[original_key]
            if meta[key] != wanted or provenance[key] != wanted:
                raise ValueError("Core model identity mismatch")
        csv_path = scores / (tag + ".csv.gz")
        csv_hash = digest(csv_path)
        if (csv_hash != provenance["csv_sha256"]
                or csv_hash != complete["runs"][tag]["csv_sha256"]):
            raise ValueError("Core score CSV changed")
        tokens, values = {}, []
        with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                score = float(row["score"])
                if (row["token"] in tokens or row["valid"].lower() not in ("true", "1")
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("Invalid/duplicate core score")
                tokens[row["token"]] = row["log_name"]
                values.append(score)
        if len(tokens) != count or complete["runs"][tag]["n"] != count or not tokens:
            raise ValueError("Incomplete core token coverage")
        if reference is not None and tokens != reference:
            raise ValueError("Unpaired core tokens/logs")
        reference = tokens
        means[tag] = math.fsum(values) / count
    if {label for label, row in report["summary"]["contrasts"].items() if row["primary"]} != set(PRIMARY):
        raise ValueError("Registered primary contrasts changed")
    for label in PRIMARY:
        left, right = label.split("-")
        row = report["summary"]["contrasts"][label]
        effects = {str(seed): means[f"{left}_f1_steps_s{seed}"] - means[f"{right}_f1_steps_s{seed}"]
                   for seed in (0, 1, 2)}
        if (set(row["seed_effects"]) != set(effects)
                or not math.isfinite(row["metric_effects"]["score"])
                or abs(math.fsum(effects.values()) / 3 - row["metric_effects"]["score"]) > 1e-10
                or any(not math.isfinite(row["seed_effects"][seed])
                       or abs(value - row["seed_effects"][seed]) > 1e-10 for seed, value in effects.items())):
            raise ValueError("Gate-driving effects differ from actual paired CSVs")
    return report


def verify_gate(registration_path, base_path, policy_path, results, core):
    if not registration_path.exists():
        raise ValueError("Five-seed registration must already be frozen")
    registration = freeze(registration_path, base_path, policy_path, results)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if (core["protocol_sha256"] != registration["base_protocol_sha256"]
            or core["donors_sha256"] != registration["base_donors_sha256"]
            or core["source_sha256"] != registration["training_source_sha256"]):
        raise ValueError("Core differs from registered base evidence")
    reports = [verified_effects(results / "analysis" / split / "report.json",
                               results / "scores" / split, core, base, split) for split in SPLITS]
    original = json.loads(policy_path.read_text(encoding="utf-8"))
    expected = decide(*reports, original)
    expected.update(policy_sha256=digest(policy_path),
                    validation_report_sha256=digest(results / "analysis/validation/report.json"),
                    confirmation_report_sha256=digest(results / "analysis/confirmation/report.json"))
    if json.loads((results / "gate_confirmation.json").read_text(encoding="utf-8")) != expected:
        raise ValueError("Final core gate is not the hash-bound paired decision")
    triggers = expected["seed_discordance_requiring_followup_or_withheld_claim"]
    return {"required": bool(triggers), "trigger_contrasts": triggers,
            "registration_sha256": digest(registration_path),
            "core_gate_sha256": digest(results / "gate_confirmation.json"),
            "additional_jobs": registration["matrix"] if triggers else [],
            "actual_added_models_complete": False,
            "claim_boundary": registration["claim_boundary"]}


def extension_protocol(base, registration):
    """In-memory recipe only; activation/materialization is a separate step."""
    if (registration["matrix"] != matrix(base) or registration["additional_seeds"] != [3, 4]
            or registration["reused_seeds"] != [0, 1, 2]):
        raise ValueError("Registered extension matrix changed")
    extended = copy.deepcopy(base)
    extended["training"]["seeds"] = [0, 1, 2, 3, 4]
    extended["seed_extension"] = {"kind": registration["kind"],
                                  "base_protocol_sha256": registration["base_protocol_sha256"],
                                  "new_seeds": [3, 4], "reused_core_seeds": [0, 1, 2],
                                  "disclosure": registration["claim_boundary"]}
    return extended


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--freeze-only", action="store_true")
    args = ap.parse_args()
    if not args.freeze_only:
        raise ValueError("This component registers only; no training/evaluation runner is implemented here")
    results = ROOT / "results/mechanism_v2"
    registration = freeze(results / "seed_protocol/registration.json",
                          results / "protocol/protocol.json",
                          results / "protocol/followup_policy.json", results)
    print("FIVE-SEED DESIGN REGISTERED ONLY", registration["captured_utc"], flush=True)


if __name__ == "__main__":
    main()
