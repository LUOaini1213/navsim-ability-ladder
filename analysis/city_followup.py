"""Freeze a prospective city execution gate, then verify it against actual CSVs."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from run_mechanism_curves import verified_gate
from verify_mechanism_core import ROOT, digest


def freeze(path, base, city, results):
    existing = json.loads((base.parent / "followup_policy.json").read_text(encoding="utf-8"))
    signature = {"kind": "prospective-same-city-execution-gate-v1", "base_protocol_sha256": digest(base),
                 "city_protocol_sha256": digest(city), "original_followup_policy_sha256": digest(base.parent / "followup_policy.json"),
                 "gate_source_sha256": digest(Path(__file__)), "source_split": "validation",
                 "candidate_contrasts": ["M1-M2", "M4-M5"], "minimum_absolute_pdms_effect": existing["minimum_absolute_pdms_effect"],
                 "trigger": "At least one candidate absolute validation mean reaches the existing practical threshold, including negative effects",
                 "matrix": "Six M2/M5 same-city auxiliary models, three seeds, same full-data update budget; matched real anchors reused",
                 "scope": "Additional execution rule fixed after core training and earlier historical/smoke evidence, before complete controlled comparisons; not an original-plan numeric preregistration claim",
                 "no_trigger": "Withhold a clean current-scene-alignment interpretation; base donor contrasts remain compound diagnostics",
                 "claim_boundary": "Execution alone is not significance, novelty, causality, or evidence of physical interaction use"}
    if signature["minimum_absolute_pdms_effect"] != .01:
        raise ValueError("Unexpected original practical threshold")
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if any(prior.get(key) != value for key, value in signature.items()):
            raise ValueError("City execution gate changed after freezing")
        return prior
    if any((results / "analysis" / split / "report.json").exists() for split in ("validation", "confirmation", "navtest")):
        raise ValueError("Cannot label a new city execution rule pre-comparison after results exist")
    signature.update(captured_utc=datetime.now(timezone.utc).isoformat(), complete_core_reports_present_at_binding=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return signature


def decide(effects, threshold=.01):
    if set(effects) != {"M1-M2", "M4-M5"} or threshold != .01 or not all(math.isfinite(value) and abs(value) <= 1 for value in effects.values()):
        raise ValueError("Invalid frozen city gate effects/threshold")
    triggers = sorted(name for name, effect in effects.items() if abs(effect) >= threshold)
    return {"required": bool(triggers), "trigger_contrasts": triggers, "validation_effects": effects,
            "claim_boundary": "Same-city training/evaluation still needed before any clean-alignment interpretation; no physical-counterfactual or causal-safety claim",
            "clean_alignment_claim_withheld_without_control": True}


def verify_gate(policy_path, base_path, city_path, results, core):
    if not policy_path.exists():
        raise ValueError("Prospective city execution policy must already be frozen")
    policy = freeze(policy_path, base_path, city_path, results)
    report_path = results / "analysis/validation/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["split"] != "validation":
        raise ValueError("Only validation may trigger the city execution gate")
    scores = results / "scores/validation"
    # This validates all 18 actual score CSVs, model identities and primary gate.
    verified_gate(report_path, scores, results / "gate_validation.json", base_path.parent / "followup_policy.json", core)
    means = {}
    for config in ("M1", "M2", "M4", "M5"):
        for seed in (0, 1, 2):
            tag = f"{config}_f1_steps_s{seed}"
            with gzip.open(scores / (tag + ".csv.gz"), "rt", encoding="utf-8", newline="") as stream:
                values = [float(row["score"]) for row in csv.DictReader(stream)]
            means[tag] = sum(values) / len(values)
    effects = {}
    for label in policy["candidate_contrasts"]:
        left, right = label.split("-")
        per_seed = {str(seed): means[f"{left}_f1_steps_s{seed}"] - means[f"{right}_f1_steps_s{seed}"] for seed in (0, 1, 2)}
        mean = sum(per_seed.values()) / 3
        row = report["summary"]["contrasts"][label]
        if abs(mean - row["metric_effects"]["score"]) > 1e-10 or any(abs(value - row["seed_effects"][seed]) > 1e-10 for seed, value in per_seed.items()):
            raise ValueError("City gate effect differs from actual paired score CSVs")
        effects[label] = mean
    return dict(decide(effects), execution_policy_sha256=digest(policy_path), validation_report_sha256=digest(report_path))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--freeze-only", action="store_true")
    args = ap.parse_args()
    if not args.freeze_only:
        raise ValueError("Use the city runner for actual verified decisions")
    results = ROOT / "results/mechanism_v2"
    policy = freeze(results / "city_protocol/execution_policy.json", results / "protocol/protocol.json",
                    results / "city_protocol/protocol.json", results)
    print("CITY EXECUTION RULE FROZEN ONLY", policy["captured_utc"], flush=True)


if __name__ == "__main__":
    main()
