"""Apply frozen execution/claim gates to fully audited controlled score reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_mechanism_core import ROOT, digest


def evidence(row, threshold):
    mean = row["metric_effects"]["score"]
    effects = row["seed_effects"]
    if set(effects) != {"0", "1", "2"}:
        raise ValueError("Exactly three paired core seeds required")
    direction = 1 if mean > 0 else -1 if mean < 0 else 0
    interval = row["multiplicity_interval"]
    excludes = (direction > 0 and interval["low"] is not None and interval["low"] > 0
                or direction < 0 and interval["high"] is not None and interval["high"] < 0)
    consistent = sum(value*direction > 0 for value in effects.values())
    return {"mean": mean, "direction": direction, "practically_nonzero": abs(mean) >= threshold,
            "same_direction_seeds": consistent, "adjusted_interval_excludes_zero": bool(excludes),
            "eligible_for_cross_protocol_gate": abs(mean) >= threshold and consistent >= 2 and bool(excludes)}


def decide(validation, confirmation, policy):
    if validation["split"] != "validation" or validation["protocol_sha256"] != policy["protocol_sha256"]:
        raise ValueError("Validation protocol/split mismatch")
    if len(validation["provenance"]) != 18:
        raise ValueError("Complete 18-run audited report required")
    if confirmation is not None:
        if (confirmation["split"] != "confirmation" or confirmation["protocol_sha256"] != policy["protocol_sha256"]
            or len(confirmation["provenance"]) != 18):
            raise ValueError("Confirmation protocol/split mismatch")
        if {run: row["checkpoint_sha256"] for run, row in validation["provenance"].items()} != {
                run: row["checkpoint_sha256"] for run, row in confirmation["provenance"].items()}:
            raise ValueError("Validation and confirmation must use identical fitted models")
    contrasts = [label for label, row in validation["summary"]["contrasts"].items() if row["primary"]]
    if set(contrasts) != {"M1-M0", "M4-M1", "M4-M5"}:
        raise ValueError("Registered primary contrasts missing/changed")
    threshold = policy["minimum_absolute_pdms_effect"]
    negative = policy["seed_discordance"]["negative_threshold"]
    positive = policy["seed_discordance"]["positive_threshold"]
    records, curve_reasons, cross_reasons, discordant = {}, [], [], []
    for label in contrasts:
        row = validation["summary"]["contrasts"][label]
        va = evidence(row, threshold)
        cf = evidence(confirmation["summary"]["contrasts"][label], threshold) if confirmation is not None else None
        if va["practically_nonzero"]:
            curve_reasons.append(label)
        for report in (validation, confirmation):
            if report is None:
                continue
            effects = report["summary"]["contrasts"][label]["seed_effects"].values()
            if min(effects) <= negative and max(effects) >= positive:
                discordant.append(report["split"]+":"+label)
        if (cf is not None and va["eligible_for_cross_protocol_gate"] and cf["eligible_for_cross_protocol_gate"]
            and va["direction"] == cf["direction"]):
            cross_reasons.append(label)
        records[label] = {"validation": va, "confirmation": cf}
    return {"learning_curve_required": bool(curve_reasons), "learning_curve_trigger_contrasts": curve_reasons,
            "cross_protocol_status": "pending_confirmation" if confirmation is None else "attempt_feasibility" if cross_reasons else "not_triggered",
            "cross_protocol_trigger_contrasts": cross_reasons,
            "seed_discordance_requiring_followup_or_withheld_claim": sorted(set(discordant)),
            "evidence": records, "disclosure": "Gates allocate experiments and constrain claims; they are not a novelty, significance or safety certificate"}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--confirmation", type=Path)
    ap.add_argument("--policy", type=Path, default=ROOT / "results/mechanism_v2/protocol/followup_policy.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    confirmation = json.loads(args.confirmation.read_text(encoding="utf-8")) if args.confirmation else None
    result = decide(validation, confirmation, policy)
    result.update(policy_sha256=digest(args.policy), validation_report_sha256=digest(args.validation),
                  confirmation_report_sha256=digest(args.confirmation) if args.confirmation else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
