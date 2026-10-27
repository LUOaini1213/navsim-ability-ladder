"""Verify frozen same-city mappings without changing the pre-outcome generator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mechanism_data import session_id, split_indices
from verify_mechanism_core import ROOT, digest


def audit_mapping(data, indices, donors, base_indices):
    indices, donors, base_indices = map(lambda row: np.asarray(row, int), (indices, donors, base_indices))
    if (indices.ndim != 1 or donors.shape != indices.shape or not len(indices)
        or len(set(indices)) != len(indices) or not np.isin(indices, base_indices).all()
        or not np.isin(donors, indices).all()):
        raise ValueError("Invalid/incomplete city partition mapping")
    cities = data["map_name"]
    sessions = np.array([session_id(log) for log in data["log_name"]])
    if not np.all(cities[indices] == cities[donors]):
        raise ValueError("Same-city invariant violated")
    if np.any(sessions[indices] == sessions[donors]):
        raise ValueError("Different-session invariant violated")
    supported_cities = {str(city) for city in set(cities[base_indices])
                        if len(set(sessions[base_indices][cities[base_indices] == city])) >= 2}
    expected = np.array([index for index in base_indices if str(cities[index]) in supported_cities])
    if not np.array_equal(indices, expected):
        raise ValueError("City coverage differs from the input-only supported cohort")
    return {"n_scenes": len(indices), "n_original_sessions": len(set(sessions[indices])),
            "same_city_fraction": 1., "different_session_fraction": 1.,
            "omitted_scenes": len(base_indices) - len(indices),
            "eligible_token_order_sha256": hashlib.sha256("\n".join(map(str, data["token"][indices])).encode("utf-8")).hexdigest()}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--city", type=Path, default=ROOT / "results/mechanism_v2/city_protocol")
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2/city_inputs.json")
    args = ap.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    protocol_path = args.city / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    complete = json.loads((args.city / "completed.json").read_text(encoding="utf-8"))
    design = json.loads((args.city / "design.json").read_text(encoding="utf-8"))
    generator_hash = digest(ROOT / "analysis/prepare_city_control.py")
    if (complete["protocol_sha256"] != digest(protocol_path) or complete["donors_sha256"] != digest(args.city / "donors.npz")
        or complete["design_sha256"] != digest(args.city / "design.json")
        or complete["generator_sha256"] != generator_hash or design["generator_sha256"] != generator_hash
        or design["base_protocol_sha256"] != digest(args.base) or design["outcomes_used_for_design"]
        or complete["outcomes_read"] or design["complete_core_reports_present_at_binding"]):
        raise ValueError("City preparation/design provenance mismatch")
    for name, checksum in design["input_helpers_sha256"].items():
        if digest(ROOT / "analysis" / name) != checksum:
            raise ValueError("Bound city input helper changed")
    if protocol["sources"] != base["sources"] or protocol["fractions"] != base["fractions"] or protocol["training"] != base["training"]:
        raise ValueError("City source/training budget/subsets changed")
    results = {}
    with np.load(args.city / "donors.npz", allow_pickle=False) as mappings:
        for source_name in ("navtrain", "navtest"):
            source = base["sources"][source_name]
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("Frozen city feature source changed")
            with np.load(source["path"], allow_pickle=False) as archive:
                data = {key: archive[key] for key in ("token", "log_name", "map_name")}
            partitions = ("train", "validation", "confirmation") if source_name == "navtrain" else ("navtest",)
            for split in partitions:
                indices, donors = mappings[split + "_indices"], mappings[split + "_donors"]
                row = audit_mapping(data, indices, donors, split_indices(data, base, split))
                if (not np.array_equal(indices, split_indices(data, protocol, split))
                    or not np.array_equal(mappings[split + "_tokens"], data["token"][indices])):
                    raise ValueError("City protocol/token alignment mismatch")
                coverage = complete["coverage"][split]
                if (row["n_scenes"] != coverage["eligible_scenes"] or row["n_original_sessions"] != coverage["eligible_sessions"]
                    or row["eligible_token_order_sha256"] != coverage["eligible_token_order_sha256"]):
                    raise ValueError("City coverage receipt differs from actual inputs")
                if split == "train" and row["omitted_scenes"]:
                    raise ValueError("Base real anchors require the unchanged training cohort")
                results[split] = row
    report = {"passed": True, "city_protocol_sha256": digest(protocol_path), "donors_sha256": complete["donors_sha256"],
              "design_sha256": complete["design_sha256"], "audit_source_sha256": digest(Path(__file__)),
              "partitions": results, "outcomes_used": False,
              "scope": "Actual frozen input mappings only; six auxiliary models and matched-coverage PDMS remain pending",
              "limitation": "City and different-session constraints are verified, but route geometry and actor compatibility are not; not a physically valid intervention",
              "inherited_diagnostic_text_note": "diagnostics.json reuses a base-helper limitation string saying city is unconstrained; this is inapplicable to these separately verified strict-city mappings. Frozen generator/helper files and receipts were not rewritten."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CITY INPUT AUDIT PASS", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
