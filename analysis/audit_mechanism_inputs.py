"""Audit donor input distributions and the historical feature-eligible cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mechanism_data import digest, session_id, split_indices
from runtime import ROOT, configure


def donor_diagnostics(data, indices, donors):
    indices, donors = np.asarray(indices), np.asarray(donors)
    if len(indices) != len(donors) or not np.isin(donors, indices).all():
        raise ValueError("Donors outside their partition")
    source_sessions = np.array([session_id(log) for log in data["log_name"][indices]])
    donor_sessions = np.array([session_id(log) for log in data["log_name"][donors]])
    if np.any(source_sessions == donor_sessions):
        raise ValueError("Same-session donor")
    counts = data["agents"][:, :, 10].sum(1)
    speed_difference = np.abs(data["speed"][indices]-data["speed"][donors])
    return {"n": len(indices), "different_session_fraction": 1.,
            "original_sessions_by_city": {str(city): len(set(source_sessions[data["map_name"][indices] == city]))
                                          for city in sorted(set(data["map_name"][indices]))},
            "same_city_fraction": float((data["map_name"][indices] == data["map_name"][donors]).mean()),
            "same_command_fraction": float((data["cmd"][indices] == data["cmd"][donors]).mean()),
            "same_speed_bin_fraction": float((np.digitize(data["speed"][indices], [2, 5, 10, 20])
                                              == np.digitize(data["speed"][donors], [2, 5, 10, 20])).mean()),
            "same_count_bin_fraction": float((np.digitize(counts[indices], [1, 3, 6])
                                              == np.digitize(counts[donors], [1, 3, 6])).mean()),
            "exact_object_count_fraction": float((counts[indices] == counts[donors]).mean()),
            "median_absolute_speed_difference_mps": float(np.median(speed_difference)),
            "p95_absolute_speed_difference_mps": float(np.quantile(speed_difference, .95)),
            "empty_to_nonempty_count": int(((counts[indices] == 0) & (counts[donors] > 0)).sum()),
            "nonempty_to_empty_count": int(((counts[indices] > 0) & (counts[donors] == 0)).sum()),
            "limitation": "Matching does not constrain city, route geometry, traffic handedness or actor interactions; controls are distribution-matched only on the declared bins, not physically valid counterfactuals"}


def cohort(data, split, workspace, ledger):
    from build_features import scene_filter_for
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig

    feature_tokens = set(map(str, data["token"]))
    sf = scene_filter_for(split, None)
    listed_logs = sorted(sf.log_names)
    expected_tokens = set(sf.tokens) if sf.tokens is not None else None
    raw_path = workspace / "dataset/navsim_logs" / ("test" if split == "navtest" else "trainval")
    present = {path.stem for path in raw_path.glob("*.pkl")}
    missing_logs = sorted(set(listed_logs)-present)
    selected = set()
    exclusions = 0
    for number, log in enumerate(listed_logs):
        if log not in present:
            continue
        # Compose the large official filter once, then vary only the log. No
        # Scene/map construction is needed to enumerate its selected tokens.
        sf.log_names = [log]
        loader = SceneLoader(raw_path, None, sf, SensorConfig.build_no_sensors())
        raw_tokens = set(loader.tokens)
        if selected & raw_tokens:
            raise ValueError("Duplicate tokens across raw logs")
        selected.update(raw_tokens)
        excluded = sorted(raw_tokens-feature_tokens)
        exclusions += len(excluded)
        for token in excluded:
            ledger.write(json.dumps({"split": split, "log_name": log, "token": token,
                                      "reason": "not_in_historical_feature_archive; original extraction reason was not recorded"})+"\n")
        del loader
        if (number+1) % 50 == 0:
            ledger.flush()
            print("cohort", split, number+1, "/", len(listed_logs), "logs", flush=True)
    extra = feature_tokens-selected
    if extra:
        raise ValueError(f"{split}: {len(extra)} feature tokens outside the available official raw cohort")
    return {"listed_logs": len(listed_logs), "missing_logs": missing_logs,
            "official_token_filter_n": len(expected_tokens) if expected_tokens is not None else None,
            "available_selected_raw_scenes": len(selected), "feature_eligible_scenes": len(feature_tokens),
            "selected_raw_scenes_without_features": exclusions,
            "official_filtered_tokens_not_in_available_selected_raw_cohort": len(expected_tokens-selected) if expected_tokens is not None else None,
            "scope": "Counts on available logs under the unchanged official scene filter; excluded extraction reasons cannot be reconstructed from the historical archive"}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2/input_controls.json")
    ap.add_argument("--cohort", action="store_true", help="Enumerate both complete raw split cohorts without constructing maps")
    args = ap.parse_args()
    workspace = configure()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    donor_path = args.protocol.parent / "donors.npz"
    report = {"protocol_sha256": digest(args.protocol), "donors_sha256": digest(donor_path), "donors": {}, "cohorts": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(donor_path) as donor_archive, open(args.out.with_suffix(".excluded.jsonl"), "w", encoding="utf-8") as ledger:
        for split in ("navtrain", "navtest"):
            source = protocol["sources"][split]
            if digest(source["path"]) != source["sha256"]:
                raise ValueError("Changed features")
            with np.load(source["path"], allow_pickle=False) as archive:
                data = {key: archive[key] for key in ("token", "log_name", "map_name", "cmd", "speed", "agents")}
            for partition in (("train", "validation", "confirmation") if split == "navtrain" else ("navtest",)):
                indices = split_indices(data, protocol, partition)
                if not np.array_equal(indices, donor_archive[partition+"_indices"]):
                    raise ValueError("Donor index mismatch")
                report["donors"][partition] = donor_diagnostics(data, indices, donor_archive[partition+"_donors"])
            if args.cohort:
                report["cohorts"][split] = cohort(data, split, workspace, ledger)
            del data
    report["passed"] = True
    report["cohort_enumerated"] = args.cohort
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("INPUT CONTROL AUDIT PASS", "raw cohort="+str(args.cohort), flush=True)


if __name__ == "__main__":
    main()
