"""Freeze a same-city cross-session donor extension without reading outcome arrays."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from run_mechanism_core import available_mb
from verify_mechanism_core import ROOT, digest


def supported_indices(data, indices):
    import numpy as np
    from mechanism_data import session_id
    by_city = defaultdict(set)
    for index in indices:
        by_city[str(data["map_name"][index])].add(session_id(data["log_name"][index]))
    supported = {city for city, sessions in by_city.items() if len(sessions) >= 2}
    keep = np.array([int(index) for index in indices if str(data["map_name"][index]) in supported], np.int64)
    omitted = {city: {"sessions": sorted(sessions),
                      "n_scenes": sum(str(data["map_name"][index]) == city for index in indices),
                      "reason": "No same-city different-session donor in this partition"}
               for city, sessions in sorted(by_city.items()) if city not in supported}
    return keep, omitted


def city_donors(data, indices, seed):
    """Never relax city/session independence; relax count, speed, then command."""
    import numpy as np
    from mechanism_data import session_id
    indices = np.asarray(indices, np.int64)
    if not len(indices):
        raise ValueError("Empty eligible partition")
    city = data["map_name"][indices]
    sessions = np.array([session_id(log) for log in data["log_name"][indices]])
    command = data["cmd"][indices]
    speed = np.digitize(data["speed"][indices], [2., 5., 10., 20.])
    counts = np.digitize(data["agents"][indices, :, 10].sum(1), [1, 3, 6])
    keys = [[(str(c), int(m), int(v), int(n)) for c, m, v, n in zip(city, command, speed, counts)],
            [(str(c), int(m), int(v)) for c, m, v in zip(city, command, speed)],
            [(str(c), int(m)) for c, m in zip(city, command)], [(str(c),) for c in city]]
    tables = []
    for level_keys in keys:
        table = defaultdict(lambda: defaultdict(list))
        for position, key in enumerate(level_keys):
            table[key][str(sessions[position])].append(int(indices[position]))
        tables.append(table)
    rng = np.random.RandomState(seed)
    donors, levels = [], []
    for position in range(len(indices)):
        for level in range(4):
            pool = tables[level][keys[level][position]]
            other_sessions = sorted(session for session in pool if session != sessions[position])
            if not other_sessions:
                continue
            selected = other_sessions[int(rng.randint(len(other_sessions)))]
            candidates = pool[selected]
            donors.append(candidates[int(rng.randint(len(candidates)))])
            levels.append(level)
            break
        else:
            raise ValueError("No same-city cross-session donor; do not silently relax the restriction")
    donors = np.array(donors, np.int64)
    if np.any(data["map_name"][indices] != data["map_name"][donors]):
        raise AssertionError("City independence invariant violated")
    if any(session_id(data["log_name"][a]) == session_id(data["log_name"][b]) for a, b in zip(indices, donors)):
        raise AssertionError("Same-session donor")
    return donors, {str(level): int(count) for level, count in sorted(Counter(levels).items())}


def frozen_write(path, value):
    encoded = json.dumps(value, indent=2)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Frozen city-extension artifact differs: "+str(path))
        return
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def freeze_design(out, base, results_root):
    """Bind the input-driven generator before any complete core score report exists."""
    path = out / "design.json"
    signature = {"kind": "same-city-cross-session-negative-control-v1", "base_protocol_sha256": digest(base),
                 "generator_sha256": digest(Path(__file__)),
                 "input_helpers_sha256": {name: digest(ROOT / "analysis" / name) for name in ("mechanism_data.py", "audit_mechanism_inputs.py")},
                 "outcomes_used_for_design": False,
                 "known_prior_evidence": "Historical ladder outcomes and an ineligible smoke score were already known",
                 "city_and_session_constraints": "Same city and different original session; never relaxed",
                 "coverage": "Exclude unsupported partition-city cells; unchanged training cohort required for real-anchor reuse",
                 "training": "Three seeds; full-data fixed-update M2/M5 auxiliary runs; reuse matched M1/M4 anchors",
                 "selection": "Input-only matching; same frozen count/speed/command fallback; no future arrays",
                 "scope": "Pre-outcome auxiliary design binding, not completed donor preparation, training or evaluation"}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in signature.items()):
            raise ValueError("City design/source changed after binding; use an explicitly new exploratory extension")
        return path
    reports = [results_root / "analysis" / split / "report.json" for split in ("validation", "confirmation", "navtest")]
    if any(report.exists() for report in reports):
        raise ValueError("Complete core comparison already exists; cannot falsely bind a pre-outcome city design")
    out.mkdir(parents=True, exist_ok=True)
    frozen_write(path, dict(signature, captured_utc=datetime.now(timezone.utc).isoformat(),
                           complete_core_reports_present_at_binding=[]))
    return path


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/mechanism_v2/city_protocol")
    ap.add_argument("--min-free-mb", type=int, default=2200)
    ap.add_argument("--freeze-design-only", action="store_true", help="Lightweight source/policy binding; does not load feature arrays")
    args = ap.parse_args()
    design_path = freeze_design(args.out, args.base, ROOT / "results/mechanism_v2")
    if args.freeze_design_only:
        print("CITY DESIGN BOUND ONLY; donor mappings/models/results still pending", design_path, flush=True)
        return
    free = available_mb()
    if free is not None and free < args.min_free_mb:
        raise RuntimeError(f"Memory guard: {free:.0f} MB free; city donor preparation not started")
    import numpy as np
    from mechanism_data import session_id, split_indices
    from audit_mechanism_inputs import donor_diagnostics

    base = json.loads(args.base.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(base)
    args.out.mkdir(parents=True, exist_ok=True)
    donor_arrays, coverage, diagnostics = {}, {}, {}
    for source_name in ("navtrain", "navtest"):
        source = base["sources"][source_name]
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Base frozen feature source changed")
        with np.load(source["path"], allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("token", "log_name", "map_name", "cmd", "speed", "agents")}
        partitions = ("train", "validation", "confirmation") if source_name == "navtrain" else ("navtest",)
        for partition in partitions:
            original = split_indices(data, base, partition)
            indices, omitted = supported_indices(data, original)
            if partition == "train" and not np.array_equal(indices, original):
                raise ValueError("Training cohort would change; real-input anchors must be retrained under a new extension design")
            seed = base["seed"]+901+("train", "validation", "confirmation", "navtest").index(partition)
            donors, levels = city_donors(data, indices, seed)
            donor_arrays[partition+"_indices"] = indices
            donor_arrays[partition+"_donors"] = donors
            donor_arrays[partition+"_tokens"] = data["token"][indices]
            diagnostics[partition] = dict(donor_diagnostics(data, indices, donors), matching_levels=levels, seed=seed)
            sessions = sorted({session_id(log) for log in data["log_name"][indices]})
            coverage[partition] = {"base_scenes": len(original), "eligible_scenes": len(indices),
                                   "eligible_sessions": len(sessions), "omitted_cities": omitted,
                                   "eligible_token_order_sha256": __import__("hashlib").sha256("\n".join(map(str, data["token"][indices])).encode("utf-8")).hexdigest()}
            if partition != "navtest":
                protocol["split_sessions"][partition] = sessions
                protocol["split_counts"][partition] = {"scenes": len(indices), "sessions": len(sessions), "logs": len(set(data["log_name"][indices]))}
        del data
    extension = {"kind": "same-city-cross-session-negative-control-v1", "base_protocol_sha256": digest(args.base), "design_sha256": digest(design_path),
                 "freeze_scope": "Input-driven design after the core study began; before complete core PDMS comparisons. Historical ladder outcomes and an ineligible smoke score were already known.",
                 "generator_sha256": digest(Path(__file__)), "coverage": coverage,
                 "conditions": {"C1": "Reuse base M1 real concat", "C2": "New M2 configuration with same-city donors",
                                "C4": "Reuse base M4 real set", "C5": "New M5 configuration with same-city donors"},
                 "seeds": [0, 1, 2], "contrasts": ["C1-C2", "C4-C5", "C4-C5-C1+C2"],
                 "comparison": "Exactly common registered eligible tokens/logs; preserve base train cohort, last-checkpoint policy and training-only normalization",
                 "confirmation_gate": "All 18 verified base core models and all six verified full-budget auxiliary models before auxiliary confirmation inference",
                 "inference": "Use a separately audited auxiliary exporter; base exporter must not be bypassed or falsely told that six models comprise an 18-model core",
                 "multiplicity": "Three auxiliary Bonferroni intervals, explicitly conditional/exploratory follow-up; no unqualified global confirmatory family-wise claim",
                 "selection": "No top-scene selection, no future trajectories/annotations as matching variables; city/session constraints never relaxed",
                 "limitations": "Same-city matching still changes route/actor compatibility; it is not a physically valid counterfactual, causal safety effect or complete perception test"}
    protocol["extension"] = extension
    protocol["selection"] = "Auxiliary last checkpoint, same fixed full-data update budget; compatible real-input anchors reused; explicit reduced validation/confirmation coverage"
    donor_path = args.out / "donors.npz"
    if donor_path.exists():
        with np.load(donor_path, allow_pickle=False) as previous:
            if set(previous.files) != set(donor_arrays) or any(not np.array_equal(previous[key], value) for key, value in donor_arrays.items()):
                raise ValueError("Frozen same-city donor mappings differ")
    else:
        temporary = args.out / "donors.tmp.npz"
        np.savez_compressed(temporary, **donor_arrays)
        temporary.replace(donor_path)
    frozen_write(args.out / "protocol.json", protocol)
    frozen_write(args.out / "diagnostics.json", diagnostics)
    frozen_write(args.out / "completed.json", {"protocol_sha256": digest(args.out / "protocol.json"),
                 "donors_sha256": digest(donor_path), "generator_sha256": digest(Path(__file__)), "design_sha256": digest(design_path), "outcomes_read": False,
                 "status": "inputs frozen; no auxiliary training/evaluation performed", "coverage": coverage})
    print("CITY CONTROL INPUTS FROZEN; NO MODEL/PDMS CLAIM", json.dumps(coverage), flush=True)


if __name__ == "__main__":
    main()
