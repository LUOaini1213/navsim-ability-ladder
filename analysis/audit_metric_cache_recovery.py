"""Exercise three corrupt-cache recoveries in a newly created isolated smoke root."""
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from mechanism_scoring import make_engine, score_one
from runtime import ROOT, configure
from verify_mechanism_core import digest, verify


def owned_target(path, trial):
    path, trial = path.resolve(), trial.resolve()
    if not path.is_relative_to(trial) or path.name != "metric_cache.pkl":
        raise ValueError("Refusing corruption outside the newly created isolated trial")
    return path


def corrupt_bytes(kind, original):
    if kind == "empty":
        return b""
    if kind == "truncated_lzma":
        return original[:max(1, len(original) // 3)]
    if kind == "invalid_pickle":
        return lzma.compress(b"not a pickle payload", preset=0)
    raise ValueError("Unregistered corruption kind")


def semantic_signature(cache):
    return {"ego_pose": list(cache.ego_state.rear_axle.serialize()),
            "ego_velocity": list(cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.array),
            "reference_poses": [list(state.rear_axle.serialize()) for state in cache.trajectory.get_sampled_trajectory()],
            "route_lane_ids": list(cache.route_lane_ids),
            "object_tokens": sorted(map(str, cache.observation.unique_objects))}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2/cache_recovery.json")
    args = ap.parse_args()
    workspace = configure(args.workspace)
    if args.out.exists():
        raise ValueError("Preserve the existing recovery audit; use a new report path")
    core = verify(workspace / "exp/mechanism_v2/models", args.protocol)
    prediction_path = workspace / "exp/mechanism_v2/predictions/validation/M0_f1_steps_s0.npz"
    receipt = json.loads(prediction_path.with_suffix(".completed.json").read_text(encoding="utf-8"))
    if receipt["predictions_sha256"] != digest(prediction_path):
        raise ValueError("Recovery-audit predictions changed")
    with np.load(prediction_path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        poses = dict(zip(map(str, archive["token"]), archive["poses"]))
    anchor = next(row for row in core["runs"] if row["tag"] == "M0_f1_steps_s0")
    if (meta != receipt["meta"] or not meta["eligible_for_paper"] or meta["split"] != "validation"
        or meta["protocol_sha256"] != core["protocol_sha256"] or meta["checkpoint_sha256"] != anchor["checkpoint_sha256"]):
        raise ValueError("Recovery-audit anchor provenance mismatch")
    parent = workspace / "exp/mechanism_v2/cache_recovery_trials"
    parent.mkdir(parents=True, exist_ok=True)
    trial = Path(tempfile.mkdtemp(prefix="isolated-", dir=str(parent))).resolve()
    report = {"passed": False, "trial_root": str(trial), "protocol_sha256": digest(args.protocol),
              "cache_builder_sha256": digest(ROOT / "analysis/cache_mechanism.py"), "audit_source_sha256": digest(Path(__file__)),
              "scientific_result_eligible": False, "attempts": [],
              "scope": "Two actual raw validation scenes, three intentional isolated-file corruptions; not full cache coverage or a model performance result"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    def build(label):
        command = [sys.executable, "-u", str(ROOT / "analysis/cache_mechanism.py"), "--split", "validation",
                   "--limit", "2", "--protocol", str(args.protocol), "--out", str(trial)]
        path = trial / (label + ".log")
        with open(path, "w", encoding="utf-8") as stream:
            code = subprocess.call(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
        if code != 0:
            raise RuntimeError("Isolated cache attempt failed: " + str(path))
    try:
        build("baseline")
        files = sorted(trial.glob("*/*/*/metric_cache.pkl"))
        if len(files) != 2:
            raise ValueError("Exactly two newly created cache files required")
        target = owned_target(files[0], trial)
        untouched = files[1]
        untouched_hash = digest(untouched)
        original = target.read_bytes()
        with lzma.open(target, "rb") as stream:
            baseline = pickle.load(stream)
        signature = semantic_signature(baseline)
        token = target.parent.name
        simulator, scorer = make_engine(workspace)
        baseline_score = score_one(baseline, poses[token], simulator, scorer)
        report.update(token=token, baseline_sha256=digest(target), untouched_sha256=untouched_hash)
        save()
        for kind in ("empty", "truncated_lzma", "invalid_pickle"):
            owned_target(target, trial).write_bytes(corrupt_bytes(kind, original))
            corrupted_hash = digest(target)
            # Prove the deliberately broken payload actually fails to load.
            try:
                with lzma.open(target, "rb") as stream:
                    pickle.load(stream)
            except (EOFError, lzma.LZMAError, pickle.UnpicklingError) as error:
                detected = type(error).__name__
            else:
                raise ValueError("Fault payload unexpectedly remained valid")
            build(kind + "_recovery")
            with lzma.open(target, "rb") as stream:
                restored = pickle.load(stream)
            fresh = score_one(restored, poses[token], simulator, scorer)
            error = max(abs(fresh[key] - baseline_score[key]) for key in baseline_score)
            ledger = [json.loads(line) for line in (trial / "validation_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line][-2:]
            if (semantic_signature(restored) != signature or error > 1e-8
                or digest(untouched) != untouched_hash
                or not any(row.get("token") == token and row.get("reason") == "computed" and row["valid"] for row in ledger)):
                raise ValueError("Recovery semantics/score/untouched-file or ledger mismatch")
            report["attempts"].append({"fault": kind, "detected_exception": detected, "corrupted_sha256": corrupted_hash,
                                       "recovered_sha256": digest(target), "baseline_byte_equal": target.read_bytes() == original,
                                       "semantic_signature_equal": True, "maximum_official_submetric_error": error,
                                       "untouched_file_preserved": True, "ledger_records_recomputation": True})
            save()
        report.update(passed=True, official_submetric_names=list(baseline_score),
                      historical_and_full_cache_roots_modified=False,
                      boundary="No historical/full-cache path was passed for mutation; final full historical-artifact preservation audit remains separate")
        save()
        print("ISOLATED ACTUAL CACHE RECOVERY PASS", len(report["attempts"]), "faults", trial, flush=True)
    except BaseException as error:
        report["exception"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
