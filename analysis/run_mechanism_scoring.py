"""Score the complete core from finished caches without repeating raw-log audits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from guarded_pipeline import Pipeline
from run_mechanism_evaluation import acquire, upstream_binding
from verify_mechanism_core import CONFIGS, ROOT, digest, verify


def verify_cache_receipts(cache, protocol_path, resume_attempt):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    state = json.loads(resume_attempt.read_text(encoding="utf-8"))
    if state["status"] != "caches_completed":
        raise ValueError("Cache-only controller has not completed both full splits")
    for name, checksum in state["frozen_sha256"].items():
        if digest(name) != checksum:
            raise ValueError("Completed cache source/audit receipt changed: " + name)
    for split in ("validation", "confirmation"):
        jobs = [row for row in state["jobs"] if row["split"] == split]
        if not jobs or jobs[-1]["exit_code"] != 0 or jobs[-1]["memory_interruption"] is not None:
            raise ValueError("No successful final full-cache attempt: " + split)
        complete = json.loads((cache / (split + "_completed.json")).read_text(encoding="utf-8"))
        if (complete["protocol_sha256"] != digest(protocol_path) or complete["split"] != split
            or complete["limit"] is not None or complete["scenes"] != protocol["split_counts"][split]["scenes"]):
            raise ValueError("Incomplete/wrong-protocol cache receipt: " + split)
    return state


def prediction_matrix():
    return [(split, f"{config}_f1_steps_s{seed}") for split in ("validation", "confirmation", "navtest")
            for seed in (0, 1, 2) for config in CONFIGS]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--wait-for-cache", action="store_true")
    args = ap.parse_args()
    root = args.workspace / "exp/mechanism_v2"
    results = ROOT / "results/mechanism_v2"
    protocol = results / "protocol/protocol.json"
    models = root / "models"
    pipeline = Pipeline("core_scoring", results / "scoring_status.json", root / "scoring_logs", results / "run_attempts",
                        "Core predictions, official scores, paired analysis, cases, gates and figures only; conditional follow-ups, manuscript, reproducibility and final audit remain")
    pipeline.state["status"] = "waiting_for_cache_lock"
    pipeline.save()
    lock = None
    try:
        lock = acquire(root / "core.lock", wait=args.wait_for_cache,
                       on_wait=lambda: print("SCORING WAIT: cache controller owns the heavy-job lock", flush=True))
        for key, value in {"NAVSIM_WORKSPACE": args.workspace, "NAVSIM_DEVKIT_ROOT": args.workspace / "navsim",
                           "OPENSCENE_DATA_ROOT": args.workspace / "dataset", "NUPLAN_MAPS_ROOT": args.workspace / "dataset/maps",
                           "NAVSIM_EXP_ROOT": args.workspace / "exp"}.items():
            os.environ[key] = str(value)
        state = json.loads((results / "cache_resume_status.json").read_text(encoding="utf-8"))
        resume_attempt = Path(state["attempt_receipt"])
        cached = verify_cache_receipts(root / "metric_cache", protocol, resume_attempt)
        core = verify(models, protocol)
        files = [ROOT / "analysis" / name for name in
                 ("guarded_pipeline.py", "run_mechanism_scoring.py", "verify_mechanism_core.py", "predict_mechanism.py",
                  "score_mechanism.py", "mechanism_scoring.py", "analyze_mechanism.py", "mechanism_cases.py", "mechanism_gates.py", "plot_mechanism.py")]
        files.extend([resume_attempt, protocol, protocol.parent / "donors.npz", protocol.parent / "diagnostics.json",
                      protocol.parent / "followup_policy.json", results / "runtime_snapshot/runtime.json"])
        pipeline.bind(files, existing={**cached["frozen_sha256"], **upstream_binding(args.workspace, results / "runtime_snapshot/runtime.json")})
        pipeline.state.update(status="preflight_passed", audit_reuse_disclosure=cached["audit_reuse_disclosure"],
                              cache_attempt_sha256=digest(resume_attempt), n_core_models=core["n_runs"])
        pipeline.save()
        def stage(name, script, arguments, resumable=True):
            pipeline.stage(name, [sys.executable, "-u", ROOT / "analysis" / script, *arguments], resumable=resumable)
        for split in ("validation", "confirmation", "navtest"):
            # No source/weight/cohort changes between the separately scored splits.
            if verify(models, protocol) != core:
                raise ValueError("Core training artifacts changed after preflight")
            paths = []
            predictions = root / "predictions" / split
            for _, tag in (row for row in prediction_matrix() if row[0] == split):
                path = predictions / (tag + ".npz")
                stage("predict_" + split + "_" + tag, "predict_mechanism.py",
                      ["--run", models / tag, "--protocol", protocol, "--split", split, "--out", path, "--threads", "2"])
                paths.append(path)
            scores, analysis = results / "scores" / split, results / "analysis" / split
            stage("score_" + split, "score_mechanism.py", ["--predictions", *paths, "--out", scores])
            stage("analyze_" + split, "analyze_mechanism.py", ["--scores", scores, "--protocol", protocol, "--out", analysis])
            stage("cases_" + split, "mechanism_cases.py", ["--scores", scores, "--predictions", predictions,
                  "--protocol", protocol, "--out", results / "cases" / split])
            if split in ("validation", "confirmation"):
                gate = ["--validation", results / "analysis/validation/report.json", "--out", results / ("gate_" + split + ".json")]
                if split == "confirmation":
                    gate.extend(["--confirmation", analysis / "report.json"])
                stage("gate_" + split, "mechanism_gates.py", gate)
            stage("figures_" + split, "plot_mechanism.py", ["--report", analysis / "report.json", "--scores", scores,
                  "--runs", models, "--protocol", protocol, "--out", results / "figures" / split])
        pipeline.finish("core_scoring_completed")
        print("CORE SCORING COMPLETED; visual inspection and other full-goal requirements remain", flush=True)
    except BaseException as error:
        pipeline.fail(error)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
