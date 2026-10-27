"""Serial matched-cohort city prediction/scoring/analysis after verified training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from city_followup import verify_gate
from city_models import CONDITIONS, verify_all
from guarded_pipeline import Pipeline
from run_mechanism_evaluation import acquire, upstream_binding
from verify_mechanism_core import ROOT, digest


def prediction_matrix():
    return [(split, condition, seed) for split in ("validation", "confirmation", "navtest")
            for seed in (0, 1, 2) for condition in CONDITIONS]


def completed_attempt(path, status):
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["status"] != status:
        raise ValueError("Prerequisite controller has not completed: " + status)
    attempt = Path(state["attempt_receipt"])
    recorded = json.loads(attempt.read_text(encoding="utf-8"))
    if recorded != state or not state.get("jobs"):
        raise ValueError("Prerequisite progress snapshot is not its completed attempt")
    last = {}
    for job in state["jobs"]:
        last[job["stage"]] = job
    if any(job["exit_code"] != 0 or job["memory_interruption"] is not None for job in last.values()):
        raise ValueError("Prerequisite has an unsuccessful final stage")
    for name, checksum in state["frozen_sha256"].items():
        if digest(name) != checksum:
            raise ValueError("Prerequisite bound source/input changed")
    return attempt, state["frozen_sha256"]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    results = ROOT / "results/mechanism_v2"
    root = args.workspace / "exp/mechanism_v2"
    base, city = results / "protocol/protocol.json", results / "city_protocol/protocol.json"
    audit = ROOT / "results/audit_v2/city_inputs.json"
    pipeline = Pipeline("city_evaluation", results / "city_evaluation_status.json", root / "city_evaluation_logs", results / "run_attempts",
                        "36 common-cohort exports, three official score batches, audited city reports and rendered figures; visual QA, interpretation and full-goal final audit remain")
    lock = None
    try:
        lock = acquire(root / "core.lock", wait=args.wait, on_wait=lambda: print("CITY EVALUATION WAIT: heavy-job lock owned", flush=True))
        for key, value in {"NAVSIM_WORKSPACE": args.workspace, "NAVSIM_DEVKIT_ROOT": args.workspace / "navsim",
                           "OPENSCENE_DATA_ROOT": args.workspace / "dataset", "NUPLAN_MAPS_ROOT": args.workspace / "dataset/maps",
                           "NAVSIM_EXP_ROOT": args.workspace / "exp"}.items():
            os.environ[key] = str(value)
        core_attempt, core_binding = completed_attempt(results / "scoring_status.json", "core_scoring_completed")
        train_attempt, train_binding = completed_attempt(results / "city_training_status.json", "six_city_models_verified")
        checked = verify_all(root / "models", root / "city_models", base, city, audit)
        gate = verify_gate(city.parent / "execution_policy.json", base, city, results, checked[0])
        if not gate["required"] or json.loads((results / "gate_city.json").read_text(encoding="utf-8")) != gate:
            raise ValueError("No matching actual-score city training trigger")
        files = [core_attempt, train_attempt, base, city, city.parent / "donors.npz", audit, results / "gate_city.json",
                 city.parent / "execution_policy.json"]
        files.extend(ROOT / "analysis" / name for name in ("run_city_evaluation.py", "city_models.py", "city_followup.py", "predict_city.py",
                     "analyze_city.py", "analyze_mechanism.py", "plot_followups.py", "score_mechanism.py", "mechanism_scoring.py", "guarded_pipeline.py", "resource_guard.py"))
        pipeline.bind(files, existing={**core_binding, **train_binding, **upstream_binding(args.workspace, results / "runtime_snapshot/runtime.json")})
        pipeline.state["matrix"] = prediction_matrix()
        pipeline.save()
        def stage(name, script, arguments):
            pipeline.stage(name, [sys.executable, "-u", ROOT / "analysis" / script, *arguments], resumable=True)
        for split in ("validation", "confirmation", "navtest"):
            if verify_all(root / "models", root / "city_models", base, city, audit) != checked:
                raise ValueError("City/core model artifacts changed after preflight")
            predictions = root / "city_predictions" / split
            paths = []
            for _, condition, seed in (row for row in prediction_matrix() if row[0] == split):
                tag = f"{condition}_f1_steps_s{seed}"
                path = predictions / (tag + ".npz")
                stage("predict_" + split + "_" + tag, "predict_city.py",
                      ["--condition", condition, "--seed", seed, "--split", split, "--workspace", args.workspace, "--out", path])
                paths.append(path)
            scores = results / "city_scores" / split
            stage("score_" + split, "score_mechanism.py", ["--predictions", *paths, "--out", scores])
            stage("analyze_" + split, "analyze_city.py", ["--workspace", args.workspace, "--scores", scores,
                  "--predictions", predictions, "--out", results / "city_analysis" / split])
            stage("figures_" + split, "plot_followups.py", ["--kind", "city", "--workspace", args.workspace,
                  "--report", results / "city_analysis" / split / "report.json", "--scores", scores,
                  "--predictions", predictions, "--out", results / "figures/city" / split])
        pipeline.finish("city_evaluation_completed")
        print("CITY EVALUATION COMPLETED; visual inspection, claim audit and other project requirements remain", flush=True)
    except BaseException as error:
        pipeline.fail(error)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
