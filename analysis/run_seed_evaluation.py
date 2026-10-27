"""Serial added-seed inference/scoring, retaining original core score identities."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from run_city_evaluation import completed_attempt
from run_mechanism_evaluation import acquire, upstream_binding
from run_seed_training import SeedPipeline
from seed_models import verify_all
from verify_mechanism_core import CONFIGS, ROOT, digest


def prediction_matrix():
    return [(split, config, seed) for split in ("validation", "confirmation", "navtest")
            for seed in (3, 4) for config in CONFIGS]


def scoring_command(python, predictions, scores):
    # Only the twelve extended-protocol predictions may share this scorer batch.
    # Original-protocol predictions are NEVER falsely relabelled or mixed here.
    expected = {f"{config}_f1_steps_s{seed}.npz" for config in CONFIGS for seed in (3, 4)}
    if (len(predictions) != 12 or {Path(path).name for path in predictions} != expected
            or len({Path(path).parent.resolve() for path in predictions}) != 1):
        raise ValueError("Exactly twelve added-seed predictions from one split folder required")
    return [python, "-u", ROOT / "analysis/score_mechanism.py", "--predictions", *predictions, "--out", scores]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    results = ROOT / "results/mechanism_v2"
    work = args.workspace / "exp/mechanism_v2"
    base = results / "protocol/protocol.json"
    registration = results / "seed_protocol/registration.json"
    protocol = registration.parent / "active/protocol.json"
    pipeline = SeedPipeline("seed_evaluation", results / "seed_evaluation_status.json", work / "seed_evaluation_logs",
                            results / "run_attempts", "36 added-seed exports, three separate twelve-run official score batches and audited thirty-run/five-seed reports; figure QA, interpretation and final project audit remain")
    lock = None
    try:
        lock = acquire(work / "core.lock", wait=args.wait, on_wait=lambda: print("SEED EVALUATION WAIT: heavy-job lock owned", flush=True))
        for key, value in {"NAVSIM_WORKSPACE": args.workspace, "NAVSIM_DEVKIT_ROOT": args.workspace / "navsim",
                           "OPENSCENE_DATA_ROOT": args.workspace / "dataset", "NUPLAN_MAPS_ROOT": args.workspace / "dataset/maps",
                           "NAVSIM_EXP_ROOT": args.workspace / "exp"}.items():
            os.environ[key] = str(value)
        core_attempt, core_binding = completed_attempt(results / "scoring_status.json", "core_scoring_completed")
        train_attempt, train_binding = completed_attempt(results / "seed_training_status.json", "twelve_added_seed_models_verified")
        checked = verify_all(work / "models", work / "seed_models", base, base.parent / "followup_policy.json", registration, results)
        if not checked[3]["required"]:
            raise ValueError("No verified five-seed extension trigger")
        files = [core_attempt, train_attempt, base, registration, protocol, protocol.parent / "donors.npz", protocol.parent / "completed.json"]
        files.extend(ROOT / "analysis" / name for name in ("run_seed_evaluation.py", "predict_seed.py", "analyze_seed.py", "seed_models.py",
                     "seed_followup.py", "run_seed_training.py", "run_city_evaluation.py", "score_mechanism.py", "mechanism_scoring.py",
                     "analyze_mechanism.py", "guarded_pipeline.py", "resource_guard.py"))
        pipeline.bind(files, existing={**core_binding, **train_binding,
                                      **upstream_binding(args.workspace, results / "runtime_snapshot/runtime.json")})
        pipeline.state["matrix"] = prediction_matrix()
        pipeline.save()
        for split in ("validation", "confirmation", "navtest"):
            if verify_all(work / "models", work / "seed_models", base, base.parent / "followup_policy.json", registration, results) != checked:
                raise ValueError("Core/added-seed artifacts changed after evaluation preflight")
            paths = []
            for _, config, seed in (row for row in prediction_matrix() if row[0] == split):
                tag = f"{config}_f1_steps_s{seed}"
                path = work / "seed_predictions" / split / (tag + ".npz")
                pipeline.stage("predict_" + split + "_" + tag, [sys.executable, "-u", ROOT / "analysis/predict_seed.py",
                               "--workspace", args.workspace, "--config", config, "--seed", seed, "--split", split, "--out", path])
                paths.append(path)
            pipeline.stage("score_" + split, scoring_command(sys.executable, paths, results / "seed_scores" / split))
            pipeline.stage("analyze_" + split, [sys.executable, "-u", ROOT / "analysis/analyze_seed.py", "--workspace", args.workspace,
                           "--split", split, "--out", results / "seed_analysis" / split])
        pipeline.finish("seed_evaluation_completed")
        print("FIVE-SEED SCORE/ANALYSIS PIPELINE COMPLETE; scientific figures, visual QA and final goal audit remain", flush=True)
    except BaseException as error:
        pipeline.fail(error)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
