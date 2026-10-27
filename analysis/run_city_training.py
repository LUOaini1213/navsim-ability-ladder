"""Train six same-city auxiliary models only after the prospective actual-score gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from city_followup import verify_gate
from city_models import matrix, verify_all, verify_inputs, verify_one
from guarded_pipeline import Pipeline
from run_mechanism_evaluation import acquire
from verify_mechanism_core import ROOT, digest, verify


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--wait-for-core-scoring", action="store_true")
    args = ap.parse_args()
    results = ROOT / "results/mechanism_v2"
    base_path, city_path = results / "protocol/protocol.json", results / "city_protocol/protocol.json"
    audit_path = ROOT / "results/audit_v2/city_inputs.json"
    root = args.workspace / "exp/mechanism_v2"
    models = root / "models"
    auxiliary = root / "city_models"
    pipeline = Pipeline("city_training", results / "city_training_status.json", root / "city_training_logs", results / "run_attempts",
                        "Six auxiliary training models only; matched-coverage scoring, interpretation and final project audit remain")
    pipeline.state["status"] = "waiting_for_core_scoring"
    pipeline.save()
    lock = None
    try:
        lock = acquire(root / "core.lock", wait=args.wait_for_core_scoring,
                       on_wait=lambda: print("CITY TRAINING WAIT: heavy-job lock owned", flush=True))
        os.environ["NAVSIM_WORKSPACE"] = str(args.workspace)
        core = verify(models, base_path)
        protocol = verify_inputs(base_path, city_path, audit_path)
        policy_path = city_path.parent / "execution_policy.json"
        gate = verify_gate(policy_path, base_path, city_path, results, core)
        gate_path = results / "gate_city.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        pipeline.state.update(gate_sha256=digest(gate_path), gate=gate)
        if not gate["required"]:
            pipeline.finish("not_triggered_clean_alignment_claim_withheld")
            print("CITY CONTROL NOT TRIGGERED; clean-alignment interpretation withheld", flush=True)
            return
        files = [base_path, city_path, city_path.parent / "donors.npz", city_path.parent / "design.json",
                 city_path.parent / "completed.json", audit_path, policy_path, gate_path,
                 results / "analysis/validation/report.json", results / "gate_validation.json"]
        files.extend(ROOT / "analysis" / name for name in ("run_city_training.py", "city_followup.py", "city_models.py", "guarded_pipeline.py",
                     "train_mechanism.py", "mechanism_data.py", "run_mechanism_curves.py", "resource_guard.py"))
        files.append(ROOT / "agents/mechanism_model.py")
        pipeline.bind(files)
        pipeline.state["matrix"] = matrix()
        pipeline.save()
        for row in matrix():
            pipeline.stage("train_" + row["tag"], [sys.executable, "-u", ROOT / "analysis/train_mechanism.py",
                           "--workspace", args.workspace, "--protocol", city_path, "--config", row["config"], "--seed", row["seed"],
                           "--threads", "2", "--out-root", auxiliary], resumable=True)
            verify_one(auxiliary, row, protocol, city_path, core)
        verify_all(models, auxiliary, base_path, city_path, audit_path)
        pipeline.finish("six_city_models_verified")
        print("SIX CITY MODELS VERIFIED; auxiliary prediction/score/analysis still required", flush=True)
    except BaseException as error:
        pipeline.fail(error)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
