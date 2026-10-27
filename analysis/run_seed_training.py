"""Train a verified five-seed extension serially; no scores or final-result claim."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from guarded_pipeline import Pipeline
from resource_guard import MemoryInterrupted, stop_owned_child, wait_for_headroom, wait_guarded
from run_mechanism_evaluation import acquire
from seed_models import activate, verify_all, verify_one
from verify_mechanism_core import ROOT, digest

LAUNCH_POLICY = {"min_free_mb": 6000, "stable_seconds": 30, "poll_seconds": .5}
RUNNING_POLICY = {"critical_mb": 2000, "emergency_mb": 1500, "low_samples": 1, "poll_seconds": .5}


class SeedPipeline(Pipeline):
    """Use the proven conservative headroom policy, with zero automatic retries."""
    def stage(self, name, command, resumable=False):
        if resumable:
            raise ValueError("No automatic memory retries; explicit future invocation can resume checkpoints")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or ".." in name:
            raise ValueError("Invalid seed-stage log name")
        self.check()
        self.state.update(stage=name, status="waiting_for_memory", child_pid=None,
                          memory_policy={"launch": LAUNCH_POLICY, "running": RUNNING_POLICY,
                                         "automatic_retries": 0})
        self.save()
        wait_for_headroom(**LAUNCH_POLICY, on_event=lambda details: self.event(name, details))
        self.check()
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "2"
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
        log = self.logs / (name + ".log")
        error, interruption = None, None
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\nSINGLE ATTEMPT " + datetime.now(timezone.utc).isoformat() + "\n")
            stream.flush()
            child = subprocess.Popen(list(map(str, command)), cwd=str(ROOT), env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
            try:
                self.state.update(status="running", stage=name, child_pid=child.pid, log_path=str(log))
                self.save()
                code = wait_guarded(child, **RUNNING_POLICY,
                                    on_event=lambda details: self.event(name, details))
            except MemoryInterrupted as caught:
                error, interruption, code = caught, caught.event, caught.event["exit_code"]
            except BaseException as caught:
                error, code = caught, stop_owned_child(child)
        self.state["jobs"].append({"stage": name, "attempt": 1, "exit_code": code,
                                   "memory_interruption": interruption,
                                   "exception": repr(error) if error else None,
                                   "command": list(map(str, command)), "log_path": str(log)})
        self.state["child_pid"] = None
        self.save()
        if error is not None:
            raise error
        if code != 0:
            raise RuntimeError("Added-seed stage failed; no skip or automatic retry: " + name)
        self.check()


def training_command(python, workspace, protocol_path, out, row):
    return [python, "-u", ROOT / "analysis/train_mechanism.py", "--workspace", workspace,
            "--protocol", protocol_path, "--config", row["config"], "--seed", row["seed"],
            "--fraction", "1", "--budget", "steps", "--threads", "2", "--out-root", out]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--wait-for-core-scoring", action="store_true")
    args = ap.parse_args()
    results = ROOT / "results/mechanism_v2"
    base = results / "protocol/protocol.json"
    policy = base.parent / "followup_policy.json"
    registration = results / "seed_protocol/registration.json"
    work = args.workspace / "exp/mechanism_v2"
    models, added = work / "models", work / "seed_models"
    pipeline = SeedPipeline("seed_training", results / "seed_training_status.json",
                            work / "seed_training_logs", results / "run_attempts",
                            "Twelve gated full-budget added-seed models only; inference, scores, five-seed analysis and final project audit remain")
    pipeline.state["status"] = "waiting_for_core_scoring"
    pipeline.save()
    lock = None
    try:
        lock = acquire(work / "core.lock", wait=args.wait_for_core_scoring,
                       on_wait=lambda: print("SEED TRAINING WAIT: heavy-job lock owned", flush=True))
        os.environ["NAVSIM_WORKSPACE"] = str(args.workspace)
        path, core, gate = activate(models, base, policy, registration, results)
        pipeline.state["gate"] = gate
        if path is None:
            pipeline.finish("not_triggered_no_added_training")
            print("FIVE-SEED EXTENSION NOT TRIGGERED; no additional jobs", flush=True)
            return
        registered = json.loads(registration.read_text(encoding="utf-8"))
        files = [base, base.parent / "donors.npz", policy, registration, path,
                 path.parent / "donors.npz", path.parent / "completed.json",
                 results / "gate_confirmation.json"]
        for split in ("validation", "confirmation"):
            scores = results / "scores" / split
            files.extend([results / "analysis" / split / "report.json", scores / "settings.json", scores / "completed.json"])
            files.extend(scores / (row["tag"] + ".csv.gz") for row in core["runs"])
        files.extend(ROOT / "analysis" / name for name in
                     ("run_seed_training.py", "seed_models.py", "seed_followup.py", "guarded_pipeline.py",
                      "resource_guard.py", "train_mechanism.py", "mechanism_data.py", "verify_mechanism_core.py", "mechanism_gates.py"))
        files.append(ROOT / "agents/mechanism_model.py")
        pipeline.bind(files)
        pipeline.state["matrix"] = registered["matrix"]
        pipeline.save()
        base_protocol = json.loads(base.read_text(encoding="utf-8"))
        for row in registered["matrix"]:
            pipeline.stage("train_" + row["tag"], training_command(sys.executable, args.workspace, path, added, row))
            verify_one(added, row, path, base_protocol, core)
        verify_all(models, added, base, policy, registration, results)
        pipeline.finish("twelve_added_seed_models_verified")
        print("TWELVE ADDED MODELS VERIFIED; five-seed inference/scoring/analysis still required", flush=True)
    except BaseException as error:
        pipeline.fail(error)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
