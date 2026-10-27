"""Resume only the two full caches after reviewed memory stops; never score/train.

Previously completed audits are required and bound, not claimed to be rerun.
This isolates cache work from an unnecessarily repeated high-memory raw audit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from resource_guard import MemoryInterrupted, stop_owned_child, wait_for_headroom, wait_guarded
from run_mechanism_evaluation import acquire, upstream_binding
from verify_mechanism_core import ROOT, digest, verify


def verify_prior_attempt(path):
    previous = json.loads(path.read_text(encoding="utf-8"))
    required = {"core_artifacts", "input_controls", "process_recovery", "live_parity", "cache_smoke"}
    succeeded = {row["stage"] for row in previous["jobs"] if row["exit_code"] == 0}
    if not required.issubset(succeeded) or previous["stage"] != "cache_validation" or previous["status"] != "failed_or_interrupted":
        raise ValueError("Prior successful audit stages and reviewed full-cache interruption required")
    for name, checksum in previous["launch_source_and_policy_sha256"].items():
        if digest(name) != checksum:
            raise ValueError("Previously audited source/policy changed: " + name)
    return previous["launch_source_and_policy_sha256"]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--previous-attempt", type=Path, default=ROOT / "results/mechanism_v2/run_attempts/2026-10-09_memory_stop/evaluation_status.json")
    args = ap.parse_args()
    root = args.workspace / "exp/mechanism_v2"
    results = ROOT / "results/mechanism_v2"
    protocol = results / "protocol/protocol.json"
    policy = json.loads(protocol.read_text(encoding="utf-8"))
    for key, value in {"NAVSIM_WORKSPACE": args.workspace, "NAVSIM_DEVKIT_ROOT": args.workspace / "navsim",
                       "OPENSCENE_DATA_ROOT": args.workspace / "dataset", "NUPLAN_MAPS_ROOT": args.workspace / "dataset/maps",
                       "NAVSIM_EXP_ROOT": args.workspace / "exp"}.items():
        os.environ[key] = str(value)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + str(os.getpid())
    attempt = results / "run_attempts" / ("cache_resume_" + stamp + ".json")
    status = results / "cache_resume_status.json"
    state = {"status": "preflight", "runner_pid": os.getpid(), "attempt_receipt": str(attempt), "jobs": [], "memory_events": [],
             "scope": "Cache only; no scores, follow-up models, gate decisions or project-completion claim"}
    def save():
        for path in (attempt, status):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
    save()
    lock = acquire(root / "core.lock")
    try:
        binding = dict(verify_prior_attempt(args.previous_attempt))
        binding[str(args.previous_attempt)] = digest(args.previous_attempt)
        core = verify(root / "models", protocol)
        reports = {name: results.parent / "audit_v2" / filename for name, filename in
                   (("core_artifacts", "core_manifest.json"), ("input_controls", "input_controls.json"),
                    ("process_recovery", "process_recovery.json"), ("live_parity", "mechanism_live.json"), ("cache_recovery", "cache_recovery.json"))}
        for name, path in reports.items():
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report["passed"]:
                raise ValueError("Required completed audit not passed: " + name)
            if name == "input_controls" and (not report["cohort_enumerated"] or report["protocol_sha256"] != core["protocol_sha256"]):
                raise ValueError("Prior full raw-cohort audit is not for this protocol")
            binding[str(path)] = digest(path)
        snapshot = results / "runtime_snapshot/runtime.json"
        binding.update(upstream_binding(args.workspace, snapshot))
        for path in (snapshot, Path(__file__), ROOT / "analysis/resource_guard.py", ROOT / "analysis/run_mechanism_evaluation.py"):
            binding[str(path)] = digest(path)
        state.update(status="preflight_passed", frozen_sha256=binding,
                     audit_reuse_disclosure="Actual earlier completed technical audits are bound now, not rerun or retrospectively claimed immutable; repeated raw audit stopped for memory. Full dataset/scoring/preservation checks remain separate.")
        save()
        logs = root / "cache_resume_logs"
        logs.mkdir(parents=True, exist_ok=True)
        for split in ("validation", "confirmation"):
            count = 0
            def event(details):
                state["memory_events"].append(dict(details, split=split, observed_at_utc=datetime.now(timezone.utc).isoformat()))
                state.update(stage=split, status=details["kind"])
                save()
                print("CACHE MEMORY", split, details, flush=True)
            while True:
                wait_for_headroom(on_event=event)
                if any(digest(path) != checksum for path, checksum in binding.items()):
                    raise ValueError("Cache resume source/policy/audit receipt changed")
                count += 1
                log = logs / (split + "_" + stamp + ".log")
                interruption = None
                flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
                with open(log, "a", encoding="utf-8") as stream:
                    stream.write("\nATTEMPT " + str(count) + " " + datetime.now(timezone.utc).isoformat() + "\n")
                    stream.flush()
                    child = subprocess.Popen([sys.executable, "-u", str(ROOT / "analysis/cache_mechanism.py"), "--split", split,
                                              "--protocol", str(protocol)], cwd=str(ROOT), stdout=stream,
                                             stderr=subprocess.STDOUT, creationflags=flags)
                    state.update(status="running", stage=split, child_pid=child.pid, log_path=str(log))
                    save()
                    print("CACHE START", split, "attempt", count, "pid", child.pid, flush=True)
                    try:
                        code = wait_guarded(child, on_event=event)
                    except MemoryInterrupted as error:
                        interruption, code = error.event, error.event["exit_code"]
                    except BaseException:
                        stop_owned_child(child)
                        raise
                state["jobs"].append({"split": split, "attempt": count, "exit_code": code,
                                      "memory_interruption": interruption, "log_path": str(log)})
                state["child_pid"] = None
                save()
                if interruption is not None:
                    continue
                if code != 0:
                    raise RuntimeError("Full cache failed; inspect log; no ordinary failure retry/skip")
                complete = json.loads((root / "metric_cache" / (split + "_completed.json")).read_text(encoding="utf-8"))
                if (complete["protocol_sha256"] != core["protocol_sha256"] or complete["split"] != split
                    or complete["limit"] is not None or complete["scenes"] != policy["split_counts"][split]["scenes"]):
                    raise ValueError("Full cache completion receipt mismatch")
                break
        state.update(status="caches_completed", stage=None, child_pid=None)
        save()
        print("BOTH FULL CACHES COMPLETED; scoring and project work remain", flush=True)
    except BaseException as error:
        state.update(status="failed_or_interrupted", exception=repr(error), child_pid=None)
        save()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
