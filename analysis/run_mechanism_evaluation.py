"""Wait for the existing core runner, then audit/cache/score serially; never train."""
from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from resource_guard import MemoryInterrupted, stop_owned_child, wait_for_headroom, wait_guarded
from verify_mechanism_core import CONFIGS, ROOT, digest


def acquire(path, wait=False, on_wait=None):
    """Hold an OS lock, not a stale PID file. Do not read a locked byte on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "a+b")
    if path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    notified = False
    while True:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return stream
        except OSError as error:
            if not wait or error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                stream.close()
                raise
            if not notified:
                if on_wait:
                    on_wait()
                notified = True
            time.sleep(3)


def upstream_binding(workspace, snapshot_path):
    """Bind the preserved current-runtime baseline, not retrospective provenance."""
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    devkit = workspace / "navsim"
    commit = subprocess.check_output(["git", "-C", str(devkit), "rev-parse", "HEAD"], text=True).strip()
    names = subprocess.check_output(["git", "-C", str(devkit), "ls-files", "-z"]).decode("utf-8").split("\0")
    names = [name for name in names if name]
    recorded = snapshot["tracked_devkit_sha256"]
    if commit != snapshot["devkit_base_commit"] or set(names) != set(recorded):
        raise ValueError("Upstream checkout differs from the preserved runtime baseline")
    current = {name: digest(devkit / name) for name in names}
    if current != recorded:
        raise ValueError("Upstream tracked source/config changed; preserve baseline and investigate before scoring")
    return {str(devkit / name): value for name, value in current.items()}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min-free-mb", type=int, default=2400)
    ap.add_argument("--critical-free-mb", type=int, default=1000)
    ap.add_argument("--emergency-free-mb", type=int, default=512)
    args = ap.parse_args()
    if args.threads < 1 or not 0 < args.emergency_free_mb <= args.critical_free_mb < args.min_free_mb:
        raise ValueError("Invalid thread or memory policy")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    for key, value in {"NAVSIM_WORKSPACE": args.workspace, "NAVSIM_DEVKIT_ROOT": args.workspace / "navsim",
                       "OPENSCENE_DATA_ROOT": args.workspace / "dataset", "NUPLAN_MAPS_ROOT": args.workspace / "dataset/maps",
                       "NAVSIM_EXP_ROOT": args.workspace / "exp"}.items():
        os.environ[key] = str(value)
    root = args.workspace / "exp/mechanism_v2"
    logs = root / "evaluation_logs"
    logs.mkdir(parents=True, exist_ok=True)
    models = root / "models"
    protocol = ROOT / "results/mechanism_v2/protocol/protocol.json"
    results = ROOT / "results/mechanism_v2"
    state_path = results / "evaluation_status.json"
    state = {"runner_pid": os.getpid(), "stage": "waiting_for_core", "status": "running", "jobs": [],
             "memory_policy": {"launch_free_mb": args.min_free_mb, "launch_stable_seconds": 20,
                               "critical_free_mb": args.critical_free_mb, "emergency_free_mb": args.emergency_free_mb,
                               "critical_samples": 2, "poll_seconds": 5}, "memory_events": []}
    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
        temporary.replace(state_path)
    eval_lock = acquire(root / "evaluation.lock")
    core_lock = None
    save()
    files = [ROOT / "analysis" / name for name in ("verify_mechanism_core.py", "audit_mechanism_recovery.py",
             "audit_mechanism_live.py", "audit_mechanism_inputs.py", "cache_mechanism.py", "predict_mechanism.py", "score_mechanism.py",
             "mechanism_scoring.py", "analyze_mechanism.py", "mechanism_cases.py", "mechanism_gates.py")]
    files.extend([protocol, protocol.parent / "donors.npz", protocol.parent / "diagnostics.json", protocol.parent / "followup_policy.json"])
    try:
        source_hashes = {str(path): digest(path) for path in files}
        snapshot_path = results / "runtime_snapshot/runtime.json"
        source_hashes.update(upstream_binding(args.workspace, snapshot_path))
        source_hashes[str(snapshot_path)] = digest(snapshot_path)
    except BaseException as error:
        state.update(status="failed_or_interrupted", exception=repr(error))
        save()
        eval_lock.close()
        raise
    state["upstream_binding_scope"] = "Matches the 2026-10-09 current-runtime snapshot, including the preserved dataloader patch; not a pre-original-launch or historical source-immutability claim"
    state["launch_source_and_policy_sha256"] = source_hashes
    supervisor_hashes = {str(ROOT / "analysis" / name): digest(ROOT / "analysis" / name)
                         for name in ("run_mechanism_evaluation.py", "resource_guard.py")}
    state["supervisor_sha256"] = supervisor_hashes
    save()
    def stage(name, script, arguments):
        command = [sys.executable, "-u", str(ROOT / "analysis" / script), *map(str, arguments)]
        path = logs / (name+".log")
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
        def event(details):
            details = dict(details, stage=name, observed_at_utc=datetime.now(timezone.utc).isoformat())
            state["memory_events"].append(details)
            state.update(stage=name, status=("waiting_for_memory" if details["kind"] == "waiting_for_memory"
                                            else "memory_protection" if details["kind"] == "memory_interruption"
                                            else "launch_ready"))
            save()
            print("MEMORY EVENT", json.dumps(details), flush=True)
        attempt = 0
        while True:
            for binding in (source_hashes, supervisor_hashes):
                if any(digest(path) != checksum for path, checksum in binding.items()):
                    raise ValueError("Evaluation source changed after launch; stopping before the next child")
            state.update(stage=name, child_pid=None)
            wait_for_headroom(args.min_free_mb, on_event=event)
            # Headroom waiting may be long; check bindings again before launch.
            for binding in (source_hashes, supervisor_hashes):
                if any(digest(path) != checksum for path, checksum in binding.items()):
                    raise ValueError("Evaluation source changed while waiting for memory")
            attempt += 1
            interruption = None
            with open(path, "a", encoding="utf-8") as output:
                output.write("\nSUPERVISED ATTEMPT " + str(attempt) + " " + datetime.now(timezone.utc).isoformat() + "\n")
                output.flush()
                child = subprocess.Popen(command, cwd=str(ROOT), stdout=output, stderr=subprocess.STDOUT, creationflags=flags)
                state.update(stage=name, status="running", child_pid=child.pid, log_path=str(path), stage_attempt=attempt)
                save()
                print("EVAL START", name, "attempt", attempt, "pid", child.pid, "log", path, flush=True)
                try:
                    code = wait_guarded(child, critical_mb=args.critical_free_mb,
                                        emergency_mb=args.emergency_free_mb, on_event=event)
                except MemoryInterrupted as error:
                    interruption = error.event
                    code = interruption["exit_code"]
                except BaseException:
                    stop_owned_child(child)
                    raise
            job = {"stage": name, "attempt": attempt, "exit_code": code, "log_path": str(path)}
            if interruption is not None:
                job["memory_interruption"] = interruption
            state["jobs"].append(job)
            state["child_pid"] = None
            save()
            if interruption is not None:
                if name not in ("cache_validation", "cache_confirmation"):
                    raise RuntimeError(name + " interrupted for memory; explicit review required before restart")
                print("CACHE RESUME WAIT", name, "preserving existing files; no failure skipped", flush=True)
                continue
            if code != 0:
                raise RuntimeError(name+" failed; inspect "+str(path)+"; not silently skipping/restarting")
            print("EVAL DONE", name, flush=True)
            return
    try:
        # Acquiring the training lock proves no core child is active. Completion
        # is checked separately; a killed/failed runner must never unlock scoring.
        core_lock = acquire(root / "core.lock", wait=True, on_wait=lambda: print("WAITING: existing core training holds its lock", flush=True))
        core_status = json.loads((results / "core_status.json").read_text(encoding="utf-8"))
        if core_status["status"] != "completed" or len(core_status["jobs"]) != 18:
            raise ValueError("Existing core runner did not finish all 18 jobs; evaluation remains gated")
        stage("core_artifacts", "verify_mechanism_core.py", ["--runs", models, "--protocol", protocol])
        stage("input_controls", "audit_mechanism_inputs.py", ["--protocol", protocol, "--cohort"])
        stage("process_recovery", "audit_mechanism_recovery.py", ["--threads", args.threads])
        stage("live_parity", "audit_mechanism_live.py", ["--runs", *[models / f"{config}_f1_steps_s0" for config in ("M0", "M1", "M3", "M4")], "--scenes", "16"])
        stage("cache_smoke", "cache_mechanism.py", ["--split", "validation", "--limit", "2", "--protocol", protocol])
        for split in ("validation", "confirmation", "navtest"):
            if split != "navtest":
                stage("cache_"+split, "cache_mechanism.py", ["--split", split, "--protocol", protocol])
            pred_folder = root / "predictions" / split
            paths = []
            for seed in range(3):
                for config in CONFIGS:
                    tag = f"{config}_f1_steps_s{seed}"
                    path = pred_folder / (tag+".npz")
                    stage("predict_"+split+"_"+tag, "predict_mechanism.py", ["--run", models / tag, "--protocol", protocol,
                          "--split", split, "--out", path, "--threads", args.threads])
                    paths.append(path)
            scores = results / "scores" / split
            analysis = results / "analysis" / split
            stage("score_"+split, "score_mechanism.py", ["--predictions", *paths, "--out", scores])
            stage("analyze_"+split, "analyze_mechanism.py", ["--scores", scores, "--protocol", protocol, "--out", analysis])
            stage("cases_"+split, "mechanism_cases.py", ["--scores", scores, "--predictions", pred_folder, "--protocol", protocol,
                  "--out", results / "cases" / split])
            if split in ("validation", "confirmation"):
                gate_args = ["--validation", results / "analysis/validation/report.json", "--out", results / ("gate_"+split+".json")]
                if split == "confirmation":
                    gate_args.extend(["--confirmation", analysis / "report.json"])
                stage("gate_"+split, "mechanism_gates.py", gate_args)
        state.update(stage="core_evaluation_completed", status="completed", child_pid=None,
                     scope="Core scoring/analysis/cases only; follow-up curves, v2 feasibility, full figures/manuscript and final audit still required")
        save()
        print("CORE EVALUATION COMPLETED; follow-up goal work remains", flush=True)
    except BaseException as error:
        state.update(status="failed_or_interrupted", exception=repr(error))
        save()
        raise
    finally:
        if core_lock is not None:
            core_lock.close()
        eval_lock.close()


if __name__ == "__main__":
    main()
