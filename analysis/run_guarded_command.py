"""Run one explicit local check under the shared heavy-job lock and RAM watchdog."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from resource_guard import stop_owned_child, wait_for_headroom, wait_guarded
from run_mechanism_evaluation import acquire


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--log", type=Path, help="Default: the receipt path with a .log extension")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Explicit command required; no shell evaluation")
    if args.receipt.exists():
        raise ValueError("Preserve old check receipt; choose a new attempt path")
    log_path = args.log or args.receipt.with_suffix(".log")
    if log_path.exists():
        raise ValueError("Preserve old check log; choose a new attempt path")
    state = {"command": command, "status": "waiting", "events": [],
             "started_utc": datetime.now(timezone.utc).isoformat(), "runner_pid": os.getpid(), "log_path": str(log_path)}
    def save():
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.receipt)
    def event(details):
        state["events"].append(dict(details, observed_at_utc=datetime.now(timezone.utc).isoformat()))
        save()
        print("GUARDED CHECK", details, flush=True)
    save()
    lock = acquire(args.workspace / "exp/mechanism_v2/core.lock")
    child = None
    try:
        wait_for_headroom(on_event=event)
        environment = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "2"
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "x", encoding="utf-8") as output:
            child = subprocess.Popen(command, env=environment, creationflags=flags, stdout=output, stderr=subprocess.STDOUT)
            state.update(status="running", child_pid=child.pid)
            save()
            code = wait_guarded(child, on_event=event)
        state.update(status="completed" if code == 0 else "failed", exit_code=code, child_pid=None,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        if code != 0:
            raise RuntimeError("Guarded check failed with exit " + str(code))
        print("GUARDED CHECK COMPLETED; full output:", log_path, flush=True)
    except BaseException as error:
        if child is not None:
            stop_owned_child(child)
        state.update(status="failed_or_interrupted", exception=repr(error), child_pid=None)
        save()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
