"""Run the 18 preregistered core models serially, safely resuming completed jobs."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from runtime import ROOT, configure


def available_mb():
    if os.name != "nt":
        return None
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                    ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong)]
    memory = Memory()
    memory.length = ctypes.sizeof(Memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("Cannot read system memory")
    return memory.available/1024**2


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min-free-mb", type=int, default=1600)
    ap.add_argument("--configs", nargs="+", choices=["M0", "M1", "M2", "M3", "M4", "M5"],
                    default=["M0", "M1", "M2", "M3", "M4", "M5"])
    args = ap.parse_args()
    workspace = configure(args.workspace)
    destination = workspace / "exp/mechanism_v2"
    destination.mkdir(parents=True, exist_ok=True)
    logs = destination / "logs"
    logs.mkdir(exist_ok=True)
    # Hold an OS-level lock. A stale PID/JSON file alone is not a running job.
    lock = open(destination / "core.lock", "a+b")
    if os.name == "nt":
        import msvcrt
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {"runner_pid": os.getpid(), "stage": "core_training", "status": "running", "jobs": []}
    status_path = ROOT / "results/mechanism_v2/core_status.json"
    def save():
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        temporary.replace(status_path)
    save()
    try:
        for seed in range(3):
            for config in args.configs:
                free = available_mb()
                if free is not None and free < args.min_free_mb:
                    manifest["status"] = "waiting_for_memory"
                    save()
                    print(f"Memory guard: {free:.0f} MB free; waiting before next job", flush=True)
                    while free < args.min_free_mb:
                        time.sleep(30)
                        free = available_mb()
                tag = f"{config}_f1_steps_s{seed}"
                path = logs / (tag + ".log")
                command = [sys.executable, "-u", str(ROOT / "analysis/train_mechanism.py"),
                           "--config", config, "--seed", str(seed), "--threads", str(args.threads)]
                creationflags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
                with open(path, "a", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                               creationflags=creationflags)
                    manifest.update(status="running", current_job=tag, child_pid=process.pid, log_path=str(path))
                    save()
                    print("START", tag, "pid", process.pid, "log", path, flush=True)
                    code = process.wait()
                manifest["jobs"].append({"tag": tag, "exit_code": code, "log_path": str(path)})
                if code != 0:
                    raise RuntimeError(f"{tag} failed with exit {code}; inspect {path}; not restarting automatically")
                print("DONE", tag, flush=True)
                save()
        manifest.update(status="completed", child_pid=None, current_job=None)
        save()
        print("CORE TRAINING COMPLETED", len(manifest["jobs"]), "jobs", flush=True)
    except BaseException as error:
        manifest.update(status="failed_or_interrupted", exception=repr(error))
        save()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
