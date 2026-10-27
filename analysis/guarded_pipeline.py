"""Logged serial child stages with immutable inputs and owned-process RAM guards."""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from resource_guard import MemoryInterrupted, stop_owned_child, wait_for_headroom, wait_guarded
from verify_mechanism_core import ROOT, digest


class Pipeline:
    def __init__(self, label, status, logs, attempt_folder, scope):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + str(os.getpid())
        self.status = Path(status)
        self.logs = Path(logs)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.attempt = Path(attempt_folder) / (label + "_" + stamp + ".json")
        self.state = {"status": "preflight", "runner_pid": os.getpid(), "jobs": [], "memory_events": [],
                      "scope": scope, "attempt_receipt": str(self.attempt), "started_utc": datetime.now(timezone.utc).isoformat()}
        self.binding = {}
        self.save()

    def save(self):
        for path in (self.attempt, self.status):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

    def bind(self, files, existing=None):
        self.binding.update(existing or {})
        self.binding.update({str(path): digest(path) for path in files})
        self.check()
        self.state["frozen_sha256"] = self.binding
        self.save()

    def check(self):
        for path, checksum in self.binding.items():
            if digest(path) != checksum:
                raise ValueError("Pipeline bound input/source/receipt changed: " + path)

    def event(self, name, details):
        self.state["memory_events"].append(dict(details, stage=name, observed_at_utc=datetime.now(timezone.utc).isoformat()))
        self.state.update(stage=name, status=details["kind"])
        self.save()
        print("PIPELINE MEMORY", name, details, flush=True)

    def stage(self, name, command, resumable=False):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or ".." in name:
            raise ValueError("Invalid stage log name")
        attempt = 0
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "2"
        while True:
            self.check()
            self.state.update(stage=name, status="waiting_for_memory", child_pid=None)
            self.save()
            wait_for_headroom(on_event=lambda details: self.event(name, details))
            self.check()
            attempt += 1
            log = self.logs / (name + ".log")
            interrupted = None
            flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
            with open(log, "a", encoding="utf-8") as stream:
                stream.write("\nATTEMPT " + str(attempt) + " " + datetime.now(timezone.utc).isoformat() + "\n")
                stream.flush()
                child = subprocess.Popen(list(map(str, command)), cwd=str(ROOT), env=environment,
                                         stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                self.state.update(status="running", stage=name, child_pid=child.pid, log_path=str(log))
                self.save()
                print("PIPELINE START", name, "attempt", attempt, "pid", child.pid, flush=True)
                try:
                    code = wait_guarded(child, on_event=lambda details: self.event(name, details))
                except MemoryInterrupted as error:
                    interrupted, code = error.event, error.event["exit_code"]
                except BaseException:
                    stop_owned_child(child)
                    raise
            self.state["jobs"].append({"stage": name, "attempt": attempt, "exit_code": code,
                                       "memory_interruption": interrupted, "command": list(map(str, command)), "log_path": str(log)})
            self.state["child_pid"] = None
            self.save()
            if interrupted is not None and resumable:
                continue
            if interrupted is not None or code != 0:
                raise RuntimeError("Pipeline stage failed/interrupted; no ordinary failure skip/retry: " + name)
            print("PIPELINE DONE", name, flush=True)
            return

    def finish(self, status):
        self.check()
        self.state.update(status=status, child_pid=None, finished_utc=datetime.now(timezone.utc).isoformat())
        self.save()

    def fail(self, error):
        self.state.update(status="failed_or_interrupted", exception=repr(error), child_pid=None)
        self.save()
