"""Monitor free RAM while an owned child runs; never stop unrelated processes."""
from __future__ import annotations

import os
import subprocess
import time

from run_mechanism_core import available_mb


class MemoryInterrupted(RuntimeError):
    def __init__(self, event):
        self.event = event
        super().__init__("Owned child stopped for low memory: " + repr(event))


def stop_owned_child(child):
    """The still-open Popen handle owns this PID; no name/glob process matching."""
    if child.poll() is not None:
        return child.returncode
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
            capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=20,
        )
        if result.returncode != 0 and child.poll() is None:
            raise RuntimeError("Cannot stop owned child tree: " + result.stderr)
    else:
        # Non-Windows memory measurement is unavailable in this project. This
        # branch only cleans up the direct child on caller cancellation.
        child.terminate()
    return child.wait(timeout=20)


def wait_for_headroom(min_free_mb=2400, stable_seconds=20, poll_seconds=5,
                      read_memory=available_mb, sleep=time.sleep,
                      monotonic=time.monotonic, on_event=None):
    if min_free_mb <= 0 or stable_seconds < 0 or poll_seconds <= 0:
        raise ValueError("Invalid memory launch policy")
    good_since = None
    waiting_reported = False
    while True:
        free = read_memory()
        if free is None:
            if os.name == "nt":
                raise RuntimeError("Windows RAM measurement unavailable; refusing launch")
            if on_event:
                on_event({"kind": "memory_monitor_unavailable", "platform": os.name})
            return
        now = monotonic()
        if free >= min_free_mb:
            if good_since is None:
                good_since = now
            if now - good_since >= stable_seconds:
                if on_event:
                    on_event({"kind": "memory_headroom_ready", "available_mb": free,
                              "stable_seconds": stable_seconds})
                return
        else:
            good_since = None
        if not waiting_reported and on_event:
            on_event({"kind": "waiting_for_memory", "available_mb": free,
                      "minimum_mb": min_free_mb, "stable_seconds": stable_seconds})
            waiting_reported = True
        sleep(poll_seconds)


def wait_guarded(child, critical_mb=1000, emergency_mb=512, low_samples=2,
                 poll_seconds=5, read_memory=available_mb, sleep=time.sleep,
                 stop=stop_owned_child, on_event=None):
    if not 0 < emergency_mb <= critical_mb or low_samples < 1 or poll_seconds <= 0:
        raise ValueError("Invalid running memory policy")
    consecutive_low = 0
    recent = []
    while True:
        code = child.poll()
        if code is not None:
            return code
        try:
            free = read_memory()
        except Exception:
            stop(child)
            raise
        if free is not None:
            recent = (recent + [float(free)])[-low_samples:]
            consecutive_low = consecutive_low + 1 if free < critical_mb else 0
            if free < emergency_mb or consecutive_low >= low_samples:
                event = {"kind": "memory_interruption", "child_pid": child.pid,
                         "available_mb": free, "recent_samples_mb": recent,
                         "critical_mb": critical_mb, "emergency_mb": emergency_mb,
                         "required_low_samples": low_samples,
                         "termination_scope": "owned_Popen_child_tree_only"}
                if on_event:
                    on_event(event)
                event["exit_code"] = stop(child)
                raise MemoryInterrupted(event)
        elif os.name == "nt":
            stop(child)
            raise RuntimeError("Windows RAM measurement lost during owned child")
        sleep(poll_seconds)
