"""Interrupt ONLY a newly created smoke child, then compare actual resumed training."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from runtime import ROOT, configure


def launch(root, log, threads):
    command = [sys.executable, "-u", str(ROOT / "analysis/train_mechanism.py"), "--config", "M4", "--seed", "0",
               "--max-steps", "1000", "--threads", str(threads), "--out-root", str(root)]
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    return process, command


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2/process_recovery.json")
    args = ap.parse_args()
    workspace = configure()
    trial = workspace / "exp/mechanism_v2/recovery" / uuid.uuid4().hex
    trial.mkdir(parents=True)
    baseline_root, interrupted_root = trial / "baseline", trial / "interrupted"
    tag = "M4_f1_steps_s0_smoke1000"
    with open(trial / "baseline.log", "w", encoding="utf-8") as stream:
        baseline, baseline_command = launch(baseline_root, stream, args.threads)
        if baseline.wait() != 0:
            raise RuntimeError("Baseline smoke failed; inspect "+str(trial))
    with open(trial / "interrupted.log", "w", encoding="utf-8") as stream:
        child, command = launch(interrupted_root, stream, args.threads)
        deadline = time.monotonic()+600
        resume_path = interrupted_root / tag / "resume.pt"
        history_path = interrupted_root / tag / "history.json"
        while child.poll() is None and time.monotonic() < deadline:
            if resume_path.exists() and history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if history and 0 < history[-1]["step"] < 1000:
                    break
            time.sleep(.1)
        else:
            # Do not terminate on timeout: this diagnostic never kills a job
            # without the specific intended recovery checkpoint.
            raise RuntimeError("No interruptible checkpoint observed; child PID "+str(child.pid)+"; inspect "+str(trial))
        # This PID is exclusively the newly launched, uniquely rooted smoke child.
        # Windows Python launchers may spawn the actual interpreter; target only
        # this owned child tree, never the full core runner or unrelated jobs.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            child.terminate()
        exit_code = child.wait(timeout=30)
        if exit_code == 0 or (interrupted_root / tag / "completed.json").exists():
            raise RuntimeError("Child completed instead of being interrupted; no recovery claim")
    with open(trial / "resumed.log", "w", encoding="utf-8") as stream:
        child, resumed_command = launch(interrupted_root, stream, args.threads)
        if child.wait() != 0:
            raise RuntimeError("Resumed smoke failed; inspect "+str(trial))
    match = re.search(r"steps=(\d+)/1000", (trial / "resumed.log").read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("Resume start step missing from actual child log")
    resumed_from = int(match.group(1))
    # Keep the parent lightweight until every child has stopped; no duplicate
    # Torch runtime competes with a training child's feature tensors.
    import torch
    baseline_state = torch.load(baseline_root / tag / "resume.pt", map_location="cpu")
    resumed_state = torch.load(interrupted_root / tag / "resume.pt", map_location="cpu")
    def compare(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and set(left) == set(right) and all(compare(left[key], right[key]) for key in left)
        if isinstance(left, (tuple, list)):
            return isinstance(right, type(left)) and len(left) == len(right) and all(compare(a, b) for a, b in zip(left, right))
        return left == right
    fields = ("settings", "model", "optimizer", "rng", "step", "order", "position")
    checks = {key: compare(baseline_state[key], resumed_state[key]) for key in fields}
    validation_match = [row["validation"] for row in baseline_state["history"]] == [row["validation"] for row in resumed_state["history"]]
    checks["validation_history"] = validation_match
    for root in (baseline_root, interrupted_root):
        complete = json.loads((root / tag / "completed.json").read_text(encoding="utf-8"))
        if complete["eligible_for_paper"]:
            raise ValueError("Recovery smoke cannot be paper-eligible")
    report = {"passed": all(checks.values()) and 0 < resumed_from < 1000, "resumed_from_step": resumed_from,
              "final_step": resumed_state["step"], "checks": checks, "trial_directory": str(trial),
              "interrupted_command": command, "interrupted_exit_code": exit_code,
              "baseline_command": baseline_command, "resumed_command": resumed_command,
              "scope": "Actual interrupted process; exact model, Adam, sampler, RNG and validation at 1000 smoke updates; elapsed time and CPU latency intentionally not compared"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("PROCESS RECOVERY", report["passed"], "from", resumed_from, "->", trial, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
