"""Score all registered curve models only after their full training/gate audit."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from analyze_mechanism_curves import verify_training
from resource_guard import MemoryInterrupted, stop_owned_child, wait_for_headroom, wait_guarded
from run_mechanism_curves import verified_gate
from run_mechanism_evaluation import acquire, upstream_binding
from verify_mechanism_core import ROOT, digest, verify


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--wait-for-heavy-job", action="store_true")
    args = ap.parse_args()
    for key, value in {"NAVSIM_WORKSPACE": args.workspace, "NAVSIM_DEVKIT_ROOT": args.workspace / "navsim",
                       "OPENSCENE_DATA_ROOT": args.workspace / "dataset", "NUPLAN_MAPS_ROOT": args.workspace / "dataset/maps",
                       "NAVSIM_EXP_ROOT": args.workspace / "exp"}.items():
        os.environ[key] = str(value)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    results = ROOT / "results/mechanism_v2"
    root = args.workspace / "exp/mechanism_v2"
    models = root / "models"
    policy = args.protocol.parent / "followup_policy.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + str(os.getpid())
    attempt_path = results / "run_attempts" / ("curve_evaluation_" + stamp + ".json")
    state_path = results / "curve_evaluation_status.json"
    state = {"status": "waiting_for_heavy_job", "runner_pid": os.getpid(), "jobs": [], "memory_events": [],
             "attempt_receipt": str(attempt_path), "scope": "Curve scoring/analysis/rendered figures only; visual QA, other extensions and final goal audit remain"}
    def save():
        for path in (attempt_path, state_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
    save()
    lock = acquire(root / "core.lock", wait=args.wait_for_heavy_job,
                   on_wait=lambda: print("CURVE EVALUATION WAIT: shared heavy-job lock owned", flush=True))
    try:
        core = verify(models, args.protocol)
        gate = verified_gate(results / "analysis/validation/report.json", results / "scores/validation",
                             results / "gate_validation.json", policy, core)
        state["gate_sha256"] = digest(results / "gate_validation.json")
        if not gate["learning_curve_required"]:
            state.update(status="not_triggered", reason="Verified frozen validation gate did not require curves")
            save()
            print("CURVE EVALUATION NOT TRIGGERED; record omission", flush=True)
            return
        _, trained = verify_training(models, args.protocol, policy)
        files = [ROOT / "analysis" / name for name in ("analyze_mechanism_curves.py", "run_mechanism_curve_evaluation.py",
                 "analyze_mechanism.py", "mechanism_gates.py", "verify_mechanism_core.py", "mechanism_data.py",
                 "predict_mechanism.py", "score_mechanism.py", "mechanism_scoring.py", "run_mechanism_curves.py",
                 "run_mechanism_evaluation.py", "resource_guard.py", "plot_followups.py")]
        files.extend([args.protocol, args.protocol.parent / "donors.npz", policy,
                      results / "gate_validation.json", results / "analysis/validation/report.json"])
        binding = {str(path): digest(path) for path in files}
        snapshot = results / "runtime_snapshot/runtime.json"
        binding.update(upstream_binding(args.workspace, snapshot))
        binding[str(snapshot)] = digest(snapshot)
        state.update(status="training_verified", n_models=len(trained), frozen_sha256=binding)
        save()
        logs = root / "curve_evaluation_logs"
        logs.mkdir(parents=True, exist_ok=True)
        def stage(name, script, arguments, resumable=False):
            attempts = 0
            def event(details):
                state["memory_events"].append(dict(details, stage=name, observed_at_utc=datetime.now(timezone.utc).isoformat()))
                state.update(status=details["kind"], stage=name)
                save()
                print("CURVE EVALUATION MEMORY", name, details, flush=True)
            while True:
                wait_for_headroom(on_event=event)
                if any(digest(path) != value for path, value in binding.items()):
                    raise ValueError("Curve evaluation source/policy changed")
                attempts += 1
                path = logs / (name + ".log")
                flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
                interrupted = None
                with open(path, "a", encoding="utf-8") as stream:
                    child = subprocess.Popen([sys.executable, "-u", str(ROOT / "analysis" / script), *map(str, arguments)],
                                             cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                    state.update(status="running", stage=name, child_pid=child.pid, log_path=str(path))
                    save()
                    print("CURVE EVALUATION START", name, "attempt", attempts, "pid", child.pid, flush=True)
                    try:
                        code = wait_guarded(child, on_event=event)
                    except MemoryInterrupted as error:
                        interrupted, code = error.event, error.event["exit_code"]
                    except BaseException:
                        stop_owned_child(child)
                        raise
                state["jobs"].append({"stage": name, "attempt": attempts, "exit_code": code,
                                      "memory_interruption": interrupted, "log_path": str(path)})
                state["child_pid"] = None
                save()
                if interrupted is not None and resumable:
                    continue
                if interrupted is not None or code != 0:
                    raise RuntimeError("Curve stage failed/interrupted; no silent skip: " + name)
                print("CURVE EVALUATION DONE", name, flush=True)
                return
        for split in ("validation", "confirmation", "navtest"):
            # Recheck all 24 follow-ups, not just the model about to be exported.
            # This is the additional guard beyond the base 18-model exporter.
            _, current = verify_training(models, args.protocol, policy)
            if current != trained:
                raise ValueError("Curve artifacts changed after matrix audit")
            if split != "navtest":
                cache = json.loads((root / "metric_cache" / (split + "_completed.json")).read_text(encoding="utf-8"))
                protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
                if (cache["protocol_sha256"] != digest(args.protocol) or cache["split"] != split
                    or cache["limit"] is not None or cache["scenes"] != protocol["split_counts"][split]["scenes"]):
                    raise ValueError("Full matching split cache required for curves")
            predictions = root / "curve_predictions" / split
            paths = []
            for tag in sorted(trained):
                path = predictions / (tag + ".npz")
                stage("predict_" + split + "_" + tag, "predict_mechanism.py",
                      ["--run", models / tag, "--protocol", args.protocol, "--split", split, "--out", path, "--threads", "2"])
                paths.append(path)
            scores = results / "curve_scores" / split
            stage("score_" + split, "score_mechanism.py", ["--predictions", *paths, "--out", scores], resumable=True)
            stage("analyze_" + split, "analyze_mechanism_curves.py", ["--scores", scores, "--models", models,
                  "--protocol", args.protocol, "--workspace", args.workspace, "--out", results / "curve_analysis" / split])
            stage("figures_" + split, "plot_followups.py", ["--kind", "curves", "--workspace", args.workspace,
                  "--protocol", args.protocol, "--report", results / "curve_analysis" / split / "report.json",
                  "--scores", scores, "--out", results / "figures/curves" / split], resumable=True)
        state.update(status="curve_evaluation_completed", child_pid=None)
        save()
        print("CURVE EVALUATION COMPLETED; real figure inspection and remaining project checks still required", flush=True)
    except BaseException as error:
        state.update(status="failed_or_interrupted", exception=repr(error), child_pid=None)
        save()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
