"""Train the registered 24 small-data follow-ups only after a verified curve gate.

This runner covers training and artifact verification, not scoring or goal completion.
It shares the existing heavy-job lock and cannot overlap the core evaluator.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from mechanism_gates import decide
from resource_guard import MemoryInterrupted, stop_owned_child, wait_for_headroom, wait_guarded
from run_mechanism_evaluation import acquire
from verify_mechanism_core import CONFIGS, ROOT, digest, verify


def matrix(protocol, policy):
    rows = []
    for budget, name in (("steps", "fixed_updates"), ("epochs", "epoch_sensitivity")):
        branch = policy["learning_curve"][name]
        if branch != {"configs": ["M1", "M4"], "fractions": [.1, .3], "seeds": [0, 1, 2]}:
            raise ValueError("Unregistered learning-curve matrix")
        for fraction in branch["fractions"]:
            n_train = protocol["fractions"][str(float(fraction))]["n_scenes"]
            full_n = protocol["split_counts"]["train"]["scenes"]
            steps = math.ceil((full_n if budget == "steps" else n_train)/protocol["training"]["batch_size"])
            steps *= protocol["training"]["full_data_epochs_equivalent"]
            for seed in branch["seeds"]:
                for config in branch["configs"]:
                    rows.append({"tag": f"{config}_f{fraction:g}_{budget}_s{seed}", "config": config,
                                 "fraction": fraction, "budget": budget, "seed": seed,
                                 "n_train": n_train, "optimizer_steps": steps})
    if len(rows) != 24 or len({row["tag"] for row in rows}) != 24:
        raise ValueError("Exactly 24 distinct smaller-data jobs required")
    return rows


def verified_gate(report_path, scores, gate_path, policy_path, core):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected = {row["tag"]: row for row in core["runs"]}
    if (report["protocol_sha256"] != core["protocol_sha256"] or set(report["provenance"]) != set(expected)
        or {row["n_validation"] for row in expected.values()} != {report["n_scenes"]}):
        raise ValueError("Core/report mismatch")
    calculated = decide(report, None, policy)
    calculated.update(policy_sha256=digest(policy_path), validation_report_sha256=digest(report_path), confirmation_report_sha256=None)
    if gate != calculated:
        raise ValueError("Curve gate is not the hash-bound frozen validation decision")
    complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
    settings = json.loads((scores / "settings.json").read_text(encoding="utf-8"))
    if (not complete["eligible_for_paper"] or complete["split"] != "validation" or settings["limit"] is not None
        or complete["settings_sha256"] != digest(scores / "settings.json")
        or report["score_settings_sha256"] != complete["settings_sha256"]
        or set(complete["runs"]) != set(expected) or set(settings["runs"]) != set(expected)
        or complete.get("final_invalid_scores", complete.get("failures", 0)) != 0):
        raise ValueError("Incomplete/smoke validation score archive")
    means, reference = {}, None
    for tag, trained in expected.items():
        provenance = report["provenance"][tag]
        meta = settings["runs"][tag]["meta"]
        if (not meta["eligible_for_paper"] or meta["protocol_sha256"] != core["protocol_sha256"]
            or meta["run"] != tag or meta["split"] != "validation" or meta["fallbacks"] != 0
            or meta["scenes"] != report["n_scenes"]
            or {key.replace("\\", "/"): value for key, value in meta["source_sha256"].items()} != core["source_sha256"]):
            raise ValueError("Validation prediction provenance mismatch")
        for name in ("checkpoint_sha256", "normalization_sha256", "training_settings_sha256", "donors_sha256"):
            trained_name = "settings_sha256" if name == "training_settings_sha256" else name
            wanted = core["donors_sha256"] if name == "donors_sha256" else trained[trained_name]
            if provenance[name] != wanted or meta[name] != wanted:
                raise ValueError("Trained-model provenance changed: "+tag)
        path = scores / (tag+".csv.gz")
        if digest(path) != provenance["csv_sha256"] or digest(path) != complete["runs"][tag]["csv_sha256"]:
            raise ValueError("Score CSV changed: "+tag)
        rows, total = {}, 0.
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                value = float(row["score"])
                if row["token"] in rows or row["valid"].lower() not in ("true", "1") or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid/duplicate validation score")
                rows[row["token"]] = row["log_name"]
                total += value
        if len(rows) != report["n_scenes"] or len(rows) != complete["runs"][tag]["n"] or not rows:
            raise ValueError("Incomplete validation score coverage")
        if reference is None:
            reference = rows
        elif rows != reference:
            raise ValueError("Unpaired validation tokens/logs")
        means[tag] = total/len(rows)
    for label in ("M1-M0", "M4-M1", "M4-M5"):
        left, right = label.split("-")
        row = report["summary"]["contrasts"][label]
        effects = [means[f"{left}_f1_steps_s{seed}"]-means[f"{right}_f1_steps_s{seed}"] for seed in range(3)]
        if (abs(sum(effects)/3-row["metric_effects"]["score"]) > 1e-10
            or any(abs(effect-row["seed_effects"][str(seed)]) > 1e-10 for seed, effect in enumerate(effects))):
            raise ValueError("Gate-driving effect differs from actual score CSVs")
    return gate


def verify_run(folder, row, protocol, protocol_path):
    run = folder / row["tag"]
    complete = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    required = {key: row[key] for key in ("config", "seed", "fraction", "budget")}
    required.update(protocol_sha256=digest(protocol_path), donors_sha256=digest(protocol_path.parent / "donors.npz"),
                    smoke_max_steps=None, training=protocol["training"])
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError("Follow-up settings mismatch: "+row["tag"])
    sources = {key.replace("\\", "/"): value for key, value in settings["source_sha256"].items()}
    expected_sources = {name: digest(ROOT / name) for name in ("agents/mechanism_model.py", "analysis/mechanism_data.py", "analysis/train_mechanism.py")}
    if sources != expected_sources:
        raise ValueError("Follow-up source changed")
    if (not complete["eligible_for_paper"] or complete["tag"] != row["tag"] or complete["n_train"] != row["n_train"]
        or complete["n_validation"] != protocol["split_counts"]["validation"]["scenes"]
        or complete["optimizer_steps"] != row["optimizer_steps"] or not history
        or history[-1]["step"] != row["optimizer_steps"] or history[-1]["validation"] != complete["final_validation"]
        or any(not math.isfinite(value) for value in complete["final_validation"].values())):
        raise ValueError("Incomplete/smoke/wrong-budget follow-up")
    for key, filename in (("settings_sha256", "settings.json"), ("checkpoint_sha256", "model.pt"), ("normalization_sha256", "normalization.json")):
        if complete[key] != digest(run / filename):
            raise ValueError("Follow-up artifact changed")
    return complete


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--min-free-mb", type=int, default=2400)
    ap.add_argument("--wait-for-evaluator", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    if args.threads != 2:
        raise ValueError("Registered CPU-thread budget is two")
    if args.min_free_mb <= 1000:
        raise ValueError("Launch headroom must exceed the running critical threshold")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    results = ROOT / "results/mechanism_v2"
    model_root = args.workspace / "exp/mechanism_v2/models"
    lock = acquire(args.workspace / "exp/mechanism_v2/core.lock", wait=args.wait_for_evaluator,
                   on_wait=lambda: print("CURVES WAIT: the heavy-job lock is still owned", flush=True))
    state_path = results / "curve_training_status.json"
    state = {"runner_pid": os.getpid(), "scope": "Training/artifacts only; driving-score curves and goal completion remain separate",
             "jobs": [], "attempts": [], "memory_events": []}
    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
        temporary.replace(state_path)
    try:
        core = verify(model_root, args.protocol)
        policy_path = args.protocol.parent / "followup_policy.json"
        gate = verified_gate(results / "analysis/validation/report.json", results / "scores/validation",
                             results / "gate_validation.json", policy_path, core)
        state.update(gate_sha256=digest(results / "gate_validation.json"), policy_sha256=digest(policy_path))
        if not gate["learning_curve_required"]:
            state.update(status="not_triggered", reason="Frozen validation gate did not trigger data-budget experiments")
            save()
            print("CURVES NOT TRIGGERED; omission must be disclosed", flush=True)
            return
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        jobs = matrix(protocol, json.loads(policy_path.read_text(encoding="utf-8")))
        state.update(status="planned" if args.plan_only else "running", matrix=jobs,
                     frozen_sha256={str(path): digest(path) for path in (args.protocol, args.protocol.parent / "donors.npz", policy_path,
                         ROOT / "agents/mechanism_model.py", ROOT / "analysis/mechanism_data.py", ROOT / "analysis/train_mechanism.py",
                         ROOT / "analysis/resource_guard.py", Path(__file__))})
        save()
        if args.plan_only:
            print("VERIFIED CURVE PLAN ONLY; no new model trained", flush=True)
            return
        logs = args.workspace / "exp/mechanism_v2/curve_logs"
        logs.mkdir(parents=True, exist_ok=True)
        for row in jobs:
            if any(digest(path) != checksum for path, checksum in state["frozen_sha256"].items()):
                raise ValueError("Frozen follow-up sources/policy changed")
            command = [sys.executable, "-u", str(ROOT / "analysis/train_mechanism.py"), "--workspace", str(args.workspace),
                       "--protocol", str(args.protocol), "--config", row["config"], "--seed", str(row["seed"]),
                       "--fraction", str(row["fraction"]), "--budget", row["budget"], "--threads", "2", "--out-root", str(model_root)]
            flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
            path = logs / (row["tag"]+".log")
            def event(details):
                state["memory_events"].append(dict(details, tag=row["tag"]))
                state.update(status=details["kind"], current_job=row["tag"])
                save()
                print("CURVE MEMORY", row["tag"], details, flush=True)
            attempt = 0
            while True:
                wait_for_headroom(args.min_free_mb, on_event=event)
                if any(digest(path) != checksum for path, checksum in state["frozen_sha256"].items()):
                    raise ValueError("Frozen follow-up source changed while waiting")
                attempt += 1
                interruption = None
                with open(path, "a", encoding="utf-8") as stream:
                    child = subprocess.Popen(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                    state.update(status="running", current_job=row["tag"], child_pid=child.pid, log_path=str(path))
                    save()
                    print("CURVE START", row["tag"], "attempt", attempt, "pid", child.pid, flush=True)
                    try:
                        code = wait_guarded(child, on_event=event)
                    except MemoryInterrupted as error:
                        interruption = error.event
                        code = interruption["exit_code"]
                    except BaseException:
                        stop_owned_child(child)
                        raise
                state["attempts"].append({"tag": row["tag"], "attempt": attempt, "exit_code": code,
                                          "memory_interruption": interruption, "log_path": str(path)})
                state["child_pid"] = None
                save()
                if interruption is not None:
                    # Atomic model/Adam/RNG/sampler recovery was tested in an
                    # actual stopped process. Never treat this attempt as done.
                    continue
                if code != 0:
                    raise RuntimeError("Follow-up failed; no silent skip/restart: "+row["tag"])
                break
            complete = verify_run(model_root, row, protocol, args.protocol)
            state["jobs"].append({"tag": row["tag"], "verified": complete})
            save()
        normalizers = {fraction: {item["verified"]["normalization_sha256"] for item in state["jobs"]
                                 if next(row for row in jobs if row["tag"] == item["tag"])["fraction"] == fraction} for fraction in (.1, .3)}
        if any(len(hashes) != 1 for hashes in normalizers.values()):
            raise ValueError("Same-subset normalizers differ across budgets/seeds/configs")
        state.update(status="training_completed", child_pid=None, subset_normalizers={str(key): next(iter(value)) for key, value in normalizers.items()})
        save()
        print("24 CURVE MODELS VERIFIED; scoring/analysis/figures still required", flush=True)
    except BaseException as error:
        state.update(status="failed_or_interrupted", exception=repr(error))
        save()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
