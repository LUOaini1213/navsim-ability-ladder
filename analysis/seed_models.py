"""Materialize and audit a gated seed extension, preserving the original core."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from seed_followup import extension_protocol, freeze, matrix, verify_gate
from verify_mechanism_core import CONFIGS, digest, verify

VALIDATION_METRICS = {"mixed_l1", "xy_l1_m", "heading_l1_rad", "ade_m", "fde_m"}


def write_new_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def input_signature(base_path, policy_path, registration_path, gate):
    return {"kind": "activated-five-seed-training-protocol-v1",
            "base_protocol_sha256": digest(base_path),
            "base_donors_sha256": digest(base_path.parent / "donors.npz"),
            "original_followup_policy_sha256": digest(policy_path),
            "registration_sha256": digest(registration_path),
            "core_gate_sha256": gate["core_gate_sha256"],
            "activation_source_sha256": digest(Path(__file__)),
            "trigger_contrasts": gate["trigger_contrasts"],
            "training_protocol_materialized": True,
            "actual_added_models_verified": False,
            "scope": "Input materialization only; training, confirmation access, inference, scores and five-seed analysis remain"}


def verify_inputs(base_path, policy_path, registration_path, results, core):
    registration = freeze(registration_path, base_path, policy_path, results)
    gate = verify_gate(registration_path, base_path, policy_path, results, core)
    if not gate["required"]:
        raise ValueError("Five-seed extension is not triggered")
    folder = registration_path.parent / "active"
    protocol_path = folder / "protocol.json"
    if folder.is_symlink() or any((folder / name).is_symlink() for name in ("protocol.json", "donors.npz", "completed.json")):
        raise ValueError("Seed extension output must not be a symbolic link")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    complete = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    expected = input_signature(base_path, policy_path, registration_path, gate)
    if any(complete.get(key) != value for key, value in expected.items()):
        raise ValueError("Seed activation lineage/source changed")
    if (protocol != extension_protocol(json.loads(base_path.read_text(encoding="utf-8")), registration)
            or complete["protocol_sha256"] != digest(protocol_path)
            or complete["donors_sha256"] != digest(folder / "donors.npz")
            or complete["donors_sha256"] != registration["base_donors_sha256"]):
        raise ValueError("Seed protocol/cohorts/donors changed")
    return protocol_path, protocol, registration, gate


def activate(base_models, base_path, policy_path, registration_path, results):
    """Actual-score-gated fresh copy; never overwrite or repair a partial copy."""
    if not registration_path.exists():
        raise ValueError("Prospective seed registration must already exist")
    core = verify(base_models, base_path)
    registration = freeze(registration_path, base_path, policy_path, results)
    gate = verify_gate(registration_path, base_path, policy_path, results, core)
    if not gate["required"]:
        return None, core, gate
    folder = registration_path.parent / "active"
    if folder.exists() or folder.is_symlink():
        path, _, _, _ = verify_inputs(base_path, policy_path, registration_path, results, core)
        return path, core, gate
    before = input_signature(base_path, policy_path, registration_path, gate)
    folder.mkdir(exist_ok=False)
    protocol_path = folder / "protocol.json"
    write_new_json(protocol_path, extension_protocol(json.loads(base_path.read_text(encoding="utf-8")), registration))
    with (base_path.parent / "donors.npz").open("rb") as source, (folder / "donors.npz").open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if before != input_signature(base_path, policy_path, registration_path, gate):
        raise ValueError("Base/registration changed during activation; preserve the partial output")
    donor_hash = digest(folder / "donors.npz")
    if donor_hash != before["base_donors_sha256"]:
        raise ValueError("Copied donors differ; preserve the partial output")
    write_new_json(folder / "completed.json", dict(before,
                   protocol_sha256=digest(protocol_path), donors_sha256=donor_hash))
    verify_inputs(base_path, policy_path, registration_path, results, core)
    return protocol_path, core, gate


def verify_one(folder, row, protocol_path, base_protocol, core):
    expected_rows = matrix(base_protocol)
    if row not in expected_rows:
        raise ValueError("Unregistered added seed/model/budget")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run = folder / row["tag"]
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    requirements = {"protocol_sha256": digest(protocol_path),
                    "donors_sha256": digest(protocol_path.parent / "donors.npz"),
                    "config": row["config"], "seed": row["seed"], "fraction": 1.,
                    "budget": "steps", "smoke_max_steps": None, "training": protocol["training"]}
    if any(settings.get(key) != value for key, value in requirements.items()):
        raise ValueError("Added-seed settings/budget changed")
    if {name.replace("\\", "/"): value for name, value in settings["source_sha256"].items()} != core["source_sha256"]:
        raise ValueError("Added seed did not use the original frozen training implementation")
    anchor = next(model for model in core["runs"] if model["tag"] == f"{row['config']}_f1_steps_s0")
    final = complete["final_validation"]
    if (complete.get("eligible_for_paper") is not True or complete["tag"] != row["tag"]
            or complete["optimizer_steps"] != row["optimizer_steps"] or not history
            or complete["n_train"] != row["n_train"]
            or complete["n_validation"] != base_protocol["split_counts"]["validation"]["scenes"]
            or complete["parameters"] != anchor["parameters"]
            or history[-1]["step"] != row["optimizer_steps"] or history[-1]["validation"] != final
            or set(final) != VALIDATION_METRICS
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) and value >= 0 for value in final.values())):
        raise ValueError("Incomplete/smoke/wrong-budget added-seed model")
    for key, name in (("settings_sha256", "settings.json"), ("checkpoint_sha256", "model.pt"),
                      ("normalization_sha256", "normalization.json")):
        if digest(run / name) != complete[key]:
            raise ValueError("Added-seed model artifact changed")
    if complete["normalization_sha256"] != anchor["normalization_sha256"]:
        raise ValueError("Added-seed training-only normalizer differs from core")
    return dict(complete, original_config=row["config"], seed=row["seed"],
                history_sha256=digest(run / "history.json"))


def verify_all(base_models, added_models, base_path, policy_path, registration_path, results):
    """Required before ANY added confirmation inference; all twelve must exist."""
    core = verify(base_models, base_path)
    protocol_path, protocol, registration, gate = verify_inputs(base_path, policy_path, registration_path, results, core)
    base_protocol = json.loads(base_path.read_text(encoding="utf-8"))
    added = {row["tag"]: verify_one(added_models, row, protocol_path, base_protocol, core)
             for row in registration["matrix"]}
    if len(added) != 12 or {row["seed"] for row in added.values()} != {3, 4}:
        raise ValueError("All twelve added models required before confirmation inference")
    return core, protocol, added, gate
