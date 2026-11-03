"""Audit six same-city models and their compatibility with unchanged core anchors."""
from __future__ import annotations

import json
import math
from pathlib import Path

from verify_mechanism_core import ROOT, digest, verify

CONDITIONS = {"C1": "M1", "C2": "M2", "C4": "M4", "C5": "M5"}
VALIDATION_METRICS = {"mixed_l1", "xy_l1_m", "heading_l1_rad", "ade_m", "fde_m"}


def matrix():
    return [{"condition": condition, "config": config, "seed": seed, "tag": f"{config}_f1_steps_s{seed}"}
            for seed in (0, 1, 2) for condition, config in (("C2", "M2"), ("C5", "M5"))]


def verify_inputs(base_path, city_path, audit_path):
    base = json.loads(base_path.read_text(encoding="utf-8"))
    city = json.loads(city_path.read_text(encoding="utf-8"))
    complete = json.loads((city_path.parent / "completed.json").read_text(encoding="utf-8"))
    design = json.loads((city_path.parent / "design.json").read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (not audit["passed"] or audit["outcomes_used"] or complete["outcomes_read"]
        or complete["protocol_sha256"] != digest(city_path) or audit["city_protocol_sha256"] != digest(city_path)
        or complete["donors_sha256"] != digest(city_path.parent / "donors.npz") or audit["donors_sha256"] != complete["donors_sha256"]
        or complete["design_sha256"] != digest(city_path.parent / "design.json")
        or audit["design_sha256"] != complete["design_sha256"] or design["outcomes_used_for_design"]
        or city["extension"]["base_protocol_sha256"] != digest(base_path)
        or design["base_protocol_sha256"] != digest(base_path)):
        raise ValueError("Same-city input/audit lineage mismatch")
    if (complete["coverage"] != city["extension"]["coverage"]
        or city["sources"] != base["sources"] or city["fractions"] != base["fractions"] or city["training"] != base["training"]
        or city["split_sessions"]["train"] != base["split_sessions"]["train"]
        or city["split_counts"]["train"] != base["split_counts"]["train"]
        or city["extension"]["seeds"] != [0, 1, 2]
        or city["training"]["seeds"] != [0, 1, 2]
        or city["extension"]["contrasts"] != ["C1-C2", "C4-C5", "C4-C5-C1+C2"]):
        raise ValueError("Same-city training cohort/budget/design changed")
    generator = digest(ROOT / "analysis/prepare_city_control.py")
    if generator != design["generator_sha256"] or generator != complete["generator_sha256"]:
        raise ValueError("Pre-outcome city generator changed")
    for name, checksum in design["input_helpers_sha256"].items():
        if digest(ROOT / "analysis" / name) != checksum:
            raise ValueError("Bound city input helper changed")
    if audit["audit_source_sha256"] != digest(ROOT / "analysis/audit_city_inputs.py"):
        raise ValueError("City audit source changed")
    for split, row in audit["partitions"].items():
        expected = city["extension"]["coverage"][split]
        if (row["same_city_fraction"] != 1. or row["different_session_fraction"] != 1.
            or row["n_scenes"] != expected["eligible_scenes"]
            or row["eligible_token_order_sha256"] != expected["eligible_token_order_sha256"]):
            raise ValueError("City invariant/coverage audit mismatch")
    if set(audit["partitions"]) != {"train", "validation", "confirmation", "navtest"}:
        raise ValueError("Incomplete city input audit")
    return city


def verify_one(folder, row, protocol, protocol_path, core):
    run = folder / row["tag"]
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    requirements = {"protocol_sha256": digest(protocol_path), "donors_sha256": digest(protocol_path.parent / "donors.npz"),
                    "config": row["config"], "seed": row["seed"], "fraction": 1., "budget": "steps", "smoke_max_steps": None,
                    "training": protocol["training"]}
    if any(settings.get(key) != value for key, value in requirements.items()):
        raise ValueError("Wrong city training settings: " + row["tag"])
    if {name.replace("\\", "/"): value for name, value in settings["source_sha256"].items()} != core["source_sha256"]:
        raise ValueError("City model did not use the frozen core training implementation")
    expected_steps = math.ceil(protocol["split_counts"]["train"]["scenes"] / protocol["training"]["batch_size"]) * protocol["training"]["full_data_epochs_equivalent"]
    anchor = next(model for model in core["runs"] if model["tag"] == row["tag"])
    final = complete["final_validation"]
    if (not complete["eligible_for_paper"] or complete["tag"] != row["tag"] or complete["optimizer_steps"] != expected_steps
        or complete["n_train"] != protocol["split_counts"]["train"]["scenes"]
        or complete["n_validation"] != protocol["split_counts"]["validation"]["scenes"]
        or complete["parameters"] != anchor["parameters"] or not history or history[-1]["step"] != expected_steps
        or history[-1]["validation"] != final or set(final) != VALIDATION_METRICS
        or set(anchor["final_validation"]) != VALIDATION_METRICS
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) and value >= 0 for value in final.values())):
        raise ValueError("Incomplete/smoke/wrong-budget city model: " + row["tag"])
    for key, filename in (("checkpoint_sha256", "model.pt"), ("normalization_sha256", "normalization.json"), ("settings_sha256", "settings.json")):
        if digest(run / filename) != complete[key]:
            raise ValueError("City model artifact changed: " + row["tag"])
    if complete["normalization_sha256"] != anchor["normalization_sha256"]:
        raise ValueError("City/real anchor training-only normalizers differ")
    return dict(complete, condition=row["condition"], original_config=row["config"], seed=row["seed"])


def verify_anchor_metadata(meta, model, core, base_path, source, n_scenes, tag, split):
    """Bind reused real predictions to the actual full core artifacts and inputs."""
    expected = {"eligible_for_paper": True, "fallbacks": 0, "run": tag, "split": split,
                "scenes": n_scenes, "protocol_sha256": digest(base_path),
                "donors_sha256": core["donors_sha256"], "checkpoint_sha256": model["checkpoint_sha256"],
                "normalization_sha256": model["normalization_sha256"], "training_settings_sha256": model["settings_sha256"],
                "features_sha256": source["sha256"]}
    sources = {name.replace("\\", "/"): value for name, value in meta.get("source_sha256", {}).items()}
    if any(meta.get(key) != value for key, value in expected.items()) or sources != core["source_sha256"]:
        raise ValueError("Original anchor model/protocol/source metadata mismatch")


def verify_all(base_models, city_models, base_path, city_path, audit_path):
    core = verify(base_models, base_path)
    city = verify_inputs(base_path, city_path, audit_path)
    auxiliary = {row["condition"] + "_s" + str(row["seed"]): verify_one(city_models, row, city, city_path, core) for row in matrix()}
    if len(auxiliary) != 6:
        raise ValueError("All six city models required before city confirmation inference")
    return core, city, auxiliary
