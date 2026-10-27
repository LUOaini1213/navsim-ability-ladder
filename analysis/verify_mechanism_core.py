"""Verify all full-budget core artifacts without loading Torch or feature arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("M0", "M1", "M2", "M3", "M4", "M5")


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(root, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = digest(protocol_path)
    donors_hash = digest(protocol_path.parent / "donors.npz")
    full_steps = math.ceil(protocol["split_counts"]["train"]["scenes"] / protocol["training"]["batch_size"])
    expected_steps = full_steps * protocol["training"]["full_data_epochs_equivalent"]
    sources = {name: digest(ROOT / name) for name in ("agents/mechanism_model.py", "analysis/mechanism_data.py", "analysis/train_mechanism.py")}
    rows = []
    for seed in protocol["training"]["seeds"]:
        for config in CONFIGS:
            tag = f"{config}_f1_steps_s{seed}"
            run = root / tag
            settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
            complete = json.loads((run / "completed.json").read_text(encoding="utf-8"))
            history = json.loads((run / "history.json").read_text(encoding="utf-8"))
            requirements = {"protocol_sha256": protocol_hash, "donors_sha256": donors_hash, "config": config,
                            "seed": seed, "fraction": 1., "budget": "steps", "smoke_max_steps": None,
                            "training": protocol["training"], "source_sha256": sources}
            # Normalize separators in source-hash keys only; do not loosen source identity.
            settings["source_sha256"] = {name.replace("\\", "/"): value for name, value in settings["source_sha256"].items()}
            if any(settings.get(key) != value for key, value in requirements.items()):
                raise ValueError("Settings/source mismatch: "+tag)
            if (not complete["eligible_for_paper"] or complete["tag"] != tag
                or complete["optimizer_steps"] != expected_steps or not history or history[-1]["step"] != expected_steps
                or complete["n_train"] != protocol["split_counts"]["train"]["scenes"]
                or complete["n_validation"] != protocol["split_counts"]["validation"]["scenes"]):
                raise ValueError("Incomplete/incorrect training budget: "+tag)
            for key, filename in (("settings_sha256", "settings.json"), ("checkpoint_sha256", "model.pt"),
                                  ("normalization_sha256", "normalization.json")):
                if complete[key] != digest(run / filename):
                    raise ValueError("Changed artifact: "+tag+"/"+filename)
            if complete["final_validation"] != history[-1]["validation"]:
                raise ValueError("Last-checkpoint policy mismatch: "+tag)
            if any(not math.isfinite(value) for value in complete["final_validation"].values()):
                raise ValueError("Nonfinite diagnostic: "+tag)
            rows.append({**complete, "tag": tag})
    parameters = [row["parameters"] for row in rows]
    if max(parameters)/min(parameters) > 1.1:
        raise ValueError("Preregistered parameter-count tolerance violated")
    if len({row["normalization_sha256"] for row in rows}) != 1:
        raise ValueError("Full-data core normalizers differ")
    return {"passed": True, "protocol_sha256": protocol_hash, "donors_sha256": donors_hash,
            "source_sha256": sources, "optimizer_steps_per_run": expected_steps,
            "n_runs": len(rows), "parameter_ratio": max(parameters)/min(parameters), "runs": rows,
            "scope": "Hashes, sources, training budget and last-checkpoint diagnostics; inference/live extraction audited separately"}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--runs", type=Path, default=Path("E:/navsim_workspace/exp/mechanism_v2/models"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2/core_manifest.json")
    args = ap.parse_args()
    report = verify(args.runs, args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("CORE ARTIFACT AUDIT PASS", report["n_runs"], "runs", report["optimizer_steps_per_run"], "steps", flush=True)


if __name__ == "__main__":
    main()
