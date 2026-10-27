"""Export strict cached-feature predictions; confirmation requires all core models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechanism_data import CONFIGS, digest, load_features, split_indices, transform
from runtime import ROOT, configure


def verify_core(root, protocol_hash):
    for seed in range(3):
        for config in CONFIGS:
            model_dir = root / f"{config}_f1_steps_s{seed}"
            complete = json.loads((model_dir / "completed.json").read_text(encoding="utf-8"))
            settings = json.loads((model_dir / "settings.json").read_text(encoding="utf-8"))
            if (not complete["eligible_for_paper"] or settings["protocol_sha256"] != protocol_hash
                or settings["config"] != config or settings["seed"] != seed
                or settings["fraction"] != 1. or settings["budget"] != "steps"
                or complete["checkpoint_sha256"] != digest(model_dir / "model.pt")
                or complete["normalization_sha256"] != digest(model_dir / "normalization.json")
                or complete["settings_sha256"] != digest(model_dir / "settings.json")):
                raise ValueError("Unverified core checkpoint " + str(model_dir))


def export(run, protocol_path, split, out, threads=2, allow_smoke=False):
    configure()
    from navsim.agents.mechanism_model import MechanismModel
    torch.set_num_threads(threads)
    torch.set_flush_denormal(True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "completed.json").read_text(encoding="utf-8"))
    if settings["protocol_sha256"] != digest(protocol_path):
        raise ValueError("Protocol mismatch")
    for key, filename in (("checkpoint_sha256", "model.pt"), ("normalization_sha256", "normalization.json"),
                          ("settings_sha256", "settings.json")):
        if complete[key] != digest(run / filename):
            raise ValueError("Artifact hash mismatch: " + filename)
    for name, checksum in settings["source_sha256"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError("Trained model source changed: "+name)
    if not complete["eligible_for_paper"] and not allow_smoke:
        raise ValueError("Smoke models cannot enter paper evaluation")
    if split == "confirmation":
        if allow_smoke:
            raise ValueError("Smoke models cannot access confirmation")
        verify_core(run.parent, digest(protocol_path))
    source = protocol["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen features changed")
    data = load_features(source["path"])
    idx = split_indices(data, protocol, split)
    _, shuffled = CONFIGS[settings["config"]]
    donor_path = protocol_path.parent / "donors.npz"
    if digest(donor_path) != settings["donors_sha256"]:
        raise ValueError("Frozen donors changed")
    with np.load(donor_path) as archive:
        if not np.array_equal(idx, archive[split+"_indices"]):
            raise ValueError("Donor alignment mismatch")
        donors = archive[split+"_donors"] if shuffled else None
    meta = {"schema": protocol["schema"], "run": run.name, "split": split, "scenes": len(idx),
            "protocol_sha256": digest(protocol_path), "checkpoint_sha256": complete["checkpoint_sha256"],
            "normalization_sha256": complete["normalization_sha256"], "training_settings_sha256": complete["settings_sha256"],
            "donors_sha256": settings["donors_sha256"], "source_sha256": settings["source_sha256"],
            "features_sha256": source["sha256"], "eligible_for_paper": complete["eligible_for_paper"],
            "fallbacks": 0, "execution": "strict frozen-feature inference; input extraction failures excluded by historical feature archive and separately audited"}
    receipt_path = out.with_suffix(".completed.json")
    if out.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["meta"] != meta or receipt["predictions_sha256"] != digest(out):
            raise ValueError("Existing prediction provenance changed; use a new output path")
        with np.load(out, allow_pickle=False) as previous:
            if (not np.array_equal(previous["token"], data["token"][idx])
                or not np.array_equal(previous["log_name"], data["log_name"][idx])
                or previous["poses"].shape != (len(idx), 8, 3) or not np.isfinite(previous["poses"]).all()
                or json.loads(str(previous["meta"])) != meta):
                raise ValueError("Existing prediction coverage/content invalid")
        print("verified exported", run.name, split, "->", out, flush=True)
        return meta
    norm = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
    context, objects = transform(data, idx, norm, donors)
    checkpoint = torch.load(run / "model.pt", map_location="cpu")
    model = MechanismModel(checkpoint["mode"], checkpoint["hidden"], checkpoint["layers"], checkpoint["object_hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(idx), 512):
            predictions.append(model(torch.from_numpy(context[offset:offset+512]),
                                     torch.from_numpy(objects[offset:offset+512])).numpy().reshape(-1, 8, 3))
    poses = np.concatenate(predictions)
    if not np.isfinite(poses).all():
        raise FloatingPointError("No silent fallback: nonfinite output")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, token=data["token"][idx], log_name=data["log_name"][idx], poses=poses,
                        meta=np.array(json.dumps(meta)))
    temporary.replace(out)
    receipt_tmp = receipt_path.with_suffix(".json.tmp")
    receipt_tmp.write_text(json.dumps({"meta": meta, "predictions_sha256": digest(out)}, indent=2)+"\n", encoding="utf-8")
    receipt_tmp.replace(receipt_path)
    print(json.dumps(meta), "->", out, flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--split", choices=["validation", "confirmation", "navtest"], default="validation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow-smoke", action="store_true")
    args = ap.parse_args()
    export(args.run, args.protocol, args.split, args.out, args.threads, args.allow_smoke)


if __name__ == "__main__":
    main()
