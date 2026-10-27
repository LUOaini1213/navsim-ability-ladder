"""Separate same-city exporter: all 18 core and all six city models are required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from city_models import CONDITIONS, verify_all, verify_anchor_metadata
from mechanism_data import digest, load_features, split_indices, transform
from runtime import ROOT, configure


def select_anchor(tokens, logs, poses, wanted, wanted_logs):
    if (not len(tokens) or not len(wanted) or len(set(tokens)) != len(tokens) or len(tokens) != len(logs) or poses.shape != (len(tokens), 8, 3)
        or not np.isfinite(poses).all() or len(wanted) != len(wanted_logs) or len(set(wanted)) != len(wanted)):
        raise ValueError("Invalid anchor predictions/coverage")
    lookup = {str(token): index for index, token in enumerate(tokens)}
    if not set(map(str, wanted)).issubset(lookup):
        raise ValueError("Anchor does not contain the registered eligible city tokens")
    positions = np.array([lookup[str(token)] for token in wanted])
    if not np.array_equal(logs[positions], wanted_logs):
        raise ValueError("Anchor eligible token/log misalignment")
    return poses[positions]


def export(condition, seed, split, base_models, city_models, base_path, city_path, base_predictions, out, threads=2, workspace=None):
    if condition not in CONDITIONS or seed not in (0, 1, 2) or threads != 2 or split not in ("validation", "confirmation", "navtest"):
        raise ValueError("Unregistered city condition/seed/split/thread budget")
    core, city, auxiliary = verify_all(base_models, city_models, base_path, city_path, ROOT / "results/audit_v2/city_inputs.json")
    source = city["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen city features changed")
    with np.load(source["path"], allow_pickle=False) as archive:
        identity = {key: archive[key] for key in ("token", "log_name")}
    indices = split_indices(identity, city, split)
    with np.load(city_path.parent / "donors.npz", allow_pickle=False) as mappings:
        if not np.array_equal(indices, mappings[split + "_indices"]) or not np.array_equal(identity["token"][indices], mappings[split + "_tokens"]):
            raise ValueError("Frozen city donor/token alignment mismatch")
        donors = mappings[split + "_donors"]
    config = CONDITIONS[condition]
    original_tag = f"{config}_f1_steps_s{seed}"
    anchor = condition in ("C1", "C4")
    model = next(row for row in core["runs"] if row["tag"] == original_tag) if anchor else auxiliary[condition + "_s" + str(seed)]
    meta = {"schema": "same-city-auxiliary-predictions-v1", "run": f"{condition}_f1_steps_s{seed}", "original_run": original_tag,
            "condition": condition, "seed": seed, "split": split, "scenes": len(indices), "eligible_for_paper": True, "fallbacks": 0,
            "protocol_sha256": digest(city_path), "base_protocol_sha256": digest(base_path), "donors_sha256": digest(city_path.parent / "donors.npz"),
            "checkpoint_sha256": model["checkpoint_sha256"], "normalization_sha256": model["normalization_sha256"],
            "training_settings_sha256": model["settings_sha256"], "source_sha256": core["source_sha256"], "features_sha256": source["sha256"],
            "city_input_audit_sha256": digest(ROOT / "results/audit_v2/city_inputs.json"), "exporter_sha256": digest(Path(__file__)),
            "execution": "Same registered eligible coverage; all 18 core and six auxiliary artifacts audited; privileged frozen features, not a physical counterfactual"}
    if anchor:
        path = base_predictions / (original_tag + ".npz")
        receipt_path = path.with_suffix(".completed.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["predictions_sha256"] != digest(path):
            raise ValueError("Original real-anchor prediction archive changed")
        with np.load(path, allow_pickle=False) as archive:
            original_meta = json.loads(str(archive["meta"]))
            tokens, logs, poses = archive["token"], archive["log_name"], archive["poses"]
        base_indices = split_indices(identity, json.loads(base_path.read_text(encoding="utf-8")), split)
        verify_anchor_metadata(original_meta, model, core, base_path, source, len(base_indices), original_tag, split)
        if (original_meta != receipt["meta"]
            or not np.array_equal(tokens, identity["token"][base_indices]) or not np.array_equal(logs, identity["log_name"][base_indices])):
            raise ValueError("Original anchor model/protocol/coverage mismatch")
        meta.update(anchor_original_predictions_sha256=digest(path), anchor_original_receipt_sha256=digest(receipt_path),
                    anchor_original_metadata=original_meta, anchor_filter="Input-only frozen eligible cohort, never outcome/top-scene filtering")
        poses = select_anchor(tokens, logs, poses, identity["token"][indices], identity["log_name"][indices])
    receipt_path = out.with_suffix(".completed.json")
    if out.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["meta"] != meta or receipt["predictions_sha256"] != digest(out):
            raise ValueError("Existing city prediction provenance changed; use a new output path")
        with np.load(out, allow_pickle=False) as archive:
            if (json.loads(str(archive["meta"])) != meta or not np.array_equal(archive["token"], identity["token"][indices])
                or not np.array_equal(archive["log_name"], identity["log_name"][indices])
                or archive["poses"].shape != (len(indices), 8, 3) or not np.isfinite(archive["poses"]).all()):
                raise ValueError("Existing city prediction content/coverage invalid")
        print("CITY PREDICTIONS VERIFIED", meta["run"], split, flush=True)
        return meta
    if out.exists():
        # An interruption can leave the new NPZ without its atomic receipt.
        # Only recompute an archive declaring this exact owned-output signature.
        with np.load(out, allow_pickle=False) as previous:
            if json.loads(str(previous["meta"])) != meta:
                raise ValueError("Refusing to overwrite an unrelated unreceipted archive")
    if not anchor:
        import torch
        configure(workspace)
        from navsim.agents.mechanism_model import MechanismModel
        torch.set_num_threads(threads)
        torch.set_flush_denormal(True)
        data = load_features(source["path"])
        run = city_models / original_tag
        norm = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
        context, objects = transform(data, indices, norm, donors)
        checkpoint = torch.load(run / "model.pt", map_location="cpu")
        network = MechanismModel(checkpoint["mode"], checkpoint["hidden"], checkpoint["layers"], checkpoint["object_hidden"])
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        with torch.no_grad():
            rows = [network(torch.from_numpy(context[offset:offset + 512]), torch.from_numpy(objects[offset:offset + 512])).numpy().reshape(-1, 8, 3)
                    for offset in range(0, len(indices), 512)]
        poses = np.concatenate(rows)
    if poses.shape != (len(indices), 8, 3) or not np.isfinite(poses).all():
        raise FloatingPointError("Strict city inference: no nonfinite/shape fallback")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, token=identity["token"][indices], log_name=identity["log_name"][indices], poses=poses, meta=np.array(json.dumps(meta)))
    temporary.replace(out)
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"meta": meta, "predictions_sha256": digest(out)}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    print("CITY PREDICTIONS EXPORTED", meta["run"], split, len(indices), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["validation", "confirmation", "navtest"], required=True)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = ROOT / "results/mechanism_v2"
    root = args.workspace / "exp/mechanism_v2"
    export(args.condition, args.seed, args.split, root / "models", root / "city_models", results / "protocol/protocol.json",
           results / "city_protocol/protocol.json", root / "predictions" / args.split, args.out, workspace=args.workspace)


if __name__ == "__main__":
    main()
