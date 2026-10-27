"""Strict added-seed inference, gated by all eighteen core plus twelve added fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seed_models import verify_all, write_new_json
from verify_mechanism_core import CONFIGS, ROOT, digest

SPLITS = ("validation", "confirmation", "navtest")


def metadata(config, seed, split, count, core, added, base_path, protocol_path, registration_path):
    if config not in CONFIGS or seed not in (3, 4) or split not in SPLITS or count <= 0:
        raise ValueError("Unregistered added-seed prediction")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = protocol["sources"]["navtest" if split == "navtest" else "navtrain"]
    tag = f"{config}_f1_steps_s{seed}"
    model = added[tag]
    return {"schema": protocol["schema"], "run": tag, "split": split, "scenes": count,
            "protocol_sha256": digest(protocol_path), "base_protocol_sha256": digest(base_path),
            "registration_sha256": digest(registration_path),
            "checkpoint_sha256": model["checkpoint_sha256"],
            "normalization_sha256": model["normalization_sha256"],
            "training_settings_sha256": model["settings_sha256"],
            "donors_sha256": core["donors_sha256"], "source_sha256": core["source_sha256"],
            "prediction_wrapper_sha256": digest(Path(__file__)),
            "features_sha256": source["sha256"], "eligible_for_paper": True, "fallbacks": 0,
            "execution": "Strict frozen-feature added-seed inference after all thirty full-budget artifacts passed; no silent fallback"}


def check_existing(out, meta, tokens, logs):
    import numpy as np
    receipt = out.with_suffix(".completed.json")
    if out.exists() != receipt.exists():
        raise ValueError("Partial seed prediction/receipt preserved; use a reviewed fresh output path")
    if not out.exists():
        return False
    bound = json.loads(receipt.read_text(encoding="utf-8"))
    if bound["meta"] != meta or bound["predictions_sha256"] != digest(out):
        raise ValueError("Existing seed prediction provenance changed")
    with np.load(out, allow_pickle=False) as archive:
        if (not np.array_equal(archive["token"], tokens) or not np.array_equal(archive["log_name"], logs)
                or archive["poses"].shape != (len(tokens), 8, 3)
                or not np.isfinite(archive["poses"]).all()
                or json.loads(str(archive["meta"])) != meta):
            raise ValueError("Existing seed prediction content/coverage changed")
    return True


def export(config, seed, split, workspace, results, out):
    if config not in CONFIGS or seed not in (3, 4) or split not in SPLITS:
        raise ValueError("Unregistered added-seed condition/seed/split")
    base = results / "protocol/protocol.json"
    registration = results / "seed_protocol/registration.json"
    protocol_path = registration.parent / "active/protocol.json"
    work = workspace / "exp/mechanism_v2"
    # This guard runs before feature loading, importing Torch or accessing outcomes.
    core, protocol, added, gate = verify_all(work / "models", work / "seed_models",
                                            base, base.parent / "followup_policy.json", registration, results)
    if not gate["required"]:
        raise ValueError("Untriggered added-seed inference refused")
    from runtime import configure
    configure(workspace)
    import numpy as np
    import torch
    from mechanism_data import CONFIGS as MODES, load_features, split_indices, transform
    from navsim.agents.mechanism_model import MechanismModel

    torch.set_num_threads(2)
    torch.set_flush_denormal(True)
    source = protocol["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen features changed before added-seed inference")
    data = load_features(source["path"])
    indices = split_indices(data, protocol, split)
    tag = f"{config}_f1_steps_s{seed}"
    run = work / "seed_models" / tag
    donor_path = protocol_path.parent / "donors.npz"
    if digest(donor_path) != core["donors_sha256"]:
        raise ValueError("Added-seed donors changed")
    with np.load(donor_path, allow_pickle=False) as donors:
        if not np.array_equal(indices, donors[split + "_indices"]):
            raise ValueError("Added-seed donor alignment changed")
        donor = donors[split + "_donors"] if MODES[config][1] else None
    tokens, logs = data["token"][indices], data["log_name"][indices]
    if not len(tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Empty/duplicate frozen prediction cohort")
    meta = metadata(config, seed, split, len(indices), core, added, base, protocol_path, registration)
    if check_existing(out, meta, tokens, logs):
        print("VERIFIED ADDED-SEED EXPORT", tag, split, flush=True)
        return meta
    normalizer = json.loads((run / "normalization.json").read_text(encoding="utf-8"))
    context, objects = transform(data, indices, normalizer, donor)
    checkpoint = torch.load(run / "model.pt", map_location="cpu")
    model = MechanismModel(checkpoint["mode"], checkpoint["hidden"], checkpoint["layers"], checkpoint["object_hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    values = []
    with torch.no_grad():
        for offset in range(0, len(indices), 512):
            values.append(model(torch.from_numpy(context[offset:offset + 512]),
                                torch.from_numpy(objects[offset:offset + 512])).numpy().reshape(-1, 8, 3))
    poses = np.concatenate(values)
    if poses.shape != (len(indices), 8, 3) or not np.isfinite(poses).all():
        raise FloatingPointError("No fallback for nonfinite/wrong-shape added-seed output")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive writes deliberately leave a partial artifact on interruption.
    with out.open("xb") as stream:
        np.savez_compressed(stream, token=tokens, log_name=logs, poses=poses, meta=np.array(json.dumps(meta)))
    write_new_json(out.with_suffix(".completed.json"), {"meta": meta, "predictions_sha256": digest(out)})
    print("ADDED-SEED EXPORT COMPLETED", tag, split, len(indices), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--seed", type=int, choices=(3, 4), required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    export(args.config, args.seed, args.split, args.workspace, ROOT / "results/mechanism_v2", args.out)


if __name__ == "__main__":
    main()
