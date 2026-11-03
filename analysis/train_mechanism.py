"""Train one controlled model with resumable, fixed-update CPU training."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from mechanism_data import (CONFIGS, SCHEMA, digest, fit_normalizer, load_features,
                            split_indices, transform)
from runtime import ROOT, configure


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def errors(pred, truth):
    delta = (pred-truth).reshape(-1, 8, 3)
    return {"mixed_l1": float(delta.abs().mean()), "xy_l1_m": float(delta[:, :, :2].abs().mean()),
            "heading_l1_rad": float(delta[:, :, 2].abs().mean()),
            "ade_m": float(delta[:, :, :2].square().sum(2).sqrt().mean()),
            "fde_m": float(delta[:, -1, :2].square().sum(1).sqrt().mean())}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace")
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fraction", type=float, choices=[.1, .3, 1.], default=1.)
    ap.add_argument("--budget", choices=["steps", "epochs"], default="steps")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-steps", type=int, help="Smoke only: produces a separately named, ineligible checkpoint")
    ap.add_argument("--out-root", type=Path)
    args = ap.parse_args()
    workspace = configure(args.workspace)
    from navsim.agents.mechanism_model import MechanismModel

    torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["schema"] != SCHEMA or args.seed not in protocol["training"]["seeds"]:
        raise ValueError("Unregistered protocol/seed")
    source = Path(protocol["sources"]["navtrain"]["path"])
    if digest(source) != protocol["sources"]["navtrain"]["sha256"]:
        raise ValueError("Features changed since protocol freeze")
    tag = f"{args.config}_f{args.fraction:g}_{args.budget}_s{args.seed}"
    if args.max_steps is not None:
        tag += f"_smoke{args.max_steps}"
    out = (args.out_root or workspace / "exp/mechanism_v2/models") / tag
    out.mkdir(parents=True, exist_ok=True)
    settings = {"schema": SCHEMA, "protocol_sha256": digest(args.protocol), "config": args.config,
                "donors_sha256": digest(args.protocol.parent / "donors.npz"),
                "seed": args.seed, "fraction": args.fraction, "budget": args.budget,
                "smoke_max_steps": args.max_steps, "training": protocol["training"],
                "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                                  [ROOT / "agents/mechanism_model.py", ROOT / "analysis/mechanism_data.py",
                                   ROOT / "analysis/train_mechanism.py"]}}
    settings_path = out / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise RuntimeError("Settings/source changed; use a fresh versioned output root")
    else:
        atomic_json(settings_path, settings)
    if (out / "completed.json").exists():
        complete = json.loads((out / "completed.json").read_text(encoding="utf-8"))
        if complete["checkpoint_sha256"] != digest(out / "model.pt"):
            raise ValueError("Completed checkpoint hash mismatch")
        print("verified completed", tag, flush=True)
        return
    started = time.monotonic()
    data = load_features(source)
    tr = split_indices(data, protocol, "train", args.fraction)
    va = split_indices(data, protocol, "validation")
    norm = fit_normalizer(data, tr)
    mode, shuffled = CONFIGS[args.config]
    # Shuffled variants exist only at full data in the preregistered core matrix.
    if shuffled and args.fraction != 1.:
        raise ValueError("Shuffled learning curves are not registered")
    donors_path = args.protocol.parent / "donors.npz"
    with np.load(donors_path) as donors:
        tr_donor = donors["train_donors"] if shuffled else None
        va_donor = donors["validation_donors"] if shuffled else None
        if shuffled and not np.array_equal(donors["train_indices"], tr):
            raise ValueError("Frozen training donor alignment mismatch")
    context, objects = transform(data, tr, norm, tr_donor)
    vc, vo = transform(data, va, norm, va_donor)
    x, obj, y = torch.from_numpy(context), torch.from_numpy(objects), torch.from_numpy(data["y_pose"][tr].copy())
    xv, ov, yv = torch.from_numpy(vc), torch.from_numpy(vo), torch.from_numpy(data["y_pose"][va].copy())
    del data, context, objects, vc, vo
    hp = protocol["training"]
    model = MechanismModel(mode, hp["hidden"], hp["layers"], hp["object_hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    full_steps = math.ceil(protocol["split_counts"]["train"]["scenes"]/hp["batch_size"])
    steps_per_epoch = math.ceil(len(x)/hp["batch_size"])
    target_steps = hp["full_data_epochs_equivalent"] * (full_steps if args.budget == "steps" else steps_per_epoch)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("Positive smoke budget required")
        target_steps = min(target_steps, args.max_steps)
    step, order, position, history, previous_seconds = 0, torch.randperm(len(x)), 0, [], 0.
    resume = out / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu")
        if state["settings"] != settings:
            raise ValueError("Resume settings mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        step, order, position = state["step"], state["order"], state["position"]
        history, previous_seconds = state["history"], state["elapsed_seconds"]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"[{tag}] train={len(x)} val={len(xv)} params={parameters} steps={step}/{target_steps} threads={args.threads}", flush=True)
    accumulated, observations = 0., 0
    while step < target_steps:
        model.train()
        if position >= len(order):
            order, position = torch.randperm(len(x)), 0
        ids = order[position:position+hp["batch_size"]]
        position += len(ids)
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(x[ids], obj[ids]), y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        opt.step()
        accumulated += float(loss)*len(ids)
        observations += len(ids)
        step += 1
        if step % full_steps == 0 or step == target_steps:
            model.eval()
            with torch.no_grad():
                predictions = torch.cat([model(xv[i:i+512], ov[i:i+512]) for i in range(0, len(xv), 512)])
            if not torch.isfinite(predictions).all():
                raise FloatingPointError("Nonfinite validation predictions")
            elapsed = previous_seconds + time.monotonic()-started
            metrics = dict(step=step, virtual_epoch=step/full_steps, train_mixed_l1=accumulated/observations,
                           validation=errors(predictions, yv), elapsed_seconds=elapsed)
            history.append(metrics)
            atomic_json(out / "history.json", history)
            atomic_torch(resume, {"settings": settings, "model": model.state_dict(), "optimizer": opt.state_dict(),
                                  "rng": torch.get_rng_state(), "step": step, "order": order, "position": position,
                                  "history": history, "elapsed_seconds": elapsed})
            print(f"  step {step}/{target_steps} val mixed L1={metrics['validation']['mixed_l1']:.4f} "
                  f"ADE={metrics['validation']['ade_m']:.3f}m elapsed={elapsed:.1f}s", flush=True)
            accumulated, observations = 0., 0
    atomic_json(out / "normalization.json", norm)
    atomic_torch(out / "model.pt", {"schema": SCHEMA, "mode": mode, "hidden": hp["hidden"], "layers": hp["layers"],
                                    "object_hidden": hp["object_hidden"], "state_dict": model.state_dict()})
    model.eval()
    with torch.no_grad():
        benchmark_context, benchmark_objects = xv[:1], ov[:1]
        for _ in range(10):
            model(benchmark_context, benchmark_objects)
        t0 = time.perf_counter()
        for _ in range(200):
            model(benchmark_context, benchmark_objects)
        latency = (time.perf_counter()-t0)/200*1000
        # Same-scene ordering intervention is a diagnostic, never a retraining control.
        permutation = torch.arange(ov.shape[1]-1, -1, -1)
        normal = model(xv[:512], ov[:512])
        reversed_objects = model(xv[:512], ov[:512, permutation])
        order_difference = float((normal-reversed_objects).abs().max())
    complete = {"schema": SCHEMA, "tag": tag, "eligible_for_paper": args.max_steps is None,
                "settings_sha256": digest(settings_path), "checkpoint_sha256": digest(out / "model.pt"),
                "normalization_sha256": digest(out / "normalization.json"),
                "parameters": parameters, "n_train": len(x), "n_validation": len(xv),
                "optimizer_steps": step, "virtual_epochs": step/full_steps,
                "approx_data_epochs": step/steps_per_epoch, "cpu_single_scene_ms": latency,
                "inference_threads": args.threads, "same_scene_reverse_order_max_pose_difference": order_difference,
                "final_validation": history[-1]["validation"],
                "elapsed_seconds": previous_seconds+time.monotonic()-started}
    atomic_json(out / "completed.json", complete)
    print("COMPLETED", tag, json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
