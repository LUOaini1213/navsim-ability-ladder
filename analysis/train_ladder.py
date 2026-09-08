"""Train one cell of the ladder tables from frozen features (scripts/build_features.py).

  --see-map / --see-agents      which privileged blocks the MLP receives (2x2 factorial)
  --target pose|ds              pose: 8 ego-frame poses (learned path)   ds: 8 arc-length
                                offsets along the rule centerline (rule path + learned speed)
  --seed                        training seed (report >= 3 seeds per cell)
  --select last|best            checkpoint choice; "last" avoids selecting by open-loop loss

Writes <out>/model.pt + norm.json (consumed by LadderMLPAgent) and metrics.json
with train/val open-loop L1 per epoch, so the L2-vs-PDMS analysis has its x-axis.

Example
  python scripts/train_ladder.py --train features_navtrain.npz --see-map --target ds --seed 0 --tag map_ds_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(os.environ["NAVSIM_DEVKIT_ROOT"])))
from navsim.agents.ladder_features import assemble, input_dim  # noqa: E402
from navsim.agents.ladder_mlp_agent import LadderMLP  # noqa: E402


def load_split(path: str, see_map: bool, see_agents: bool, target: str, val_frac: float, seed: int, log_holdout=None):
    with np.load(path, allow_pickle=False) as z:  # NpzFile is lazy: materialise each array exactly once
        d = {k: z[k] for k in ["ego", "cl", "agents", "lead", "y_pose", "y_ds", "ds_valid", "token", "log_name"]}
    n = len(d["token"])
    blocks = [d["ego"].astype(np.float32)]
    if see_map:
        blocks.append(d["cl"].reshape(n, -1).astype(np.float32))
    if see_agents:
        blocks.append(d["agents"].reshape(n, -1).astype(np.float32))
        blocks.append(d["lead"].astype(np.float32))
    X = np.concatenate(blocks, axis=1)  # same block order as ladder_features.assemble
    assert np.allclose(X[0], assemble(d["ego"][0], d["cl"][0], d["agents"][0], d["lead"][0], see_map, see_agents))
    if target == "ds":
        Y = d["y_ds"].astype(np.float32)
        keep = d["ds_valid"] > 0.5
    else:
        Y = d["y_pose"].astype(np.float32)
        keep = np.ones(n, bool)
    logs = d["log_name"]
    # hold out whole logs for validation (no scene-level leakage between train and val)
    uniq = np.unique(logs)
    rng = np.random.RandomState(1234)  # fixed: the val logs are the same for every seed / cell
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_logs = set(uniq[:n_val]) if log_holdout is None else set(log_holdout)
    is_val = np.array([l in val_logs for l in logs])
    tr, va = keep & ~is_val, keep & is_val
    return X[tr], Y[tr], X[va], Y[va], sorted(val_logs), int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="features_navtrain.npz")
    ap.add_argument("--see-map", action="store_true")
    ap.add_argument("--see-agents", action="store_true")
    ap.add_argument("--target", choices=["pose", "ds"], default="pose")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=["last", "best"], default="last")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default=str(Path(os.environ["NAVSIM_EXP_ROOT"]) / "ladder"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Weight decay drives many weights into the subnormal range; x86 handles subnormals in microcode,
    # which made the wd runs 20-50x slower per epoch. Flush-to-zero keeps CPU training at full speed.
    torch.set_flush_denormal(True)
    out = Path(args.out_root) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    Xtr, Ytr, Xva, Yva, val_logs, n_keep = load_split(args.train, args.see_map, args.see_agents, args.target, args.val_frac, args.seed)
    assert Xtr.shape[1] == input_dim(args.see_map, args.see_agents)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd).astype(np.float32)  # constant / near-constant features: leave unscaled
    Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd
    print(f"[{args.tag}] see_map={args.see_map} see_agents={args.see_agents} target={args.target} "
          f"n_train={len(Xtr)} n_val={len(Xva)} (val logs {len(val_logs)}) in_dim={Xtr.shape[1]} load={time.time()-t0:.0f}s", flush=True)

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    out_dim = Ytr.shape[1]
    model = LadderMLP(Xtr.shape[1], args.hidden, out_dim, args.dropout, args.layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    xt, yt = torch.tensor(Xtr_n), torch.tensor(Ytr)
    xv, yv = torch.tensor(Xva_n), torch.tensor(Yva)
    hist, best, best_ep = [], 1e9, -1
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), args.bs):
            idx = perm[i:i + args.bs]
            opt.zero_grad()
            loss = torch.nn.functional.l1_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(xv)
            vl1 = float(torch.nn.functional.l1_loss(pv, yv))
            vl2 = float(((pv - yv) ** 2).mean().sqrt())
        hist.append({"epoch": ep + 1, "train_l1": tot / len(xt), "val_l1": vl1, "val_rmse": vl2})
        is_best = vl1 < best
        if is_best:
            best, best_ep = vl1, ep + 1
            if args.select == "best":
                torch.save({"state_dict": model.state_dict(), "in_dim": int(Xtr.shape[1]), "hidden": args.hidden,
                            "out_dim": int(out_dim), "dropout": args.dropout, "layers": args.layers}, out / "model.pt")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  train L1 {tot/len(xt):.4f}  val L1 {vl1:.4f}{'  *' if is_best else ''}  ({time.time()-t0:.0f}s)", flush=True)
    if args.select == "last":
        torch.save({"state_dict": model.state_dict(), "in_dim": int(Xtr.shape[1]), "hidden": args.hidden,
                    "out_dim": int(out_dim), "dropout": args.dropout, "layers": args.layers}, out / "model.pt")
    norm = {"mu": mu.tolist(), "sd": sd.tolist(), "see_map": args.see_map, "see_agents": args.see_agents,
            "target": args.target, "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout, "wd": args.wd,
            "lr": args.lr, "bs": args.bs, "epochs": args.epochs, "seed": args.seed, "select": args.select,
            "best_val_l1": best, "best_epoch": best_ep, "final_val_l1": hist[-1]["val_l1"],
            "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_kept": n_keep, "val_logs": val_logs, "train_file": args.train}
    json.dump(norm, open(out / "norm.json", "w", encoding="utf-8"), indent=1)
    json.dump(hist, open(out / "metrics.json", "w", encoding="utf-8"), indent=1)
    print(f"[{args.tag}] done: best val L1 {best:.4f} @ {best_ep}, final {hist[-1]['val_l1']:.4f}  -> {out}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
