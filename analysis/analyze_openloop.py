"""Open-loop fit vs simulation score (E6): per-run and per-scene.

For every trained cell under $NAVSIM_EXP_ROOT/ladder/<tag> that has a navtest PDM
result (exp/l_<tag>_navtest), compute on the frozen navtest features
  - open-loop L1 / L2 (ADE over the 8 poses for pose targets; L1 over the 8 arc-length offsets for ds targets)
  - join with per-scene PDMS  ->  run-level table + scene-level Spearman(L2, PDMS)
Writes exp/summary/openloop_navtest.md and openloop_navtest.csv, plus a scatter png
(final val L1 on navtrain-val vs navtest PDMS) if matplotlib is available.

Usage: python scripts/analyze_openloop.py [--split navtest]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from navsim.agents.ladder_features import assemble
from navsim.agents.ladder_mlp_agent import load_run


def newest_csv(exp_dir: Path):
    files = sorted(glob.glob(str(exp_dir / "*" / "*.csv")))
    return files[-1] if files else None


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    args = ap.parse_args()
    root = Path(os.environ["NAVSIM_EXP_ROOT"])
    with np.load(root / "features" / f"features_{args.split}.npz", allow_pickle=False) as z:  # lazy NpzFile: read once
        feats = {k: z[k] for k in z.files}
    tokens = feats["token"].astype(str)
    n_sc = len(tokens)
    rows, scene_rows = [], []
    for run_dir in sorted((root / "ladder").glob("*/")):
        tag = run_dir.name
        if tag.startswith("_trial") or not (run_dir / "model.pt").exists():
            continue
        model, mu, sd, norm = load_run(str(run_dir))
        see_map, see_agents, target = bool(norm["see_map"]), bool(norm["see_agents"]), norm["target"]
        blocks = [feats["ego"].astype(np.float32)]
        if see_map:
            blocks.append(feats["cl"].reshape(n_sc, -1).astype(np.float32))
        if see_agents:
            blocks.append(feats["agents"].reshape(n_sc, -1).astype(np.float32))
            blocks.append(feats["lead"].astype(np.float32))
        X = np.concatenate(blocks, axis=1)  # same block order as ladder_features.assemble
        assert np.allclose(X[0], assemble(feats["ego"][0], feats["cl"][0], feats["agents"][0], feats["lead"][0], see_map, see_agents))
        with torch.no_grad():
            pred = model(torch.tensor((X - mu) / sd)).numpy()
        if target == "pose":
            y = feats["y_pose"]
            l1 = np.abs(pred - y).mean(1)
            ade = np.linalg.norm((pred - y).reshape(-1, 8, 3)[:, :, :2], axis=2).mean(1)
        else:
            y = feats["y_ds"]
            l1 = np.abs(pred - y).mean(1)
            ade = l1
        for suffix, mode in [("", "learned"), ("_rs", "rule")]:
            if target == "ds" and suffix:
                continue
            f = newest_csv(root / f"l_{tag}{suffix}_{args.split}")
            if not f:
                continue
            pdm = pd.read_csv(f)
            pdm = pdm[(pdm["token"] != "average") & (pdm["valid"] == True)].set_index("token")  # noqa: E712
            common = np.isin(tokens, pdm.index.to_numpy())
            s = pdm.loc[tokens[common], "score"].to_numpy()
            rows.append({"run": f"{tag}{suffix}", "target": target, "see_map": see_map, "see_agents": see_agents,
                         "speed_mode": mode, "wd": norm.get("wd"), "dropout": norm.get("dropout"), "hidden": norm.get("hidden"),
                         "seed": norm.get("seed"), "val_l1_navtrain": norm.get("final_val_l1"),
                         f"l1_{args.split}": float(l1[common].mean()), f"ade_{args.split}": float(ade[common].mean()),
                         "PDMS": float(s.mean()), "n": int(common.sum()),
                         "spearman_ade_pdms": spearman(ade[common], s)})
            scene_rows.append(pd.DataFrame({"run": f"{tag}{suffix}", "token": tokens[common], "ade": ade[common], "l1": l1[common], "PDMS": s}))
    if not rows:
        raise SystemExit("no scored runs found")
    df = pd.DataFrame(rows).sort_values(["target", "see_map", "see_agents", "wd", "dropout", "seed"])
    out = root / "summary"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"openloop_{args.split}.csv", index=False)
    pd.concat(scene_rows).to_csv(out / f"openloop_scenes_{args.split}.csv", index=False)
    md = [f"# Open-loop fit vs PDMS on {args.split}\n",
          "| run | target | map | agents | speed | wd | dropout | hidden | seed | val L1 (navtrain-val) | ADE navtest | PDMS | Spearman(ADE,PDMS) |",
          "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        md.append(f"| {r['run']} | {r['target']} | {int(r['see_map'])} | {int(r['see_agents'])} | {r['speed_mode']} | {r['wd']} | {r['dropout']} | {r['hidden']} | {r['seed']} | {r['val_l1_navtrain']:.4f} | {r[f'ade_{args.split}']:.3f} | **{r['PDMS']:.3f}** | {r['spearman_ade_pdms']:+.3f} |")
    (out / f"openloop_{args.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # pose-target runs only: the ds models' "ADE" is an arc-length L1 and is not comparable
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        sub = df[(df["target"] == "pose") & (df["speed_mode"] == "learned")]
        groups = [("pose_kin", "kin", "tab:gray"), ("pose_agents", "kin+agents", "tab:orange"),
                  ("pose_map", "kin+map", "tab:blue"), ("pose_map_agents", "kin+map+agents", "tab:green"),
                  ("reg_pose_map", "kin+map, regularised", "tab:red")]
        for prefix, label, color in groups:
            g = sub[sub["run"].str.startswith(prefix + "_")]
            if len(g):
                ax.scatter(g[f"ade_{args.split}"], g["PDMS"], label=label, color=color, s=28)
        for _, r in sub[sub["run"].str.startswith("reg_")].iterrows():
            ax.annotate(r["run"].replace("reg_pose_map_", "").replace("_s0", ""), (r[f"ade_{args.split}"], r["PDMS"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel(f"open-loop ADE on {args.split} (m)  ← better")
        ax.set_ylabel("PDMS  ↑ better")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"openloop_{args.split}.png", dpi=160)
        print("figure ->", out / f"openloop_{args.split}.png")
    except Exception as e:  # pragma: no cover
        print("no figure:", e)


if __name__ == "__main__":
    main()
