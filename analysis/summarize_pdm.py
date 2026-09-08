"""Aggregate NAVSIM per-scene PDM CSVs into ladder tables with paired bootstrap CIs.

Reads every experiment directory under $NAVSIM_EXP_ROOT matching --glob (default
"*_navtest"), takes the newest CSV in each, and writes
  <out>/ladder_<split>.md      one row per agent: n, PDMS, NC, DAC, TTC, C, EP, DDC (+ fail rates)
  <out>/pairs_<split>.md       paired deltas with 95% bootstrap CIs for --pairs a:b,...
  <out>/scores_<split>.csv     wide per-scene table (token x agent) for downstream analysis

Usage
  python scripts/summarize_pdm.py --split navtest --pairs cv_navtest:kin_navtest,kin_navtest:privmapkin_navtest
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
           "ego_progress", "comfort", "driving_direction_compliance"]
SHORT = {"score": "PDMS", "no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC",
         "time_to_collision_within_bound": "TTC", "ego_progress": "EP", "comfort": "C",
         "driving_direction_compliance": "DDC"}


def newest_csv(exp_dir: Path):
    files = sorted(glob.glob(str(exp_dir / "*" / "*.csv")))
    return files[-1] if files else None


def load_experiments(root: Path, pattern: str):
    out = {}
    for d in sorted(root.glob(pattern)):
        if d.name.startswith("metric_cache") or not d.is_dir():
            continue
        f = newest_csv(d)
        if not f:
            continue
        df = pd.read_csv(f)
        if "token" not in df.columns or "score" not in df.columns:
            continue
        df = df[df["token"] != "average"]
        df = df[df["valid"] == True]  # noqa: E712
        out[d.name] = df.set_index("token")
    return out


def bootstrap_ci(delta: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(os.environ["NAVSIM_EXP_ROOT"])
    out = Path(args.out or root / "summary")
    out.mkdir(parents=True, exist_ok=True)
    exps = load_experiments(root, args.glob or f"*_{args.split}")
    if not exps:
        raise SystemExit("no experiments found")

    rows = []
    for name, df in exps.items():
        r = {"agent": name.replace(f"_{args.split}", ""), "n": len(df)}
        for m in METRICS:
            if m in df:
                r[SHORT[m]] = df[m].mean()
        for m in ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound"]:
            if m in df:
                r[SHORT[m] + "_fail"] = float((df[m] < 0.5).mean())
        rows.append(r)
    table = pd.DataFrame(rows).sort_values("PDMS")
    md = [f"# Ladder on {args.split} (per-scene means, newest CSV per experiment)\n",
          "| agent | n | PDMS | NC | DAC | TTC | C | EP | DDC | NC fail | DAC fail | TTC fail |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        md.append("| {agent} | {n} | **{PDMS:.3f}** | {NC:.3f} | {DAC:.3f} | {TTC:.3f} | {C:.3f} | {EP:.3f} | {DDC:.3f} | {NC_fail:.1%} | {DAC_fail:.1%} | {TTC_fail:.1%} |".format(**{k: r.get(k, float('nan')) for k in ["agent", "n", "PDMS", "NC", "DAC", "TTC", "C", "EP", "DDC", "NC_fail", "DAC_fail", "TTC_fail"]}))
    (out / f"ladder_{args.split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    wide = pd.DataFrame({name: df["score"] for name, df in exps.items()})
    wide.to_csv(out / f"scores_{args.split}.csv")

    if args.pairs:
        pmd = [f"# Paired deltas on {args.split} (b - a, scenes scored by both; 95% bootstrap CI, 10k resamples)\n",
               "| a | b | n | dPDMS | 95% CI | dDAC | dNC | dEP | verdict |", "|---|---|---:|---:|---|---:|---:|---:|---|"]
        for pair in args.pairs.split(","):
            a, b = pair.split(":")
            if a not in exps or b not in exps:
                pmd.append(f"| {a} | {b} | - | missing | | | | | |")
                continue
            common = exps[a].index.intersection(exps[b].index)
            da, db = exps[a].loc[common], exps[b].loc[common]
            mean, lo, hi = bootstrap_ci((db["score"] - da["score"]).to_numpy())
            ddac = float((db["drivable_area_compliance"] - da["drivable_area_compliance"]).mean())
            dnc = float((db["no_at_fault_collisions"] - da["no_at_fault_collisions"]).mean())
            dep = float((db["ego_progress"] - da["ego_progress"]).mean())
            verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI spans 0"
            pmd.append(f"| {a} | {b} | {len(common)} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {ddac:+.3f} | {dnc:+.3f} | {dep:+.3f} | {verdict} |")
        (out / f"pairs_{args.split}.md").write_text("\n".join(pmd) + "\n", encoding="utf-8")
        print("\n".join(pmd))


if __name__ == "__main__":
    main()
