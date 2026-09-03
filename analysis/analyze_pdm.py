"""Compare PDM CSVs, write a table + failure tokens, dump a few hard scenes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('NAVSIM_WORKSPACE', '.'))
EXP = ROOT / "exp"
OUT = EXP / "analysis"
PY = os.environ.get("NAVSIM_PYTHON", sys.executable)
CKPT = EXP / "training_ego_mlp_mini" / "ego_status_mlp_mini.ckpt"

METRICS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
]
SHORT = {
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "ego_progress": "EP",
    "time_to_collision_within_bound": "TTC",
    "comfort": "C",
    "driving_direction_compliance": "DDC",
    "score": "PDMS",
}

AGENTS = {
    "CV": EXP / "cv_agent_mini",
    "Kinematic": EXP / "kinematic_agent_mini",
    "MLP": EXP / "mlp_agent_mini",
    "PrivBrake": EXP / "privileged_brake_mini",
    "PrivMap": EXP / "privileged_centerline_mini",
    "PrivMapKin": EXP / "privileged_centerline_kin",
    "PrivMapGTSpd": EXP / "privileged_centerline_gtspeed",
    "PrivGTPathKin": EXP / "privileged_gtpath_kin",
    "Human": EXP / "human_agent_mini",
}


def _latest_csv(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    csvs = [p for p in folder.rglob("*.csv") if p.stat().st_size > 0]
    if not csvs:
        return None

    def _mean_score(path: Path) -> float:
        try:
            df = pd.read_csv(path)
            df = df[df["token"].notna() & (df["token"].astype(str) != "average")]
            return float(df["score"].mean()) if len(df) else -1.0
        except Exception:
            return -1.0

    return max(csvs, key=_mean_score)


def _load(name: str, folder: Path) -> pd.DataFrame | None:
    csv = _latest_csv(folder)
    if csv is None:
        print(f"skip {name}: no csv under {folder}")
        return None
    df = pd.read_csv(csv)
    df = df[df["token"].notna() & (df["token"].astype(str) != "average")]
    print(f"{name}: {csv.name} n={len(df)} PDMS={df['score'].mean():.4f}")
    return df


def _fail_tag(row: pd.Series) -> str:
    if row["no_at_fault_collisions"] < 1:
        return "NC"
    if row["drivable_area_compliance"] < 1:
        return "DAC"
    if row["time_to_collision_within_bound"] < 1:
        return "TTC"
    if row["score"] <= 0:
        return "other"
    return "ok"


def _dump(token: str, tag: str) -> None:
    dump = ROOT / "dump_one_trajectory.py"
    cmd = [
        PY,
        str(ROOT / "run_navsim_step.py"),
        str(dump),
        "--ckpt",
        str(CKPT),
        "--token",
        token,
        "--out-dir",
        str(EXP / "trajectories"),
        "--split",
        "warmup",
        "--dying",
        tag,
    ]
    print("dump", token, tag)
    subprocess.check_call(cmd)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, folder in AGENTS.items():
        df = _load(name, folder)
        if df is not None:
            frames[name] = df.set_index("token")

    if "CV" not in frames or "MLP" not in frames:
        print("need at least CV and MLP CSVs")
        return 1

    rows = []
    for name, df in frames.items():
        rec = {"agent": name, "n": int(len(df)), "score_zero": float((df["score"] <= 0).mean())}
        for m in METRICS:
            rec[SHORT[m]] = float(df[m].mean())
        rec["NC_fail"] = float((df["no_at_fault_collisions"] < 1).mean())
        rec["DAC_fail"] = float((df["drivable_area_compliance"] < 1).mean())
        rec["TTC_fail"] = float((df["time_to_collision_within_bound"] < 1).mean())
        rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "pdm_summary.csv", index=False)

    # pairwise vs CV on shared tokens
    base = frames["CV"]
    cmp_rows = []
    for name, df in frames.items():
        common = base.index.intersection(df.index)
        if len(common) == 0:
            continue
        delta = df.loc[common, "score"] - base.loc[common, "score"]
        cmp_rows.append(
            {
                "agent": name,
                "n_common": int(len(common)),
                "mean_delta_vs_CV": float(delta.mean()),
                "frac_better_than_CV": float((delta > 0).mean()),
                "frac_worse_than_CV": float((delta < 0).mean()),
            }
        )
    pd.DataFrame(cmp_rows).to_csv(OUT / "pdm_vs_cv.csv", index=False)

    # MLP failure taxonomy (relative to Human if present)
    mlp = frames["MLP"].copy()
    mlp["fail"] = mlp.apply(_fail_tag, axis=1)
    fail_counts = mlp["fail"].value_counts().to_dict()
    (OUT / "mlp_fail_counts.json").write_text(json.dumps(fail_counts, indent=2), encoding="utf-8")

    hard = mlp.sort_values("score").head(8)
    hard[["score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress"]].to_csv(
        OUT / "mlp_worst8.csv"
    )

    # dump 1 token per fail family + 1 high-score success
    dumped = []
    for tag in ("NC", "DAC", "TTC"):
        hit = mlp[mlp["fail"] == tag]
        if len(hit):
            tok = str(hit.sort_values("score").index[0])
            try:
                _dump(tok, tag)
                dumped.append({"token": tok, "tag": tag})
            except Exception as e:
                print("dump failed", tok, e)
    ok = mlp[mlp["fail"] == "ok"]
    if len(ok):
        tok = str(ok.sort_values("score", ascending=False).index[0])
        try:
            _dump(tok, "ok")
            dumped.append({"token": tok, "tag": "ok"})
        except Exception as e:
            print("dump failed", tok, e)

    md = ["# NAVSIM warmup_test_e2e PDM comparison", ""]
    md.append("Split: `warmup_test_e2e` (same 563-scene cache as CV 0.233).")
    md.append("")
    md.append("| agent | n | PDMS | NC | DAC | EP | TTC | C | score=0 | NC fail | DAC fail | TTC fail |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['agent']} | {r['n']} | {r['PDMS']:.3f} | {r['NC']:.3f} | {r['DAC']:.3f} | {r['EP']:.3f} | {r['TTC']:.3f} | {r['C']:.3f} | {r['score_zero']:.3f} | {r['NC_fail']:.3f} | {r['DAC_fail']:.3f} | {r['TTC_fail']:.3f} |"
        )
    md.append("")
    md.append("## vs ConstantVelocity")
    md.append("")
    for r in cmp_rows:
        md.append(
            f"- **{r['agent']}**: ΔPDMS={r['mean_delta_vs_CV']:+.3f}, better than CV {r['frac_better_than_CV']:.1%}"
        )
    md.append("")
    md.append("## MLP failure families")
    md.append("")
    for k, v in fail_counts.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("Dumped trajectories: " + ", ".join(f"{d['tag']}={d['token']}" for d in dumped))
    md.append("")
    md.append("## How to read this")
    md.append("")
    md.append("- **Human** is privileged GT future — upper bound, not a deployable planner.")
    md.append("- **CV** is naive (constant speed, heading 0). Low EP + TTC/NC because it cannot slow or turn.")
    md.append("- **Kinematic** uses accel + left/right command yaw. Tests how far kinematics go without sensors.")
    md.append("- **MLP** is still blind (no camera/LiDAR) but *learns* the mapping from (v,a,command) → 4s trajectory.")
    md.append("- **PrivBrake** is not deployable: kinematics + IDM brake from GT boxes.")
    md.append("- **PrivMap** / **PrivMapKin** / **PrivMapGTSpd**: centerline with IDM, kinematic, or Human arc-length.")
    md.append("- **PrivGTPathKin**: Human logged path with kinematic speed (geometry vs timing).")
    md.append("- Remaining MLP zeros are mostly NC/DAC/TTC: scenes that need perception, not better kinematics.")
    (OUT / "pdm_report.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "pdm_report.md")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
