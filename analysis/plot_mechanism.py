"""Draw audited core-study figures, with explicitly ineligible layout smoke mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_mechanism_core import available_mb
from verify_mechanism_core import CONFIGS, ROOT, digest, verify

COLORS = {"M0": "#333333", "M1": "#0072b2", "M2": "#56b4e9", "M3": "#e69f00", "M4": "#009e73", "M5": "#cc79a7"}
NAMES = {"M0": "ego/map", "M1": "concat", "M2": "concat/shuffled", "M3": "lead summary", "M4": "set", "M5": "set/shuffled"}
PRIMARY = ("M1-M0", "M4-M1", "M4-M5")
CITY_NAMES = {"sg-one-north": "Singapore", "us-ma-boston": "Boston", "us-nv-las-vegas-strip": "Las Vegas",
              "us-pa-pittsburgh-hazelwood": "Pittsburgh"}


def save_figure(fig, out, name, report):
    if report.get("layout_fixture", False):
        fig.text(.5, -.15, "LAYOUT SMOKE ONLY - SYNTHETIC NUMBERS - NOT A PAPER RESULT", ha="center", va="top", color="#a00000", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(out / (name+"."+suffix), dpi=180, bbox_inches="tight")


def report_figures(report, out):
    """Pure rendering; the production CLI verifies score/model artifacts first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, label in enumerate(PRIMARY):
        row = summary["contrasts"][label]
        color = COLORS[label.split("-")[0]]
        adjusted, ordinary = row["multiplicity_interval"], row["session_interval"]
        if adjusted["low"] is not None:
            axis.plot([adjusted["low"], adjusted["high"]], [i, i], color=color, lw=1.5)
        if ordinary["low"] is not None:
            axis.plot([ordinary["low"], ordinary["high"]], [i, i], color=color, lw=5)
        axis.scatter([row["metric_effects"]["score"]], [i], marker="D", color=color, s=65, zorder=4)
        for seed, offset in zip(("0", "1", "2"), (-.12, 0, .12)):
            axis.scatter([row["seed_effects"][seed]], [i+offset], marker="x", color="black", s=22, zorder=5)
    axis.axvline(0, color="0.6", lw=1, linestyle="--")
    axis.set(yticks=range(len(PRIMARY)), yticklabels=PRIMARY, xlabel="Paired PDMS difference (left minus right)",
             title=report["split"]+": primary contrasts")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.text(.5, -.23, "Diamond: three-seed mean; crosses: individual seeds\nThick: 95% session CI; thin: 98.33% adjusted session CI",
              transform=axis.transAxes, ha="center", fontsize=9)
    save_figure(fig, out, "primary_contrasts", report)
    plt.close(fig)

    groups = list(summary["subgroups"])
    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, .42*len(groups))), sharey=True, constrained_layout=True)
    labels = []
    for group in groups:
        row = summary["subgroups"][group]
        name = "city: "+CITY_NAMES.get(group[5:], group[5:]) if group.startswith("city:") else group.replace("_", " ")
        labels.append(f"{name} (n={row['n_scenes']}, sessions={row['n_sessions']})")
    for axis, contrast in zip(axes, PRIMARY):
        for i, group in enumerate(groups):
            row = summary["subgroups"][group]["contrasts"].get(contrast)
            if row is None:
                axis.text(.5, i, "no scenes", transform=axis.get_yaxis_transform(), va="center", fontsize=8, color="0.5")
                continue
            interval = row["session_interval"]
            color = COLORS[contrast.split("-")[0]]
            if interval["low"] is not None:
                axis.plot([interval["low"], interval["high"]], [i, i], color=color)
            axis.scatter([row["metric_effects"]["score"]], [i], color=color, s=25,
                         marker="o" if interval["low"] is not None else "x")
        axis.axvline(0, color="0.6", lw=1, linestyle="--")
        axis.set(title=contrast, xlabel="Paired PDMS difference")
        axis.grid(axis="x", alpha=.2)
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(labels, fontsize=9)
    # Text-only empty groups do not update scatter data limits. Keep their rows
    # inside the axes, rather than drawing the last label on the bottom spine.
    axes[0].set_ylim(len(groups)-.5, -.5)
    fig.suptitle(report["split"]+": overlapping input-defined strata; exploratory 95% session CIs\n"
                 "Crossing is a straight-corridor proxy; x marker means interval not estimable (<2 sessions)", fontsize=11)
    save_figure(fig, out, "subgroup_effects", report)
    plt.close(fig)

    metrics = [("no_at_fault_collisions", "NC"), ("drivable_area_compliance", "DAC"), ("time_to_collision_within_bound", "TTC")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    markers = ("o", "s", "^", "P", "D", "v")
    for axis, (metric, abbreviation) in zip(axes, metrics):
        for config, marker in zip(CONFIGS, markers):
            row = summary["cells"][config]
            points = list(row["seed_metrics"].values())
            axis.scatter([value["ego_progress"] for value in points], [value[metric] for value in points],
                         c=COLORS[config], marker=marker, s=16, alpha=.35)
            axis.scatter([row["mean"]["ego_progress"]], [row["mean"][metric]], c=COLORS[config], marker=marker,
                         s=80, edgecolors="white", linewidths=.6, label=config+" "+NAMES[config])
        axis.set(xlabel="EP (ego progress)", ylabel=abbreviation, title=abbreviation+" vs progress")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.1), ncol=3, fontsize=9)
    fig.suptitle(report["split"]+": metric-progress trade-offs; small symbols are individual seeds\n"
                 "Axes zoom to observed values; no significance or deployment-safety claim follows from this scatter", fontsize=10)
    save_figure(fig, out, "metric_progress", report)
    plt.close(fig)


def training_figure(core, model_root, out, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    histories, history_hashes = {}, {}
    for row in core["runs"]:
        run = model_root / row["tag"]
        history = json.loads((run / "history.json").read_text(encoding="utf-8"))
        if not history or history[-1]["validation"] != row["final_validation"] or history[-1]["step"] != row["optimizer_steps"]:
            raise ValueError("History/final-checkpoint mismatch: "+row["tag"])
        histories[row["tag"]] = history
        history_hashes[row["tag"]] = digest(run / "history.json")
    metrics = [("ade_m", "Validation ADE (m)"), ("xy_l1_m", "Validation coordinate L1 (m)"),
               ("heading_l1_rad", "Validation raw-heading L1 (rad)"), ("mixed_l1", "Validation mixed L1 (m/rad)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for config in CONFIGS:
            runs = [histories[f"{config}_f1_steps_s{seed}"] for seed in range(3)]
            steps = [row["step"] for row in runs[0]]
            if any([row["step"] for row in history] != steps for history in runs):
                raise ValueError("Unequal diagnostic steps across seeds")
            values = np.array([[row["validation"][metric] for row in history] for history in runs])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite convergence metric")
            axis.plot(steps, values.mean(0), color=COLORS[config], label=config)
            axis.fill_between(steps, values.min(0), values.max(0), color=COLORS[config], alpha=.1)
        axis.set(xlabel="Optimizer updates", ylabel=label)
        axis.grid(alpha=.2)
    axes[0, 0].legend(ncol=3, fontsize=9)
    fig.suptitle("Full-budget core convergence: mean and min/max across three fitted seeds (not a confidence band)", fontsize=11)
    save_figure(fig, out, "core_convergence", report)
    plt.close(fig)
    return history_hashes


def layout_fixture():
    """Only for exercising label/layout branches; cannot pass production checks."""
    cells, contrasts = {}, {}
    metric_names = ("score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress", "comfort", "driving_direction_compliance")
    for number, config in enumerate(CONFIGS):
        mean = dict(zip(metric_names, (.65+number*.01, .94+number*.003, .91+number*.004, .87+number*.004,
                                       .8-number*.005, .9, .98)))
        cells[config] = {"mean": mean, "seed_metrics": {str(seed): {key: value+(seed-1)*.002 for key, value in mean.items()} for seed in range(3)}}
    for i, label in enumerate(PRIMARY):
        effect = (i-1)*.03
        contrasts[label] = {"metric_effects": {"score": effect}, "seed_effects": {str(seed): effect+(seed-1)*.01 for seed in range(3)},
                            "session_interval": {"low": effect-.025, "high": effect+.025},
                            "multiplicity_interval": {"low": effect-.035, "high": effect+.035}}
    groups = {}
    for name in ("all", "near_objects", "no_near_objects", "low_speed", "high_speed", "turn_command", "straight_command", "close_lead",
                 "potential_crossing_proxy", "city:sg-one-north", "city:us-ma-boston", "city:us-nv-las-vegas-strip", "city:us-pa-pittsburgh-hazelwood", "empty"):
        count, sessions = (0, 0) if name == "empty" else (100, 1) if name == "city:sg-one-north" else (5000, 16)
        rows = {key: {"metric_effects": value["metric_effects"], "session_interval": {"low": None, "high": None} if sessions < 2 else value["session_interval"]}
                for key, value in contrasts.items()} if count else {}
        groups[name] = {"n_scenes": count, "n_sessions": sessions, "contrasts": rows}
    return {"layout_fixture": True, "split": "LAYOUT SMOKE", "summary": {"cells": cells, "contrasts": contrasts, "subgroups": groups}}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--runs", type=Path, default=Path("E:/navsim_workspace/exp/mechanism_v2/models"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, required=True)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--smoke-layout", action="store_true")
    modes.add_argument("--training-only", action="store_true", help="Audited real convergence only; no driving-score figure")
    ap.add_argument("--min-free-mb", type=int, default=1800)
    args = ap.parse_args()
    free = available_mb()
    if free is not None and free < args.min_free_mb:
        raise RuntimeError(f"Memory guard: {free:.0f} MB free, needs {args.min_free_mb}; no renderer launched")
    if args.smoke_layout:
        if args.report or args.scores:
            raise ValueError("Layout smoke cannot consume paper scores")
        report_figures(layout_fixture(), args.out)
        (args.out / "receipt.json").write_text(json.dumps({"eligible_for_paper": False, "purpose": "synthetic layout smoke only"}, indent=2)+"\n", encoding="utf-8")
        print("LAYOUT SMOKE ONLY", args.out, flush=True)
        return
    if args.training_only:
        if args.report or args.scores:
            raise ValueError("Training-only rendering cannot consume driving-score reports")
        core = verify(args.runs, args.protocol)
        args.out.mkdir(parents=True, exist_ok=True)
        signature = {"artifact_kind": "verified training convergence, not PDMS", "protocol_sha256": core["protocol_sha256"],
                     "checkpoint_sha256": {row["tag"]: row["checkpoint_sha256"] for row in core["runs"]}}
        path = args.out / "settings.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Training figure inputs changed; use a new output folder")
        path.write_text(json.dumps(signature, indent=2)+"\n", encoding="utf-8")
        histories = training_figure(core, args.runs, args.out, {})
        receipt = {"eligible_for_paper": True, **signature, "scope": "Training diagnostics only; no driving-score or safety finding",
                   "render_source_sha256": digest(Path(__file__)), "history_sha256": histories,
                   "figures": {file.name: digest(file) for file in sorted(args.out.iterdir()) if file.suffix in (".png", ".svg")},
                   "visual_qa_status": "pending inspection"}
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        print("REAL CORE CONVERGENCE RENDERED; visual QA still required", args.out, flush=True)
        return
    if args.report is None or args.scores is None:
        raise ValueError("Production requires both audited report and score archive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("layout_fixture", False):
        raise ValueError("Synthetic fixture cannot enter paper figures")
    core = verify(args.runs, args.protocol)
    expected = {row["tag"] for row in core["runs"]}
    if (set(report["provenance"]) != expected or report["protocol_sha256"] != core["protocol_sha256"]
        or report["score_settings_sha256"] != digest(args.scores / "settings.json")):
        raise ValueError("Report/core/score provenance mismatch")
    complete = json.loads((args.scores / "completed.json").read_text(encoding="utf-8"))
    if not complete["eligible_for_paper"] or complete["split"] != report["split"] or set(complete["runs"]) != expected:
        raise ValueError("Incomplete/smoke scoring cannot enter figures")
    for row in core["runs"]:
        if report["provenance"][row["tag"]]["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("Reported checkpoint changed")
        if digest(args.scores / (row["tag"]+".csv.gz")) != report["provenance"][row["tag"]]["csv_sha256"]:
            raise ValueError("Reported CSV changed")
    args.out.mkdir(parents=True, exist_ok=True)
    signature = {"report_sha256": digest(args.report), "protocol_sha256": digest(args.protocol), "split": report["split"]}
    path = args.out / "settings.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != signature:
        raise ValueError("Figure data changed; use a new output folder")
    path.write_text(json.dumps(signature, indent=2)+"\n", encoding="utf-8")
    report_figures(report, args.out)
    histories = training_figure(core, args.runs, args.out, report)
    receipt = {"eligible_for_paper": True, **signature, "render_source_sha256": digest(Path(__file__)), "history_sha256": histories,
               "figures": {path.name: digest(path) for path in sorted(args.out.iterdir()) if path.suffix in (".png", ".svg")},
               "visual_qa_status": "not yet inspected; successful rendering does not prove publication-ready layout"}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print("CORE FIGURES RENDERED; visual QA still required", args.out, flush=True)


if __name__ == "__main__":
    main()
