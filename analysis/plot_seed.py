"""Source-rebuilt five-seed figures; synthetic layouts can never be paper results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_mechanism_core import available_mb
from verify_mechanism_core import CONFIGS, ROOT, digest

SEEDS = (0, 1, 2, 3, 4)
PRIMARY = ("M1-M0", "M4-M1", "M4-M5")
CONTRASTS = PRIMARY + ("M1-M2", "M3-M0", "M4-M0", "M5-M2")
METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
           "time_to_collision_within_bound", "ego_progress", "comfort", "driving_direction_compliance")
LABELS = ("PDMS", "NC", "DAC", "TTC", "EP", "C", "DDC")
FIGURES = ("five_seed_contrasts", "seed_extension_diagnostics", "five_seed_strata", "five_seed_submetrics")
SEED_COLORS = ("#444444", "#0072b2", "#009e73", "#e69f00", "#cc79a7")


def effect_text(value):
    return "0.000" if round(value, 3) == 0 else f"{value:+.3f}"


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite five-seed report")
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def verify_report(report, rebuild):
    """Production callback audits models, both score archives and actual source arrays."""
    if (report.get("layout_fixture") or report.get("schema") != "gated-five-seed-analysis-v1"
            or report.get("split") not in ("validation", "confirmation", "navtest")):
        raise ValueError("Synthetic/unknown five-seed report")
    expected = {f"{config}_f1_steps_s{seed}" for config in CONFIGS for seed in SEEDS}
    if (set(report.get("provenance", {})) != expected or report.get("paired_seeds") != list(SEEDS)
            or (report.get("reused_core_runs"), report.get("added_runs")) != (18, 12)
            or report.get("gate", {}).get("required") is not True
            or not report.get("analysis_source_sha256")
            or any(type(report.get(key)) is not int or report[key] <= 0 for key in ("n_scenes", "n_logs", "n_sessions"))):
        raise ValueError("Incomplete five-seed provenance/coverage/gate")
    if not report.get("base_protocol_sha256") or not report.get("extension_protocol_sha256") or report["base_protocol_sha256"] == report["extension_protocol_sha256"]:
        raise ValueError("Distinct original and extension training protocols required")
    for tag, row in report["provenance"].items():
        expected_hash = report["base_protocol_sha256"] if int(tag[-1]) < 3 else report["extension_protocol_sha256"]
        if row.get("training_protocol_sha256") != expected_hash:
            raise ValueError("Original/added training protocol relabelled")
    finite_tree(report)
    summary = report["summary"]
    if set(summary["cells"]) != set(CONFIGS) or set(summary["contrasts"]) != set(CONTRASTS):
        raise ValueError("All six cells and seven paired contrasts required")
    seed_keys = {str(seed) for seed in SEEDS}
    for row in summary["cells"].values():
        if set(row["mean"]) != set(METRICS) or set(row["seed_metrics"]) != seed_keys or any(set(values) != set(METRICS) for values in row["seed_metrics"].values()):
            raise ValueError("All five seeds and seven cell metrics required")
    for name, row in summary["contrasts"].items():
        if set(row["seed_effects"]) != seed_keys or set(row["metric_effects"]) != set(METRICS) or row["primary"] != (name in PRIMARY):
            raise ValueError("All five effects and primary/secondary identities required")
        if name in PRIMARY and not math.isclose(row["multiplicity_interval"]["alpha"], .05 / 3):
            raise ValueError("Registered three-primary adjusted interval required")
    actual = rebuild()
    if report != actual:
        raise ValueError("Report differs from statistics rebuilt from actual artifacts")
    return actual


def grouped_effects(row):
    """Descriptive fitted-seed subsets, not new independent confidence intervals."""
    values = row["seed_effects"]
    return (math.fsum(values[str(seed)] for seed in (0, 1, 2)) / 3,
            math.fsum(values[str(seed)] for seed in (3, 4)) / 2,
            math.fsum(values[str(seed)] for seed in SEEDS) / 5)


def layout_fixture():
    def interval(value, alpha=.05, estimable=True):
        width = .018 if alpha == .05 else .027
        return {"mean": value, "low": value-width if estimable else None, "high": value+width if estimable else None,
                "alpha": alpha, "draws": 10000, "n_clusters": 4 if estimable else 1}
    contrasts = {}
    for index, name in enumerate(CONTRASTS):
        value = (index-3) * .012
        contrasts[name] = {"primary": name in PRIMARY, "metric_effects": {metric: value * (1 if j == 0 else (j-3)/3) for j, metric in enumerate(METRICS)},
            "seed_effects": {str(seed): value+(seed-2)*.018 for seed in SEEDS}, "session_interval": interval(value)}
        if name in PRIMARY:
            contrasts[name]["multiplicity_interval"] = interval(value, .05/3)
    cells = {}
    for index, config in enumerate(CONFIGS):
        cells[config] = {"mean": {metric: .55 + index*.02 for metric in METRICS},
                         "seed_metrics": {str(seed): {metric: .55+index*.02+(seed-2)*.005 for metric in METRICS} for seed in SEEDS}}
    groups = {}
    for name, count, logs, sessions in (("all", 10, 6, 4), ("near_objects", 6, 4, 3),
                                      ("potential_crossing_proxy", 3, 2, 2), ("city:single-session", 2, 1, 1), ("empty", 0, 0, 0)):
        groups[name] = {"n_scenes": count, "n_logs": logs, "n_sessions": sessions, "contrasts": {}}
        if count:
            groups[name]["contrasts"] = {label: {"metric_effects": dict(contrasts[label]["metric_effects"]),
                "session_interval": interval(contrasts[label]["metric_effects"]["score"], estimable=sessions > 1)} for label in PRIMARY}
    return {"layout_fixture": True, "split": "LAYOUT SMOKE", "n_scenes": 10, "n_logs": 6, "n_sessions": 4,
            "summary": {"cells": cells, "contrasts": contrasts, "subgroups": groups}}


def save(fig, out, name, report, caption):
    import matplotlib.pyplot as plt
    fig.text(.5, -.035, caption, ha="center", va="top", fontsize=9)
    if report.get("layout_fixture"):
        fig.text(.5, -.13, "LAYOUT SMOKE ONLY - SYNTHETIC NUMBERS - NOT A PAPER RESULT", ha="center", va="top", color="#a00000", fontsize=11)
    for suffix in ("png", "svg"):
        path = out / (name + "." + suffix)
        if path.exists():
            raise ValueError("Preserve existing/partial five-seed figures; use a fresh output version")
        with path.open("xb") as stream:
            fig.savefig(stream, format=suffix, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_figures(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [report["summary"]["contrasts"][name] for name in CONTRASTS]
    labels = [name + (" *" if name in PRIMARY else "") for name in CONTRASTS]
    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    for index, row in enumerate(rows):
        for key, width in (("multiplicity_interval", 1), ("session_interval", 4)):
            if key in row and row[key]["low"] is not None:
                axis.plot([row[key]["low"], row[key]["high"]], [index, index], color="#0072b2", lw=width)
        axis.scatter(row["metric_effects"]["score"], index, marker="D", color="#0072b2", s=50, zorder=4)
        for seed, color in zip(SEEDS, SEED_COLORS):
            axis.scatter(row["seed_effects"][str(seed)], index+(seed-2)*.08, color=color, marker="x", s=25,
                         label="seed " + str(seed) if index == 0 else None, zorder=5)
    axis.axvline(0, color=".55", lw=1, linestyle="--")
    axis.set(yticks=range(len(rows)), yticklabels=labels, ylim=(len(rows)-.5, -.5), xlabel="Paired PDMS difference",
             title=report["split"] + ": all five paired seeds; * = original primary contrast")
    axis.legend(ncol=5, fontsize=8, loc="best")
    axis.grid(axis="x", alpha=.2)
    save(fig, out, FIGURES[0], report, "Diamond: five-seed mean; crosses: seeds 0-4. Thick: 95% session CI; thin: 98.33% CI for * only.\nIntervals condition on these fitted seeds; outcome gating does not create fresh independent confirmation.")

    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    names = ("original seeds 0/1/2", "added seeds 3/4", "all five seeds")
    for index, row in enumerate(rows):
        values = grouped_effects(row)
        axis.plot([min(values), max(values)], [index, index], color=".8", lw=1)
        for value, offset, color, marker, name in zip(values, (-.13, 0, .13), ("#0072b2", "#e69f00", "#009e73"), ("o", "s", "D"), names):
            axis.scatter(value, index+offset, color=color, marker=marker, s=45, label=name if index == 0 else None)
    axis.axvline(0, color=".55", lw=1, linestyle="--")
    axis.set(yticks=range(len(rows)), yticklabels=labels, ylim=(len(rows)-.5, -.5), xlabel="Mean paired PDMS difference",
             title=report["split"] + ": descriptive original-three vs added-two diagnostic")
    axis.legend(ncol=3, fontsize=9, loc="best")
    axis.grid(axis="x", alpha=.2)
    save(fig, out, FIGURES[1], report, "Symbols are fitted-seed subset means; gray segments only connect their range, not confidence intervals.\nNo new holdout, population-seed uncertainty or significance claim follows from this descriptive comparison.")

    groups = report["summary"]["subgroups"]
    fig, axes = plt.subplots(1, 3, figsize=(17, max(4.8, .42*len(groups))), sharey=True, constrained_layout=True)
    group_labels = [f"{name.replace('_', ' ')} (n={row['n_scenes']}, logs={row['n_logs']}, sessions={row['n_sessions']})" for name, row in groups.items()]
    for axis, label in zip(axes, PRIMARY):
        for index, group in enumerate(groups.values()):
            row = group["contrasts"].get(label)
            if row is None:
                axis.text(.5, index, "no scenes", transform=axis.get_yaxis_transform(), ha="center", va="center", fontsize=8, color=".5")
                continue
            bounds = row["session_interval"]
            if bounds["low"] is not None:
                axis.plot([bounds["low"], bounds["high"]], [index, index], color="#0072b2", lw=1)
            axis.scatter(row["metric_effects"]["score"], index, marker="o" if bounds["low"] is not None else "x", color="#0072b2", s=28)
        axis.axvline(0, color=".55", lw=1, linestyle="--")
        axis.set(title=label, xlabel="Paired PDMS difference")
        axis.grid(axis="x", alpha=.2)
    axes[0].set(yticks=range(len(groups)), yticklabels=group_labels, ylim=(len(groups)-.5, -.5))
    fig.suptitle(report["split"] + ": overlapping input-defined strata; five fitted seeds", fontsize=11)
    save(fig, out, FIGURES[2], report, "Exploratory 95% session intervals; x = not estimable (<2 sessions); empty groups are retained.\nCrossing is a straight-corridor proxy. No causal interaction or deployment-safety interpretation.")

    values = np.array([[row["metric_effects"][metric] for metric in METRICS] for row in rows])
    limit = max(.005, float(np.abs(values).max()))
    fig, axis = plt.subplots(figsize=(11, 5.3), constrained_layout=True)
    drawing = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(METRICS)):
            axis.text(j, i, effect_text(values[i, j]), ha="center", va="center", fontsize=9,
                      color="white" if abs(values[i, j]) > .6*limit else "black")
    axis.set(xticks=range(len(METRICS)), xticklabels=LABELS, yticks=range(len(rows)), yticklabels=labels,
             title=report["split"] + ": all seven submetric differences (five-seed means)")
    fig.colorbar(drawing, ax=axis, label="Paired submetric difference")
    save(fig, out, FIGURES[3], report, "Symmetric scale zooms to observed effects; this panel has no uncertainty bars.\nDDC is diagnostic only in pinned NAVSIM v1; no matched sensor-SOTA or general driving-safety claim.")


def write_figures(report, out, report_hash, renderer=render_figures, source_check=None):
    """Refuse overwrite, changed bindings or partial renders; exact completed bytes may reuse."""
    smoke = bool(report.get("layout_fixture"))
    if not smoke and not report_hash:
        raise ValueError("Scientific render must bind a source report hash")
    signature = {"kind": "five-seed-figures-v1", "layout_smoke": smoke, "split": report["split"],
                 "report_sha256": report_hash, "renderer_source_sha256": digest(Path(__file__)), "expected_figures": list(FIGURES)}
    expected_files = {name + "." + suffix for name in FIGURES for suffix in ("png", "svg")}
    settings, receipt_path = out / "settings.json", out / "receipt.json"
    if out.exists() and any(out.iterdir()):
        if not settings.exists() or not receipt_path.exists() or json.loads(settings.read_text(encoding="utf-8")) != signature:
            raise ValueError("Preserve unrelated/partial/different five-seed render; use a fresh output version")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (any(receipt.get(key) != value for key, value in signature.items())
                or receipt.get("eligible_for_paper") is not (not smoke) or set(receipt.get("figures", {})) != expected_files):
            raise ValueError("Incomplete/different five-seed render receipt")
        for name, checksum in receipt["figures"].items():
            if digest(out / name) != checksum:
                raise ValueError("Completed five-seed figure changed")
        if source_check is not None and not source_check():
            raise ValueError("Source report changed; do not accept this render")
        return receipt
    out.mkdir(parents=True, exist_ok=True)
    with settings.open("x", encoding="utf-8") as stream:
        json.dump(signature, stream, indent=2)
    renderer(report, out)
    if source_check is not None and not source_check():
        raise ValueError("Source report changed while rendering; no completion receipt")
    receipt = dict(signature, eligible_for_paper=not smoke, figures={name: digest(out / name) for name in sorted(expected_files)},
                   visual_qa_status="pending inspection", scope="Synthetic layout only" if smoke else
                   "Audited source-rebuilt outcome-gated five-seed sensitivity; no fresh confirmation or deployment-safety result")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--report", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout-smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.layout_smoke:
        if args.report or not args.out.resolve().is_relative_to((ROOT / "tmp").resolve()):
            raise ValueError("Synthetic layout requires tmp-only output and no scientific report")
    elif args.report is None:
        raise ValueError("Source-audited five-seed report required")
    free = available_mb()
    required = 2000 if args.layout_smoke else 6000
    if free is not None and free < required:
        raise RuntimeError(f"Memory guard: needs {required} MB before imports/rebuild; no renderer launched")
    if args.layout_smoke:
        report, report_hash = layout_fixture(), None
    else:
        from analyze_seed import build_report
        report_hash = digest(args.report)
        report = json.loads(args.report.read_text(encoding="utf-8"))
        verify_report(report, lambda: build_report(args.workspace, ROOT / "results/mechanism_v2", report["split"]))
    receipt = write_figures(report, args.out, report_hash,
                           source_check=(lambda: digest(args.report) == report_hash) if args.report else None)
    print("FIVE-SEED FIGURES RENDERED/VERIFIED; visual QA and final audit remain", receipt["split"], flush=True)


if __name__ == "__main__":
    main()
