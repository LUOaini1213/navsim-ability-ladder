"""Render source-rebuilt city/data-budget figures; smoke layouts are ineligible."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_mechanism_core import available_mb
from verify_mechanism_core import ROOT, digest

METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress", "comfort", "driving_direction_compliance")
LABELS = ("PDMS", "NC", "DAC", "TTC", "EP", "C", "DDC")
CITY_CONTRASTS = ("C1-C2", "C4-C5", "C4-C5-C1+C2")
CURVE_CELLS = tuple(f"{budget}/f{fraction:g}" for budget in ("steps", "epochs") for fraction in (.1, .3, 1.))
FIGURES = {"city": ("city_contrasts", "city_strata", "city_submetrics"),
           "curves": ("learning_curves", "encoder_curves", "budget_interactions", "curve_submetrics")}
COLORS = {"M1": "#0072b2", "M4": "#009e73"}


def verify_report(kind, report, rebuild):
    """Rebuild must inspect actual artifacts; equality alone is not a file audit."""
    if kind not in FIGURES or report.get("layout_fixture") or report.get("split") not in ("validation", "confirmation", "navtest"):
        raise ValueError("Synthetic/unknown reports cannot enter scientific follow-up figures")
    expected = {f"C{number}_f1_steps_s{seed}" for number in (1, 2, 4, 5) for seed in (0, 1, 2)} if kind == "city" else {
        f"{config}_f{fraction:g}_{budget}_s{seed}" for config in ("M1", "M4") for seed in (0, 1, 2)
        for fraction, budget in ((.1, "steps"), (.3, "steps"), (.1, "epochs"), (.3, "epochs"), (1., "steps"))}
    if (set(report.get("provenance", {})) != expected or not report.get("analysis_source_sha256")
        or report.get("n_scenes", 0) <= 0 or report.get("n_sessions", 0) <= 0):
        raise ValueError("Incomplete follow-up report provenance/coverage")
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite report value")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    finite(report)
    actual = rebuild()
    if report != actual:
        raise ValueError("Figure report does not equal statistics rebuilt from actual artifacts")
    return actual


def interval(value, alpha=.05, estimable=True):
    return {"mean": value, "low": value - .018 if estimable else None, "high": value + .021 if estimable else None,
            "alpha": alpha, "draws": 10000, "n_clusters": 4 if estimable else 1}


def effect(value, adjusted_key, alpha):
    return {"metric_effects": {metric: value * (1 if index == 0 else (index - 3) / 3) for index, metric in enumerate(METRICS)},
            "seed_effects": {str(seed): value + (seed - 1) * .012 for seed in (0, 1, 2)},
            "session_interval": interval(value), adjusted_key: interval(value, alpha)}


def layout_fixture(kind):
    if kind not in FIGURES:
        raise ValueError("Unknown follow-up layout")
    report = {"layout_fixture": True, "split": "LAYOUT SMOKE", "n_scenes": 10, "n_sessions": 4,
              "summary": {"cells": {}, "contrasts": {}, "subgroups": {}}, "scope": "Synthetic layout only; no trained-model or driving evidence"}
    summary = report["summary"]
    if kind == "city":
        summary["contrasts"] = {name: effect(value, "auxiliary_family_interval", .05 / 3)
                                for name, value in zip(CITY_CONTRASTS, (-.025, .015, .04))}
        for name, scenes, sessions in (("all", 10, 4), ("near_objects", 6, 3), ("potential_crossing_proxy", 3, 2),
                                        ("city:single-session", 2, 1), ("empty", 0, 0)):
            rows = {}
            if scenes:
                for label in CITY_CONTRASTS:
                    value = summary["contrasts"][label]["metric_effects"]["score"]
                    rows[label] = {"metric_effects": {"score": value}, "session_interval": interval(value, estimable=sessions > 1)}
            summary["subgroups"][name] = {"n_scenes": scenes, "n_sessions": sessions, "contrasts": rows}
    else:
        interactions = {}
        for budget in ("steps", "epochs"):
            for fraction in (.1, .3, 1.):
                key = f"{budget}/f{fraction:g}"
                value = .04 - .06 * fraction + (.005 if budget == "epochs" and fraction != 1. else 0)
                cell = {"fraction": fraction, "budget": budget, "n_train": round(10000 * fraction),
                        "reused_full_data_anchor": fraction == 1., "models": {}, "M4-M1": effect(value, "unused", .05)}
                for config in ("M1", "M4"):
                    score = .55 + .1 * fraction + (value if config == "M4" else 0)
                    cell["models"][config] = {"metric_means": {"score": score}, "score_interval": interval(score),
                                               "seed_metrics": {str(seed): {"score": score + (seed - 1) * .01} for seed in (0, 1, 2)}}
                summary["cells"][key] = cell
                if fraction != 1.:
                    interactions[key + "-full"] = effect(value + .02, "four_contrast_interval", .05 / 4)
        summary["small_minus_full_encoder_effect"] = interactions
    return report


def save(fig, out, name, report, caption):
    import matplotlib.pyplot as plt
    fig.text(.5, -.04, caption, ha="center", va="top", fontsize=9)
    if report.get("layout_fixture"):
        fig.text(.5, -.12, "LAYOUT SMOKE ONLY - SYNTHETIC NUMBERS - NOT A PAPER RESULT", ha="center", va="top", color="#a00000", fontsize=11)
    for suffix in ("png", "svg"):
        temporary = out / (name + ".tmp." + suffix)
        fig.savefig(temporary, format=suffix, dpi=160, bbox_inches="tight")
        temporary.replace(out / (name + "." + suffix))
    plt.close(fig)


def forest(axis, labels, rows, adjusted_key=None):
    for i, row in enumerate(rows):
        if adjusted_key:
            bounds = row[adjusted_key]
            if bounds["low"] is not None:
                axis.plot([bounds["low"], bounds["high"]], [i, i], color="#0072b2", lw=1)
        bounds = row["session_interval"]
        if bounds["low"] is not None:
            axis.plot([bounds["low"], bounds["high"]], [i, i], color="#0072b2", lw=4)
        axis.scatter([row["metric_effects"]["score"]], [i], color="#0072b2", marker="D", s=55, zorder=4)
        for seed, offset in zip(("0", "1", "2"), (-.13, 0, .13)):
            axis.scatter(row["seed_effects"][seed], i + offset, color="#555555", marker="x", s=23, zorder=5)
    axis.axvline(0, color=".55", linestyle="--", lw=1)
    axis.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Paired PDMS difference", ylim=(len(labels) - .5, -.5))
    axis.grid(axis="x", alpha=.2)


def heatmap(report, rows, names, out, filename):
    import matplotlib.pyplot as plt
    import numpy as np
    values = np.array([[row["metric_effects"][metric] for metric in METRICS] for row in rows])
    limit = max(.005, float(np.abs(values).max()))
    fig, axis = plt.subplots(figsize=(11, max(3.7, .6 * len(rows))), constrained_layout=True)
    drawing = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(METRICS)):
            axis.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=9,
                      color="white" if abs(values[i, j]) > .6 * limit else "black")
    axis.set(xticks=range(len(METRICS)), xticklabels=LABELS, yticks=range(len(names)), yticklabels=names,
             title=report["split"] + ": paired submetric differences (three fitted seeds)")
    fig.colorbar(drawing, ax=axis, label="Mean paired submetric difference")
    save(fig, out, filename, report, "Color scale is symmetric and zoomed to observed effects; this panel has no uncertainty bars.\nDDC is diagnostic only (zero score weight in the pinned NAVSIM v1 scorer); no causal-safety interpretation.")


def city_figures(report, out):
    import matplotlib.pyplot as plt
    summary = report["summary"]
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    forest(axis, CITY_CONTRASTS, [summary["contrasts"][name] for name in CITY_CONTRASTS], "auxiliary_family_interval")
    axis.set_title(report["split"] + ": same-city, cross-session controls on identical eligible scenes")
    save(fig, out, "city_contrasts", report, "Diamond: three-seed score mean; crosses: individual seed effects. Thick: 95% session CI; thin: 98.33% auxiliary CI.\nConditional exploratory follow-up; not a physical counterfactual or global confirmatory family.")
    groups = summary["subgroups"]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, .42 * len(groups))), sharey=True, constrained_layout=True)
    labels = [f"{name.replace('_', ' ')} (n={row['n_scenes']}, sessions={row['n_sessions']})" for name, row in groups.items()]
    for axis, contrast in zip(axes, CITY_CONTRASTS):
        for i, row in enumerate(groups.values()):
            if contrast not in row["contrasts"]:
                axis.text(.5, i, "no scenes", transform=axis.get_yaxis_transform(), va="center", ha="center", color=".5", fontsize=8)
                continue
            value = row["contrasts"][contrast]
            bounds = value["session_interval"]
            if bounds["low"] is not None:
                axis.plot([bounds["low"], bounds["high"]], [i, i], color="#0072b2")
            axis.scatter(value["metric_effects"]["score"], i, color="#0072b2", marker="o" if bounds["low"] is not None else "x", s=28)
        axis.axvline(0, color=".6", linestyle="--", lw=1)
        axis.set(title=contrast, xlabel="Paired PDMS difference")
        axis.grid(axis="x", alpha=.2)
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels) - .5, -.5))
    fig.suptitle(report["split"] + ": overlapping input-defined strata; exploratory 95% session intervals", fontsize=11)
    save(fig, out, "city_strata", report, "x: interval not estimable (<2 original sessions); empty strata are retained. Crossing is a straight-corridor proxy.\nCommon-cohort exclusions must be read alongside the report; no full-city or physical-interaction claim.")
    heatmap(report, [summary["contrasts"][name] for name in CITY_CONTRASTS], CITY_CONTRASTS, out, "city_submetrics")


def curve_figures(report, out):
    import matplotlib.pyplot as plt
    summary = report["summary"]
    for filename, difference in (("learning_curves", False), ("encoder_curves", True)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True, constrained_layout=True)
        for axis, budget in zip(axes, ("steps", "epochs")):
            for config in (("M4-M1",) if difference else ("M1", "M4")):
                color = "#7b3294" if difference else COLORS[config]
                means = []
                for fraction in (.1, .3, 1.):
                    cell = summary["cells"][f"{budget}/f{fraction:g}"]
                    row = cell["M4-M1"] if difference else cell["models"][config]
                    mean = row["metric_effects"]["score"] if difference else row["metric_means"]["score"]
                    bounds = row["session_interval"] if difference else row["score_interval"]
                    x = fraction * 100
                    if bounds["low"] is not None:
                        axis.plot([x, x], [bounds["low"], bounds["high"]], color=color, lw=1.5)
                    seed_values = row["seed_effects"] if difference else {seed: values["score"] for seed, values in row["seed_metrics"].items()}
                    axis.scatter([x] * 3, [seed_values[str(seed)] for seed in (0, 1, 2)], color=color, marker="x", alpha=.45, s=25)
                    axis.scatter(x, mean, color=color, marker="D" if fraction != 1. else "o", s=60, zorder=4)
                    means.append(mean)
                axis.plot([10, 30, 100], means, color=color, label=config + (" set - concat" if difference else (" concat" if config == "M1" else " set")))
            if difference:
                axis.axhline(0, color=".6", linestyle="--", lw=1)
            axis.set(xticks=[10, 30, 100], xlabel="Registered training fraction (%)", title="Fixed 13,000 updates" if budget == "steps" else "Equal-epoch sensitivity")
            axis.grid(alpha=.2)
        axes[0].set_ylabel("Paired M4-M1 PDMS difference" if difference else "PDMS (mean of fitted-seed scores)")
        axes[1].legend(loc="best", fontsize=9)
        fig.suptitle(report["split"] + ": registered data/update-budget follow-up", fontsize=12)
        save(fig, out, filename, report, "Diamonds: seed-score means; crosses: individual seeds; vertical segments: 95% original-session intervals, conditional on fitted seeds.\nThe circular 100% point is the same reused full-data anchor in both panels, not independent replication; axes zoom to observed values.")
    names = [f"{budget}/f{fraction:g}-full" for budget in ("steps", "epochs") for fraction in (.1, .3)]
    fig, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    forest(axis, names, [summary["small_minus_full_encoder_effect"][name] for name in names], "four_contrast_interval")
    axis.set_title(report["split"] + ": (M4-M1) at smaller data minus (M4-M1) at full data")
    save(fig, out, "budget_interactions", report, "Diamond: paired difference of differences; crosses: seeds. Thick: 95% session CI; thin: 98.75% four-contrast sensitivity CI.\nOutcome-gated, fitted-seed-conditional sensitivity analysis; no unconditional global confirmatory guarantee.")
    heatmap(report, [summary["cells"][name]["M4-M1"] for name in CURVE_CELLS], CURVE_CELLS, out, "curve_submetrics")


def render_figures(kind, report, out):
    import matplotlib
    matplotlib.use("Agg")
    if kind == "city":
        city_figures(report, out)
    elif kind == "curves":
        curve_figures(report, out)
    else:
        raise ValueError("Unknown figure kind")


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--kind", choices=FIGURES, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--scores", type=Path)
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout-smoke", action="store_true")
    args = ap.parse_args()
    if args.layout_smoke:
        if not args.out.resolve().is_relative_to((ROOT / "tmp").resolve()) or args.report or args.scores or args.predictions:
            raise ValueError("Smoke figures require a tmp-only destination and no scientific inputs")
        report = layout_fixture(args.kind)
        report_hash = None
    else:
        if args.report is None or args.scores is None or (args.kind == "city" and args.predictions is None):
            raise ValueError("Actual report, score archive and city predictions required")
        report_hash = digest(args.report)
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if args.kind == "city":
            from analyze_city import build_report
            rebuild = lambda: build_report(args.scores, args.predictions, args.workspace)
        else:
            from analyze_mechanism_curves import build_report
            rebuild = lambda: build_report(args.scores, args.workspace / "exp/mechanism_v2/models", args.protocol, workspace=args.workspace)
        verify_report(args.kind, report, rebuild)
    free = available_mb()
    if free is not None and free < 2000:
        raise RuntimeError("Less than 2,000 MB available for rendering; no figure launched")
    signature = {"kind": args.kind, "layout_smoke": args.layout_smoke, "report_sha256": report_hash,
                 "renderer_source_sha256": digest(Path(__file__)), "expected_figures": list(FIGURES[args.kind])}
    args.out.mkdir(parents=True, exist_ok=True)
    settings = args.out / "settings.json"
    if settings.exists():
        if json.loads(settings.read_text(encoding="utf-8")) != signature:
            raise ValueError("Different follow-up render source/report; preserve prior outputs")
    elif any(args.out.iterdir()):
        raise ValueError("Refusing an unrelated nonempty figure directory")
    else:
        settings.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    receipt_path = args.out / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_files = {name + "." + suffix for name in FIGURES[args.kind] for suffix in ("png", "svg")}
        if (any(receipt.get(key) != value for key, value in signature.items())
            or receipt.get("eligible_for_paper") != (not args.layout_smoke) or set(receipt.get("figures", {})) != expected_files):
            raise ValueError("Incomplete/different follow-up render receipt")
        for name, checksum in receipt["figures"].items():
            if digest(args.out / name) != checksum:
                raise ValueError("Completed follow-up figure changed")
        print("FOLLOW-UP FIGURES VERIFIED", args.out, flush=True)
        return
    render_figures(args.kind, report, args.out)
    if not args.layout_smoke and digest(args.report) != report_hash:
        raise ValueError("Source report changed while rendering")
    receipt = dict(signature, eligible_for_paper=not args.layout_smoke,
                   scope="Layout-only synthetic fixture" if args.layout_smoke else "Audited source-rebuilt conditional follow-up figures; visual QA still required",
                   figures={name + "." + suffix: digest(args.out / (name + "." + suffix)) for name in FIGURES[args.kind] for suffix in ("png", "svg")},
                   visual_qa_status="pending inspection")
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    print("FOLLOW-UP FIGURES RENDERED", args.kind, args.out, flush=True)


if __name__ == "__main__":
    main()
