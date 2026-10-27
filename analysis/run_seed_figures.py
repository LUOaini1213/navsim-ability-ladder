"""Guarded three-split five-seed figures after actual added-seed evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_city_evaluation import completed_attempt
from run_mechanism_evaluation import acquire, upstream_binding
from run_seed_training import SeedPipeline
from verify_mechanism_core import ROOT


def figure_commands(python, workspace, results, out_root=None):
    out_root = results / "seed_figures" if out_root is None else out_root
    return [(split, [python, "-u", ROOT / "analysis/plot_seed.py", "--workspace", workspace,
                     "--report", results / "seed_analysis" / split / "report.json",
                     "--out", out_root / split]) for split in ("validation", "confirmation", "navtest")]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results/mechanism_v2/seed_figures",
                    help="Use a fresh output version after an interrupted/changed render; never overwrite")
    args = ap.parse_args()
    results = ROOT / "results/mechanism_v2"
    pipeline = SeedPipeline("seed_figures", results / "seed_figure_status.json", args.workspace / "exp/mechanism_v2/seed_figure_logs",
                            results / "run_attempts", "Four source-rebuilt figures per split; actual PNG/SVG inspection and final project audit remain")
    lock = None
    try:
        lock = acquire(args.workspace / "exp/mechanism_v2/core.lock", wait=args.wait,
                       on_wait=lambda: print("SEED FIGURES WAIT: heavy-job lock owned", flush=True))
        attempt, binding = completed_attempt(results / "seed_evaluation_status.json", "seed_evaluation_completed")
        files = [attempt, results / "seed_protocol/registration.json", results / "seed_protocol/active/protocol.json",
                 results / "seed_protocol/active/completed.json"]
        files.extend(ROOT / "analysis" / name for name in ("run_seed_figures.py", "plot_seed.py", "analyze_seed.py", "analyze_mechanism.py",
                     "run_seed_training.py", "run_city_evaluation.py", "seed_models.py", "seed_followup.py", "predict_seed.py",
                     "verify_mechanism_core.py", "run_mechanism_core.py", "runtime.py", "guarded_pipeline.py", "resource_guard.py"))
        files.extend(results / "seed_analysis" / split / "report.json" for split in ("validation", "confirmation", "navtest"))
        pipeline.bind(files, existing={**binding, **upstream_binding(args.workspace, results / "runtime_snapshot/runtime.json")})
        for split, command in figure_commands(sys.executable, args.workspace, results, args.out_root):
            pipeline.stage("figures_" + split, command)
        pipeline.finish("seed_figures_rendered_visual_qa_pending")
        print("FIVE-SEED FIGURES COMPLETE; mandatory scientific PNG/SVG visual QA and final audit remain", flush=True)
    except BaseException as error:
        pipeline.fail(error)
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
