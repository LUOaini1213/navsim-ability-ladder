"""Reproduce archived PDMS/subscores through the new frozen-feature score path."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from mechanism_data import load_features
from mechanism_scoring import make_engine, score_one
from runtime import ROOT, configure


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--scenes", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "results/audit_v2/score_path.json")
    args = ap.parse_args()
    if args.scenes < 1:
        raise ValueError("At least one comparison scene required")
    workspace = configure()
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.agents.ladder_features import assemble
    from navsim.agents.ladder_mlp_agent import load_run
    torch.set_num_threads(2)
    data = load_features(workspace / "exp/features/features_navtest.npz")
    index = {str(token): i for i, token in enumerate(data["token"])}
    tokens = sorted(index)
    chosen = [tokens[int(i)] for i in np.unique(np.linspace(0, len(tokens)-1, args.scenes).astype(int))]
    cache = MetricCacheLoader(workspace / "exp/metric_cache_navtest")
    simulator, scorer = make_engine(workspace)
    ledger = []
    for tag in ("pose_kin_s0", "pose_agents_s0", "pose_map_s0", "pose_map_agents_s0"):
        model, mu, sd, norm = load_run(str(workspace / "exp/ladder" / tag))
        with gzip.open(ROOT / "results/navtest/per_scene" / ("l_"+tag+".csv.gz"), "rt", encoding="utf-8") as stream:
            archived = {row["token"]: row for row in csv.DictReader(stream)}
        for token in chosen:
            i = index[token]
            x = assemble(data["ego"][i], data["cl"][i], data["agents"][i], data["lead"][i], norm["see_map"], norm["see_agents"])
            with torch.no_grad():
                poses = model(torch.from_numpy((x-mu)/sd)[None]).numpy().reshape(8, 3)
            fresh = score_one(cache.get_from_token(token), poses, simulator, scorer)
            errors = {key: abs(value-float(archived[token][key])) for key, value in fresh.items()}
            ledger.append({"run": tag, "token": token, "errors": errors, "passed": max(errors.values()) <= 1e-5})
        print("score parity", tag, len(chosen), "scenes", flush=True)
    report = {"passed": all(row["passed"] for row in ledger), "comparisons": len(ledger),
              "sampling": "Evenly spaced sorted tokens, fixed before score comparison",
              "maximum_metric_error": max(max(row["errors"].values()) for row in ledger), "ledger": ledger}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("SCORE PATH", "PASS" if report["passed"] else "FAIL", report["maximum_metric_error"], flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
