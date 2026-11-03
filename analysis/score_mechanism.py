"""Score aligned prediction files with per-token SQLite recovery and fail-fast errors."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from mechanism_data import digest
from mechanism_scoring import make_engine, score_one
from runtime import ROOT, configure


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--predictions", type=Path, nargs="+", required=True)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="Smoke subset; marked ineligible, never reported as full evaluation")
    args = ap.parse_args()
    workspace = configure()
    from navsim.common.dataloader import MetricCacheLoader
    predictions = {}
    all_tokens = None
    reference_logs = None
    provenance = {}
    for path in args.predictions:
        with np.load(path, allow_pickle=False) as archive:
            tokens, logs, poses = archive["token"], archive["log_name"], archive["poses"]
            meta = json.loads(str(archive["meta"]))
        if (len(set(tokens)) != len(tokens) or len(logs) != len(tokens) or poses.shape != (len(tokens), 8, 3)
            or meta["scenes"] != len(tokens) or not np.isfinite(poses).all()):
            raise ValueError("Invalid predictions " + str(path))
        run = meta["run"]
        if run in predictions:
            raise ValueError("Duplicate run: " + run)
        if all_tokens is not None and set(tokens) != all_tokens:
            raise ValueError("Predictions do not cover identical scenes")
        all_tokens = set(tokens)
        token_logs = {str(token): str(log) for token, log in zip(tokens, logs)}
        if reference_logs is not None and token_logs != reference_logs:
            raise ValueError("Prediction token/log alignment differs")
        reference_logs = token_logs
        predictions[run] = {str(token): (str(log), pose) for token, log, pose in zip(tokens, logs, poses)}
        provenance[run] = {"predictions_sha256": digest(path), "meta": meta}
    splits = {value["meta"]["split"] for value in provenance.values()}
    protocols = {value["meta"]["protocol_sha256"] for value in provenance.values()}
    if len(splits) != 1 or len(protocols) != 1:
        raise ValueError("Mixed split/protocol")
    split = next(iter(splits))
    cache_path = args.cache or workspace / "exp" / ("metric_cache_navtest" if split == "navtest" else "mechanism_v2/metric_cache")
    loader = MetricCacheLoader(cache_path)
    tokens = sorted(all_tokens)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Positive smoke count required")
        tokens = tokens[:args.limit]
    missing = set(tokens)-set(loader.tokens)
    if missing:
        raise ValueError(f"Missing {len(missing)} metric caches; no silent skipping")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {"runs": provenance, "split": split, "cache_root": str(cache_path), "limit": args.limit,
                "scoring_config_sha256": digest(workspace / "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"),
                "wrapper_source_sha256": {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")}}
    settings_path = args.out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Scoring settings changed; use a new output directory")
    settings_path.write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    db = sqlite3.connect(str(args.out / "scores.sqlite"))
    db.execute("CREATE TABLE IF NOT EXISTS scores (run TEXT, token TEXT, record TEXT, PRIMARY KEY(run,token))")
    existing = set(db.execute("SELECT run,token FROM scores"))
    simulator, scorer = make_engine(workspace)
    started = time.monotonic()
    for number, token in enumerate(tokens):
        todo = [run for run in predictions if (run, token) not in existing]
        if not todo:
            continue
        try:
            cache = loader.get_from_token(token)
        except Exception as error:
            db.commit()
            with open(args.out / "failures.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"run": "__cache_load__", "token": token, "exception": repr(error)})+"\n")
            raise
        for run in todo:
            log, pose = predictions[run][token]
            try:
                values = score_one(cache, pose, simulator, scorer)
                if not all(np.isfinite(value) for value in values.values()):
                    raise FloatingPointError("Nonfinite metric")
                record = dict(token=token, log_name=log, valid=True, **values)
                db.execute("INSERT INTO scores VALUES (?,?,?)", (run, token, json.dumps(record)))
            except Exception as error:
                db.commit()
                with open(args.out / "failures.jsonl", "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"run": run, "token": token, "exception": repr(error)})+"\n")
                raise
        if (number+1) % 100 == 0:
            db.commit()
            print("scored", number+1, "/", len(tokens), "scenes x", len(predictions), "runs; elapsed", round(time.monotonic()-started), "s", flush=True)
    db.commit()
    summaries = {}
    for run in predictions:
        rows = [json.loads(row[0]) for row in db.execute("SELECT record FROM scores WHERE run=? ORDER BY token", (run,))]
        if {row["token"] for row in rows} != set(tokens):
            raise ValueError("Incomplete/extra results for " + run)
        path = args.out / (run+".csv.gz")
        with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summaries[run] = {"n": len(rows), "mean_score": sum(row["score"] for row in rows)/len(rows),
                          "csv_sha256": digest(path)}
    failure_path = args.out / "failures.jsonl"
    recorded_failures = sum(1 for line in failure_path.read_text(encoding="utf-8").splitlines() if line) if failure_path.exists() else 0
    complete = {"eligible_for_paper": args.limit is None and all(p["meta"]["eligible_for_paper"] for p in provenance.values()),
                "settings_sha256": digest(settings_path), "split": split, "runs": summaries,
                "elapsed_seconds": time.monotonic()-started, "final_invalid_scores": 0,
                "attempt_failures_recorded": recorded_failures}
    (args.out / "completed.json").write_text(json.dumps(complete, indent=2)+"\n", encoding="utf-8")
    db.close()
    print("SCORING COMPLETED", json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
