"""Build separate, resumable v1 metric caches for validation/confirmation only."""
from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

from mechanism_data import digest, load_features, split_indices
from runtime import ROOT, configure


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--split", choices=["validation", "confirmation"], required=True)
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, help="Explicit smoke subset, stored in a separate cache root")
    args = ap.parse_args()
    workspace = configure()
    from build_features import scene_filter_for
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = protocol["sources"]["navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Features changed")
    data = load_features(source["path"])
    indices = split_indices(data, protocol, args.split)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Positive smoke count required")
        indices = indices[:args.limit]
    out = args.out or workspace / "exp/mechanism_v2" / ("metric_cache" if args.limit is None else "metric_cache_smoke")
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for idx in indices:
        groups[str(data["log_name"][idx])].append(str(data["token"][idx]))
    del data
    processor = MetricCacheProcessor(str(out), False)
    paths = []
    started = time.monotonic()
    # Per-split provenance prevents resuming a different protocol into this directory.
    signature = {"protocol_sha256": digest(args.protocol), "split": args.split, "scenes": len(indices), "limit": args.limit}
    signature_path = out / (args.split+"_signature.json")
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise ValueError("Cache protocol/signature mismatch")
    signature_path.write_text(json.dumps(signature, indent=2)+"\n", encoding="utf-8")
    with open(out / (args.split+"_ledger.jsonl"), "a", encoding="utf-8") as ledger:
        for log, tokens in sorted(groups.items()):
            sf = scene_filter_for("navtrain", [log])
            sf.tokens = tokens
            loader = SceneLoader(workspace / "dataset/navsim_logs/trainval", None, sf, SensorConfig.build_no_sensors())
            if set(loader.tokens) != set(tokens):
                raise ValueError("Missing raw scenes in " + log)
            for token in sorted(tokens):
                path = out / log / "unknown" / token / "metric_cache.pkl"
                try:
                    valid_existing = False
                    if path.exists():
                        try:
                            with lzma.open(path, "rb") as stream:
                                existing = pickle.load(stream)
                            valid_existing = existing.ego_state is not None and existing.observation is not None
                            del existing
                        except (EOFError, pickle.UnpicklingError, lzma.LZMAError):
                            valid_existing = False
                    if not valid_existing:
                        scene = loader.get_scene_from_token(token)
                        scenario = NavSimScenario(scene, os.environ["NUPLAN_MAPS_ROOT"], os.environ["NUPLAN_MAP_VERSION"])
                        # Regenerate only a corrupt file in this new cache, never alter historical caches.
                        processor._force_feature_computation = path.exists()
                        entry = processor.compute_metric_cache(scenario)
                        if entry is None:
                            raise RuntimeError("Cache processor returned no entry")
                    paths.append(str(path))
                    ledger.write(json.dumps({"token": token, "log_name": log, "reason": "existing" if valid_existing else "computed",
                                             "valid": True, "path": str(path)})+"\n")
                except Exception as error:
                    ledger.write(json.dumps({"token": token, "valid": False, "exception": repr(error)})+"\n")
                    ledger.flush()
                    raise
            ledger.flush()
            print(args.split, "cached", len(paths), "/", len(indices), "elapsed", round(time.monotonic()-started), "s", flush=True)
    metadata = out / "metadata"
    metadata.mkdir(exist_ok=True)
    # Union completed partitions, so the standard NAVSIM cache loader sees both.
    all_paths = sorted(str(p) for p in out.glob("*/*/*/metric_cache.pkl"))
    temporary = metadata / "metric_cache_metadata.csv.tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name"])
        for path in all_paths:
            writer.writerow([path])
    temporary.replace(metadata / "metric_cache_metadata.csv")
    (out / (args.split+"_completed.json")).write_text(json.dumps(dict(signature, elapsed_seconds=time.monotonic()-started), indent=2)+"\n", encoding="utf-8")
    print("CACHE COMPLETED", args.split, len(paths), flush=True)


if __name__ == "__main__":
    main()
