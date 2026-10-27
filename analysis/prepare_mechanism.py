"""Freeze all partitions/controls before reading new experiment outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from mechanism_data import freeze_protocol, load_features, matched_donors, split_indices
from runtime import ROOT, configure


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace")
    ap.add_argument("--out", type=Path, default=ROOT / "results/mechanism_v2/protocol")
    args = ap.parse_args()
    workspace = configure(args.workspace)
    train_path = workspace / "exp/features/features_navtrain.npz"
    test_path = workspace / "exp/features/features_navtest.npz"
    train, test = load_features(train_path), load_features(test_path)
    protocol = freeze_protocol(train, test, train_path, test_path)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "protocol.json"
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Refusing to replace a different frozen protocol")
    else:
        dest.write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    arrays, report = {}, {}
    for number, split in enumerate(("train", "validation", "confirmation", "navtest")):
        data = test if split == "navtest" else train
        idx = split_indices(data, protocol, split)
        donor, levels = matched_donors(data, idx, protocol["seed"]+number)
        arrays[split+"_indices"], arrays[split+"_donors"] = idx, donor
        report[split] = {"n": len(idx), "matching_levels": levels}
    control_path = args.out / "donors.npz"
    if control_path.exists():
        with np.load(control_path) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise RuntimeError("Refusing to replace different frozen donors")
    else:
        np.savez_compressed(control_path, **arrays)
    (args.out / "donor_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"splits": protocol["split_counts"], "fractions": protocol["fractions"],
                      "donors": report}, indent=2))


if __name__ == "__main__":
    main()
