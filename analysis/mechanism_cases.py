"""Select frozen-rule diagnostic cases and draw current-input/plan comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_mechanism import METRICS, read_score_archive
from mechanism_data import CONFIGS, digest, session_id, split_indices
from runtime import ROOT


def choose_cases(arrays, tokens, logs, policy, split):
    """Hash sampling inside outcome bins, with no best/worst ranking or relaxation."""
    tokens, logs = np.asarray(tokens), np.asarray(logs)
    if len(tokens) != len(logs) or len(set(tokens)) != len(tokens):
        raise ValueError("Unique aligned tokens/logs required")
    seeds, rules = policy["seeds"], policy["case_selection"]
    if rules["maximum_per_original_session_per_contrast"] != 1:
        raise ValueError("Only the frozen one-case-per-session policy is implemented")
    cases, counts = [], {}
    for contrast in policy["contrasts"]:
        left, right = contrast.split("-")
        delta = np.stack([arrays[f"{left}_f1_steps_s{seed}"][:, 0]
                          - arrays[f"{right}_f1_steps_s{seed}"][:, 0] for seed in seeds])
        if delta.shape != (len(seeds), len(tokens)) or not np.isfinite(delta).all():
            raise ValueError("Misaligned/nonfinite scores")
        effect = delta.mean(0)
        masks = {"negative": effect <= rules["negative_max"], "positive": effect >= rules["positive_min"],
                 "neutral": np.abs(effect) <= rules["neutral_absolute_max"]}
        used_sessions = set()
        counts[contrast] = {}
        for kind in rules["bin_order"]:
            candidates = np.flatnonzero(masks[kind])
            def rank(index):
                value = "|".join([rules["salt"], split, contrast, kind, str(tokens[index])])
                return hashlib.sha256(value.encode("utf-8")).hexdigest(), str(tokens[index])
            selected = []
            for index in sorted(candidates, key=rank):
                session = session_id(logs[index])
                if session in used_sessions:
                    continue
                used_sessions.add(session)
                selected.append(int(index))
                cases.append({"contrast": contrast, "bin": kind, "token": str(tokens[index]),
                              "log_name": str(logs[index]), "session": session, "delta_pdms": float(effect[index]),
                              "per_seed_delta": {str(seed): float(delta[j, index]) for j, seed in enumerate(seeds)},
                              "metrics": {config: {str(seed): dict(zip(METRICS, arrays[f"{config}_f1_steps_s{seed}"][index].tolist()))
                                                   for seed in seeds} for config in (left, right)},
                              "hash_rank": rank(index)[0]})
                if len(selected) == rules["per_bin"]:
                    break
            counts[contrast][kind] = {"candidate_scenes": len(candidates),
                                     "candidate_sessions": len({session_id(logs[i]) for i in candidates}),
                                     "selected": len(selected), "requested": rules["per_bin"],
                                     "shortfall": rules["per_bin"]-len(selected)}
    return {"cases": cases, "counts": counts, "disclosure": rules["disclosure"]}


def box_corners(obj):
    x, y, _, _, length, width, cosine, sine = obj[:8]
    local = np.array([[length/2, width/2], [length/2, -width/2], [-length/2, -width/2], [-length/2, width/2]])
    return local @ np.array([[cosine, sine], [-sine, cosine]]) + [x, y]


def plot_case(case, data, index, predictions, donor_index, policy, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    left, right = case["contrast"].split("-")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    centerline = data["cl"][index].reshape(-1, 4)[:, :2]
    expert = data["y_pose"][index].reshape(8, 3)
    points = [centerline, expert[:, :2], np.zeros((1, 2))]
    for axis, config in zip(axes, (left, right)):
        axis.plot(centerline[:, 0], centerline[:, 1], "--", color="0.5", lw=1, label="reference centerline")
        axis.plot(expert[:, 0], expert[:, 1], "ko-", lw=1, ms=3, label="logged future (eval only)")
        axis.scatter([0], [0], marker="^", c="black", s=45, zorder=5)
        for j, obj in enumerate(data["agents"][index]):
            if obj[10] > .5:
                points.append(box_corners(obj))
                axis.add_patch(Polygon(box_corners(obj), facecolor="0.85", edgecolor="0.4", alpha=.8,
                                       label="current privileged boxes" if j == 0 else None))
        if CONFIGS[config][1]:
            for j, obj in enumerate(data["agents"][donor_index]):
                if obj[10] > .5:
                    points.append(box_corners(obj))
                    axis.add_patch(Polygon(box_corners(obj), fill=False, edgecolor="#b95b00", linestyle=":",
                                           label="shuffled donor boxes" if j == 0 else None))
        for seed, color in zip(policy["seeds"], ("#0072b2", "#009e73", "#cc79a7")):
            pose = predictions[f"{config}_f1_steps_s{seed}"][case["token"]]
            points.append(pose[:, :2])
            score = case["metrics"][config][str(seed)]["score"]
            axis.plot(pose[:, 0], pose[:, 1], ".-", color=color, lw=2 if seed == 0 else 1,
                      label=f"seed {seed}, PDMS {score:.3f}")
        axis.set(title=config+" / "+CONFIGS[config][0]+(" / shuffled" if CONFIGS[config][1] else ""),
                 xlabel="forward x (m)", ylabel="left y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, loc="best")
    coordinates = np.concatenate(points)
    low, high = coordinates.min(0)-5, coordinates.max(0)+5
    # Fixed minimum window; all reference/plan points remain visible, even outliers.
    for axis in axes:
        axis.set_xlim(min(-10, low[0]), max(45, high[0]))
        axis.set_ylim(min(-15, low[1]), max(15, high[1]))
    fig.suptitle(f"{case['contrast']} / {case['bin']} / mean paired delta {case['delta_pdms']:+.3f}\n"
                 +case["token"]+" | current inputs and plans, not simulated rollout", fontsize=11)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True, help="Folder with the 18 corresponding .npz files")
    ap.add_argument("--protocol", type=Path, default=ROOT / "results/mechanism_v2/protocol/protocol.json")
    ap.add_argument("--policy", type=Path, default=ROOT / "results/mechanism_v2/protocol/diagnostics.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if policy["protocol_sha256"] != digest(args.protocol) or policy["contrasts"] != protocol["primary_contrasts"]:
        raise ValueError("Diagnostic policy/protocol mismatch")
    complete = json.loads((args.scores / "completed.json").read_text(encoding="utf-8"))
    split = complete["split"]
    if split not in ("validation", "confirmation", "navtest"):
        raise ValueError("Unknown split")
    source = protocol["sources"]["navtest" if split == "navtest" else "navtrain"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Changed frozen features")
    with np.load(source["path"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("token", "log_name", "cl", "y_pose", "agents")}
    indices = split_indices(data, protocol, split)
    tokens, logs = data["token"][indices], data["log_name"][indices]
    arrays, provenance, _ = read_score_archive(args.scores, protocol, digest(args.protocol), tokens, logs)
    report = choose_cases(arrays, tokens, logs, policy, split)
    report.update(split=split, protocol_sha256=digest(args.protocol), policy_sha256=digest(args.policy), provenance=provenance,
                  source_sha256=source["sha256"], score_settings_sha256=digest(args.scores / "settings.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    signature = {key: report[key] for key in ("split", "protocol_sha256", "policy_sha256", "provenance", "score_settings_sha256")}
    signature_path = args.out / "settings.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise ValueError("Case inputs changed; use a new output folder")
    signature_path.write_text(json.dumps(signature, indent=2)+"\n", encoding="utf-8")
    wanted = {case["token"] for case in report["cases"]}
    settings = json.loads((args.scores / "settings.json").read_text(encoding="utf-8"))
    predictions = {}
    for run in settings["runs"]:
        path = args.predictions / (run+".npz")
        if digest(path) != settings["runs"][run]["predictions_sha256"]:
            raise ValueError("Prediction hash mismatch: "+run)
        with np.load(path, allow_pickle=False) as archive:
            predictions[run] = {str(token): pose for token, pose in zip(archive["token"], archive["poses"]) if token in wanted}
        if set(predictions[run]) != wanted:
            raise ValueError("Missing case predictions")
    donor_path = args.protocol.parent / "donors.npz"
    # Every prediction is already bound to a training settings hash by the score
    # archive. Bind the displayed donors to the original full-budget settings too.
    donor_digest = digest(donor_path)
    model_root = Path(protocol["sources"]["navtrain"]["path"]).parents[1] / "mechanism_v2/models"
    for run in settings["runs"]:
        training_settings = json.loads((model_root / run / "settings.json").read_text(encoding="utf-8"))
        if training_settings["donors_sha256"] != donor_digest:
            raise ValueError("Displayed donor provenance mismatch")
    with np.load(donor_path) as archive:
        if not np.array_equal(indices, archive[split+"_indices"]):
            raise ValueError("Donor index mismatch")
        donors = archive[split+"_donors"]
    lookup = {str(token): i for i, token in enumerate(tokens)}
    for number, case in enumerate(report["cases"]):
        position = lookup[case["token"]]
        filename = f"{number+1:02d}_{case['contrast']}_{case['bin']}.png"
        plot_case(case, data, indices[position], predictions, donors[position], policy, args.out / filename)
        case["figure"] = filename
    (args.out / "cases.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    lines = ["# Frozen-rule diagnostic cases", "", report["disclosure"], "", policy["freeze_scope"], ""]
    for contrast, bins in report["counts"].items():
        for kind, row in bins.items():
            lines.append(f"- {contrast}/{kind}: {row['candidate_scenes']} candidates in {row['candidate_sessions']} sessions; "
                         f"selected {row['selected']}/{row['requested']}; shortfall {row['shortfall']}.")
    lines.extend(["", "All seeds are shown, with fixed seed 0 emphasized. Logged future is evaluation-only.",
                  "Current boxes/centerline are privileged inputs, not full road geometry or official simulated rollouts.", ""])
    for case in report["cases"]:
        lines.extend([f"## {case['contrast']} / {case['bin']}", "", case["log_name"]+" / "+case["token"], "",
                      f"![plan diagnostic]({case['figure']})", ""])
    (args.out / "cases.md").write_text("\n".join(lines), encoding="utf-8")
    print("DIAGNOSTIC CASES", len(report["cases"]), "->", args.out, flush=True)


if __name__ == "__main__":
    main()
