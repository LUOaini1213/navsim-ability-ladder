"""Extract per-scene features for the ability-ladder experiments, one npz per log.

Decouples the slow part (scene loading, map route search) from training, so
every seed / factor combination trains from the same frozen feature file.

Per scene we store
  ego        (8,)     vx, vy, ax, ay, driving command one-hot(4)      [kinematics]
  cl         (K*4,)   on-route centerline ahead, ego frame            [GT map]
  agents     (N, 11)  nearest dynamic boxes: x, y, vx, vy, l, w, cos h, sin h,
                      is_vehicle, is_pedestrian, mask                 [GT agents]
  lead       (2,)     same-lane lead bumper gap (m, 200 if none), lead forward speed
  y_pose     (24,)    future 8 poses (x, y, heading) in ego frame     [target A]
  y_ds       (8,)     arc-length progress along the centerline        [target B]
  ds_valid   ()       1 if y_ds is a clean monotone projection
  meta: token, log_name, map_name, cmd index, ego speed

Usage
  python scripts/build_features.py --split navtest  --workers 6
  python scripts/build_features.py --split navtrain --workers 6 [--max-logs 50]
  python scripts/build_features.py --split navtest --merge      # concat per-log npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("NAVSIM_DEVKIT_ROOT", ""))
from navsim.agents.ladder_features import (K_POINTS, LOOKAHEAD_M, N_AGENTS, SEARCH_DEPTH,  # noqa: E402
                                           agent_features, lead_features)

SPLIT_TO_DATA = {"navtest": "test", "navtrain": "trainval", "warmup_test_e2e": "mini", "navmini": "mini"}


def scene_filter_for(split: str, log_names):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = str(Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/common/train_test_split/scene_filter")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        sf = instantiate(compose(config_name=split))
    if log_names is not None:
        sf.log_names = list(log_names)
    return sf


def process_log(args):
    split, log_name, out_dir = args
    out_path = Path(out_dir) / f"{log_name}.npz"
    if out_path.exists():
        return log_name, -1, 0.0
    t0 = time.time()
    try:
        from shapely.geometry import Point

        from navsim.agents.centerline_util import centerline_features, ego_features_from_status, route_path, to_global
        from navsim.common.dataclasses import SensorConfig
        from navsim.common.dataloader import SceneLoader

        sf = scene_filter_for(split, [log_name])
        loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / SPLIT_TO_DATA[split],
            scene_filter=sf,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        rec = {k: [] for k in ["ego", "cl", "agents", "lead", "y_pose", "y_ds", "ds_valid", "token", "map_name", "cmd", "speed"]}
        for tok in loader.tokens:
            try:
                scene = loader.get_scene_from_token(tok)
                frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
                ego = ego_features_from_status(frame.ego_status)
                rp = route_path(scene, SEARCH_DEPTH)
                cl = centerline_features(scene, K_POINTS, LOOKAHEAD_M, SEARCH_DEPTH)
                if rp is None or cl is None:
                    continue
                path, ego_pose, s0 = rp
                fut = np.asarray(scene.get_future_trajectory(num_trajectory_frames=8).poses, dtype=np.float64)
                if fut.shape != (8, 3):
                    continue
                g = to_global(fut, ego_pose)
                ds = np.array([float(path.project(Point(float(p[0]), float(p[1])))) - s0 for p in g], dtype=np.float32)
                ds_valid = 1.0 if (ds.min() >= -2.0 and not np.any(np.diff(ds) < -1.0)) else 0.0
                ds = np.maximum.accumulate(np.clip(ds, 0.0, None))
                speed = float(np.hypot(ego[0], ego[1]))
                rec["ego"].append(ego.astype(np.float32))
                rec["cl"].append(cl.reshape(-1).astype(np.float32))
                rec["agents"].append(agent_features(frame))
                rec["lead"].append(lead_features(scene, speed))
                rec["y_pose"].append(fut.reshape(-1).astype(np.float32))
                rec["y_ds"].append(ds)
                rec["ds_valid"].append(ds_valid)
                rec["token"].append(tok)
                rec["map_name"].append(scene.scene_metadata.map_name)
                rec["cmd"].append(int(np.argmax(frame.ego_status.driving_command)))
                rec["speed"].append(speed)
            except Exception:
                continue
        n = len(rec["token"])
        np.savez_compressed(
            out_path,
            ego=np.stack(rec["ego"]) if n else np.zeros((0, 8), np.float32),
            cl=np.stack(rec["cl"]) if n else np.zeros((0, K_POINTS * 4), np.float32),
            agents=np.stack(rec["agents"]) if n else np.zeros((0, N_AGENTS, 11), np.float32),
            lead=np.stack(rec["lead"]) if n else np.zeros((0, 2), np.float32),
            y_pose=np.stack(rec["y_pose"]) if n else np.zeros((0, 24), np.float32),
            y_ds=np.stack(rec["y_ds"]) if n else np.zeros((0, 8), np.float32),
            ds_valid=np.asarray(rec["ds_valid"], np.float32),
            token=np.asarray(rec["token"]),
            log_name=np.asarray([log_name] * n),
            map_name=np.asarray(rec["map_name"]),
            cmd=np.asarray(rec["cmd"], np.int64),
            speed=np.asarray(rec["speed"], np.float32),
        )
        return log_name, n, time.time() - t0
    except Exception:
        traceback.print_exc()
        return log_name, -2, time.time() - t0


def merge(split: str, out_dir: Path):
    files = sorted(out_dir.glob("*.npz"))
    parts = [np.load(f, allow_pickle=False) for f in files]
    keys = ["ego", "cl", "agents", "lead", "y_pose", "y_ds", "ds_valid", "token", "log_name", "map_name", "cmd", "speed"]
    merged = {k: np.concatenate([p[k] for p in parts if len(p["token"])]) for k in keys}
    out = out_dir.parent / f"features_{split}.npz"
    np.savez_compressed(out, **merged)
    print(f"merged {len(files)} logs -> {out}  scenes={len(merged['token'])}  ds_valid={int(merged['ds_valid'].sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-logs", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out or (Path(os.environ["NAVSIM_EXP_ROOT"]) / "features" / args.split))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args.split, out_dir)
        return

    sf = scene_filter_for(args.split, None)
    data_dir = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / SPLIT_TO_DATA[args.split]
    present = {p.stem for p in data_dir.glob("*.pkl")}
    logs = [l for l in sf.log_names if l in present]
    missing = len(sf.log_names) - len(logs)
    if args.max_logs:
        logs = logs[: args.max_logs]
    print(f"[{args.split}] {len(logs)} logs to process ({missing} listed logs missing on disk), workers={args.workers}", flush=True)

    import multiprocessing as mp

    t0 = time.time()
    done = 0
    total = 0
    with mp.Pool(processes=args.workers) as pool:
        for log_name, n, dt in pool.imap_unordered(process_log, [(args.split, l, str(out_dir)) for l in logs]):
            done += 1
            if n >= 0:
                total += n
            status = "cached" if n == -1 else ("FAILED" if n == -2 else f"{n} scenes")
            print(f"  [{done}/{len(logs)}] {log_name}: {status} ({dt:.0f}s)  total={total}  elapsed={time.time()-t0:.0f}s", flush=True)
    merge(args.split, out_dir)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
