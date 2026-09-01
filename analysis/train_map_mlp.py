# -*- coding: utf-8 -*-
"""Train a small MLP that CAN see the on-route map centerline.

Fills the one gap in the ladder: every map-using agent so far is hand-written
(PrivMap / PrivMapKin). This asks whether a learned model can use the same
centerline features, holding the feature source fixed.

Features (per scene, all in the ego frame of the current frame):
  - ego status 8-dim: velocity(2), acceleration(2), driving_command(4)
  - centerline: K points sampled ahead along the on-route lane centerline,
    as (dx, dy, cos(dtheta), sin(dtheta)) -> 4K dims
Target: the same 8x3 future poses the official EgoStatusMLP regresses.

Train logs / val logs come from available_mini_logs.yaml, so the split matches
the clean ladder. Usage:  python train_map_mlp.py --epochs 60
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'navsim'))

from pathlib import Path
from shapely.geometry import Point

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra

K_POINTS = 20
LOOKAHEAD_M = 60.0
SEARCH_DEPTH = 15
LOGS_YAML = os.path.join(ROOT, 'navsim', 'navsim', 'planning', 'script',
                         'config', 'training', 'available_mini_logs.yaml')
CKPT = os.path.join(ROOT, 'exp', 'map_mlp', 'map_mlp.pt')
NORM = os.path.join(ROOT, 'exp', 'map_mlp', 'norm.json')


def split_logs():
    train, val, sec = [], [], None
    for raw in io.open(LOGS_YAML, encoding='utf-8').read().splitlines():
        line = raw.strip()
        if line.startswith('train_logs:'):
            sec = 't'
            continue
        if line.startswith('val_logs:'):
            sec = 'v'
            continue
        if line.startswith('- ') and sec:
            (train if sec == 't' else val).append(line[2:].strip().strip('"').strip("'"))
    return train, val


def centerline_features(scene):
    """K x 4 centerline features in the ego frame; None if no route."""
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    ids = list(dict.fromkeys(frame.roadblock_ids))
    blocks, lanes = {}, {}
    for rid in ids:
        b = scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK)
        b = b or scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if b is None:
            continue
        blocks[b.id] = b
        for lane in b.interior_edges:
            lanes[lane.id] = lane
    if not lanes:
        return None
    ego = np.asarray(frame.ego_status.ego_pose, dtype=np.float64)
    ego_pt = Point(float(ego[0]), float(ego[1]))
    best, best_d = None, 1e9
    for lane in lanes.values():
        try:
            d = float(lane.baseline_path.linestring.distance(ego_pt))
        except Exception:
            continue
        if d < best_d:
            best_d, best = d, lane
    if best is None:
        return None
    block_ids = list(blocks.keys())
    block_list = list(blocks.values())
    try:
        start_idx = int(np.argmax(np.array(block_ids) == best.get_roadblock_id()))
        window = block_list[start_idx:start_idx + SEARCH_DEPTH] or block_list[:SEARCH_DEPTH]
        route_plan, _ = Dijkstra(best, list(lanes.keys())).search(window[-1])
        discrete = []
        for lane in route_plan:
            discrete.extend(lane.baseline_path.discrete_path)
    except Exception:
        discrete = []
    if len(discrete) < 2:
        discrete = list(best.baseline_path.discrete_path)
    if len(discrete) < 2:
        return None

    path = PDMPath(discrete)
    s0 = float(path.project(ego_pt))
    ss = s0 + np.linspace(0.0, LOOKAHEAD_M, K_POINTS)
    try:
        poses = path.interpolate(ss, as_array=True)
    except Exception:
        return None

    yaw = float(ego[2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    rel = (poses[:, :2] - ego[:2]) @ R.T
    dth = poses[:, 2] - yaw
    return np.concatenate([rel, np.cos(dth)[:, None], np.sin(dth)[:, None]], axis=1).astype(np.float32)


def ego_features(scene):
    st = scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status
    return np.concatenate([np.asarray(st.ego_velocity, dtype=np.float32),
                           np.asarray(st.ego_acceleration, dtype=np.float32),
                           np.asarray(st.driving_command, dtype=np.float32)]).astype(np.float32)


def build(logs, tag):
    data_path = Path(os.environ['OPENSCENE_DATA_ROOT']) / 'navsim_logs' / 'mini'
    sensor_path = Path(os.environ['OPENSCENE_DATA_ROOT']) / 'sensor_blobs' / 'mini'
    filt = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True,
                       log_names=logs, max_scenes=None)
    loader = SceneLoader(data_path, sensor_path, filt, SensorConfig.build_no_sensors())
    toks = loader.tokens
    X, C, Y, keep = [], [], [], []
    t0 = time.time()
    for i, tok in enumerate(toks):
        try:
            scene = loader.get_scene_from_token(tok)
            cl = centerline_features(scene)
            if cl is None:
                continue
            fut = scene.get_future_trajectory(num_trajectory_frames=8)
            y = np.asarray(fut.poses, dtype=np.float32)
            if y.shape != (8, 3):
                continue
            X.append(ego_features(scene))
            C.append(cl.reshape(-1))
            Y.append(y.reshape(-1))
            keep.append(tok)
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            print('  [%s] %d/%d  kept=%d  %.0fs' % (tag, i + 1, len(toks), len(keep), time.time() - t0))
    print('  [%s] done: %d scenes kept of %d (%.0fs)' % (tag, len(keep), len(toks), time.time() - t0))
    if not keep:
        return None
    return (np.stack(X), np.stack(C), np.stack(Y), keep)


class MapMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=512, out_dim=24):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=64)
    args = ap.parse_args()

    tr_logs, va_logs = split_logs()
    print('train logs %d / val logs %d' % (len(tr_logs), len(va_logs)))
    print('building train set...')
    tr = build(tr_logs, 'train')
    print('building val set...')
    va = build(va_logs, 'val')
    if tr is None or va is None:
        raise SystemExit('empty dataset')

    Xtr = np.concatenate([tr[0], tr[1]], axis=1)
    Xva = np.concatenate([va[0], va[1]], axis=1)
    Ytr, Yva = tr[2], va[2]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev, ' in_dim:', Xtr.shape[1], ' n_train:', len(Xtr), ' n_val:', len(Xva))
    model = MapMLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(Ytr, device=dev)
    xv = torch.tensor(Xva, device=dev)
    yv = torch.tensor(Yva, device=dev)

    best = 1e9
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        tot = 0.0
        for i in range(0, len(xt), args.bs):
            idx = perm[i:i + args.bs]
            opt.zero_grad()
            loss = torch.nn.functional.l1_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vl = float(torch.nn.functional.l1_loss(model(xv), yv))
        if vl < best:
            best = vl
            torch.save({'state_dict': model.state_dict(), 'in_dim': int(Xtr.shape[1])}, CKPT)
        if (ep + 1) % 10 == 0 or ep == 0:
            print('  epoch %3d  train L1 %.4f  val L1 %.4f%s'
                  % (ep + 1, tot / len(xt), vl, '  *' if vl == best else ''))

    io.open(NORM, 'w', encoding='utf-8').write(json.dumps(
        {'mu': mu.tolist(), 'sd': sd.tolist(), 'k_points': K_POINTS,
         'lookahead_m': LOOKAHEAD_M, 'search_depth': SEARCH_DEPTH}))
    print('best val L1 %.4f' % best)
    print('ckpt', CKPT)
    print('norm', NORM)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
