# -*- coding: utf-8 -*-
"""Learn ONLY the speed profile; take the geometry from the map centerline.

Motivation (from the ladder): on the map centerline, swapping the hand-written
kinematic speed for the human arc-length profile is the single biggest
remaining gain (0.802 -> 0.866 on 563; +0.064 on the clean split). So instead of
asking a small MLP to regress full xyθ poses — which it failed to do — ask it
only for progress along the path: 8 arc-length offsets Δs_k. The path itself
stays hand-written. This is the "rules draw the line, the model paces it" split.

Target: Δs_k = project(GT future pose k onto path) - s0, k = 1..8.
Inference: s_k = s0 + cummax(relu(Δs_k)); poses = path.interpolate(s_k) -> ego frame.

Usage:  python train_speed_mlp.py --epochs 80 --hidden 128 --dropout 0.2 --wd 1e-3
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'navsim'))
sys.path.insert(0, ROOT)

from shapely.geometry import Point  # noqa: E402

from navsim.agents.centerline_util import centerline_features, ego_features, route_path, to_global  # noqa: E402
import train_map_mlp as base  # noqa: E402


def build(logs, tag):
    loader = base.make_loader(logs)
    toks = loader.tokens
    X, C, Y, keep = [], [], [], []
    t0 = time.time()
    for i, tok in enumerate(toks):
        try:
            scene = loader.get_scene_from_token(tok)
            rp = route_path(scene)
            cl = centerline_features(scene)
            if rp is None or cl is None:
                continue
            path, ego, s0 = rp
            fut = np.asarray(scene.get_future_trajectory(num_trajectory_frames=8).poses, dtype=np.float64)
            if fut.shape != (8, 3):
                continue
            g = to_global(fut, ego)
            ds = np.array([float(path.project(Point(float(p[0]), float(p[1])))) - s0 for p in g],
                          dtype=np.float32)
            # progress must be non-negative and non-decreasing; if the projection is
            # badly off the route (lane change away from it), skip the sample.
            if ds.min() < -2.0 or np.any(np.diff(ds) < -1.0):
                continue
            ds = np.maximum.accumulate(np.clip(ds, 0.0, None))
            X.append(ego_features(scene))
            C.append(cl.reshape(-1))
            Y.append(ds)
            keep.append(tok)
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print('  [%s] %d/%d  kept=%d  %.0fs' % (tag, i + 1, len(toks), len(keep), time.time() - t0))
    print('  [%s] done: %d scenes kept of %d (%.0fs)' % (tag, len(keep), len(toks), time.time() - t0))
    return (np.stack(X), np.stack(C), np.stack(Y), keep) if keep else None


def main():
    args = base.parse()
    tr_logs, va_logs = base.split_logs()
    print('train logs %d / val logs %d' % (len(tr_logs), len(va_logs)))
    tr = build(tr_logs, 'train')
    va = build(va_logs, 'val')
    if tr is None or va is None:
        raise SystemExit('empty dataset')
    Xtr = np.concatenate([tr[0], tr[1]], axis=1)
    Xva = np.concatenate([va[0], va[1]], axis=1)
    out_dir = os.path.join(ROOT, 'exp', 'speed_mlp' + ('_' + args.tag if args.tag else ''))
    print('hidden=%d dropout=%.2f wd=%g  n_train=%d n_val=%d in_dim=%d  (target = 8 arc-length offsets)'
          % (args.hidden, args.dropout, args.wd, len(Xtr), len(Xva), Xtr.shape[1]))
    base.fit(Xtr, tr[2], Xva, va[2], args, out_dir, 8, 'speed_mlp')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
