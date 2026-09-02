# -*- coding: utf-8 -*-
"""Train a small MLP that CAN see the on-route map centerline.

Fills the one gap in the ladder: every map-using agent so far is hand-written
(PrivMap / PrivMapKin). This asks whether a learned model can use the same
centerline features, holding the feature source fixed.

Features (ego frame): ego status 8-dim + K centerline points x (dx, dy, cos, sin).
Target: the same 8x3 future poses the official EgoStatusMLP regresses.

Usage:
  python train_map_mlp.py --epochs 80                       # baseline (hidden 512)
  python train_map_mlp.py --epochs 80 --hidden 128 --dropout 0.2 --wd 1e-3 --tag reg
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

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.agents.centerline_util import (K_POINTS, LOOKAHEAD_M, SEARCH_DEPTH,
                                           centerline_features, ego_features)

LOGS_YAML = os.path.join(ROOT, 'navsim', 'navsim', 'planning', 'script',
                         'config', 'training', 'available_mini_logs.yaml')


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


def make_loader(logs):
    data_path = Path(os.environ['OPENSCENE_DATA_ROOT']) / 'navsim_logs' / 'mini'
    sensor_path = Path(os.environ['OPENSCENE_DATA_ROOT']) / 'sensor_blobs' / 'mini'
    filt = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True,
                       log_names=logs, max_scenes=None)
    return SceneLoader(data_path, sensor_path, filt, SensorConfig.build_no_sensors())


def build(logs, tag):
    loader = make_loader(logs)
    toks = loader.tokens
    X, C, Y, keep = [], [], [], []
    t0 = time.time()
    for i, tok in enumerate(toks):
        try:
            scene = loader.get_scene_from_token(tok)
            cl = centerline_features(scene)
            if cl is None:
                continue
            y = np.asarray(scene.get_future_trajectory(num_trajectory_frames=8).poses, dtype=np.float32)
            if y.shape != (8, 3):
                continue
            X.append(ego_features(scene))
            C.append(cl.reshape(-1))
            Y.append(y.reshape(-1))
            keep.append(tok)
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print('  [%s] %d/%d  kept=%d  %.0fs' % (tag, i + 1, len(toks), len(keep), time.time() - t0))
    print('  [%s] done: %d scenes kept of %d (%.0fs)' % (tag, len(keep), len(toks), time.time() - t0))
    return (np.stack(X), np.stack(C), np.stack(Y), keep) if keep else None


class MapMLP(torch.nn.Module):
    """Dropout layers are always present (p may be 0) so state_dict keys never shift."""

    def __init__(self, in_dim, hidden=512, out_dim=24, dropout=0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def fit(Xtr, Ytr, Xva, Yva, args, out_dir, out_dim, name):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MapMLP(Xtr.shape[1], args.hidden, out_dim, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    xv, yv = torch.tensor(Xva, device=dev), torch.tensor(Yva, device=dev)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, name + '.pt')
    best, best_ep = 1e9, -1
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
            best, best_ep = vl, ep + 1
            torch.save({'state_dict': model.state_dict(), 'in_dim': int(Xtr.shape[1]),
                        'hidden': args.hidden, 'dropout': args.dropout, 'out_dim': out_dim}, ckpt)
        if (ep + 1) % 10 == 0 or ep == 0:
            print('  epoch %3d  train L1 %.4f  val L1 %.4f%s'
                  % (ep + 1, tot / len(xt), vl, '  *' if vl == best else ''))
    io.open(os.path.join(out_dir, 'norm.json'), 'w', encoding='utf-8').write(json.dumps(
        {'mu': mu.tolist(), 'sd': sd.tolist(), 'k_points': K_POINTS,
         'lookahead_m': LOOKAHEAD_M, 'search_depth': SEARCH_DEPTH,
         'hidden': args.hidden, 'dropout': args.dropout, 'wd': args.wd,
         'best_val_l1': best, 'best_epoch': best_ep, 'n_train': int(len(Xtr)), 'n_val': int(len(Xva))}))
    print('best val L1 %.4f @ epoch %d  ->  %s' % (best, best_ep, ckpt))


def parse(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--tag', type=str, default='')
    if extra:
        extra(ap)
    return ap.parse_args()


def main():
    args = parse()
    tr_logs, va_logs = split_logs()
    print('train logs %d / val logs %d' % (len(tr_logs), len(va_logs)))
    tr = build(tr_logs, 'train')
    va = build(va_logs, 'val')
    if tr is None or va is None:
        raise SystemExit('empty dataset')
    Xtr = np.concatenate([tr[0], tr[1]], axis=1)
    Xva = np.concatenate([va[0], va[1]], axis=1)
    out_dir = os.path.join(ROOT, 'exp', 'map_mlp' + ('_' + args.tag if args.tag else ''))
    print('hidden=%d dropout=%.2f wd=%g  n_train=%d n_val=%d in_dim=%d'
          % (args.hidden, args.dropout, args.wd, len(Xtr), len(Xva), Xtr.shape[1]))
    fit(Xtr, tr[2], Xva, va[2], args, out_dir, 24, 'map_mlp')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
