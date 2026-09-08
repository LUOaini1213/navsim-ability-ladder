# -*- coding: utf-8 -*-
"""Copy the per-scene CSVs of a NAVSIM workspace run into this repository.

The warmup CSVs (``results/per_scene/``) are small enough to commit verbatim.
navtest is 12,146 scenes x 56 runs, so those are written gzipped and with the
float columns rounded to six decimals -- ``analysis/ladder_io.py`` reads
``.csv.gz`` transparently, and every table under ``results/navtest/`` is
recomputed from these files by ``analysis/analyze_navtest.py``.

    python analysis/export_per_scene.py --split navtest        # exp/*_navtest -> results/navtest/per_scene
    python analysis/export_per_scene.py --split navtest --dry-run

Needs the workspace (``NAVSIM_EXP_ROOT``); the committed output does not.
"""
import argparse
import csv
import glob
import gzip
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_io import REPO  # noqa: E402

COLS = ['token', 'valid', 'no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
        'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance', 'score']


def newest_csv(exp_dir):
    files = sorted(glob.glob(os.path.join(exp_dir, '*', '*.csv')))
    return files[-1] if files else None


def compact(src):
    """(rows, n) of one run CSV, floats rounded to 6 dp, average row dropped."""
    out = []
    with io.open(src, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tok = (r.get('token') or '').strip()
            if not tok or tok == 'average' or r.get('valid') != 'True':
                continue
            out.append([tok, 'True'] + ['%g' % round(float(r[c]), 6) for c in COLS[2:]])
    return out


def write_gz(path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)
    w.writerows(rows)
    with gzip.GzipFile(path, 'wb', compresslevel=9, mtime=0) as g:  # mtime=0: byte-stable output
        g.write(buf.getvalue().encode('utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='navtest')
    ap.add_argument('--exp-root', default=os.environ.get('NAVSIM_EXP_ROOT'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not a.exp_root or not os.path.isdir(a.exp_root):
        raise SystemExit('set NAVSIM_EXP_ROOT or pass --exp-root')
    out = a.out or os.path.join(REPO, 'results', a.split, 'per_scene')
    if not a.dry_run:
        os.makedirs(out, exist_ok=True)

    suffix = '_' + a.split
    total = 0
    for d in sorted(glob.glob(os.path.join(a.exp_root, '*' + suffix))):
        name = os.path.basename(d)
        if name.startswith('metric_cache'):
            continue
        src = newest_csv(d)
        if not src:
            continue
        rows = compact(src)
        dst = os.path.join(out, name[: -len(suffix)] + '.csv.gz')
        print('%-38s %6d scenes  <- %s' % (os.path.basename(dst), len(rows), os.path.relpath(src, a.exp_root)))
        if not a.dry_run:
            write_gz(dst, COLS, rows)
            total += os.path.getsize(dst)

    cache = os.path.join(a.exp_root, 'metric_cache_' + a.split)
    if os.path.isdir(cache):
        pairs = []
        for log in sorted(os.listdir(cache)):
            p = os.path.join(cache, log, 'unknown')
            if os.path.isdir(p):
                pairs += [[tok, log] for tok in sorted(os.listdir(p))]
        print('token_log.csv.gz %d tokens across %d logs' % (len(pairs), len({p[1] for p in pairs})))
        if not a.dry_run:
            dst = os.path.join(out, 'token_log.csv.gz')
            write_gz(dst, ['token', 'log'], sorted(pairs))
            total += os.path.getsize(dst)
    if not a.dry_run:
        print('wrote %s (%.1f MB)' % (out, total / 1024 / 1024))


if __name__ == '__main__':
    sys.exit(main())
