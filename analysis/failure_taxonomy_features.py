# -*- coding: utf-8 -*-
"""Extract the pre-registered scene features for every navtest scene.

    $env:NAVSIM_WORKSPACE = 'E:\\navsim_workspace'; . .\\scripts\\env.ps1
    & $env:NAVSIM_PYTHON -X utf8 analysis\\failure_taxonomy_features.py --out results\\failure_taxonomy

Inputs (read only)
  results/navtest/per_scene/token_log.csv.gz      the 12,146 navtest tokens and their logs
  <workspace>/dataset/navsim_logs/test/<log>.pkl  raw OpenScene logs (no sensor blobs)
  <workspace>/exp/features/features_navtest.npz   the study's frozen inputs (cl, lead, y_pose)
  <workspace>/dataset/maps/<city>/<ver>/map.gpkg  layer `intersections` only
Outputs (refuses to overwrite)
  <out>/features.csv.gz        one row per scene, raw values and categories
  <out>/features_manifest.json inputs, counts, consistency checks against the archive

Runs as one process; numerical libraries are pinned to --threads (default 1, at most 2
while other heavy jobs share the machine). Before each raw log is unpickled it waits for
free memory (--min-free-mb plus five times the file size), so it never pushes a shared
16-GB machine into its commit limit. Category rules live in failure_taxonomy_core.py.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import failure_taxonomy_core as core  # noqa: E402


def write_lf(path, text):
    """Write text with LF line endings on every platform (the repository normalises to LF)."""
    with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


TOKEN_LOG = REPO / 'results' / 'navtest' / 'per_scene' / 'token_log.csv.gz'
COLUMNS = ['token', 'log', 'session', 'city', 'speed_mps', 'speed', 'command', 'lead_gap_m',
           'lead_class', 'lead', 'ego_in_junction', 'junction_dist_m', 'junction', 'curvature_deg',
           'curvature', 'lane_change_m', 'lane_change', 'v_future_min', 'v_future_max', 'stop_go',
           'logged_heading_change_deg', 'logged_length_m', 'route_len_m']


def available_mb():
    """(available physical MB, available commit MB) or (None, None) off Windows."""
    if os.name != 'nt':
        return None, None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    st = Status()
    st.dwLength = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 2**20, st.ullAvailPageFile / 2**20


def wait_for_memory(need_mb, label, log):
    waited = 0
    while True:
        phys, commit = available_mb()
        if phys is None or (commit >= need_mb and phys >= need_mb / 2):
            if waited:
                log['memory_waits'].append({'before': label, 'seconds': waited})
            return
        if waited % 300 == 0:
            print('  waiting for memory before %s: commit %.0f MB, phys %.0f MB, need %.0f' % (
                label, commit, phys, need_mb), flush=True)
        time.sleep(15)
        waited += 15


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def yaw_of(q):
    """Yaw of a (w, x, y, z) quaternion exactly as NAVSIM gets it: pyquaternion's
    yaw_pitch_roll[0] (normalise, then atan2(2(wz - xy), 1 - 2(y^2 + z^2)))."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return math.atan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y * y + z * z))


def to_local(x, y, yaw, ox, oy, oyaw):
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    dx, dy = x - ox, y - oy
    return c * dx - s * dy, s * dx + c * dy, math.atan2(math.sin(yaw - oyaw), math.cos(yaw - oyaw))


def map_file(maps_root, city_location):
    folder = maps_root / city_location
    files = sorted(folder.glob('*/map.gpkg'))
    if len(files) != 1:
        raise SystemExit('expected one map.gpkg under %s, found %d' % (folder, len(files)))
    return files[0]


def load_intersections(gpkg):
    """Union of the city's intersection polygons in its projected (UTM) system, prepared,
    reading only that layer, the way nuplan's GPKGMapsDB.load_vector_layer projects it."""
    import pyogrio
    import shapely
    meta = pyogrio.read_dataframe(str(gpkg), layer='meta')
    crs = meta.loc[meta['key'] == 'projectedCoordSystem', 'value'].iloc[0]
    df = pyogrio.read_dataframe(str(gpkg), layer='intersections').to_crs(crs)
    union = shapely.union_all(df.geometry.values)
    shapely.prepare(union)
    return union, len(df), crs


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ''
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        return ('%.' + str(nd) + 'f') % v
    return str(v)


def city_mix(args):
    """Scenes and sessions per city in the frozen navtrain and navtest archives (context for the
    city breakdown; reads only the map_name and log_name arrays)."""
    import numpy as np
    target = Path(args.out) / 'train_test_city_mix.json'
    if target.exists():
        raise SystemExit('refusing to overwrite %s' % target)
    out = {}
    for split in ('navtrain', 'navtest'):
        path = Path(args.workspace) / 'exp' / 'features' / ('features_%s.npz' % split)
        z = np.load(str(path), allow_pickle=False)
        maps, logs = [str(m) for m in z['map_name']], [str(l) for l in z['log_name']]
        per = {}
        for m, l in zip(maps, logs):
            rec = per.setdefault(core.city_of(m), {'scenes': 0, 'sessions': set(), 'logs': set()})
            rec['scenes'] += 1
            rec['sessions'].add(core.session_of(l))
            rec['logs'].add(l)
        out[split] = {'scenes': len(maps), 'archive': str(path), 'sha256': sha256(path),
                      'cities': {c: {'scenes': v['scenes'], 'share': round(v['scenes'] / len(maps), 4),
                                     'logs': len(v['logs']), 'sessions': len(v['sessions'])}
                                 for c, v in sorted(per.items())}}
    write_lf(target, json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workspace', default=os.environ.get('NAVSIM_WORKSPACE', 'E:/navsim_workspace'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--min-free-mb', type=float, default=1200.0)
    ap.add_argument('--max-logs', type=int, default=None, help='smoke test on the first N logs')
    ap.add_argument('--city-mix', action='store_true', help='only write train_test_city_mix.json')
    args = ap.parse_args()
    if args.city_mix:
        os.environ['OMP_NUM_THREADS'] = str(args.threads)
        return city_mix(args)
    if args.threads < 1 or args.threads > 4:
        raise SystemExit('--threads must be 1..4')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = str(args.threads)
    import numpy as np
    import shapely

    out = Path(args.out)
    target = out / 'features.csv.gz'
    if target.exists() or (out / 'features_manifest.json').exists():
        raise SystemExit('refusing to overwrite %s; choose a fresh --out' % out)
    out.mkdir(parents=True, exist_ok=True)

    ws = Path(args.workspace)
    logs_dir = ws / 'dataset' / 'navsim_logs' / 'test'
    maps_root = ws / 'dataset' / 'maps'
    archive_path = ws / 'exp' / 'features' / 'features_navtest.npz'
    t_start = time.time()
    record = {'memory_waits': []}

    with io.TextIOWrapper(gzip.open(TOKEN_LOG, 'rb'), encoding='utf-8') as f:
        token_log = {r['token']: r['log'] for r in csv.DictReader(f)}
    by_log = {}
    for tok, log in token_log.items():
        by_log.setdefault(log, []).append(tok)
    logs = sorted(by_log)
    if args.max_logs:
        logs = logs[:args.max_logs]

    archive = np.load(str(archive_path), allow_pickle=False)
    a_tok = [str(t) for t in archive['token']]
    a_idx = {t: i for i, t in enumerate(a_tok)}
    a_cl, a_lead, a_pose = archive['cl'], archive['lead'], archive['y_pose']
    a_speed, a_cmd = archive['speed'], archive['cmd']

    unions, union_info = {}, {}
    rows, checks = [], {'speed_abs_diff_max': 0.0, 'command_mismatch': 0, 'lead_gap_abs_diff_max': 0.0,
                        'lead_presence_mismatch': 0, 'logged_xy_abs_diff_max': 0.0,
                        'logged_heading_abs_diff_max': 0.0, 'missing_in_archive': 0,
                        'window_incomplete': 0, 'all_zero_command': 0}
    sanity = {}  # command -> sign of logged heading change (no outcome involved)
    for n_log, log in enumerate(logs, 1):
        pkl = logs_dir / (log + '.pkl')
        size_mb = pkl.stat().st_size / 2**20
        wait_for_memory(args.min_free_mb + 5 * size_mb, log, record)
        with open(pkl, 'rb') as f:
            frames = pickle.load(f)
        index = {fr['token']: i for i, fr in enumerate(frames)}
        for tok in sorted(by_log[log]):
            i = index.get(tok)
            if i is None or i - 3 < 0 or i + 10 >= len(frames):
                checks['window_incomplete'] += 1
                continue
            fr = frames[i]
            vx, vy = float(fr['ego_dynamic_state'][0]), float(fr['ego_dynamic_state'][1])
            v0 = math.hypot(vx, vy)
            cmd_vals = [float(v) for v in fr['driving_command']]
            if max(cmd_vals) <= 0:
                checks['all_zero_command'] += 1
            command = core.command_of(cmd_vals)
            anns = fr['anns']
            gap, _, lead_cls = core.lead_gap(anns['gt_boxes'], anns['gt_names'], anns['gt_velocity_3d'])
            ox, oy = float(fr['ego2global_translation'][0]), float(fr['ego2global_translation'][1])
            oyaw = yaw_of(fr['ego2global_rotation'])
            future, poses = [], [(0.0, 0.0, 0.0)]
            for k in range(1, 9):
                fk = frames[i + k]
                future.append(math.hypot(float(fk['ego_dynamic_state'][0]), float(fk['ego_dynamic_state'][1])))
                poses.append(to_local(float(fk['ego2global_translation'][0]), float(fk['ego2global_translation'][1]),
                                      yaw_of(fk['ego2global_rotation']), ox, oy, oyaw))
            p8 = poses[-1]
            heading_change = math.degrees(p8[2])
            length = core.polyline_length([(p[0], p[1]) for p in poses])

            city_location = str(fr['map_location'])
            if city_location not in unions:
                unions[city_location], n_poly, crs = load_intersections(map_file(maps_root, city_location))
                union_info[city_location] = {'intersections': n_poly, 'projected_crs': crs}
            union = unions[city_location]
            ego_in = bool(shapely.contains_xy(union, ox, oy))

            j = a_idx.get(tok)
            cl_rows, first_s, curv, dlat, route_len = None, None, None, None, None
            if j is None:
                checks['missing_in_archive'] += 1
            else:
                cl_rows = [tuple(float(v) for v in r) for r in a_cl[j].reshape(-1, 4)]
                s = core.cl_arc_lengths(len(cl_rows))
                c, sn = math.cos(oyaw), math.sin(oyaw)
                gx = [ox + c * r[0] - sn * r[1] for r in cl_rows]
                gy = [oy + sn * r[0] + c * r[1] for r in cl_rows]
                inside = shapely.contains_xy(union, np.asarray(gx), np.asarray(gy))
                hits = [s[q] for q in range(len(s)) if inside[q]]
                first_s = 0.0 if ego_in else (hits[0] if hits else None)
                curv = core.curvature_deg(cl_rows)
                dlat = core.lane_change_delta(cl_rows, (p8[0], p8[1]))
                route_len = core.route_length(cl_rows)
                checks['speed_abs_diff_max'] = max(checks['speed_abs_diff_max'], abs(float(a_speed[j]) - v0))
                checks['command_mismatch'] += int(int(a_cmd[j]) != core.COMMANDS.index(command))
                a_gap = float(a_lead[j][0])
                a_has = a_gap < 199.0
                if a_has != (gap is not None):
                    checks['lead_presence_mismatch'] += 1
                elif gap is not None:
                    checks['lead_gap_abs_diff_max'] = max(checks['lead_gap_abs_diff_max'], abs(a_gap - gap))
                ap8 = a_pose[j].reshape(8, 3)[-1]
                checks['logged_xy_abs_diff_max'] = max(checks['logged_xy_abs_diff_max'],
                                                       abs(float(ap8[0]) - p8[0]), abs(float(ap8[1]) - p8[1]))
                dh = math.atan2(math.sin(float(ap8[2]) - p8[2]), math.cos(float(ap8[2]) - p8[2]))
                checks['logged_heading_abs_diff_max'] = max(checks['logged_heading_abs_diff_max'], abs(dh))

            sign = 'left(+)' if heading_change > 5 else ('right(-)' if heading_change < -5 else 'flat')
            sanity.setdefault(command, {}).setdefault(sign, 0)
            sanity[command][sign] += 1
            rows.append({
                'token': tok, 'log': log, 'session': core.session_of(log),
                'city': core.city_of(city_location), 'speed_mps': v0, 'speed': core.speed_band(v0),
                'command': command, 'lead_gap_m': gap, 'lead_class': lead_cls or '',
                'lead': core.lead_band(gap), 'ego_in_junction': ego_in, 'junction_dist_m': first_s,
                'junction': core.junction_band(ego_in if cl_rows is not None else None, first_s),
                'curvature_deg': curv, 'curvature': core.curvature_band(curv),
                'lane_change_m': dlat, 'lane_change': core.lane_change_band(dlat),
                'v_future_min': min(future), 'v_future_max': max(future),
                'stop_go': core.stop_go(v0, future), 'logged_heading_change_deg': heading_change,
                'logged_length_m': length, 'route_len_m': route_len,
            })
        del frames, index
        gc.collect()
        print('[%d/%d] %s: %d scenes (%.0f s)' % (n_log, len(logs), log, len(by_log[log]), time.time() - t_start),
              flush=True)

    rows.sort(key=lambda r: r['token'])
    with gzip.open(target, 'wt', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([fmt(r[c]) for c in COLUMNS])

    counts = {}
    for name in core.FEATURE_NAMES:
        counts[name] = {}
        for r in rows:
            counts[name][r[name]] = counts[name].get(r[name], 0) + 1
    phys, commit = available_mb()
    manifest = {
        'script': 'analysis/failure_taxonomy_features.py',
        'preregistration': 'docs/FAILURE_TAXONOMY.md',
        'scenes': len(rows), 'logs': len({r['log'] for r in rows}),
        'sessions': len({r['session'] for r in rows}),
        'expected_scenes': len(token_log) if not args.max_logs else None,
        'inputs': {
            'token_log': {'path': str(TOKEN_LOG.relative_to(REPO)).replace('\\', '/'), 'sha256': sha256(TOKEN_LOG)},
            'feature_archive': {'path': str(archive_path), 'sha256': sha256(archive_path),
                                'scenes': len(a_tok)},
            'raw_logs': {'path': str(logs_dir), 'files_read': len(logs)},
            'maps': {k: dict(v, gpkg=str(map_file(maps_root, k))) for k, v in sorted(union_info.items())},
        },
        'consistency_with_archive': checks,
        'command_vs_logged_heading_sign': sanity,
        'category_counts': counts,
        'threads': args.threads,
        'memory_waits': record['memory_waits'],
        'runtime_s': round(time.time() - t_start, 1),
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'shapely': shapely.__version__},
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'available_mb_at_end': {'phys': phys, 'commit': commit},
    }
    write_lf((out / 'features_manifest.json'), json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'scenes': len(rows), 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
