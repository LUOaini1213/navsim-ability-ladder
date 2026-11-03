# -*- coding: utf-8 -*-
"""Pure-Python core of the navtest failure taxonomy (docs/FAILURE_TAXONOMY.md).

Standard library only, so the tests run in CI with no dataset and no devkit.
Everything that decides a category, a failure, a bootstrap interval or a
root-cause label lives here; the extraction, analysis and case scripts only
feed it data.

Sections
  scene features   fixed bins of the pre-registration
  outcomes         per-scene failure indicators
  bootstrap        the log/session cluster bootstrap of analyze_clustered.py,
                   plus a replay of its random stream for many statistics
  root causes      the labelling rules and the stratified case sample
"""
from __future__ import annotations

import hashlib
import math
import random

B = 10000
SEED = 12345

# --------------------------------------------------------------------------- features

CITY = {'us-nv-las-vegas-strip': 'vegas', 'us-pa-pittsburgh-hazelwood': 'pittsburgh',
        'us-ma-boston': 'boston', 'sg-one-north': 'singapore'}
COMMANDS = ('left', 'straight', 'right', 'unknown')
DYNAMIC = ('vehicle', 'pedestrian', 'bicycle', 'generic_object')

# feature -> categories in display order ('n/a' is appended when present)
FEATURES = [
    ('city', ('vegas', 'pittsburgh', 'boston', 'singapore')),
    ('speed', ('stopped', 'slow', 'medium', 'fast')),
    ('command', COMMANDS),
    ('lead', ('none', 'close', 'mid', 'far')),
    ('junction', ('in', 'near', 'none')),
    ('curvature', ('straight', 'bend', 'turn')),
    ('lane_change', ('no', 'yes')),
    ('stop_go', ('stationary', 'starting', 'stopping', 'moving')),
]
FEATURE_NAMES = [f for f, _ in FEATURES]
NA = 'n/a'

# lead rule of PrivilegedBrakeAgent._lead_gap_speed (agents/privileged_brake_agent.py)
LANE_HALF_WIDTH = 2.3
EGO_FRONT = 4.0
LEAD_X_MIN, LEAD_X_MAX = 1.5, 55.0

CL_POINTS, CL_LOOKAHEAD = 20, 60.0
JUNCTION_NEAR_M = 30.0
CURVATURE_AT_M = 30.0


def session_of(log_name):
    """Vehicle + start time: the log name without its last two fields."""
    return log_name.rsplit('_', 2)[0]


def city_of(map_location):
    return CITY.get(str(map_location), NA)


def speed_band(v0):
    if v0 is None or not math.isfinite(v0):
        return NA
    if v0 < 0.5:
        return 'stopped'
    if v0 < 5.0:
        return 'slow'
    if v0 < 10.0:
        return 'medium'
    return 'fast'


def command_of(onehot):
    """argmax of the 4-way driving command, first maximum on ties (numpy's rule)."""
    values = [float(v) for v in onehot]
    if len(values) != 4:
        return NA
    return COMMANDS[values.index(max(values))]


def lead_gap(boxes, names, velocities):
    """Closest same-corridor dynamic object ahead at t0, ego frame.

    boxes: rows (x, y, z, length, width, height, heading); velocities: rows (vx, vy, ...).
    Returns (gap_m, lead_forward_speed, class) or (None, 0.0, None). Same arithmetic as
    PrivilegedBrakeAgent._lead_gap_speed, so the gap equals the boxes models' lead input.
    """
    best_gap, best_v, best_cls = None, 0.0, None
    for box, name, vel in zip(boxes, names, velocities):
        name = str(name)
        if name not in DYNAMIC:
            continue
        x, y = float(box[0]), float(box[1])
        length, width = float(box[3]), float(box[4])
        half = LANE_HALF_WIDTH + 0.25 * width
        if name == 'pedestrian':
            half = max(half, 3.0)
        if x < LEAD_X_MIN or x > LEAD_X_MAX or abs(y) > half:
            continue
        gap = x - 0.5 * length - EGO_FRONT
        if best_gap is None or gap < best_gap:
            best_gap, best_v, best_cls = gap, float(vel[0]), name
    return best_gap, best_v, best_cls


def lead_band(gap):
    if gap is None:
        return 'none'
    if gap < 10.0:
        return 'close'
    if gap < 25.0:
        return 'mid'
    return 'far'


def junction_band(ego_inside, first_inside_s):
    """ego_inside: bool or None; first_inside_s: arc length (m) of the first centerline
    sample inside an intersection, or None."""
    if ego_inside is None:
        return NA
    if ego_inside:
        return 'in'
    if first_inside_s is not None and first_inside_s <= JUNCTION_NEAR_M + 1e-9:
        return 'near'
    return 'none'


def cl_arc_lengths(k=CL_POINTS, lookahead=CL_LOOKAHEAD):
    """Arc length of each centerline sample: np.linspace(0, lookahead, k)."""
    return [lookahead * i / (k - 1) for i in range(k)]


def unwrap(angles):
    out, prev, offset = [], None, 0.0
    for a in angles:
        if prev is not None:
            d = a - prev
            if d > math.pi:
                offset -= 2 * math.pi
            elif d < -math.pi:
                offset += 2 * math.pi
        out.append(a + offset)
        prev = a
    return out


def interp(x, xs, ys):
    """np.interp for increasing xs, clamped at the ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def curvature_deg(cl_rows, at_m=CURVATURE_AT_M):
    """abs heading change (deg) of the route centerline between 0 and at_m ahead.

    cl_rows: K rows (dx, dy, cos dtheta, sin dtheta) as stored in the frozen `cl` block.
    """
    if not cl_rows:
        return None
    heads = unwrap([math.atan2(r[3], r[2]) for r in cl_rows])
    s = cl_arc_lengths(len(cl_rows))
    return abs(math.degrees(interp(at_m, s, heads) - heads[0]))


def curvature_band(deg):
    if deg is None or not math.isfinite(deg):
        return NA
    if deg < 15.0:
        return 'straight'
    if deg < 45.0:
        return 'bend'
    return 'turn'


def _seg_project(px, py, ax, ay, bx, by, extend_end=False):
    """(distance, signed lateral, t) of point p against segment a->b; t clamped to [0,1]
    (or [0, inf) when extend_end). Signed lateral is positive to the left of a->b."""
    vx, vy = bx - ax, by - ay
    seg2 = vx * vx + vy * vy
    if seg2 <= 0.0:
        dx, dy = px - ax, py - ay
        return math.hypot(dx, dy), 0.0, 0.0
    t = ((px - ax) * vx + (py - ay) * vy) / seg2
    t = max(0.0, t) if extend_end else min(1.0, max(0.0, t))
    cx, cy = ax + t * vx, ay + t * vy
    dist = math.hypot(px - cx, py - cy)
    cross = vx * (py - ay) - vy * (px - ax)
    lateral = math.copysign(dist, cross) if dist > 0 else 0.0
    return dist, lateral, t


def dedupe(polyline, tol=1e-4):
    """Drop consecutive repeats: a route shorter than the 60-m lookahead is stored with its
    end point repeated, and the extension must start from the last real segment."""
    out = []
    for p in polyline:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def route_length(cl_rows):
    """Arc length of the stored centerline (60 m unless the route ends earlier)."""
    return polyline_length(dedupe([(r[0], r[1]) for r in cl_rows])) if cl_rows else None


def signed_offset(point, polyline):
    """Signed lateral offset (left positive) of point from a polyline whose last segment is
    extended in a straight line beyond its end (repeated end points dropped first)."""
    polyline = dedupe(polyline)
    if len(polyline) < 2:
        return None
    best = None
    n = len(polyline)
    for i in range(n - 1):
        (ax, ay), (bx, by) = polyline[i][:2], polyline[i + 1][:2]
        dist, lateral, _ = _seg_project(point[0], point[1], ax, ay, bx, by, extend_end=(i == n - 2))
        if best is None or dist < best[0]:
            best = (dist, lateral)
    return best[1]


def lane_change_delta(cl_rows, end_xy):
    """Δd = d(end) − d(origin) relative to the centerline polyline (ego frame)."""
    if not cl_rows:
        return None
    poly = [(r[0], r[1]) for r in cl_rows]
    d0, d1 = signed_offset((0.0, 0.0), poly), signed_offset(end_xy, poly)
    if d0 is None or d1 is None:
        return None
    return d1 - d0


def logged_polyline(logged):
    """Logged 4-s path as a polyline from the origin through the logged poses. If the ego
    barely moves (< 0.5 m), a ray straight ahead stands in, so the extension of the last
    segment (signed_offset) is still defined."""
    pts = dedupe([(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in logged])
    if len(pts) < 2 or polyline_length(pts) < 0.5:
        return [(0.0, 0.0), (1.0, 0.0)]
    return pts


def lane_change_band(delta):
    if delta is None or not math.isfinite(delta):
        return NA
    return 'yes' if abs(delta) >= 2.0 else 'no'


def stop_go(v0, future):
    """Logged speed regime over the 4-s horizon (v0 at t0, future = 8 logged speeds)."""
    if v0 is None or not future or any(v is None for v in future):
        return NA
    if max([v0] + list(future)) < 0.5:
        return 'stationary'
    if v0 < 1.0 and max(future) >= 2.0:
        return 'starting'
    if v0 >= 2.0 and min(future) < 0.5:
        return 'stopping'
    return 'moving'


# --------------------------------------------------------------------------- outcomes

OUTCOMES = ['pdms', 'fail_pdms', 'fail_nc', 'fail_dac', 'fail_ttc', 'fail_c', 'fail_ep']
OUTCOME_LABEL = {'pdms': 'PDMS', 'fail_pdms': 'PDMS<0.5', 'fail_nc': 'NC fail',
                 'fail_dac': 'DAC fail', 'fail_ttc': 'TTC fail', 'fail_c': 'C fail',
                 'fail_ep': 'EP<0.5'}


def outcomes(row):
    """Per-scene outcome values from one per-scene CSV row (floats)."""
    nc = row['no_at_fault_collisions']
    dac = row['drivable_area_compliance']
    return {
        'pdms': row['score'],
        'fail_pdms': 1.0 if row['score'] < 0.5 else 0.0,
        'fail_nc': 1.0 if nc < 1 else 0.0,
        'fail_dac': 1.0 if dac < 1 else 0.0,
        'fail_ttc': 1.0 if row['time_to_collision_within_bound'] < 1 else 0.0,
        'fail_c': 1.0 if row['comfort'] < 1 else 0.0,
        'fail_ep': 1.0 if (row['ego_progress'] < 0.5 and nc * dac > 0) else 0.0,
    }


# --------------------------------------------------------------------------- bootstrap

def cluster_ci(values, groups, b=10000, seed=12345):
    """Resample entire logs; retain scene-weighted mean (ratio of sums/counts).

    Verbatim copy of ``cluster_ci`` in analysis/analyze_clustered.py (controlled-study
    branch, commit ad049d1), kept here because this branch starts from main.
    """
    if not values or len(values) != len(groups) or b < 100:
        raise ValueError('nonempty aligned values/groups and >=100 draws required')
    totals = {}
    for value, group in zip(values, groups):
        if not math.isfinite(value):
            raise ValueError('nonfinite score')
        total, count = totals.get(group, (0.0, 0))
        totals[group] = (total + value, count + 1)
    blocks = [totals[g] for g in sorted(totals)]
    if len(blocks) < 2:
        raise ValueError('at least two logs required')
    rng = random.Random(seed)
    samples = []
    for _ in range(b):
        picked = rng.choices(blocks, k=len(blocks))
        samples.append(sum(x[0] for x in picked) / sum(x[1] for x in picked))
    samples.sort()
    return sum(values) / len(values), samples[int(.025*b)], samples[int(.975*b)]


_INDEX_CACHE = {}


def replay_indices(n_clusters, b=B, seed=SEED):
    """The block indices ``random.Random(seed).choices(blocks, k=n)`` picks in each of the
    b draws of cluster_ci (choices without weights draws floor(random() * n))."""
    key = (n_clusters, b, seed)
    if key not in _INDEX_CACHE:
        rnd = random.Random(seed).random
        n = float(n_clusters)
        _INDEX_CACHE[key] = [[int(rnd() * n) for _ in range(n_clusters)] for _ in range(b)]
    return _INDEX_CACHE[key]


def _np():
    try:
        import numpy
        return numpy
    except ImportError:  # CI runs the standard library only
        return None


class Clusters(object):
    """Cluster bootstrap for many statistics over one fixed set of scenes.

    groups: cluster label per scene. Clusters are sorted by label, as in cluster_ci, and
    resampled with cluster_ci's random stream, so ``ratio_ci`` of one value vector equals
    cluster_ci(values, groups) (to floating-point summation order).
    """

    def __init__(self, groups, b=B, seed=SEED, use_numpy=True):
        self.labels = sorted(set(groups))
        if len(self.labels) < 2:
            raise ValueError('at least two clusters required')
        pos = {g: i for i, g in enumerate(self.labels)}
        self.member = [pos[g] for g in groups]
        self.n = len(groups)
        self.k = len(self.labels)
        self.b = b
        self.idx = replay_indices(self.k, b, seed)
        self.np = _np() if use_numpy else None
        self._w = None

    def _weights(self):
        if self._w is None:
            np = self.np
            idx = np.asarray(self.idx, dtype=np.int64)
            flat = (idx + (np.arange(self.b, dtype=np.int64) * self.k)[:, None]).ravel()
            self._w = np.bincount(flat, minlength=self.b * self.k).reshape(self.b, self.k).astype(np.float64)
        return self._w

    def block_sums(self, values, mask=None):
        """Per-cluster (sum, count) of values over the scenes where mask is true."""
        sums, counts = [0.0] * self.k, [0] * self.k
        for i, v in enumerate(values):
            if mask is not None and not mask[i]:
                continue
            c = self.member[i]
            sums[c] += v
            counts[c] += 1
        return sums, counts

    def ratio_draws(self, sums, counts):
        """Bootstrap draws of sum/count; None where a draw holds no scene at all."""
        if self.np is not None:
            np = self.np
            w = self._weights()
            num = w @ np.asarray(sums, dtype=np.float64)
            den = w @ np.asarray(counts, dtype=np.float64)
            with np.errstate(invalid='ignore', divide='ignore'):
                out = num / den
            return [None if d == 0 else float(v) for v, d in zip(out.tolist(), den.tolist())]
        draws = []
        for row in self.idx:
            den = sum(counts[i] for i in row)
            draws.append(None if den == 0 else sum(sums[i] for i in row) / den)
        return draws

    def ratio_draws_many(self, sums_cols, counts_cols):
        """Several statistics at once (lists of per-cluster columns). numpy only."""
        np = self.np
        w = self._weights()
        num = w @ np.asarray(sums_cols, dtype=np.float64).T
        den = w @ np.asarray(counts_cols, dtype=np.float64).T
        with np.errstate(invalid='ignore', divide='ignore'):
            return num / den, den


def percentile_ci(draws, b=None):
    """2.5/97.5 percentiles at cluster_ci's positions; empty draws are dropped first.

    Returns (low, high, n_dropped). With no empty draw the positions are exactly
    int(.025*b) and int(.975*b) of the sorted draws."""
    kept = sorted(d for d in draws if d is not None and not (isinstance(d, float) and math.isnan(d)))
    dropped = len(draws) - len(kept)
    if len(kept) < 100:
        return None, None, dropped
    m = len(kept)
    return kept[int(.025 * m)], kept[int(.975 * m)], dropped


def mean_ci(values, groups, b=B, seed=SEED, use_numpy=True):
    """cluster_ci through the replay path: (mean, low, high)."""
    cl = Clusters(groups, b, seed, use_numpy)
    sums, counts = cl.block_sums(values)
    lo, hi, _ = percentile_ci(cl.ratio_draws(sums, counts))
    return sum(values) / len(values), lo, hi


def diff_ci(values, groups, in_a, in_b, b=B, seed=SEED, use_numpy=True):
    """Δ_A − Δ_B (scene-weighted means of two disjoint scene sets) with all clusters
    resampled together. Returns (diff, low, high, dropped_draws)."""
    cl = Clusters(groups, b, seed, use_numpy)
    sa, ca = cl.block_sums(values, in_a)
    sb, cb = cl.block_sums(values, in_b)
    da, db = cl.ratio_draws(sa, ca), cl.ratio_draws(sb, cb)
    draws = [None if (x is None or y is None) else x - y for x, y in zip(da, db)]
    na, nb = sum(ca), sum(cb)
    if not na or not nb:
        return None, None, None, b
    point = sum(sa) / na - sum(sb) / nb
    lo, hi, dropped = percentile_ci(draws)
    return point, lo, hi, dropped


def verdict(lo, hi):
    if lo is None or hi is None:
        return 'n/a'
    return 'excludes 0' if (lo > 0 or hi < 0) else 'spans 0'


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------- root causes

STRATA = [
    # id, label, failing run template, contrast run template, condition
    ('S1', 'rule map-fixed', 'kin', 'privmapkin'),
    ('S2', 'learned map-fixed', 'l_pose_kin_s{s}', 'l_pose_map_s{s}'),
    ('S3', 'box-induced', 'l_pose_map_agents_s{s}', 'l_pose_map_s{s}'),
    ('S4', 'learned residual', 'l_pose_map_s{s}', None),
    ('S5', 'rule residual', 'privmapkin', None),
]
PER_STRATUM = 24
SEEDS = (0, 1, 2)
SALT = 'p10-root-cause'


def stratum_member(stratum, failing, contrast):
    """Is a scene in the stratum, given the failing run's and the contrast run's PDMS?"""
    if stratum in ('S1', 'S2'):
        return failing < 0.5 and contrast >= 0.5
    if stratum == 'S3':
        return contrast >= 0.5 and failing < 0.5
    if stratum in ('S4', 'S5'):
        return failing < 0.5
    raise ValueError(stratum)


def hash_rank(stratum, seed, token):
    return hashlib.sha256('|'.join([SALT, stratum, str(seed), token]).encode('utf-8')).hexdigest()


def select_cases(stratum, candidates_by_seed, token_log, per_seed):
    """Hash-ordered sample, at most one case per log within the stratum.

    candidates_by_seed: {seed: [token, ...]} (seed None for single-run strata).
    Seeds are filled in ascending order; logs used by an earlier seed stay used.
    """
    used, out = set(), []
    for seed in sorted(candidates_by_seed, key=lambda s: -1 if s is None else s):
        ranked = sorted(candidates_by_seed[seed], key=lambda t: (hash_rank(stratum, seed, t), t))
        taken = 0
        for token in ranked:
            log = token_log[token]
            if log in used:
                continue
            used.add(log)
            out.append({'stratum': stratum, 'seed': seed, 'token': token, 'log': log,
                        'hash': hash_rank(stratum, seed, token)})
            taken += 1
            if taken == per_seed:
                break
    return out


def primary_failure(d):
    """Earliest multiplicative event (NC vs DAC, ties -> DAC), then TTC, comfort, other."""
    events = []
    if d['dac'] < 1 and d.get('dac_step') is not None:
        events.append((d['dac_step'], 0, 'dac'))
    if d['nc'] < 1 and d.get('nc_step') is not None:
        events.append((d['nc_step'], 1, 'nc'))
    if events:
        return min(events)[2]
    if d['ttc'] < 1:
        return 'ttc'
    if d['c'] < 1:
        return 'comfort'
    return 'other'


def root_cause(d):
    """Root-cause label of one re-scored failure (rules in docs/FAILURE_TAXONOMY.md)."""
    if not d.get('reproduced', True):
        return 'unreproduced'
    if d.get('fallback'):
        return 'fallback'
    kind = primary_failure(d)
    if kind == 'dac':
        if d['dac_lateral_m'] >= 1.0:
            return 'dac_lateral_curve' if abs(d['logged_heading_change_deg']) >= 15.0 else 'dac_lateral_straight'
        if d['dac_travelled_m'] >= d['logged_length_m'] + 1.0:
            return 'dac_overshoot'
        return 'dac_near_logged_path'
    if kind == 'nc':
        if d['nc'] == 0.5:
            return 'nc_static'
        if d.get('nc_object_type') in ('pedestrian', 'bicycle'):
            return 'nc_vru'
        if d.get('nc_collision_type') == 'active_lateral':
            return 'nc_lateral'
        if d.get('nc_heading_diff_deg') is not None and d['nc_heading_diff_deg'] < 30.0 and d.get('nc_object_ahead'):
            return 'nc_rear_end'
        return 'nc_crossing'
    if kind == 'ttc':
        if d.get('ttc_heading_diff_deg') is not None and d['ttc_heading_diff_deg'] < 30.0 and d.get('ttc_object_ahead'):
            return 'ttc_following'
        return 'ttc_crossing'
    return kind


def plan_difference(box_len, map_len, box_end, map_end, map_heading):
    """S3 descriptor: how the boxes model's plan differs from the map-only plan."""
    if box_len <= map_len - 2.0:
        return 'slower'
    if box_len >= map_len + 2.0:
        return 'faster'
    dx, dy = box_end[0] - map_end[0], box_end[1] - map_end[1]
    lateral = abs(-math.sin(map_heading) * dx + math.cos(map_heading) * dy)
    return 'lateral' if lateral >= 0.5 else 'similar'


def polyline_length(points):
    return sum(math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
               for i in range(len(points) - 1))


def heading_diff_deg(a, b):
    """Absolute wrapped difference of two headings, degrees in [0, 180]."""
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))
