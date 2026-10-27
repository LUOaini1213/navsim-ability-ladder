# -*- coding: utf-8 -*-
"""Where do the planners fail, and which scene types carry the headline effects?

    python analysis/failure_taxonomy_analyze.py              # -> results/failure_taxonomy/*.md|csv|json
    python analysis/failure_taxonomy_analyze.py --check      # recompute point estimates, compare

Reads the cached scene features (results/failure_taxonomy/features.csv.gz, written by
failure_taxonomy_features.py) and the committed per-scene PDM scores
(results/navtest/per_scene/*.csv.gz). It never re-scores anything. Statistics follow
docs/FAILURE_TAXONOMY.md: scene-weighted category means with the log- and session-cluster
bootstrap of analyze_clustered.py (10,000 draws, seed 12345), contributions of each
category to the headline effects, the four pre-registered hypotheses, and the iid
scene bootstrap of the published tables as a reference for where clustering matters.

numpy makes the bootstrap fast (about a minute); without it the same numbers come out of
the pure-Python path in failure_taxonomy_core, only much more slowly. ``--check`` needs the
standard library only.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import failure_taxonomy_core as core  # noqa: E402
from ladder_io import REPO, bootstrap_large, open_text  # noqa: E402

OUT = os.path.join(REPO, 'results', 'failure_taxonomy')
PER_SCENE = os.path.join(REPO, 'results', 'navtest', 'per_scene')
FEATURES = os.path.join(OUT, 'features.csv.gz')
SEEDS = (0, 1, 2)
COLS = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
        'time_to_collision_within_bound', 'comfort', 'score']

RULE = ['cv', 'kin', 'privbrake', 'egomlp_s0', 'egomlp_s1', 'egomlp_s2', 'privmap_idm', 'privmapkin',
        'privgtpath_kin', 'privmap_gtspd', 'human']
CELLS = ['pose_kin', 'pose_agents', 'pose_map', 'pose_map_agents', 'ds_kin', 'ds_map', 'ds_map_agents']
# display name -> (label, member runs averaged per scene)
AGENTS = [
    ('cv', 'ConstantVelocity', ['cv']),
    ('kin', 'Kinematic rule', ['kin']),
    ('privbrake', 'PrivBrake rule (+GT boxes)', ['privbrake']),
    ('egomlp', 'EgoStatusMLP (official, 3 seeds)', ['egomlp_s0', 'egomlp_s1', 'egomlp_s2']),
    ('privmap_idm', 'PrivMap rule, IDM', ['privmap_idm']),
    ('privmapkin', 'PrivMapKin rule (+GT map)', ['privmapkin']),
    ('privgtpath_kin', 'PrivGTPathKin', ['privgtpath_kin']),
    ('privmap_gtspd', 'PrivMapGTSpd', ['privmap_gtspd']),
    ('human', 'Human', ['human']),
] + [(c, 'MLP %s (3 seeds)' % c, ['l_%s_s%d' % (c, s) for s in SEEDS]) for c in CELLS] \
  + [('%s_s%d' % (c, s), 'MLP %s seed %d' % (c, s), ['l_%s_s%d' % (c, s)]) for c in CELLS for s in SEEDS] \
  + [('egomlp_s%d' % s, 'EgoStatusMLP seed %d' % s, ['egomlp_s%d' % s]) for s in SEEDS]
HEADLINE = ['kin', 'privbrake', 'privmapkin', 'pose_kin', 'pose_agents', 'pose_map', 'pose_map_agents']
# column order of every markdown table (must match the table headers)
DISPLAY = ['pdms', 'fail_pdms', 'fail_dac', 'fail_nc', 'fail_ttc', 'fail_ep', 'fail_c']
PDMS_TABLE = ['cv', 'kin', 'privbrake', 'egomlp', 'privmapkin', 'pose_kin', 'pose_agents', 'pose_map',
              'pose_map_agents', 'ds_map', 'privmap_gtspd', 'human']

EFFECT_LABEL = {'E1': 'rule map gain (privmapkin - kin)',
                'E2': 'learned map gain (pose_map - pose_kin)',
                'E3': 'boxes added to the map-aware MLP (pose_map_agents - pose_map)',
                'E4': 'map x boxes interaction'}


def effect_terms(eff, seed):
    """{run: weight} of an effect for one seed ('avg' averages the three seeds)."""
    if eff == 'E1':
        return {'privmapkin': 1.0, 'kin': -1.0}
    seeds = SEEDS if seed == 'avg' else (seed,)
    w = 1.0 / len(seeds)
    out = {}
    for s in seeds:
        if eff == 'E2':
            terms = {'l_pose_map_s%d' % s: 1, 'l_pose_kin_s%d' % s: -1}
        elif eff == 'E3':
            terms = {'l_pose_map_agents_s%d' % s: 1, 'l_pose_map_s%d' % s: -1}
        elif eff == 'E4':
            terms = {'l_pose_map_agents_s%d' % s: 1, 'l_pose_map_s%d' % s: -1,
                     'l_pose_agents_s%d' % s: -1, 'l_pose_kin_s%d' % s: 1}
        else:
            raise ValueError(eff)
        for k, v in terms.items():
            out[k] = out.get(k, 0.0) + w * v
    return out


EFFECTS = [('E1', None)] + [(e, s) for e in ('E2', 'E3', 'E4') for s in ('avg',) + SEEDS]


def effect_name(eff, seed):
    return eff if seed is None else '%s_%s' % (eff, seed if seed == 'avg' else 's%d' % seed)


# --------------------------------------------------------------------------- loading

def load_features(path=FEATURES):
    with open_text(path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r['token'])
    return rows


def load_run(stem, tokens):
    """Per-scene outcome series of one committed run, aligned to tokens."""
    with open_text(os.path.join(PER_SCENE, stem + '.csv.gz')) as f:
        by_tok = {}
        for r in csv.DictReader(f):
            by_tok[r['token']] = core.outcomes({c: float(r[c]) for c in COLS})
    if set(by_tok) != set(tokens):
        raise SystemExit('%s: token set differs from the feature cache' % stem)
    return {m: [by_tok[t][m] for t in tokens] for m in core.OUTCOMES}


def build_series(tokens):
    runs = {}
    for stem in RULE + ['l_%s_s%d' % (c, s) for c in CELLS for s in SEEDS]:
        runs[stem] = load_run(stem, tokens)
    series = {}
    n = len(tokens)
    for name, _label, members in AGENTS:
        series[name] = {m: [sum(runs[r][m][i] for r in members) / len(members) for i in range(n)]
                        for m in core.OUTCOMES}
    for eff, seed in EFFECTS:
        terms = effect_terms(eff, seed)
        series[effect_name(eff, seed)] = {
            m: [sum(w * runs[r][m][i] for r, w in terms.items()) for i in range(n)] for m in core.OUTCOMES}
    return runs, series


# --------------------------------------------------------------------------- statistics

class Stats(object):
    """Means and cluster intervals of many series over scene subsets."""

    def __init__(self, series, logs, sessions, draws, use_numpy):
        self.names = sorted(series)
        self.series = series
        self.logs, self.sessions = logs, sessions
        self.draws = draws
        self.np = core._np() if use_numpy else None
        if self.np is not None:
            np = self.np
            self.matrix = np.array([series[s][m] for s in self.names for m in core.OUTCOMES], dtype=np.float64).T
            self.columns = [(s, m) for s in self.names for m in core.OUTCOMES]

    def subset_ci(self, idx, level, wanted=None):
        """{(series, metric): (low, high)} for the scenes idx, clusters = logs or sessions,
        resampling only the clusters present in the subset (cluster_ci applied to the subset)."""
        groups = [(self.logs if level == 'log' else self.sessions)[i] for i in idx]
        cl = core.Clusters(groups, self.draws, core.SEED, use_numpy=self.np is not None)
        out = {}
        if self.np is not None:
            np = self.np
            onehot = np.zeros((cl.k, len(idx)))
            onehot[np.asarray(cl.member), np.arange(len(idx))] = 1.0
            sums = onehot @ self.matrix[np.asarray(idx)]
            counts = onehot.sum(axis=1)
            w = cl._weights()
            den = w @ counts
            num = w @ sums
            draws = num / den[:, None]
            draws.sort(axis=0)
            lo_pos, hi_pos = int(.025 * self.draws), int(.975 * self.draws)
            for j, key in enumerate(self.columns):
                if wanted is None or key in wanted:
                    out[key] = (float(draws[lo_pos, j]), float(draws[hi_pos, j]))
            return out
        for key in wanted or [(s, m) for s in self.names for m in core.OUTCOMES]:
            vals = [self.series[key[0]][key[1]][i] for i in idx]
            sums, counts = cl.block_sums(vals)
            lo, hi, _ = core.percentile_ci(cl.ratio_draws(sums, counts))
            out[key] = (lo, hi)
        return out


def mean(values, idx):
    return sum(values[i] for i in idx) / len(idx) if idx else float('nan')


def small(n, n_logs, n_sessions):
    return n < 50 or n_logs < 5 or n_sessions < 3


def categories(rows, feature):
    present = {r[feature] for r in rows}
    cats = [c for c in dict(core.FEATURES)[feature] if c in present]
    return cats + sorted(present - set(cats))


# --------------------------------------------------------------------------- formatting

def f3(x):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else '%.3f' % x


def s3(x):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else '%+.3f' % x


def pct(x):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else '%.1f%%' % (100 * x)


def ci3(lo, hi, signed=False):
    if lo is None:
        return '--'
    f = s3 if signed else f3
    return '[%s, %s]' % (f(lo), f(hi))


def write_csv(path, header, rows):
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        for r in rows:
            w.writerow(['' if v is None else ('%.6f' % v if isinstance(v, float) else v) for v in r])


# --------------------------------------------------------------------------- analysis

def analyse(draws=core.B, use_numpy=True, iid=True, log=print):
    t0 = time.time()
    rows = load_features()
    tokens = [r['token'] for r in rows]
    logs = [r['log'] for r in rows]
    sessions = [r['session'] for r in rows]
    N = len(tokens)
    _runs, series = build_series(tokens)
    del _runs
    st = Stats(series, logs, sessions, draws, use_numpy)
    log('loaded %d scenes, %d series (%.0f s)' % (N, len(series), time.time() - t0))

    subsets = [('all', 'all', list(range(N)))]
    for feature in core.FEATURE_NAMES:
        for cat in categories(rows, feature):
            subsets.append((feature, cat, [i for i in range(N) if rows[i][feature] == cat]))

    cat_rows, eff_rows, meta = [], [], {}
    for feature, cat, idx in subsets:
        n, nl, ns = len(idx), len({logs[i] for i in idx}), len({sessions[i] for i in idx})
        is_small = small(n, nl, ns)
        meta[(feature, cat)] = dict(n=n, logs=nl, sessions=ns, small=is_small)
        cis = {}
        if not is_small:
            cis['log'] = st.subset_ci(idx, 'log')
            cis['session'] = st.subset_ci(idx, 'session')
        for name in st.names:
            for m in core.OUTCOMES:
                lo_l, hi_l = cis['log'][(name, m)] if cis else (None, None)
                lo_s, hi_s = cis['session'][(name, m)] if cis else (None, None)
                rec = [feature, cat, name, m, n, nl, ns, int(is_small), mean(series[name][m], idx),
                       lo_l, hi_l, lo_s, hi_s]
                (eff_rows if name[:2] in ('E1', 'E2', 'E3', 'E4') else cat_rows).append(rec)
        log('  %-12s %-11s n=%5d logs=%3d sessions=%2d%s (%.0f s)' % (
            feature, cat, n, nl, ns, ' small' if is_small else '', time.time() - t0))

    # contributions of each category to each effect (all scenes, values masked)
    contrib = []
    eff_names = [effect_name(e, s) for e, s in EFFECTS]
    for level, groups in (('log', logs), ('session', sessions)):
        cl = core.Clusters(groups, draws, core.SEED, use_numpy=use_numpy)
        for feature, cat, idx in subsets[1:]:
            inset = set(idx)
            for name in eff_names:
                vals = series[name]['pdms']
                masked = [v if i in inset else 0.0 for i, v in enumerate(vals)]
                sums, counts = cl.block_sums(masked)
                lo, hi, _ = core.percentile_ci(cl.ratio_draws(sums, counts))
                contrib.append((level, feature, cat, name, sum(masked) / N, lo, hi))
    log('contributions done (%.0f s)' % (time.time() - t0))

    # iid scene bootstrap of the published tables, effect category means only
    iid_rows = {}
    if iid:
        for feature, cat, idx in subsets:
            if meta[(feature, cat)]['small']:
                continue
            for name in ('E1', 'E2_avg', 'E3_avg', 'E4_avg'):
                m, lo, hi = bootstrap_large([series[name]['pdms'][i] for i in idx])
                iid_rows[(feature, cat, name)] = (m, lo, hi)
        log('iid reference done (%.0f s)' % (time.time() - t0))

    hyps = hypotheses(rows, series, logs, sessions, draws, use_numpy)
    log('hypotheses done (%.0f s)' % (time.time() - t0))

    # exploratory, added after the city breakdown was seen: each effect without one city
    loco = []
    wanted = {(effect_name(e, s), 'pdms') for e, s in EFFECTS}
    for city in categories(rows, 'city'):
        idx = [i for i in range(N) if rows[i]['city'] != city]
        cis = {lv: st.subset_ci(idx, lv, wanted) for lv in ('log', 'session')}
        for e, s in EFFECTS:
            name = effect_name(e, s)
            loco.append(dict(effect=name, without=city, scenes=len(idx),
                             sessions=len({sessions[i] for i in idx}), delta=mean(series[name]['pdms'], idx),
                             log=list(cis['log'][(name, 'pdms')]), session=list(cis['session'][(name, 'pdms')])))
    log('leave-one-city-out done (%.0f s)' % (time.time() - t0))
    return dict(rows=rows, series=series, subsets=subsets, meta=meta, cat_rows=cat_rows,
                eff_rows=eff_rows, contrib=contrib, iid=iid_rows, hyps=hyps, loco=loco, N=N, draws=draws,
                seconds=time.time() - t0)


HYPOTHESES = [
    ('H1', 'E2', 'curvature', ('bend', 'turn'), ('straight',), '>'),
    ('H1r', 'E1', 'curvature', ('bend', 'turn'), ('straight',), '>'),
    ('H3a', 'E3', 'lead', ('none',), None, '<'),
    ('H3b', 'E3', 'lead', ('close', 'mid', 'far'), ('none',), '<'),
]


def hypotheses(rows, series, logs, sessions, draws, use_numpy):
    out = []
    for hid, eff, feature, group_a, group_b, sign in HYPOTHESES:
        seeds = [None] if eff == 'E1' else ['avg', 0, 1, 2]
        in_a = [r[feature] in group_a for r in rows]
        in_b = [r[feature] in group_b for r in rows] if group_b else None
        for seed in seeds:
            name = effect_name(eff, seed)
            vals = series[name]['pdms']
            for level, groups in (('log', logs), ('session', sessions)):
                if group_b:
                    d, lo, hi, dropped = core.diff_ci(vals, groups, in_a, in_b, draws, core.SEED, use_numpy)
                else:
                    idx = [i for i, a in enumerate(in_a) if a]
                    d, lo, hi = core.mean_ci([vals[i] for i in idx], [groups[i] for i in idx], draws,
                                             core.SEED, use_numpy)
                    dropped = 0
                ok = (lo is not None) and ((lo > 0) if sign == '>' else (hi < 0))
                out.append(dict(hypothesis=hid, effect=name, level=level, estimate=d, low=lo, high=hi,
                                dropped_draws=dropped, supported_here=ok))
    return out


# --------------------------------------------------------------------------- reports

def index_rows(recs):
    return {(r[0], r[1], r[2], r[3]): r for r in recs}


def write_outputs(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    head = ['feature', 'category', 'series', 'metric', 'scenes', 'logs', 'sessions', 'small', 'mean',
            'log_low', 'log_high', 'session_low', 'session_high']
    write_csv(os.path.join(out_dir, 'category_outcomes.csv'), head, res['cat_rows'])
    eff = index_rows(res['eff_rows'])
    iid = res['iid']
    eff_out = []
    for r in res['eff_rows']:
        key = (r[0], r[1], r[2])
        i = iid.get(key) if r[3] == 'pdms' else None
        eff_out.append(r + [i[1] if i else None, i[2] if i else None])
    write_csv(os.path.join(out_dir, 'category_effects.csv'), head + ['iid_low', 'iid_high'], eff_out)
    write_csv(os.path.join(out_dir, 'category_contributions.csv'),
              ['level', 'feature', 'category', 'effect', 'contribution', 'low', 'high'], res['contrib'])
    with io.open(os.path.join(out_dir, 'hypotheses.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(res['hyps'], f, indent=2, sort_keys=True)
        f.write('\n')
    with io.open(os.path.join(out_dir, 'city_sensitivity.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(res['loco'], f, indent=2, sort_keys=True)
        f.write('\n')
    cat = index_rows(res['cat_rows'])
    texts = {
        'where_planners_fail.md': where_md(res, cat),
        'headline_effects.md': effects_md(res, eff),
        'hypotheses.md': hypotheses_md(res, eff),
        'clustering.md': clustering_md(res, eff),
        'city_sensitivity.md': city_sensitivity_md(res),
    }
    for name, text in texts.items():
        io.open(os.path.join(out_dir, name), 'w', encoding='utf-8', newline='\n').write(text)
    summary = summary_json(res, cat, eff)
    with io.open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write('\n')
    return texts


PREAMBLE = ['Scene-weighted means over the scenes of each category. Intervals: percentile 95%, '
            '10,000 draws, seed 12345,', 'whole logs (136) or whole sessions (44) resampled '
            '(`cluster_ci` of analyze_clustered.py, applied to the',
            "category's scenes). Learned cells average their three seeds per scene; their intervals "
            'condition on', 'those fitted models. *small* = fewer than 50 scenes, 5 logs or 3 sessions '
            '(no interval). Exploratory unless', 'listed in `hypotheses.md`.']


def where_md(res, cat):
    L = ['# Where the planners fail (navtest, 12,146 scenes)', '',
         'Generated by `analysis/failure_taxonomy_analyze.py` from `features.csv.gz` and the committed',
         'per-scene scores. Taxonomy and rules: `docs/FAILURE_TAXONOMY.md`. Every value with both',
         'intervals: `category_outcomes.csv`.', ''] + PREAMBLE + ['']
    label = {a: l for a, l, _ in AGENTS}
    L += ['Failure definitions: PDMS<0.5; NC fail = NC < 1; DAC fail = DAC < 1; TTC fail = TTC < 1;',
          'C fail = comfort < 1; EP<0.5 = ego progress below 0.5 while NC and DAC are non-zero.', '']
    meta = res['meta']
    for feature in ['all'] + core.FEATURE_NAMES:
        cats = ['all'] if feature == 'all' else categories(res['rows'], feature)
        L += ['## %s' % ('All scenes' if feature == 'all' else 'Feature: `%s`' % feature), '']
        L += ['### PDMS by category', '',
              '| category | scenes | logs | ' + ' | '.join(PDMS_TABLE) + ' |',
              '|---|---:|---:|' + '---:|' * len(PDMS_TABLE)]
        for c in cats:
            m = meta[(feature, c)]
            vals = ' | '.join(f3(cat[(feature, c, a, 'pdms')][8]) for a in PDMS_TABLE)
            L.append('| %s%s | %d | %d | %s |' % (c, ' *(small)*' if m['small'] else '', m['n'], m['logs'], vals))
        L += ['', '### Failure rates of the headline planners, with log-cluster 95% intervals', '',
              '| category | planner | PDMS | PDMS<0.5 | DAC fail | NC fail | TTC fail | EP<0.5 | C fail |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for c in cats:
            for a in HEADLINE:
                cells = []
                for metric in DISPLAY:
                    r = cat[(feature, c, a, metric)]
                    if metric == 'pdms':
                        cells.append('%s %s' % (f3(r[8]), ci3(r[9], r[10])))
                    else:
                        cells.append('%s%s' % (pct(r[8]), '' if r[9] is None else ' [%s, %s]' % (
                            pct(r[9]).rstrip('%'), pct(r[10]))))
                L.append('| %s | %s | %s |' % (c, a, ' | '.join(cells)))
        L.append('')
    L += ['Planner key: ' + '; '.join('`%s` %s' % (a, label[a]) for a in PDMS_TABLE) + '.', '']
    return '\n'.join(L)


def effects_md(res, eff):
    L = ['# Which scene types carry the headline effects', '',
         'Per category: the mean paired effect inside the category (Δ_c) with log- and session-cluster',
         '95% intervals, and its **contribution** κ_c = Σ_{t in c} Δ_t / N, so the categories of one',
         'feature add up to the overall effect. *share* = κ_c / Δ; *conc.* = share / scene share =',
         'Δ_c / Δ (above 1: the category carries more than its share of scenes). Per-seed values in',
         '`category_effects.csv` and `category_contributions.csv`. Exploratory except where',
         '`hypotheses.md` says otherwise.', '']
    contrib = {(lv, f, c, e): (k, lo, hi) for lv, f, c, e, k, lo, hi in res['contrib']}
    meta = res['meta']
    N = res['N']
    for e, seeds in (('E1', [None]), ('E2', ['avg', 0, 1, 2]), ('E3', ['avg', 0, 1, 2]), ('E4', ['avg', 0, 1, 2])):
        main_name = effect_name(e, seeds[0])
        total = eff[('all', 'all', main_name, 'pdms')]
        L += ['## %s: %s' % (e, EFFECT_LABEL[e]), '',
              'Overall%s: Δ = %s, log %s, session %s.' % (
                  '' if e == 'E1' else ' (seed average)', s3(total[8]), ci3(total[9], total[10], True),
                  ci3(total[11], total[12], True)), '']
        if e != 'E1':
            per = [eff[('all', 'all', effect_name(e, s), 'pdms')] for s in (0, 1, 2)]
            L += ['Per seed: ' + '; '.join('s%d %s log %s session %s' % (s, s3(p[8]), ci3(p[9], p[10], True),
                                                                          ci3(p[11], p[12], True))
                                           for s, p in zip((0, 1, 2), per)) + '.', '']
        hdr = '| feature | category | scenes | Δ_c | log 95% | session 95% | κ_c | share | conc. |'
        sep = '|---|---|---:|---:|---|---|---:|---:|---:|'
        if e != 'E1':
            hdr += ' Δ_c s0 / s1 / s2 | share s0 / s1 / s2 |'
            sep += '---|---|'
        L += [hdr, sep]
        for feature in core.FEATURE_NAMES:
            for c in categories(res['rows'], feature):
                m = meta[(feature, c)]
                r = eff[(feature, c, main_name, 'pdms')]
                k = contrib[('log', feature, c, main_name)]
                share = k[0] / total[8] if total[8] else float('nan')
                conc = share / (m['n'] / N)
                line = '| %s | %s%s | %d | %s | %s | %s | %s | %.0f%% | %.2f |' % (
                    feature, c, ' *(small)*' if m['small'] else '', m['n'], s3(r[8]),
                    ci3(r[9], r[10], True), ci3(r[11], r[12], True), s3(k[0]), 100 * share, conc)
                if e != 'E1':
                    ds, shs = [], []
                    for s in (0, 1, 2):
                        nm = effect_name(e, s)
                        ds.append(s3(eff[(feature, c, nm, 'pdms')][8]))
                        tot_s = eff[('all', 'all', nm, 'pdms')][8]
                        shs.append('%.0f%%' % (100 * contrib[('log', feature, c, nm)][0] / tot_s))
                    line += ' %s | %s |' % (' / '.join(ds), ' / '.join(shs))
                L.append(line)
        L += ['', '### How the effect arises: change in failure rates inside each category (%s)' % (
            'single run' if e == 'E1' else 'seed average'), '',
              '| feature | category | ΔPDMS<0.5 | ΔDAC fail | ΔNC fail | ΔTTC fail | ΔEP<0.5 | ΔC fail |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
        for feature in core.FEATURE_NAMES:
            for c in categories(res['rows'], feature):
                cells = [eff[(feature, c, main_name, mm)][8] for mm in DISPLAY[1:]]
                L.append('| %s | %s | %s |' % (feature, c, ' | '.join('%+.1f pp' % (100 * v) for v in cells)))
        L.append('')
    L += ['Rows sum: within one feature the κ_c add up to the overall Δ (up to rounding).', '']
    return '\n'.join(L)


HYP_TEXT = {
    'H1': 'learned map gain larger on curved routes: Δ_E2(bend ∪ turn) − Δ_E2(straight) > 0',
    'H1r': 'rule map gain larger on curved routes: Δ_E1(bend ∪ turn) − Δ_E1(straight) > 0',
    'H3a': 'box penalty present without a lead: Δ_E3(lead = none) < 0',
    'H3b': 'box penalty larger with a lead: Δ_E3(lead ∈ close, mid, far) − Δ_E3(none) < 0',
}


def hypotheses_md(res, eff):
    L = ['# Pre-registered hypotheses (docs/FAILURE_TAXONOMY.md)', '',
         'Decided under both log- and session-cluster bootstrap (10,000 draws, seed 12345); for E2/E3',
         'the seed average and each of the three seeds must agree. "confirmatory" here means "specified',
         'before the analysis"; navtest had already been used in this project, so this is not fresh-data',
         'confirmation. Group differences resample all clusters together; draws in which a group is',
         'empty are dropped (counted below).', '',
         '| hypothesis | effect | estimate | log 95% | session 95% | dropped draws (log/session) | holds here |',
         '|---|---|---:|---|---|---|---|']
    by = {}
    for h in res['hyps']:
        by.setdefault((h['hypothesis'], h['effect']), {})[h['level']] = h
    verdicts = {}
    for (hid, name), lv in by.items():
        lg, se = lv['log'], lv['session']
        ok = lg['supported_here'] and se['supported_here']
        verdicts.setdefault(hid, []).append(ok)
        L.append('| %s | %s | %s | %s | %s | %d / %d | %s |' % (
            hid, name, s3(lg['estimate']), ci3(lg['low'], lg['high'], True), ci3(se['low'], se['high'], True),
            lg['dropped_draws'], se['dropped_draws'], 'yes' if ok else 'no'))
    L += ['', '## Verdicts', '']
    for hid in ('H1', 'H1r', 'H3a', 'H3b'):
        ok = all(verdicts[hid])
        L.append('- **%s** (%s): %s.' % (hid, HYP_TEXT[hid], 'supported' if ok else
                                         'not supported (%d of %d effect/seed rows hold under both clusterings)'
                                         % (sum(verdicts[hid]), len(verdicts[hid]))))
    L.append('')
    return '\n'.join(L)


def clustering_md(res, eff):
    L = ['# Where clustering changes a conclusion', '',
         'For the seed-averaged effects (E1 single run), each category mean Δ_c gets three 95% intervals:',
         'the iid scene bootstrap of the published tables (`bootstrap_large`, 2,000 draws), the log-cluster',
         'and the session-cluster bootstrap (10,000 draws). Listed: every category where the three',
         'verdicts ("excludes 0" / "spans 0") are not all the same.', '',
         '| effect | feature | category | scenes | Δ_c | iid 95% | log 95% | session 95% | verdicts iid / log / session |',
         '|---|---|---|---:|---:|---|---|---|---|']
    changed, total = 0, 0
    for (feature, c, name), (m, lo, hi) in sorted(res['iid'].items()):
        r = eff[(feature, c, name, 'pdms')]
        v = (core.verdict(lo, hi), core.verdict(r[9], r[10]), core.verdict(r[11], r[12]))
        total += 1
        if len(set(v)) > 1:
            changed += 1
            L.append('| %s | %s | %s | %d | %s | %s | %s | %s | %s |' % (
                name, feature, c, r[4], s3(r[8]), ci3(lo, hi, True), ci3(r[9], r[10], True),
                ci3(r[11], r[12], True), ' / '.join(v)))
    L += ['', '%d of %d category-level effect intervals change verdict between the three bootstraps.' % (
        changed, total), '']
    widths = []
    for (feature, c, name), (m, lo, hi) in sorted(res['iid'].items()):
        r = eff[(feature, c, name, 'pdms')]
        if r[9] is not None and hi > lo:
            widths.append(((r[10] - r[9]) / (hi - lo), (r[12] - r[11]) / (hi - lo)))
    if widths:
        wl = sorted(w[0] for w in widths)
        ws = sorted(w[1] for w in widths)
        L += ['Interval width relative to iid (median over the %d rows): log %.2fx, session %.2fx.' % (
            len(widths), wl[len(wl) // 2], ws[len(ws) // 2]), '']
    return '\n'.join(L)


def city_sensitivity_md(res):
    L = ['# Exploratory: each headline effect with one city left out', '',
         'Added after the city breakdown in `headline_effects.md` had been seen (Singapore carries a',
         'large share of the box penalty from 16% of the scenes and only 2 sessions), so this is a',
         'post-hoc sensitivity check, not a pre-registered test. Each row drops every scene of one city',
         'and recomputes the scene-weighted effect with log- and session-cluster 95% intervals',
         '(10,000 draws, seed 12345).', '',
         '| effect | city left out | scenes | sessions | Δ | log 95% | session 95% | verdict log / session |',
         '|---|---|---:|---:|---:|---|---|---|']
    for r in res['loco']:
        L.append('| %s | %s | %d | %d | %s | %s | %s | %s / %s |' % (
            r['effect'], r['without'], r['scenes'], r['sessions'], s3(r['delta']),
            ci3(r['log'][0], r['log'][1], True), ci3(r['session'][0], r['session'][1], True),
            core.verdict(*r['log']), core.verdict(*r['session'])))
    L.append('')
    return '\n'.join(L)


def summary_json(res, cat, eff):
    out = {'scenes': res['N'], 'draws': res['draws'], 'runtime_s': round(res['seconds'], 1),
           'overall': {}, 'categories': {}}
    for name in ['E1'] + [effect_name(e, s) for e in ('E2', 'E3', 'E4') for s in ('avg', 0, 1, 2)]:
        r = eff[('all', 'all', name, 'pdms')]
        out['overall'][name] = {'delta': r[8], 'log': [r[9], r[10]], 'session': [r[11], r[12]]}
    for (feature, c), m in res['meta'].items():
        out['categories']['%s=%s' % (feature, c)] = m
    return out


# --------------------------------------------------------------------------- check

def check(out_dir):
    """Recompute every point estimate of the committed CSVs (standard library only)."""
    rows = load_features()
    tokens = [r['token'] for r in rows]
    _runs, series = build_series(tokens)
    N = len(tokens)
    problems = 0
    idx_of = {'all': {'all': list(range(N))}}
    for feature in core.FEATURE_NAMES:
        idx_of[feature] = {}
        for i, r in enumerate(rows):
            idx_of[feature].setdefault(r[feature], []).append(i)
    for name in ('category_outcomes.csv', 'category_effects.csv'):
        with io.open(os.path.join(out_dir, name), encoding='utf-8') as f:
            for r in csv.DictReader(f):
                idx = idx_of[r['feature']][r['category']]
                got = mean(series[r['series']][r['metric']], idx)
                if abs(got - float(r['mean'])) > 2e-6 or int(r['scenes']) != len(idx):
                    problems += 1
                    if problems < 10:
                        print('MISMATCH', name, r['feature'], r['category'], r['series'], r['metric'], got, r['mean'])
    with io.open(os.path.join(out_dir, 'category_contributions.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            idx = idx_of[r['feature']][r['category']]
            got = sum(series[r['effect']]['pdms'][i] for i in idx) / N
            if abs(got - float(r['contribution'])) > 2e-6:
                problems += 1
    print('point estimates: %s' % ('all match' if not problems else '%d mismatch(es)' % problems))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--draws', type=int, default=core.B)
    ap.add_argument('--no-numpy', action='store_true')
    ap.add_argument('--no-iid', action='store_true')
    ap.add_argument('--check', action='store_true', help='recompute point estimates of the committed outputs')
    a = ap.parse_args()
    if a.check:
        return 1 if check(a.out) else 0
    res = analyse(a.draws, not a.no_numpy, not a.no_iid)
    write_outputs(res, a.out)
    print('wrote results to %s (%.0f s)' % (a.out, res['seconds']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
