# -*- coding: utf-8 -*-
"""Paired bootstrap confidence intervals over the 9-agent ladder.

Reads the per-scene CSVs already produced by run_pdm_score.py — does NOT
re-run any evaluation. Writes exp/analysis/ci_report.md.

Usage:  python analyze_ci.py
"""
import csv
import glob
import io
import os
import random
import sys

EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp')
OUT = os.path.join(EXP, 'analysis', 'ci_report.md')
B = 10000
SEED = 12345

# Canonical run = the FIRST csv per agent, matching exp/analysis/pdm_report.md.
# NOTE: privileged_brake_mini has two runs (12:52 PDMS 0.602 / 12:55 PDMS 0.593).
# pdm_report.md used the first; we keep that for consistency and flag it below.
AGENTS = [
    ('CV', 'cv_agent_mini', 'speed only'),
    ('Kinematic', 'kinematic_agent_mini', 'v, a, command'),
    ('PrivBrake', 'privileged_brake_mini', 'GT boxes, no map'),
    ('MLP', 'mlp_agent_mini', 'learned kinematics (blind)'),
    ('MapMLP (learned+map)', 'map_mlp_mini', 'learned, sees centerline'),
    ('PrivMap(IDM)', 'privileged_centerline_mini', 'map centerline + IDM'),
    ('PrivMapKin', 'privileged_centerline_kin', 'map centerline + kinematic'),
    ('PrivGTPathKin', 'privileged_gtpath_kin', 'logged path + kinematic'),
    ('PrivMapGTSpd', 'privileged_centerline_gtspeed', 'map centerline + human arc-length'),
    ('Human', 'human_agent_mini', 'logged future'),
]

PAIRS = [
    ('CV', 'Kinematic', 'Add kinematics (accel + command yaw)'),
    ('Kinematic', 'MLP', 'Learned blind planner vs hand rule'),
    ('Kinematic', 'PrivBrake', 'Add GT detection boxes + brake'),
    ('MLP', 'PrivMapKin', 'Add on-route map centerline'),
    ('PrivMap(IDM)', 'PrivMapKin', 'Drop IDM on the centerline (counter-intuitive)'),
    ('PrivMapKin', 'PrivMapGTSpd', 'Swap in human arc-length speed (geometry fixed)'),
    ('PrivMapKin', 'PrivGTPathKin', 'Swap in logged path (speed fixed)'),
    ('PrivMapGTSpd', 'Human', 'Remaining gap to Human'),
    ('MLP', 'MapMLP (learned+map)', 'Give the LEARNED model the centerline too'),
    ('MapMLP (learned+map)', 'PrivMapKin', 'Same centerline: hand rule vs learned'),
]


def load(subdir):
    files = sorted(glob.glob(os.path.join(EXP, subdir, '*', '*.csv')))
    if not files:
        raise SystemExit('no csv under ' + subdir)
    rows = {}
    with io.open(files[0], encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tok = (r.get('token') or '').strip()
            if not tok or tok.lower() == 'average':
                continue
            try:
                rows[tok] = {
                    'score': float(r['score']),
                    'dac': float(r['drivable_area_compliance']),
                }
            except (ValueError, KeyError):
                continue
    return rows, os.path.relpath(files[0], EXP)


def bootstrap(diffs, seed=SEED):
    n = len(diffs)
    mean = sum(diffs) / n
    rnd = random.Random(seed)
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rnd.randrange(n)]
        means.append(s / n)
    means.sort()
    return mean, means[int(0.025 * B)], means[int(0.975 * B)]


def main():
    data, srcs = {}, {}
    for name, sub, _ in AGENTS:
        data[name], srcs[name] = load(sub)

    common = set(data[AGENTS[0][0]])
    for name, _, _ in AGENTS[1:]:
        common &= set(data[name])
    common = sorted(common)

    L = []
    L.append('# Paired bootstrap CIs over the agent ladder\n')
    L.append('Split `warmup_test_e2e`, %d scenes common to all %d agents, '
             'same metric cache. %d bootstrap resamples, seed %d. '
             'No evaluation was re-run — this reads the CSVs already in `exp/`.\n'
             % (len(common), len(AGENTS), B, SEED))

    L.append('\n## Ladder\n')
    L.append('| agent | sees | PDMS | DAC | DAC fail |')
    L.append('|---|---|---:|---:|---:|')
    for name, _, sees in AGENTS:
        d = data[name]
        n = len(common)
        pdms = sum(d[t]['score'] for t in common) / n
        dac = sum(d[t]['dac'] for t in common) / n
        fail = sum(1 for t in common if d[t]['dac'] < 0.5) / n
        L.append('| %s | %s | %.3f | %.3f | %.1f%% |' % (name, sees, pdms, dac, 100 * fail))

    L.append('\n## Paired deltas (positive = the second agent is better)\n')
    L.append('| comparison | dPDMS | 95% CI | verdict |')
    L.append('|---|---:|---|---|')
    for a, b, label in PAIRS:
        diffs = [data[b][t]['score'] - data[a][t]['score'] for t in common]
        m, lo, hi = bootstrap(diffs)
        verdict = 'CI excludes 0' if (lo > 0 or hi < 0) else 'CI spans 0'
        L.append('| %s | %+.4f | [%+.4f, %+.4f] | %s |' % (label, m, lo, hi, verdict))

    # difference-in-differences: speed vs geometry
    base, spd, geo = data['PrivMapKin'], data['PrivMapGTSpd'], data['PrivGTPathKin']
    dd = [(spd[t]['score'] - base[t]['score']) - (geo[t]['score'] - base[t]['score'])
          for t in common]
    m, lo, hi = bootstrap(dd, seed=7)
    L.append('\n## Speed vs geometry (difference in differences)\n')
    L.append('Gain from swapping in human speed **minus** gain from swapping in the '
             'logged path, same 563 paired scenes:\n')
    L.append('- %+.4f, 95%% CI [%+.4f, %+.4f] — %s\n'
             % (m, lo, hi, 'speed is worth significantly more'
                if lo > 0 else ('geometry is worth significantly more' if hi < 0
                                else 'not separable')))

    L.append('\n## Caveats\n')
    L.append('- `privileged_brake_mini` has **two** runs with different results '
             '(12:52 PDMS 0.602 / DAC 0.766; 12:55 PDMS 0.593 / DAC 0.785). '
             'This report and `pdm_report.md` both use the first. '
             'The gap is unexplained and should be pinned down before the number is quoted.')
    L.append('- MLP and MapMLP are the two learned agents; on this split both scores are '
             'contaminated (428 of 563 scenes were training logs). '
             'The clean comparison is the held-out 135 (MLP 0.475 vs CV 0.181).')
    L.append('- Priv* and Human consume ground truth or map privilege and are **not '
             'deployable** — they are upper bounds, not results.')
    L.append('- Ten comparisons were made; no multiplicity correction is applied. '
             'The two smallest effects (drop-IDM, logged-path) would not survive a '
             'strict Bonferroni threshold and should be reported as suggestive.')
    L.append('- `warmup_test_e2e` is not the official navtest leaderboard.')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('wrote', OUT)
    print('scenes=%d agents=%d' % (len(common), len(AGENTS)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
