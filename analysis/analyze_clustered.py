"""Paired log-cluster bootstrap sensitivity analysis of frozen navtest runs.

Run from a plain clone: python analysis/analyze_clustered.py
Seed-averaged effects condition on the three fitted models; intervals quantify
evaluation-log variation, not uncertainty over new training seeds.
"""
import argparse
import json
import math
import random
from pathlib import Path

from analyze_navtest import Runs, PER_SCENE, OUT
from ladder_io import token_to_log


def cluster_ci(values, groups, b=10000, seed=12345):
    """Resample entire logs; retain scene-weighted mean (ratio of sums/counts)."""
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


def cell(base):
    return {'l_%s_s%d' % (base, s): 1/3 for s in range(3)}


def contrast(a, b):
    result = {k: -v for k, v in a.items()}
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def comparisons():
    kin, agents, m, ma = [cell(x) for x in
                           ('pose_kin', 'pose_agents', 'pose_map', 'pose_map_agents')]
    return [
        ('rule: add map', contrast({'kin': 1}, {'privmapkin': 1})),
        ('rule: add boxes + brake', contrast({'kin': 1}, {'privbrake': 1})),
        ('pose: add map, no boxes', contrast(kin, m)),
        ('pose: add boxes, no map', contrast(kin, agents)),
        ('pose: add boxes, with map', contrast(m, ma)),
        ('pose: map x boxes interaction', contrast(contrast(kin, agents), contrast(m, ma))),
        ('rule path: learn speed from ego', contrast({'privmapkin': 1}, cell('ds_kin'))),
        ('rule path: learn speed from ego + map', contrast({'privmapkin': 1}, cell('ds_map'))),
        ('rule path: add boxes to speed model', contrast(cell('ds_map'), cell('ds_map_agents'))),
        ('map input: rule path vs full pose', contrast(m, cell('ds_map'))),
        ('pose seed 0: dropout 0.2', contrast({'l_pose_map_s0': 1}, {'l_reg_pose_map_do0.2_s0': 1})),
        ('pose seed 0: weight decay 1e-4', contrast({'l_pose_map_s0': 1}, {'l_reg_pose_map_wd1e-4_s0': 1})),
    ]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--cluster', choices=['log', 'session'], default='log')
    args = ap.parse_args()
    runs = Runs(PER_SCENE, '')
    mapping = token_to_log(PER_SCENE)
    rows = []
    for label, weights in comparisons():
        data = {k: runs[k] for k in weights}
        tokens = sorted(mapping)
        for name, scores in data.items():
            if set(scores) != set(tokens):
                raise ValueError('incomplete or extra tokens: ' + name)
        values = [sum(weights[k]*data[k][t]['score'] for k in weights) for t in tokens]
        groups = [mapping[t] if args.cluster == 'log' else mapping[t].rsplit('_', 2)[0] for t in tokens]
        mean, lo, hi = cluster_ci(values, groups, args.draws)
        rows.append(dict(comparison=label, n=len(tokens), logs=len(set(groups)),
                         delta=mean, low=lo, high=hi, weights=weights))
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    result = dict(draws=args.draws, seed=12345, cluster=args.cluster, estimand='scene-weighted mean', rows=rows)
    stem = 'clustered' if args.cluster == 'log' else 'session_clustered'
    (dest/(stem+'.json')).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = ['# '+args.cluster.capitalize()+'-cluster bootstrap on navtest', '',
             'Whole '+args.cluster+'s are resampled with replacement; the statistic remains scene-weighted.',
             'Learned-cell contrasts average paired scores over the three fixed training seeds.',
             'Intervals condition on those fitted models and do not estimate training-seed uncertainty.',
             'Exploratory contrasts, percentile 95% intervals; no multiplicity correction.', '',
             '| comparison (b - a) | scenes | '+args.cluster+'s | delta PDMS | 95% '+args.cluster+'-cluster CI |',
             '|---|---:|---:|---:|---|']
    for r in rows:
        lines.append('| {comparison} | {n} | {logs} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] |'.format(**r))
    (dest/(stem+'.md')).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
