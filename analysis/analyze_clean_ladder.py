# -*- coding: utf-8 -*-
"""Recompute the whole 9-agent ladder on the CLEAN held-out val logs only.

Why: the mini dataset has 64 logs and 62 of them are the warmup_test_e2e logs,
so an MLP trained on `mini` cannot be evaluated cleanly on the full 563 scenes —
428 of them were in its training set. Retraining on "mini minus warmup" is
impossible (only 2 logs would remain). The correct fix is therefore to score
*every* agent on the same held-out val logs, which makes the learned row
directly comparable to the eight non-learned ones.

Reads only existing per-scene CSVs + the metric cache layout. Nothing is re-run.
Writes exp/analysis/clean_ladder.md.
"""
import csv
import glob
import io
import os
import random
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, 'exp')
LOGS_YAML = os.path.join(ROOT, 'navsim', 'navsim', 'planning', 'script',
                         'config', 'training', 'available_mini_logs.yaml')
OUT = os.path.join(EXP, 'analysis', 'clean_ladder.md')
B = 10000
SEED = 12345

AGENTS = [
    ('CV', 'cv_agent_mini', 'speed only', 'yes'),
    ('Kinematic', 'kinematic_agent_mini', 'v, a, command', 'yes'),
    ('PrivBrake', 'privileged_brake_mini', 'GT boxes, no map', 'no'),
    ('MLP (learned)', 'mlp_agent_mini', 'learned kinematics, blind', 'yes'),
    ('MapMLP (learned+map)', 'map_mlp_mini', 'learned, sees centerline', 'no'),
    ('MapMLP-reg (learned+map)', 'map_mlp_reg_mini', 'learned, sees centerline, regularized', 'no'),
    ('PrivMap(IDM)', 'privileged_centerline_mini', 'map centerline + IDM', 'no'),
    ('PrivMapKin', 'privileged_centerline_kin', 'map centerline + kinematic', 'no'),
    ('SpeedMLP (hand path+learned speed)', 'speed_mlp_mini', 'map centerline + LEARNED arc-length', 'no'),
    ('SpeedMLP-e200', 'speed_mlp_e200_mini', 'same, 200 epochs', 'no'),
    ('PrivGTPathKin', 'privileged_gtpath_kin', 'logged path + kinematic', 'no'),
    ('PrivMapGTSpd', 'privileged_centerline_gtspeed', 'map centerline + human speed', 'no'),
    ('Human', 'human_agent_mini', 'logged future', 'no'),
]


def split_logs():
    train, val, sec = set(), set(), None
    for raw in io.open(LOGS_YAML, encoding='utf-8').read().splitlines():
        line = raw.strip()
        if line.startswith('train_logs:'):
            sec = 't'
            continue
        if line.startswith('val_logs:'):
            sec = 'v'
            continue
        if line.startswith('- ') and sec:
            name = line[2:].strip().strip('"').strip("'")
            (train if sec == 't' else val).add(name)
    return train, val


def token2log():
    root = os.path.join(EXP, 'metric_cache')
    m = {}
    for log in os.listdir(root):
        p = os.path.join(root, log, 'unknown')
        if os.path.isdir(p):
            for tok in os.listdir(p):
                m[tok] = log
    return m


def load(subdir):
    files = sorted(glob.glob(os.path.join(EXP, subdir, '*', '*.csv')))
    rows = {}
    with io.open(files[0], encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tok = (r.get('token') or '').strip()
            if not tok or tok.lower() == 'average':
                continue
            try:
                rows[tok] = {'score': float(r['score']),
                             'dac': float(r['drivable_area_compliance'])}
            except (ValueError, KeyError):
                continue
    return rows


def boot(diffs, seed=SEED):
    n = len(diffs)
    mean = sum(diffs) / n
    rnd = random.Random(seed)
    ms = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rnd.randrange(n)]
        ms.append(s / n)
    ms.sort()
    return mean, ms[int(0.025 * B)], ms[int(0.975 * B)]


def main():
    train_logs, val_logs = split_logs()
    t2l = token2log()
    data = {name: load(sub) for name, sub, _, _ in AGENTS}

    common = set(data[AGENTS[0][0]])
    for name, _, _, _ in AGENTS[1:]:
        common &= set(data[name])

    val_tok = sorted(t for t in common if t2l.get(t) in val_logs)
    tr_tok = sorted(t for t in common if t2l.get(t) in train_logs)

    L = []
    L.append('# Clean ladder — held-out val logs only\n')
    L.append('The mini dataset has 64 logs; **62 of them are the warmup_test_e2e logs**. '
             'Training on `mini` therefore contaminates the full 563-scene evaluation for the '
             'learned agents, and retraining on "mini minus warmup" is not possible — only '
             '2 logs would remain. So instead of dropping the learned row, every agent is '
             'scored on the same held-out val logs.\n')
    L.append('- train logs %d / val logs %d (`available_mini_logs.yaml`)' % (len(train_logs), len(val_logs)))
    L.append('- scenes: **%d held-out val** vs %d train-overlap (%d total)\n'
             % (len(val_tok), len(tr_tok), len(common)))

    L.append('\n## Ladder on the held-out val scenes (n=%d)\n' % len(val_tok))
    L.append('| agent | sees | deployable | PDMS (val) | PDMS (train-overlap) | gap |')
    L.append('|---|---|---|---:|---:|---:|')
    for name, _, sees, dep in AGENTS:
        d = data[name]
        v = sum(d[t]['score'] for t in val_tok) / len(val_tok)
        o = sum(d[t]['score'] for t in tr_tok) / len(tr_tok)
        L.append('| %s | %s | %s | **%.3f** | %.3f | %+.3f |'
                 % (name, sees, dep, v, o, o - v))

    L.append('\nOnly the learned row should show a systematic train-overlap advantage; '
             'the eight non-learned agents have no training set, so their gap is scene '
             'difficulty, not leakage.\n')

    L.append('\n## Paired deltas on the clean val scenes\n')
    L.append('| comparison | dPDMS | 95% CI | verdict |')
    L.append('|---|---:|---|---|')
    PAIRS = [
        ('CV', 'Kinematic', 'Add kinematics'),
        ('Kinematic', 'MLP (learned)', 'Learned blind planner vs hand rule'),
        ('Kinematic', 'PrivBrake', 'Add GT boxes + brake'),
        ('MLP (learned)', 'PrivMapKin', 'Add on-route map centerline'),
        ('PrivMap(IDM)', 'PrivMapKin', 'Drop IDM on the centerline'),
        ('PrivMapKin', 'PrivMapGTSpd', 'Swap in human speed (geometry fixed)'),
        ('PrivMapKin', 'PrivGTPathKin', 'Swap in logged path (speed fixed)'),
        ('PrivMapGTSpd', 'Human', 'Remaining gap to Human'),
        ('MLP (learned)', 'MapMLP (learned+map)', 'Give the LEARNED model the centerline too'),
        ('MapMLP (learned+map)', 'PrivMapKin', 'Same centerline: hand rule vs learned'),
        ('MapMLP (learned+map)', 'MapMLP-reg (learned+map)', 'Regularize the learned map model'),
        ('MapMLP-reg (learned+map)', 'PrivMapKin', 'Regularized learned vs hand rule (same centerline)'),
        ('PrivMapKin', 'SpeedMLP (hand path+learned speed)', 'Hand path: LEARNED speed vs hand kinematic speed'),
        ('SpeedMLP (hand path+learned speed)', 'PrivMapGTSpd', 'Learned speed vs human speed (same path, upper bound)'),
        ('SpeedMLP (hand path+learned speed)', 'SpeedMLP-e200', 'Speed model: 80 vs 200 epochs'),
    ]
    for a, b, label in PAIRS:
        diffs = [data[b][t]['score'] - data[a][t]['score'] for t in val_tok]
        m, lo, hi = boot(diffs)
        L.append('| %s | %+.4f | [%+.4f, %+.4f] | %s |'
                 % (label, m, lo, hi, 'CI excludes 0' if (lo > 0 or hi < 0) else 'CI spans 0'))

    base, spd, geo = data['PrivMapKin'], data['PrivMapGTSpd'], data['PrivGTPathKin']
    dd = [(spd[t]['score'] - base[t]['score']) - (geo[t]['score'] - base[t]['score'])
          for t in val_tok]
    m, lo, hi = boot(dd, seed=7)
    L.append('\n## Speed vs geometry on the clean val scenes\n')
    L.append('- %+.4f, 95%% CI [%+.4f, %+.4f] — %s\n'
             % (m, lo, hi, 'speed worth more' if lo > 0
                else ('geometry worth more' if hi < 0 else 'not separable at n=%d' % len(val_tok))))

    L.append('\n## Caveats\n')
    L.append('- n=%d is small; CIs are correspondingly wide. Effects that were significant '
             'on 563 scenes may not separate here — that is the honest cost of removing '
             'the contaminated scenes, not a defect of the analysis.' % len(val_tok))
    L.append('- Priv* and Human are upper bounds, not deployable systems.')
    L.append('- `privileged_brake_mini` has two runs (0.602 / 0.593 on the full split); '
             'the first is used everywhere for consistency and the discrepancy is unexplained.')
    L.append('- `warmup_test_e2e` is not the official navtest leaderboard.')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('wrote', OUT)
    print('val scenes = %d, train-overlap = %d' % (len(val_tok), len(tr_tok)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
