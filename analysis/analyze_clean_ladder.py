# -*- coding: utf-8 -*-
"""Recompute the whole ladder on the CLEAN held-out val logs only.

Why: the mini dataset has 64 logs and 62 of them are the warmup_test_e2e logs,
so an MLP trained on `mini` cannot be evaluated cleanly on the full 563 scenes —
428 of them were in its training set. Retraining on "mini minus warmup" is
impossible (only 2 logs would remain). The correct fix is therefore to score
*every* agent on the same held-out val logs, which makes the learned rows
directly comparable to the non-learned ones.

Reads only existing per-scene CSVs + the token-to-log map. Nothing is re-run.
Runs from the NAVSIM workspace or from results/per_scene/ (ladder_io).

Usage:  python analysis/analyze_clean_ladder.py [--csv-root DIR] [--out FILE]
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_io import (REPO, bootstrap, is_per_scene, load_scores,  # noqa: E402
                       resolve_csv_root, split_logs, token_to_log)

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


def render(root):
    train_logs, val_logs = split_logs(root)
    t2l = token_to_log(root)
    data = {name: load_scores(root, sub) for name, sub, _, _ in AGENTS}

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
    for a, b, label in PAIRS:
        diffs = [data[b][t]['score'] - data[a][t]['score'] for t in val_tok]
        m, lo, hi = bootstrap(diffs)
        L.append('| %s | %+.4f | [%+.4f, %+.4f] | %s |'
                 % (label, m, lo, hi, 'CI excludes 0' if (lo > 0 or hi < 0) else 'CI spans 0'))

    base, spd, geo = data['PrivMapKin'], data['PrivMapGTSpd'], data['PrivGTPathKin']
    dd = [(spd[t]['score'] - base[t]['score']) - (geo[t]['score'] - base[t]['score'])
          for t in val_tok]
    m, lo, hi = bootstrap(dd, seed=7)
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
    return '\n'.join(L) + '\n', len(val_tok), len(tr_tok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv-root', help='workspace exp/ or results/per_scene (auto-detected)')
    ap.add_argument('--out', help='report path')
    a = ap.parse_args(argv)
    root = resolve_csv_root(a.csv_root)
    out = a.out or (os.path.join(REPO, 'results', 'clean_ladder.md') if is_per_scene(root)
                    else os.path.join(root, 'analysis', 'clean_ladder.md'))
    text, n_val, n_tr = render(root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    io.open(out, 'w', encoding='utf-8', newline='\n').write(text)
    print('wrote', out)
    print('val scenes = %d, train-overlap = %d' % (n_val, n_tr))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
