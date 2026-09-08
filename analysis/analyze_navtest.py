# -*- coding: utf-8 -*-
"""Regenerate every table under ``results/navtest/`` from the committed per-scene CSVs.

    python analysis/analyze_navtest.py                 # rewrite results/navtest/*.md
    python analysis/analyze_navtest.py --stdout        # print instead
    python analysis/analyze_navtest.py --check README.md   # exit 1 on any mismatch

Reads ``results/navtest/per_scene/*.csv.gz`` (or a NAVSIM workspace ``exp/`` via
``--csv-root``, where the same runs live in ``<stem>_navtest/<run>/<ts>.csv``).
Standard library only; the bootstrap seed is fixed, so a rerun reproduces every
interval to the last digit.

Tables
  ladder.md          the rule / official ladder, sub-scores and failure rates
  factorial.md       2x2, what a learned trajectory model is allowed to see
  path_speed.md      who draws the path and who paces it
  regularisation.md  open-loop regularisation vs the simulation score
"""
import argparse
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_io import REPO, bootstrap_large, load_rows  # noqa: E402

OUT = os.path.join(REPO, 'results', 'navtest')
PER_SCENE = os.path.join(OUT, 'per_scene')
SEEDS = (0, 1, 2)

SUB = [('score', 'PDMS'), ('no_at_fault_collisions', 'NC'), ('drivable_area_compliance', 'DAC'),
       ('time_to_collision_within_bound', 'TTC'), ('comfort', 'C'), ('ego_progress', 'EP'),
       ('driving_direction_compliance', 'DDC')]

# label, stem, what it sees, deployable
LADDER = [
    ('ConstantVelocity', 'cv', 'speed only', 'yes'),
    ('Kinematic rule', 'kin', 'v, a, driving command', 'yes'),
    ('PrivBrake rule', 'privbrake', '+ GT boxes', 'no'),
    ('EgoStatusMLP (official, seed 0)', 'egomlp_s0', 'learned kinematics, blind', 'yes'),
    ('EgoStatusMLP (official, seed 1)', 'egomlp_s1', 'learned kinematics, blind', 'yes'),
    ('EgoStatusMLP (official, seed 2)', 'egomlp_s2', 'learned kinematics, blind', 'yes'),
    ('PrivMap rule, IDM speed', 'privmap_idm', '+ GT map centerline', 'no'),
    ('PrivMapKin rule', 'privmapkin', '+ GT map centerline', 'no'),
    ('PrivGTPathKin', 'privgtpath_kin', 'logged path, rule speed', 'no'),
    ('PrivMapGTSpd', 'privmap_gtspd', 'GT map path, human speed', 'no'),
    ('Human', 'human', 'logged future', 'no'),
]


class Runs(object):
    """Lazily loaded per-scene rows, keyed by committed stem."""

    def __init__(self, root, suffix):
        self.root, self.suffix, self._c = root, suffix, {}

    def __getitem__(self, stem):
        if stem not in self._c:
            self._c[stem] = load_rows(self.root, stem + self.suffix, newest=bool(self.suffix))
        return self._c[stem]

    def mean(self, stem, col='score'):
        d = self[stem]
        return sum(r[col] for r in d.values()) / len(d)

    def fail(self, stem, col):
        d = self[stem]
        return sum(1 for r in d.values() if r[col] < 1) / len(d)

    def delta(self, a, b, col='score'):
        """(n, mean, lo, hi) of b - a over the scenes both scored."""
        da, db = self[a], self[b]
        common = sorted(set(da) & set(db))
        diffs = [db[t][col] - da[t][col] for t in common]
        mean, lo, hi = bootstrap_large(diffs)
        return len(common), mean, lo, hi

    def dsub(self, a, b, col):
        da, db = self[a], self[b]
        common = set(da) & set(db)
        return sum(db[t][col] - da[t][col] for t in common) / len(common)


def load_cells():
    """stem -> final open-loop validation L1, from results/navtest/cells.csv."""
    out = {}
    with io.open(os.path.join(OUT, 'cells.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out['l_' + r['cell']] = float(r['final_val_l1'])
    return out


def verdict(lo, hi):
    return 'CI excludes 0' if (lo > 0 or hi < 0) else 'CI spans 0'


def row_delta(R, a, b, label):
    n, m, lo, hi = R.delta(a, b)
    return '| %s | %d | %+.4f | [%+.4f, %+.4f] | %+.3f | %+.3f | %+.3f | %s |' % (
        label, n, m, lo, hi, R.dsub(a, b, 'drivable_area_compliance'),
        R.dsub(a, b, 'no_at_fault_collisions'), R.dsub(a, b, 'ego_progress'), verdict(lo, hi))


DELTA_HEAD = ['| comparison | n | dPDMS | 95% CI | dDAC | dNC | dEP | verdict |',
              '|---|---:|---:|---|---:|---:|---:|---|']


def ladder_md(R):
    L = ['# The ladder on navtest (12,146 scenes)', '',
         'Per-scene means of the official PDM sub-scores. `Priv*` and `Human` consume ground truth',
         'or map privilege: they are upper bounds, not deployable planners.', '',
         '| agent | what it sees | deployable | ' + ' | '.join(s for _, s in SUB) + ' | NC fail | DAC fail | TTC fail |',
         '|---|---|---|' + '---:|' * (len(SUB) + 3)]
    for label, stem, sees, dep in LADDER:
        vals = ' | '.join(('**%.3f**' if c == 'score' else '%.3f') % R.mean(stem, c) for c, _ in SUB)
        L.append('| %s | %s | %s | %s | %.1f%% | %.1f%% | %.1f%% |' % (
            label, sees, dep, vals, 100 * R.fail(stem, 'no_at_fault_collisions'),
            100 * R.fail(stem, 'drivable_area_compliance'),
            100 * R.fail(stem, 'time_to_collision_within_bound')))
    L += ['', '## Paired deltas', ''] + DELTA_HEAD
    for a, b, label in [
        ('cv', 'kin', 'add kinematics'),
        ('kin', 'privbrake', 'add GT boxes + brake'),
        ('kin', 'privmapkin', 'add GT map centerline'),
        ('privbrake', 'privmapkin', 'GT boxes -> GT map'),
        ('privmap_idm', 'privmapkin', 'drop IDM on the centerline'),
        ('privmapkin', 'privmap_gtspd', 'swap in human speed (path fixed)'),
        ('privmapkin', 'privgtpath_kin', 'swap in the logged path (speed fixed)'),
        ('privmap_gtspd', 'human', 'remaining gap to Human'),
        ('kin', 'egomlp_s0', 'hand kinematics -> learned kinematics (official, seed 0)'),
        ('egomlp_s0', 'privmapkin', 'learned blind -> hand rule with the map'),
    ]:
        L.append(row_delta(R, a, b, label))
    L += ['', 'The map is worth about seven times what ground-truth detection boxes are worth, and it',
          'buys drivable-area compliance; the boxes buy collision avoidance and give back progress.']
    return '\n'.join(L) + '\n'


def factorial_md(R):
    cells = [('kin', 'pose_kin'), ('kin+agents', 'pose_agents'),
             ('kin+map', 'pose_map'), ('kin+map+agents', 'pose_map_agents')]
    L = ['# 2x2: what a learned trajectory model is allowed to see (navtest)', '',
         'One MLP per cell regresses the eight future poses. Inputs are ego kinematics (8),',
         'optionally the on-route map centerline (20x4) and optionally the nearest eight dynamic',
         'GT boxes plus the lead gap (8x11 + 2). Three seeds per cell, trained on navtrain.', '',
         '| inputs | ' + ' | '.join('seed %d' % s for s in SEEDS) + ' | mean |',
         '|---|---:|---:|---:|---:|']
    for label, base in cells:
        v = [R.mean('l_%s_s%d' % (base, s)) for s in SEEDS]
        L.append('| %s | %s | **%.3f** |' % (label, ' | '.join('%.3f' % x for x in v), sum(v) / len(v)))
    L += ['', '## Main effects and the interaction, per seed', ''] + DELTA_HEAD
    for s in SEEDS:
        for a, b, label in [
            ('pose_kin', 'pose_map', 'add map (no boxes)'),
            ('pose_kin', 'pose_agents', 'add boxes (no map)'),
            ('pose_agents', 'pose_map_agents', 'add map (with boxes)'),
            ('pose_map', 'pose_map_agents', 'add boxes (with map)'),
        ]:
            L.append(row_delta(R, 'l_%s_s%d' % (a, s), 'l_%s_s%d' % (b, s), '%s, seed %d' % (label, s)))
    L += ['', 'The map effect is large and consistent. The box effect is not significant in two of the',
          'three seeds on its own, and is significantly **negative** in all three once the model already',
          'has the map -- while the open-loop validation loss barely moves. Ground-truth perception is',
          'not a free input: the capacity spent on it does not pay off under this score.']
    return '\n'.join(L) + '\n'


def path_speed_md(R):
    L = ['# Who draws the path, who paces it (navtest)', '',
         'The rule path is the on-route map centerline; the rule speed is the constant-acceleration',
         'kinematic law. `ds` models predict only progress along the rule path; `pose` models predict',
         'the whole trajectory, and the `_rs` variant keeps that geometry but re-times it with the rule',
         'speed. Human speed is the upper bound at fixed geometry.', '',
         '| path | speed | ' + ' | '.join('seed %d' % s for s in SEEDS) + ' | mean |',
         '|---|---|---:|---:|---:|---:|']

    def line(path, speed, stems):
        v = [R.mean(x) for x in stems]
        cells = ' | '.join('%.3f' % x for x in v)
        if len(v) == 1:
            cells = '%.3f | %s | %s' % (v[0], '--', '--')
        return '| %s | %s | %s | **%.3f** |' % (path, speed, cells, sum(v) / len(v))

    L.append(line('rule (map centerline)', 'rule (kinematic)', ['privmapkin']))
    L.append(line('rule (map centerline)', 'learned, blind', ['l_ds_kin_s%d' % s for s in SEEDS]))
    L.append(line('rule (map centerline)', 'learned, sees map', ['l_ds_map_s%d' % s for s in SEEDS]))
    L.append(line('rule (map centerline)', 'learned, map + boxes', ['l_ds_map_agents_s%d' % s for s in SEEDS]))
    L.append(line('learned (sees map)', 'rule (kinematic)', ['l_pose_map_s%d_rs' % s for s in SEEDS]))
    L.append(line('learned (sees map)', 'learned', ['l_pose_map_s%d' % s for s in SEEDS]))
    L.append(line('rule (map centerline)', 'human (upper bound)', ['privmap_gtspd']))
    L += ['', '## Paired deltas, per seed', ''] + DELTA_HEAD
    for s in SEEDS:
        L.append(row_delta(R, 'privmapkin', 'l_ds_map_s%d' % s, 'rule speed -> learned speed, rule path, seed %d' % s))
        L.append(row_delta(R, 'l_ds_map_s%d' % s, 'privmap_gtspd', 'learned speed -> human speed, rule path, seed %d' % s))
        L.append(row_delta(R, 'privmapkin', 'l_pose_map_s%d_rs' % s, 'rule path -> learned path, rule speed, seed %d' % s))
        L.append(row_delta(R, 'l_ds_map_s%d' % s, 'l_pose_map_s%d' % s, 'rule path -> learned path, learned speed, seed %d' % s))
        L.append(row_delta(R, 'l_ds_map_s%d' % s, 'l_ds_map_agents_s%d' % s, 'learned speed also sees boxes, seed %d' % s))
    L += ['', 'Given the centerline, the learned component earns its place in the speed profile and',
          'nowhere else: letting the model draw the line instead of pacing it is not separable from',
          'the hand rule, while pacing a hand-drawn line beats the hand speed law in every seed.']
    return '\n'.join(L) + '\n'


def regularisation_md(R):
    variants = [('none (baseline)', 'l_pose_map_s0'), ('wd 1e-4', 'l_reg_pose_map_wd1e-4_s0'),
                ('wd 1e-3', 'l_reg_pose_map_wd1e-3_s0'), ('wd 1e-2', 'l_reg_pose_map_wd1e-2_s0'),
                ('dropout 0.2', 'l_reg_pose_map_do0.2_s0'),
                ('dropout 0.2 + wd 1e-3', 'l_reg_pose_map_do0.2_wd1e-3_s0'),
                ('hidden 128', 'l_reg_pose_map_h128_s0')]
    L = ['# Regularising the map-conditioned model (navtest, seed 0)', '',
         'Same architecture and data; only the regularisation changes. The checkpoint is always the',
         'last epoch, so the open-loop validation L1 below (60 held-out navtrain logs) is the number a',
         'loss-based model selection would actually see -- it is not itself a selection criterion here.',
         '',
         '| regularisation | val L1 (navtrain) | dL1 vs baseline | PDMS | dPDMS vs baseline | 95% CI | verdict |',
         '|---|---:|---:|---:|---:|---|---|']
    val_l1 = load_cells()
    base = 'l_pose_map_s0'
    for label, stem in variants:
        if stem == base:
            L.append('| %s | %.4f | -- | **%.3f** | -- | -- | -- |' % (label, val_l1[stem], R.mean(stem)))
            continue
        _n, m, lo, hi = R.delta(base, stem)
        L.append('| %s | %.4f | %+.4f | **%.3f** | %+.4f | [%+.4f, %+.4f] | %s |' % (
            label, val_l1[stem], val_l1[stem] - val_l1[base], R.mean(stem), m, lo, hi, verdict(lo, hi)))
    L += ['', 'Two regularisers improve the open-loop fit by a comparable amount and move the simulation',
          'score in opposite directions: weight decay 1e-4 (dL1 -0.032) is worth +0.024 PDMS, dropout 0.2',
          '(dL1 -0.024) costs -0.012, and both intervals exclude zero. Heavy weight decay ruins both',
          'numbers, so the loss is not useless -- across these seven runs its rank correlation with PDMS',
          'is strong. It is among the models that are actually competitive that the ordering breaks:',
          'the unregularised baseline has the worst open-loop fit of the six that still work, and still',
          'outscores dropout 0.2. Treat the open-loop loss as a coarse filter, not a selection criterion.']
    return '\n'.join(L) + '\n'


import re  # noqa: E402


def check_readme(R, readme):
    """Verify the navtest numbers README quotes against the per-scene CSVs."""
    text = io.open(readme, encoding='utf-8').read()
    start = text.find('## The ladder on navtest')
    if start < 0:
        return ['README has no "## The ladder on navtest" section']
    end = text.find('\n## ', start + 1)
    section = text[start:end if end > 0 else None]
    problems, seen = [], 0
    for label, stem, _sees, _dep in LADDER:
        m = re.search(r'^\| ' + re.escape(label) + r' \|[^|]*\| \**(0\.\d{3})\** \|', section, re.M)
        if not m:
            continue
        seen += 1
        got = R.mean(stem)
        if abs(got - float(m.group(1))) > 0.0006:
            problems.append('%s: README %s vs CSV %.3f' % (label, m.group(1), got))
    if seen != len(LADDER):
        problems.append('navtest ladder: found %d of %d rows' % (seen, len(LADDER)))
    for label, stems in [('kin', ['pose_kin']), ('kin+agents', ['pose_agents']),
                         ('kin+map', ['pose_map']), ('kin+map+agents', ['pose_map_agents'])]:
        m = re.search(r'^\| ' + re.escape(label) + r' \|[^|]*\| \**(0\.\d{3})\** \|', text, re.M)
        if not m:
            problems.append('factorial row for %s not found in README' % label)
            continue
        got = sum(R.mean('l_%s_s%d' % (stems[0], s)) for s in SEEDS) / len(SEEDS)
        if abs(got - float(m.group(1))) > 0.0006:
            problems.append('%s: README %s vs CSV %.3f' % (label, m.group(1), got))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-root', default=PER_SCENE)
    ap.add_argument('--suffix', default='', help='"_navtest" when reading a workspace exp/')
    ap.add_argument('--stdout', action='store_true')
    ap.add_argument('--check', metavar='README', help='verify the navtest numbers quoted in this file')
    a = ap.parse_args()
    R = Runs(a.csv_root, a.suffix)
    if a.check:
        problems = check_readme(R, a.check)
        for p in problems:
            print('MISMATCH:', p)
        print('%s: navtest numbers match the per-scene CSVs' % a.check if not problems
              else '%d mismatch(es)' % len(problems))
        return 1 if problems else 0
    for name, fn in [('ladder.md', ladder_md), ('factorial.md', factorial_md),
                     ('path_speed.md', path_speed_md), ('regularisation.md', regularisation_md)]:
        text = fn(R)
        if a.stdout:
            print(text)
        else:
            io.open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='\n').write(text)
            print('wrote', os.path.join('results', 'navtest', name))
    return 0


if __name__ == '__main__':
    sys.exit(main())
