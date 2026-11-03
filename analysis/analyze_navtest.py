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

# path cell, speed cell, the runs averaged into that cell. README quotes the mean column
# of path_speed.md as a single PDMS value per row, minus the "learned, map + boxes" row.
PATH_SPEED = [
    ('rule (map centerline)', 'rule (kinematic)', ['privmapkin']),
    ('rule (map centerline)', 'learned, blind', ['l_ds_kin_s%d' % s for s in SEEDS]),
    ('rule (map centerline)', 'learned, sees map', ['l_ds_map_s%d' % s for s in SEEDS]),
    ('rule (map centerline)', 'learned, map + boxes', ['l_ds_map_agents_s%d' % s for s in SEEDS]),
    ('learned (sees map)', 'rule (kinematic)', ['l_pose_map_s%d_rs' % s for s in SEEDS]),
    ('learned (sees map)', 'learned', ['l_pose_map_s%d' % s for s in SEEDS]),
    ('rule (map centerline)', 'human (upper bound)', ['privmap_gtspd']),
]

# label, run stem. The first entry is the baseline every delta is taken against.
REG_VARIANTS = [
    ('none (baseline)', 'l_pose_map_s0'),
    ('wd 1e-4', 'l_reg_pose_map_wd1e-4_s0'),
    ('wd 1e-3', 'l_reg_pose_map_wd1e-3_s0'),
    ('wd 1e-2', 'l_reg_pose_map_wd1e-2_s0'),
    ('dropout 0.2', 'l_reg_pose_map_do0.2_s0'),
    ('dropout 0.2 + wd 1e-3', 'l_reg_pose_map_do0.2_wd1e-3_s0'),
    ('hidden 128', 'l_reg_pose_map_h128_s0'),
]

# label, base stem of the 2x2 cell (one run per seed).
FACTORIAL = [('kin', 'pose_kin'), ('kin+agents', 'pose_agents'),
             ('kin+map', 'pose_map'), ('kin+map+agents', 'pose_map_agents')]


class Runs(object):
    """Lazily loaded per-scene rows, keyed by committed stem."""

    def __init__(self, root, suffix):
        self.root, self.suffix, self._c, self._d = root, suffix, {}, {}

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
        """(n, mean, lo, hi) of b - a over the scenes both scored. Memoised: the bootstrap
        seed is fixed, so a repeat call would give the same interval anyway."""
        key = (a, b, col)
        if key not in self._d:
            da, db = self[a], self[b]
            common = sorted(set(da) & set(db))
            diffs = [db[t][col] - da[t][col] for t in common]
            mean, lo, hi = bootstrap_large(diffs)
            self._d[key] = (len(common), mean, lo, hi)
        return self._d[key]

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
    cells = FACTORIAL
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

    for path, speed, stems in PATH_SPEED:
        L.append(line(path, speed, stems))
    L += ['', '## Paired deltas, per seed', ''] + DELTA_HEAD
    for s in SEEDS:
        L.append(row_delta(R, 'privmapkin', 'l_ds_map_s%d' % s, 'rule speed -> learned speed, rule path, seed %d' % s))
        L.append(row_delta(R, 'l_ds_map_s%d' % s, 'privmap_gtspd', 'learned speed -> human speed, rule path, seed %d' % s))
        L.append(row_delta(R, 'privmapkin', 'l_pose_map_s%d_rs' % s, 'rule path -> learned path, rule speed, seed %d' % s))
        L.append(row_delta(R, 'l_ds_map_s%d' % s, 'l_pose_map_s%d' % s, 'rule path -> learned path, learned speed, seed %d' % s))
        L.append(row_delta(R, 'l_ds_map_s%d' % s, 'l_ds_map_agents_s%d' % s, 'learned speed also sees boxes, seed %d' % s))
    # The closing paragraph is counted from the deltas above, not asserted: for each swap,
    # how many seeds land significantly better and how many significantly worse.
    def tally(pairs):
        ds = [R.delta(a, b) for a, b in pairs]
        return (sum(1 for _n, _m, lo, _hi in ds if lo > 0),
                sum(1 for _n, _m, _lo, hi in ds if hi < 0))

    n = len(SEEDS)
    sp_up, sp_dn = tally([('privmapkin', 'l_ds_map_s%d' % s) for s in SEEDS])
    pl_up, pl_dn = tally([('l_ds_map_s%d' % s, 'l_pose_map_s%d' % s) for s in SEEDS])
    pr_up, pr_dn = tally([('privmapkin', 'l_pose_map_s%d_rs' % s) for s in SEEDS])
    L += ['', 'Counting the seeds whose interval excludes zero. Rule speed -> learned speed, on the rule',
          'path: %d of %d seeds better, %d worse. Rule path -> learned path at the learned speed: %d better,'
          % (sp_up, n, sp_dn, pl_up),
          '%d worse. The same path swap at the rule speed -- the only comparison in which nothing but the'
          % pl_dn,
          'path changes -- is %d of %d better, %d worse. The speed effect reproduces across seeds; the sign'
          % (pr_up, n, pr_dn),
          'of the path effect does not, so the path numbers are per-seed outcomes rather than an effect.']
    return '\n'.join(L) + '\n'


def regularisation_md(R):
    variants = REG_VARIANTS
    L = ['# Regularising the map-conditioned model (navtest, seed 0)', '',
         'Same architecture and data; only the regularisation changes. The checkpoint is always the',
         'last epoch, so the open-loop validation L1 below (60 held-out navtrain logs) is the number a',
         'loss-based model selection would actually see -- it is not itself a selection criterion here.',
         '',
         '| regularisation | val L1 (navtrain) | dL1 vs baseline | PDMS | dPDMS vs baseline | 95% CI | verdict |',
         '|---|---:|---:|---:|---:|---|---|']
    val_l1 = load_cells()
    base = variants[0][1]
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


def md_section(text, title):
    """The markdown section starting at ``title``, or None if there is no such heading."""
    start = text.find(title)
    if start < 0:
        return None
    end = text.find('\n## ', start + 1)
    return text[start:end if end > 0 else None]


def md_num(cell):
    """A numeric README cell as a float. README prose uses U+2212, the reports use '-'."""
    return float(cell.replace(u'−', '-'))


def _cmp(problems, what, quoted, got, tol=0.0006):
    # Report the ASCII form: the message is printed to whatever console runs the check.
    if abs(md_num(quoted) - got) > tol:
        problems.append('%s: README %s vs CSV %+.4f' % (what, quoted.replace(u'−', '-'), got))


def _check_path_speed(R, text, problems):
    """README's condensed path/speed table against the mean column of path_speed.md."""
    section = md_section(text, '## Who draws the path, who paces it')
    if section is None:
        problems.append('README has no "## Who draws the path, who paces it" section')
        return
    seen, quoted = 0, 0
    for path, speed, stems in PATH_SPEED:
        m = re.search(r'^\| ' + re.escape(path) + r' \| ' + re.escape(speed)
                      + r' \| \**(0\.\d{3})\** \|', section, re.M)
        seen += 1
        if not m:
            continue  # README shows a subset of the rows; the count check below catches drops
        quoted += 1
        _cmp(problems, 'path/speed "%s" + "%s"' % (path, speed), m.group(1),
             sum(R.mean(x) for x in stems) / len(stems))
    if seen != len(PATH_SPEED) or quoted < len(PATH_SPEED) - 1:
        problems.append('path/speed table: matched %d of the %d rows in path_speed.md'
                        % (quoted, len(PATH_SPEED)))


def _check_regularisation(R, text, problems):
    """README's regularisation table: val L1 and dL1 from cells.csv, PDMS/dPDMS/CI from the CSVs."""
    section = md_section(text, '## How far can you trust the open-loop loss?')
    if section is None:
        problems.append('README has no "## How far can you trust the open-loop loss?" section')
        return
    val_l1, base, seen = load_cells(), REG_VARIANTS[0][1], 0
    for label, stem in REG_VARIANTS:
        m = re.search(r'^\| ' + re.escape(label) + r' \| (\S+) \| (\S+) \| \**(0\.\d{3})\** \| ([^|]*)\|',
                      section, re.M)
        if not m:
            continue
        seen += 1
        l1, dl1, pdms, dpdms = m.group(1), m.group(2), m.group(3), m.group(4).strip().strip('*')
        _cmp(problems, '%s val L1' % label, l1, val_l1[stem], tol=0.00006)
        _cmp(problems, '%s PDMS' % label, pdms, R.mean(stem))
        if stem == base:
            continue
        _cmp(problems, '%s dL1' % label, dl1, val_l1[stem] - val_l1[base])
        _n, mean, lo, hi = R.delta(base, stem)
        g = re.match(r'([+−-][\d.]+) \[([+−-][\d.]+), ([+−-][\d.]+)\]$', dpdms)
        if not g:
            problems.append('%s: cannot parse the dPDMS cell %r'
                            % (label, dpdms.replace(u'−', '-')))
            continue
        for name, cell, got in (('dPDMS', g.group(1), mean), ('CI low', g.group(2), lo),
                                ('CI high', g.group(3), hi)):
            _cmp(problems, '%s %s' % (label, name), cell, got)
    if seen != len(REG_VARIANTS):
        problems.append('regularisation table: found %d of %d rows' % (seen, len(REG_VARIANTS)))


def _check_train_samples(text, problems):
    """The training-set size README quotes, against n_train in cells.csv."""
    m = re.search(r'With ([\d,]+) training samples', text)
    if not m:
        problems.append('README does not state the training-sample count')
        return
    with io.open(os.path.join(OUT, 'cells.csv'), encoding='utf-8') as f:
        n_train = {r['cell']: int(r['n_train']) for r in csv.DictReader(f)}
    want = n_train['pose_map_s0']
    if int(m.group(1).replace(',', '')) != want:
        problems.append('training samples: README %s vs cells.csv n_train %d' % (m.group(1), want))


# ---------------------------------------------------------------- README prose
#
# Every signed number the prose of README's navtest sections quotes -- intervals, per-seed
# lists, sub-score deltas -- is matched by one rule below and recomputed from the per-scene
# CSVs. A signed number that no rule covers is itself a mismatch, so a claim cannot be added
# or reworded past the check. Signs are compared exactly as README writes them: every delta
# is b - a, as in the reports.

SN = r'([+-]\d*\.\d+)'
IV = SN + r' \[' + SN + r', ' + SN + r'\]'
PROSE_SECTIONS = ('## The ladder on navtest', '## What a learned trajectory model is allowed to see',
                  '## Who draws the path, who paces it', '## How far can you trust the open-loop loss?',
                  '## What changed when the split changed')


def _prose(text):
    """README's navtest sections as one line with ASCII minus signs. Tables are left out
    (the table checks above cover them), except the split-comparison table, which has no
    other check."""
    parts = []
    for title in PROSE_SECTIONS:
        sec = md_section(text, title)
        if sec is None:
            parts.append('[missing section %s]' % title)
            continue
        keep = [ln for ln in sec.split('\n')
                if title == PROSE_SECTIONS[-1] or not ln.lstrip().startswith('|')]
        parts.append(' '.join(keep))
    return re.sub(r'\s+', ' ', ' '.join(parts)).replace(u'−', '-').replace(u'—', '--')


class _Prose(object):
    def __init__(self, text, problems):
        self.text, self.problems, self.covered = text, problems, []

    def find(self, name, pattern):
        m = re.search(pattern, self.text)
        if not m:
            self.problems.append('README prose: cannot find %s' % name)
            return None
        self.covered += [m.span(i) for i in range(1, (m.lastindex or 0) + 1)]
        return m

    def num(self, what, cell, got):
        """``cell`` must be ``got`` rounded to the decimals README shows, sign included."""
        dp = len(cell.split('.')[1])
        want = ('%+.*f' if cell[0] in '+-' else '%.*f') % (dp, got)
        if want != cell:
            self.problems.append('%s: README %s vs CSV %+.4f' % (what, cell, got))

    def iv(self, what, m, first, d):
        _n, mean, lo, hi = d
        for k, (name, got) in enumerate((('', mean), (' CI low', lo), (' CI high', hi))):
            self.num(what + name, m.group(first + k), got)

    def verdicts(self, what, ds, want):
        got = ['excl' if (lo > 0 or hi < 0) else 'spans' for _n, _m, lo, hi in ds]
        if got != want:
            self.problems.append('%s: README says intervals %s, CSV gives %s' % (what, want, got))

    def uncovered(self):
        for m in re.finditer(SN, self.text):
            if not any(s <= m.start(1) and m.end(1) <= e for s, e in self.covered):
                self.problems.append('README prose: signed number %s is not checked ("...%s")'
                                     % (m.group(1), self.text[max(0, m.start() - 50):m.start()]))


def _check_prose(R, text, readme, problems):
    import analyze_openloop_csv as ol
    P = _Prose(_prose(text), problems)

    def seeds(fmt):
        return [fmt % s for s in SEEDS]

    # 1. map vs boxes
    m = P.find('claim 1, map gain', r'adding the on-route centerline is \*\*' + IV + r'\*\*')
    if m:
        P.iv('claim 1 map gain', m, 1, R.delta('kin', 'privmapkin'))
    m = P.find('claim 1, box gain', r'adding ground-truth detection boxes and a brake is \*\*' + IV + r'\*\*')
    if m:
        P.iv('claim 1 box gain', m, 1, R.delta('kin', 'privbrake'))
    m = P.find('claim 1, sub-scores', r'\(DAC ' + SN + r'\); the boxes buy collision avoidance \(NC ' + SN
               + r'\) and hand back progress \(EP ' + SN + r'\)')
    if m:
        P.num('claim 1 map dDAC', m.group(1), R.dsub('kin', 'privmapkin', 'drivable_area_compliance'))
        P.num('claim 1 boxes dNC', m.group(2), R.dsub('kin', 'privbrake', 'no_at_fault_collisions'))
        P.num('claim 1 boxes dEP', m.group(3), R.dsub('kin', 'privbrake', 'ego_progress'))
    if P.find('claim 1, ratio', r'about (seven) times'):
        ratio = R.delta('kin', 'privmapkin')[1] / R.delta('kin', 'privbrake')[1]
        if round(ratio) != 7:
            problems.append('claim 1: "about seven times" but the ratio is %.2f' % ratio)

    # 2. speed vs path headroom, as scored and with comfort neutralised
    m = P.find('claim 2, human speed', r'swapping the rule speed law for the human speed profile is \*\*'
               + IV + r'\*\*')
    if m:
        P.iv('claim 2 human speed', m, 1, R.delta('privmapkin', 'privmap_gtspd'))
    m = P.find('claim 2, logged path', r'swapping the centerline for the logged path is \*\*' + IV + r'\*\*')
    if m:
        P.iv('claim 2 logged path', m, 1, R.delta('privmapkin', 'privgtpath_kin'))
    # 3. the 2x2
    m = P.find('claim 3, map effect range', r'large in every seed \(' + SN + ' to ' + SN + r'\)')
    if m:
        ds = [R.delta(a, b) for a, b in zip(seeds('l_pose_kin_s%d'), seeds('l_pose_map_s%d'))]
        P.num('claim 3 smallest map effect', m.group(1), min(d[1] for d in ds))
        P.num('claim 3 largest map effect', m.group(2), max(d[1] for d in ds))
        P.verdicts('claim 3 map effect', ds, ['excl'] * 3)
    if P.find('claim 3, boxes without map', r'box effect on its own is not significant in (two) seeds of three'):
        ds = [R.delta(a, b) for a, b in zip(seeds('l_pose_kin_s%d'), seeds('l_pose_agents_s%d'))]
        n = sum(1 for _n, _m, lo, hi in ds if not (lo > 0 or hi < 0))
        if n != 2:
            problems.append('claim 3: boxes without the map span zero in %d seeds, README says two' % n)
    m = P.find('claim 3, boxes with map', r'\*\*negative in all three\*\*: ' + SN + ', ' + SN + ', ' + SN
               + ', every interval excluding zero')
    if m:
        ds = [R.delta(a, b) for a, b in zip(seeds('l_pose_map_s%d'), seeds('l_pose_map_agents_s%d'))]
        for k, d in enumerate(ds):
            P.num('claim 3 boxes with map, seed %d' % k, m.group(k + 1), d[1])
        P.verdicts('claim 3 boxes with map', ds, ['excl'] * 3)

    # 5. who draws the path, who paces it
    speed = [R.delta('privmapkin', b) for b in seeds('l_ds_map_s%d')]
    m = P.find('claim 5, learned speed', r'beats the hand speed law in every seed \(' + SN + ', ' + SN + ', '
               + SN + r', all three intervals excluding zero\)')
    if m:
        for k, d in enumerate(speed):
            P.num('claim 5 learned speed, seed %d' % k, m.group(k + 1), d[1])
        P.verdicts('claim 5 learned speed', speed, ['excl'] * 3)
    if P.find('claim 5, share of the human-speed gap', r'recovers about (two thirds) of the distance'):
        share = sum(d[1] for d in speed) / len(speed) / R.delta('privmapkin', 'privmap_gtspd')[1]
        if round(3 * share) != 2:
            problems.append('claim 5: "about two thirds" but the learned speed recovers %.2f' % share)
    m = P.find('claim 5, path swap at the learned speed',
               r'the swap is ' + SN + r' \(interval spans zero\), ' + SN + ' and ' + SN
               + r' -- the last two intervals both exclude zero and point opposite ways')
    if m:
        ds = [R.delta(a, b) for a, b in zip(seeds('l_ds_map_s%d'), seeds('l_pose_map_s%d'))]
        for k, d in enumerate(ds):
            P.num('claim 5 path swap at the learned speed, seed %d' % k, m.group(k + 1), d[1])
        P.verdicts('claim 5 path swap at the learned speed', ds, ['spans', 'excl', 'excl'])
        if not ds[1][1] * ds[2][1] < 0:
            problems.append('claim 5: the two significant path swaps do not point opposite ways')
    m = P.find('claim 5, path swap at the rule speed',
               r'it is ' + SN + r' \(spans zero\), ' + SN + ' and ' + SN + r': two of three significantly '
               r"positive, three-seed mean (0\.\d{3}) against the rule path's (0\.\d{3})")
    if m:
        ds = [R.delta('privmapkin', b) for b in seeds('l_pose_map_s%d_rs')]
        for k, d in enumerate(ds):
            P.num('claim 5 path swap at the rule speed, seed %d' % k, m.group(k + 1), d[1])
        P.verdicts('claim 5 path swap at the rule speed', ds, ['spans', 'excl', 'excl'])
        if not all(d[1] > 0 for d in ds[1:]):
            problems.append('claim 5: the significant rule-speed path swaps are not positive')
        P.num('claim 5 learned path, rule speed', m.group(4),
              sum(R.mean(x) for x in seeds('l_pose_map_s%d_rs')) / len(SEEDS))
        P.num('claim 5 rule path, rule speed', m.group(5), R.mean('privmapkin'))

    # 6. regularisation, and the rank correlations
    m = P.find('claim 6, opposite moves', r'move the score ' + SN + ' and ' + SN
               + ', both intervals excluding zero')
    if m:
        ds = [R.delta('l_pose_map_s0', b) for b in ('l_reg_pose_map_wd1e-4_s0', 'l_reg_pose_map_do0.2_s0')]
        P.num('claim 6 wd 1e-4', m.group(1), ds[0][1])
        P.num('claim 6 dropout 0.2', m.group(2), ds[1][1])
        P.verdicts('claim 6 regularisers', ds, ['excl', 'excl'])
    # The three correlations are recomputed from openloop.csv by analyze_openloop_csv; this only
    # marks them as covered, and folds that script's verdict in so one check covers the prose.
    for name, pattern in (('claim 6, nine-run correlation', r'displacement error and PDMS is ' + SN),
                          ('claim 6, drop-one correlation', r'Drop the worst run and the correlation is ' + SN),
                          ('claim 6, seven-run correlation', r'the seven that remain give ' + SN)):
        P.find(name, pattern)
    problems.extend('open-loop: ' + p for p in ol.check_readme(ol.load_runs(), readme))

    # What changed when the split changed (navtest column; the 563-scene column is make_summary's)
    m = P.find('split table, map vs boxes', r'held, ' + SN + ' vs ' + SN + ', intervals tight')
    if m:
        P.num('split table map gain', m.group(1), R.delta('kin', 'privmapkin')[1])
        P.num('split table box gain', m.group(2), R.delta('kin', 'privbrake')[1])
    m = P.find('split table, learned speed', r'\| ' + SN + ' to ' + SN + r', significant in all seeds \|')
    if m:
        P.num('split table smallest learned-speed gain', m.group(1), min(d[1] for d in speed))
        P.num('split table largest learned-speed gain', m.group(2), max(d[1] for d in speed))
        P.verdicts('split table learned speed', speed, ['excl'] * 3)
    P.find('split table, 563-scene learned speed', r'\| ' + SN + r', interval spanned zero \|')
    P.uncovered()


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
    # The 2x2 table quotes two columns: the mean open-loop val L1 (cells.csv) and the PDMS.
    fac = md_section(text, '## What a learned trajectory model is allowed to see')
    if fac is None:
        problems.append('README has no "## What a learned trajectory model is allowed to see" section')
        fac = ''
    val_l1 = load_cells()
    for label, base in FACTORIAL:
        m = re.search(r'^\| ' + re.escape(label) + r' \| \**(0\.\d{3})\** \| \**(0\.\d{3})\** \|',
                      fac, re.M)
        if not m:
            problems.append('factorial row for %s not found in README' % label)
            continue
        runs = ['l_%s_s%d' % (base, s) for s in SEEDS]
        _cmp(problems, '%s open-loop L1' % label, m.group(1),
             sum(val_l1[x] for x in runs) / len(runs))
        _cmp(problems, '%s PDMS' % label, m.group(2),
             sum(R.mean(x) for x in runs) / len(runs))
    _check_path_speed(R, text, problems)
    _check_regularisation(R, text, problems)
    _check_train_samples(text, problems)
    _check_prose(R, text, readme, problems)
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
