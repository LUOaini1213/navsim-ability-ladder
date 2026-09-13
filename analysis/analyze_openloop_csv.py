# -*- coding: utf-8 -*-
"""Recompute the open-loop-error-vs-simulation-score numbers from the committed CSV.

    python analysis/analyze_openloop_csv.py                    # -> results/navtest/openloop_summary.md
    python analysis/analyze_openloop_csv.py --stdout           # print instead
    python analysis/analyze_openloop_csv.py --check README.md  # exit 1 on any mismatch

``results/navtest/openloop.csv`` is one row per scoring run, written by the workspace-side
``analyze_openloop.py`` (which needs the devkit and the frozen features). Two of its columns
can be checked against the rest of the archive and are, in ``tests/test_navtest_reports.py``:
``PDMS`` is the per-scene mean of the matching file under ``per_scene/`` and
``val_l1_navtrain`` is ``final_val_l1`` in ``cells.csv``. The displacement-error columns come
from the training runs. This script does the arithmetic README quotes on top of that CSV --
rank correlations and the subset ranges -- so a plain clone can recompute them.

Spearman is Pearson on average ranks; standard library only, no scipy.
"""
import argparse
import csv
import io
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_io import REPO  # noqa: E402

OUT = os.path.join(REPO, 'results', 'navtest')
CSV_PATH = os.path.join(OUT, 'openloop.csv')
REPORT = 'openloop_summary.md'

# How far the "drop the weakest run" ladder goes. 9 map-conditioned runs down to 6.
MIN_KEPT = 6


def load_runs():
    with io.open(CSV_PATH, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def map_conditioned(rows):
    """The trajectory models that see the centerline and no boxes, on their own speed.

    This is the group README's conclusion 6 is about. Sorted by PDMS, best first, so
    "drop the weakest" is well defined.
    """
    g = [r for r in rows if r['target'] == 'pose' and r['see_map'] == 'True'
         and r['see_agents'] == 'False' and r['speed_mode'] == 'learned']
    return sorted(g, key=lambda r: (-float(r['PDMS']), r['run']))


def ranks(xs):
    """1-based ranks, ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a)) * math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / den


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def ade(r):
    return float(r['ade_navtest'])


def pdms(r):
    return float(r['PDMS'])


def corr(group):
    return spearman([ade(r) for r in group], [pdms(r) for r in group])


def ladder(group):
    """(kept, dropped run or None, ade lo/hi, pdms lo/hi, spearman) dropping the weakest."""
    out = []
    for k in range(len(group), MIN_KEPT - 1, -1):
        s = group[:k]
        out.append((k, None if k == len(group) else group[k]['run'],
                    min(ade(r) for r in s), max(ade(r) for r in s),
                    min(pdms(r) for r in s), max(pdms(r) for r in s), corr(s)))
    return out


def per_scene_extreme(rows):
    """The run whose per-scene ADE/PDMS rank correlation has the largest magnitude."""
    return max(rows, key=lambda r: abs(float(r['spearman_ade_pdms'])))


def report(rows):
    g = map_conditioned(rows)
    L = ['# Open-loop error against the simulation score (navtest)', '',
         'Recomputed from `openloop.csv` (%d scoring runs) with the standard library only.' % len(rows),
         'Spearman is Pearson on average ranks. The `PDMS` column of that CSV is the per-scene mean',
         'of the matching file under `per_scene/` and `val_l1_navtrain` is `final_val_l1` in',
         '`cells.csv`; the tests check both. The displacement errors come from the training runs.', '',
         '## The %d map-conditioned runs' % len(g), '',
         'Trajectory models that see the centerline and no GT boxes, scored on their own speed --',
         'the cluster the ordering question is about. Ranks are 1 = lowest value.', '',
         '| run | val L1 (navtrain) | ADE (navtest) | PDMS | ADE rank | PDMS rank |',
         '|---|---:|---:|---:|---:|---:|']
    ra, rp = ranks([ade(r) for r in g]), ranks([pdms(r) for r in g])
    for r, a, p in zip(g, ra, rp):
        L.append('| %s | %.4f | %.4f | %.4f | %g | %g |'
                 % (r['run'], float(r['val_l1_navtrain']), ade(r), pdms(r), a, p))
    L += ['', '## What the correlation is made of', '',
          'Dropping the lowest-scoring run one at a time, from all %d down to %d.' % (len(g), MIN_KEPT), '',
          '| runs kept | run dropped | ADE range | PDMS range | Spearman(ADE, PDMS) |',
          '|---:|---|---|---|---:|']
    for k, dropped, alo, ahi, plo, phi, s in ladder(g):
        L.append('| %d | %s | %.4f - %.4f | %.4f - %.4f | %+.4f |'
                 % (k, dropped or '--', alo, ahi, plo, phi, s))
    # Both claims below are decided from the numbers above, not asserted.
    seq = [s for _k, _d, _alo, _ahi, _plo, _phi, s in ladder(g)]
    weakens = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    base = next(r for r in g if r['run'] == 'pose_map_s0')
    do = next(r for r in g if r['run'] == 'reg_pose_map_do0.2_s0')
    inverts = ade(do) < ade(base) and pdms(do) < pdms(base)
    L += ['',
          'Down that ladder the correlation goes %s.' % ' -> '.join('%+.4f' % s for s in seq)]
    L += (['It weakens at every step, so the full-group figure is carried by the distance between the',
           'working models and the broken ones more than by any ordering among the runs worth shipping.']
          if weakens else
          ['It does not weaken at every step, so it is not simply a working-versus-broken effect.'])
    L += ['Dropout 0.2 against the unregularised baseline: ADE %.4f against %.4f, PDMS %.4f against'
          % (ade(do), ade(base), pdms(do)),
          '%.4f -- %s.' % (pdms(base),
                           'the better open-loop fit is the worse score' if inverts
                           else 'the two orderings agree here'),
          '', '## Within a single run', '',
          'Per-scene rank correlation between displacement error and PDMS, over all %d runs in the'
          % len(rows),
          'CSV. The five largest by magnitude:', '',
          '| run | per-scene Spearman(ADE, PDMS) |', '|---|---:|']
    top = sorted(rows, key=lambda r: -abs(float(r['spearman_ade_pdms'])))[:5]
    for r in top:
        L.append('| %s | %+.4f |' % (r['run'], float(r['spearman_ade_pdms'])))
    worst = per_scene_extreme(rows)
    L += ['',
          'Largest magnitude over all %d runs: %.4f (`%s`).'
          % (len(rows), abs(float(worst['spearman_ade_pdms'])), worst['run'])]
    return '\n'.join(L) + '\n'


def _dec(cell):
    """Decimal places quoted, so a 2dp figure is checked to 2dp and a 4dp one to 4dp."""
    return len(cell.split('.')[1])


def _num(cell):
    return float(cell.replace(u'−', '-'))


def _cmp(problems, what, cell, got):
    tol = 0.5 * 10 ** -_dec(cell) + 1e-9
    if abs(_num(cell) - got) > tol:
        problems.append('%s: README %s vs CSV %+.4f' % (what, cell.replace(u'−', '-'), got))


WORDS = {'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
NUM = r'[−+-]?\d+\.\d+'


def _count(cell):
    return WORDS.get(cell, None) if not cell.isdigit() else int(cell)


def check_readme(rows, readme):
    """Verify the open-loop numbers README quotes against openloop.csv."""
    text = re.sub(r'\s+', ' ', io.open(readme, encoding='utf-8').read())
    problems = []
    g = map_conditioned(rows)
    steps = {k: (alo, ahi, plo, phi, s) for k, _d, alo, ahi, plo, phi, s in ladder(g)}

    def find(name, pattern):
        m = re.search(pattern, text, re.I)
        if not m:
            problems.append('README does not state %s in the expected form' % name)
        return m

    m = find('the full-group rank correlation',
             r'over the (\w+) map-conditioned runs, rank correlation between navtest'
             r' displacement error and PDMS is (' + NUM + r')')
    if m:
        if _count(m.group(1)) != len(g):
            problems.append('map-conditioned group: README says %s, CSV has %d' % (m.group(1), len(g)))
        _cmp(problems, 'Spearman over all %d' % len(g), m.group(2), steps[len(g)][4])

    m = find('the one-run-dropped correlation',
             r'drop the worst run and the correlation is (' + NUM + r')')
    if m:
        _cmp(problems, 'Spearman over %d' % (len(g) - 1), m.group(1), steps[len(g) - 1][4])

    subset = len(g) - 2
    m = find('the competitive-subset correlation',
             r'the (\w+) that remain give (' + NUM + r')')
    if m:
        n = _count(m.group(1))
        if n not in steps:
            problems.append('competitive subset: README says %s runs, not on the drop ladder' % m.group(1))
        else:
            subset = n
            _cmp(problems, 'Spearman over %d' % n, m.group(2), steps[n][4])

    # the ranges belong to whichever subset the sentence above named
    m = find('the competitive-subset ranges',
             r'\(ADE (\d+\.\d+)[–-](\d+\.\d+), PDMS (\d+\.\d+)[–-](\d+\.\d+)\)')
    if m:
        alo, ahi, plo, phi, _s = steps[subset]
        _cmp(problems, 'competitive ADE low', m.group(1), alo)
        _cmp(problems, 'competitive ADE high', m.group(2), ahi)
        _cmp(problems, 'competitive PDMS low', m.group(3), plo)
        _cmp(problems, 'competitive PDMS high', m.group(4), phi)

    m = find('the dropout-0.2 inversion',
             r'dropout 0\.2 fits better than the unregularised baseline \(ADE (\d+\.\d+) vs'
             r' (\d+\.\d+)\) and scores worse \((\d+\.\d+) vs (\d+\.\d+)\)')
    if m:
        base = next(r for r in g if r['run'] == 'pose_map_s0')
        do = next(r for r in g if r['run'] == 'reg_pose_map_do0.2_s0')
        _cmp(problems, 'dropout 0.2 ADE', m.group(1), ade(do))
        _cmp(problems, 'baseline ADE', m.group(2), ade(base))
        _cmp(problems, 'dropout 0.2 PDMS', m.group(3), pdms(do))
        _cmp(problems, 'baseline PDMS', m.group(4), pdms(base))

    m = find('the per-scene correlation bound', r'never exceeds (\d+\.\d+) in magnitude')
    if m:
        got = abs(float(per_scene_extreme(rows)['spearman_ade_pdms']))
        d = _dec(m.group(1))
        tight = math.ceil(got * 10 ** d) / 10.0 ** d
        if abs(float(m.group(1)) - tight) > 1e-9:
            problems.append('per-scene bound: README %s, the tightest %d-dp bound on %.4f is %.*f'
                            % (m.group(1), d, got, d, tight))

    m = find('the number of runs in the figure', r'all (\d+) learned runs')
    if m and int(m.group(1)) != len(rows):
        problems.append('openloop figure: README says %s runs, the CSV has %d' % (m.group(1), len(rows)))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stdout', action='store_true')
    ap.add_argument('--check', metavar='README', help='verify the open-loop numbers quoted in this file')
    a = ap.parse_args()
    rows = load_runs()
    if a.check:
        problems = check_readme(rows, a.check)
        for p in problems:
            print('MISMATCH:', p)
        print('%s: open-loop numbers match openloop.csv' % a.check if not problems
              else '%d mismatch(es)' % len(problems))
        return 1 if problems else 0
    text = report(rows)
    if a.stdout:
        print(text)
    else:
        io.open(os.path.join(OUT, REPORT), 'w', encoding='utf-8', newline='\n').write(text)
        print('wrote', os.path.join('results', 'navtest', REPORT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
