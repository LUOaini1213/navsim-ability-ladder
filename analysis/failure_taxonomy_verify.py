# -*- coding: utf-8 -*-
"""Two checks on the committed failure taxonomy (standard library only).

    python analysis/failure_taxonomy_verify.py compare FRESH_DIR [--cases FRESH_CASES_DIR]
    python analysis/failure_taxonomy_verify.py readme README.md

compare  FRESH_DIR is the output of ``failure_taxonomy_analyze.py --out FRESH_DIR`` (and
         FRESH_CASES_DIR that of ``failure_taxonomy_cases.py select`` / ``report --no-plots``
         run on a copy of the scored cases). Every markdown and CSV file must equal the
         committed one byte for byte. The JSON files carry full-precision floats whose last
         digit depends on the numpy build's summation order, so they are compared value by
         value: same keys, same strings and integers, floats within 1e-9 (the wall-clock
         ``runtime_s`` is skipped).
readme   Recomputes every number README's "Where the planners fail" section and the
         failure-taxonomy caveats of claim 3 quote, from the committed outputs, and fails on a
         mismatch -- or on a signed number in that section no rule covers, so new prose cannot
         slip past the check.

Exit status 1 on any problem; each problem is printed.
"""
import argparse
import csv
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, 'results', 'failure_taxonomy')
CASES = os.path.join(OUT, 'cases')

# What failure_taxonomy_analyze.write_outputs writes, and what the cases script writes from
# the committed scored_*.jsonl without the devkit (select; report --no-plots).
ANALYZE_FILES = ['category_outcomes.csv', 'category_effects.csv', 'category_contributions.csv',
                 'where_planners_fail.md', 'headline_effects.md', 'hypotheses.md', 'clustering.md',
                 'city_sensitivity.md', 'hypotheses.json', 'city_sensitivity.json', 'summary.json']
CASES_FILES = ['sample.csv', 'sample_counts.json', 'cases.csv', 'root_causes.md']
FLOAT_TOL = 1e-9
# Wall-clock time of the run that wrote summary.json: not a result.
VOLATILE_KEYS = {'runtime_s'}


# ---------------------------------------------------------------- compare


def _json_diff(a, b, path, out):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            out.append('%s: %r vs %r' % (path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                out.append('%s: %r vs %r' % (path, a, b))
        elif abs(a - b) > FLOAT_TOL:
            out.append('%s: %r vs %r' % (path, a, b))
    elif isinstance(a, dict) and isinstance(b, dict):
        if sorted(a) != sorted(b):
            out.append('%s: keys differ' % path)
        for k in sorted((set(a) & set(b)) - VOLATILE_KEYS):
            _json_diff(a[k], b[k], '%s/%s' % (path, k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append('%s: length %d vs %d' % (path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            _json_diff(x, y, '%s/%d' % (path, i), out)
    elif a != b:
        out.append('%s: %r vs %r' % (path, a, b))


def compare_dir(fresh, committed, names):
    problems = []
    for name in names:
        f, c = os.path.join(fresh, name), os.path.join(committed, name)
        if not os.path.exists(f):
            problems.append('%s: not regenerated' % name)
            continue
        if name.endswith('.json'):
            with io.open(f, encoding='utf-8') as fa, io.open(c, encoding='utf-8') as fb:
                diffs = []
                _json_diff(json.load(fb), json.load(fa), name, diffs)
            problems += diffs[:5] + (['%s: %d more' % (name, len(diffs) - 5)] if len(diffs) > 5 else [])
        else:
            with open(f, 'rb') as fa, open(c, 'rb') as fb:
                if fa.read() != fb.read():
                    problems.append('%s: differs from the committed file' % name)
    return problems


# ---------------------------------------------------------------- readme


def _rows(name, base=OUT):
    with io.open(os.path.join(base, name), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _json(name, base=OUT):
    with io.open(os.path.join(base, name), encoding='utf-8') as f:
        return json.load(f)


def _num(s):
    return float(s.replace(',', ''))


def _excludes(lo, hi):
    return lo > 0 or hi < 0


class Sources(object):
    """The committed outputs, indexed the way README quotes them."""

    def __init__(self):
        self.eff = {(r['feature'], r['category'], r['series'], r['metric']): r
                    for r in _rows('category_effects.csv')}
        self.contrib = {(r['feature'], r['category'], r['effect']): float(r['contribution'])
                        for r in _rows('category_contributions.csv') if r['level'] == 'log'}
        self.summary = _json('summary.json')
        self.loco = {(r['without'], r['effect']): r for r in _json('city_sensitivity.json')}
        self.hyps = {(r['hypothesis'], r['effect'], r['level']): r for r in _json('hypotheses.json')}
        self.cases = _rows('cases.csv', CASES)
        with io.open(os.path.join(CASES, 'root_causes.md'), encoding='utf-8') as f:
            self.root_md = f.read()

    def mean(self, feature, category, series, metric='pdms'):
        return float(self.eff[(feature, category, series, metric)]['mean'])

    def overall(self, effect):
        return self.summary['overall'][effect]

    def share(self, feature, category, effect):
        return self.contrib[(feature, category, effect)] / self.overall(effect)['delta']

    def clustering(self):
        """(changed, total, median log width ratio, median session width ratio) over the
        category-level PDMS effect rows that carry an iid interval -- the same rows, verdicts
        ('n/a' for a category too small for a cluster interval) and upper median as
        clustering_md in failure_taxonomy_analyze.py, recomputed from category_effects.csv."""
        rows = [r for r in self.eff.values() if r['metric'] == 'pdms' and r['iid_low'] != '']
        changed, wl, ws = 0, [], []
        for r in rows:
            iv = {k: (float(r[k + '_low']), float(r[k + '_high'])) if r[k + '_low'] != '' else None
                  for k in ('iid', 'log', 'session')}
            verdicts = {_excludes(*v) if v is not None else None for v in iv.values()}
            changed += len(verdicts) > 1
            w0 = iv['iid'][1] - iv['iid'][0]
            if iv['log'] and iv['session'] and w0 > 0:
                wl.append((iv['log'][1] - iv['log'][0]) / w0)
                ws.append((iv['session'][1] - iv['session'][0]) / w0)
        return changed, len(rows), _upper_median(wl), _upper_median(ws)


def _median(v):
    v = sorted(v)
    n = len(v)
    return (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2) if n else float('nan')


def _upper_median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float('nan')


def _section(text, title):
    start = text.find(title)
    if start < 0:
        return None
    end = text.find('\n## ', start + 1)
    return text[start:end if end > 0 else None]


def _norm(text):
    """One line, ASCII minus and times, so patterns do not depend on wrapping or typography."""
    return re.sub(r'\s+', ' ', text).replace(u'−', '-').replace(u'×', 'x').replace(u'—', '--')


SN = r'([+-]\d*\.\d+)'          # a signed number as README writes it
IV = SN + r' \[' + SN + r', ' + SN + r'\]'


class Checker(object):
    def __init__(self, text):
        self.text = text
        self.problems = []
        self.covered = []           # (start, end) spans of numbers a rule has checked

    def find(self, name, pattern):
        m = re.search(pattern, self.text)
        if not m:
            self.problems.append('README does not state %s in the expected form' % name)
            return None
        self.covered += [m.span(i) for i in range(1, (m.lastindex or 0) + 1)]
        return m

    def val(self, what, cell, got, dp=None):
        """README's cell must be ``got`` rounded to the decimals README shows."""
        q = _num(cell)
        if dp is None:
            dp = len(cell.split('.')[1]) if '.' in cell else 0
        if round(got, dp) != round(q, dp) or (q != 0 and (q > 0) != (got > 0) and round(got, dp) != 0):
            self.problems.append('%s: README %s vs outputs %.*f' % (what, cell, dp + 1, got))

    def truth(self, what, ok):
        if not ok:
            self.problems.append(what)

    def uncovered(self, section):
        """Every signed number in ``section`` must sit inside a checked span."""
        start = self.text.find(section)
        if start < 0:
            return
        for m in re.finditer(SN, section):
            a, b = start + m.start(1), start + m.end(1)
            if not any(s <= a and b <= e for s, e in self.covered):
                self.problems.append('unchecked number %s in "%s..."' % (m.group(1), section[max(0, m.start() - 40):m.start()]))


def check_readme(readme):
    raw = io.open(readme, encoding='utf-8').read()
    sec = _section(raw, '## Where the planners fail')
    claim3 = re.search(r'\*\*3\. .*?(?=\n\n)', raw, re.S)
    if sec is None or claim3 is None:
        return ['README has no "## Where the planners fail" section or no claim 3']
    S = Sources()
    C = Checker(_norm(claim3.group(0)) + '\n' + _norm(sec))
    e2, e3 = S.overall('E2_avg'), S.overall('E3_avg')
    sg = S.summary['categories']['city=singapore']

    # claim 3: the two caveats the taxonomy puts on "negative in all three seeds"
    m = C.find('the session-clustered seed-1 box interval',
               r"clustered by driving session, seed 1's interval reaches zero \(\[" + SN + ', ' + SN + r'\]\)')
    if m:
        lo, hi = S.overall('E3_s1')['session']
        C.val('claim 3 seed 1 session low', m.group(1), lo)
        C.val('claim 3 seed 1 session high', m.group(2), hi)
        C.truth('claim 3: seed 1 session interval does not reach zero', round(hi, 3) >= 0)
    m = C.find('the Singapore caveat of claim 3',
               r"Singapore's ([\d,]+) scenes, from only two sessions, carry (\d+)% of the penalty -- "
               r'without Singapore it is ' + SN + ' and seed 1 is no longer significant')
    if m:
        C.val('claim 3 Singapore scenes', m.group(1), sg['n'], 0)
        C.truth('claim 3: Singapore is not two sessions', sg['sessions'] == 2)
        C.val('claim 3 Singapore share (%)', m.group(2), 100 * S.share('city', 'singapore', 'E3_avg'), 0)
        C.val('claim 3 box penalty without Singapore', m.group(3), S.loco[('singapore', 'E3_avg')]['delta'])
        s1 = S.loco[('singapore', 'E3_s1')]
        C.truth('claim 3: seed 1 without Singapore is still significant',
                not _excludes(*s1['log']) and not _excludes(*s1['session']))

    # bullet 1: the map gain
    m = C.find('the no-lead share of the learned map gain',
               r'Scenes without a lead vehicle carry (\d+)% of the learned map gain \(' + SN
               + r' overall\), and in fast scenes it is ' + SN)
    if m:
        C.val('no-lead share of E2 (%)', m.group(1), 100 * S.share('lead', 'none', 'E2_avg'), 0)
        C.val('E2 overall', m.group(2), e2['delta'])
        C.val('E2 in fast scenes', m.group(3), S.mean('speed', 'fast', 'E2_avg'))
    m = C.find('H1', r'pre-registered H1: ' + IV + r', not supported\)')
    if m:
        h = S.hyps[('H1', 'E2_avg', 'log')]
        for i, v in enumerate((h['estimate'], h['low'], h['high']), 1):
            C.val('H1 (log) value %d' % i, m.group(i), v)
        C.truth('H1 is supported in hypotheses.json', not h['supported_here']
                and not S.hyps[('H1', 'E2_avg', 'session')]['supported_here'])
    m = C.find('the DAC failure-rate drop of E2', r'the DAC failure rate drops by ([\d.]+) points')
    if m:
        C.val('E2 DAC failure-rate drop (points)', m.group(1),
              -100 * S.mean('all', 'all', 'E2_avg', 'fail_dac'))

    # bullet 2: the box penalty
    m = C.find('the E3 mechanism',
               r'Adding boxes to the map model \(' + SN + r'\) adds drivable-area failures \(' + SN
               + r' points\) and TTC failures \(' + SN + r' points\), but not slower plans \(' + SN
               + r' points\)')
    if m:
        C.val('E3 overall', m.group(1), e3['delta'])
        C.val('E3 DAC failures (points)', m.group(2), 100 * S.mean('all', 'all', 'E3_avg', 'fail_dac'))
        C.val('E3 TTC failures (points)', m.group(3), 100 * S.mean('all', 'all', 'E3_avg', 'fail_ttc'))
        C.val('E3 EP<0.5 (points)', m.group(4), 100 * S.mean('all', 'all', 'E3_avg', 'fail_ep'))
        ep = S.eff[('all', 'all', 'E3_avg', 'fail_ep')]
        C.truth('E3: "not slower plans" but the EP<0.5 log interval excludes zero',
                not _excludes(float(ep['log_low']), float(ep['log_high'])))
    m = C.find('the Singapore share of E3',
               r"Singapore's ([\d,]+) scenes, from two sessions, carry (\d+)% of it; without Singapore it is "
               + SN + r' and seed 1 is no longer significant, while the map x boxes interaction survives \('
               + SN + r'\)')
    if m:
        C.val('Singapore scenes', m.group(1), sg['n'], 0)
        C.val('Singapore share of E3 (%)', m.group(2), 100 * S.share('city', 'singapore', 'E3_avg'), 0)
        C.val('E3 without Singapore', m.group(3), S.loco[('singapore', 'E3_avg')]['delta'])
        e4 = S.loco[('singapore', 'E4_avg')]
        C.val('E4 without Singapore', m.group(4), e4['delta'])
        C.truth('E4 without Singapore does not survive',
                _excludes(*e4['log']) and _excludes(*e4['session']))

    # bullet 3: clustering
    m = C.find('the clustering count', r'(\d+) of (\d+) category-level intervals change verdict')
    changed, total, wl, ws = S.clustering()
    if m:
        C.val('intervals that change verdict', m.group(1), changed, 0)
        C.val('category-level intervals', m.group(2), total, 0)
    m = C.find('the clustered width ratio', r'median ([\d.]+)x wider')
    if m:
        C.val('median width ratio, log clustering', m.group(1), wl)
        C.val('median width ratio, session clustering', m.group(1), ws)

    # bullet 4: root causes
    m = C.find('the root-cause sample',
               r'(\d+) failures \((\d+) per stratum, at most one per log\), re-scored with the unchanged '
               r'scorer, reproduce the committed scores to ([\d.e-]+)\. DAC is the first failure in (\d+) of '
               r'them; when the plan first leaves the road it is a median ([\d.]+) m from the human path, '
               r'and none of those corners')
    if m:
        cases = S.cases
        per = {}
        for r in cases:
            per.setdefault(r['stratum'], []).append(r)
        C.val('root-cause cases', m.group(1), len(cases), 0)
        C.truth('root-cause sample is not %s per stratum' % m.group(2),
                all(len(v) == int(m.group(2)) for v in per.values()))
        C.truth('root-cause sample repeats a log within a stratum',
                all(len({r['log'] for r in v}) == len(v) for v in per.values()))
        C.truth('not every case reproduces', all(r['reproduced'] == 'True' for r in cases))
        errs = [float(x) for x in re.findall(r'largest absolute sub-score error ([\d.e-]+)\)', S.root_md)]
        C.truth('reproduction error: README %s vs root_causes.md max %s'
                % (m.group(3), max(errs) if errs else None), bool(errs) and float(m.group(3)) == max(errs))
        dac = [r for r in cases if r['primary'] == 'dac']
        C.val('DAC-primary cases', m.group(4), len(dac), 0)
        C.val('median distance to the logged path (m)', m.group(5), _median([float(r['dac_lateral_m']) for r in dac]))
        C.truth('a DAC failure corner lies on a mapped lane', not any(r['dac_corner_on_lane'] == 'True' for r in dac))

    C.uncovered(_norm(sec))
    return C.problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('compare')
    c.add_argument('fresh')
    c.add_argument('--cases', help='regenerated cases/ outputs to compare as well')
    r = sub.add_parser('readme')
    r.add_argument('readme')
    a = ap.parse_args()
    if a.cmd == 'compare':
        problems = compare_dir(a.fresh, OUT, ANALYZE_FILES)
        if a.cases:
            problems += compare_dir(a.cases, CASES, CASES_FILES)
        ok = 'regenerated failure-taxonomy outputs match the committed files'
    else:
        problems = check_readme(a.readme)
        ok = '%s: failure-taxonomy numbers match the committed outputs' % a.readme
    for p in problems:
        print('MISMATCH:', p)
    print(ok if not problems else '%d mismatch(es)' % len(problems))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
