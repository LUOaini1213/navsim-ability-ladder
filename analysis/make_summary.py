# -*- coding: utf-8 -*-
"""Regenerate results/pdm_summary.csv for every agent from the per-scene CSVs,
or check that the numbers README.md quotes are the numbers the CSVs give.

    python analysis/make_summary.py                    # rewrite results/pdm_summary.csv (13 agents)
    python analysis/make_summary.py --check README.md  # exit 1 on any mismatch

Column definitions follow analyze_pdm.py (which needs pandas and the
workspace): sub-score columns are means over the 563 scenes, ``score_zero`` is
the share of scenes with PDMS <= 0, ``*_fail`` the share where that sub-score
is below 1. Standard library only.
"""
import argparse
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_io import (REPO, load_rows, load_scores, resolve_csv_root,  # noqa: E402
                       split_logs, token_to_log)

METRICS = [('no_at_fault_collisions', 'NC'), ('drivable_area_compliance', 'DAC'),
           ('ego_progress', 'EP'), ('time_to_collision_within_bound', 'TTC'),
           ('comfort', 'C'), ('driving_direction_compliance', 'DDC'), ('score', 'PDMS')]

# summary name -> agent_subdir, in ladder order
AGENTS = [
    ('CV', 'cv_agent_mini'),
    ('Kinematic', 'kinematic_agent_mini'),
    ('PrivBrake', 'privileged_brake_mini'),
    ('MLP', 'mlp_agent_mini'),
    ('MapMLP', 'map_mlp_mini'),
    ('MapMLP-reg', 'map_mlp_reg_mini'),
    ('PrivMap', 'privileged_centerline_mini'),
    ('PrivMapKin', 'privileged_centerline_kin'),
    ('SpeedMLP', 'speed_mlp_mini'),
    ('SpeedMLP-e200', 'speed_mlp_e200_mini'),
    ('PrivGTPathKin', 'privileged_gtpath_kin'),
    ('PrivMapGTSpd', 'privileged_centerline_gtspeed'),
    ('Human', 'human_agent_mini'),
]

# README ladder table row label -> agent_subdir
README_ROWS = {
    'ConstantVelocity': 'cv_agent_mini', 'Kinematic': 'kinematic_agent_mini',
    'PrivBrake': 'privileged_brake_mini', 'EgoStatusMLP': 'mlp_agent_mini',
    'MapMLP': 'map_mlp_mini', 'PrivMap': 'privileged_centerline_mini',
    'PrivMapKin': 'privileged_centerline_kin', 'SpeedMLP': 'speed_mlp_mini',
    'PrivGTPathKin': 'privileged_gtpath_kin', 'PrivMapGTSpd': 'privileged_centerline_gtspeed',
    'Human': 'human_agent_mini',
}
# README learned-variants table: label -> agent_subdir (clean n=135 column)
README_CLEAN_ROWS = {
    'MapMLP': 'map_mlp_mini', 'MapMLP-reg': 'map_mlp_reg_mini', 'SpeedMLP': 'speed_mlp_mini',
    'SpeedMLP, 200 epochs': 'speed_mlp_e200_mini', 'PrivMapGTSpd': 'privileged_centerline_gtspeed',
}


def summary_rows(root):
    out = []
    for name, sub in AGENTS:
        rows = load_rows(root, sub)
        n = len(rows)
        rec = {'agent': name, 'n': n,
               'score_zero': sum(1 for r in rows.values() if r['score'] <= 0) / n}
        for col, short in METRICS:
            rec[short] = sum(r[col] for r in rows.values()) / n
        rec['NC_fail'] = sum(1 for r in rows.values() if r['no_at_fault_collisions'] < 1) / n
        rec['DAC_fail'] = sum(1 for r in rows.values() if r['drivable_area_compliance'] < 1) / n
        rec['TTC_fail'] = sum(1 for r in rows.values() if r['time_to_collision_within_bound'] < 1) / n
        out.append(rec)
    return out


def write_summary(root, path):
    cols = ['agent', 'n', 'score_zero'] + [s for _, s in METRICS] + ['NC_fail', 'DAC_fail', 'TTC_fail']
    lines = [','.join(cols)]
    for rec in summary_rows(root):
        lines.append(','.join(str(rec[c]) if c == 'agent' else repr(rec[c]) if isinstance(rec[c], float) else str(rec[c])
                              for c in cols))
    io.open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')


def clean_val_tokens(root, subdirs):
    _train, val = split_logs(root)
    t2l = token_to_log(root)
    common = None
    for sub in subdirs:
        toks = set(load_scores(root, sub))
        common = toks if common is None else common & toks
    return sorted(t for t in common if t2l.get(t) in val)


def check_readme(root, readme):
    text = io.open(readme, encoding='utf-8').read()
    problems = []
    scores = {sub: load_scores(root, sub) for sub in set(README_ROWS.values()) | set(README_CLEAN_ROWS.values())}
    common = None
    for s in scores.values():
        common = set(s) if common is None else common & set(s)
    common = sorted(common)
    seen = 0
    for m in re.finditer(r'^\| \**([A-Za-z]+)\** \|[^|]*\|[^|]*\| \**(0\.\d{3})\** \| \**(0\.\d{3})\** \|', text, re.M):
        label, pdms, dac = m.group(1), float(m.group(2)), float(m.group(3))
        sub = README_ROWS.get(label)
        if not sub:
            continue
        seen += 1
        d = scores[sub]
        got_p = sum(d[t]['score'] for t in common) / len(common)
        got_d = sum(d[t]['dac'] for t in common) / len(common)
        if abs(got_p - pdms) > 0.0006 or abs(got_d - dac) > 0.0006:
            problems.append(f'{label}: README {pdms:.3f}/{dac:.3f} vs CSV {got_p:.3f}/{got_d:.3f}')
    if seen != len(README_ROWS):
        problems.append(f'ladder table: found {seen} of {len(README_ROWS)} expected rows')
    val = clean_val_tokens(root, list(README_CLEAN_ROWS.values()) + ['privileged_centerline_kin'])
    if len(val) != 135:
        problems.append(f'clean split has {len(val)} scenes, README says 135')
    # The learned-variants table lives in its own section; its 4th cell is the
    # clean n=135 PDMS. Match the label cell exactly (MapMLP vs MapMLP-reg ...).
    start = text.find('## Putting a learned part back in')
    end = text.find('\n## ', start + 1)
    section = text[start:end if end > 0 else None]
    cell = {
        'MapMLP': r'MapMLP', 'MapMLP-reg': r'MapMLP-reg \([^|]*\)', 'SpeedMLP': r'SpeedMLP',
        'SpeedMLP, 200 epochs': r'SpeedMLP, 200 epochs', 'PrivMapGTSpd': r'PrivMapGTSpd \([^|]*\)',
    }
    for label, sub in README_CLEAN_ROWS.items():
        pat = r'^\| \**' + cell[label] + r'\** \|[^|]*\|[^|]*\| \**(0\.\d{3})\** \|'
        m = re.search(pat, section, re.M)
        if not m:
            problems.append(f'clean-split row for {label} not found')
            continue
        d = scores[sub]
        got = sum(d[t]['score'] for t in val) / len(val)
        if abs(got - float(m.group(1))) > 0.0006:
            problems.append(f'{label} clean n=135: README {m.group(1)} vs CSV {got:.3f}')
    hand = scores['privileged_centerline_kin']
    got = sum(hand[t]['score'] for t in val) / len(val)
    if 'PrivMapKin (0.730)' in text and abs(got - 0.730) > 0.0006:
        problems.append(f'PrivMapKin clean n=135: README 0.730 vs CSV {got:.3f}')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv-root', help='workspace exp/ or results/per_scene (auto)')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'pdm_summary.csv'))
    ap.add_argument('--check', metavar='README', help='verify the ladder numbers quoted in this file')
    a = ap.parse_args()
    root = resolve_csv_root(a.csv_root)
    if a.check:
        problems = check_readme(root, a.check)
        for p in problems:
            print('MISMATCH:', p)
        print(f'{a.check}: ladder and clean-split numbers match the per-scene CSVs under {root}'
              if not problems else f'{len(problems)} mismatch(es)')
        return 1 if problems else 0
    write_summary(root, a.out)
    print('wrote', a.out, 'from', root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
