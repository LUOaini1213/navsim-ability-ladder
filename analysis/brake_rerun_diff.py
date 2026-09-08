# -*- coding: utf-8 -*-
"""Characterize the two privileged_brake_mini runs (12:52 and 12:55 on 2026-08-20).

Both runs were launched from identical code and configuration snapshots (the
hydra `code/` directories differ only in `output_dir`), yet 130 of the 563
scenes score differently. This script says exactly where they differ so the
"unexplained discrepancy" in the reports is at least a characterized one.

    python analysis/brake_rerun_diff.py      # writes results/brake_rerun_diff.md
"""
import io
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_io import PER_SCENE, REPO, load_rows  # noqa: E402

METRICS = [
    ('no_at_fault_collisions', 'NC'), ('drivable_area_compliance', 'DAC'), ('ego_progress', 'EP'),
    ('time_to_collision_within_bound', 'TTC'), ('comfort', 'C'),
    ('driving_direction_compliance', 'DDC'), ('score', 'PDMS'),
]


def render(root=PER_SCENE):
    a = load_rows(root, 'privileged_brake_mini')
    b = load_rows(root, 'privileged_brake_mini_run2')
    common = sorted(set(a) & set(b))
    differ = [t for t in common if abs(a[t]['score'] - b[t]['score']) > 1e-9]
    which = Counter()
    for t in differ:
        for key, _ in METRICS:
            if key in a[t] and key in b[t] and abs(a[t][key] - b[t][key]) > 1e-9:
                which[key] += 1
    run2_higher = sum(1 for t in differ if b[t]['score'] > a[t]['score'])
    mean_a = sum(a[t]['score'] for t in common) / len(common)
    mean_b = sum(b[t]['score'] for t in common) / len(common)
    diffs = sorted(differ, key=lambda t: -abs(a[t]['score'] - b[t]['score']))

    L = ['# privileged_brake_mini: the two runs, scene by scene\n']
    L.append('Run 1 = `2026.08.20.12.52.28` (used in every report), run 2 = `2026.08.20.12.55.16`. '
             'Identical code and configuration snapshots (the hydra `code/` folders differ only in '
             '`output_dir`); same metric cache; %d scenes in both.\n' % len(common))
    L.append('| | run 1 | run 2 |')
    L.append('|---|---:|---:|')
    L.append('| mean PDMS | %.4f | %.4f |' % (mean_a, mean_b))
    L.append('| scenes with a different score | %d of %d | |' % (len(differ), len(common)))
    L.append('| run 2 higher / lower | %d / %d | |' % (run2_higher, len(differ) - run2_higher))
    L.append('\n## Which sub-score moves\n')
    L.append('| sub-score | scenes where it differs |')
    L.append('|---|---:|')
    for key, short in METRICS:
        L.append('| %s | %d |' % (short, which.get(key, 0)))
    L.append('\nEgo progress differs in almost every affected scene, i.e. the *trajectory* itself '
             'differed between the runs, not only its scoring. With identical code and inputs that '
             'points upstream of the agent — at the scene loading / annotation order the agent '
             'consumes, or at the parallel scorer — rather than at a parameter change. It has not '
             'been pinned down; every report keeps using run 1 and says so.\n')
    L.append('\n## Largest per-scene differences\n')
    L.append('| token | run 1 | run 2 | NC 1→2 | DAC 1→2 | EP 1→2 |')
    L.append('|---|---:|---:|---|---|---|')
    for t in diffs[:15]:
        L.append('| `%s` | %.3f | %.3f | %.2f→%.2f | %.2f→%.2f | %.2f→%.2f |' % (
            t, a[t]['score'], b[t]['score'],
            a[t]['no_at_fault_collisions'], b[t]['no_at_fault_collisions'],
            a[t]['drivable_area_compliance'], b[t]['drivable_area_compliance'],
            a[t]['ego_progress'], b[t]['ego_progress']))
    return '\n'.join(L) + '\n', len(common), len(differ)


def main():
    text, n, d = render()
    out = os.path.join(REPO, 'results', 'brake_rerun_diff.md')
    io.open(out, 'w', encoding='utf-8', newline='\n').write(text)
    print('wrote', out, '(%d scenes, %d differ)' % (n, d))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
