# -*- coding: utf-8 -*-
"""Why the two privileged_brake_mini runs of 2026-08-20 differ, scene by scene.

Run 1 (started 12:52:28, CSV written 12:54:17) and run 2 (started 12:55:16) were first written
up here as coming from identical code, with the cause of the difference left open. They did not
come from identical code. Hydra's ``code/`` folder, read as a code snapshot, holds only the three
config files; the workspace copy of ``agents/privileged_brake_agent.py`` was saved at 12:54:58,
after run 1 finished and before run 2 started. Both runs used ``worker=sequential``. Scoring the
current agent again on 2026-09-24 reproduces run 2 in every scene (results/rerun_check.md), so run
2 is the canonical run and run 1 is kept as ``per_scene/privileged_brake_mini_superseded.csv``.
This script characterizes what the edit changed.

    python analysis/brake_rerun_diff.py      # writes results/brake_rerun_diff.md
"""
import io
import json
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
    a = load_rows(root, 'privileged_brake_mini_superseded')     # run 1
    b = load_rows(root, 'privileged_brake_mini')                # run 2, what the current code gives
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
    dac_a = sum(a[t]['drivable_area_compliance'] for t in common) / len(common)
    dac_b = sum(b[t]['drivable_area_compliance'] for t in common) / len(common)
    diffs = sorted(differ, key=lambda t: -abs(a[t]['score'] - b[t]['score']))
    reruns = json.load(io.open(os.path.join(REPO, 'results', 'rerun_check.json'),
                               encoding='utf-8'))['agents']['privileged_brake_mini']['reruns']
    same = [r for r in reruns if r['identical_scenes'] == len(common)]

    L = ['# privileged_brake_mini: the superseded run against the canonical one\n']
    L.append('Run 1 = `2026.08.20.12.52.28` (CSV written 12:54:17), run 2 = `2026.08.20.12.55.16`: same '
             'configuration, same metric cache, `worker=sequential` in both, %d scenes in both.\n' % len(common))
    L.append('**What changed between the two runs was the agent.** Hydra\'s `code/` folder, which an '
             'earlier version of this note read as a code snapshot, holds only the three config files. The '
             'workspace copy of `agents/privileged_brake_agent.py` was saved at 12:54:58, after run 1 wrote '
             'its CSV and before run 2 started. Scoring the current agent again on 2026-09-24 (%s) '
             'reproduces run 2 in all %d scenes in %d of %d re-runs ([rerun_check.md](rerun_check.md)), so '
             'run 2 is the canonical run and every report uses it. Run 1 is kept as '
             '`per_scene/privileged_brake_mini_superseded.csv`. What the edit was cannot be recovered: the '
             'workspace agent file is not under version control.\n'
             % ('; '.join(r['worker'] for r in reruns), len(common), len(same), len(reruns)))
    L.append('| | run 1 (superseded) | run 2 (canonical) |')
    L.append('|---|---:|---:|')
    L.append('| mean PDMS | %.4f | %.4f |' % (mean_a, mean_b))
    L.append('| mean DAC | %.4f | %.4f |' % (dac_a, dac_b))
    L.append('| scenes with a different score | %d of %d | |' % (len(differ), len(common)))
    L.append('| run 2 higher / lower | %d / %d | |' % (run2_higher, len(differ) - run2_higher))
    L.append('\n## Which sub-score moves\n')
    L.append('| sub-score | scenes where it differs |')
    L.append('|---|---:|')
    for key, short in METRICS:
        L.append('| %s | %d |' % (short, which.get(key, 0)))
    L.append('\nEgo progress differs in almost every affected scene: the edit changed the trajectories the '
             'agent produces, not how they are scored.\n')
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
