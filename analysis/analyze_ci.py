# -*- coding: utf-8 -*-
"""Paired bootstrap confidence intervals over the agent ladder.

Reads the per-scene CSVs already produced by run_pdm_score.py — does NOT
re-run any evaluation. Runs from the NAVSIM workspace (`exp/`) or from the
CSVs committed under `results/per_scene/` (see ladder_io.resolve_csv_root).

Usage:  python analysis/analyze_ci.py [--csv-root DIR] [--out FILE]
        default output: results/ci_report.md (repo) or <exp>/analysis/ci_report.md
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_io import (B, REPO, SEED, bootstrap, is_per_scene, load_scores,  # noqa: E402
                       resolve_csv_root)

# Canonical run = the FIRST csv per agent, matching exp/analysis/pdm_report.md.
# NOTE: privileged_brake_mini has two runs (12:52 PDMS 0.602 / 12:55 PDMS 0.593).
# pdm_report.md used the first; we keep that for consistency and flag it below.
AGENTS = [
    ('CV', 'cv_agent_mini', 'speed only'),
    ('Kinematic', 'kinematic_agent_mini', 'v, a, command'),
    ('PrivBrake', 'privileged_brake_mini', 'GT boxes, no map'),
    ('MLP', 'mlp_agent_mini', 'learned kinematics (blind)'),
    ('MapMLP (learned+map)', 'map_mlp_mini', 'learned, sees centerline'),
    ('MapMLP-reg (learned+map)', 'map_mlp_reg_mini', 'learned, sees centerline; h128 / drop 0.2 / wd 1e-3'),
    ('PrivMap(IDM)', 'privileged_centerline_mini', 'map centerline + IDM'),
    ('PrivMapKin', 'privileged_centerline_kin', 'map centerline + kinematic'),
    ('SpeedMLP (hand path+learned speed)', 'speed_mlp_mini', 'map centerline + LEARNED arc-length'),
    ('SpeedMLP-e200', 'speed_mlp_e200_mini', 'same, 200 epochs'),
    ('PrivGTPathKin', 'privileged_gtpath_kin', 'logged path + kinematic'),
    ('PrivMapGTSpd', 'privileged_centerline_gtspeed', 'map centerline + human arc-length'),
    ('Human', 'human_agent_mini', 'logged future'),
]

PAIRS = [
    ('CV', 'Kinematic', 'Add kinematics (accel + command yaw)'),
    ('Kinematic', 'MLP', 'Learned blind planner vs hand rule'),
    ('Kinematic', 'PrivBrake', 'Add GT detection boxes + brake'),
    ('MLP', 'PrivMapKin', 'Add on-route map centerline'),
    ('PrivMap(IDM)', 'PrivMapKin', 'Drop IDM on the centerline (counter-intuitive)'),
    ('PrivMapKin', 'PrivMapGTSpd', 'Swap in human arc-length speed (geometry fixed)'),
    ('PrivMapKin', 'PrivGTPathKin', 'Swap in logged path (speed fixed)'),
    ('PrivMapGTSpd', 'Human', 'Remaining gap to Human'),
    ('MLP', 'MapMLP (learned+map)', 'Give the LEARNED model the centerline too'),
    ('MapMLP (learned+map)', 'PrivMapKin', 'Same centerline: hand rule vs learned'),
    ('MapMLP (learned+map)', 'MapMLP-reg (learned+map)', 'Regularize the learned map model'),
    ('MapMLP-reg (learned+map)', 'PrivMapKin', 'Regularized learned vs hand rule (same centerline)'),
    ('PrivMapKin', 'SpeedMLP (hand path+learned speed)', 'Hand path: LEARNED speed vs hand kinematic speed'),
    ('SpeedMLP (hand path+learned speed)', 'PrivMapGTSpd', 'Learned speed vs human speed (same path, upper bound)'),
    ('SpeedMLP (hand path+learned speed)', 'SpeedMLP-e200', 'Speed model: 80 vs 200 epochs'),
]


def render(root):
    data = {name: load_scores(root, sub) for name, sub, _ in AGENTS}

    common = set(data[AGENTS[0][0]])
    for name, _, _ in AGENTS[1:]:
        common &= set(data[name])
    common = sorted(common)

    L = []
    L.append('# Paired bootstrap CIs over the agent ladder\n')
    L.append('Split `warmup_test_e2e`, %d scenes common to all %d agents, '
             'same metric cache. %d bootstrap resamples, seed %d. '
             'No evaluation was re-run — this reads the CSVs already in `exp/`.\n'
             % (len(common), len(AGENTS), B, SEED))

    L.append('\n## Ladder\n')
    L.append('| agent | sees | PDMS | DAC | DAC fail |')
    L.append('|---|---|---:|---:|---:|')
    for name, _, sees in AGENTS:
        d = data[name]
        n = len(common)
        pdms = sum(d[t]['score'] for t in common) / n
        dac = sum(d[t]['dac'] for t in common) / n
        fail = sum(1 for t in common if d[t]['dac'] < 0.5) / n
        L.append('| %s | %s | %.3f | %.3f | %.1f%% |' % (name, sees, pdms, dac, 100 * fail))

    L.append('\n## Paired deltas (positive = the second agent is better)\n')
    L.append('| comparison | dPDMS | 95% CI | verdict |')
    L.append('|---|---:|---|---|')
    for a, b, label in PAIRS:
        diffs = [data[b][t]['score'] - data[a][t]['score'] for t in common]
        m, lo, hi = bootstrap(diffs)
        verdict = 'CI excludes 0' if (lo > 0 or hi < 0) else 'CI spans 0'
        L.append('| %s | %+.4f | [%+.4f, %+.4f] | %s |' % (label, m, lo, hi, verdict))

    # difference-in-differences: speed vs geometry
    base, spd, geo = data['PrivMapKin'], data['PrivMapGTSpd'], data['PrivGTPathKin']
    dd = [(spd[t]['score'] - base[t]['score']) - (geo[t]['score'] - base[t]['score'])
          for t in common]
    m, lo, hi = bootstrap(dd, seed=7)
    L.append('\n## Speed vs geometry (difference in differences)\n')
    L.append('Gain from swapping in human speed **minus** gain from swapping in the '
             'logged path, same 563 paired scenes:\n')
    L.append('- %+.4f, 95%% CI [%+.4f, %+.4f] — %s\n'
             % (m, lo, hi, 'speed is worth significantly more'
                if lo > 0 else ('geometry is worth significantly more' if hi < 0
                                else 'not separable')))

    L.append('\n## Caveats\n')
    L.append('- `privileged_brake_mini` has **two** runs with different results '
             '(12:52 PDMS 0.602 / DAC 0.766; 12:55 PDMS 0.593 / DAC 0.785). '
             'This report and `pdm_report.md` both use the first. '
             'The gap is unexplained and should be pinned down before the number is quoted.')
    L.append('- MLP, MapMLP(-reg) and SpeedMLP are learned; on this split their scores are '
             'contaminated (428 of 563 scenes were training logs). '
             'The clean comparison is the held-out 135 (MLP 0.475 vs CV 0.181).')
    L.append('- Priv* and Human consume ground truth or map privilege and are **not '
             'deployable** — they are upper bounds, not results.')
    L.append('- Multiple comparisons were made; no multiplicity correction is applied. '
             'The two smallest effects (drop-IDM, logged-path) would not survive a '
             'strict Bonferroni threshold and should be reported as suggestive.')
    L.append('- `warmup_test_e2e` is not the official navtest leaderboard.')
    return '\n'.join(L) + '\n', len(common)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv-root', help='workspace exp/ or results/per_scene (auto-detected)')
    ap.add_argument('--out', help='report path')
    a = ap.parse_args(argv)
    root = resolve_csv_root(a.csv_root)
    out = a.out or (os.path.join(REPO, 'results', 'ci_report.md') if is_per_scene(root)
                    else os.path.join(root, 'analysis', 'ci_report.md'))
    text, n = render(root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    io.open(out, 'w', encoding='utf-8', newline='\n').write(text)
    print('wrote', out)
    print('scenes=%d agents=%d' % (n, len(AGENTS)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
