# -*- coding: utf-8 -*-
"""Every committed report is exactly what the committed per-scene CSVs give.

Regenerates results/ci_report.md and results/clean_ladder.md from
results/per_scene/ (fixed seed, 10000 resamples) and compares byte for byte;
rebuilds results/pdm_summary.csv and compares numerically; checks the ladder
numbers quoted in README.md. No dataset, no devkit, standard library only.
"""
import csv
import io
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'analysis')
PER_SCENE = os.path.join(ROOT, 'results', 'per_scene')
sys.path.insert(0, ANALYSIS)

import analyze_ci  # noqa: E402
import analyze_clean_ladder  # noqa: E402
import make_summary  # noqa: E402


class PerSceneArchive(unittest.TestCase):
    def test_every_agent_has_563_scenes_and_the_split_has_135_val(self):
        subs = [s for _, s in make_summary.AGENTS]
        for sub in subs:
            with self.subTest(agent=sub):
                self.assertEqual(len(make_summary.load_scores(PER_SCENE, sub)), 563)
        self.assertEqual(len(make_summary.clean_val_tokens(PER_SCENE, subs)), 135)
        train, val = make_summary.split_logs(PER_SCENE)
        self.assertEqual((len(train), len(val)), (51, 13))


class ReportsReproduce(unittest.TestCase):
    def test_ci_report_is_regenerated_byte_for_byte(self):
        text, n = analyze_ci.render(PER_SCENE)
        self.assertEqual(n, 563)
        committed = io.open(os.path.join(ROOT, 'results', 'ci_report.md'), encoding='utf-8').read()
        self.assertEqual(text, committed)

    def test_clean_ladder_is_regenerated_byte_for_byte(self):
        text, n_val, n_tr = analyze_clean_ladder.render(PER_SCENE)
        self.assertEqual((n_val, n_tr), (135, 428))
        committed = io.open(os.path.join(ROOT, 'results', 'clean_ladder.md'), encoding='utf-8').read()
        self.assertEqual(text, committed)

    def test_pdm_summary_matches_numerically(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'pdm_summary.csv')
            make_summary.write_summary(PER_SCENE, out)
            fresh = {r['agent']: r for r in csv.DictReader(io.open(out, encoding='utf-8'))}
        committed = {r['agent']: r for r in csv.DictReader(
            io.open(os.path.join(ROOT, 'results', 'pdm_summary.csv'), encoding='utf-8'))}
        self.assertEqual(set(committed), set(fresh))
        for agent, row in committed.items():
            for key, value in row.items():
                if key == 'agent':
                    continue
                self.assertAlmostEqual(float(value), float(fresh[agent][key]), places=9, msg=f'{agent}.{key}')

    def test_readme_ladder_numbers_match_the_csvs(self):
        problems = make_summary.check_readme(PER_SCENE, os.path.join(ROOT, 'README.md'))
        self.assertEqual(problems, [], '\n'.join(problems))

    def test_scripts_run_from_a_clone_with_no_workspace(self):
        env = dict(os.environ)
        env.pop('NAVSIM_EXP_ROOT', None)
        with tempfile.TemporaryDirectory() as tmp:
            for script in ('analyze_ci.py', 'analyze_clean_ladder.py'):
                out = os.path.join(tmp, script + '.md')
                r = subprocess.run([sys.executable, os.path.join(ANALYSIS, script), '--out', out],
                                   capture_output=True, text=True, env=env, cwd=tmp)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(os.path.getsize(out) > 1000)


if __name__ == '__main__':
    unittest.main()
