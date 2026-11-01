# -*- coding: utf-8 -*-
"""The committed per-scene CSVs are what the current code produces.

results/rerun_check.json records, for every agent on the 563-scene split, the fingerprint of
each re-run made from the code in this repository (analysis/rerun_check.py, in the workspace).
Here, with no workspace: every committed CSV must still have that fingerprint, every agent must
have been re-run, the one agent whose August runs disagreed must have been re-run under more
than one worker, and the superseded run must be the one that differs. Standard library only.
"""
import io
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_SCENE = os.path.join(ROOT, 'results', 'per_scene')
sys.path.insert(0, os.path.join(ROOT, 'analysis'))

import make_summary  # noqa: E402
import rerun_check  # noqa: E402
from ladder_io import fingerprint  # noqa: E402


class RerunCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.load(io.open(os.path.join(ROOT, 'results', 'rerun_check.json'), encoding='utf-8'))

    def test_every_ladder_agent_was_rerun(self):
        self.assertEqual(set(self.data['agents']), {sub for _, sub in make_summary.AGENTS})
        for sub, rec in self.data['agents'].items():
            with self.subTest(agent=sub):
                self.assertGreaterEqual(len(rec['reruns']), 1)

    def test_committed_csvs_are_what_every_rerun_produced(self):
        for sub, rec in self.data['agents'].items():
            with self.subTest(agent=sub):
                sha, n = fingerprint(os.path.join(PER_SCENE, sub + '.csv'))
                self.assertEqual((sha, n), (rec['committed_sha256'], 563))
                self.assertEqual(rec['canonical_sha256'], sha, 'committed copy differs from the workspace run')
                for r in rec['reruns']:
                    # bit-identical, or float noise far below the three decimals anything is reported to
                    self.assertLessEqual(r['max_abs_diff'], 1e-9, r['run'])
                    self.assertEqual(r['sha256'] == sha, r['identical_scenes'] == 563, r['run'])

    def test_brake_agent_was_rerun_sequentially_and_in_parallel(self):
        workers = [r['worker'] for r in self.data['agents']['privileged_brake_mini']['reruns']]
        self.assertGreaterEqual(workers.count('sequential'), 2, workers)
        self.assertTrue(any('pool' in w for w in workers), workers)

    def test_the_superseded_run_is_the_one_that_differs(self):
        s = self.data['superseded']
        sha, n = fingerprint(os.path.join(ROOT, *s['file'].split('/')))
        self.assertEqual((sha, n), (s['sha256'], 563))
        self.assertNotEqual(sha, self.data['agents']['privileged_brake_mini']['committed_sha256'])
        self.assertLess(s['identical_scenes_vs_canonical'], 563)

    def test_markdown_is_regenerated_byte_for_byte(self):
        committed = io.open(os.path.join(ROOT, 'results', 'rerun_check.md'), encoding='utf-8').read()
        self.assertEqual(rerun_check.render_md(self.data), committed)

    def test_fingerprint_ignores_row_order_but_not_values(self):
        import tempfile
        src = io.open(os.path.join(PER_SCENE, 'cv_agent_mini.csv'), encoding='utf-8').read().splitlines()
        head, body = src[0], [l for l in src[1:] if ',average,' not in l]
        with tempfile.TemporaryDirectory() as tmp:
            shuffled = os.path.join(tmp, 'a.csv')
            io.open(shuffled, 'w', encoding='utf-8').write('\n'.join([head] + body[::-1]) + '\n')
            self.assertEqual(fingerprint(shuffled), fingerprint(os.path.join(PER_SCENE, 'cv_agent_mini.csv')))
            cells = body[0].split(',')
            cells[-1] = repr(float(cells[-1]) + 1e-12)
            changed = os.path.join(tmp, 'b.csv')
            io.open(changed, 'w', encoding='utf-8').write('\n'.join([head, ','.join(cells)] + body[1:]) + '\n')
            self.assertNotEqual(fingerprint(changed)[0], fingerprint(shuffled)[0])


if __name__ == '__main__':
    unittest.main()
