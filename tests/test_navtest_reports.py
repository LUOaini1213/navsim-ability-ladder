# -*- coding: utf-8 -*-
"""The navtest tables are exactly what the committed per-scene CSVs give.

Regenerates results/navtest/*.md from results/navtest/per_scene/*.csv.gz (fixed
seed) and compares byte for byte, checks the archive is complete and consistent,
and checks the navtest numbers quoted in README.md. No dataset, no devkit,
standard library only.
"""
import csv
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

import analyze_navtest as nav  # noqa: E402

PER_SCENE = nav.PER_SCENE
N_SCENES = 12146
N_LOGS = 136


class Archive(unittest.TestCase):
    def setUp(self):
        self.R = nav.Runs(PER_SCENE, '')

    def test_every_run_has_the_same_12146_scenes(self):
        stems = [s for _, s, _, _ in nav.LADDER]
        stems += ['l_%s_s%d' % (b, s) for s in nav.SEEDS
                  for b in ('pose_kin', 'pose_agents', 'pose_map', 'pose_map_agents',
                            'ds_kin', 'ds_map', 'ds_map_agents')]
        reference = None
        for stem in stems:
            with self.subTest(run=stem):
                tokens = set(self.R[stem])
                self.assertEqual(len(tokens), N_SCENES)
                if reference is None:
                    reference = tokens
                self.assertEqual(tokens, reference)

    def test_token_log_covers_every_scene(self):
        from ladder_io import token_to_log
        t2l = token_to_log(PER_SCENE)
        self.assertEqual(len(t2l), N_SCENES)
        self.assertEqual(len(set(t2l.values())), N_LOGS)
        self.assertTrue(set(self.R['cv']).issubset(t2l))

    def test_cells_csv_describes_27_trained_cells(self):
        with io.open(os.path.join(nav.OUT, 'cells.csv'), encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 27)
        self.assertEqual({int(r['seed']) for r in rows}, {0, 1, 2})
        for r in rows:
            with self.subTest(cell=r['cell']):
                # the checkpoint is the last epoch, never the one validation liked best
                self.assertEqual(r['select'], 'last')
                self.assertEqual(int(r['n_val_logs']), 60)
                self.assertGreater(float(r['final_val_l1']), 0.0)


class ReportsReproduce(unittest.TestCase):
    def setUp(self):
        self.R = nav.Runs(PER_SCENE, '')

    def _same(self, name, fn):
        committed = io.open(os.path.join(nav.OUT, name), encoding='utf-8').read()
        self.assertEqual(fn(self.R), committed, name + ' is not what the CSVs give')

    def test_ladder_regenerates_byte_for_byte(self):
        self._same('ladder.md', nav.ladder_md)

    def test_factorial_regenerates_byte_for_byte(self):
        self._same('factorial.md', nav.factorial_md)

    def test_path_speed_regenerates_byte_for_byte(self):
        self._same('path_speed.md', nav.path_speed_md)

    def test_regularisation_regenerates_byte_for_byte(self):
        self._same('regularisation.md', nav.regularisation_md)

    def test_readme_navtest_numbers_match_the_csvs(self):
        problems = nav.check_readme(self.R, os.path.join(ROOT, 'README.md'))
        self.assertEqual(problems, [], '\n'.join(problems))


class HeadlineClaims(unittest.TestCase):
    """The claims the README makes in bold, asserted against the CSVs."""

    def setUp(self):
        self.R = nav.Runs(PER_SCENE, '')

    def test_map_beats_gt_boxes_by_about_seven_times(self):
        _n, map_gain, lo, _hi = self.R.delta('kin', 'privmapkin')
        _n, box_gain, blo, _bhi = self.R.delta('kin', 'privbrake')
        self.assertGreater(lo, 0)
        self.assertGreater(blo, 0)
        self.assertGreater(map_gain / box_gain, 6.0)

    def test_gt_boxes_hurt_once_the_model_has_the_map(self):
        for s in nav.SEEDS:
            with self.subTest(seed=s):
                _n, mean, lo, hi = self.R.delta('l_pose_map_s%d' % s, 'l_pose_map_agents_s%d' % s)
                self.assertLess(mean, 0)
                self.assertLess(hi, 0, 'interval should exclude zero')

    def test_learned_speed_on_the_rule_path_beats_the_rule_speed(self):
        for s in nav.SEEDS:
            with self.subTest(seed=s):
                _n, mean, lo, _hi = self.R.delta('privmapkin', 'l_ds_map_s%d' % s)
                self.assertGreater(mean, 0)
                self.assertGreater(lo, 0)

    def test_equal_open_loop_gains_move_the_score_in_opposite_directions(self):
        """wd 1e-4 and dropout 0.2 improve the validation L1 by comparable amounts,
        and the simulation score goes up for one and down for the other."""
        val = nav.load_cells()
        base = val['l_pose_map_s0']
        d_wd = base - val['l_reg_pose_map_wd1e-4_s0']
        d_do = base - val['l_reg_pose_map_do0.2_s0']
        for name, gain in (('wd 1e-4', d_wd), ('dropout 0.2', d_do)):
            with self.subTest(config=name):
                self.assertGreater(gain, 0.01, 'expected an open-loop improvement')
        self.assertLess(abs(d_wd - d_do), 0.015, 'the two gains should be comparable')

        _n, m_wd, lo_wd, _hi = self.R.delta('l_pose_map_s0', 'l_reg_pose_map_wd1e-4_s0')
        _n, m_do, _lo, hi_do = self.R.delta('l_pose_map_s0', 'l_reg_pose_map_do0.2_s0')
        self.assertGreater(lo_wd, 0, 'wd 1e-4 should be a significant gain')
        self.assertLess(hi_do, 0, 'dropout 0.2 should be a significant loss')
        self.assertGreater(m_wd, 0)
        self.assertLess(m_do, 0)


if __name__ == '__main__':
    unittest.main()
