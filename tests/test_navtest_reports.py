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
import analyze_openloop_csv as ol  # noqa: E402

PER_SCENE = nav.PER_SCENE
N_SCENES = 12146
N_LOGS = 136
N_OPENLOOP_RUNS = 45


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

    def test_openloop_summary_regenerates_byte_for_byte(self):
        committed = io.open(os.path.join(nav.OUT, ol.REPORT), encoding='utf-8').read()
        self.assertEqual(ol.report(ol.load_runs()), committed,
                         ol.REPORT + ' is not what openloop.csv gives')

    def test_readme_navtest_numbers_match_the_csvs(self):
        problems = nav.check_readme(self.R, os.path.join(ROOT, 'README.md'))
        self.assertEqual(problems, [], '\n'.join(problems))

    def test_readme_openloop_numbers_match_the_csv(self):
        problems = ol.check_readme(ol.load_runs(), os.path.join(ROOT, 'README.md'))
        self.assertEqual(problems, [], '\n'.join(problems))


class OpenLoopArchive(unittest.TestCase):
    """openloop.csv is one row per run; two of its columns must agree with the rest.

    The displacement errors in that CSV come from the training runs and cannot be
    rebuilt here, but the score and the validation loss can be, so they are: if the
    file were stale or hand-edited, these would catch it.
    """

    def setUp(self):
        self.rows = ol.load_runs()
        self.R = nav.Runs(PER_SCENE, '')

    def test_one_row_per_scored_run(self):
        self.assertEqual(len(self.rows), N_OPENLOOP_RUNS)
        self.assertEqual(len({r['run'] for r in self.rows}), N_OPENLOOP_RUNS)

    def test_pdms_column_is_the_per_scene_mean(self):
        for r in self.rows:
            with self.subTest(run=r['run']):
                self.assertEqual(int(r['n']), N_SCENES)
                self.assertAlmostEqual(self.R.mean('l_' + r['run']), float(r['PDMS']), places=6)

    def test_val_l1_column_matches_cells_csv(self):
        cells = nav.load_cells()
        for r in self.rows:
            stem = r['run'][:-3] if r['run'].endswith('_rs') else r['run']
            with self.subTest(run=r['run']):
                # a _rs run re-times the same checkpoint, so it shares the cell's loss
                self.assertAlmostEqual(cells['l_' + stem], float(r['val_l1_navtrain']), places=5)

    def test_spearman_matches_a_known_case(self):
        # rank correlation of two exactly opposed orderings is -1, of two equal ones +1
        self.assertAlmostEqual(ol.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(ol.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        # ties take the average rank: 1, 2.5, 2.5, 4
        self.assertEqual(ol.ranks([5, 6, 6, 7]), [1.0, 2.5, 2.5, 4.0])


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


class ReadmeProse(unittest.TestCase):
    """check_readme covers the prose, not only the tables: a changed sign, a changed digit
    or a new unchecked number in README's navtest sections is a mismatch. Each case edits
    a scratch copy of README and expects the check to fail on exactly that edit."""

    @classmethod
    def setUpClass(cls):
        cls.R = nav.Runs(PER_SCENE, '')
        with io.open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
            cls.text = f.read()

    def _problems(self, old, new):
        import tempfile
        self.assertEqual(self.text.count(old), 1, old)
        fd, path = tempfile.mkstemp(suffix='.md')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
                f.write(self.text.replace(old, new))
            return nav.check_readme(self.R, path)
        finally:
            os.remove(path)

    def test_claim_5_with_the_signs_main_used_to_quote_is_caught(self):
        # Until this check existed, README quoted the learned-speed path swap as
        # "-0.004 ..., +0.008, -0.022": each value is the negation of path_speed.md.
        problems = self._problems(u'swap is +0.004 (interval spans zero),\n−0.008 and +0.022',
                                  u'swap is −0.004 (interval spans zero),\n+0.008 and −0.022')
        flagged = [p for p in problems if p.startswith('claim 5 path swap at the learned speed')]
        self.assertEqual(len(flagged), 3, problems)

    def test_a_flipped_interval_bound_is_caught(self):
        problems = self._problems('**+0.296 [+0.288,', u'**+0.296 [−0.288,')
        self.assertEqual(problems, ['claim 1 map gain CI low: README -0.288 vs CSV +0.2876'])

    def test_a_wrong_verdict_is_caught(self):
        problems = self._problems('box effect on its own is not significant in\ntwo seeds of three',
                                  'box effect on its own is not significant in\none seeds of three')
        self.assertTrue(any('claim 3, boxes without map' in p for p in problems), problems)

    def test_an_unchecked_signed_number_is_caught(self):
        problems = self._problems('Giving the speed model the GT', 'It moves +0.123. Giving the speed model the GT')
        self.assertTrue(any('signed number +0.123 is not checked' in p for p in problems), problems)


if __name__ == '__main__':
    unittest.main()
