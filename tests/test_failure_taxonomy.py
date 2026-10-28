# -*- coding: utf-8 -*-
"""Unit tests for the pure-Python core of the failure taxonomy (standard library only).

Covers the pre-registered bins (docs/FAILURE_TAXONOMY.md), the lead-gap rule shared with
PrivilegedBrakeAgent, the centerline geometry, the outcome indicators, the cluster
bootstrap (including the replay of cluster_ci's random stream) and the root-cause rules.
"""
import math
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))

import failure_taxonomy_core as core  # noqa: E402


def straight_cl(k=20, length=60.0, y=0.0):
    return [(length * i / (k - 1), y, 1.0, 0.0) for i in range(k)]


def arc_cl(radius, k=20, length=60.0, left=True):
    """Centerline that turns with a constant radius from the origin."""
    rows = []
    sgn = 1.0 if left else -1.0
    for i in range(k):
        s = length * i / (k - 1)
        th = s / radius
        rows.append((radius * math.sin(th), sgn * radius * (1 - math.cos(th)), math.cos(th), sgn * math.sin(th)))
    return rows


class Bins(unittest.TestCase):
    def test_speed_edges(self):
        cases = [(0.0, 'stopped'), (0.4999, 'stopped'), (0.5, 'slow'), (4.999, 'slow'), (5.0, 'medium'),
                 (9.999, 'medium'), (10.0, 'fast'), (30.0, 'fast'), (float('nan'), 'n/a'), (None, 'n/a')]
        for v, want in cases:
            with self.subTest(v=v):
                self.assertEqual(core.speed_band(v), want)

    def test_command_argmax_first_on_ties(self):
        self.assertEqual(core.command_of([1, 0, 0, 0]), 'left')
        self.assertEqual(core.command_of([0, 1, 0, 0]), 'straight')
        self.assertEqual(core.command_of([0, 0, 1, 0]), 'right')
        self.assertEqual(core.command_of([0, 0, 0, 1]), 'unknown')
        self.assertEqual(core.command_of([0, 0, 0, 0]), 'left')  # numpy argmax rule, as the agents
        self.assertEqual(core.command_of([0, 1]), 'n/a')

    def test_lead_band_edges(self):
        self.assertEqual(core.lead_band(None), 'none')
        self.assertEqual(core.lead_band(-1.0), 'close')
        self.assertEqual(core.lead_band(9.99), 'close')
        self.assertEqual(core.lead_band(10.0), 'mid')
        self.assertEqual(core.lead_band(24.99), 'mid')
        self.assertEqual(core.lead_band(25.0), 'far')

    def test_junction_band(self):
        self.assertEqual(core.junction_band(True, 0.0), 'in')
        self.assertEqual(core.junction_band(False, 30.0), 'near')
        self.assertEqual(core.junction_band(False, 31.6), 'none')
        self.assertEqual(core.junction_band(False, None), 'none')
        self.assertEqual(core.junction_band(None, None), 'n/a')

    def test_curvature_and_lane_change_bands(self):
        self.assertEqual(core.curvature_band(14.99), 'straight')
        self.assertEqual(core.curvature_band(15.0), 'bend')
        self.assertEqual(core.curvature_band(44.99), 'bend')
        self.assertEqual(core.curvature_band(45.0), 'turn')
        self.assertEqual(core.curvature_band(None), 'n/a')
        self.assertEqual(core.lane_change_band(1.99), 'no')
        self.assertEqual(core.lane_change_band(-2.0), 'yes')
        self.assertEqual(core.lane_change_band(None), 'n/a')

    def test_stop_go(self):
        self.assertEqual(core.stop_go(0.2, [0.1] * 8), 'stationary')
        self.assertEqual(core.stop_go(0.2, [0.1] * 7 + [0.49]), 'stationary')
        self.assertEqual(core.stop_go(0.9, [0.5, 1, 1.5, 2.0, 2, 2, 2, 2]), 'starting')
        self.assertEqual(core.stop_go(0.9, [0.5, 1, 1.5, 1.9, 1.9, 1.9, 1.9, 1.9]), 'moving')
        self.assertEqual(core.stop_go(5.0, [4, 3, 2, 1, 0.4, 0, 0, 0]), 'stopping')
        self.assertEqual(core.stop_go(1.5, [1, 0.5, 0.2, 0, 0, 0, 0, 0]), 'moving')  # v0 < 2: not 'stopping'
        self.assertEqual(core.stop_go(8.0, [8] * 8), 'moving')
        self.assertEqual(core.stop_go(None, [1] * 8), 'n/a')

    def test_city_and_session(self):
        self.assertEqual(core.city_of('us-nv-las-vegas-strip'), 'vegas')
        self.assertEqual(core.city_of('sg-one-north'), 'singapore')
        self.assertEqual(core.city_of('elsewhere'), 'n/a')
        self.assertEqual(core.session_of('2021.05.25.14.16.10_veh-35_01690_02183'), '2021.05.25.14.16.10_veh-35')

    def test_feature_table_matches_preregistration(self):
        self.assertEqual(core.FEATURE_NAMES, ['city', 'speed', 'command', 'lead', 'junction', 'curvature',
                                              'lane_change', 'stop_go'])


class LeadRule(unittest.TestCase):
    """Same arithmetic as PrivilegedBrakeAgent._lead_gap_speed."""

    @staticmethod
    def box(x, y, length=4.0, width=2.0):
        return (x, y, 0.0, length, width, 1.5, 0.0)

    def test_gap_and_closest(self):
        boxes = [self.box(20.0, 0.0), self.box(10.0, 0.5)]
        gap, v, cls = core.lead_gap(boxes, ['vehicle', 'vehicle'], [(3.0, 0, 0), (1.5, 0, 0)])
        self.assertAlmostEqual(gap, 10.0 - 2.0 - 4.0)
        self.assertEqual((v, cls), (1.5, 'vehicle'))

    def test_corridor_limits(self):
        # vehicle half-width 2.3 + 0.25 * 2.0 = 2.8
        self.assertIsNone(core.lead_gap([self.box(10, 2.9)], ['vehicle'], [(0, 0, 0)])[0])
        self.assertIsNotNone(core.lead_gap([self.box(10, 2.8)], ['vehicle'], [(0, 0, 0)])[0])
        # pedestrians get at least 3.0 m
        self.assertIsNotNone(core.lead_gap([self.box(10, 2.9, 0.5, 0.5)], ['pedestrian'], [(0, 0, 0)])[0])
        self.assertIsNone(core.lead_gap([self.box(1.4, 0)], ['vehicle'], [(0, 0, 0)])[0])
        self.assertIsNone(core.lead_gap([self.box(55.1, 0)], ['vehicle'], [(0, 0, 0)])[0])
        self.assertIsNone(core.lead_gap([self.box(10, 0)], ['traffic_cone'], [(0, 0, 0)])[0])

    def test_negative_gap_is_close(self):
        gap, _, _ = core.lead_gap([self.box(3.0, 0.0)], ['bicycle'], [(0, 0, 0)])
        self.assertLess(gap, 0)
        self.assertEqual(core.lead_band(gap), 'close')


class Geometry(unittest.TestCase):
    def test_curvature(self):
        self.assertAlmostEqual(core.curvature_deg(straight_cl()), 0.0, places=6)
        # radius 30 m: 30 m of arc is one radian
        self.assertAlmostEqual(core.curvature_deg(arc_cl(30.0)), math.degrees(1.0), places=3)
        self.assertAlmostEqual(core.curvature_deg(arc_cl(30.0, left=False)), math.degrees(1.0), places=3)
        self.assertIsNone(core.curvature_deg([]))

    def test_curvature_unwraps_across_pi(self):
        rows = [(float(i), 0.0, math.cos(math.pi - 0.01 + 0.002 * i), math.sin(math.pi - 0.01 + 0.002 * i))
                for i in range(20)]
        self.assertLess(core.curvature_deg(rows), 5.0)

    def test_signed_offset_left_positive_and_extended(self):
        cl = [(r[0], r[1]) for r in straight_cl()]
        self.assertAlmostEqual(core.signed_offset((10.0, 1.5), cl), 1.5)
        self.assertAlmostEqual(core.signed_offset((10.0, -2.0), cl), -2.0)
        self.assertAlmostEqual(core.signed_offset((80.0, 3.0), cl), 3.0)  # beyond the end

    def test_short_route_repeated_end_points(self):
        # a 25-m route stored with its end repeated to fill the 20 samples
        rows = [(min(60.0 * i / 19, 25.0), 0.0, 1.0, 0.0) for i in range(20)]
        self.assertAlmostEqual(core.signed_offset((46.0, 1.0), [(r[0], r[1]) for r in rows]), 1.0)
        self.assertAlmostEqual(core.route_length(rows), 25.0)
        self.assertAlmostEqual(core.lane_change_delta(rows, (46.0, 0.5)), 0.5)

    def test_lane_change_delta(self):
        self.assertAlmostEqual(core.lane_change_delta(straight_cl(), (40.0, 3.5)), 3.5)
        self.assertEqual(core.lane_change_band(core.lane_change_delta(straight_cl(), (40.0, 3.5))), 'yes')
        self.assertEqual(core.lane_change_band(core.lane_change_delta(straight_cl(), (40.0, 0.7))), 'no')

    def test_logged_polyline_and_lateral_distance(self):
        logged = [(5.0 * i, 0.0, 0.0) for i in range(1, 9)]
        path = core.logged_polyline(logged)
        self.assertEqual(path[0], (0.0, 0.0))
        self.assertAlmostEqual(core.signed_offset((20.0, 1.2), path), 1.2)
        self.assertAlmostEqual(core.signed_offset((55.0, -0.4), path), -0.4)  # beyond the logged end
        # a standing ego: a ray straight ahead, so a plan that goes straight has no lateral error
        still = core.logged_polyline([(0.01 * i, 0.0, 0.0) for i in range(1, 9)])
        self.assertEqual(still, [(0.0, 0.0), (1.0, 0.0)])
        self.assertAlmostEqual(core.signed_offset((12.0, 0.0), still), 0.0)

    def test_helpers(self):
        self.assertEqual(core.cl_arc_lengths(3, 60.0), [0.0, 30.0, 60.0])
        self.assertAlmostEqual(core.interp(15.0, [0, 30], [0, 1]), 0.5)
        self.assertEqual(core.interp(-1, [0, 30], [2, 3]), 2)
        self.assertAlmostEqual(core.polyline_length([(0, 0), (3, 4), (3, 10)]), 11.0)
        self.assertAlmostEqual(core.heading_diff_deg(math.pi - 0.1, -math.pi + 0.1), math.degrees(0.2))


class Outcomes(unittest.TestCase):
    @staticmethod
    def row(nc=1.0, dac=1.0, ep=1.0, ttc=1.0, c=1.0):
        score = nc * dac * (5 * ep + 5 * ttc + 2 * c) / 12
        return {'no_at_fault_collisions': nc, 'drivable_area_compliance': dac, 'ego_progress': ep,
                'time_to_collision_within_bound': ttc, 'comfort': c, 'score': score}

    def test_indicators(self):
        o = core.outcomes(self.row())
        self.assertEqual(o['pdms'], 1.0)
        self.assertEqual(sum(v for k, v in o.items() if k != 'pdms'), 0)
        o = core.outcomes(self.row(nc=0.5, ttc=0.0))
        self.assertEqual((o['fail_nc'], o['fail_ttc'], o['fail_pdms']), (1.0, 1.0, 1.0))
        o = core.outcomes(self.row(dac=0.0, ep=0.0))
        self.assertEqual((o['fail_dac'], o['fail_ep'], o['fail_pdms']), (1.0, 0.0, 1.0))  # EP zeroed by DAC
        o = core.outcomes(self.row(ep=0.3))
        self.assertEqual((o['fail_ep'], o['fail_pdms']), (1.0, 0.0))
        o = core.outcomes(self.row(c=0.0))
        self.assertEqual(o['fail_c'], 1.0)


class Bootstrap(unittest.TestCase):
    def test_vendored_cluster_ci_known_cases(self):
        # the cases of tests/test_clustered.py on the controlled-study branch
        self.assertEqual(core.cluster_ci([.25] * 6, ['a'] * 2 + ['b'] * 4, 100), (.25, .25, .25))
        mean, lo, hi = core.cluster_ci([0, 1, 1, 1], ['a', 'b', 'b', 'b'], 1000)
        self.assertEqual((mean, lo, hi), (.75, 0, 1))
        self.assertEqual(core.cluster_ci([0, 0, 1, 1], ['a', 'a', 'b', 'b'], 1000), (.5, 0, 1))
        for values, groups in [([], []), ([1], ['a']), ([1, float('nan')], ['a', 'b']), ([1], ['a', 'b'])]:
            with self.assertRaises(ValueError):
                core.cluster_ci(values, groups, 100)

    def test_replay_matches_random_choices(self):
        for k in (2, 7, 44):
            rng = random.Random(core.SEED)
            want = [rng.choices(list(range(k)), k=k) for _ in range(50)]
            self.assertEqual(core.replay_indices(k, 50, core.SEED), want)

    def synthetic(self, n=400, seed=3):
        rnd = random.Random(seed)
        groups = ['log%02d' % rnd.randrange(23) for _ in range(n)]
        values = [rnd.random() + (0.3 if g < 'log10' else 0.0) for g in groups]
        return values, groups

    def test_pure_replay_equals_cluster_ci(self):
        values, groups = self.synthetic()
        want = core.cluster_ci(values, groups, 2000)
        self.assertEqual(core.mean_ci(values, groups, 2000, use_numpy=False), want)

    @unittest.skipIf(core._np() is None, 'numpy not installed')
    def test_numpy_replay_equals_cluster_ci(self):
        values, groups = self.synthetic(seed=5)
        want = core.cluster_ci(values, groups, 2000)
        got = core.mean_ci(values, groups, 2000, use_numpy=True)
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=12)

    def test_contribution_sums_to_effect(self):
        values, groups = self.synthetic(seed=7)
        mask = [i % 3 == 0 for i in range(len(values))]
        part = [v if m else 0.0 for v, m in zip(values, mask)]
        rest = [0.0 if m else v for v, m in zip(values, mask)]
        total = core.mean_ci(values, groups, 500, use_numpy=False)[0]
        a = core.mean_ci(part, groups, 500, use_numpy=False)[0]
        b = core.mean_ci(rest, groups, 500, use_numpy=False)[0]
        self.assertAlmostEqual(a + b, total, places=12)

    def test_diff_ci(self):
        rnd = random.Random(11)
        groups, values, in_a = [], [], []
        for i in range(600):
            g = 'g%02d' % (i % 30)
            a = i % 2 == 0
            groups.append(g)
            in_a.append(a)
            values.append((1.0 if a else 0.0) + rnd.gauss(0, 0.1))
        in_b = [not a for a in in_a]
        d, lo, hi, dropped = core.diff_ci(values, groups, in_a, in_b, 1000, use_numpy=False)
        self.assertAlmostEqual(d, 1.0, delta=0.05)
        self.assertTrue(lo < d < hi)
        self.assertEqual(dropped, 0)
        self.assertEqual(core.verdict(lo, hi), 'excludes 0')

    def test_diff_ci_drops_empty_draws(self):
        groups = ['g%d' % (i % 20) for i in range(200)]
        in_a = [g == 'g0' for g in groups]  # group A lives in one cluster of twenty
        in_b = [not a for a in in_a]
        values = [1.0 if a else 0.0 for a in in_a]
        _, _, _, dropped = core.diff_ci(values, groups, in_a, in_b, 1000, use_numpy=False)
        self.assertGreater(dropped, 200)  # about (19/20)^20 = 36% of draws miss g0
        self.assertLess(dropped, 500)

    def test_percentile_positions_and_verdict(self):
        draws = list(range(1000))
        self.assertEqual(core.percentile_ci(draws), (25, 975, 0))
        self.assertEqual(core.percentile_ci([None] * 950 + list(range(50))), (None, None, 950))
        self.assertEqual(core.verdict(-0.1, 0.2), 'spans 0')
        self.assertEqual(core.verdict(0.1, 0.2), 'excludes 0')
        self.assertEqual(core.verdict(None, 0.2), 'n/a')

    def test_wilson(self):
        lo, hi = core.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=4)
        lo, hi = core.wilson(12, 24)
        self.assertAlmostEqual(lo, 0.3143, places=4)
        self.assertAlmostEqual(hi, 0.6857, places=4)
        self.assertEqual(core.wilson(0, 0), (None, None))


class ReportColumns(unittest.TestCase):
    """The markdown tables print DISPLAY in the order their headers name."""

    def test_display_order_matches_headers(self):
        import failure_taxonomy_analyze as fa
        labels = [core.OUTCOME_LABEL[m] for m in fa.DISPLAY]
        self.assertEqual(labels, ['PDMS', 'PDMS<0.5', 'DAC fail', 'NC fail', 'TTC fail', 'EP<0.5', 'C fail'])
        self.assertEqual(sorted(fa.DISPLAY), sorted(core.OUTCOMES))

    def test_effect_weights(self):
        import failure_taxonomy_analyze as fa
        self.assertEqual(fa.effect_terms('E1', None), {'privmapkin': 1.0, 'kin': -1.0})
        e4 = fa.effect_terms('E4', 0)
        self.assertEqual(e4, {'l_pose_map_agents_s0': 1, 'l_pose_map_s0': -1, 'l_pose_agents_s0': -1,
                              'l_pose_kin_s0': 1})
        avg = fa.effect_terms('E3', 'avg')
        self.assertEqual(len(avg), 6)
        self.assertAlmostEqual(sum(avg.values()), 0.0)
        self.assertAlmostEqual(avg['l_pose_map_agents_s2'], 1 / 3)


class CommittedResults(unittest.TestCase):
    """The committed feature cache is complete and every committed point estimate (category means,
    effects, contributions) is what the per-scene CSVs give. Intervals need the bootstrap and are
    regenerated by `python analysis/failure_taxonomy_analyze.py`."""

    OUT = os.path.join(ROOT, 'results', 'failure_taxonomy')

    def test_feature_cache_complete_and_in_preregistered_bins(self):
        import csv
        import gzip
        import io
        path = os.path.join(self.OUT, 'features.csv.gz')
        with io.TextIOWrapper(gzip.open(path, 'rb'), encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 12146)
        self.assertEqual(len({r['token'] for r in rows}), 12146)
        self.assertEqual(len({r['log'] for r in rows}), 136)
        self.assertEqual(len({r['session'] for r in rows}), 44)
        allowed = dict(core.FEATURES)
        for r in rows:
            for feature, cats in allowed.items():
                self.assertIn(r[feature], cats + (core.NA,), (feature, r['token']))

    def test_point_estimates_reproduce(self):
        import failure_taxonomy_analyze as fa
        self.assertEqual(fa.check(self.OUT), 0)


class SameAsControlledStudyBranch(unittest.TestCase):
    """If analyze_clustered.py is present (after the controlled-study branch merges), the
    vendored copy must give identical results."""

    def test_identical_when_available(self):
        try:
            import analyze_clustered
        except Exception:
            self.skipTest('analysis/analyze_clustered.py not on this branch')
        values = [random.Random(1).random() for _ in range(300)]
        groups = ['g%d' % (i % 17) for i in range(300)]
        self.assertEqual(analyze_clustered.cluster_ci(values, groups, 1000),
                         core.cluster_ci(values, groups, 1000))


class RootCause(unittest.TestCase):
    @staticmethod
    def diag(**kw):
        d = dict(reproduced=True, fallback=False, nc=1.0, dac=1.0, ttc=1.0, c=1.0, ep=1.0,
                 nc_step=None, dac_step=None, ttc_step=None, dac_lateral_m=0.0, dac_travelled_m=0.0,
                 logged_length_m=20.0, logged_heading_change_deg=0.0, nc_object_type='vehicle',
                 nc_collision_type='active_front', nc_heading_diff_deg=5.0, nc_object_ahead=True,
                 ttc_heading_diff_deg=5.0, ttc_object_ahead=True)
        d.update(kw)
        return d

    def test_primary_failure_order(self):
        self.assertEqual(core.primary_failure(self.diag(dac=0.0, dac_step=12, nc=0.0, nc_step=8)), 'nc')
        self.assertEqual(core.primary_failure(self.diag(dac=0.0, dac_step=8, nc=0.0, nc_step=8)), 'dac')
        self.assertEqual(core.primary_failure(self.diag(ttc=0.0)), 'ttc')
        self.assertEqual(core.primary_failure(self.diag(c=0.0)), 'comfort')
        self.assertEqual(core.primary_failure(self.diag()), 'other')

    def test_dac_labels(self):
        d = self.diag(dac=0.0, dac_step=10, dac_lateral_m=1.5, logged_heading_change_deg=-40.0)
        self.assertEqual(core.root_cause(d), 'dac_lateral_curve')
        d['logged_heading_change_deg'] = 5.0
        self.assertEqual(core.root_cause(d), 'dac_lateral_straight')
        d.update(dac_lateral_m=0.4, dac_travelled_m=21.5)
        self.assertEqual(core.root_cause(d), 'dac_overshoot')
        d['dac_travelled_m'] = 20.5
        self.assertEqual(core.root_cause(d), 'dac_near_logged_path')

    def test_nc_labels(self):
        base = dict(nc=0.0, nc_step=5)
        self.assertEqual(core.root_cause(self.diag(**base)), 'nc_rear_end')
        self.assertEqual(core.root_cause(self.diag(nc=0.5, nc_step=5)), 'nc_static')
        self.assertEqual(core.root_cause(self.diag(nc_object_type='pedestrian', **base)), 'nc_vru')
        self.assertEqual(core.root_cause(self.diag(nc_collision_type='active_lateral', **base)), 'nc_lateral')
        self.assertEqual(core.root_cause(self.diag(nc_heading_diff_deg=80.0, **base)), 'nc_crossing')
        self.assertEqual(core.root_cause(self.diag(nc_object_ahead=False, **base)), 'nc_crossing')

    def test_ttc_and_overrides(self):
        self.assertEqual(core.root_cause(self.diag(ttc=0.0)), 'ttc_following')
        self.assertEqual(core.root_cause(self.diag(ttc=0.0, ttc_heading_diff_deg=60.0)), 'ttc_crossing')
        self.assertEqual(core.root_cause(self.diag(c=0.0)), 'comfort')
        self.assertEqual(core.root_cause(self.diag(dac=0.0, dac_step=3, fallback=True)), 'fallback')
        self.assertEqual(core.root_cause(self.diag(dac=0.0, dac_step=3, reproduced=False)), 'unreproduced')

    def test_plan_difference(self):
        self.assertEqual(core.plan_difference(10.0, 12.5, (10, 0), (12.5, 0), 0.0), 'slower')
        self.assertEqual(core.plan_difference(15.0, 12.5, (15, 0), (12.5, 0), 0.0), 'faster')
        self.assertEqual(core.plan_difference(12.0, 12.5, (12, 0.8), (12.5, 0), 0.0), 'lateral')
        self.assertEqual(core.plan_difference(12.0, 12.5, (12, 0.1), (12.5, 0), 0.0), 'similar')

    def test_strata_membership(self):
        self.assertTrue(core.stratum_member('S1', 0.2, 0.9))
        self.assertFalse(core.stratum_member('S1', 0.2, 0.3))
        self.assertTrue(core.stratum_member('S3', 0.1, 0.8))
        self.assertTrue(core.stratum_member('S4', 0.49, None))
        self.assertFalse(core.stratum_member('S5', 0.5, None))
        with self.assertRaises(ValueError):
            core.stratum_member('S9', 0, 0)

    def test_select_cases_one_per_log_and_deterministic(self):
        token_log = {'t%03d' % i: 'log%02d' % (i % 10) for i in range(100)}
        cands = {0: sorted(token_log)[:60], 1: sorted(token_log)[40:], 2: sorted(token_log)}
        first = core.select_cases('S3', cands, token_log, 3)
        self.assertEqual(first, core.select_cases('S3', cands, token_log, 3))
        self.assertEqual(len(first), 9)
        self.assertEqual(len({c['log'] for c in first}), 9)
        self.assertEqual([c['seed'] for c in first], [0] * 3 + [1] * 3 + [2] * 3)
        ranks = [c['hash'] for c in first if c['seed'] == 0]
        self.assertEqual(ranks, sorted(ranks))
        single = core.select_cases('S1', {None: sorted(token_log)}, token_log, 24)
        self.assertEqual(len(single), 10)  # only ten logs exist


if __name__ == '__main__':
    unittest.main()
