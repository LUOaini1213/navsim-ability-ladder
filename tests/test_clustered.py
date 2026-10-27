import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from analyze_clustered import cluster_ci, contrast


class ClusterBootstrap(unittest.TestCase):
    def test_constant_effect(self):
        self.assertEqual(cluster_ci([.25]*6, ['a']*2+['b']*4, 100), (.25, .25, .25))

    def test_scene_weighted_not_log_weighted(self):
        mean, lo, hi = cluster_ci([0, 1, 1, 1], ['a', 'b', 'b', 'b'], 1000)
        self.assertEqual(mean, .75)
        self.assertEqual((lo, hi), (0, 1))

    def test_aligned_cluster_and_reproducible(self):
        first = cluster_ci([0, 0, 1, 1], ['a', 'a', 'b', 'b'], 1000)
        self.assertEqual(first, cluster_ci([0, 0, 1, 1], ['a', 'a', 'b', 'b'], 1000))
        self.assertEqual(first, (.5, 0, 1))

    def test_invalid(self):
        for values, groups in [([], []), ([1], ['a']), ([1, float('nan')], ['a', 'b']), ([1], ['a', 'b'])]:
            with self.assertRaises(ValueError):
                cluster_ci(values, groups, 100)

    def test_interaction_sign(self):
        effect = contrast(contrast({'k': 1}, {'a': 1}), contrast({'m': 1}, {'ma': 1}))
        self.assertEqual(effect, {'k': 1, 'a': -1, 'm': -1, 'ma': 1})
