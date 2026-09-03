# -*- coding: utf-8 -*-
"""The paired-bootstrap routine behind every confidence interval in results/."""
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))

from ladder_io import bootstrap  # noqa: E402


class Bootstrap(unittest.TestCase):
    def test_constant_shift_is_recovered_with_a_tight_interval(self):
        diffs = [0.1] * 200
        mean, lo, hi = bootstrap(diffs, seed=1, b=2000)
        self.assertAlmostEqual(mean, 0.1)
        self.assertAlmostEqual(lo, 0.1)
        self.assertAlmostEqual(hi, 0.1)

    def test_noise_around_a_shift_brackets_the_shift(self):
        rnd = random.Random(0)
        diffs = [0.05 + rnd.gauss(0, 0.3) for _ in range(563)]
        mean, lo, hi = bootstrap(diffs, seed=1, b=2000)
        self.assertLess(lo, 0.05)
        self.assertGreater(hi, 0.05)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertLess(hi - lo, 0.12)  # ~ 2 * 1.96 * 0.3 / sqrt(563)

    def test_pure_noise_spans_zero(self):
        rnd = random.Random(3)
        diffs = [rnd.gauss(0, 0.2) for _ in range(300)]
        _mean, lo, hi = bootstrap(diffs, seed=1, b=2000)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_same_seed_same_interval(self):
        rnd = random.Random(5)
        diffs = [rnd.gauss(0.02, 0.1) for _ in range(100)]
        self.assertEqual(bootstrap(diffs, seed=12345, b=500), bootstrap(diffs, seed=12345, b=500))
        self.assertNotEqual(bootstrap(diffs, seed=12345, b=500), bootstrap(diffs, seed=7, b=500))


if __name__ == '__main__':
    unittest.main()
