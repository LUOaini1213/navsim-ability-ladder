import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
try:
    import numpy as np
    from mechanism_cases import box_corners, choose_cases
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires optional numpy")
class MechanismCases(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "results/mechanism_v2/protocol/diagnostics.json").read_text(encoding="utf-8"))
        self.policy["contrasts"] = ["M1-M0"]
        self.tokens = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
        self.logs = np.array([f"drive{j//2}_0_1" for j in range(8)])
        self.arrays = {}
        for seed in self.policy["seeds"]:
            self.arrays[f"M0_f1_steps_s{seed}"] = np.full((8, 7), .5)
            value = np.full((8, 7), .5)
            value[:, 0] += [-.1, -.4, .1, .4, 0., 0., .03, -.03]
            self.arrays[f"M1_f1_steps_s{seed}"] = value

    def test_deterministic_and_input_order_independent(self):
        report = choose_cases(self.arrays, self.tokens, self.logs, self.policy, "validation")
        order = np.arange(7, -1, -1)
        reverse = choose_cases({key: value[order] for key, value in self.arrays.items()},
                               self.tokens[order], self.logs[order], self.policy, "validation")
        self.assertEqual(report, reverse)

    def test_session_limit_and_shortfall_no_threshold_relaxation(self):
        report = choose_cases(self.arrays, self.tokens, self.logs, self.policy, "validation")
        self.assertEqual(len(report["cases"]), 3)
        self.assertEqual(len({case["session"] for case in report["cases"]}), 3)
        self.assertTrue(all(row["shortfall"] == 1 for row in report["counts"]["M1-M0"].values()))
        self.assertFalse({"g", "h"} & {case["token"] for case in report["cases"]})

    def test_averages_scores_not_best_seed(self):
        for seed, effect in enumerate([-.3, .1, .2]):
            self.arrays[f"M1_f1_steps_s{seed}"][:, 0] = .5+effect
        report = choose_cases(self.arrays, self.tokens, self.logs, self.policy, "validation")
        self.assertEqual(report["counts"]["M1-M0"]["positive"]["candidate_scenes"], 0)
        self.assertEqual(report["counts"]["M1-M0"]["negative"]["candidate_scenes"], 0)
        self.assertEqual(report["counts"]["M1-M0"]["neutral"]["candidate_scenes"], 8)

    def test_empty_bins_are_not_replaced(self):
        for seed in self.policy["seeds"]:
            self.arrays[f"M1_f1_steps_s{seed}"][:, 0] = .53
        report = choose_cases(self.arrays, self.tokens, self.logs, self.policy, "confirmation")
        self.assertEqual(report["cases"], [])
        self.assertTrue(all(row["shortfall"] == 2 for row in report["counts"]["M1-M0"].values()))

    def test_box_rotation(self):
        obj = np.array([10, 20, 0, 0, 4, 2, 0, 1, 1, 0, 1])
        corners = box_corners(obj)
        np.testing.assert_array_equal(corners.mean(0), [10, 20])
        np.testing.assert_array_equal(corners.max(0)-corners.min(0), [2, 4])


if __name__ == "__main__":
    unittest.main()
