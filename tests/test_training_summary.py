import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from summarize_mechanism_training import distribution, markdown


class TrainingSummary(unittest.TestCase):
    def test_seed_range_not_confidence_interval(self):
        row = distribution([.5, .6, .7])
        self.assertAlmostEqual(row["mean"], .6)
        self.assertEqual(row["seeds"], {"0": .5, "1": .6, "2": .7})
        self.assertNotIn("ci", row)

    def test_incomplete_and_nonfinite_refused(self):
        for values in ([.1, .2], [.1, .2, math.nan], [.1, -.2, .3]):
            with self.assertRaises(ValueError):
                distribution(values)

    def test_rendered_table_has_scope_limits(self):
        table = markdown({"scope": "No PDMS result", "n_runs": 18, "optimizer_steps_per_run": 13000,
                          "parameter_ratio": 1.08, "summary": {}, "limitations": ["Not end-to-end latency"]})
        self.assertIn("not confidence intervals", table)
        self.assertIn("No PDMS result", table)
        self.assertIn("Not end-to-end latency", table)


if __name__ == "__main__":
    unittest.main()
