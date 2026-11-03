import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from plot_mechanism import layout_fixture, PRIMARY


class MechanismPlot(unittest.TestCase):
    def test_fixture_is_explicitly_synthetic(self):
        report = layout_fixture()
        self.assertTrue(report["layout_fixture"])
        self.assertIn("SMOKE", report["split"])
        self.assertEqual(set(report["summary"]["contrasts"]), set(PRIMARY))

    def test_fixture_exercises_nonestimable_and_empty_rows(self):
        groups = layout_fixture()["summary"]["subgroups"]
        self.assertEqual(groups["empty"]["n_scenes"], 0)
        self.assertEqual(groups["empty"]["contrasts"], {})
        self.assertEqual(groups["city:sg-one-north"]["n_sessions"], 1)
        self.assertTrue(all(row["session_interval"]["low"] is None for row in groups["city:sg-one-north"]["contrasts"].values()))


if __name__ == "__main__":
    unittest.main()
