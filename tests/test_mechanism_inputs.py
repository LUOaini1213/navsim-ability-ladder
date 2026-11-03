import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
try:
    import numpy as np
    from audit_mechanism_inputs import donor_diagnostics
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires optional numpy")
class MechanismInputs(unittest.TestCase):
    def setUp(self):
        self.data = {"log_name": np.array(["driveA_0_1", "driveB_0_1", "driveC_0_1"]),
                     "map_name": np.array(["left", "right", "left"]), "cmd": np.array([1, 1, 1]),
                     "speed": np.array([3., 4., 4.5]), "agents": np.zeros((3, 8, 11))}
        self.data["agents"][:, 0, 10] = 1

    def test_cross_city_matching_is_disclosed_not_assumed(self):
        row = donor_diagnostics(self.data, [0, 1, 2], [1, 2, 0])
        self.assertAlmostEqual(row["same_city_fraction"], 1/3)
        self.assertEqual(row["same_speed_bin_fraction"], 1.)
        self.assertEqual(row["different_session_fraction"], 1.)

    def test_partition_or_session_violation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            donor_diagnostics(self.data, [0, 1], [1, 2])
        with self.assertRaisesRegex(ValueError, "Same-session"):
            donor_diagnostics(self.data, [0, 1, 2], [0, 2, 1])


if __name__ == "__main__":
    unittest.main()
