import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from audit_city_inputs import audit_mapping


class CityInputAudit(unittest.TestCase):
    def fixture(self):
        return {"map_name": np.array(["a", "a", "b"]), "token": np.array(["0", "1", "2"]),
                "log_name": np.array(["2021.01.01.00.00.00_veh-a_0_1", "2021.01.02.00.00.00_veh-b_0_1", "2021.01.03.00.00.00_veh-c_0_1"])}

    def test_explicit_unsupported_city_exclusion(self):
        report = audit_mapping(self.fixture(), [0, 1], [1, 0], [0, 1, 2])
        self.assertEqual(report["omitted_scenes"], 1)
        self.assertEqual(report["same_city_fraction"], 1.)

    def test_same_city_and_other_session_required(self):
        with self.assertRaisesRegex(ValueError, "Same-city"):
            audit_mapping(self.fixture(), [0, 1, 2], [1, 2, 0], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "Different-session"):
            audit_mapping(self.fixture(), [0, 1], [0, 1], [0, 1, 2])

    def test_coverage_or_donor_partition_cannot_be_relaxed(self):
        with self.assertRaises(ValueError):
            audit_mapping(self.fixture(), [0, 1], [2, 0], [0, 1, 2])
        with self.assertRaises(ValueError):
            audit_mapping(self.fixture(), [1, 0], [0, 1], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
