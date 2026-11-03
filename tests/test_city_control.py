import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from prepare_city_control import freeze_design
try:
    import numpy as np
    from prepare_city_control import city_donors, supported_indices
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires optional numpy")
class CityControl(unittest.TestCase):
    def fixture(self):
        return {"map_name": np.array(["A", "A", "A", "A", "B"]),
                "log_name": np.array(["drive0_0_1", "drive0_2_3", "drive1_0_1", "drive1_2_3", "drive2_0_1"]),
                "cmd": np.array([1, 1, 1, 1, 2]), "speed": np.array([3., 3.5, 4., 4.5, 10.]),
                "agents": np.zeros((5, 8, 11))}

    def test_single_session_city_excluded_not_relaxed(self):
        data = self.fixture()
        indices, omitted = supported_indices(data, np.arange(5))
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])
        self.assertEqual(omitted["B"]["n_scenes"], 1)
        with self.assertRaisesRegex(ValueError, "same-city"):
            city_donors(data, np.arange(5), 7)

    def test_full_matching_is_deterministic_and_cross_session(self):
        data = self.fixture()
        indices, _ = supported_indices(data, np.arange(5))
        donors, levels = city_donors(data, indices, 7)
        np.testing.assert_array_equal(donors, city_donors(data, indices, 7)[0])
        self.assertEqual(levels, {"0": 4})
        np.testing.assert_array_equal(data["map_name"][indices], data["map_name"][donors])
        self.assertTrue(all(data["log_name"][a].split("_")[0] != data["log_name"][b].split("_")[0] for a, b in zip(indices, donors)))

    def test_fallback_preserves_city_and_session(self):
        data = self.fixture()
        data["cmd"][2:4] = 3
        donors, levels = city_donors(data, [0, 1, 2, 3], 9)
        self.assertEqual(levels, {"3": 4})
        self.assertTrue(all(value in [2, 3] for value in donors[:2]))
        self.assertTrue(all(value in [0, 1] for value in donors[2:]))

    def test_matching_never_accesses_future_keys(self):
        class CurrentOnly(dict):
            def __getitem__(self, key):
                if key.startswith("y_") or "future" in key:
                    raise AssertionError("Future accessed")
                return super().__getitem__(key)
        data = CurrentOnly(self.fixture())
        city_donors(data, [0, 1, 2, 3], 9)


class CityDesignBinding(unittest.TestCase):
    def test_repeated_binding_preserves_time_and_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.json"
            base.write_text("{}", encoding="utf-8")
            path = freeze_design(root / "city", base, root / "results")
            original = path.read_bytes()
            self.assertEqual(freeze_design(root / "city", base, root / "results").read_bytes(), original)
            base.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after binding"):
                freeze_design(root / "city", base, root / "results")

    def test_complete_core_report_prevents_false_preoutcome_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.json"
            base.write_text("{}", encoding="utf-8")
            report = root / "results/analysis/validation/report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot falsely bind"):
                freeze_design(root / "city", base, root / "results")


if __name__ == "__main__":
    unittest.main()
