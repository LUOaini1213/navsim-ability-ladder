import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from city_followup import decide, freeze, verify_gate
from verify_mechanism_core import digest
import test_curve_runner as curve_fixtures


class CityFollowup(unittest.TestCase):
    def fixture(self, folder):
        base = folder / "base/protocol.json"
        city = folder / "city/protocol.json"
        base.parent.mkdir()
        city.parent.mkdir()
        base.write_text('{"fixture":"base"}', encoding="utf-8")
        city.write_text('{"fixture":"city"}', encoding="utf-8")
        (base.parent / "followup_policy.json").write_text('{"minimum_absolute_pdms_effect":0.01}', encoding="utf-8")
        return folder / "city/execution_policy.json", base, city, folder / "results"

    def test_positive_and_negative_practical_effects_trigger(self):
        for effects, wanted in (({"M1-M2": .01, "M4-M5": 0.}, ["M1-M2"]),
                                ({"M1-M2": 0., "M4-M5": -.01}, ["M4-M5"]),
                                ({"M1-M2": -.02, "M4-M5": .02}, ["M1-M2", "M4-M5"])):
            with self.subTest(effects=effects):
                gate = decide(effects)
                self.assertTrue(gate["required"])
                self.assertEqual(gate["trigger_contrasts"], wanted)

    def test_small_effect_withholds_clean_alignment_claim(self):
        gate = decide({"M1-M2": .009, "M4-M5": -.009})
        self.assertFalse(gate["required"])
        self.assertTrue(gate["clean_alignment_claim_withheld_without_control"])

    def test_invalid_missing_effects_and_threshold_cannot_change(self):
        for effects in ({"M4-M5": .1}, {"M1-M2": float("nan"), "M4-M5": 0.}, {"M1-M2": 2., "M4-M5": 0.}):
            with self.assertRaises(ValueError):
                decide(effects)
        with self.assertRaises(ValueError):
            decide({"M1-M2": .01, "M4-M5": 0.}, threshold=.005)

    def test_freeze_idempotent_unchanged_then_refuses_input_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            first = freeze(*paths)
            self.assertEqual(freeze(*paths), first)
            self.assertIn("not an original-plan", first["scope"])
            paths[2].write_text('{"fixture":"changed"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after freezing"):
                freeze(*paths)

    def test_cannot_retroactively_freeze_after_a_complete_report(self):
        for split in ("validation", "confirmation", "navtest"):
            with tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                report = paths[3] / "analysis" / split / "report.json"
                report.parent.mkdir(parents=True)
                report.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "after results exist"):
                    freeze(*paths)
                self.assertFalse(paths[0].exists())

    def test_evaluation_never_auto_creates_a_missing_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "must already be frozen"):
                verify_gate(*paths, {})

    def actual_csv_fixture(self, folder):
        helper = curve_fixtures.CurveRunner()
        helper.setUp()
        report_path, scores, gate_path, original_policy, core = helper.gate_fixture(folder)
        base = folder / "protocol.json"
        city = folder / "city/protocol.json"
        city.parent.mkdir()
        city.write_text('{"fixture":"city"}', encoding="utf-8")
        shutil.copyfile(original_policy, base.parent / "followup_policy.json")
        results = folder / "results"
        policy = city.parent / "execution_policy.json"
        freeze(policy, base, city, results)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        effect = .50 - .49
        report["summary"]["contrasts"]["M1-M2"] = {"primary": False, "metric_effects": {"score": effect},
                                                        "seed_effects": {str(seed): effect for seed in (0, 1, 2)}}
        report_path = results / "analysis/validation/report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        shutil.copytree(scores, results / "scores/validation")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate["validation_report_sha256"] = digest(report_path)
        (results / "gate_validation.json").write_text(json.dumps(gate), encoding="utf-8")
        return policy, base, city, results, core

    def test_city_gate_recomputes_both_effects_from_all_core_csvs(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.actual_csv_fixture(Path(temporary))
            gate = verify_gate(*paths)
            self.assertTrue(gate["required"])
            self.assertAlmostEqual(gate["validation_effects"]["M1-M2"], .01)
            self.assertAlmostEqual(gate["validation_effects"]["M4-M5"], .02)

    def test_resealed_secondary_report_cannot_replace_actual_csv_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.actual_csv_fixture(Path(temporary))
            report_path = paths[3] / "analysis/validation/report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["summary"]["contrasts"]["M1-M2"]["metric_effects"]["score"] = .03
            report_path.write_text(json.dumps(report), encoding="utf-8")
            gate_path = paths[3] / "gate_validation.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["validation_report_sha256"] = digest(report_path)
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "City gate effect differs"):
                verify_gate(*paths)


if __name__ == "__main__":
    unittest.main()
