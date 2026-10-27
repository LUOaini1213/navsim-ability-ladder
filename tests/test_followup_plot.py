import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from plot_followups import CITY_CONTRASTS, CURVE_CELLS, FIGURES, effect_text, layout_fixture, verify_report


class FollowupPlot(unittest.TestCase):
    def fixture_report(self, kind):
        # Rebuild callbacks are unit fixtures only; no scientific files/figures are produced.
        report = layout_fixture(kind)
        report.update(layout_fixture=False, split="validation", analysis_source_sha256={"fixture": "fixture"})
        if kind == "city":
            names = {f"C{number}_f1_steps_s{seed}" for number in (1, 2, 4, 5) for seed in (0, 1, 2)}
        else:
            names = {f"{config}_f{fraction:g}_{budget}_s{seed}" for config in ("M1", "M4") for seed in (0, 1, 2)
                     for fraction, budget in ((.1, "steps"), (.3, "steps"), (.1, "epochs"), (.3, "epochs"), (1., "steps"))}
        report["provenance"] = {name: {"fixture": True} for name in names}
        return report

    def test_all_seven_figure_layouts_explicitly_ineligible(self):
        self.assertEqual(sum(map(len, FIGURES.values())), 7)
        for kind in FIGURES:
            report = layout_fixture(kind)
            self.assertTrue(report["layout_fixture"])
            self.assertIn("SMOKE", report["split"])
            callback = Mock()
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                verify_report(kind, report, callback)
            callback.assert_not_called()

    def test_city_empty_and_one_session_strata_retained(self):
        summary = layout_fixture("city")["summary"]
        self.assertEqual(set(summary["contrasts"]), set(CITY_CONTRASTS))
        self.assertEqual(summary["subgroups"]["empty"]["contrasts"], {})
        self.assertTrue(all(row["session_interval"]["low"] is None for row in summary["subgroups"]["city:single-session"]["contrasts"].values()))

    def test_smoke_exercises_both_interval_widths_not_just_one_line(self):
        for row in layout_fixture("city")["summary"]["contrasts"].values():
            self.assertLess(row["auxiliary_family_interval"]["low"], row["session_interval"]["low"])
            self.assertGreater(row["auxiliary_family_interval"]["high"], row["session_interval"]["high"])
        for row in layout_fixture("curves")["summary"]["small_minus_full_encoder_effect"].values():
            self.assertLess(row["four_contrast_interval"]["low"], row["session_interval"]["low"])
            self.assertGreater(row["four_contrast_interval"]["high"], row["session_interval"]["high"])

    def test_heatmap_labels_round_near_zero_without_false_sign(self):
        self.assertEqual(effect_text(-0.), "0.000")
        self.assertEqual(effect_text(-.00001), "0.000")
        self.assertEqual(effect_text(.01), "+0.010")
        self.assertEqual(effect_text(-.01), "-0.010")

    def test_both_budgets_reuse_identical_full_anchor(self):
        cells = layout_fixture("curves")["summary"]["cells"]
        self.assertEqual(set(cells), set(CURVE_CELLS))
        self.assertEqual(cells["steps/f1"]["models"], cells["epochs/f1"]["models"])
        self.assertTrue(cells["steps/f1"]["reused_full_data_anchor"])
        self.assertTrue(cells["epochs/f1"]["reused_full_data_anchor"])

    def test_production_preflight_calls_source_rebuild_before_accepting(self):
        for kind in FIGURES:
            report = self.fixture_report(kind)
            callback = Mock(return_value=copy.deepcopy(report))
            self.assertEqual(verify_report(kind, report, callback), report)
            callback.assert_called_once_with()
            callback = Mock(side_effect=ValueError("actual artifact audit failed"))
            with self.assertRaisesRegex(ValueError, "actual artifact audit failed"):
                verify_report(kind, report, callback)

    def test_stale_or_tampered_summary_and_missing_cells_rejected(self):
        for kind in FIGURES:
            report = self.fixture_report(kind)
            changed = copy.deepcopy(report)
            changed["n_sessions"] += 1
            with self.assertRaisesRegex(ValueError, "rebuilt from actual artifacts"):
                verify_report(kind, changed, lambda: report)
            changed = copy.deepcopy(report)
            del changed["provenance"][next(iter(changed["provenance"]))]
            callback = Mock()
            with self.assertRaisesRegex(ValueError, "provenance/coverage"):
                verify_report(kind, changed, callback)
            callback.assert_not_called()

    def test_nonfinite_report_cannot_enter_rendering(self):
        report = self.fixture_report("city")
        report["summary"]["contrasts"]["C1-C2"]["metric_effects"]["score"] = float("nan")
        callback = Mock()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            verify_report("city", report, callback)
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
