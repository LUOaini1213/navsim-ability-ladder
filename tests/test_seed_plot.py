import copy
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import plot_seed
import run_seed_figures
from run_seed_figures import figure_commands


class SeedPlot(unittest.TestCase):
    def report(self):
        # A synthetic unit fixture with a mock rebuild is NOT scientific provenance.
        report = plot_seed.layout_fixture()
        report.update(layout_fixture=False, schema="gated-five-seed-analysis-v1", split="validation",
                      paired_seeds=list(plot_seed.SEEDS), reused_core_runs=18, added_runs=12,
                      gate={"required": True}, analysis_source_sha256={"unit": "fixture"},
                      base_protocol_sha256="base-fixture", extension_protocol_sha256="extension-fixture")
        report["provenance"] = {f"{config}_f1_steps_s{seed}": {"training_protocol_sha256":
            "base-fixture" if seed < 3 else "extension-fixture"} for config in plot_seed.CONFIGS for seed in plot_seed.SEEDS}
        return report

    @staticmethod
    def fake_render(report, out):
        # Only byte-ownership tests: these placeholders are NOT actual image formats.
        for name in plot_seed.FIGURES:
            for suffix in ("png", "svg"):
                (out / (name + "." + suffix)).write_bytes(b"unit placeholder - not a scientific image")

    def test_exact_four_figure_scope_and_all_five_seed_effects(self):
        report = plot_seed.layout_fixture()
        self.assertEqual(len(plot_seed.FIGURES), 4)
        self.assertEqual(len(report["summary"]["contrasts"]), 7)
        for row in report["summary"]["contrasts"].values():
            self.assertEqual(set(row["seed_effects"]), {"0", "1", "2", "3", "4"})
            self.assertEqual(set(row["metric_effects"]), set(plot_seed.METRICS))

    def test_synthetic_layout_rejected_before_source_rebuild(self):
        callback = Mock()
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            plot_seed.verify_report(plot_seed.layout_fixture(), callback)
        callback.assert_not_called()

    def test_source_rebuild_is_mandatory_and_failure_propagates(self):
        report = self.report()
        callback = Mock(return_value=copy.deepcopy(report))
        self.assertEqual(plot_seed.verify_report(report, callback), report)
        callback.assert_called_once_with()
        with self.assertRaisesRegex(ValueError, "actual model audit failed"):
            plot_seed.verify_report(report, Mock(side_effect=ValueError("actual model audit failed")))

    def test_stale_statistics_rejected_even_with_complete_provenance(self):
        report = self.report()
        changed = copy.deepcopy(report)
        changed["summary"]["contrasts"]["M1-M0"]["seed_effects"]["4"] += .1
        with self.assertRaisesRegex(ValueError, "rebuilt from actual artifacts"):
            plot_seed.verify_report(changed, lambda: report)

    def test_partial_cells_seeds_gate_and_coverage_rejected_early(self):
        for mutation in ("run", "seeds", "gate", "count", "sources", "counts"):
            report = self.report()
            if mutation == "run":
                del report["provenance"]["M0_f1_steps_s4"]
            elif mutation == "seeds":
                report["paired_seeds"] = [0, 1, 2]
            elif mutation == "gate":
                report["gate"]["required"] = False
            elif mutation == "count":
                report["n_sessions"] = True
            elif mutation == "sources":
                report["analysis_source_sha256"] = {}
            else:
                report["added_runs"] = 6
            callback = Mock()
            with self.assertRaisesRegex(ValueError, "provenance/coverage/gate"):
                plot_seed.verify_report(report, callback)
            callback.assert_not_called()

    def test_original_protocol_must_not_be_relabelled(self):
        report = self.report()
        report["provenance"]["M0_f1_steps_s0"]["training_protocol_sha256"] = "extension-fixture"
        with self.assertRaisesRegex(ValueError, "relabelled"):
            plot_seed.verify_report(report, Mock())
        report = self.report()
        report["extension_protocol_sha256"] = report["base_protocol_sha256"]
        with self.assertRaisesRegex(ValueError, "Distinct"):
            plot_seed.verify_report(report, Mock())

    def test_all_seven_cell_metrics_and_five_seed_metrics_required(self):
        for mutation in ("cell", "metric", "seed"):
            report = self.report()
            if mutation == "cell":
                del report["summary"]["cells"]["M5"]
            elif mutation == "metric":
                del report["summary"]["cells"]["M0"]["seed_metrics"]["4"]["comfort"]
            else:
                del report["summary"]["cells"]["M0"]["seed_metrics"]["4"]
            with self.assertRaises(ValueError):
                plot_seed.verify_report(report, Mock())

    def test_primary_adjusted_family_and_secondary_identity_not_changed(self):
        for mutation in ("alpha", "secondary", "contrast", "seed"):
            report = self.report()
            rows = report["summary"]["contrasts"]
            if mutation == "alpha":
                rows["M1-M0"]["multiplicity_interval"]["alpha"] = .05
            elif mutation == "secondary":
                rows["M1-M2"]["primary"] = True
            elif mutation == "contrast":
                del rows["M5-M2"]
            else:
                del rows["M1-M0"]["seed_effects"]["4"]
            with self.assertRaises(ValueError):
                plot_seed.verify_report(report, Mock())

    def test_nonfinite_statistic_rejected_before_any_rebuild(self):
        report = self.report()
        report["summary"]["cells"]["M0"]["mean"]["score"] = float("nan")
        callback = Mock()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            plot_seed.verify_report(report, callback)
        callback.assert_not_called()

    def test_descriptive_subset_means_do_not_drop_added_seed(self):
        row = {"seed_effects": {"0": .01, "1": .02, "2": .03, "3": -.1, "4": .2}}
        original, added, all_five = plot_seed.grouped_effects(row)
        self.assertAlmostEqual(original, .02)
        self.assertAlmostEqual(added, .05)
        self.assertAlmostEqual(all_five, .032)

    def test_empty_and_single_session_groups_are_retained(self):
        groups = plot_seed.layout_fixture()["summary"]["subgroups"]
        self.assertEqual(groups["empty"]["contrasts"], {})
        self.assertEqual(groups["empty"]["n_logs"], 0)
        self.assertTrue(all(row["session_interval"]["low"] is None for row in groups["city:single-session"]["contrasts"].values()))

    def test_labels_do_not_show_signed_zero(self):
        self.assertEqual(plot_seed.effect_text(-.00001), "0.000")
        self.assertEqual(plot_seed.effect_text(.01), "+0.010")

    def test_completed_byte_receipt_reuses_without_rerender(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "figures"
            renderer = Mock(side_effect=self.fake_render)
            receipt = plot_seed.write_figures(plot_seed.layout_fixture(), out, None, renderer)
            self.assertFalse(receipt["eligible_for_paper"])
            self.assertEqual(len(receipt["figures"]), 8)
            self.assertEqual(receipt["visual_qa_status"], "pending inspection")
            self.assertEqual(plot_seed.write_figures(plot_seed.layout_fixture(), out, None, renderer), receipt)
            renderer.assert_called_once()

    def test_changed_render_source_or_figure_bytes_are_not_overwritten(self):
        for mutation in ("figure", "settings", "receipt"):
            with tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary) / "figures"
                report = plot_seed.layout_fixture()
                plot_seed.write_figures(report, out, None, self.fake_render)
                if mutation == "figure":
                    (out / (plot_seed.FIGURES[0] + ".png")).write_bytes(b"changed placeholder")
                else:
                    path = out / (mutation + ".json")
                    content = json.loads(path.read_text(encoding="utf-8"))
                    content["renderer_source_sha256"] = "changed"
                    path.write_text(json.dumps(content), encoding="utf-8")
                before = {path.name: path.read_bytes() for path in out.iterdir()}
                with self.assertRaises(ValueError):
                    plot_seed.write_figures(report, out, None, Mock())
                self.assertEqual({path.name: path.read_bytes() for path in out.iterdir()}, before)

    def test_partial_render_kept_and_not_retried_automatically(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "figures"
            with self.assertRaisesRegex(RuntimeError, "unit interrupted"):
                plot_seed.write_figures(plot_seed.layout_fixture(), out, None, Mock(side_effect=RuntimeError("unit interrupted")))
            self.assertTrue((out / "settings.json").exists())
            self.assertFalse((out / "receipt.json").exists())
            callback = Mock()
            with self.assertRaisesRegex(ValueError, "partial"):
                plot_seed.write_figures(plot_seed.layout_fixture(), out, None, callback)
            callback.assert_not_called()

    def test_source_changed_during_render_has_no_completion_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "figures"
            with self.assertRaisesRegex(ValueError, "no completion receipt"):
                plot_seed.write_figures(plot_seed.layout_fixture(), out, None, self.fake_render, source_check=lambda: False)
            self.assertFalse((out / "receipt.json").exists())

    def test_nonempty_unrelated_folder_and_unbound_report_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            keep = out / "user-note.txt"
            keep.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrelated"):
                plot_seed.write_figures(plot_seed.layout_fixture(), out, None, Mock())
            self.assertEqual(keep.read_text(encoding="utf-8"), "preserve")
            with self.assertRaisesRegex(ValueError, "source report hash"):
                plot_seed.write_figures(self.report(), out, None, Mock())

    def test_memory_guard_precedes_scientific_import_rebuild_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "no-output"
            before = set(sys.modules)
            with patch.object(plot_seed, "available_mb", return_value=1422.4):
                with self.assertRaisesRegex(RuntimeError, "6000 MB before imports/rebuild"):
                    plot_seed.main(["--report", "nonexistent-scientific-report.json", "--out", str(out)])
            self.assertFalse(out.exists())
            self.assertFalse({"matplotlib", "numpy", "torch", "analyze_seed"} & (set(sys.modules)-before))

    def test_smoke_cannot_consume_scientific_report_or_escape_tmp(self):
        for options in (("--out", "results/not-smoke"), ("--report", "actual-report.json", "--out", str(plot_seed.ROOT / "tmp/smoke"))):
            with patch.object(plot_seed, "available_mb") as memory:
                with self.assertRaisesRegex(ValueError, "tmp-only"):
                    plot_seed.main(["--layout-smoke", *options])
                memory.assert_not_called()

    def test_all_three_split_commands_and_fresh_output_version(self):
        workspace, results = Path("workspace"), Path("results")
        rows = figure_commands("python", workspace, results)
        self.assertEqual([split for split, _ in rows], ["validation", "confirmation", "navtest"])
        for split, command in rows:
            command = list(map(str, command))
            self.assertNotIn("--layout-smoke", command)
            self.assertEqual(command[command.index("--report")+1], str(results / "seed_analysis" / split / "report.json"))
            self.assertEqual(command[-1], str(results / "seed_figures" / split))
        self.assertEqual(figure_commands("python", workspace, results, Path("fresh-version"))[0][1][-1], Path("fresh-version/validation"))

    def runner_mocks(self):
        pipeline, lock = Mock(), Mock()
        return pipeline, lock, patch.multiple(run_seed_figures,
            SeedPipeline=Mock(return_value=pipeline), acquire=Mock(return_value=lock),
            completed_attempt=Mock(return_value=(Path("unit-attempt.json"), {"unit": "binding"})),
            upstream_binding=Mock(return_value={}))

    def test_runner_requires_actual_completed_evaluation_before_any_stage(self):
        pipeline, lock, mocks = self.runner_mocks()
        with mocks, patch.object(sys, "argv", ["run_seed_figures.py"]):
            run_seed_figures.completed_attempt.side_effect = ValueError("actual evaluated artifacts absent")
            with self.assertRaisesRegex(ValueError, "actual evaluated artifacts absent"):
                run_seed_figures.main()
            run_seed_figures.completed_attempt.assert_called_once_with(
                run_seed_figures.ROOT / "results/mechanism_v2/seed_evaluation_status.json", "seed_evaluation_completed")
        pipeline.stage.assert_not_called()
        pipeline.bind.assert_not_called()
        pipeline.finish.assert_not_called()
        pipeline.fail.assert_called_once()
        lock.close.assert_called_once()

    def test_mocked_runner_dispatches_exact_three_guarded_stages_without_claiming_qa(self):
        # Control-flow unit test only: no real model audit, child or image render.
        pipeline, lock, mocks = self.runner_mocks()
        with mocks, patch.object(sys, "argv", ["run_seed_figures.py"]), redirect_stdout(StringIO()):
            run_seed_figures.main()
        self.assertEqual([call.args[0] for call in pipeline.stage.call_args_list],
                         ["figures_validation", "figures_confirmation", "figures_navtest"])
        pipeline.finish.assert_called_once_with("seed_figures_rendered_visual_qa_pending")
        pipeline.fail.assert_not_called()
        lock.close.assert_called_once()

    def test_owned_lock_failure_cannot_read_prerequisites_or_render(self):
        pipeline, lock, mocks = self.runner_mocks()
        with mocks, patch.object(sys, "argv", ["run_seed_figures.py"]):
            run_seed_figures.acquire.side_effect = RuntimeError("heavy-job lock owned")
            with self.assertRaisesRegex(RuntimeError, "heavy-job lock owned"):
                run_seed_figures.main()
            run_seed_figures.completed_attempt.assert_not_called()
        pipeline.stage.assert_not_called()
        pipeline.finish.assert_not_called()
        pipeline.fail.assert_called_once()
        lock.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
