import copy
import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import analyze_seed
import test_seed_predictions as prediction_fixtures
from verify_mechanism_core import ROOT, digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SeedAnalysis(unittest.TestCase):
    def fixture(self, folder):
        import numpy as np
        workspace = folder / "workspace"
        config = workspace / "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("unit fixture only", encoding="utf-8")
        scores, predictions = folder / "scores", folder / "predictions"
        scores.mkdir()
        predictions.mkdir()
        tokens, logs = np.array(["a", "b"]), np.array(["session_a_0_1", "session_b_0_1"])
        wanted, settings = {}, {"split": "validation", "limit": None,
            "cache_root": str(workspace / "exp/mechanism_v2/metric_cache"),
            "wrapper_source_sha256": {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")},
            "scoring_config_sha256": digest(config), "runs": {}}
        complete = {"eligible_for_paper": True, "split": "validation", "final_invalid_scores": 0, "attempt_failures_recorded": 2, "runs": {}}
        for config in ("M0", "M4"):
            tag = config + "_f1_steps_s3"
            meta = {"run": tag, "split": "validation", "scenes": 2, "protocol_sha256": "extension-hash",
                    "checkpoint_sha256": config, "normalization_sha256": "same-norm", "training_settings_sha256": config,
                    "donors_sha256": "same-donors", "source_sha256": {"analysis/train_mechanism.py": "frozen"},
                    "features_sha256": "features", "eligible_for_paper": True, "fallbacks": 0}
            wanted[tag] = copy.deepcopy(meta)
            # Real small NPZ and receipt; rename only this fixture's generated pair.
            path, _, _, _ = prediction_fixtures.SeedPredictions().archive(predictions, meta=meta, tokens=tokens, logs=logs)
            target = predictions / (tag + ".npz")
            path.rename(target)
            path.with_suffix(".completed.json").rename(target.with_suffix(".completed.json"))
            settings["runs"][tag] = {"meta": meta, "predictions_sha256": digest(target)}
            csv_path = scores / (tag + ".csv.gz")
            with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", *analyze_seed.METRICS])
                writer.writeheader()
                for token, log in zip(tokens, logs):
                    writer.writerow({"token": token, "log_name": log, "valid": True, **{metric: .5 for metric in analyze_seed.METRICS}})
            complete["runs"][tag] = {"csv_sha256": digest(csv_path), "n": 2, "mean_score": .5}
        write_json(scores / "settings.json", settings)
        complete["settings_sha256"] = digest(scores / "settings.json")
        write_json(scores / "completed.json", complete)
        return scores, predictions, wanted, tokens, logs, "validation", workspace

    def test_real_npz_receipts_and_all_seven_metrics_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            arrays, provenance, receipt = analyze_seed.audit_archive(*paths)
            self.assertEqual(len(arrays), 2)
            self.assertEqual(len(next(iter(arrays.values()))[0]), 7)
            self.assertEqual({row["training_protocol_sha256"] for row in provenance.values()}, {"extension-hash"})
            self.assertEqual(receipt["attempt_failures_recorded"], 2)

    def test_core_and_added_protocols_are_not_mixed_into_one_score_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            paths[2]["M0_f1_steps_s3"]["protocol_sha256"] = "original-core-hash"
            with self.assertRaisesRegex(ValueError, "separate score archives"):
                analyze_seed.audit_archive(*paths)

    def test_metadata_source_separators_normalized_but_identity_not_relaxed(self):
        self.assertTrue(analyze_seed.matches_meta({"source_sha256": {"a\\b.py": "x"}}, {"source_sha256": {"a/b.py": "x"}}))
        self.assertFalse(analyze_seed.matches_meta({"source_sha256": {"a\\b.py": "x"}}, {"source_sha256": {"a/b.py": "y"}}))

    def test_changed_prediction_or_receipt_rejected(self):
        for changed in ("archive", "receipt"):
            with tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                prediction = paths[1] / "M0_f1_steps_s3.npz"
                if changed == "archive":
                    prediction.write_bytes(b"changed")
                else:
                    receipt = prediction.with_suffix(".completed.json")
                    record = json.loads(receipt.read_text(encoding="utf-8"))
                    record["predictions_sha256"] = "different"
                    write_json(receipt, record)
                with self.assertRaisesRegex(ValueError, "prediction|Prediction"):
                    analyze_seed.audit_archive(*paths)

    def test_smoke_wrapper_yaml_cache_and_nan_summary_rejected(self):
        for altered in ("smoke", "wrapper", "yaml", "cache", "mean"):
            with tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                scores = paths[0]
                settings = json.loads((scores / "settings.json").read_text(encoding="utf-8"))
                complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
                if altered == "smoke":
                    complete["eligible_for_paper"] = False
                elif altered == "wrapper":
                    settings["wrapper_source_sha256"] = {}
                elif altered == "yaml":
                    settings["scoring_config_sha256"] = "different"
                elif altered == "cache":
                    settings["cache_root"] = str(paths[6] / "wrong")
                else:
                    complete["runs"]["M0_f1_steps_s3"]["mean_score"] = float("nan")
                write_json(scores / "settings.json", settings)
                complete["settings_sha256"] = digest(scores / "settings.json")
                write_json(scores / "completed.json", complete)
                with self.assertRaisesRegex(ValueError, "official|summary"):
                    analyze_seed.audit_archive(*paths)

    def test_resealed_csv_cannot_replace_cohort_or_nonfinite_submetric(self):
        for altered in ("token", "log", "nan", "duplicate"):
            with tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                scores = paths[0]
                target = scores / "M0_f1_steps_s3.csv.gz"
                with gzip.open(target, "rt", encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if altered == "token":
                    rows[0]["token"] = "new"
                elif altered == "log":
                    rows[0]["log_name"] = "different"
                elif altered == "nan":
                    rows[0]["ego_progress"] = "nan"
                else:
                    rows[1]["token"] = rows[0]["token"]
                with gzip.open(target, "wt", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
                complete["runs"]["M0_f1_steps_s3"]["csv_sha256"] = digest(target)
                write_json(scores / "completed.json", complete)
                with self.assertRaisesRegex(ValueError, "coverage|nonfinite|Duplicate"):
                    analyze_seed.audit_archive(*paths)

    def test_summary_uses_all_five_paired_seeds_and_all_metrics(self):
        import numpy as np
        from analyze_mechanism import summarize
        arrays = {f"M{config}_f1_steps_s{seed}": np.full((2, 7), .2 + config * .01 + seed * .02)
                  for config in range(6) for seed in range(5)}
        arrays["M1_f1_steps_s4"][:, 0] += .2
        result = summarize(arrays, np.array(["session_a_0_1", "session_b_0_1"]), {"all": [True, True]},
                           ["M1-M0", "M4-M1", "M4-M5"], [0, 1, 2, 3, 4], 10000)
        row = result["contrasts"]["M1-M0"]
        self.assertEqual(set(row["seed_effects"]), {"0", "1", "2", "3", "4"})
        self.assertAlmostEqual(row["metric_effects"]["score"], .05)
        self.assertEqual(set(row["metric_effects"]), set(analyze_seed.METRICS))
        self.assertAlmostEqual(row["multiplicity_interval"]["alpha"], .05 / 3)

    def test_registered_builder_rejects_draw_tuning_before_data_access(self):
        with self.assertRaisesRegex(ValueError, "10,000"):
            analyze_seed.build_report(Path("unused"), Path("unused"), "validation", draws=1000)

    def builder_fixture(self, folder):
        """Real small arrays/archives; only the independently tested fit audit is mocked."""
        import numpy as np
        workspace, results = folder / "workspace", folder / "results"
        feature_path = folder / "synthetic_features.npz"
        tokens = np.array(["fixture_" + str(number) for number in range(10)])
        logs = np.array(["session_a_0_1"] * 5 + ["session_b_0_1"] * 5)
        np.savez_compressed(feature_path, token=tokens, log_name=logs, map_name=np.array(["fixture_city"] * 10),
                            agents=np.zeros((10, 8, 11), np.float32), speed=np.ones(10) * 4,
                            cmd=np.ones(10), lead=np.ones((10, 2)) * 100)
        base = {"schema": "dynamic-objects-v2", "training": {"seeds": [0, 1, 2]},
                "sources": {name: {"path": str(feature_path), "sha256": digest(feature_path)} for name in ("navtrain", "navtest")},
                "split_sessions": {"validation": ["session_a", "session_b"]},
                "primary_contrasts": ["M1-M0", "M4-M1", "M4-M5"],
                "thresholds": {"near_object_distance_m": 20., "crossing_horizon_s": 3., "speed_low_mps": 2.,
                               "speed_high_mps": 5., "close_lead_gap_m": 15.},
                "disclosure": "Synthetic builder fixture, not NAVSIM experiment evidence."}
        base_path = results / "protocol/protocol.json"
        write_json(base_path, base)
        protocol = copy.deepcopy(base)
        protocol["training"]["seeds"] = [0, 1, 2, 3, 4]
        protocol["seed_extension"] = {"fixture": True}
        protocol_path = results / "seed_protocol/active/protocol.json"
        write_json(protocol_path, protocol)
        registration = results / "seed_protocol/registration.json"
        write_json(registration, {"fixture": True})
        write_json(protocol_path.parent / "completed.json", {"fixture": True})
        sources = {name: digest(ROOT / name) for name in ("agents/mechanism_model.py", "analysis/mechanism_data.py", "analysis/train_mechanism.py")}
        core = {"source_sha256": sources, "donors_sha256": "same-fixture-donors", "runs": []}
        added = {}
        for seed in range(5):
            for config in range(6):
                tag = f"M{config}_f1_steps_s{seed}"
                model = {"tag": tag, "checkpoint_sha256": tag, "normalization_sha256": "same-fixture-norm", "settings_sha256": tag + "-settings"}
                if seed < 3:
                    core["runs"].append(model)
                else:
                    added[tag] = model
        original_meta, added_meta = analyze_seed.expected_metadata("validation", 10, core, added, base_path, protocol_path, registration)
        config = workspace / "navsim/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("synthetic official-scorer-config fixture only", encoding="utf-8")
        tracked = [base_path, protocol_path, registration, feature_path]
        for label, meta_table, score_name, prediction_name in (
                ("original", original_meta, "scores", "predictions"), ("added", added_meta, "seed_scores", "seed_predictions")):
            scores = results / score_name / "validation"
            predictions = workspace / "exp/mechanism_v2" / prediction_name / "validation"
            scores.mkdir(parents=True)
            predictions.mkdir(parents=True)
            settings = {"split": "validation", "limit": None, "runs": {},
                        "cache_root": str(workspace / "exp/mechanism_v2/metric_cache"),
                        "wrapper_source_sha256": {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")},
                        "scoring_config_sha256": digest(config)}
            complete = {"eligible_for_paper": True, "split": "validation", "final_invalid_scores": 0, "attempt_failures_recorded": 0, "runs": {}}
            for tag, meta in meta_table.items():
                path = predictions / (tag + ".npz")
                np.savez_compressed(path, token=tokens, log_name=logs, poses=np.zeros((10, 8, 3), np.float32), meta=np.array(json.dumps(meta)))
                write_json(path.with_suffix(".completed.json"), {"meta": meta, "predictions_sha256": digest(path)})
                settings["runs"][tag] = {"meta": meta, "predictions_sha256": digest(path)}
                value = .3 + int(tag[1]) * .02 + int(tag[-1]) * .01
                csv_path = scores / (tag + ".csv.gz")
                with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", *analyze_seed.METRICS])
                    writer.writeheader()
                    for token, log in zip(tokens, logs):
                        writer.writerow({"token": token, "log_name": log, "valid": True, **{metric: value for metric in analyze_seed.METRICS}})
                complete["runs"][tag] = {"n": 10, "mean_score": value, "csv_sha256": digest(csv_path)}
                tracked.extend([path, path.with_suffix(".completed.json"), csv_path])
            write_json(scores / "settings.json", settings)
            complete["settings_sha256"] = digest(scores / "settings.json")
            write_json(scores / "completed.json", complete)
            tracked.extend([scores / "settings.json", scores / "completed.json"])
        return workspace, results, (core, protocol, added, {"required": True, "fixture": True}), tracked

    def test_source_rebuilt_thirty_run_report_retains_two_protocols_and_all_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, results, checked, tracked = self.builder_fixture(Path(temporary))
            before = {str(path): digest(path) for path in tracked}
            with patch.object(analyze_seed, "verify_all", return_value=checked) as guard:
                report = analyze_seed.build_report(workspace, results, "validation")
            guard.assert_called_once()
            self.assertEqual(report["paired_seeds"], [0, 1, 2, 3, 4])
            self.assertEqual(len(report["provenance"]), 30)
            self.assertEqual((report["reused_core_runs"], report["added_runs"]), (18, 12))
            self.assertEqual({row["training_protocol_sha256"] for row in report["provenance"].values()},
                             {report["base_protocol_sha256"], report["extension_protocol_sha256"]})
            self.assertAlmostEqual(report["summary"]["contrasts"]["M1-M0"]["metric_effects"]["score"], .02)
            self.assertEqual(set(report["summary"]["contrasts"]["M1-M0"]["seed_effects"]), {"0", "1", "2", "3", "4"})
            self.assertEqual(report["n_sessions"], 2)
            text = analyze_seed.render(report)
            self.assertIn("five seeds", text)
            self.assertIn("not fresh independent confirmation", text)
            self.assertEqual({str(path): digest(path) for path in tracked}, before)

    def test_source_rebuilt_report_cannot_relabel_original_scores_as_added_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, results, checked, _ = self.builder_fixture(Path(temporary))
            scores = results / "scores/validation"
            settings = json.loads((scores / "settings.json").read_text(encoding="utf-8"))
            settings["runs"]["M0_f1_steps_s0"]["meta"]["protocol_sha256"] = digest(results / "seed_protocol/active/protocol.json")
            write_json(scores / "settings.json", settings)
            complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
            complete["settings_sha256"] = digest(scores / "settings.json")
            write_json(scores / "completed.json", complete)
            with patch.object(analyze_seed, "verify_all", return_value=checked):
                with self.assertRaisesRegex(ValueError, "training-protocol identity"):
                    analyze_seed.build_report(workspace, results, "validation")


if __name__ == "__main__":
    unittest.main()
