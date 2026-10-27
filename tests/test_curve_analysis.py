import copy
import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_mechanism import METRICS
from analyze_mechanism_curves import read_archives, summarize, tag_for, verify_training
from run_mechanism_curves import matrix
from verify_mechanism_core import digest
import test_mechanism_manifest as manifest_fixtures


class CurveAnalysis(unittest.TestCase):
    def training_fixture(self, folder):
        models, protocol_path = manifest_fixtures.MechanismManifest().fixture(folder)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["fractions"] = {"0.1": {"n_scenes": 10}, "0.3": {"n_scenes": 30}, "1.0": {"n_scenes": 100}}
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        policy_path = folder / "policy.json"
        policy = json.loads((ROOT / "results/mechanism_v2/protocol/followup_policy.json").read_text(encoding="utf-8"))
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        for run in models.iterdir():
            settings_path = run / "settings.json"
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            settings["protocol_sha256"] = digest(protocol_path)
            settings_path.write_text(json.dumps(settings), encoding="utf-8")
            complete_path = run / "completed.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["settings_sha256"] = digest(settings_path)
            complete_path.write_text(json.dumps(complete), encoding="utf-8")
        for row in matrix(protocol, policy):
            anchor = models / f"{row['config']}_f1_steps_s{row['seed']}"
            run = models / row["tag"]
            run.mkdir()
            settings = json.loads((anchor / "settings.json").read_text(encoding="utf-8"))
            settings.update({key: row[key] for key in ("config", "seed", "fraction", "budget")})
            (run / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
            (run / "model.pt").write_bytes((anchor / "model.pt").read_bytes())
            (run / "normalization.json").write_text(json.dumps({"fixture_fraction": row["fraction"]}), encoding="utf-8")
            complete = json.loads((anchor / "completed.json").read_text(encoding="utf-8"))
            complete.update(tag=row["tag"], optimizer_steps=row["optimizer_steps"], n_train=row["n_train"],
                            settings_sha256=digest(run / "settings.json"), normalization_sha256=digest(run / "normalization.json"))
            (run / "history.json").write_text(json.dumps([{"step": row["optimizer_steps"], "validation": complete["final_validation"]}]), encoding="utf-8")
            (run / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
        return models, protocol_path, policy_path

    def test_complete_matrix_guard_and_missing_model_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.training_fixture(Path(temporary))
            _, trained = verify_training(*paths)
            self.assertEqual(len(trained), 30)
            self.assertEqual(sum("_f1_" in tag for tag in trained), 6)
            (paths[0] / "M4_f0.3_epochs_s2/completed.json").unlink()
            with self.assertRaises(FileNotFoundError):
                verify_training(*paths)

    def test_same_fraction_normalizers_must_match_across_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.training_fixture(Path(temporary))
            run = paths[0] / "M4_f0.3_epochs_s2"
            (run / "normalization.json").write_text('{"changed": true}', encoding="utf-8")
            path = run / "completed.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["normalization_sha256"] = digest(run / "normalization.json")
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "normalizers differ"):
                verify_training(*paths)

    def fixture(self, folder):
        core = {"protocol_sha256": "protocol", "donors_sha256": "donors", "source_sha256": {"source": "sha"}}
        trained = {"M1_f0.1_steps_s0": {"checkpoint_sha256": "checkpoint", "normalization_sha256": "norm", "settings_sha256": "train"}}
        tag = next(iter(trained))
        meta = {"eligible_for_paper": True, "fallbacks": 0, "run": tag, "split": "validation", "scenes": 3,
                "protocol_sha256": "protocol", "features_sha256": "features", "donors_sha256": "donors",
                "source_sha256": core["source_sha256"], "checkpoint_sha256": "checkpoint",
                "normalization_sha256": "norm", "training_settings_sha256": "train"}
        settings = {"limit": None, "runs": {tag: {"meta": meta, "predictions_sha256": "predictions"}},
                    "wrapper_source_sha256": {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")}}
        (folder / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        path = folder / (tag + ".csv.gz")
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", *METRICS])
            writer.writeheader()
            for token in ("a", "b", "c"):
                writer.writerow(dict(token=token, log_name="log", valid=True, **dict.fromkeys(METRICS, .5)))
        complete = {"eligible_for_paper": True, "split": "validation", "final_invalid_scores": 0,
                    "settings_sha256": digest(folder / "settings.json"),
                    "runs": {tag: {"csv_sha256": digest(path), "n": 3, "mean_score": .5}}}
        (folder / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
        return core, trained

    def test_hash_bound_archive_and_exact_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            core, trained = self.fixture(folder)
            args = (folder, core, trained, ["c", "b", "a"], ["log"] * 3, "features")
            arrays, _, _ = read_archives(*args)
            self.assertEqual(next(iter(arrays.values())).shape, (3, 7))
            (folder / "M1_f0.1_steps_s0.csv.gz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "CSV changed"):
                read_archives(*args)

    def test_wrong_normalization_and_smoke_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            core, trained = self.fixture(folder)
            wrong = copy.deepcopy(trained)
            next(iter(wrong.values()))["normalization_sha256"] = "other_subset"
            with self.assertRaisesRegex(ValueError, "trained artifact mismatch"):
                read_archives(folder, core, wrong, ["a", "b", "c"], ["log"] * 3, "features")
            complete = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
            complete["eligible_for_paper"] = False
            (folder / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke"):
                read_archives(folder, core, trained, ["a", "b", "c"], ["log"] * 3, "features")

    def test_log_misalignment_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            core, trained = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                read_archives(folder, core, trained, ["a", "b", "c"], ["log", "wrong", "log"], "features")

    def test_full_point_reused_not_an_independent_epoch_run(self):
        self.assertEqual(tag_for("M1", 1., "epochs", 0), "M1_f1_steps_s0")
        self.assertEqual(tag_for("M4", .3, "epochs", 2), "M4_f0.3_epochs_s2")

    def test_paired_interaction_and_seed_average(self):
        protocol = {"training": {"seeds": [0, 1, 2]}, "fractions": {"0.1": {"n_scenes": 10}, "0.3": {"n_scenes": 30}, "1.0": {"n_scenes": 100}}}
        arrays = {}
        for budget in ("steps", "epochs"):
            for fraction in (.1, .3, 1.):
                for config in ("M1", "M4"):
                    for seed in range(3):
                        value = .4 + seed * .01 + (0 if config == "M1" else .02 if fraction == 1. else .06)
                        arrays[tag_for(config, fraction, budget, seed)] = np.full((8, 7), value)
        logs = np.array(["2021.01.01.00.00.00_vehicle_a_000_010"] * 4 + ["2021.01.02.00.00.00_vehicle_b_000_010"] * 4)
        report = summarize(arrays, logs, protocol, draws=200)
        self.assertEqual(len(report["cells"]), 6)
        self.assertEqual(len(report["small_minus_full_encoder_effect"]), 4)
        effect = report["small_minus_full_encoder_effect"]["steps/f0.1-full"]
        self.assertAlmostEqual(effect["metric_effects"]["score"], .04)
        self.assertTrue(report["cells"]["epochs/f1"]["reused_full_data_anchor"])
        self.assertAlmostEqual(report["cells"]["steps/f0.1"]["models"]["M1"]["metric_means"]["score"], .41)


if __name__ == "__main__":
    unittest.main()
