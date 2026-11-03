import copy
import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from mechanism_gates import decide
from seed_followup import PRIMARY, extension_protocol, freeze, matrix, verify_gate
from verify_mechanism_core import CONFIGS, digest, verify
import test_mechanism_manifest as manifest_fixtures


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SeedFollowup(unittest.TestCase):
    def base_fixture(self, folder):
        models, base = manifest_fixtures.MechanismManifest().fixture(folder)
        protocol = json.loads(base.read_text(encoding="utf-8"))
        protocol.update(schema="dynamic-objects-v2", primary_contrasts=list(PRIMARY))
        protocol["split_counts"]["confirmation"] = {"scenes": 10}
        write_json(base, protocol)
        for run in models.iterdir():
            settings_path = run / "settings.json"
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            settings["protocol_sha256"] = digest(base)
            write_json(settings_path, settings)
            complete_path = run / "completed.json"
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["settings_sha256"] = digest(settings_path)
            write_json(complete_path, complete)
        core = verify(models, base)
        original = json.loads((ROOT / "results/mechanism_v2/protocol/followup_policy.json").read_text(encoding="utf-8"))
        original["protocol_sha256"] = digest(base)
        policy = folder / "followup_policy.json"
        write_json(policy, original)
        results = folder / "results"
        registration_path = results / "seed_protocol/registration.json"
        registration = freeze(registration_path, base, policy, results)
        return registration_path, base, policy, results, core, protocol, registration

    def evidence_fixture(self, folder, discordant_splits=("validation",)):
        paths = self.base_fixture(folder)
        registration, base, policy, results, core, protocol, _ = paths
        reports = []
        for split in ("validation", "confirmation"):
            scores = results / "scores" / split
            scores.mkdir(parents=True)
            settings = {"limit": None, "runs": {}}
            complete = {"eligible_for_paper": True, "split": split, "final_invalid_scores": 0, "runs": {}}
            report = {"split": split, "protocol_sha256": core["protocol_sha256"], "n_scenes": 10,
                      "provenance": {}, "summary": {"contrasts": {}}}
            means = {}
            for trained in core["runs"]:
                tag = trained["tag"]
                config, seed = tag.split("_")[0], int(tag.rsplit("s", 1)[1])
                value = {"M0": .48, "M1": .50, "M2": .49, "M3": .51, "M4": .52, "M5": .50}[config]
                if split in discordant_splits and config == "M1":
                    value = (.46, .50, .49)[seed]
                means[tag] = value
                meta = {"eligible_for_paper": True, "run": tag, "split": split, "scenes": 10,
                        "fallbacks": 0, "protocol_sha256": core["protocol_sha256"],
                        "source_sha256": core["source_sha256"], "donors_sha256": core["donors_sha256"],
                        "checkpoint_sha256": trained["checkpoint_sha256"],
                        "normalization_sha256": trained["normalization_sha256"],
                        "training_settings_sha256": trained["settings_sha256"]}
                csv_path = scores / (tag + ".csv.gz")
                with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", "score"])
                    writer.writeheader()
                    for number in range(10):
                        writer.writerow({"token": str(number), "log_name": "fixture_session_0_1", "valid": True, "score": value})
                settings["runs"][tag] = {"meta": meta}
                complete["runs"][tag] = {"csv_sha256": digest(csv_path), "n": 10}
                report["provenance"][tag] = dict(meta, csv_sha256=digest(csv_path))
            write_json(scores / "settings.json", settings)
            complete["settings_sha256"] = digest(scores / "settings.json")
            write_json(scores / "completed.json", complete)
            report["score_settings_sha256"] = complete["settings_sha256"]
            for label in PRIMARY:
                left, right = label.split("-")
                effects = {str(seed): means[f"{left}_f1_steps_s{seed}"] - means[f"{right}_f1_steps_s{seed}"] for seed in (0, 1, 2)}
                report["summary"]["contrasts"][label] = {
                    "primary": True, "seed_effects": effects,
                    "metric_effects": {"score": sum(effects.values()) / 3},
                    "multiplicity_interval": {"low": -.1, "high": .1}}
            write_json(results / "analysis" / split / "report.json", report)
            reports.append(report)
        self.reseal_gate(paths)
        return paths

    def reseal_gate(self, paths):
        _, _, policy_path, results = paths[:4]
        reports = [json.loads((results / "analysis" / split / "report.json").read_text(encoding="utf-8")) for split in ("validation", "confirmation")]
        gate = decide(*reports, json.loads(policy_path.read_text(encoding="utf-8")))
        gate.update(policy_sha256=digest(policy_path),
                    validation_report_sha256=digest(results / "analysis/validation/report.json"),
                    confirmation_report_sha256=digest(results / "analysis/confirmation/report.json"))
        write_json(results / "gate_confirmation.json", gate)

    def test_real_design_has_exact_twelve_full_budget_jobs(self):
        protocol = json.loads((ROOT / "results/mechanism_v2/protocol/protocol.json").read_text(encoding="utf-8"))
        before = copy.deepcopy(protocol)
        rows = matrix(protocol)
        self.assertEqual(protocol, before)
        self.assertEqual(len({row["tag"] for row in rows}), 12)
        self.assertEqual({row["seed"] for row in rows}, {3, 4})
        self.assertEqual({row["config"] for row in rows}, set(CONFIGS))
        self.assertEqual({row["optimizer_steps"] for row in rows}, {13000})
        self.assertEqual({row["fraction"] for row in rows}, {1.})

    def test_matrix_refuses_replaced_seeds_and_contrasts(self):
        protocol = json.loads((ROOT / "results/mechanism_v2/protocol/protocol.json").read_text(encoding="utf-8"))
        for alteration in ("seeds", "contrasts"):
            changed = copy.deepcopy(protocol)
            if alteration == "seeds":
                changed["training"]["seeds"] = [1, 2, 3]
            else:
                changed["primary_contrasts"] = ["M4-M0"]
            with self.assertRaisesRegex(ValueError, "Original"):
                matrix(changed)

    def test_registration_idempotent_and_binds_donors(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.base_fixture(Path(temporary))
            self.assertEqual(freeze(*paths[:4]), paths[6])
            (paths[1].parent / "donors.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after freezing"):
                freeze(*paths[:4])

    def test_prospective_registration_refuses_existing_reports(self):
        for split in ("validation", "confirmation", "navtest"):
            with tempfile.TemporaryDirectory() as temporary:
                paths = self.base_fixture(Path(temporary))
                # Remove only the fixture's own registration, never repository data.
                paths[0].unlink()
                write_json(paths[3] / "analysis" / split / "report.json", {})
                with self.assertRaisesRegex(ValueError, "after complete reports"):
                    freeze(*paths[:4])
                self.assertFalse(paths[0].exists())

    def test_recipe_changes_only_seed_registration_not_cohorts_or_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.base_fixture(Path(temporary))
            original = copy.deepcopy(paths[5])
            extended = extension_protocol(paths[5], paths[6])
            self.assertEqual(extended["training"]["seeds"], [0, 1, 2, 3, 4])
            extended.pop("seed_extension")
            extended["training"]["seeds"] = [0, 1, 2]
            self.assertEqual(extended, original)
            self.assertEqual(paths[5], original)
            self.assertFalse((paths[0].parent / "protocol.json").exists())

    def test_requires_both_reports_and_never_creates_missing_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.base_fixture(Path(temporary))
            with self.assertRaises(FileNotFoundError):
                verify_gate(*paths[:5])
            paths[0].unlink()
            with self.assertRaisesRegex(ValueError, "must already be frozen"):
                verify_gate(*paths[:5])
            self.assertFalse(paths[0].exists())

    def test_paired_csv_trigger_not_training_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.evidence_fixture(Path(temporary))
            gate = verify_gate(*paths[:5])
            self.assertTrue(gate["required"])
            self.assertIn("validation:M1-M0", gate["trigger_contrasts"])
            self.assertEqual(len(gate["additional_jobs"]), 12)
            self.assertFalse(gate["actual_added_models_complete"])

    def test_confirmation_only_discordance_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.evidence_fixture(Path(temporary), ("confirmation",))
            self.assertIn("confirmation:M1-M0", verify_gate(*paths[:5])["trigger_contrasts"])

    def test_no_discordance_does_not_manufacture_added_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.evidence_fixture(Path(temporary), ())
            gate = verify_gate(*paths[:5])
            self.assertFalse(gate["required"])
            self.assertEqual(gate["additional_jobs"], [])

    def test_changed_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.evidence_fixture(Path(temporary))
            (paths[3] / "scores/confirmation/M1_f1_steps_s0.csv.gz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "CSV changed"):
                verify_gate(*paths[:5])

    def test_resealed_report_cannot_replace_actual_paired_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.evidence_fixture(Path(temporary))
            report_path = paths[3] / "analysis/confirmation/report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["summary"]["contrasts"]["M1-M0"]["seed_effects"]["0"] = -.2
            write_json(report_path, report)
            self.reseal_gate(paths)
            with self.assertRaisesRegex(ValueError, "actual paired CSVs"):
                verify_gate(*paths[:5])

    def test_smoke_archive_and_changed_gate_are_rejected(self):
        for altered in ("smoke", "gate"):
            with tempfile.TemporaryDirectory() as temporary:
                paths = self.evidence_fixture(Path(temporary))
                target = paths[3] / ("scores/validation/completed.json" if altered == "smoke" else "gate_confirmation.json")
                value = json.loads(target.read_text(encoding="utf-8"))
                value["eligible_for_paper" if altered == "smoke" else "cross_protocol_status"] = False
                write_json(target, value)
                with self.assertRaisesRegex(ValueError, "smoke|hash-bound"):
                    verify_gate(*paths[:5])

    def test_core_cannot_replace_or_duplicate_one_of_eighteen_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.evidence_fixture(Path(temporary))
            changed = copy.deepcopy(paths[4])
            changed["runs"][-1] = changed["runs"][0]
            with self.assertRaisesRegex(ValueError, "core identities"):
                verify_gate(*paths[:4], changed)

    def test_recipe_refuses_post_registration_seed_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.base_fixture(Path(temporary))
            for key in ("reused_seeds", "additional_seeds"):
                changed = copy.deepcopy(paths[6])
                changed[key] = [0]
                with self.assertRaisesRegex(ValueError, "matrix changed"):
                    extension_protocol(paths[5], changed)

    def test_nan_effect_cannot_bypass_difference_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.evidence_fixture(Path(temporary))
            report_path = paths[3] / "analysis/confirmation/report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["summary"]["contrasts"]["M1-M0"]["seed_effects"]["0"] = float("nan")
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "actual paired CSVs"):
                verify_gate(*paths[:5])


if __name__ == "__main__":
    unittest.main()
