import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import seed_models
import test_seed_followup as seed_fixtures
from seed_followup import matrix
from verify_mechanism_core import digest


class SeedModels(unittest.TestCase):
    def fixture(self, folder, discordant=True):
        helper = seed_fixtures.SeedFollowup()
        paths = helper.evidence_fixture(folder, ("validation",) if discordant else ())
        registration, base, policy, results, core, protocol = paths[:6]
        models = folder / "models"
        activated, core, gate = seed_models.activate(models, base, policy, registration, results)
        return paths, activated, models, core, gate

    def added_fixture(self, folder):
        paths, activated, models, core, gate = self.fixture(folder)
        protocol = json.loads(activated.read_text(encoding="utf-8"))
        added = folder / "added_models"
        final = {metric: .5 for metric in seed_models.VALIDATION_METRICS}
        for row in matrix(paths[5]):
            run = added / row["tag"]
            run.mkdir(parents=True)
            settings = {"protocol_sha256": digest(activated), "donors_sha256": digest(activated.parent / "donors.npz"),
                        "config": row["config"], "seed": row["seed"], "fraction": 1., "budget": "steps",
                        "smoke_max_steps": None, "training": protocol["training"], "source_sha256": core["source_sha256"]}
            seed_fixtures.write_json(run / "settings.json", settings)
            (run / "model.pt").write_bytes(b"unit fixture, not a trained checkpoint")
            (run / "normalization.json").write_text("{}", encoding="utf-8")
            seed_fixtures.write_json(run / "history.json", [{"step": 20, "validation": final}])
            complete = {"eligible_for_paper": True, "tag": row["tag"], "optimizer_steps": 20,
                        "n_train": 100, "n_validation": 10, "parameters": 600000, "final_validation": final,
                        "settings_sha256": digest(run / "settings.json"), "checkpoint_sha256": digest(run / "model.pt"),
                        "normalization_sha256": digest(run / "normalization.json")}
            seed_fixtures.write_json(run / "completed.json", complete)
        return paths, activated, models, added, core

    def test_actual_gated_copy_preserves_base_protocol_and_donors(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = seed_fixtures.SeedFollowup().evidence_fixture(folder)
            registration, base, policy, results = paths[:4]
            before = digest(base), digest(base.parent / "donors.npz")
            activated, core, gate = seed_models.activate(folder / "models", base, policy, registration, results)
            self.assertTrue(gate["required"])
            self.assertEqual((digest(base), digest(base.parent / "donors.npz")), before)
            self.assertEqual(digest(activated.parent / "donors.npz"), before[1])
            extended = json.loads(activated.read_text(encoding="utf-8"))
            self.assertEqual(extended["training"]["seeds"], [0, 1, 2, 3, 4])
            self.assertFalse(json.loads((activated.parent / "completed.json").read_text(encoding="utf-8"))["actual_added_models_verified"])
            self.assertEqual(seed_models.activate(folder / "models", base, policy, registration, results)[0], activated)

    def test_no_trigger_materializes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, activated, _, _, gate = self.fixture(Path(temporary), False)
            self.assertIsNone(activated)
            self.assertFalse(gate["required"])
            self.assertFalse((paths[0].parent / "active").exists())

    def test_partial_activation_is_preserved_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            paths = seed_fixtures.SeedFollowup().evidence_fixture(folder)
            active = paths[0].parent / "active"
            active.mkdir()
            marker = active / "protocol.json"
            marker.write_text('{"partial_fixture":true}', encoding="utf-8")
            before = digest(marker)
            with self.assertRaises(FileNotFoundError):
                seed_models.activate(folder / "models", paths[1], paths[2], paths[0], paths[3])
            self.assertEqual(digest(marker), before)

    def test_changed_activation_or_donor_copy_is_rejected(self):
        for altered in ("lineage", "donors", "cohort"):
            with tempfile.TemporaryDirectory() as temporary:
                paths, activated, models, _, _ = self.fixture(Path(temporary))
                if altered == "donors":
                    (activated.parent / "donors.npz").write_bytes(b"changed")
                elif altered == "lineage":
                    target = activated.parent / "completed.json"
                    value = json.loads(target.read_text(encoding="utf-8"))
                    value["registration_sha256"] = "different"
                    seed_fixtures.write_json(target, value)
                else:
                    value = json.loads(activated.read_text(encoding="utf-8"))
                    value["split_counts"]["validation"]["scenes"] = 9
                    seed_fixtures.write_json(activated, value)
                with self.assertRaisesRegex(ValueError, "lineage|cohorts"):
                    seed_models.activate(models, paths[1], paths[2], paths[0], paths[3])

    def test_all_twelve_artifacts_required_for_confirmation_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, activated, models, added, _ = self.added_fixture(Path(temporary))
            result = seed_models.verify_all(models, added, paths[1], paths[2], paths[0], paths[3])
            self.assertEqual(len(result[2]), 12)
            self.assertEqual({row["seed"] for row in result[2].values()}, {3, 4})
            (added / "M5_f1_steps_s4/completed.json").unlink()
            with self.assertRaises(FileNotFoundError):
                seed_models.verify_all(models, added, paths[1], paths[2], paths[0], paths[3])

    def test_smoke_wrong_budget_seed_and_nonfinite_metrics_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, activated, _, added, core = self.added_fixture(Path(temporary))
            row = matrix(paths[5])[0]
            completed_path = added / row["tag"] / "completed.json"
            original = json.loads(completed_path.read_text(encoding="utf-8"))
            for change in ({"eligible_for_paper": False}, {"optimizer_steps": 19}, {"parameters": 600001},
                           {"final_validation": {}}, {"final_validation": {metric: float("nan") for metric in seed_models.VALIDATION_METRICS}}):
                seed_fixtures.write_json(completed_path, dict(original, **change))
                with self.assertRaisesRegex(ValueError, "Incomplete/smoke"):
                    seed_models.verify_one(added, row, activated, paths[5], core)
            seed_fixtures.write_json(completed_path, original)
            with self.assertRaisesRegex(ValueError, "Unregistered"):
                seed_models.verify_one(added, dict(row, seed=0), activated, paths[5], core)

    def test_checkpoint_and_resealed_normalizer_change_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, activated, _, added, core = self.added_fixture(Path(temporary))
            row = matrix(paths[5])[0]
            run = added / row["tag"]
            (run / "model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                seed_models.verify_one(added, row, activated, paths[5], core)
            complete = json.loads((run / "completed.json").read_text(encoding="utf-8"))
            complete["checkpoint_sha256"] = digest(run / "model.pt")
            (run / "normalization.json").write_text('{"changed":true}', encoding="utf-8")
            complete["normalization_sha256"] = digest(run / "normalization.json")
            seed_fixtures.write_json(run / "completed.json", complete)
            with self.assertRaisesRegex(ValueError, "normalizer differs"):
                seed_models.verify_one(added, row, activated, paths[5], core)

    def test_source_and_settings_hash_changes_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, activated, _, added, core = self.added_fixture(Path(temporary))
            row = matrix(paths[5])[0]
            target = added / row["tag"] / "settings.json"
            original = json.loads(target.read_text(encoding="utf-8"))
            changed = copy.deepcopy(original)
            changed["source_sha256"]["analysis/train_mechanism.py"] = "changed"
            seed_fixtures.write_json(target, changed)
            with self.assertRaisesRegex(ValueError, "original frozen"):
                seed_models.verify_one(added, row, activated, paths[5], core)
            seed_fixtures.write_json(target, original)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                seed_models.verify_one(added, row, activated, paths[5], core)


if __name__ == "__main__":
    unittest.main()
