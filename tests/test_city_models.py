import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import city_models
from verify_mechanism_core import digest


class CityModels(unittest.TestCase):
    def fixture(self, folder):
        path = folder / "protocol.json"
        protocol = {"split_counts": {"train": {"scenes": 100}, "validation": {"scenes": 8}},
                    "training": {"batch_size": 10, "full_data_epochs_equivalent": 2, "seeds": [0, 1, 2]}}
        path.write_text(json.dumps(protocol), encoding="utf-8")
        (folder / "donors.npz").write_bytes(b"test-only donor fixture, not scientific data")
        core = {"source_sha256": {"analysis/train_mechanism.py": "fixture"}, "donors_sha256": "base-donors", "runs": []}
        norm_hash = None
        for row in city_models.matrix():
            run = folder / "models" / row["tag"]
            run.mkdir(parents=True)
            settings = {"protocol_sha256": digest(path), "donors_sha256": digest(folder / "donors.npz"),
                        "config": row["config"], "seed": row["seed"], "fraction": 1., "budget": "steps", "smoke_max_steps": None,
                        "training": protocol["training"], "source_sha256": core["source_sha256"]}
            (run / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
            (run / "model.pt").write_bytes(b"test-only checkpoint fixture")
            (run / "normalization.json").write_text("{}", encoding="utf-8")
            norm_hash = digest(run / "normalization.json")
            final = {name: .5 for name in city_models.VALIDATION_METRICS}
            complete = {"eligible_for_paper": True, "tag": row["tag"], "optimizer_steps": 20, "n_train": 100, "n_validation": 8,
                        "parameters": 600000, "final_validation": final, "settings_sha256": digest(run / "settings.json"),
                        "checkpoint_sha256": digest(run / "model.pt"), "normalization_sha256": norm_hash}
            (run / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
            (run / "history.json").write_text(json.dumps([{"step": 20, "validation": final}]), encoding="utf-8")
            core["runs"].append(copy.deepcopy(complete))
        return folder / "models", path, protocol, core

    def test_six_full_budget_auxiliaries_and_exact_matrix(self):
        rows = city_models.matrix()
        self.assertEqual(len({row["tag"] for row in rows}), 6)
        self.assertEqual({row["seed"] for row in rows}, {0, 1, 2})
        self.assertEqual({row["condition"] for row in rows}, {"C2", "C5"})
        with tempfile.TemporaryDirectory() as temporary:
            folder, path, protocol, core = self.fixture(Path(temporary))
            for row in rows:
                self.assertEqual(city_models.verify_one(folder, row, protocol, path, core)["optimizer_steps"], 20)

    def test_empty_partial_nonfinite_or_negative_diagnostics_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, path, protocol, core = self.fixture(Path(temporary))
            row = city_models.matrix()[0]
            run = folder / row["tag"]
            original = json.loads((run / "completed.json").read_text(encoding="utf-8"))
            for final in ({}, {"ade_m": .5}, {key: float("nan") for key in city_models.VALIDATION_METRICS},
                          {key: -1. for key in city_models.VALIDATION_METRICS}):
                with self.subTest(final=final):
                    complete = dict(original, final_validation=final)
                    (run / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
                    (run / "history.json").write_text(json.dumps([{"step": 20, "validation": final}]), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Incomplete/smoke/wrong-budget"):
                        city_models.verify_one(folder, row, protocol, path, core)

    def test_smoke_budget_changed_checkpoint_and_normalizer_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, path, protocol, core = self.fixture(Path(temporary))
            row = city_models.matrix()[0]
            run = folder / row["tag"]
            original = json.loads((run / "completed.json").read_text(encoding="utf-8"))
            for change in ({"eligible_for_paper": False}, {"optimizer_steps": 19}, {"n_validation": 10}):
                (run / "completed.json").write_text(json.dumps(dict(original, **change)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    city_models.verify_one(folder, row, protocol, path, core)
            (run / "completed.json").write_text(json.dumps(original), encoding="utf-8")
            (run / "model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                city_models.verify_one(folder, row, protocol, path, core)
            changed = dict(original, checkpoint_sha256=digest(run / "model.pt"))
            (run / "normalization.json").write_text('{"changed":true}', encoding="utf-8")
            changed["normalization_sha256"] = digest(run / "normalization.json")
            (run / "completed.json").write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "normalizers differ"):
                city_models.verify_one(folder, row, protocol, path, core)

    def test_all_six_and_base_audit_required_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, path, protocol, core = self.fixture(Path(temporary))
            with patch.object(city_models, "verify", return_value=core) as base_check, \
                 patch.object(city_models, "verify_inputs", return_value=protocol):
                result = city_models.verify_all(Path("base-models"), folder, Path("base-protocol"), path, Path("audit"))
                base_check.assert_called_once_with(Path("base-models"), Path("base-protocol"))
                self.assertEqual(len(result[2]), 6)
                (folder / city_models.matrix()[-1]["tag"] / "completed.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    city_models.verify_all(Path("base-models"), folder, Path("base-protocol"), path, Path("audit"))

    def test_real_anchor_source_and_full_cohort_metadata_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, path, _, core = self.fixture(Path(temporary))
            model = core["runs"][0]
            tag = model["tag"]
            meta = {"eligible_for_paper": True, "fallbacks": 0, "run": tag, "split": "validation", "scenes": 10,
                    "protocol_sha256": digest(path), "donors_sha256": core["donors_sha256"], "checkpoint_sha256": model["checkpoint_sha256"],
                    "normalization_sha256": model["normalization_sha256"], "training_settings_sha256": model["settings_sha256"],
                    "features_sha256": "features", "source_sha256": core["source_sha256"]}
            args = (model, core, path, {"sha256": "features"}, 10, tag, "validation")
            city_models.verify_anchor_metadata(meta, *args)
            for change in ({"source_sha256": {}}, {"donors_sha256": "different"}, {"scenes": 8}, {"fallbacks": 1}):
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    city_models.verify_anchor_metadata(dict(meta, **change), *args)


if __name__ == "__main__":
    unittest.main()
