import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import predict_seed
from seed_models import write_new_json
from verify_mechanism_core import CONFIGS, digest


class SeedPredictions(unittest.TestCase):
    def archive(self, folder, meta=None, tokens=None, logs=None, poses=None):
        import numpy as np
        tokens = np.array(["a", "b"]) if tokens is None else tokens
        logs = np.array(["session_a_0_1", "session_b_0_1"]) if logs is None else logs
        poses = np.zeros((2, 8, 3), np.float32) if poses is None else poses
        meta = {"fixture": True} if meta is None else meta
        path = folder / "predictions.npz"
        np.savez_compressed(path, token=tokens, log_name=logs, poses=poses, meta=np.array(json.dumps(meta)))
        write_new_json(path.with_suffix(".completed.json"), {"meta": meta, "predictions_sha256": digest(path)})
        return path, meta, tokens, logs

    def test_metadata_retains_distinct_base_and_extension_training_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base, protocol, registration = [folder / name for name in ("base.json", "protocol.json", "registration.json")]
            base.write_text("{}", encoding="utf-8")
            protocol.write_text(json.dumps({"schema": "dynamic-objects-v2", "sources": {
                "navtrain": {"sha256": "train"}, "navtest": {"sha256": "test"}}}), encoding="utf-8")
            registration.write_text("{}", encoding="utf-8")
            core = {"donors_sha256": "same-donors", "source_sha256": {"analysis/train_mechanism.py": "frozen"}}
            added = {f"{config}_f1_steps_s{seed}": {"checkpoint_sha256": str(seed) + config,
                     "normalization_sha256": "same-norm", "settings_sha256": "added-settings"}
                     for config in CONFIGS for seed in (3, 4)}
            meta = predict_seed.metadata("M4", 3, "confirmation", 2, core, added, base, protocol, registration)
            self.assertEqual(meta["base_protocol_sha256"], digest(base))
            self.assertEqual(meta["protocol_sha256"], digest(protocol))
            self.assertNotEqual(meta["protocol_sha256"], meta["base_protocol_sha256"])
            self.assertEqual(meta["checkpoint_sha256"], "3M4")
            self.assertEqual(meta["features_sha256"], "train")
            self.assertEqual(meta["fallbacks"], 0)

    def test_unknown_or_core_seed_refused_before_dependencies_and_guard(self):
        for config, seed, split in (("M0", 0, "confirmation"), ("M6", 3, "validation"), ("M0", 3, "train")):
            with patch.object(predict_seed, "verify_all") as guard:
                with self.assertRaisesRegex(ValueError, "Unregistered"):
                    predict_seed.export(config, seed, split, Path("workspace"), Path("results"), Path("out"))
                guard.assert_not_called()

    def test_missing_thirtieth_model_blocks_confirmation_before_feature_loading(self):
        with patch.object(predict_seed, "verify_all", side_effect=FileNotFoundError("missing added model")) as guard:
            with self.assertRaisesRegex(FileNotFoundError, "missing added model"):
                predict_seed.export("M5", 4, "confirmation", Path("workspace"), Path("results"), Path("out"))
            guard.assert_called_once()

    def test_actual_npz_existing_prediction_is_verified_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, meta, tokens, logs = self.archive(Path(temporary))
            before = digest(path)
            self.assertTrue(predict_seed.check_existing(path, meta, tokens, logs))
            self.assertEqual(digest(path), before)
            self.assertFalse(predict_seed.check_existing(path.parent / "fresh.npz", meta, tokens, logs))

    def test_partial_receipt_or_archive_is_preserved(self):
        for missing in ("receipt", "archive"):
            with tempfile.TemporaryDirectory() as temporary:
                path, meta, tokens, logs = self.archive(Path(temporary))
                deleted = path.with_suffix(".completed.json") if missing == "receipt" else path
                deleted.unlink()
                with self.assertRaisesRegex(ValueError, "Partial"):
                    predict_seed.check_existing(path, meta, tokens, logs)
                self.assertEqual(path.exists(), missing == "receipt")

    def test_resealed_archive_cannot_replace_token_log_shape_or_finite_output(self):
        import numpy as np
        changes = ({"tokens": np.array(["c", "b"])}, {"logs": np.array(["different", "session_b_0_1"])},
                   {"poses": np.zeros((2, 7, 3), np.float32)}, {"poses": np.full((2, 8, 3), np.nan)})
        for change in changes:
            with tempfile.TemporaryDirectory() as temporary:
                path, meta, _, _ = self.archive(Path(temporary), **change)
                with self.assertRaisesRegex(ValueError, "content/coverage"):
                    predict_seed.check_existing(path, meta, np.array(["a", "b"]), np.array(["session_a_0_1", "session_b_0_1"]))

    def test_changed_meta_and_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, meta, tokens, logs = self.archive(Path(temporary))
            with self.assertRaisesRegex(ValueError, "provenance"):
                predict_seed.check_existing(path, {"fixture": False}, tokens, logs)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                predict_seed.check_existing(path, meta, tokens, logs)


if __name__ == "__main__":
    unittest.main()
