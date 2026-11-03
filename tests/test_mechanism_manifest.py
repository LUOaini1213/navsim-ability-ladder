import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from verify_mechanism_core import CONFIGS, digest, verify


class MechanismManifest(unittest.TestCase):
    def fixture(self, folder):
        protocol = {"split_counts": {"train": {"scenes": 100}, "validation": {"scenes": 10}},
                    "training": {"batch_size": 10, "full_data_epochs_equivalent": 2, "seeds": [0, 1, 2]}}
        path = folder / "protocol.json"
        path.write_text(json.dumps(protocol), encoding="utf-8")
        (folder / "donors.npz").write_bytes(b"fixture only; not an experimental archive")
        sources = {name: digest(ROOT / name) for name in ("agents/mechanism_model.py", "analysis/mechanism_data.py", "analysis/train_mechanism.py")}
        for seed in range(3):
            for config in CONFIGS:
                tag = f"{config}_f1_steps_s{seed}"
                run = folder / "models" / tag
                run.mkdir(parents=True)
                settings = {"protocol_sha256": digest(path), "donors_sha256": digest(folder / "donors.npz"), "config": config,
                            "seed": seed, "fraction": 1., "budget": "steps", "smoke_max_steps": None, "training": protocol["training"],
                            "source_sha256": sources}
                (run / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
                (run / "model.pt").write_bytes(b"fixture only; not a trained model")
                (run / "normalization.json").write_text("{}", encoding="utf-8")
                (run / "history.json").write_text(json.dumps([{"step": 20, "validation": {"ade_m": .5}}]), encoding="utf-8")
                complete = {"eligible_for_paper": True, "tag": tag, "optimizer_steps": 20, "n_train": 100, "n_validation": 10,
                            "final_validation": {"ade_m": .5}, "parameters": 600000,
                            "settings_sha256": digest(run / "settings.json"), "checkpoint_sha256": digest(run / "model.pt"),
                            "normalization_sha256": digest(run / "normalization.json")}
                (run / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
        return folder / "models", path

    def test_complete_manifest_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            models, protocol = self.fixture(Path(temporary))
            self.assertEqual(verify(models, protocol)["n_runs"], 18)
            (models / "M4_f1_steps_s1/model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed artifact"):
                verify(models, protocol)

    def test_wrong_budget_or_smoke_never_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            models, protocol = self.fixture(Path(temporary))
            path = models / "M0_f1_steps_s0/completed.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["optimizer_steps"] = 19
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "budget"):
                verify(models, protocol)

    def test_source_changes_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            models, protocol = self.fixture(Path(temporary))
            path = models / "M1_f1_steps_s0/settings.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["source_sha256"]["agents/mechanism_model.py"] = "changed"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                verify(models, protocol)


if __name__ == "__main__":
    unittest.main()
