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
from run_mechanism_curves import matrix, verified_gate, verify_run
from mechanism_gates import decide
from verify_mechanism_core import digest, verify
import test_mechanism_manifest as manifest_fixtures


class CurveRunner(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "results/mechanism_v2/protocol/protocol.json").read_text(encoding="utf-8"))
        self.policy = json.loads((ROOT / "results/mechanism_v2/protocol/followup_policy.json").read_text(encoding="utf-8"))

    def test_exact_additional_matrix_and_full_data_reuse(self):
        rows = matrix(self.protocol, self.policy)
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["fraction"] for row in rows}, {.1, .3})
        self.assertEqual({row["config"] for row in rows}, {"M1", "M4"})
        self.assertEqual(sum(row["budget"] == "steps" for row in rows), 12)
        self.assertEqual(sum(row["budget"] == "epochs" for row in rows), 12)
        self.assertEqual({row["optimizer_steps"] for row in rows if row["budget"] == "steps"}, {13000})
        self.assertTrue(all(row["optimizer_steps"] < 13000 for row in rows if row["budget"] == "epochs"))

    def test_matrix_cannot_silently_expand_or_pick_seeds(self):
        for key, value in (("configs", ["M1", "M4", "M5"]), ("seeds", [0]), ("fractions", [.1, .3, 1.])):
            altered = copy.deepcopy(self.policy)
            altered["learning_curve"]["fixed_updates"][key] = value
            with self.assertRaises(ValueError):
                matrix(self.protocol, altered)

    def test_training_verifier_refuses_wrong_budget_or_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            models, protocol_path = manifest_fixtures.MechanismManifest().fixture(Path(temporary))
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            row = {"tag": "M1_f1_steps_s0", "config": "M1", "fraction": 1., "budget": "steps", "seed": 0,
                   "n_train": 100, "optimizer_steps": 20}
            self.assertEqual(verify_run(models, row, protocol, protocol_path)["optimizer_steps"], 20)
            wrong = dict(row, optimizer_steps=19)
            with self.assertRaisesRegex(ValueError, "wrong-budget"):
                verify_run(models, wrong, protocol, protocol_path)
            path = models / row["tag"] / "completed.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["eligible_for_paper"] = False
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke"):
                verify_run(models, row, protocol, protocol_path)

    def test_training_verifier_refuses_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            models, protocol_path = manifest_fixtures.MechanismManifest().fixture(Path(temporary))
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            row = {"tag": "M1_f1_steps_s0", "config": "M1", "fraction": 1., "budget": "steps", "seed": 0,
                   "n_train": 100, "optimizer_steps": 20}
            (models / row["tag"] / "model.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_run(models, row, protocol, protocol_path)

    def gate_fixture(self, folder):
        models, protocol_path = manifest_fixtures.MechanismManifest().fixture(folder)
        core = verify(models, protocol_path)
        policy = copy.deepcopy(self.policy)
        policy["protocol_sha256"] = core["protocol_sha256"]
        policy_path = folder / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        scores = folder / "scores"
        scores.mkdir()
        report = {"split": "validation", "protocol_sha256": core["protocol_sha256"], "n_scenes": 10,
                  "provenance": {}, "summary": {"contrasts": {}}}
        settings = {"limit": None, "runs": {}}
        complete = {"eligible_for_paper": True, "split": "validation", "final_invalid_scores": 0, "runs": {}}
        values = {"M0": .48, "M1": .50, "M2": .49, "M3": .51, "M4": .52, "M5": .50}
        for trained in core["runs"]:
            tag, config = trained["tag"], trained["tag"].split("_")[0]
            meta = {"eligible_for_paper": True, "run": tag, "split": "validation", "scenes": 10, "fallbacks": 0,
                    "protocol_sha256": core["protocol_sha256"], "source_sha256": core["source_sha256"],
                    "checkpoint_sha256": trained["checkpoint_sha256"], "normalization_sha256": trained["normalization_sha256"],
                    "training_settings_sha256": trained["settings_sha256"], "donors_sha256": core["donors_sha256"]}
            path = scores / (tag+".csv.gz")
            with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", "score"])
                writer.writeheader()
                for number in range(10):
                    writer.writerow({"token": str(number), "log_name": "synthetic_session_0_1", "valid": True, "score": values[config]})
            settings["runs"][tag] = {"meta": meta}
            complete["runs"][tag] = {"csv_sha256": digest(path), "n": 10}
            report["provenance"][tag] = dict(meta, csv_sha256=digest(path))
        settings_path = scores / "settings.json"
        settings_path.write_text(json.dumps(settings), encoding="utf-8")
        complete["settings_sha256"] = digest(settings_path)
        (scores / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
        report["score_settings_sha256"] = digest(settings_path)
        for label in ("M1-M0", "M4-M1", "M4-M5"):
            left, right = label.split("-")
            effect = values[left]-values[right]
            report["summary"]["contrasts"][label] = {"primary": True, "metric_effects": {"score": effect},
                "seed_effects": {str(seed): effect for seed in range(3)}, "multiplicity_interval": {"low": .01, "high": .03}}
        report_path = folder / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        gate = decide(report, None, policy)
        gate.update(policy_sha256=digest(policy_path), validation_report_sha256=digest(report_path), confirmation_report_sha256=None)
        gate_path = folder / "gate.json"
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        return report_path, scores, gate_path, policy_path, core

    def test_gate_requires_actual_bound_score_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.gate_fixture(Path(temporary))
            self.assertTrue(verified_gate(*paths)["learning_curve_required"])
            path = paths[1] / "M1_f1_steps_s0.csv.gz"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "CSV changed"):
                verified_gate(*paths)

    def test_gate_rejects_changed_decision_and_smoke_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.gate_fixture(Path(temporary))
            gate = json.loads(paths[2].read_text(encoding="utf-8"))
            gate["learning_curve_required"] = False
            paths[2].write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash-bound"):
                verified_gate(*paths)
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.gate_fixture(Path(temporary))
            path = paths[1] / "completed.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["eligible_for_paper"] = False
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke"):
                verified_gate(*paths)


if __name__ == "__main__":
    unittest.main()
