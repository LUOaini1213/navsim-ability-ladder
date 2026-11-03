import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from mechanism_gates import decide


class MechanismGates(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "results/mechanism_v2/protocol/followup_policy.json").read_text(encoding="utf-8"))
        self.validation = {"split": "validation", "protocol_sha256": self.policy["protocol_sha256"],
                           "provenance": {str(i): {"checkpoint_sha256": str(i)} for i in range(18)}, "summary": {"contrasts": {}}}
        for label in ("M1-M0", "M4-M1", "M4-M5"):
            self.validation["summary"]["contrasts"][label] = {"primary": True, "metric_effects": {"score": .02},
                "seed_effects": {"0": .02, "1": .03, "2": .01}, "multiplicity_interval": {"low": .01, "high": .03}}
        self.confirmation = copy.deepcopy(self.validation)
        self.confirmation["split"] = "confirmation"

    def test_validation_only_does_not_unlock_v2(self):
        result = decide(self.validation, None, self.policy)
        self.assertTrue(result["learning_curve_required"])
        self.assertEqual(result["cross_protocol_status"], "pending_confirmation")

    def test_negative_effects_also_trigger_curves(self):
        for row in self.validation["summary"]["contrasts"].values():
            row["metric_effects"]["score"] = -.02
        self.assertTrue(decide(self.validation, None, self.policy)["learning_curve_required"])

    def test_confirmation_reversal_never_unlocks_v2(self):
        for row in self.confirmation["summary"]["contrasts"].values():
            row.update(metric_effects={"score": -.02}, seed_effects={"0": -.02, "1": -.03, "2": -.01},
                       multiplicity_interval={"low": -.03, "high": -.01})
        self.assertEqual(decide(self.validation, self.confirmation, self.policy)["cross_protocol_status"], "not_triggered")

    def test_same_fitted_checkpoints_required(self):
        self.confirmation["provenance"]["0"]["checkpoint_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "identical fitted"):
            decide(self.validation, self.confirmation, self.policy)

    def test_seed_discordance_is_not_hidden(self):
        row = self.validation["summary"]["contrasts"]["M4-M1"]
        row["seed_effects"] = {"0": -.02, "1": .05, "2": .03}
        result = decide(self.validation, self.confirmation, self.policy)
        self.assertIn("validation:M4-M1", result["seed_discordance_requiring_followup_or_withheld_claim"])

    def test_small_effect_or_zero_crossing_interval_is_not_confirmation(self):
        for row in self.confirmation["summary"]["contrasts"].values():
            row["multiplicity_interval"]["low"] = -.001
        self.assertEqual(decide(self.validation, self.confirmation, self.policy)["cross_protocol_status"], "not_triggered")
        for row in self.validation["summary"]["contrasts"].values():
            row["metric_effects"]["score"] = .001
        self.assertFalse(decide(self.validation, None, self.policy)["learning_curve_required"])


if __name__ == "__main__":
    unittest.main()
