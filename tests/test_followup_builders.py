import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import analyze_city
import analyze_mechanism_curves


class FollowupBuilders(unittest.TestCase):
    def test_city_rebuild_requires_all_models_and_actual_trigger_before_features(self):
        core = {"fixture": "verified-core"}
        with patch.object(analyze_city, "verify_all", return_value=(core, {}, {})) as models, \
             patch.object(analyze_city, "verify_gate", return_value={"required": False}) as gate, \
             patch.object(analyze_city.np, "load") as features:
            with self.assertRaisesRegex(ValueError, "actual frozen execution trigger"):
                analyze_city.build_report(Path("scores"), Path("predictions"), Path("workspace"))
            models.assert_called_once()
            gate.assert_called_once()
            self.assertIs(gate.call_args.args[-1], core)
            features.assert_not_called()

    def test_curve_rebuild_requires_complete_training_and_gate_before_features(self):
        core = {"fixture": "verified-core"}
        with patch.object(analyze_mechanism_curves, "verify_training", return_value=(core, {})) as models, \
             patch.object(analyze_mechanism_curves, "verified_gate", return_value={"learning_curve_required": False}) as gate, \
             patch.object(analyze_mechanism_curves.np, "load") as features:
            with self.assertRaisesRegex(ValueError, "execution gate"):
                analyze_mechanism_curves.build_report(Path("scores"), Path("models"), Path("protocol.json"))
            models.assert_called_once()
            gate.assert_called_once()
            self.assertIs(gate.call_args.args[-1], core)
            features.assert_not_called()


if __name__ == "__main__":
    unittest.main()
