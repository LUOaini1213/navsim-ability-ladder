import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_mechanism_scoring import prediction_matrix, verify_cache_receipts
from verify_mechanism_core import digest


class ScoringRunner(unittest.TestCase):
    def fixture(self, folder):
        protocol = folder / "protocol.json"
        protocol.write_text(json.dumps({"split_counts": {"validation": {"scenes": 10}, "confirmation": {"scenes": 8}}}), encoding="utf-8")
        cache = folder / "cache"
        cache.mkdir()
        for split, scenes in (("validation", 10), ("confirmation", 8)):
            (cache / (split + "_completed.json")).write_text(json.dumps({"split": split, "limit": None, "scenes": scenes,
                                                                         "protocol_sha256": digest(protocol)}), encoding="utf-8")
        state = {"status": "caches_completed", "frozen_sha256": {str(protocol): digest(protocol)},
                 "jobs": [{"split": split, "exit_code": 0, "memory_interruption": None} for split in ("validation", "confirmation")]}
        attempt = folder / "attempt.json"
        attempt.write_text(json.dumps(state), encoding="utf-8")
        return cache, protocol, attempt, state

    def test_exact_54_exports_not_picked_seeds(self):
        rows = prediction_matrix()
        self.assertEqual(len(rows), 54)
        self.assertEqual(len(set(rows)), 54)
        self.assertEqual({row[0] for row in rows}, {"validation", "confirmation", "navtest"})

    def test_complete_matching_cache_receipts_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, protocol, attempt, _ = self.fixture(Path(temporary))
            self.assertEqual(verify_cache_receipts(cache, protocol, attempt)["status"], "caches_completed")
            value = json.loads((cache / "confirmation_completed.json").read_text(encoding="utf-8"))
            value["limit"] = 2
            (cache / "confirmation_completed.json").write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify_cache_receipts(cache, protocol, attempt)

    def test_interrupted_controller_is_not_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, protocol, attempt, state = self.fixture(Path(temporary))
            state["status"] = "waiting_for_memory"
            attempt.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not completed"):
                verify_cache_receipts(cache, protocol, attempt)


if __name__ == "__main__":
    unittest.main()
