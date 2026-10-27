import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_city_evaluation import completed_attempt, prediction_matrix
from verify_mechanism_core import digest


class CityEvaluation(unittest.TestCase):
    def test_exact_36_exports_same_seeds_and_three_splits(self):
        rows = prediction_matrix()
        self.assertEqual(len(rows), 36)
        self.assertEqual(len(set(rows)), 36)
        self.assertEqual({row[1] for row in rows}, {"C1", "C2", "C4", "C5"})
        self.assertEqual({row[2] for row in rows}, {0, 1, 2})

    def test_completed_attempt_not_snapshot_or_failed_last_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "source"
            source.write_text("fixture", encoding="utf-8")
            attempt, snapshot = folder / "attempt.json", folder / "status.json"
            state = {"status": "completed", "attempt_receipt": str(attempt), "frozen_sha256": {str(source): digest(source)},
                     "jobs": [{"stage": "training", "exit_code": 1, "memory_interruption": {"kind": "memory_interruption"}},
                              {"stage": "training", "exit_code": 0, "memory_interruption": None}]}
            for path in (attempt, snapshot):
                path.write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(completed_attempt(snapshot, "completed")[0], attempt)
            state["jobs"][-1]["exit_code"] = 1
            for path in (attempt, snapshot):
                path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsuccessful"):
                completed_attempt(snapshot, "completed")
            state["jobs"][-1]["exit_code"] = 0
            snapshot.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot"):
                completed_attempt(snapshot, "completed")

    def test_changed_bound_sources_and_waiting_are_not_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, attempt, snapshot = folder / "source", folder / "attempt.json", folder / "status.json"
            source.write_text("fixture", encoding="utf-8")
            state = {"status": "waiting_for_memory", "attempt_receipt": str(attempt), "frozen_sha256": {str(source): digest(source)},
                     "jobs": [{"stage": "train", "exit_code": 0, "memory_interruption": None}]}
            for path in (attempt, snapshot):
                path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has not completed"):
                completed_attempt(snapshot, "completed")
            state["status"] = "completed"
            for path in (attempt, snapshot):
                path.write_text(json.dumps(state), encoding="utf-8")
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source/input changed"):
                completed_attempt(snapshot, "completed")


if __name__ == "__main__":
    unittest.main()
