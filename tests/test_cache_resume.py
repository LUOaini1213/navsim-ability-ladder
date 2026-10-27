import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from resume_mechanism_cache import cache_memory_policy, check_cache_exit, verify_prior_attempt
from verify_mechanism_core import digest
from resource_guard import MemoryInterrupted, wait_guarded


class CacheResume(unittest.TestCase):
    def test_conservative_headroom_and_no_automatic_pressure_retry(self):
        policy = cache_memory_policy()
        self.assertEqual(policy["launch"]["min_free_mb"], 6000)
        self.assertEqual(policy["launch"]["stable_seconds"], 30)
        self.assertEqual(policy["running"]["poll_seconds"], .5)
        self.assertEqual(policy["running"]["low_samples"], 1)
        self.assertEqual(policy["automatic_retries_after_memory_interruption"], 0)
        with self.assertRaisesRegex(RuntimeError, "automatic retry disabled"):
            check_cache_exit(1, {"kind": "memory_interruption"})
        with self.assertRaisesRegex(RuntimeError, "no ordinary failure retry"):
            check_cache_exit(1, None)
        check_cache_exit(0, None)

    def test_effective_policy_stops_at_first_two_gb_warning(self):
        class OwnedFixture:
            pid = 123456
            def poll(self):
                return None
        samples = iter([2500., 1999.])
        stopped, sleeps = [], []
        with self.assertRaises(MemoryInterrupted) as caught:
            wait_guarded(OwnedFixture(), **cache_memory_policy()["running"], read_memory=lambda: next(samples),
                         sleep=sleeps.append, stop=lambda child: stopped.append(child.pid) or 1)
        self.assertEqual(stopped, [123456])
        self.assertEqual(sleeps, [.5])
        self.assertEqual(caught.exception.event["available_mb"], 1999.)
        self.assertEqual(caught.exception.event["required_low_samples"], 1)

    def fixture(self, folder):
        source = folder / "source.py"
        source.write_text("# fixture\n", encoding="utf-8")
        value = {"stage": "cache_validation", "status": "failed_or_interrupted",
                 "jobs": [{"stage": stage, "exit_code": 0} for stage in
                          ("core_artifacts", "input_controls", "process_recovery", "live_parity", "cache_smoke")],
                 "launch_source_and_policy_sha256": {str(source): digest(source)}}
        path = folder / "attempt.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return source, value, path

    def test_bound_successful_audits_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, value, path = self.fixture(Path(temporary))
            self.assertEqual(verify_prior_attempt(path)[str(source)], digest(source))
            value["jobs"][1]["exit_code"] = 1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "successful audit"):
                verify_prior_attempt(path)

    def test_changed_source_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _, path = self.fixture(Path(temporary))
            source.write_text("# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source/policy changed"):
                verify_prior_attempt(path)


if __name__ == "__main__":
    unittest.main()
