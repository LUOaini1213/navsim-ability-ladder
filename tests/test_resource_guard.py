import sys
import os
import subprocess
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from resource_guard import MemoryInterrupted, wait_for_headroom, wait_guarded


class FakeClock:
    def __init__(self):
        self.now = 0

    def sleep(self, seconds):
        self.now += seconds


class FakeChild:
    pid = 123456

    def __init__(self, polls):
        self.polls = iter(polls)

    def poll(self):
        return next(self.polls)


class ResourceGuardTests(unittest.TestCase):
    def run_child(self, readings, polls, events=None):
        stopped = []
        def stop(child):
            stopped.append(child.pid)
            return 1
        return wait_guarded(FakeChild(polls), read_memory=iter(readings).__next__,
                            sleep=lambda _: None, stop=stop, on_event=(events or []).append), stopped

    def test_normal_exit_never_kills(self):
        code, stopped = self.run_child([1800, 2000], [None, None, 0])
        self.assertEqual(code, 0)
        self.assertEqual(stopped, [])

    def test_nonzero_exit_not_reclassified_as_memory(self):
        code, stopped = self.run_child([], [7])
        self.assertEqual(code, 7)
        self.assertEqual(stopped, [])

    def test_sustained_low_stops_only_owned_pid(self):
        stopped = []
        with self.assertRaises(MemoryInterrupted) as caught:
            wait_guarded(FakeChild([None, None]), read_memory=iter([950, 800]).__next__,
                         sleep=lambda _: None, stop=lambda child: stopped.append(child.pid) or 1)
        self.assertEqual(stopped, [123456])
        self.assertEqual(caught.exception.event["recent_samples_mb"], [950., 800.])

    def test_single_transient_low_resets(self):
        code, stopped = self.run_child([900, 1600, 950, 1400], [None, None, None, None, 0])
        self.assertEqual(code, 0)
        self.assertFalse(stopped)

    def test_emergency_stops_on_first_sample(self):
        with self.assertRaises(MemoryInterrupted) as caught:
            self.run_child([400], [None])
        self.assertEqual(caught.exception.event["available_mb"], 400)

    def test_memory_probe_failure_cleans_up_owned_child(self):
        stopped = []
        def broken():
            raise OSError("probe failed")
        with self.assertRaises(OSError):
            wait_guarded(FakeChild([None]), read_memory=broken,
                         stop=lambda child: stopped.append(child.pid))
        self.assertEqual(stopped, [123456])

    def test_launch_requires_continuous_headroom(self):
        clock = FakeClock()
        events = []
        wait_for_headroom(2400, 10, 5, read_memory=iter([2500, 2000, 2500, 2600, 2700]).__next__,
                          sleep=clock.sleep, monotonic=lambda: clock.now, on_event=events.append)
        self.assertEqual(clock.now, 20)
        self.assertEqual(events[-1]["kind"], "memory_headroom_ready")

    def test_invalid_policy_refused(self):
        with self.assertRaises(ValueError):
            wait_guarded(FakeChild([]), emergency_mb=1001, critical_mb=1000)
        with self.assertRaises(ValueError):
            wait_for_headroom(stable_seconds=-1)

    @unittest.skipUnless(os.name == "nt", "Project memory monitor targets Windows")
    def test_actual_owned_child_stop_preserves_unrelated_child(self):
        code = [sys.executable, "-c", "import time; time.sleep(30)"]
        owned = subprocess.Popen(code, creationflags=subprocess.CREATE_NO_WINDOW)
        unrelated = subprocess.Popen(code, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            with self.assertRaises(MemoryInterrupted):
                wait_guarded(owned, read_memory=lambda: 400)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            for child in (owned, unrelated):
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
