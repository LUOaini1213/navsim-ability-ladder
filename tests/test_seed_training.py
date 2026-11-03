import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import run_seed_training as runner
from resource_guard import MemoryInterrupted


class SeedTraining(unittest.TestCase):
    def make(self, folder):
        return runner.SeedPipeline("seed_fixture", folder / "status.json", folder / "logs",
                                   folder / "attempts", "Unit fixture only")

    def test_real_tiny_child_logged_with_conservative_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make(Path(temporary))
            with patch.object(runner, "wait_for_headroom") as headroom, \
                 patch.object(runner, "wait_guarded", side_effect=lambda child, **_: child.wait()) as guard:
                pipeline.stage("fixture", [sys.executable, "-c", "print('tiny fixture only')"])
            for key, value in runner.LAUNCH_POLICY.items():
                self.assertEqual(headroom.call_args.kwargs[key], value)
            for key, value in runner.RUNNING_POLICY.items():
                self.assertEqual(guard.call_args.kwargs[key], value)
            self.assertIn("tiny fixture only", (pipeline.logs / "fixture.log").read_text(encoding="utf-8"))
            self.assertEqual(len(pipeline.state["jobs"]), 1)
            self.assertIsNone(pipeline.state["child_pid"])
            self.assertEqual(pipeline.state["memory_policy"]["automatic_retries"], 0)

    def test_owned_memory_interruption_is_recorded_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make(Path(temporary))
            def interrupt(child, **_):
                # Only this tiny fixture process is owned; no unrelated PID lookup.
                child.wait()
                raise MemoryInterrupted({"kind": "fixture_interruption", "exit_code": 0, "child_pid": child.pid})
            with patch.object(runner, "wait_for_headroom") as headroom, \
                 patch.object(runner, "wait_guarded", side_effect=interrupt):
                with self.assertRaises(MemoryInterrupted):
                    pipeline.stage("fixture", [sys.executable, "-c", "pass"])
            self.assertEqual(headroom.call_count, 1)
            self.assertEqual(len(pipeline.state["jobs"]), 1)
            self.assertEqual(pipeline.state["jobs"][0]["memory_interruption"]["kind"], "fixture_interruption")

    def test_ordinary_failure_not_skipped_or_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make(Path(temporary))
            with patch.object(runner, "wait_for_headroom"), \
                 patch.object(runner, "wait_guarded", side_effect=lambda child, **_: child.wait()):
                with self.assertRaisesRegex(RuntimeError, "no skip or automatic retry"):
                    pipeline.stage("fixture", [sys.executable, "-c", "import sys; sys.exit(7)"])
            self.assertEqual(len(pipeline.state["jobs"]), 1)
            self.assertEqual(pipeline.state["jobs"][0]["exit_code"], 7)

    def test_source_change_while_waiting_prevents_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "bound.txt"
            source.write_text("before", encoding="utf-8")
            pipeline = self.make(folder)
            pipeline.bind([source])
            with patch.object(runner, "wait_for_headroom", side_effect=lambda **_: source.write_text("changed", encoding="utf-8")), \
                 patch.object(runner.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "bound input/source"):
                    pipeline.stage("fixture", [sys.executable, "-c", "pass"])
            launch.assert_not_called()

    def test_post_launch_status_failure_cleans_up_owned_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make(Path(temporary))
            original_save = pipeline.save
            def fail_running_save():
                if pipeline.state["status"] == "running" and pipeline.state.get("child_pid") is not None:
                    raise OSError("fixture receipt write failure")
                original_save()
            with patch.object(runner, "wait_for_headroom"), \
                 patch.object(pipeline, "save", side_effect=fail_running_save), \
                 patch.object(runner, "stop_owned_child", side_effect=lambda child: child.wait()) as stop:
                with self.assertRaisesRegex(OSError, "receipt write failure"):
                    pipeline.stage("fixture", [sys.executable, "-c", "pass"])
            stop.assert_called_once()
            self.assertEqual(len(pipeline.state["jobs"]), 1)
            self.assertIsNone(pipeline.state["child_pid"])

    def test_no_resumable_auto_retry_or_log_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make(Path(temporary))
            with self.assertRaisesRegex(ValueError, "No automatic"):
                pipeline.stage("fixture", [], resumable=True)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                pipeline.stage("../outside", [])

    def test_command_uses_isolated_outputs_frozen_trainer_and_fixed_budget(self):
        command = list(map(str, runner.training_command("python", Path("workspace"), Path("active/protocol.json"),
                                                       Path("seed_models"), {"config": "M4", "seed": 3})))
        self.assertTrue(command[2].replace("\\", "/").endswith("analysis/train_mechanism.py"))
        for key, value in (("--seed", "3"), ("--fraction", "1"), ("--budget", "steps"),
                           ("--threads", "2"), ("--out-root", "seed_models")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertNotIn("--max-steps", command)


if __name__ == "__main__":
    unittest.main()
