import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from guarded_pipeline import Pipeline


class GuardedPipeline(unittest.TestCase):
    def make(self, folder):
        return Pipeline("fixture", folder / "status.json", folder / "logs", folder / "attempts", "Unit test only")

    def test_real_child_logged_and_ordinary_failure_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pipeline = self.make(folder)
            with patch("guarded_pipeline.wait_for_headroom"), patch("guarded_pipeline.wait_guarded", lambda child, **_: child.wait()):
                pipeline.stage("ok", [sys.executable, "-c", "print('fixture output')"])
                self.assertIn("fixture output", (folder / "logs/ok.log").read_text(encoding="utf-8"))
                with self.assertRaisesRegex(RuntimeError, "no ordinary failure"):
                    pipeline.stage("bad", [sys.executable, "-c", "import sys; sys.exit(7)"], resumable=True)
            self.assertEqual(len(pipeline.state["jobs"]), 2)
            self.assertEqual(pipeline.state["jobs"][-1]["exit_code"], 7)
            self.assertTrue(pipeline.attempt.exists())

    def test_source_change_during_memory_wait_prevents_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "source.py"
            source.write_text("# before\n", encoding="utf-8")
            pipeline = self.make(folder)
            pipeline.bind([source])
            with patch("guarded_pipeline.wait_for_headroom", side_effect=lambda **_: source.write_text("# changed\n", encoding="utf-8")), patch("guarded_pipeline.subprocess.Popen") as popen:
                with self.assertRaisesRegex(ValueError, "bound input/source"):
                    pipeline.stage("guard", [sys.executable, "-c", "pass"])
                popen.assert_not_called()

    def test_log_traversal_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = self.make(Path(temporary))
            with self.assertRaises(ValueError):
                pipeline.stage("../outside", [sys.executable, "-c", "pass"])


if __name__ == "__main__":
    unittest.main()
