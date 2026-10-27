import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_mechanism_evaluation import upstream_binding
from verify_mechanism_core import digest


class UpstreamBinding(unittest.TestCase):
    def fixture(self, folder):
        devkit = folder / "navsim"
        subprocess.run(["git", "init", "--quiet", str(devkit)], check=True)
        path = devkit / "scoring.yaml"
        path.write_text("weight: 5\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(devkit), "add", "scoring.yaml"], check=True)
        subprocess.run(["git", "-C", str(devkit), "-c", "user.name=LocalTest", "-c", "user.email=local-test@example.invalid",
                        "commit", "--quiet", "-m", "fixture only"], check=True)
        # Existing dirty patches are accepted only if included in the baseline.
        path.write_text("weight: 2\n", encoding="utf-8")
        commit = subprocess.check_output(["git", "-C", str(devkit), "rev-parse", "HEAD"], text=True).strip()
        snapshot = folder / "snapshot.json"
        snapshot.write_text(json.dumps({"devkit_base_commit": commit, "tracked_devkit_sha256": {"scoring.yaml": digest(path)}}), encoding="utf-8")
        return devkit, path, snapshot

    def test_existing_patch_bound_and_subsequent_change_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _, path, snapshot = self.fixture(folder)
            self.assertEqual(upstream_binding(folder, snapshot)[str(path)], digest(path))
            path.write_text("weight: 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source/config changed"):
                upstream_binding(folder, snapshot)

    def test_tracked_inventory_change_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            devkit, _, snapshot = self.fixture(folder)
            (devkit / "new.py").write_text("pass\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(devkit), "add", "new.py"], check=True)
            with self.assertRaisesRegex(ValueError, "checkout differs"):
                upstream_binding(folder, snapshot)


if __name__ == "__main__":
    unittest.main()
