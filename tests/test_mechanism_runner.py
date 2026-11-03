import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from run_mechanism_evaluation import acquire


class MechanismRunner(unittest.TestCase):
    def child_code(self, path, wait):
        return ("import sys; from pathlib import Path; sys.path.insert(0, "+repr(str(ROOT / "analysis"))+"); "
                "from run_mechanism_evaluation import acquire\n"
                "try:\n    lock=acquire(Path("+repr(str(path))+"), wait="+str(wait)+")\n"
                "except OSError:\n    sys.exit(7)\n"
                "else:\n    lock.close(); sys.exit(0)\n")

    def test_os_lock_rejects_another_process_not_stale_pids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.lock"
            lock = acquire(path)
            try:
                result = subprocess.run([sys.executable, "-c", self.child_code(path, False)], timeout=5)
                self.assertEqual(result.returncode, 7)
            finally:
                lock.close()
            result = subprocess.run([sys.executable, "-c", self.child_code(path, False)], timeout=5)
            self.assertEqual(result.returncode, 0)

    def test_wait_acquires_only_after_current_owner_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.lock"
            lock = acquire(path)
            timer = threading.Timer(.5, lock.close)
            timer.start()
            try:
                result = subprocess.run([sys.executable, "-c", self.child_code(path, True)], timeout=6)
                self.assertEqual(result.returncode, 0)
            finally:
                timer.join()


if __name__ == "__main__":
    unittest.main()
