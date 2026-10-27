import lzma
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from audit_metric_cache_recovery import corrupt_bytes, owned_target


class CacheRecoveryAudit(unittest.TestCase):
    def test_corruption_cannot_target_outside_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary) / "new-trial"
            target = trial / "log" / "unknown" / "token" / "metric_cache.pkl"
            self.assertEqual(owned_target(target, trial), target.resolve())
            with self.assertRaises(ValueError):
                owned_target(Path(temporary) / "old-cache" / "metric_cache.pkl", trial)
            with self.assertRaises(ValueError):
                owned_target(trial / "protocol.json", trial)

    def test_registered_faults_are_really_invalid(self):
        original = lzma.compress(pickle.dumps({"valid": True, "padding": "x" * 100}), preset=0)
        for kind in ("empty", "truncated_lzma", "invalid_pickle"):
            with self.subTest(kind=kind):
                with self.assertRaises((EOFError, lzma.LZMAError, pickle.UnpicklingError)):
                    pickle.loads(lzma.decompress(corrupt_bytes(kind, original)))
        with self.assertRaises(ValueError):
            corrupt_bytes("unregistered", original)


if __name__ == "__main__":
    unittest.main()
