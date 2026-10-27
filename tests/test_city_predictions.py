import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from predict_city import select_anchor


class CityPredictions(unittest.TestCase):
    def test_exact_input_only_subset_and_registered_order(self):
        tokens = np.array(["a", "b", "c"])
        logs = np.array(["log-a", "log-b", "log-c"])
        poses = np.arange(72, dtype=np.float32).reshape(3, 8, 3)
        selected = select_anchor(tokens, logs, poses, np.array(["c", "a"]), np.array(["log-c", "log-a"]))
        np.testing.assert_array_equal(selected, poses[[2, 0]])

    def test_duplicate_missing_or_misaligned_subset_is_rejected(self):
        poses = np.zeros((2, 8, 3), dtype=np.float32)
        for tokens, wanted, logs in ((["a", "a"], ["a"], ["l"]),
                                    (["a", "b"], ["x"], ["l"]),
                                    (["a", "b"], ["a", "a"], ["l", "l"]),
                                    (["a", "b"], ["a"], ["wrong"])):
            with self.assertRaises(ValueError):
                select_anchor(np.array(tokens), np.array(["l", "l"]), poses, np.array(wanted), np.array(logs))

    def test_nonfinite_or_wrong_shape_predictions_not_promoted(self):
        for poses in (np.full((1, 8, 3), np.nan), np.zeros((1, 7, 3))):
            with self.assertRaises(ValueError):
                select_anchor(np.array(["a"]), np.array(["l"]), poses, np.array(["a"]), np.array(["l"]))

    def test_empty_subset_not_a_complete_evaluation(self):
        with self.assertRaises(ValueError):
            select_anchor(np.array(["a"]), np.array(["l"]), np.zeros((1, 8, 3)), np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()
