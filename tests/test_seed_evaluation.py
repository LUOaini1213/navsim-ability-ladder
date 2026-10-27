import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from run_seed_evaluation import prediction_matrix, scoring_command


class SeedEvaluation(unittest.TestCase):
    def test_exact_thirty_six_added_exports_no_seed_selection_or_core_relabel(self):
        rows = prediction_matrix()
        self.assertEqual(len(rows), 36)
        self.assertEqual(len(set(rows)), 36)
        self.assertEqual({row[2] for row in rows}, {3, 4})
        self.assertEqual({row[1] for row in rows}, {"M0", "M1", "M2", "M3", "M4", "M5"})
        self.assertEqual({row[0] for row in rows}, {"validation", "confirmation", "navtest"})

    def test_generic_official_scorer_has_no_smoke_limit_and_separate_outputs(self):
        predictions = [Path("seed_predictions") / (f"M{config}_f1_steps_s{seed}.npz")
                       for seed in (3, 4) for config in range(6)]
        command = list(map(str, scoring_command("python", predictions, Path("seed_scores"))))
        self.assertNotIn("--limit", command)
        self.assertEqual(len(command[command.index("--predictions") + 1:command.index("--out")]), 12)
        self.assertTrue(command[2].replace("\\", "/").endswith("analysis/score_mechanism.py"))
        self.assertEqual(command[-1], "seed_scores")

    def test_missing_duplicate_core_seed_or_mixed_split_batch_rejected(self):
        original = [Path("seed_predictions/validation") / (f"M{config}_f1_steps_s{seed}.npz")
                    for seed in (3, 4) for config in range(6)]
        variants = [original[:-1], [original[0], *original[1:-1], original[0]],
                    [Path("seed_predictions/validation/M0_f1_steps_s0.npz"), *original[1:]],
                    [Path("seed_predictions/confirmation/M0_f1_steps_s3.npz"), *original[1:]]]
        for paths in variants:
            with self.assertRaisesRegex(ValueError, "Exactly twelve"):
                scoring_command("python", paths, Path("seed_scores"))


if __name__ == "__main__":
    unittest.main()
