import copy
import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
try:
    import numpy as np
    from analyze_mechanism import cluster_interval, summarize, read_score_archive, METRICS
    from mechanism_data import digest, CONFIGS
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires optional numpy")
class MechanismAnalysis(unittest.TestCase):
    def test_ratio_estimator_keeps_scene_weighting(self):
        row = cluster_interval([0, 1, 1, 1], ["a", "b", "b", "b"], 1000)
        self.assertEqual(row["mean"], .75)
        self.assertEqual((row["low"], row["high"]), (0., 1.))

    def test_single_session_does_not_get_zero_width_certainty(self):
        row = cluster_interval([1, 1], ["a", "a"], 100)
        self.assertIsNone(row["low"])
        self.assertEqual(row["status"], "insufficient_clusters")

    def test_multiplicity_interval_is_no_narrower(self):
        normal = cluster_interval([0, .2, .3, .9], list("abcd"), 10000)
        adjusted = cluster_interval([0, .2, .3, .9], list("abcd"), 10000, .05/3)
        self.assertLessEqual(adjusted["low"], normal["low"])
        self.assertGreaterEqual(adjusted["high"], normal["high"])
        self.assertEqual(normal, cluster_interval([0, .2, .3, .9], list("abcd"), 10000))

    def fixture(self):
        return {f"{config}_f1_steps_s{seed}": np.full((4, 7), .1*int(config[1])+seed*.01)
                for config in CONFIGS for seed in range(3)}

    def test_paired_seed_scores_not_trajectory_ensembles(self):
        result = summarize(self.fixture(), ["driveA_0_1"]*2+["driveB_0_1"]*2,
                           {"all": np.ones(4, bool), "empty": np.zeros(4, bool)},
                           ["M1-M0", "M4-M1", "M4-M5"], [0, 1, 2], 100)
        self.assertAlmostEqual(result["contrasts"]["M4-M1"]["metric_effects"]["score"], .3)
        self.assertEqual(result["subgroups"]["empty"]["n_scenes"], 0)
        self.assertEqual(len(result["contrasts"]["M4-M1"]["seed_effects"]), 3)
        self.assertAlmostEqual(result["contrasts"]["M4-M1"]["multiplicity_interval"]["alpha"], .05/3)

    def archive(self, folder):
        tokens, logs = ["a", "b"], ["driveA_0_1", "driveB_0_1"]
        protocol = {"training": {"seeds": [0, 1, 2]}, "sources": {"navtest": {"sha256": "features"}}}
        settings, completed = {"limit": None, "runs": {}}, {"eligible_for_paper": True, "runs": {}, "split": "navtest"}
        root = Path(__file__).resolve().parents[1]
        source_hashes = {name: digest(root / name) for name in ("agents/mechanism_model.py", "analysis/mechanism_data.py", "analysis/train_mechanism.py")}
        for config in CONFIGS:
            for seed in range(3):
                run = f"{config}_f1_steps_s{seed}"
                path = folder/(run+".csv.gz")
                with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["token", "log_name", "valid", *METRICS])
                    writer.writeheader()
                    for token, log in zip(tokens, logs):
                        writer.writerow(dict(token=token, log_name=log, valid=True, **dict.fromkeys(METRICS, .5)))
                completed["runs"][run] = {"n": 2, "mean_score": .5, "csv_sha256": digest(path)}
                settings["runs"][run] = {"predictions_sha256": "predictions", "meta": {"run": run, "eligible_for_paper": True,
                    "protocol_sha256": "protocol", "split": "navtest", "fallbacks": 0, "scenes": 2, "features_sha256": "features",
                    "checkpoint_sha256": "model", "normalization_sha256": "normalizer", "training_settings_sha256": "training",
                    "donors_sha256": "donors", "source_sha256": source_hashes}}
        (folder/"settings.json").write_text(json.dumps(settings), encoding="utf-8")
        completed["settings_sha256"] = digest(folder/"settings.json")
        (folder/"completed.json").write_text(json.dumps(completed), encoding="utf-8")
        return protocol, tokens, logs, completed

    def test_exact_coverage_and_hashes_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            protocol, tokens, logs, complete = self.archive(folder)
            arrays, provenance, split = read_score_archive(folder, protocol, "protocol", tokens, logs)
            self.assertEqual((len(arrays), split), (18, "navtest"))
            with self.assertRaisesRegex(ValueError, "misalignment"):
                read_score_archive(folder, protocol, "protocol", tokens, logs[::-1])
            complete["runs"].pop("M0_f1_steps_s0")
            (folder/"completed.json").write_text(json.dumps(complete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "18 core"):
                read_score_archive(folder, protocol, "protocol", tokens, logs)

    def test_smoke_is_never_promoted(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            protocol, tokens, logs, complete = self.archive(folder)
            complete["eligible_for_paper"] = False
            (folder/"completed.json").write_text(json.dumps(complete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Smoke"):
                read_score_archive(folder, protocol, "protocol", tokens, logs)
