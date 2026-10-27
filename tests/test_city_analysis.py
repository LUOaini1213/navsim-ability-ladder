import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_city import METRICS, read_scores, summarize, tags
from verify_mechanism_core import digest


class CityAnalysis(unittest.TestCase):
    def fixture(self, folder):
        # These tiny files are fixture provenance only, not training/scientific evidence.
        base, city_path, audit = folder / "base/protocol.json", folder / "city/protocol.json", folder / "audit.json"
        base.parent.mkdir()
        city_path.parent.mkdir()
        for path in (base, city_path, audit):
            path.write_text('{"fixture":true}', encoding="utf-8")
        (city_path.parent / "donors.npz").write_bytes(b"fixture donors")
        source = {"sha256": "fixture-features"}
        city = {"sources": {"navtrain": source}}
        core = {"source_sha256": {"analysis/train_mechanism.py": "fixture"}, "donors_sha256": "base-donors", "runs": []}
        auxiliary = {}
        scores, predictions, originals = folder / "scores", folder / "predictions", folder / "originals"
        for path in (scores, predictions, originals):
            path.mkdir()
        base_tokens = np.array(["a", "b", "c"])
        base_logs = np.array(["drive-a_0_1", "drive-b_0_1", "drive-c_0_1"])
        tokens, logs = base_tokens[[0, 2]], base_logs[[0, 2]]
        settings = {"split": "validation", "limit": None, "runs": {},
                    "wrapper_source_sha256": {name: digest(ROOT / "analysis" / name) for name in ("score_mechanism.py", "mechanism_scoring.py")}}
        complete = {"split": "validation", "eligible_for_paper": True, "final_invalid_scores": 0, "runs": {}}
        config_names = {"C1": "M1", "C2": "M2", "C4": "M4", "C5": "M5"}
        for tag in sorted(tags()):
            condition = tag.split("_")[0]
            seed = int(tag.rsplit("s", 1)[1])
            original_tag = f"{config_names[condition]}_f1_steps_s{seed}"
            model = {"tag": original_tag, "checkpoint_sha256": "checkpoint-" + original_tag,
                     "normalization_sha256": "normalization", "settings_sha256": "settings-" + original_tag}
            anchor = condition in ("C1", "C4")
            if anchor:
                core["runs"].append(model)
            else:
                auxiliary[condition + "_s" + str(seed)] = model
            poses = np.full((2, 8, 3), seed, np.float32)
            meta = {"schema": "same-city-auxiliary-predictions-v1", "run": tag, "original_run": original_tag,
                    "condition": condition, "seed": seed, "split": "validation", "scenes": 2, "eligible_for_paper": True, "fallbacks": 0,
                    "protocol_sha256": digest(city_path), "base_protocol_sha256": digest(base),
                    "donors_sha256": digest(city_path.parent / "donors.npz"), "checkpoint_sha256": model["checkpoint_sha256"],
                    "normalization_sha256": model["normalization_sha256"], "training_settings_sha256": model["settings_sha256"],
                    "features_sha256": source["sha256"], "source_sha256": core["source_sha256"], "city_input_audit_sha256": digest(audit),
                    "exporter_sha256": digest(ROOT / "analysis/predict_city.py")}
            if anchor:
                original_meta = dict(meta, run=original_tag, scenes=3, protocol_sha256=digest(base), donors_sha256=core["donors_sha256"])
                original = originals / (original_tag + ".npz")
                np.savez_compressed(original, token=base_tokens, log_name=base_logs,
                                    poses=np.full((3, 8, 3), seed, np.float32), meta=np.array(json.dumps(original_meta)))
                receipt = original.with_suffix(".completed.json")
                receipt.write_text(json.dumps({"predictions_sha256": digest(original), "meta": original_meta}), encoding="utf-8")
                meta.update(anchor_original_predictions_sha256=digest(original), anchor_original_receipt_sha256=digest(receipt),
                            anchor_original_metadata=original_meta)
            prediction = predictions / (tag + ".npz")
            np.savez_compressed(prediction, token=tokens, log_name=logs, poses=poses, meta=np.array(json.dumps(meta)))
            prediction.with_suffix(".completed.json").write_text(json.dumps({"predictions_sha256": digest(prediction), "meta": meta}), encoding="utf-8")
            settings["runs"][tag] = {"predictions_sha256": digest(prediction), "meta": meta}
            value = {"C1": .5, "C2": .4, "C4": .8, "C5": .5}[condition]
            rows = [dict(token=token, log_name=log, valid=True, **{metric: value for metric in METRICS}) for token, log in zip(tokens, logs)]
            path = scores / (tag + ".csv.gz")
            self.write_csv(path, rows)
            complete["runs"][tag] = {"n": 2, "mean_score": value, "csv_sha256": digest(path)}
        self.seal_scores(scores, settings, complete)
        return scores, predictions, originals, base, city_path, audit, core, city, auxiliary, tokens, logs, base_tokens, base_logs

    @staticmethod
    def write_csv(path, rows):
        with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def seal_scores(scores, settings, complete):
        (scores / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        complete["settings_sha256"] = digest(scores / "settings.json")
        (scores / "completed.json").write_text(json.dumps(complete), encoding="utf-8")

    def test_all_12_cells_common_coverage_and_real_anchor_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            arrays, provenance, split = read_scores(*args)
            self.assertEqual(set(arrays), tags())
            self.assertEqual(len(provenance), 12)
            self.assertEqual(split, "validation")
            summary = summarize(arrays, args[10], {"all": [True, True], "empty": [False, False]}, 100)
            self.assertAlmostEqual(summary["contrasts"]["C4-C5-C1+C2"]["metric_effects"]["score"], .2)
            self.assertEqual(summary["contrasts"]["C1-C2"]["auxiliary_family_interval"]["alpha"], .05 / 3)
            self.assertEqual(summary["subgroups"]["empty"]["contrasts"], {})

    def test_resealed_wrong_real_anchor_poses_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            scores, predictions = args[:2]
            tag = "C1_f1_steps_s0"
            path = predictions / (tag + ".npz")
            with np.load(path, allow_pickle=False) as archive:
                contents = {key: archive[key] for key in archive.files}
            contents["poses"] = contents["poses"] + 1
            np.savez_compressed(path, **contents)
            receipt = {"meta": json.loads(str(contents["meta"])), "predictions_sha256": digest(path)}
            path.with_suffix(".completed.json").write_text(json.dumps(receipt), encoding="utf-8")
            settings = json.loads((scores / "settings.json").read_text(encoding="utf-8"))
            complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
            settings["runs"][tag]["predictions_sha256"] = digest(path)
            self.seal_scores(scores, settings, complete)
            with self.assertRaisesRegex(ValueError, "filter/poses mismatch"):
                read_scores(*args)

    def test_resealed_wrong_logs_and_nonfinite_subscores_rejected(self):
        for change, message in (({"log_name": "wrong"}, "token-log coverage"), ({"comfort": "nan"}, "Invalid city metric")):
            with tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                scores, tag = args[0], "C2_f1_steps_s0"
                path = scores / (tag + ".csv.gz")
                with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                rows[0].update(change)
                self.write_csv(path, rows)
                complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
                complete["runs"][tag]["csv_sha256"] = digest(path)
                (scores / "completed.json").write_text(json.dumps(complete), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    read_scores(*args)

    def test_wrong_matrix_or_source_metadata_rejected_even_if_resealed(self):
        for change in ("matrix", "source"):
            with tempfile.TemporaryDirectory() as temporary:
                args = self.fixture(Path(temporary))
                scores = args[0]
                settings = json.loads((scores / "settings.json").read_text(encoding="utf-8"))
                complete = json.loads((scores / "completed.json").read_text(encoding="utf-8"))
                if change == "matrix":
                    del settings["runs"]["C5_f1_steps_s2"]
                else:
                    settings["runs"]["C2_f1_steps_s0"]["meta"]["normalization_sha256"] = "different"
                self.seal_scores(scores, settings, complete)
                with self.assertRaises(ValueError):
                    read_scores(*args)

    def test_summary_rejects_missing_seeds_or_unaligned_masks(self):
        arrays = {tag: np.full((2, len(METRICS)), .5) for tag in tags()}
        with self.assertRaisesRegex(ValueError, "mask misalignment"):
            summarize(arrays, ["session-a_0_1", "session-b_0_1"], {"bad": [True]}, 100)
        del arrays["C2_f1_steps_s1"]
        with self.assertRaisesRegex(ValueError, "Exactly 12"):
            summarize(arrays, ["session-a_0_1", "session-b_0_1"], {}, 100)


if __name__ == "__main__":
    unittest.main()
