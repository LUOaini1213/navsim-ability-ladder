import hashlib
import json
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import repro_source_bundle as snapshot


class SourceBundleTests(unittest.TestCase):
    def fixture(self, root):
        for name in snapshot.ROOT_FILES:
            (root / name).write_text(name + "\n", encoding="utf-8")
        for name in snapshot.DIRECTORIES:
            (root / name).mkdir()
        for name in snapshot.EXPLICIT_FILES:
            (root / name).write_text('{"scope": "public bibliography fixture"}\n', encoding="utf-8")
        (root / "analysis/helper.py").write_text("print('fixture')\n", encoding="utf-8")
        return root / "results/repro_snapshots/owned-fixture"

    def test_snapshot_is_byte_exact_deterministic_and_not_scientific_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            first = snapshot.build(root, output)
            second = snapshot.build(root, output.with_name("second-owned-fixture"))
            self.assertEqual(first, second)
            self.assertTrue(first["passed"])
            self.assertFalse(first["final_project_completed"])
            self.assertEqual(first["n_source_files"], 5)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], snapshot.SCHEMA)
            self.assertTrue(set(snapshot.EXPLICIT_FILES).issubset(manifest["files"]))

    def test_existing_or_broad_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            output.mkdir(parents=True)
            marker = output / "user.txt"
            marker.write_text("retain", encoding="utf-8")
            for target in (output, root, root / "results/repro_snapshots"):
                with self.assertRaises(ValueError):
                    snapshot.build(root, target)
            self.assertEqual(marker.read_text(encoding="utf-8"), "retain")

    def test_sensitive_and_dataset_artifacts_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            for name in (".env", "analysis/private.env", "analysis/model.pt", "docs/paper.pdf",
                         "docs/private.json", "analysis/settings.json", "docs/other_metadata.json"):
                (root / name).write_bytes(b"do not copy")
            (root / "results").mkdir()
            (root / "results/raw.csv").write_bytes(b"not source")
            snapshot.build(root, output)
            with zipfile.ZipFile(output / "source.zip") as bundle:
                self.assertEqual(set(bundle.namelist()), {*snapshot.ROOT_FILES, *snapshot.EXPLICIT_FILES,
                                                         "analysis/helper.py"})

    def test_changed_zip_fails_and_resealed_manifest_does_not_hide_changed_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            snapshot.build(root, output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            archive = output / "source.zip"
            with zipfile.ZipFile(archive) as bundle:
                entries = {name: bundle.read(name) for name in bundle.namelist()}
            # Preserve platform newline bytes and size so this exercises hashing,
            # rather than merely detecting a CRLF-versus-LF size change.
            entries["analysis/helper.py"] = entries["analysis/helper.py"].replace(b"fixture", b"altered")
            with zipfile.ZipFile(archive, "w") as bundle:
                for name, data in entries.items():
                    bundle.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "archive changed"):
                snapshot.verify_bundle(output)
            manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                snapshot.verify_bundle(output)

    def test_path_traversal_and_unsupported_members_refused(self):
        for name in ("../secret.py", "/analysis/file.py", "analysis\\file.py", "analysis/../file.py",
                     "C:/analysis/file.py", "analysis/.env", "results/score.csv", "docs/paper.pdf",
                     "analysis//file.py", "analysis/__pycache__/file.py", "docs/private.json",
                     "docs/subfolder/reference_metadata_2026-10-09.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                snapshot.safe_name(name)

    def test_required_public_metadata_missing_refuses_build_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            (root / snapshot.EXPLICIT_FILES[0]).unlink()
            with self.assertRaisesRegex(ValueError, "Required source is missing"):
                snapshot.build(root, output)
            self.assertFalse(output.exists())

    def rewrite_without_public_metadata(self, output, schema):
        archive = output / "source.zip"
        with zipfile.ZipFile(archive) as bundle:
            entries = {name: bundle.read(name) for name in bundle.namelist()
                       if name not in snapshot.EXPLICIT_FILES}
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, data in entries.items():
                bundle.writestr(name, data)
        path = output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema"] = schema
        for name in snapshot.EXPLICIT_FILES:
            manifest["files"].pop(name)
        manifest["archive_sha256"] = snapshot.digest(archive)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_legacy_snapshot_remains_verifiable_without_modifying_its_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            snapshot.build(root, output)
            self.rewrite_without_public_metadata(output, snapshot.LEGACY_SCHEMA)
            before = {name: snapshot.digest(output / name) for name in ("source.zip", "manifest.json")}
            checked = snapshot.verify_bundle(output)
            self.assertTrue(checked["passed"])
            self.assertEqual(checked["n_source_files"], 4)
            self.assertEqual(before, {name: snapshot.digest(output / name) for name in before})

    def test_current_snapshot_cannot_drop_public_metadata_even_after_resealing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            snapshot.build(root, output)
            self.rewrite_without_public_metadata(output, snapshot.SCHEMA)
            with self.assertRaisesRegex(ValueError, "Required source/license inventory missing"):
                snapshot.verify_bundle(output)

    def test_legacy_schema_cannot_claim_new_metadata_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            snapshot.build(root, output)
            path = output / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["schema"] = snapshot.LEGACY_SCHEMA
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside source-only allowlist"):
                snapshot.verify_bundle(output)

    def test_manifest_cannot_promote_source_snapshot_to_completed_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            snapshot.build(root, output)
            path = output / "manifest.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            for name in ("experimental_reproduction_completed", "final_project_completed", "publication_authorized"):
                changed = dict(original, **{name: True})
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "cannot be promoted"):
                    snapshot.verify_bundle(output)

    def test_changed_inventory_during_build_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            initial = snapshot.inventory(root)
            changed = {**initial, "analysis/extra.py": {"bytes": 1, "sha256": "x"}}
            with patch.object(snapshot, "inventory", side_effect=[initial, changed]):
                with self.assertRaisesRegex(ValueError, "changed during snapshot"):
                    snapshot.build(root, output)
            self.assertTrue((output / "source.zip").exists())
            self.assertFalse((output / "manifest.json").exists())

    def test_resealed_archive_with_missing_extra_or_duplicate_members_is_refused(self):
        for alteration in ("missing", "extra", "duplicate"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = self.fixture(root)
                snapshot.build(root, output)
                archive = output / "source.zip"
                with zipfile.ZipFile(archive) as bundle:
                    entries = {name: bundle.read(name) for name in bundle.namelist()}
                if alteration == "missing":
                    entries.pop("analysis/helper.py")
                elif alteration == "extra":
                    entries["analysis/extra.py"] = b"x"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(archive, "w") as bundle:
                        for name, data in entries.items():
                            bundle.writestr(name, data)
                        if alteration == "duplicate":
                            bundle.writestr("analysis/helper.py", entries["analysis/helper.py"])
                manifest_path = output / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["archive_sha256"] = snapshot.digest(archive)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Missing/extra/duplicate"):
                    snapshot.verify_bundle(output)

    def test_oversized_manifest_member_and_noninteger_size_are_refused_before_unzip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self.fixture(root)
            snapshot.build(root, output)
            path = output / "manifest.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            for size in (snapshot.MAX_FILE + 1, -1, True, 1.0):
                changed = json.loads(json.dumps(original))
                changed["files"]["analysis/helper.py"]["bytes"] = size
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Malformed source inventory"):
                    snapshot.verify_bundle(output)


if __name__ == "__main__":
    unittest.main()
