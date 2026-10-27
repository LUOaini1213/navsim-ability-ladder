"""Create/verify a local source snapshot, never a completed experimental release."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCHEMA = "local-reproduction-source-snapshot-v1"
SCHEMA = "local-reproduction-source-snapshot-v2"
DIRECTORIES = {
    "agents": {".py"}, "analysis": {".py", ".cjs"},
    "tests": {".py", ".cjs"}, "scripts": {".py", ".ps1", ".sh"},
    "docs": {".md", ".bib"},
}
ROOT_FILES = ("LICENSE", "README.md", "requirements.txt")
# Public bibliography evidence only; never extend this to arbitrary JSON files.
EXPLICIT_FILES = ("docs/reference_metadata_2026-10-09.json",)
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
EXCLUSIONS = ["Raw logs/maps/sensors/caches", "Model weights and derived score archives",
              "Reviewed-paper PDFs", "Private environment files and credentials",
              "Scientific completion or permission to publish/redistribute data"]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_name(name, schema=SCHEMA):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
        or any(part in (".", "..") for part in name.split("/")) or path.as_posix() != name):
        raise ValueError("Unsafe/noncanonical archive member")
    if name in ROOT_FILES:
        return
    if schema == SCHEMA and name in EXPLICIT_FILES:
        return
    if (len(path.parts) < 2 or path.parts[0] not in DIRECTORIES
        or path.suffix not in DIRECTORIES[path.parts[0]]
        or any(part.startswith(".") or part == "__pycache__" for part in path.parts)):
        raise ValueError("Member outside source-only allowlist")


def inventory(root):
    root = Path(root).resolve(strict=True)
    paths = [root / name for name in (*ROOT_FILES, *EXPLICIT_FILES)]
    for directory, suffixes in DIRECTORIES.items():
        base = root / directory
        if base.is_symlink():
            raise ValueError("Source directory symlink refused")
        if not base.is_dir():
            raise ValueError("Required source directory missing: " + directory)
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ValueError("Source symlink refused")
            if path.is_file() and path.suffix in suffixes:
                name = path.relative_to(root).as_posix()
                if any(part.startswith(".") or part == "__pycache__" for part in PurePosixPath(name).parts):
                    continue
                paths.append(path)
    rows, total = {}, 0
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Required source is missing or outside repository")
        name = path.relative_to(root).as_posix()
        safe_name(name)
        size = path.stat().st_size
        total += size
        if size > MAX_FILE or total > MAX_TOTAL:
            raise ValueError("Source-only snapshot size limit exceeded")
        rows[name] = {"bytes": size, "sha256": digest(path)}
    return rows


def build(root, output):
    root = Path(root).resolve(strict=True)
    output = Path(output)
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("Snapshot output symlink refused")
    output = output.resolve()
    allowed = root / "results/repro_snapshots"
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError("Use a fresh child directory under results/repro_snapshots; no overwrite")
    rows = inventory(root)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source.zip"
    # A failed attempt leaves its new, explicitly targeted partial directory intact.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in rows:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            executable = Path(name).suffix in {".sh", ".ps1"}
            info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
            with (root / name).open("rb") as source, bundle.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    if inventory(root) != rows:
        raise ValueError("Source inventory changed during snapshot; preserve partial attempt")
    manifest = {"schema": SCHEMA, "status": "source_snapshot_only",
                "experimental_reproduction_completed": False, "final_project_completed": False,
                "publication_authorized": False, "excluded": EXCLUSIONS,
                "files": rows, "archive_sha256": digest(archive),
                "scope": "Local byte-exact source/instruction snapshot; experiments, data licenses, scores, final manuscript and completion audit remain separate"}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_bundle(output)


def verify_bundle(folder):
    folder = Path(folder)
    manifest_path, archive = folder / "manifest.json", folder / "source.zip"
    if manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Manifest size limit exceeded")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if (schema not in (LEGACY_SCHEMA, SCHEMA) or manifest.get("status") != "source_snapshot_only"
        or any(manifest.get(key) is not False for key in ("experimental_reproduction_completed", "final_project_completed", "publication_authorized"))
        or manifest.get("excluded") != EXCLUSIONS):
        raise ValueError("Source snapshot cannot be promoted to experimental/final/publication completion")
    rows = manifest["files"]
    required = set(ROOT_FILES)
    if schema == SCHEMA:
        required.update(EXPLICIT_FILES)
    if not isinstance(rows, dict) or not required.issubset(rows):
        raise ValueError("Required source/license inventory missing")
    total = 0
    for name, row in rows.items():
        safe_name(name, schema)
        if set(row) != {"bytes", "sha256"} or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= MAX_FILE:
            raise ValueError("Malformed source inventory")
        total += row["bytes"]
    if total > MAX_TOTAL or archive.stat().st_size > MAX_TOTAL or digest(archive) != manifest["archive_sha256"]:
        raise ValueError("Source archive changed or exceeds size limit")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(rows) or set(member.filename for member in members) != set(rows):
            raise ValueError("Missing/extra/duplicate source archive member")
        for member in members:
            safe_name(member.filename, schema)
            row = rows[member.filename]
            if (member.file_size != row["bytes"] or member.is_dir()
                or stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1):
                raise ValueError("Wrong-size/directory/symlink/encrypted archive member")
            measured, sha = 0, hashlib.sha256()
            with bundle.open(member) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    measured += len(block)
                    if measured > row["bytes"]:
                        raise ValueError("Archive member exceeds declared size")
                    sha.update(block)
            if measured != row["bytes"] or sha.hexdigest() != row["sha256"]:
                raise ValueError("Archived source differs from manifest")
    return {"passed": True, "n_source_files": len(rows), "source_bytes": total,
            "archive_sha256": manifest["archive_sha256"], "manifest_sha256": digest(manifest_path),
            "scope": manifest["scope"], "final_project_completed": False}


def main():
    ap = argparse.ArgumentParser(__doc__)
    branch = ap.add_subparsers(dest="command", required=True)
    make = branch.add_parser("build")
    make.add_argument("--out", type=Path, required=True)
    check = branch.add_parser("verify")
    check.add_argument("folder", type=Path)
    args = ap.parse_args()
    report = build(ROOT, args.out) if args.command == "build" else verify_bundle(args.folder)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
