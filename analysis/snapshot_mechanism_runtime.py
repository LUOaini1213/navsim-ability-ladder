"""Record local reproducibility/resource evidence without importing ML runtimes.

This is a current snapshot, not proof that upstream files were unchanged before it.
No credentials, environment-variable dumps, dataset content or host name are saved.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_mechanism_core import available_mb
from verify_mechanism_core import ROOT, digest


def git(directory, *arguments):
    return subprocess.check_output(["git", "-C", str(directory), *arguments]).decode("utf-8").strip()


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--workspace", type=Path, default=Path("E:/navsim_workspace"))
    ap.add_argument("--out", type=Path, default=ROOT / "results/mechanism_v2/runtime_snapshot")
    args = ap.parse_args()
    devkit = args.workspace / "navsim"
    tracked = subprocess.check_output(["git", "-C", str(devkit), "ls-files", "-z"]).decode("utf-8").split("\0")
    hashes = {name: digest(devkit / name) for name in tracked if name and (devkit / name).is_file()}
    installed = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    canonical = {path.relative_to(ROOT).as_posix(): digest(path) for folder in ("agents", "analysis") for path in sorted((ROOT / folder).glob("*.py"))}
    cache = args.workspace / "exp/mechanism_v2/metric_cache"
    sizes = [path.stat().st_size for path in cache.rglob("metric_cache.pkl")] if cache.exists() else []
    disk = shutil.disk_usage(args.workspace)
    args.out.mkdir(parents=True, exist_ok=True)
    patch = subprocess.check_output(["git", "-C", str(devkit), "diff", "--binary", "HEAD", "--", "navsim/common/dataloader.py"])
    (args.out / "local_dataloader.patch").write_bytes(patch)
    report = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Current local environment/source/resource snapshot during core cache construction; not a pre-launch provenance claim or completed reproducibility package",
              "python": platform.python_version(), "os": platform.system(), "architecture": platform.machine(),
              "repo_base_commit": git(ROOT, "rev-parse", "HEAD"), "devkit_base_commit": git(devkit, "rev-parse", "HEAD"),
              "devkit_dirty_status": git(devkit, "status", "--porcelain"), "tracked_devkit_sha256": hashes,
              "canonical_source_sha256": canonical, "dependency_versions": dict(sorted(installed.items(), key=lambda item: item[0].lower())),
              "local_dataloader_patch_sha256": digest(args.out / "local_dataloader.patch"),
              "available_memory_mb_at_snapshot": available_mb(), "workspace_volume_free_bytes_at_snapshot": disk.free,
              "partial_new_metric_cache": {"files": len(sizes), "bytes": sum(sizes), "largest_bytes": max(sizes) if sizes else None,
                                           "completion": "Not inferred from these file counts; require complete per-split manifests and official token coverage"},
              "notes": ["Dependency inventory lists installed names/versions only; no direct-URL credentials or environment variables",
                        "Local tracked dataloader changes are preserved, not reverted; canonical repo agents overlay historical untracked devkit agents",
                        "Feature-source paths in the original frozen protocol are host-specific; relocation requires a separate protocol with disclosed provenance",
                        "Repository code license does not relicense NAVSIM, nuPlan/OpenScene/maps or primary-paper PDFs; dataset redistribution is not part of this work"]}
    (args.out / "runtime.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("CURRENT RUNTIME SNAPSHOT", report["python"], len(hashes), "tracked upstream hashes;", len(installed), "dependency versions", flush=True)


if __name__ == "__main__":
    main()
