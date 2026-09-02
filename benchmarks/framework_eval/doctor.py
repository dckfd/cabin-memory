from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .registry import FrameworkSpec


def inspect_frameworks(root: Path, specs: dict[str, FrameworkSpec]) -> dict:
    rows = []
    for spec in specs.values():
        local = root / spec.local_path if spec.local_path else None
        exists = bool(local and local.exists())
        commit = None
        if exists and (local / ".git").exists() and shutil.which("git"):
            run = subprocess.run(
                ["git", "-C", str(local), "rev-parse", "--short=12", "HEAD"],
                text=True, capture_output=True, check=False,
            )
            commit = run.stdout.strip() if run.returncode == 0 else None
        license_files = []
        if exists:
            license_files = [str(path.relative_to(root)) for pattern in ("LICENSE*", "COPYING*")
                             for path in local.glob(pattern) if path.is_file()]
        rows.append({
            "id": spec.framework_id,
            "name": spec.name,
            "status": spec.status,
            "adapter": spec.adapter,
            "repository": spec.repository,
            "local_path": spec.local_path,
            "downloaded": exists,
            "commit": commit,
            "declared_license": spec.license,
            "license_files": sorted(set(license_files)),
            "notes": spec.notes,
        })
    return {
        "registry_version": 1,
        "framework_count": len(rows),
        "downloaded_count": sum(bool(row["downloaded"]) for row in rows),
        "frameworks": rows,
    }


def write_doctor_report(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
