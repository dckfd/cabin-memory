from __future__ import annotations

import json
import subprocess
from pathlib import Path


def verify_locked_sources(root: Path, lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    rows = []
    for source in lock["sources"]:
        path = root / source["path"]
        actual = None
        error = None
        if not path.exists():
            error = "missing"
        elif not (path / ".git").exists():
            error = "not-a-git-checkout"
        else:
            run = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True, capture_output=True, check=False,
            )
            if run.returncode:
                error = run.stderr.strip()[-500:] or "git-rev-parse-failed"
            else:
                actual = run.stdout.strip()
                if not actual.startswith(str(source["commit"])):
                    error = "commit-mismatch"
        rows.append({
            **source,
            "exists": path.exists(),
            "actual_commit": actual,
            "pass": error is None,
            "error": error,
        })
    return {
        "lock_version": lock["version"],
        "source_count": len(rows),
        "verified_count": sum(row["pass"] for row in rows),
        "pass": all(row["pass"] for row in rows),
        "sources": rows,
        "unavailable": lock.get("unavailable", []),
    }
