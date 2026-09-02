from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameworkSpec:
    framework_id: str
    name: str
    repository: str | None
    local_path: str | None
    license: str
    adapter: str
    status: str
    notes: str
    metadata: dict[str, Any]


def load_registry(path: Path) -> dict[str, FrameworkSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = {}
    for row in data["frameworks"]:
        spec = FrameworkSpec(
            framework_id=row["id"],
            name=row["name"],
            repository=row.get("repository"),
            local_path=row.get("local_path"),
            license=row.get("license", "unknown"),
            adapter=row.get("adapter", "external_process"),
            status=row.get("status", "downloaded"),
            notes=row.get("notes", ""),
            metadata=dict(row.get("metadata", {})),
        )
        specs[spec.framework_id] = spec
    return specs
