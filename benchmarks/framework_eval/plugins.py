from __future__ import annotations

import importlib
import json
import os
import shlex
from pathlib import Path
from typing import Any

from .adapters.base import MemoryAdapter
from .adapters.external_process import ExternalProcessAdapter
from .datasets.base import DatasetAdapter
from .judges.base import DatasetJudge


KNOWN_CAPABILITIES = frozenset({
    "ingest", "search", "answer", "reflect", "wait_until_ready",
    "update", "delete", "profile", "graph", "drill_down",
})


def load_symbol(reference: str):
    try:
        module_name, symbol_name = reference.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"plugin reference must be module:symbol: {reference}") from exc
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(f"plugin symbol not found: {reference}") from exc


class PluginCatalog:
    """Declarative plugin discovery for frameworks, datasets, and judges."""

    def __init__(self, *, root: Path, framework_profiles: Path, dataset_profiles: Path) -> None:
        self.root = root.resolve()
        self.frameworks = _index(framework_profiles, "profiles")
        self.datasets = _index(dataset_profiles, "datasets")

    def dataset_root(self, dataset_id: str, override: Path | None = None) -> Path:
        if override:
            return override.resolve()
        return (self.root / self.datasets[dataset_id]["root"]).resolve()

    def create_dataset(self, dataset_id: str, *, root: Path | None = None) -> DatasetAdapter:
        profile = self.datasets[dataset_id]
        cls = load_symbol(profile["adapter"])
        instance = cls(self.dataset_root(dataset_id, root))
        if not isinstance(instance, DatasetAdapter):
            raise TypeError(f"dataset plugin {dataset_id} must implement DatasetAdapter")
        return instance

    def create_judge(self, dataset_id: str, *, root: Path | None = None) -> DatasetJudge:
        profile = self.datasets[dataset_id]
        cls = load_symbol(profile["judge"])
        dataset_root = self.dataset_root(dataset_id, root)
        kwargs = _resolve_kwargs(profile.get("judge_kwargs", {}), dataset_root=dataset_root)
        instance = cls(**kwargs)
        if not isinstance(instance, DatasetJudge):
            raise TypeError(f"judge plugin for {dataset_id} must implement DatasetJudge")
        return instance

    def create_memory_adapter(
        self,
        framework_id: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "TDAI_API_KEY",
        external_command: str | None = None,
        external_cwd: Path | None = None,
    ) -> MemoryAdapter:
        profile = self.frameworks[framework_id]
        execution = profile["execution"]
        capabilities = frozenset(profile.get("capabilities", ["ingest", "search"]))
        unknown = capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(
                f"framework {framework_id} declares unknown capabilities: "
                f"{', '.join(sorted(unknown))}"
            )
        if execution == "in_process":
            cls = load_symbol(profile["bridge"])
            instance = cls()
        elif execution == "http":
            if not profile.get("bridge"):
                raise RuntimeError(f"framework {framework_id} has no HTTP bridge")
            if not base_url:
                raise ValueError(f"framework {framework_id} requires base_url")
            cls = load_symbol(profile["bridge"])
            instance = cls(base_url, api_key=os.getenv(api_key_env, ""), adapter_id=framework_id)
        elif execution == "external_process":
            command = (shlex.split(external_command) if external_command
                       else list(profile.get("command") or []))
            if not command:
                raise RuntimeError(f"framework {framework_id} has no implemented process bridge")
            instance = ExternalProcessAdapter(
                command,
                cwd=external_cwd,
                adapter_id=framework_id,
                capabilities=capabilities,
                timeout=int(profile.get("timeout", 600)),
            )
        else:
            raise RuntimeError(
                f"framework {framework_id} is declared as {execution} but has no adapter bridge"
            )
        instance.capabilities = capabilities
        return instance


def _index(path: Path, field: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[field]
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate plugin IDs in {path}")
    return result


def _resolve_kwargs(values: dict[str, Any], *, dataset_root: Path) -> dict[str, Any]:
    return {
        key: dataset_root if value == "$dataset_root" else value
        for key, value in values.items()
    }
