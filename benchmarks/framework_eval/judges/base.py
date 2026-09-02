from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JudgeConfig:
    metrics: tuple[str, ...]
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    concurrency: int = 4
    resume: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetJudge(ABC):
    """Scoring belongs to the dataset, never to a memory framework."""

    judge_id = "abstract"
    supported_metrics: frozenset[str] = frozenset()
    requires_model = False

    def validate_metrics(self, metrics: tuple[str, ...]) -> None:
        unknown = set(metrics) - set(self.supported_metrics)
        if unknown:
            raise ValueError(
                f"judge {self.judge_id} does not support: {', '.join(sorted(unknown))}"
            )

    @abstractmethod
    def score(self, predictions_path: Path, output_dir: Path,
              config: JudgeConfig) -> dict:
        """Write auditable per-question scores and return a summary."""
