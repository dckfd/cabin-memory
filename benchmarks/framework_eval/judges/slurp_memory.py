from __future__ import annotations

from pathlib import Path

from .kvret_memory import KVRETMemoryJudge
from ..datasets.slurp_memory import SLURPMemoryDataset


class SLURPMemoryJudge(KVRETMemoryJudge):
    """Deterministic entity-value scoring; it never calls a judge model."""

    judge_id = "slurp_memory_deterministic_entity"

    def __init__(self, *, dataset_root: Path) -> None:
        self.dataset = SLURPMemoryDataset(dataset_root)
