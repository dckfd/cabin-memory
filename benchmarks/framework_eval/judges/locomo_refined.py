from __future__ import annotations

from pathlib import Path

from .base import DatasetJudge, JudgeConfig
from ..scoring import score_locomo_predictions


class LoCoMoRefinedJudge(DatasetJudge):
    judge_id = "locomo_refined_official"
    supported_metrics = frozenset({"llm", "f1", "bleu"})
    requires_model = True

    def __init__(self, *, locomo_root: Path) -> None:
        self.locomo_root = locomo_root

    def score(self, predictions_path: Path, output_dir: Path,
              config: JudgeConfig) -> dict:
        self.validate_metrics(config.metrics)
        # The official scorer resolves model endpoint/key from its supported
        # environment when they are omitted. Explicit values are forwarded by
        # the scorer CLI in a later compatibility layer, never serialized.
        return score_locomo_predictions(
            predictions_path=predictions_path,
            output_dir=output_dir,
            locomo_root=self.locomo_root,
            metrics=config.metrics,
            concurrency=config.concurrency,
            llm_judge=str(config.metadata.get("llm_judge", "refined")),
            evaluator_model=config.model,
            evaluator_base_url=config.base_url,
            evaluator_api_key=config.api_key,
        )
