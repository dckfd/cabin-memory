from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .base import DatasetJudge, JudgeConfig
from ..datasets.kvret_memory import KVRETMemoryDataset


def normalize_slot_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(re.findall(r"[a-z0-9]+", normalized))


def _canonical_times(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    result: set[str] = set()
    for hour, minute, period in re.findall(
            r"(?<!\d)(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\b",
            normalized,
        ):
        hour_value = int(hour) % 12
        if period.replace(".", "") == "pm":
            hour_value += 12
        result.add(f"{hour_value:02d}:{int(minute or '0'):02d}")
    for hour, minute in re.findall(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", normalized):
        if 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59:
            result.add(f"{int(hour):02d}:{int(minute):02d}")
    return result


def slot_value_matches(predicted: str, gold: str, *, exact: bool) -> bool:
    predicted_key = normalize_slot_value(predicted)
    gold_key = normalize_slot_value(gold)
    if not predicted_key or not gold_key:
        return False
    lexical = predicted_key == gold_key if exact else gold_key in predicted_key
    typo_equivalent = bool(
        predicted_key.isalpha()
        and gold_key.isalpha()
        and min(len(predicted_key), len(gold_key)) >= 6
        and _edit_distance_at_most_one(predicted_key, gold_key)
    )
    return (
        lexical
        or typo_equivalent
        or bool(_canonical_times(predicted) & _canonical_times(gold))
    )


def _edit_distance_at_most_one(first: str, second: str) -> bool:
    if first == second:
        return True
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        mismatches = [
            index for index, (left, right) in enumerate(zip(first, second))
            if left != right
        ]
        if len(mismatches) == 1:
            return True
        return bool(
            len(mismatches) == 2
            and mismatches[1] == mismatches[0] + 1
            and first[mismatches[0]] == second[mismatches[1]]
            and first[mismatches[1]] == second[mismatches[0]]
        )
    shorter, longer = (first, second) if len(first) < len(second) else (second, first)
    left = right = differences = 0
    while left < len(shorter) and right < len(longer):
        if shorter[left] == longer[right]:
            left += 1
            right += 1
        else:
            differences += 1
            right += 1
            if differences > 1:
                return False
    return True


class KVRETMemoryJudge(DatasetJudge):
    """Deterministic slot-value scoring; no LLM Judge tokens are required."""

    judge_id = "kvret_memory_deterministic_slot"
    supported_metrics = frozenset({"exact", "contains"})
    requires_model = False

    def __init__(self, *, dataset_root: Path) -> None:
        self.dataset = KVRETMemoryDataset(dataset_root)

    def score(
        self, predictions_path: Path, output_dir: Path, config: JudgeConfig,
    ) -> dict:
        self.validate_metrics(config.metrics)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows: list[dict] = []
        by_category: dict[str, list[dict]] = defaultdict(list)
        for prediction in predictions:
            qa_id = str(prediction["qa_id"])
            question = self.dataset.question(qa_id)
            predicted = str(prediction.get("predicted_answer") or "")
            scores = {
                "exact": any(
                    slot_value_matches(predicted, answer, exact=True)
                    for answer in question.answers
                ),
                "contains": any(
                    slot_value_matches(predicted, answer, exact=False)
                    for answer in question.answers
                ),
            }
            row = {
                "qa_id": qa_id,
                "category": question.category,
                "question": question.text,
                "gold_answers": list(question.answers),
                "predicted_answer": predicted,
                "scores": {metric: scores[metric] for metric in config.metrics},
            }
            rows.append(row)
            by_category[question.category].append(row)

        scored = output_dir / "predictions.scored.jsonl"
        scored.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary = {
            "judge_id": self.judge_id,
            "question_count": len(rows),
            "expected_question_count": len(self.dataset.questions()),
            "coverage": (
                len({row["qa_id"] for row in rows}) / len(self.dataset.questions())
                if self.dataset.questions() else None
            ),
            "metrics": {
                metric: {
                    "correct": sum(row["scores"][metric] for row in rows),
                    "mean": (
                        sum(row["scores"][metric] for row in rows) / len(rows)
                        if rows else None
                    ),
                }
                for metric in config.metrics
            },
            "by_category": {
                category: {
                    "total": len(values),
                    **{
                        metric: sum(row["scores"][metric] for row in values)
                        for metric in config.metrics
                    },
                }
                for category, values in sorted(by_category.items())
            },
            "scored": str(scored),
            "uses_llm": False,
        }
        (output_dir / "score-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
