from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from .base import DatasetJudge, JudgeConfig
from ..datasets.longmemeval import LongMemEvalDataset


class JudgeLLM(Protocol):
    def complete(self, prompt: str, config: JudgeConfig) -> tuple[str, dict]: ...


class OpenAIJudgeLLM:
    def complete(self, prompt: str, config: JudgeConfig) -> tuple[str, dict]:
        if not config.base_url or not config.model:
            raise ValueError("LongMemEval LLM judge requires base_url and model")
        body = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 16,
            "enable_thinking": False,
        }
        if "deepseek" in config.model.lower() or "deepseek" in config.base_url.lower():
            # DeepSeek V4 uses the OpenAI-compatible `thinking` object.  Its
            # older `enable_thinking` extension is silently ignored, leaving
            # short judge responses with reasoning tokens but empty content.
            body["thinking"] = {"type": "disabled"}
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = urllib.request.Request(
            config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=float(config.metadata.get("timeout", 180))
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"]), dict(payload.get("usage", {}))


class LongMemEvalJudge(DatasetJudge):
    judge_id = "longmemeval_official_task_prompts"
    supported_metrics = frozenset({"llm"})
    requires_model = True

    def __init__(self, *, dataset_root: Path, client: JudgeLLM | None = None) -> None:
        self.dataset = LongMemEvalDataset(dataset_root)
        self.client = client or OpenAIJudgeLLM()

    def score(self, predictions_path: Path, output_dir: Path,
              config: JudgeConfig) -> dict:
        self.validate_metrics(config.metrics)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "judge-results.jsonl"
        predictions = _read_jsonl(predictions_path)
        completed: dict[str, dict] = {}
        if config.resume and output.exists():
            completed = {str(row["qa_id"]): row for row in _read_jsonl(output)
                         if row.get("label") is not None}
        mode = "a" if config.resume else "w"
        results = list(completed.values())
        with output.open(mode, encoding="utf-8") as handle:
            for prediction in predictions:
                qa_id = str(prediction["qa_id"])
                if qa_id in completed:
                    continue
                question = self.dataset.question(qa_id)
                prompt = official_prompt(
                    question=question.text,
                    answers=question.answers,
                    response=str(prediction.get("predicted_answer", "")),
                    category=question.category,
                    abstention=bool(question.metadata.get("is_abstention")),
                )
                started = time.monotonic()
                try:
                    raw, usage = self.client.complete(prompt, config)
                    label = parse_yes_no(raw)
                    row = {
                        "qa_id": qa_id,
                        "category": question.category,
                        "label": label,
                        "judge_response": raw,
                        "judge_model": config.model,
                        "usage": usage,
                        "latency_seconds": time.monotonic() - started,
                        "error": None,
                    }
                except Exception as exc:
                    row = {
                        "qa_id": qa_id,
                        "category": question.category,
                        "label": None,
                        "judge_model": config.model,
                        "usage": {},
                        "latency_seconds": time.monotonic() - started,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                results.append(row)
        summary = summarize(results, expected_count=len(predictions))
        summary.update({"judge_id": self.judge_id, "judge_model": config.model,
                        "results": str(output)})
        (output_dir / "score-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def official_prompt(*, question: str, answers: tuple[str, ...], response: str,
                    category: str, abstention: bool) -> str:
    answer = "; ".join(answers)
    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. The model "
            "could say that the information is incomplete, or some other information is given but the asked "
            "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        ).format(question, answer, response)
    if category == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response from a "
            "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. The response is correct as long "
            "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
            "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        ).format(question, answer, response)
    if category == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. Please answer yes if "
            "the response contains the correct answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response should be considered as correct "
            "as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}"
            "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        ).format(question, answer, response)
    temporal = (
        " In addition, do not penalize off-by-one errors for the number of days. If the question asks for "
        "the number of days/weeks/months, etc., and the model makes an off-by-one error, the response is "
        "still correct."
        if category == "temporal-reasoning" else ""
    )
    return (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
        "correct answer or contains all the intermediate steps to get the correct answer, you should also "
        "answer yes. If the response only contains a subset of the information required by the answer, answer "
        "no.{}\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
        "correct? Answer yes or no only."
    ).format(temporal, question, answer, response)


def parse_yes_no(text: str) -> bool:
    normalized = text.strip().lower().rstrip(".!")
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    raise ValueError(f"judge returned neither yes nor no: {text[:100]!r}")


def summarize(rows: list[dict], *, expected_count: int) -> dict:
    latest = {str(row["qa_id"]): row for row in rows}
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in latest.values():
        by_category[str(row.get("category") or "unknown")].append(row)
    correct = sum(row.get("label") is True for row in latest.values())
    return {
        "expected_count": expected_count,
        "result_count": len(latest),
        "correct": correct,
        "errors": sum(row.get("label") is None for row in latest.values()),
        "accuracy": correct / expected_count if expected_count else None,
        "by_category": {
            key: {
                "total": len(values),
                "correct": sum(row.get("label") is True for row in values),
            }
            for key, values in sorted(by_category.items())
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
