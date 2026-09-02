from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_locomo_predictions(
    *,
    predictions_path: Path,
    output_dir: Path,
    locomo_root: Path,
    metrics: tuple[str, ...] = ("f1", "bleu"),
    concurrency: int = 4,
    llm_judge: str = "refined",
    evaluator_model: str | None = None,
    evaluator_base_url: str | None = None,
    evaluator_api_key: str | None = None,
) -> dict:
    """Score a complete or partial canonical prediction file with the official scorer.

    A partial question file is materialized beside the results. This preserves the
    official metric implementation without treating unselected questions as blank
    predictions.
    """
    predictions_path = predictions_path.resolve()
    output_dir = output_dir.resolve()
    locomo_root = locomo_root.resolve()
    predictions = _read_jsonl(predictions_path)
    if not predictions:
        raise ValueError(f"no predictions found in {predictions_path}")
    qa_ids = {str(row["qa_id"]) for row in predictions}
    official_questions = _read_jsonl(locomo_root / "data/public/questions.jsonl")
    selected = [row for row in official_questions if str(row.get("qa_id")) in qa_ids]
    selected_ids = {str(row["qa_id"]) for row in selected}
    missing = sorted(qa_ids - selected_ids)
    if missing:
        raise ValueError(f"unknown LoCoMo qa_id values: {', '.join(missing[:5])}")

    output_dir.mkdir(parents=True, exist_ok=True)
    question_path = output_dir / "questions.selected.jsonl"
    normalized_predictions = output_dir / "predictions.canonical.jsonl"
    scored_path = output_dir / "predictions.scored.jsonl"
    _write_jsonl(question_path, selected)
    _write_jsonl(
        normalized_predictions,
        ({"qa_id": str(row["qa_id"]), "predicted_answer": row["predicted_answer"]}
         for row in predictions),
    )

    command = [
        sys.executable,
        "-c",
        "from evaluate import main; main()",
        "--questions-path", str(question_path),
        "--predictions-path", str(normalized_predictions),
        "--output-path", str(scored_path),
        "--metrics", *metrics,
        "--concurrency", str(max(1, concurrency)),
        "--llm-judge", llm_judge,
        "--no-progress",
    ]
    env = os.environ.copy()
    if evaluator_model:
        env["EVALUATOR_MODEL"] = evaluator_model
    if evaluator_base_url:
        env["EVALUATOR_BASE_URL"] = evaluator_base_url
    if evaluator_api_key:
        env["EVALUATOR_API_KEY"] = evaluator_api_key
    source_dir = str((locomo_root / "src").resolve())
    env["PYTHONPATH"] = source_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        command,
        cwd=locomo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "official LoCoMo scorer failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    scored = _read_jsonl(scored_path)
    summary = summarize_scored(scored, metrics)
    summary.update({
        "predictions": str(predictions_path),
        "scored": str(scored_path),
        "questions": str(question_path),
        "official_scorer": str(locomo_root / "src/evaluate.py"),
        "evaluator_model": evaluator_model or env.get("EVALUATOR_MODEL") or "qwen3-14b",
        "stdout": completed.stdout.strip(),
    })
    summary_path = output_dir / "score-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_scored(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    result: dict = {"question_count": len(rows), "metrics": {}}
    for metric in metrics:
        field = f"{metric}_score"
        values: list[float] = []
        for row in rows:
            value = row.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        result["metrics"][metric] = {
            "mean": sum(values) / len(values) if values else None,
            "scored_count": len(values),
        }
    return result
