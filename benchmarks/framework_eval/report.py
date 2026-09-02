from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_run(retrieval_path: Path, predictions_path: Path | None = None) -> dict:
    retrieval = _rows(retrieval_path)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in retrieval:
        by_category[str(row["question"].get("category") or "unknown")].append(row)

    def retrieval_metrics(rows: list[dict]) -> dict:
        recalls = [row["metrics"]["evidence_recall"] for row in rows
                   if row["metrics"].get("evidence_recall") is not None]
        return {
            "questions": len(rows),
            "mean_evidence_recall": sum(recalls) / len(recalls) if recalls else None,
            "mean_context_chars": (
                sum(row["metrics"]["context_chars"] for row in rows) / len(rows) if rows else None
            ),
            "mean_search_seconds": (
                sum(row["metrics"]["search_seconds"] for row in rows) / len(rows) if rows else None
            ),
        }

    result = {
        "framework": retrieval[0]["framework"] if retrieval else None,
        "retrieval": retrieval_metrics(retrieval),
        "by_category": {key: retrieval_metrics(value) for key, value in sorted(by_category.items())},
    }
    if predictions_path is not None and predictions_path.exists():
        predictions = _rows(predictions_path)
        usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
        result["answering"] = {
            "questions": len(predictions),
            "tokens": {key: sum(int((row.get("usage") or {}).get(key, 0)) for row in predictions)
                       for key in usage_keys},
        }
    return result


def write_markdown(summary: dict, path: Path) -> None:
    retrieval = summary["retrieval"]
    lines = [
        f"# {summary.get('framework') or 'Unknown'} evaluation summary",
        "",
        f"- Questions: {retrieval['questions']}",
        f"- Mean evidence recall: {_fmt(retrieval['mean_evidence_recall'])}",
        f"- Mean context characters: {_fmt(retrieval['mean_context_chars'])}",
        f"- Mean search seconds: {_fmt(retrieval['mean_search_seconds'])}",
        "",
        "## By category",
        "",
        "| Category | Questions | Evidence recall | Context chars |",
        "|---|---:|---:|---:|",
    ]
    for category, row in summary["by_category"].items():
        lines.append(f"| {category} | {row['questions']} | {_fmt(row['mean_evidence_recall'])} | {_fmt(row['mean_context_chars'])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"
