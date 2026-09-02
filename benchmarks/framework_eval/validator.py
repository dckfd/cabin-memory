from __future__ import annotations

import json
from pathlib import Path


def validate_retrieval(path: Path, *, expected_count: int | None = None) -> dict:
    errors, warnings, ids = [], [], set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            question = row.get("question") or {}
            qa_id = str(question.get("question_id") or "")
            if not qa_id:
                errors.append(f"line {line_number}: missing question.question_id")
            elif qa_id in ids:
                errors.append(f"line {line_number}: duplicate qa_id {qa_id}")
            ids.add(qa_id)
            if not isinstance(row.get("hits"), list):
                errors.append(f"line {line_number}: hits must be a list")
            elif not row["hits"]:
                warnings.append(f"line {line_number}: empty retrieval")
            metrics = row.get("metrics") or {}
            if "context_chars" not in metrics or "search_seconds" not in metrics:
                errors.append(f"line {line_number}: incomplete retrieval metrics")
    if expected_count is not None and rows != expected_count:
        errors.append(f"expected {expected_count} rows, found {rows}")
    return {
        "artifact": str(path),
        "kind": "retrieval",
        "row_count": rows,
        "expected_count": expected_count,
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
    }
