#!/usr/bin/env python3
"""Dependency-free validation for the portable cockpit JSONL contract."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.framework_eval.datasets.cockpit_jsonl import CockpitJSONLDataset


def _rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{number}: row must be an object")
            rows.append(value)
    return rows


def _datetime(value: object, location: str, errors: list[str]) -> None:
    text = str(value or "")
    if not text:
        errors.append(f"{location}: timestamp is required")
        return
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{location}: invalid ISO-8601 timestamp {text!r}")
        return
    if parsed.tzinfo is None:
        errors.append(f"{location}: timestamp must include a timezone offset")


def validate(root: Path) -> dict:
    conversations = _rows(root / "conversations.jsonl")
    questions = _rows(root / "questions.jsonl")
    errors: list[str] = []
    conversation_ids: set[str] = set()
    question_ids: set[str] = set()
    for row_number, row in enumerate(conversations, 1):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            errors.append(f"conversations.jsonl:{row_number}: sample_id is required")
        elif sample_id in conversation_ids:
            errors.append(f"conversations.jsonl:{row_number}: duplicate sample_id {sample_id}")
        conversation_ids.add(sample_id)
        sessions = row.get("sessions")
        if not isinstance(sessions, list) or not sessions:
            errors.append(f"conversations.jsonl:{row_number}: sessions must be non-empty")
            continue
        for session_number, session in enumerate(sessions, 1):
            session_id = session.get("source_session_id") or session.get("session_id")
            if not str(session_id or ""):
                errors.append(
                    f"conversations.jsonl:{row_number}.sessions[{session_number}]: "
                    "source_session_id or session_id is required"
                )
            _datetime(
                session.get("date_time") or session.get("timestamp"),
                f"conversations.jsonl:{row_number}.sessions[{session_number}]",
                errors,
            )
            messages = session.get("messages")
            if not isinstance(messages, list) or not messages:
                errors.append(
                    f"conversations.jsonl:{row_number}.sessions[{session_number}]: "
                    "messages must be non-empty"
                )
                continue
            for message_number, message in enumerate(messages, 1):
                location = (
                    f"conversations.jsonl:{row_number}.sessions[{session_number}]"
                    f".messages[{message_number}]"
                )
                if message.get("role") not in {"user", "assistant", "system", "tool"}:
                    errors.append(f"{location}: unsupported role {message.get('role')!r}")
                if not str(message.get("content") or ""):
                    errors.append(f"{location}: content is required")
    for row_number, row in enumerate(questions, 1):
        for field in ("qa_id", "sample_id", "question"):
            if not str(row.get(field) or ""):
                errors.append(f"questions.jsonl:{row_number}: {field} is required")
        if not isinstance(row.get("answer"), list):
            errors.append(f"questions.jsonl:{row_number}: answer must be an array")
        qa_id = str(row.get("qa_id") or "")
        if qa_id in question_ids:
            errors.append(f"questions.jsonl:{row_number}: duplicate qa_id {qa_id}")
        question_ids.add(qa_id)
        if row.get("question_date"):
            _datetime(row["question_date"], f"questions.jsonl:{row_number}.question_date", errors)
    if not errors:
        errors.extend(CockpitJSONLDataset(root).validate())
    return {
        "valid": not errors,
        "conversation_count": len(conversations),
        "question_count": len(questions),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.dataset.resolve())
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
