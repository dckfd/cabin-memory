from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_changed(base: Path, candidate: Path, output: Path) -> dict[str, Any]:
    base_rows = {
        str(row["question"]["question_id"]): row for row in _read_jsonl(base)
    }
    candidate_rows = _read_jsonl(candidate)
    missing = [
        str(row["question"]["question_id"])
        for row in candidate_rows
        if str(row["question"]["question_id"]) not in base_rows
    ]
    if missing:
        raise ValueError(f"candidate contains IDs absent from base: {missing[:5]}")
    changed = [
        row for row in candidate_rows
        if str(row.get("context") or "")
        != str(base_rows[str(row["question"]["question_id"])].get("context") or "")
    ]
    _write_jsonl(output, changed)
    return {
        "base_count": len(base_rows),
        "candidate_count": len(candidate_rows),
        "changed_count": len(changed),
        "output": str(output.resolve()),
    }


def merge_predictions(
    base: Path,
    delta: Path,
    retrieval: Path,
    output: Path,
) -> dict[str, Any]:
    base_rows = {str(row["qa_id"]): row for row in _read_jsonl(base)}
    delta_rows = {str(row["qa_id"]): row for row in _read_jsonl(delta)}
    order = [
        str(row["question"]["question_id"]) for row in _read_jsonl(retrieval)
    ]
    unknown = sorted(set(delta_rows) - set(order))
    missing = [qa_id for qa_id in order if qa_id not in delta_rows and qa_id not in base_rows]
    if unknown or missing:
        raise ValueError(
            f"prediction ID mismatch: unknown_delta={unknown[:5]}, missing={missing[:5]}"
        )
    rows = [delta_rows.get(qa_id, base_rows[qa_id]) for qa_id in order]
    _write_jsonl(output, rows)
    return {
        "question_count": len(rows),
        "reused_count": sum(qa_id not in delta_rows for qa_id in order),
        "delta_count": sum(qa_id in delta_rows for qa_id in order),
        "output": str(output.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select changed retrieval rows or merge delta predictions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--base", type=Path, required=True)
    select.add_argument("--candidate", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--base", type=Path, required=True)
    merge.add_argument("--delta", type=Path, required=True)
    merge.add_argument("--retrieval", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        result = select_changed(
            args.base.resolve(), args.candidate.resolve(), args.output.resolve()
        )
    else:
        result = merge_predictions(
            args.base.resolve(),
            args.delta.resolve(),
            args.retrieval.resolve(),
            args.output.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
