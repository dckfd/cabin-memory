from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Iterable


VARIANTS = ("yesterday", "day_before_yesterday", "last_weekday")
_PREFIX = re.compile(
    r"^During the (vehicle interaction|driver voice interaction) "
    r"logged on (20\d{2}-\d{2}-\d{2})"
)


def rewrite_question(row: dict, variant: str) -> dict:
    """Turn an explicit-date cockpit question into an anchored relative one."""
    if variant not in VARIANTS:
        raise ValueError(f"unsupported temporal variant: {variant}")
    result = dict(row)
    original = str(result.get("question") or "")
    match = _PREFIX.search(original)
    if match is None:
        raise ValueError(f"question has no supported interaction date: {original}")
    interaction = match.group(1)
    source_date = datetime.strptime(match.group(2), "%Y-%m-%d").date()
    if variant == "yesterday":
        query_date = source_date + timedelta(days=1)
        replacement = f"During yesterday's {interaction}"
    elif variant == "day_before_yesterday":
        query_date = source_date + timedelta(days=2)
        replacement = (
            f"During the {interaction} from the day before yesterday"
        )
    else:
        query_date = source_date + timedelta(days=7)
        weekday = source_date.strftime("%A")
        replacement = f"During last {weekday}'s {interaction}"
    result["question"] = _PREFIX.sub(replacement, original, count=1)
    query_time = datetime.combine(
        query_date, time(hour=15, minute=30), tzinfo=timezone.utc
    )
    metadata = dict(result.get("metadata") or {})
    metadata.update({
        "query_time": query_time.isoformat().replace("+00:00", "Z"),
        "timezone": "UTC",
        "temporal_variant": variant,
        "source_interaction_date": source_date.isoformat(),
        "derived_temporal_cockpit": True,
    })
    result["metadata"] = metadata
    return result


def stratified_rows(
    rows: Iterable[dict], *, limit: int, selected_ids: set[str] | None = None
) -> list[dict]:
    """Select deterministically without looking at answers or evidence IDs."""
    eligible = [
        row for row in rows
        if selected_ids is None or str(row.get("qa_id")) in selected_ids
    ]
    groups: dict[str, deque[dict]] = defaultdict(deque)
    for row in sorted(eligible, key=lambda item: str(item.get("qa_id") or "")):
        groups[str(row.get("category") or "unknown")].append(row)
    selected: list[dict] = []
    categories = sorted(groups)
    while len(selected) < min(limit, len(eligible)):
        progressed = False
        for category in categories:
            if groups[category] and len(selected) < limit:
                selected.append(groups[category].popleft())
                progressed = True
        if not progressed:
            break
    return selected


def prepare(
    input_dir: Path,
    output_dir: Path,
    *,
    limit: int,
    selection_manifest: Path | None = None,
) -> dict:
    if limit <= 0:
        raise ValueError("limit must be positive")
    question_path = input_dir / "questions.jsonl"
    conversation_path = input_dir / "conversations.jsonl"
    rows = [
        json.loads(line)
        for line in question_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_ids = None
    if selection_manifest is not None:
        manifest = json.loads(selection_manifest.read_text(encoding="utf-8"))
        selected_ids = {str(value) for value in manifest["question_ids"]}
    selected = stratified_rows(rows, limit=limit, selected_ids=selected_ids)
    rewritten = [
        rewrite_question(row, VARIANTS[index % len(VARIANTS)])
        for index, row in enumerate(selected)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(conversation_path, output_dir / "conversations.jsonl")
    (output_dir / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rewritten),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "source": str(input_dir.resolve()),
        "questions": len(rewritten),
        "selection_manifest": (
            str(selection_manifest.resolve()) if selection_manifest else ""
        ),
        "variants": {
            variant: sum(
                row["metadata"]["temporal_variant"] == variant
                for row in rewritten
            )
            for variant in VARIANTS
        },
        "question_ids": [str(row["qa_id"]) for row in rewritten],
        "answer_and_evidence_unchanged": True,
    }
    (output_dir / "temporal-cockpit-manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build anchored relative-time cockpit retrieval questions"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--selection-manifest", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.input_dir,
        args.output_dir,
        limit=args.limit,
        selection_manifest=args.selection_manifest,
    )
    printable = {
        key: value for key, value in result.items() if key != "question_ids"
    }
    printable["question_id_count"] = len(result["question_ids"])
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
