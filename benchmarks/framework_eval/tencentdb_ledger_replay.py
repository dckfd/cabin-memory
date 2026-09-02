from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .adapters.base import MemoryAdapter
from .adapters.tencentdb_http import TencentDBHTTPAdapter
from .schema import MemoryHit


def _memory_hit(row: dict[str, Any]) -> MemoryHit:
    return MemoryHit(
        content=str(row.get("content") or ""),
        score=(float(row["score"]) if row.get("score") is not None else None),
        source_ids=tuple(str(value) for value in (row.get("source_ids") or [])),
        metadata=dict(row.get("metadata") or {}),
    )


def replace_ledger_hit(
    row: dict[str, Any],
    ledger_row: dict[str, Any],
    *,
    max_context_chars: int | None,
    base_artifact: str,
) -> dict[str, Any]:
    """Replace exactly one L1X hit while preserving every backend hit byte-for-byte."""
    hits = [dict(hit) for hit in (row.get("hits") or [])]
    ledger_indexes = [
        index for index, hit in enumerate(hits)
        if (hit.get("metadata") or {}).get("level") == "L1X"
    ]
    if len(ledger_indexes) != 1:
        raise ValueError(
            f"expected exactly one L1X hit for {row['question']['question_id']}; "
            f"found {len(ledger_indexes)}"
        )
    index = ledger_indexes[0]
    hits[index] = {
        "content": str(ledger_row.get("content") or ""),
        "score": ledger_row.get("score"),
        "source_ids": [
            str(value) for value in (ledger_row.get("source_ids") or [])
        ],
        "metadata": dict(ledger_row.get("metadata") or {}),
        "parts": [],
    }
    rendered = MemoryAdapter.render_hits(_memory_hit(hit) for hit in hits)
    truncated = bool(
        max_context_chars is not None and len(rendered) > max_context_chars
    )
    context = rendered[:max_context_chars] if truncated else rendered
    gold_sources = {
        str(value) for value in (row.get("question") or {}).get("evidence_ids") or []
    }
    retrieved_sources = {
        str(value)
        for hit in hits
        for value in (hit.get("source_ids") or [])
    }
    metrics = dict(row.get("metrics") or {})
    metrics.update({
        "hit_count": len(hits),
        "context_chars": len(context),
        "retrieved_context_chars": len(rendered),
        "context_truncated": truncated,
        "evidence_source_count": len(gold_sources),
        "evidence_source_hits": len(gold_sources & retrieved_sources),
        "evidence_recall": (
            len(gold_sources & retrieved_sources) / len(gold_sources)
            if gold_sources else None
        ),
        "evidence_recall_scope": "retrieved_hits_before_context_truncation",
    })
    result = dict(row)
    result["hits"] = hits
    result["context"] = context
    result["metrics"] = metrics
    result["replay_provenance"] = {
        "mode": "replace_adapter_ledger_only",
        "base_artifact": base_artifact,
        "preserved_hit_count": len(hits) - 1,
        "replaced_level": "L1X",
    }
    return result


def replay(
    base: Path,
    output: Path,
    *,
    adapter: TencentDBHTTPAdapter,
    max_context_chars: int | None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = 0
    changed = 0
    with base.open(encoding="utf-8") as source, output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            question = row["question"]
            ledger_row = adapter._v3_ledger_row(
                str(question["conversation_id"]), str(question["text"])
            )
            if ledger_row is None:
                raise ValueError(
                    f"new ledger produced no row for {question['question_id']}"
                )
            replaced = replace_ledger_hit(
                row,
                ledger_row,
                max_context_chars=max_context_chars,
                base_artifact=str(base.resolve()),
            )
            changed += int(str(replaced["context"]) != str(row.get("context") or ""))
            target.write(json.dumps(replaced, ensure_ascii=False) + "\n")
            count += 1
    return {
        "questions": count,
        "changed_contexts": changed,
        "seconds": time.monotonic() - started,
        "base": str(base.resolve()),
        "output": str(output.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an adapter-only TencentDB ledger change over saved backend hits"
        )
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context-chars", type=int, default=30000)
    args = parser.parse_args()
    result = replay(
        args.base.resolve(),
        args.output.resolve(),
        adapter=TencentDBHTTPAdapter("http://unused"),
        max_context_chars=args.max_context_chars,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
