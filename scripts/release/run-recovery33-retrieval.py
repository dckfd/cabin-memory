#!/usr/bin/env python3
"""Read-only retrieval over the frozen Recovery33 memory service."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.framework_eval.adapters.tencentdb_http import TencentDBHTTPAdapter
from benchmarks.framework_eval.datasets.cockpit_jsonl import CockpitJSONLDataset


def _recovery33_isolation(dataset: CockpitJSONLDataset, destination: Path) -> Path:
    conversations = {}
    prefix = "cockpit-zh-public-mix-"
    for conversation in dataset.conversations():
        if not conversation.conversation_id.startswith(prefix):
            raise ValueError(
                "automatic Recovery33 isolation is valid only for the published v7 dataset"
            )
        suffix = conversation.conversation_id.removeprefix(prefix)
        conversations[conversation.conversation_id] = {
            "agent_id": f"cockpit-zh-rc52-v7-agent-{suffix}",
            "task_id": f"cockpit-zh-rc52-v7-task-{suffix}",
        }
    value = {
        "dataset_id": "cockpit-zh-public-mix-500-v7",
        "team_id": "cockpit-zh-rc52-v7-team",
        "user_id": "cockpit-zh-rc52-v7-user",
        "conversations": conversations,
    }
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:18507")
    parser.add_argument("--isolation-map", type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = CockpitJSONLDataset(args.dataset.resolve())
    errors = dataset.validate()
    if errors:
        raise SystemExit("dataset validation failed: " + "; ".join(errors[:10]))
    isolation = args.isolation_map
    if isolation is None:
        isolation = _recovery33_isolation(dataset, args.output.with_name("isolation.json"))
    os.environ.update({
        "TDAI_HTTP_API_VERSION": "v3",
        "TDAI_EVAL_ISOLATION_MAP": str(isolation.resolve()),
        "TDAI_EVAL_MEMORY_LAYERS": "L0,L1,L2,L3",
        "TDAI_EVAL_TEMPORAL_QUERY_MODE": "interval_v1",
        "TDAI_EVAL_TEMPORAL_QUERY_RESULTS": "4",
        "TDAI_EVAL_TEMPORAL_DEFAULT_TIMEZONE": "Asia/Shanghai",
        "TDAI_EVAL_STRUCTURED_CHAIN_RETRIEVAL": "true",
    })
    adapter = TencentDBHTTPAdapter(
        args.service_url, api_key=os.getenv("TDAI_API_KEY", "")
    )
    questions = dataset.questions()
    with args.output.open("x", encoding="utf-8") as handle:
        for number, question in enumerate(questions, 1):
            started = time.monotonic()
            adapter.ensure_construction(
                dataset.conversation(question.conversation_id), timeout=30
            )
            hits = adapter.search(question, limit=args.limit)
            row = {
                "schema_version": 1,
                "framework": "tencentdb-recovery33-rc-harness-v2",
                "question": question.to_dict(),
                "hits": [hit.to_dict() for hit in hits],
                "context": adapter.render_hits(hits),
                "metrics": {
                    "hit_count": len(hits),
                    "search_seconds": time.monotonic() - started,
                    "store_reused": True,
                    "memory_rebuilt": False,
                    "evidence_source_count": len(question.evidence_ids),
                    "retrieved_source_ids": [
                        source for hit in hits for source in hit.source_ids
                    ],
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({
                "done": number,
                "qa_id": question.question_id,
                "hits": len(hits),
            }, ensure_ascii=False), flush=True)
    print(json.dumps({"completed": len(questions), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
