#!/usr/bin/env python3
"""Read-only full-500 retrieval against the preserved Recovery33 store."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmarks.framework_eval.adapters.tencentdb_http import TencentDBHTTPAdapter
from benchmarks.framework_eval.schema import Conversation, Message, Question, Session


def main() -> None:
    challenge = ROOT / "benchmarks/framework_eval/challenges/cockpit_zh_public_mix_500_v7"
    output = Path(__file__).with_name("retrieval.jsonl")
    if output.exists():
        raise SystemExit("refusing overwrite")
    conversations = {}
    for line in (challenge / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        sessions = []
        for session in raw["sessions"]:
            session_id = session["source_session_id"]
            messages = tuple(Message(
                f"{session_id}:{index:03d}", message.get("role", "user"),
                message.get("content", ""), timestamp=session.get("date_time", ""),
                metadata=message.get("metadata") or {},
            ) for index, message in enumerate(session["messages"], 1))
            sessions.append(Session(session_id, session.get("date_time", ""), messages))
        conversations[raw["sample_id"]] = Conversation(raw["sample_id"], tuple(sessions))
    questions = []
    for line in (challenge / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        questions.append(Question(
            str(raw["qa_id"]), str(raw["sample_id"]), str(raw["question"]),
            tuple(raw.get("answer") or ()), category=str(raw.get("category", "")),
            evidence_ids=tuple(raw.get("answer_session_ids") or ()),
            metadata={**(raw.get("metadata") or {}), "question_date": raw.get("question_date"),
                      "is_abstention": raw.get("is_abstention", False)},
        ))
    os.environ["TDAI_HTTP_API_VERSION"] = "v3"
    os.environ["TDAI_EVAL_ISOLATION_MAP"] = str(ROOT / "benchmarks/framework_eval_runs/cockpit-zh-final-completeness-rc52-20260830-v1/v7-official-pass1-20260830-v1/v7-isolation.json")
    os.environ["TDAI_EVAL_MEMORY_LAYERS"] = "L0,L1,L2,L3"
    os.environ["TDAI_EVAL_TEMPORAL_QUERY_MODE"] = "interval_v1"
    os.environ["TDAI_EVAL_TEMPORAL_QUERY_RESULTS"] = "4"
    os.environ["TDAI_EVAL_TEMPORAL_DEFAULT_TIMEZONE"] = "Asia/Shanghai"
    adapter = TencentDBHTTPAdapter("http://127.0.0.1:18507", api_key=os.getenv("TDAI_LLM_API_KEY", ""))
    with output.open("x", encoding="utf-8") as handle:
        for number, question in enumerate(questions, 1):
            started = time.monotonic()
            adapter.ensure_construction(conversations[question.conversation_id], timeout=30)
            hits = adapter.search(question, limit=16)
            row = {
                "schema_version": 1, "framework": "tencentdb-recovery33-full500-harness-v1",
                "question": question.to_dict(), "hits": [hit.to_dict() for hit in hits],
                "context": adapter.render_hits(hits),
                "metrics": {
                    "hit_count": len(hits), "search_seconds": time.monotonic() - started,
                    "store_reused": True, "memory_rebuilt": False,
                    "evidence_source_count": len(question.evidence_ids),
                    "retrieved_source_ids": [source for hit in hits for source in hit.source_ids],
                },
                "recovery_image": "agentmemory/memory-core:2.0.0-cockpit-zh-semantic-receipt-rc77-20260902",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"done": number, "qa_id": question.question_id, "hits": len(hits)}), flush=True)
    print(json.dumps({"completed": len(questions), "output": str(output)}))


if __name__ == "__main__":
    main()
