from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .adapters.base import MemoryAdapter
from .schema import Conversation, Question


@dataclass(frozen=True)
class NativeRunPolicy:
    """Optional lifecycle hooks for a framework's native answer track."""

    reflect_after_ingest: bool = False
    ready_timeout: float | None = None
    fail_fast: bool = False


class NativeAnswerRunner:
    """Evaluate frameworks that own retrieval and answer generation.

    This is deliberately a separate track from retrieval + shared AnswerModel:
    native answers may use a framework-specific model, prompt, or agent loop and
    therefore must not be mixed into the unified leaderboard.
    """

    def __init__(self, adapter: MemoryAdapter, output: Path, *, limit: int = 8,
                 resume: bool = False, policy: NativeRunPolicy | None = None) -> None:
        adapter.require("ingest")
        adapter.require("answer")
        self.adapter = adapter
        self.output = output
        self.limit = limit
        self.resume = resume
        self.policy = policy or NativeRunPolicy()
        output.parent.mkdir(parents=True, exist_ok=True)

    def run(self, conversations: list[Conversation], questions: list[Question]) -> dict:
        selected: dict[str, list[Question]] = {}
        for question in questions:
            selected.setdefault(question.conversation_id, []).append(question)
        completed_ids = self._completed_ids()
        completed = skipped = errors = 0
        started = time.monotonic()
        mode = "a" if self.resume else "w"
        try:
            with self.output.open(mode, encoding="utf-8") as handle:
                for conversation in conversations:
                    pending = [
                        question for question in selected.get(conversation.conversation_id, [])
                        if question.question_id not in completed_ids
                    ]
                    skipped += len(selected.get(conversation.conversation_id, [])) - len(pending)
                    if not pending:
                        continue
                    self.adapter.ingest(conversation)
                    if self.policy.reflect_after_ingest:
                        self.adapter.reflect(conversation.conversation_id)
                    if self.policy.ready_timeout is not None:
                        self.adapter.wait_until_ready(
                            conversation.conversation_id,
                            timeout=self.policy.ready_timeout,
                        )
                    for question in pending:
                        answer_started = time.monotonic()
                        try:
                            answer = self.adapter.answer(question, limit=self.limit)
                            row = {
                                "schema_version": 1,
                                "framework": self.adapter.adapter_id,
                                "answer_track": "native",
                                "qa_id": question.question_id,
                                "conversation_id": question.conversation_id,
                                "category": question.category,
                                "gold_answers": list(question.answers),
                                "predicted_answer": answer.answer,
                                "source_ids": list(answer.source_ids),
                                "usage": answer.usage,
                                "framework_metadata": answer.metadata,
                                "answer_seconds": time.monotonic() - answer_started,
                                "error": None,
                            }
                        except Exception as exc:
                            errors += 1
                            row = {
                                "schema_version": 1,
                                "framework": self.adapter.adapter_id,
                                "answer_track": "native",
                                "qa_id": question.question_id,
                                "conversation_id": question.conversation_id,
                                "category": question.category,
                                "gold_answers": list(question.answers),
                                "predicted_answer": "",
                                "source_ids": [],
                                "usage": {},
                                "framework_metadata": {},
                                "answer_seconds": time.monotonic() - answer_started,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            if self.policy.fail_fast:
                                raise
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        completed += 1
        finally:
            self.adapter.close()
        return {
            "questions": completed,
            "skipped": skipped,
            "errors": errors,
            "answer_track": "native",
            "seconds": time.monotonic() - started,
            "output": str(self.output),
        }

    def _completed_ids(self) -> set[str]:
        if not self.resume or not self.output.exists():
            return set()
        with self.output.open(encoding="utf-8") as handle:
            return {
                str(json.loads(line).get("qa_id"))
                for line in handle
                if line.strip()
            }
