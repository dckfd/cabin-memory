from __future__ import annotations

from pathlib import Path
import re

from .base import DatasetAdapter, DatasetInfo
from .locomo_refined import read_jsonl
from ..schema import Conversation, Message, Question, Session


class KVRETMemoryDataset(DatasetAdapter):
    """Derived long-memory evaluation over official KVRET test dialogues."""

    dataset_id = "kvret_memory"
    info = DatasetInfo(
        dataset_id="kvret_memory",
        version="kvret-memory-138-v2",
        modalities=frozenset({"text"}),
        task_types=frozenset({
            "qa", "task-oriented", "fragmented-command", "cross-session"
        }),
        metadata={"official_kvret_metric": False},
    )

    def __init__(self, root: Path) -> None:
        self.root = root
        conversation_rows = read_jsonl(root / "conversations.jsonl")
        question_rows = read_jsonl(root / "questions.jsonl")
        self._conversations = {
            str(row["sample_id"]): self._conversation(row)
            for row in conversation_rows
        }
        self._questions = [self._question(row) for row in question_rows]

    @staticmethod
    def _conversation(row: dict) -> Conversation:
        sessions = tuple(Session(
            session_id=str(source["session_id"]),
            timestamp=str(source.get("timestamp") or ""),
            messages=tuple(Message(
                message_id=str(message["message_id"]),
                role=str(message.get("role") or ""),
                speaker=str(message.get("speaker") or ""),
                content=str(message.get("content") or ""),
                timestamp=str(message.get("timestamp") or source.get("timestamp") or ""),
                metadata=dict(message.get("metadata") or {}),
            ) for message in source.get("messages") or []),
            metadata=dict(source.get("metadata") or {}),
        ) for source in row.get("sessions") or [])
        return Conversation(
            conversation_id=str(row["sample_id"]),
            sessions=sessions,
            metadata=dict(row.get("metadata") or {}),
        )

    @staticmethod
    def _question(row: dict) -> Question:
        return Question(
            question_id=str(row["qa_id"]),
            conversation_id=str(row["sample_id"]),
            text=str(row["question"]),
            answers=tuple(str(value) for value in row.get("answer") or []),
            category=str(row.get("category") or ""),
            evidence_ids=tuple(str(value) for value in row.get("evidence") or []),
            metadata=dict(row.get("metadata") or {}),
        )

    def conversation(self, conversation_id: str) -> Conversation:
        return self._conversations[conversation_id]

    def conversations(self) -> list[Conversation]:
        return list(self._conversations.values())

    def questions(self, conversation_ids: set[str] | None = None) -> list[Question]:
        if not conversation_ids:
            return list(self._questions)
        return [
            question for question in self._questions
            if question.conversation_id in conversation_ids
        ]

    def validate(self) -> list[str]:
        errors = super().validate()
        source_text: dict[str, str] = {}
        conversation_sources: dict[str, set[str]] = {}
        for conversation in self.conversations():
            source_ids: set[str] = set()
            for session in conversation.sessions:
                for message in session.messages:
                    source_ids.add(message.message_id)
                    source_text[message.message_id] = message.render_text()
            conversation_sources[conversation.conversation_id] = source_ids
        for question in self.questions():
            known = conversation_sources.get(question.conversation_id, set())
            missing = set(question.evidence_ids) - known
            if missing:
                errors.append(
                    f"question {question.question_id} references missing evidence: "
                    f"{', '.join(sorted(missing))}"
                )
            answer_keys = {
                "".join(re.findall(r"[a-z0-9]+", answer.casefold()))
                for answer in question.answers
            } - {""}
            evidence_key = "".join(
                re.findall(
                    r"[a-z0-9]+",
                    " ".join(source_text.get(value, "") for value in question.evidence_ids)
                    .casefold(),
                )
            )
            if not answer_keys or not any(value in evidence_key for value in answer_keys):
                errors.append(
                    f"question {question.question_id} has no grounded answer in its evidence"
                )
        return errors
