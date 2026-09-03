from __future__ import annotations

import json
from pathlib import Path

from .base import DatasetAdapter, DatasetInfo
from ..schema import Conversation, Message, Question, Session


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CockpitJSONLDataset(DatasetAdapter):
    """Portable adapter for the published Chinese cockpit JSONL contract."""

    dataset_id = "cockpit_jsonl"
    info = DatasetInfo(
        dataset_id=dataset_id,
        version="1",
        modalities=frozenset({"text"}),
        task_types=frozenset({"qa", "temporal", "multi-hop", "cross-session"}),
    )

    def __init__(self, root: Path) -> None:
        self.root = root
        conversations = _read_jsonl(root / "conversations.jsonl")
        questions = _read_jsonl(root / "questions.jsonl")
        self._conversations = {
            str(row["sample_id"]): self._conversation(row) for row in conversations
        }
        self._questions = [self._question(row) for row in questions]

    @staticmethod
    def _conversation(row: dict) -> Conversation:
        sessions: list[Session] = []
        for session_number, source in enumerate(row.get("sessions") or (), 1):
            session_id = str(
                source.get("source_session_id")
                or source.get("session_id")
                or f"{row['sample_id']}-s{session_number:03d}"
            )
            timestamp = str(source.get("date_time") or source.get("timestamp") or "")
            messages = tuple(
                Message(
                    message_id=str(
                        message.get("message_id") or f"{session_id}:{index:03d}"
                    ),
                    role=str(message.get("role") or ""),
                    content=str(message.get("content") or ""),
                    speaker=str(message.get("speaker") or ""),
                    timestamp=str(message.get("timestamp") or timestamp),
                    metadata=dict(message.get("metadata") or {}),
                )
                for index, message in enumerate(source.get("messages") or (), 1)
            )
            sessions.append(Session(
                session_id=session_id,
                timestamp=timestamp,
                messages=messages,
                metadata=dict(source.get("metadata") or {}),
            ))
        return Conversation(
            conversation_id=str(row["sample_id"]),
            sessions=tuple(sessions),
            metadata=dict(row.get("metadata") or {}),
        )

    @staticmethod
    def _question(row: dict) -> Question:
        metadata = dict(row.get("metadata") or {})
        for key in ("question_date", "is_abstention", "question_type"):
            if key in row:
                metadata[key] = row[key]
        return Question(
            question_id=str(row["qa_id"]),
            conversation_id=str(row["sample_id"]),
            text=str(row["question"]),
            answers=tuple(str(value) for value in row.get("answer") or ()),
            category=str(row.get("category") or row.get("question_type") or ""),
            evidence_ids=tuple(str(value) for value in (
                row.get("answer_session_ids") or row.get("evidence") or ()
            )),
            metadata=metadata,
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
        message_owners: dict[str, str] = {}
        session_owners: dict[str, str] = {}
        for conversation in self.conversations():
            if not conversation.sessions:
                errors.append(f"conversation {conversation.conversation_id} has no sessions")
            for session in conversation.sessions:
                if session.session_id in session_owners:
                    errors.append(f"duplicate session_id: {session.session_id}")
                session_owners[session.session_id] = conversation.conversation_id
                if not session.timestamp:
                    errors.append(f"session {session.session_id} has no timestamp")
                for message in session.messages:
                    if message.message_id in message_owners:
                        errors.append(f"duplicate message_id: {message.message_id}")
                    message_owners[message.message_id] = conversation.conversation_id
                    if message.role not in {"user", "assistant", "system", "tool"}:
                        errors.append(
                            f"message {message.message_id} has unsupported role {message.role!r}"
                        )
                    if not message.render_text():
                        errors.append(f"message {message.message_id} has no content")
        for question in self.questions():
            for evidence_id in question.evidence_ids:
                owner = message_owners.get(evidence_id) or session_owners.get(evidence_id)
                if owner is None:
                    errors.append(
                        f"question {question.question_id} references missing evidence {evidence_id}"
                    )
                elif owner != question.conversation_id:
                    errors.append(
                        f"question {question.question_id} references cross-conversation evidence "
                        f"{evidence_id}"
                    )
        return errors
