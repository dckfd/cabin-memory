from __future__ import annotations

from pathlib import Path

from .locomo_refined import read_jsonl
from ..schema import Conversation, Message, Question, Session
from .base import DatasetAdapter, DatasetInfo


class LongMemEvalDataset(DatasetAdapter):
    dataset_id = "longmemeval"
    info = DatasetInfo(
        dataset_id="longmemeval",
        version="longmemeval-s",
        modalities=frozenset({"text"}),
        task_types=frozenset({"qa", "preference", "knowledge-update", "temporal", "abstention"}),
    )

    def __init__(self, root: Path) -> None:
        conversations = read_jsonl(root / "conversations.jsonl")
        questions = read_jsonl(root / "questions.jsonl")
        self._conversations = {row["sample_id"]: self._conversation(row) for row in conversations}
        self._questions = [self._question(row) for row in questions]

    @staticmethod
    def _conversation(row: dict) -> Conversation:
        sessions = []
        for source in row.get("sessions", []):
            source_id = str(source.get("source_session_id", source["session_index"]))
            messages = tuple(Message(
                message_id=f"{source_id}:{index:03d}",
                role=str(message.get("role", "")),
                content=str(message.get("content", "")),
                timestamp=str(source.get("date_time", "")),
                metadata={"has_answer": bool(message.get("has_answer")),
                          "source_session_id": source_id},
            ) for index, message in enumerate(source.get("messages", []), 1))
            sessions.append(Session(source_id, str(source.get("date_time", "")), messages,
                                    {"source_session_id": source_id}))
        return Conversation(str(row["sample_id"]), tuple(sessions), {
            "source_question_id": row.get("source_question_id"),
        })

    @staticmethod
    def _question(row: dict) -> Question:
        return Question(
            question_id=str(row["qa_id"]),
            conversation_id=str(row["sample_id"]),
            text=str(row["question"]),
            answers=tuple(str(answer) for answer in row.get("answer", [])),
            category=str(row.get("category", row.get("question_type", ""))),
            evidence_ids=tuple(str(value) for value in row.get("answer_session_ids", [])),
            metadata={
                "question_date": row.get("question_date"),
                "is_abstention": bool(row.get("is_abstention")),
            },
        )

    def conversation(self, conversation_id: str) -> Conversation:
        return self._conversations[conversation_id]

    def conversations(self) -> list[Conversation]:
        return list(self._conversations.values())

    def questions(self, conversation_ids: set[str] | None = None) -> list[Question]:
        if not conversation_ids:
            return list(self._questions)
        return [q for q in self._questions if q.conversation_id in conversation_ids]
