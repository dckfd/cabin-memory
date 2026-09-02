from __future__ import annotations

import json
from pathlib import Path

from ..schema import ContentPart, Conversation, Message, Question, Session
from .base import DatasetAdapter, DatasetInfo


def read_jsonl(path: Path) -> list[dict]:
    # File iteration splits only on physical newlines. ``str.splitlines()`` also
    # treats Unicode line separators embedded inside valid JSON strings as row
    # boundaries, which corrupts real LongMemEval conversations.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message_content(row: dict) -> str:
    parts = [str(row.get("text", ""))]
    if row.get("blip_caption"):
        parts.append(f"[Image caption: {row['blip_caption']}]")
    if row.get("query"):
        parts.append(f"[Image query: {row['query']}]")
    return "\n".join(part for part in parts if part)


class LoCoMoRefinedDataset(DatasetAdapter):
    dataset_id = "locomo_refined"
    info = DatasetInfo(
        dataset_id="locomo_refined",
        version="public-refined",
        modalities=frozenset({"text", "image-uri", "image-caption"}),
        task_types=frozenset({"qa", "temporal", "multi-hop", "adversarial"}),
    )

    def __init__(self, root: Path) -> None:
        self.root = root
        conversations = read_jsonl(root / "data/public/conversations.jsonl")
        questions = read_jsonl(root / "data/public/questions.jsonl")
        self._conversations = {row["sample_id"]: self._conversation(row) for row in conversations}
        self._questions = [self._question(row) for row in questions]

    @staticmethod
    def _conversation(row: dict) -> Conversation:
        sessions = []
        for source in row.get("sessions", []):
            session_id = f"{row['sample_id']}-session-{int(source['session_index']):03d}"
            messages = tuple(Message(
                message_id=str(message.get("dia_id", "")),
                role=str(message.get("role", "")),
                content=message_content(message),
                speaker=str(message.get("speaker", "")),
                timestamp=str(message.get("session_date_time", source.get("date_time", ""))),
                metadata={
                    "images": message.get("images", []),
                    "has_multimodal_context": bool(message.get("has_multimodal_context")),
                    "session_index": source.get("session_index"),
                },
                parts=tuple(
                    ContentPart(
                        type="image",
                        uri=str(uri),
                        metadata={
                            "caption": message.get("blip_caption", ""),
                            "query": message.get("query", ""),
                        },
                    )
                    for uri in (message.get("images") or [])
                    if str(uri)
                ),
            ) for message in source.get("messages", []))
            sessions.append(Session(session_id, str(source.get("date_time", "")), messages,
                                    {"session_index": source.get("session_index")}))
        return Conversation(str(row["sample_id"]), tuple(sessions), {
            "speaker_a": row.get("speaker_a"),
            "speaker_b": row.get("speaker_b"),
            "multimodal_message_count": row.get("multimodal_message_count", 0),
        })

    @staticmethod
    def _question(row: dict) -> Question:
        return Question(
            question_id=str(row["qa_id"]),
            conversation_id=str(row["sample_id"]),
            text=str(row["question"]),
            answers=tuple(str(answer) for answer in row.get("answer", [])),
            category=str(row.get("category", "")),
            evidence_ids=tuple(str(value) for value in row.get("evidence", [])),
            metadata={"is_multimodal": bool(row.get("is_multi_modality"))},
        )

    def conversation(self, conversation_id: str) -> Conversation:
        return self._conversations[conversation_id]

    def conversations(self) -> list[Conversation]:
        return list(self._conversations.values())

    def questions(self, conversation_ids: set[str] | None = None) -> list[Question]:
        if not conversation_ids:
            return list(self._questions)
        return [q for q in self._questions if q.conversation_id in conversation_ids]
