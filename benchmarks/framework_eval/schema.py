from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContentPart:
    """One modality-bearing piece of a message or question."""

    type: str
    text: str = ""
    uri: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"text", "image", "audio", "video", "file"}
        if self.type not in allowed:
            raise ValueError(f"unsupported content part type: {self.type}")
        if not self.text and not self.uri:
            raise ValueError("content part requires text or uri")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContentPart":
        return cls(
            type=str(value["type"]),
            text=str(value.get("text", "")),
            uri=str(value.get("uri", "")),
            mime_type=str(value.get("mime_type", "")),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class Message:
    message_id: str
    role: str
    content: str
    speaker: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parts: tuple[ContentPart, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "parts": [part.to_dict() for part in self.parts],
        }

    def render_text(self) -> str:
        if self.content:
            return self.content
        return "\n".join(part.text for part in self.parts if part.text)


@dataclass(frozen=True)
class Session:
    session_id: str
    timestamp: str
    messages: tuple[Message, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "messages": [message.to_dict() for message in self.messages],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    sessions: tuple[Session, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "sessions": [session.to_dict() for session in self.sessions],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Question:
    question_id: str
    conversation_id: str
    text: str
    answers: tuple[str, ...]
    category: str = ""
    evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    parts: tuple[ContentPart, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "conversation_id": self.conversation_id,
            "text": self.text,
            "answers": list(self.answers),
            "category": self.category,
            "evidence_ids": list(self.evidence_ids),
            "metadata": self.metadata,
            "parts": [part.to_dict() for part in self.parts],
        }

    def render_text(self) -> str:
        if self.text:
            return self.text
        return "\n".join(part.text for part in self.parts if part.text)


@dataclass(frozen=True)
class MemoryHit:
    content: str
    score: float | None = None
    source_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    parts: tuple[ContentPart, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "source_ids": list(self.source_ids),
            "metadata": self.metadata,
            "parts": [part.to_dict() for part in self.parts],
        }


@dataclass(frozen=True)
class MemoryAnswer:
    answer: str
    source_ids: tuple[str, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
