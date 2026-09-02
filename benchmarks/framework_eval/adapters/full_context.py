from __future__ import annotations

from .base import MemoryAdapter
from ..schema import MemoryHit, Question, Session


class FullContextAdapter(MemoryAdapter):
    adapter_id = "full_context"

    def __init__(self) -> None:
        self._sessions: dict[str, list[Session]] = {}

    def prepare(self, conversation) -> None:
        self._sessions[conversation.conversation_id] = []

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        self._sessions.setdefault(conversation_id, []).append(session)

    def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for session in self._sessions.get(question.conversation_id, []):
            text = "\n".join(
                f"[{message.message_id}] {message.speaker or message.role}: {message.render_text()}"
                for message in session.messages
            )
            hits.append(MemoryHit(
                text,
                source_ids=(session.session_id,) + tuple(m.message_id for m in session.messages),
                metadata={"session_id": session.session_id, "timestamp": session.timestamp},
                parts=tuple(part for message in session.messages for part in message.parts),
            ))
        return hits
