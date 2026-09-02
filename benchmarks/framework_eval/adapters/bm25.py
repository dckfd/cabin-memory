from __future__ import annotations

import math
import re
from collections import Counter

from .base import MemoryAdapter
from ..schema import ContentPart, MemoryHit, Question, Session


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25Adapter(MemoryAdapter):
    adapter_id = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[
            str, list[tuple[str, str, str, str, str, list[str], tuple[ContentPart, ...]]]
        ] = {}

    def prepare(self, conversation) -> None:
        self._docs[conversation.conversation_id] = []

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        docs = self._docs.setdefault(conversation_id, [])
        for message in session.messages:
            content = message.render_text()
            docs.append((
                message.message_id,
                session.session_id,
                message.speaker or message.role,
                message.timestamp or session.timestamp,
                content,
                tokenize(content),
                message.parts,
            ))

    def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
        docs = self._docs.get(question.conversation_id, [])
        if not docs:
            return []
        query = tokenize(question.text)
        avgdl = sum(len(tokens) for _, _, _, _, _, tokens, _ in docs) / len(docs)
        df = Counter(
            token for token in set(query) for _, _, _, _, _, tokens, _ in docs if token in tokens
        )
        scored: list[tuple[float, str, str, str, str, tuple[ContentPart, ...]]] = []
        for message_id, session_id, speaker, timestamp, content, tokens, parts in docs:
            tf = Counter(tokens)
            score = 0.0
            for token in query:
                freq = tf[token]
                if not freq:
                    continue
                idf = math.log(1 + (len(docs) - df[token] + 0.5) / (df[token] + 0.5))
                norm = freq + self.k1 * (1 - self.b + self.b * len(tokens) / max(avgdl, 1))
                score += idf * freq * (self.k1 + 1) / norm
            if score:
                scored.append((score, message_id, session_id, speaker, timestamp, content, parts))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [MemoryHit(
            f"{speaker}: {content}" if speaker else content,
            score=score,
            source_ids=(message_id, session_id),
            metadata={"timestamp": timestamp, "session_id": session_id},
            parts=parts,
        ) for score, message_id, session_id, speaker, timestamp, content, parts in scored[:limit]]
