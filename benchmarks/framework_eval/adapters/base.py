from __future__ import annotations

from abc import ABC
from collections.abc import Iterable

from ..schema import Conversation, MemoryAnswer, MemoryHit, Message, Question, Session


class UnsupportedCapabilityError(NotImplementedError):
    pass


class MemoryAdapter(ABC):
    """Smallest common contract shared by memory frameworks.

    Implementations may be in-process, HTTP services, containers, or subprocess
    bridges. The harness owns answer generation and judging so only memory
    construction and retrieval vary between frameworks.
    """

    adapter_id = "abstract"
    capabilities = frozenset({"ingest", "search"})

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, capability: str) -> None:
        if not self.supports(capability):
            raise UnsupportedCapabilityError(
                f"adapter {self.adapter_id} does not support {capability}"
            )

    def prepare(self, conversation: Conversation) -> None:
        """Create an isolated namespace for one conversation."""

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Persist one complete source session."""
        self.require("ingest")
        raise UnsupportedCapabilityError(
            "ingest capability is declared but ingest_session is not implemented"
        )

    def finalize(self, conversation_id: str) -> None:
        """Flush asynchronous extraction/indexing before retrieval."""

    def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
        """Return ranked evidence for one question."""
        self.require("search")
        raise UnsupportedCapabilityError("search capability is declared but not implemented")

    def close(self) -> None:
        """Release external resources."""

    def construction_metrics(self) -> dict:
        """Return model-free construction counters for the current process."""
        return {}

    def ingest(self, conversation: Conversation) -> None:
        self.require("ingest")
        self.prepare(conversation)
        for session in conversation.sessions:
            self.ingest_session(conversation.conversation_id, session)
        self.finalize(conversation.conversation_id)

    def answer(self, question: Question, *, limit: int) -> MemoryAnswer:
        self.require("answer")
        raise UnsupportedCapabilityError("answer capability is declared but not implemented")

    def reflect(self, conversation_id: str) -> None:
        self.require("reflect")
        raise UnsupportedCapabilityError("reflect capability is declared but not implemented")

    def wait_until_ready(self, conversation_id: str, *, timeout: float) -> None:
        if self.supports("wait_until_ready"):
            raise UnsupportedCapabilityError(
                "wait_until_ready capability is declared but not implemented"
            )

    def ensure_construction(
        self, conversation: Conversation, *, timeout: float | None
    ) -> dict:
        """Audit and, when supported, repair asynchronous construction.

        The default is deliberately a no-op.  Remote adapters can use this
        hook before readiness/search to detect a completed-but-empty pipeline
        result without teaching the framework-neutral runner about their
        storage layout or retry protocol.
        """
        return {}

    def delete(self, conversation_id: str, memory_id: str) -> None:
        self.require("delete")
        raise UnsupportedCapabilityError("delete capability is declared but not implemented")

    def update(self, conversation_id: str, memory_id: str, message: Message) -> None:
        self.require("update")
        raise UnsupportedCapabilityError("update capability is declared but not implemented")

    def get_profile(self, conversation_id: str) -> dict:
        self.require("profile")
        raise UnsupportedCapabilityError("profile capability is declared but not implemented")

    def get_graph(self, conversation_id: str) -> dict:
        self.require("graph")
        raise UnsupportedCapabilityError("graph capability is declared but not implemented")

    def drill_down(self, conversation_id: str, node_id: str) -> MemoryHit:
        self.require("drill_down")
        raise UnsupportedCapabilityError("drill_down capability is declared but not implemented")

    @staticmethod
    def render_hits(hits: Iterable[MemoryHit]) -> str:
        blocks = []
        for index, hit in enumerate(hits, 1):
            sources = f" sources={','.join(hit.source_ids)}" if hit.source_ids else ""
            score = f" score={hit.score:.6f}" if hit.score is not None else ""
            # Time is first-class evidence in long-horizon QA.  Adapters keep
            # it in metadata so indexes do not have to tokenize the timestamp;
            # render it back into the answer context, where relative phrases
            # such as "yesterday" can be resolved without dataset-specific
            # logic.  ``authored_at`` is used by several external frameworks.
            timestamp = str(
                hit.metadata.get("timestamp")
                or hit.metadata.get("authored_at")
                or hit.metadata.get("session_timestamp")
                or ""
            ).strip()
            time_label = f" time={timestamp}" if timestamp else ""
            blocks.append(f"[{index}{score}{sources}{time_label}]\n{hit.content}")
        return "\n\n".join(blocks)
