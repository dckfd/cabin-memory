from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .base import MemoryAdapter
from ..schema import ContentPart, Conversation, MemoryAnswer, MemoryHit, Message, Question, Session


class ExternalProcessAdapter(MemoryAdapter):
    """Framework bridge using a stable JSON request/response contract.

    The configured command is invoked once per operation. It receives one JSON
    object on stdin and must return one JSON object on stdout. Keeping third-party
    frameworks out-of-process prevents dependency conflicts between their SDKs.
    """

    adapter_id = "external_process"

    def __init__(self, command: list[str], *, cwd: Path | None = None,
                 environment: dict[str, str] | None = None, timeout: int = 600,
                 adapter_id: str = "external_process",
                 capabilities: set[str] | frozenset[str] | None = None) -> None:
        self.adapter_id = adapter_id
        self.capabilities = frozenset(capabilities or {"ingest", "search"})
        self.command = command
        self.cwd = cwd
        self.environment = {**os.environ, **(environment or {})}
        self.timeout = timeout

    def _call(self, operation: str, payload: dict) -> dict:
        request = json.dumps({"protocol": 1, "operation": operation, "payload": payload})
        completed = subprocess.run(
            self.command,
            input=request,
            text=True,
            capture_output=True,
            cwd=self.cwd,
            env=self.environment,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"adapter command failed ({completed.returncode}): {completed.stderr[-2000:]}"
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"adapter returned invalid JSON: {completed.stdout[-2000:]}") from exc
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return response

    def prepare(self, conversation: Conversation) -> None:
        self._call("prepare", conversation.to_dict())

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        self._call("ingest_session", {
            "conversation_id": conversation_id,
            "session": session.to_dict(),
        })

    def finalize(self, conversation_id: str) -> None:
        self._call("finalize", {"conversation_id": conversation_id})

    def search(self, question: Question, *, limit: int) -> list[MemoryHit]:
        response = self._call("search", {"question": question.to_dict(), "limit": limit})
        return [MemoryHit(
            content=str(hit["content"]),
            score=float(hit["score"]) if hit.get("score") is not None else None,
            source_ids=tuple(str(value) for value in hit.get("source_ids", [])),
            metadata=dict(hit.get("metadata", {})),
            parts=tuple(ContentPart.from_dict(part) for part in hit.get("parts", [])),
        ) for hit in response.get("hits", [])]

    def close(self) -> None:
        self._call("close", {})

    def answer(self, question: Question, *, limit: int) -> MemoryAnswer:
        self.require("answer")
        response = self._call("answer", {"question": question.to_dict(), "limit": limit})
        return MemoryAnswer(
            answer=str(response.get("answer", "")),
            source_ids=tuple(str(value) for value in response.get("source_ids", [])),
            usage=dict(response.get("usage", {})),
            metadata=dict(response.get("metadata", {})),
        )

    def reflect(self, conversation_id: str) -> None:
        self.require("reflect")
        self._call("reflect", {"conversation_id": conversation_id})

    def wait_until_ready(self, conversation_id: str, *, timeout: float) -> None:
        self.require("wait_until_ready")
        self._call("wait_until_ready", {
            "conversation_id": conversation_id,
            "timeout": timeout,
        })

    def delete(self, conversation_id: str, memory_id: str) -> None:
        self.require("delete")
        self._call("delete", {"conversation_id": conversation_id, "memory_id": memory_id})

    def update(self, conversation_id: str, memory_id: str, message: Message) -> None:
        self.require("update")
        self._call("update", {
            "conversation_id": conversation_id,
            "memory_id": memory_id,
            "message": message.to_dict(),
        })

    def get_profile(self, conversation_id: str) -> dict:
        self.require("profile")
        return self._call("get_profile", {"conversation_id": conversation_id})

    def get_graph(self, conversation_id: str) -> dict:
        self.require("graph")
        return self._call("get_graph", {"conversation_id": conversation_id})

    def drill_down(self, conversation_id: str, node_id: str) -> MemoryHit:
        self.require("drill_down")
        row = self._call("drill_down", {
            "conversation_id": conversation_id,
            "node_id": node_id,
        })
        return MemoryHit(
            content=str(row.get("content", "")),
            score=float(row["score"]) if row.get("score") is not None else None,
            source_ids=tuple(str(value) for value in row.get("source_ids", [])),
            metadata=dict(row.get("metadata", {})),
            parts=tuple(ContentPart.from_dict(part) for part in row.get("parts", [])),
        )
