from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schema import Conversation, Question


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    version: str = "unknown"
    modalities: frozenset[str] = frozenset({"text"})
    task_types: frozenset[str] = frozenset({"qa"})
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Canonical input boundary for any benchmark dataset.

    Implementations may read conversations, documents, event streams, or
    interactive episodes, but must expose stable questions and isolated units
    of memory construction to the evaluator.
    """

    info = DatasetInfo("abstract")

    @abstractmethod
    def conversation(self, conversation_id: str) -> Conversation:
        """Return one isolated ingestion unit."""

    @abstractmethod
    def conversations(self) -> list[Conversation]:
        """Return all ingestion units in deterministic order."""

    @abstractmethod
    def questions(self, conversation_ids: set[str] | None = None) -> list[Question]:
        """Return evaluation tasks, optionally filtered by ingestion unit."""

    def question(self, question_id: str) -> Question:
        for item in self.questions():
            if item.question_id == question_id:
                return item
        raise KeyError(question_id)

    def validate(self) -> list[str]:
        """Return structural errors without invoking a model."""
        errors: list[str] = []
        conversation_ids = {item.conversation_id for item in self.conversations()}
        seen: set[str] = set()
        for question in self.questions():
            if question.question_id in seen:
                errors.append(f"duplicate question_id: {question.question_id}")
            seen.add(question.question_id)
            if question.conversation_id not in conversation_ids:
                errors.append(
                    f"question {question.question_id} references missing conversation "
                    f"{question.conversation_id}"
                )
        return errors
