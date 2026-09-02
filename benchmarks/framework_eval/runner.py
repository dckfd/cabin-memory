from __future__ import annotations

import json
import time
from pathlib import Path

from .adapters.base import MemoryAdapter
from .schema import Conversation, Question


class RetrievalRunner:
    """Run memory construction/retrieval without requiring an answer LLM."""

    def __init__(self, adapter: MemoryAdapter, output: Path, *, limit: int = 8,
                 resume: bool = False, max_context_chars: int | None = None,
                 reflect_after_ingest: bool = False,
                 ready_timeout: float | None = None,
                 ingest: bool = True,
                 ingest_only: bool = False) -> None:
        if ingest_only and not ingest:
            raise ValueError("ingest_only requires ingest=True")
        if ingest:
            adapter.require("ingest")
        if not ingest_only:
            adapter.require("search")
        self.adapter = adapter
        self.output = output
        self.limit = limit
        self.resume = resume
        self.max_context_chars = max_context_chars
        self.reflect_after_ingest = reflect_after_ingest
        self.ready_timeout = ready_timeout
        self.ingest_enabled = ingest
        self.ingest_only = ingest_only
        output.parent.mkdir(parents=True, exist_ok=True)

    def run(self, conversations: list[Conversation], questions: list[Question]) -> dict:
        by_conversation: dict[str, list[Question]] = {}
        for question in questions:
            by_conversation.setdefault(question.conversation_id, []).append(question)
        completed_ids: set[str] = set()
        if self.resume and self.output.exists():
            with self.output.open(encoding="utf-8") as existing:
                completed_ids = {
                    str(json.loads(line)["question"]["question_id"])
                    for line in existing if line.strip()
                }
        completed = 0
        skipped = 0
        ingested_conversations = 0
        construction_assurance: dict[str, dict] = {}
        started = time.monotonic()
        with self.output.open("a" if self.resume else "w", encoding="utf-8") as handle:
            for conversation in conversations:
                selected = by_conversation.get(conversation.conversation_id, [])
                if not selected:
                    continue
                pending = [
                    question for question in selected
                    if question.question_id not in completed_ids
                ]
                # Resume must skip a fully completed conversation *before*
                # ingest. Rebuilding its store would duplicate memories and
                # spend extraction tokens even though every QA row is already
                # present in the output file.
                if self.resume and not pending:
                    skipped += len(selected)
                    continue
                ingest_seconds = 0.0
                if self.ingest_enabled:
                    ingest_started = time.monotonic()
                    self.adapter.ingest(conversation)
                    if self.reflect_after_ingest:
                        self.adapter.reflect(conversation.conversation_id)
                    ingest_seconds = time.monotonic() - ingest_started
                    ingested_conversations += 1
                if self.ingest_only:
                    continue
                assurance_started = time.monotonic()
                assurance = self.adapter.ensure_construction(
                    conversation, timeout=self.ready_timeout
                )
                assurance_seconds = time.monotonic() - assurance_started
                construction_assurance[conversation.conversation_id] = assurance
                readiness_seconds = 0.0
                if self.ready_timeout is not None:
                    readiness_started = time.monotonic()
                    self.adapter.wait_until_ready(
                        conversation.conversation_id, timeout=self.ready_timeout
                    )
                    readiness_seconds = time.monotonic() - readiness_started
                lifecycle_seconds = ingest_seconds + readiness_seconds
                for question in selected:
                    if question.question_id in completed_ids:
                        skipped += 1
                        continue
                    search_started = time.monotonic()
                    hits = self.adapter.search(question, limit=self.limit)
                    retrieved_sources = {
                        source for hit in hits for source in hit.source_ids
                    }
                    gold_sources = set(question.evidence_ids)
                    evidence_hits = retrieved_sources & gold_sources
                    rendered_context = self.adapter.render_hits(hits)
                    route_metadata = hits[0].metadata if hits else {}
                    route_budget = route_metadata.get(
                        "retrieval_context_budget_chars"
                    )
                    budgets = [
                        int(value) for value in (
                            self.max_context_chars, route_budget,
                        )
                        if value is not None and int(value) > 0
                    ]
                    effective_context_budget = min(budgets) if budgets else None
                    line_overflow_limit = max(0, int(
                        route_metadata.get(
                            "retrieval_context_line_overflow_chars", 0
                        ) or 0
                    ))
                    (
                        context,
                        context_truncated,
                        context_budget_exceeded,
                        context_budget_overflow,
                        context_boundary_action,
                    ) = _bounded_complete_line_context(
                        rendered_context,
                        effective_context_budget,
                        max_line_overflow_chars=line_overflow_limit,
                    )
                    row = {
                        "schema_version": 1,
                        "framework": self.adapter.adapter_id,
                        "question": question.to_dict(),
                        "hits": [hit.to_dict() for hit in hits],
                        "context": context,
                        "metrics": {
                            "hit_count": len(hits),
                            "context_chars": len(context),
                            "retrieved_context_chars": len(rendered_context),
                            "context_truncated": context_truncated,
                            "context_budget_exceeded": context_budget_exceeded,
                            "context_budget_chars": effective_context_budget,
                            "context_budget_overflow_chars": (
                                context_budget_overflow
                            ),
                            "context_line_overflow_limit_chars": (
                                line_overflow_limit
                            ),
                            "context_boundary_action": context_boundary_action,
                            "evidence_source_count": len(gold_sources),
                            "evidence_source_hits": len(evidence_hits),
                            "evidence_recall": (
                                len(evidence_hits) / len(gold_sources) if gold_sources else None
                            ),
                            "evidence_recall_scope": "retrieved_hits_before_context_truncation",
                            "search_seconds": time.monotonic() - search_started,
                            "conversation_ingest_seconds": ingest_seconds,
                            "construction_assurance_seconds": assurance_seconds,
                            "construction_assurance_status": assurance.get(
                                "status", "not_required"
                            ),
                            "construction_repair_attempts": assurance.get(
                                "repair_attempts", 0
                            ),
                            "readiness_seconds": readiness_seconds,
                            "conversation_lifecycle_seconds": lifecycle_seconds,
                            "store_reused": not self.ingest_enabled,
                            "retrieval_policy": route_metadata.get(
                                "retrieval_policy", "fixed"
                            ),
                            "retrieval_route": route_metadata.get(
                                "retrieval_route", "fixed"
                            ),
                            "adaptive_fallback": route_metadata.get(
                                "adaptive_fallback"
                            ),
                            "adaptive_reason": route_metadata.get(
                                "adaptive_reason"
                            ),
                            "adaptive_search_calls": route_metadata.get(
                                "adaptive_search_calls"
                            ),
                            "adaptive_profile_hits": route_metadata.get(
                                "adaptive_profile_hits"
                            ),
                            "adaptive_profile_levels": route_metadata.get(
                                "adaptive_profile_levels"
                            ),
                            "l23_schedule": route_metadata.get("l23_schedule"),
                            "temporal_query_mode": route_metadata.get(
                                "temporal_query_mode", "disabled"
                            ),
                            "temporal_query_active": route_metadata.get(
                                "temporal_query_active", False
                            ),
                            "temporal_query_relative": route_metadata.get(
                                "temporal_query_relative", False
                            ),
                            "temporal_query_anchor": route_metadata.get(
                                "temporal_query_anchor", ""
                            ),
                            "temporal_query_timezone": route_metadata.get(
                                "temporal_query_timezone", ""
                            ),
                            "temporal_query_spans": route_metadata.get(
                                "temporal_query_spans", []
                            ),
                            "temporal_candidate_hits": route_metadata.get(
                                "temporal_candidate_hits", 0
                            ),
                            "temporal_normalization_seconds": route_metadata.get(
                                "temporal_normalization_seconds", 0.0
                            ),
                            "temporal_candidate_seconds": route_metadata.get(
                                "temporal_candidate_seconds", 0.0
                            ),
                            "temporal_short_circuit_enabled": route_metadata.get(
                                "temporal_short_circuit_enabled", False
                            ),
                            "temporal_short_circuit_used": route_metadata.get(
                                "temporal_short_circuit_used", False
                            ),
                            "typed_episode_short_circuit_enabled": (
                                route_metadata.get(
                                    "typed_episode_short_circuit_enabled",
                                    False,
                                )
                            ),
                            "typed_episode_short_circuit_used": (
                                route_metadata.get(
                                    "typed_episode_short_circuit_used", False
                                )
                            ),
                            "typed_episode_candidate_hits": route_metadata.get(
                                "typed_episode_candidate_hits", 0
                            ),
                            "typed_episode_candidate_seconds": (
                                route_metadata.get(
                                    "typed_episode_candidate_seconds", 0.0
                                )
                            ),
                        },
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
        construction = self.adapter.construction_metrics()
        self.adapter.close()
        return {"questions": completed, "skipped": skipped,
                "ingested_conversations": ingested_conversations,
                "construction_assurance": construction_assurance,
                "construction": construction,
                "seconds": time.monotonic() - started,
                "output": str(self.output)}


def _bounded_complete_line_context(
    value: str,
    budget: int | None,
    *,
    max_line_overflow_chars: int = 0,
) -> tuple[str, bool, bool, int, str]:
    """Apply a character budget without emitting a partial dialogue line.

    The legacy hard character cap is preserved unless a route explicitly
    opts into a bounded line overflow. With the opt-in enabled, the current
    line is completed when its remainder fits the allowance; otherwise the
    partial line is removed. This bounds cost while avoiding misleading text
    fragments for weaker answer models.
    """
    if budget is None or budget <= 0 or len(value) <= budget:
        return value, False, False, 0, "none"

    hard_context = value[:budget]
    if max_line_overflow_chars <= 0:
        return hard_context, True, True, 0, "hard_char_limit"

    # If the hard cut is already at a line boundary there is nothing to fix.
    if value[budget - 1:budget] == "\n" or value[budget:budget + 1] == "\n":
        return hard_context, True, True, 0, "already_complete_line"

    newline = value.find("\n", budget)
    forward_boundary = len(value) if newline < 0 else newline + 1
    required_overflow = forward_boundary - budget
    if required_overflow <= max_line_overflow_chars:
        context = value[:forward_boundary]
        return (
            context,
            len(context) < len(value),
            True,
            required_overflow,
            "completed_current_line",
        )

    previous_boundary = value.rfind("\n", 0, budget)
    context = value[:previous_boundary + 1] if previous_boundary >= 0 else ""
    return context, len(context) < len(value), True, 0, "removed_partial_line"
