from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answering import _load_config_data, _resolve_env
from .plugins import PluginCatalog
from .schema import Conversation, Message, Session


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "benchmarks/framework_eval/profiles.json"
DATASETS = ROOT / "benchmarks/framework_eval/datasets.json"

LEDGER_CATEGORIES = frozenset({
    "identity", "residence", "relationship", "family", "pet", "education",
    "career", "skill", "instrument", "book", "music", "media", "event",
    "travel", "hobby", "art", "purchase", "possession", "health",
    "emotion", "preference", "plan", "community", "other",
})

LEDGER_SYSTEM_PROMPT = """You are a lossless long-term-memory compiler.

Extract reusable facts stated by each dialogue actor in the supplied dialogue. This
is memory construction, not question answering. You will not receive questions,
reference answers, or evidence labels beyond the source IDs attached to messages.

Requirements:
- Preserve secondary clauses as well as the main point of a message.
- Preserve exact names, titles, quantities, relationships, possessions, skills,
  purchases, plans, events, dates, and emotions together with their causes.
- Attribute a fact only to the actor whose message states it. Other actors may
  provide conversational context but are not evidence for that actor's memory.
- In task-oriented dialogue, preserve user requests and parameters as well as
  assistant selections, confirmations, tool outcomes, and recommendations. Never
  rewrite an assistant-provided value as if the user had stated it.
- Normalize an unambiguous relative date against source_date and put the result in
  event_date. Do not invent precision; leave event_date empty when ambiguous.
- Keep separate facts when later information adds another item to a set. Never
  discard an earlier item merely because a newer message mentions a different one.
- Lines labelled [Image caption: ...] and [Image query: ...] are supplied textual
  observations attached to that source message. Preserve their explicit names,
  visible text, colors, object counts, places, and other visual attributes when
  they complement what the speaker says. For example, if a speaker says they made
  a painting and its supplied caption describes a pink sky, retain both facts.
  Do not perform OCR, guess an unseen title, or add details absent from those lines.
- Do not infer unsupported facts or answer an implicit question.

Return only a JSON array. Every item must contain exactly these fields:
{
  "subject": "exact actor label shown in the dialogue",
  "category": "one category from the list below",
  "topic": "short queryable noun phrase",
  "statement": "standalone factual statement",
  "values": ["exact names, titles, quantities, or key values"],
  "event_date": "ISO date/date range or empty string",
  "source_ids": ["source message IDs"]
}

Categories:
identity, residence, relationship, family, pet, education, career, skill,
instrument, book, music, media, event, travel, hobby, art, purchase, possession,
health, emotion, preference, plan, community, other.

Use instrument for instruments somebody plays, book for books somebody read, music
for performers/bands/concerts, purchase for acquired items, and pet for pet species
or names. Use the dialogue's language for topic and statement."""

LEDGER_EXTRACT_VERSION = 3

LEDGER_ROLLUP_SYSTEM_PROMPT = """You are a source-grounded memory consolidator.

The input is a list of atomic memories for exactly one dialogue actor. Build compact
aggregate memories only where two or more atomic facts can be usefully combined.
You receive no questions, reference answers, or benchmark evidence.

General consolidation rules:
- Union exact members of sets such as pets, family members, instruments, books,
  performers seen, purchases, possessions, places, activities, and communities.
- Consolidate only facts with the same semantic relation. For example, keep
  "instruments played", "artists liked", and "performers actually seen" as three
  different memories even though all belong to music. Likewise, do not mix a
  purchased item with an item merely owned or mentioned.
- Produce each meaningful compatible aggregate supported by two or more facts;
  do not omit a purchase/instrument/pet/book inventory just because another topic
  has more source facts.
- Resolve entities only when atomic facts logically entail the resolution. Preserve
  uncertainty or conflicts instead of choosing an unsupported interpretation.
- For family counts, account for overlapping descriptions. A daughter may be one
  of "two younger kids" rather than an additional child. Comparative wording such
  as "two younger kids" entails at least one older child; report a minimum count
  when an exact count is not logically guaranteed.
- A count may be derived only when distinct entities or events are logically
  entailed. Deduplicate repeated descriptions using source IDs and dates.
- Distinguish an actual event from a habit, preference, plan, or frequency claim.
- Keep temporal versions separate when a set changed over time.
- Never drop exact names, titles, quantities, dates, or source IDs used to support
  the aggregate memory. Do not introduce a value absent from or not logically
  entailed by the atomic facts.

Return only a JSON array. Each aggregate item must contain:
{
  "subject": "the input subject",
  "category": "one category from the atomic-memory category list",
  "topic": "short queryable noun phrase",
  "statement": "standalone aggregate fact",
  "values": ["exact members, names, titles, quantities, or key values"],
  "event_date": "ISO date/date range or empty string",
  "source_ids": ["all supporting source IDs"],
  "derived": true,
  "derivation": "brief description of the source-grounded merge or inference"
}

Do not repeat a single atomic fact as an aggregate."""

LEDGER_ROLLUP_VERSION = 3

LEDGER_REASONING_ROLLUP_SYSTEM_PROMPT = """You are a source-grounded cross-session memory reasoner.

The input contains atomic memories for exactly one dialogue actor. Produce only compact
memories that require linking two or more atomic facts; ordinary single-fact
restatements and simple inventories are handled elsewhere. You receive no
questions, answers, or benchmark evidence.

Audit the memories for these general long-dialogue relations:
- Resolve a later definite or elliptical reference (for example, "the book X
  recommended") only when earlier facts identify a unique compatible entity.
- Derive exact or minimum entity counts from distinct family members, pets, or
  other entities. Plural people plus their separately mentioned sibling entails
  at least three people; "two younger children" entails at least one older child.
- Count distinct occurrences of the same activity within a stated period. Count
  actual events only, not habits, preferences, plans, or repeated descriptions of
  one event. Use source dates and source IDs to deduplicate.
- Link causes, outcomes, state changes, and temporal versions when the atomic
  facts explicitly support the relationship.

Be conservative. If multiple entities or event groupings remain possible, state
the uncertainty or a minimum count rather than inventing an exact answer. Preserve
every source ID used in the inference and never introduce an entity absent from
the atomic facts.

Return only a JSON array. Each item must contain:
{
  "subject": "the input subject",
  "category": "one category from the atomic-memory category list",
  "topic": "short queryable noun phrase",
  "statement": "standalone cross-session fact",
  "values": ["exact values or conservative derived values"],
  "event_date": "ISO date/date range or empty string",
  "source_ids": ["all supporting source IDs"],
  "derived": true,
  "derivation": "brief source-grounded reasoning"
}

Return an empty array when no safe cross-session inference exists."""

LEDGER_REASONING_ROLLUP_VERSION = 2

LEDGER_RELATION_ROLLUP_SYSTEM_PROMPT = """You are a source-grounded cross-actor memory linker.

The input contains a compact subset of raw dialogue messages and atomic memories
selected without access to benchmark questions or answers. Build only durable
relations that require evidence from at least two dialogue actors. Typical useful
relations include recommendations, advice followed, transfers, referrals, shared
plans, requests followed by selections, and confirmations.

Rules:
- Resolve a referred item (for example, "the book you recommended") only when the
  supplied messages identify one unique compatible item. If two candidates remain,
  return no relation instead of guessing.
- Preserve direction: who recommended/requested/offered what to whom. Set subject
  to the actor whose resulting action, state, or preference the memory describes.
- Cite every message needed to establish both ends of the relation. A relation must
  use at least two source IDs whose speakers include at least two different actors.
- Confidence is epistemic support from the supplied evidence, not importance. Use
  0.90 or above only for a unique, explicit chain; omit uncertain chains.
- Do not turn conversational politeness into a durable relation and do not infer
  facts from benchmark conventions.

Return only a JSON array. Every item must contain:
{
  "subject": "actor whose resulting state/action is stored",
  "category": "one atomic-memory category",
  "topic": "short queryable noun phrase",
  "statement": "standalone directional relation",
  "values": ["exact items, names, or values"],
  "event_date": "ISO date/date range or empty string",
  "source_ids": ["all supporting source message IDs"],
  "participants": ["all actors in the relation"],
  "relation_type": "recommendation|advice|transfer|referral|shared_plan|request_selection|confirmation|other",
  "confidence": 0.0,
  "derived": true,
  "derivation": "brief auditable link across the cited messages"
}

Return an empty array when no high-confidence cross-actor relation exists."""

LEDGER_RELATION_ROLLUP_VERSION = 1
LEDGER_RELATION_MIN_CONFIDENCE = 0.85

LEDGER_EVENT_ROLLUP_SYSTEM_PROMPT = """You are a source-grounded event-memory auditor.

The input contains a compact subset of raw dialogue messages and atomic memories
selected without access to benchmark questions, answers, or evidence labels. Build
only memories that improve event identity, deduplication, auditable counting, or a
conservative near-term projection from an explicitly severe negative outcome. The
input includes candidate_clusters created by deterministic lexical grouping. Audit
every supplied cluster in order. Return exactly one decision object for every
candidate_cluster_id, in the same order. Do not omit a cluster merely because
another cluster seems more important.

Rules:
- Treat source_date as the time a message was spoken. Treat event_date as the event
  time only when the text or an atomic fact supports it.
- Separate actual occurrences from habits, preferences, hypothetical examples, and
  future plans. A photo introduced as a real recent/past activity is event evidence;
  a generic statement such as "we love camping" is not another occurrence.
- Group repeated descriptions of one occurrence into one event_group. Different
  dates, destinations, or explicit words such as "another" can establish distinct
  event groups. Never count messages as events.
- For event_count, event_count must equal the number of distinct event_groups and
  each group must list its own supporting source IDs. Use a stated period only when
  all counted occurrences fall inside it.
- Preserve useful nested scopes. A specific place/activity cluster and a broader
  activity cluster are separate audit tasks; a broad count is not a substitute for
  a safe, source-grounded count in the narrower cluster.
- A causal_projection is allowed only for a near-term repeat decision that follows
  directly from an explicitly severe negative outcome or stated negative emotion.
  Phrase it as likely/unlikely, never as a permanent preference or certainty.
  For example, a recent journey that the same actor describes across messages as
  involving an accident, a bad start, and fear supports "unlikely to repeat that
  journey soon"; it never supports "will never travel again".
- Cite at least two source messages spoken by the subject. Preserve uncertainty and
  return no item if identity or causality is ambiguous.

Return only a JSON array. An accepted decision must contain:
{
  "candidate_cluster_id": "exact ID from the input",
  "decision": "accept",
  "subject": "exact dialogue actor",
  "category": "event, travel, plan, emotion, or another atomic-memory category",
  "topic": "short queryable noun phrase",
  "statement": "standalone audited event memory",
  "values": ["exact values or conservative derived values"],
  "event_date": "ISO date/date range/period or empty string",
  "source_ids": ["all supporting source message IDs"],
  "memory_kind": "event_identity|event_count|causal_projection",
  "event_groups": [
    {"event_key": "stable descriptive identity", "event_date": "date or empty", "source_ids": ["IDs for this occurrence"]}
  ],
  "event_count": 0,
  "projection_horizon": "near_term or empty",
  "confidence": 0.0,
  "derived": true,
  "derivation": "brief auditable identity/count/causal reasoning"
}

For event_identity, provide one event_group backed by multiple descriptions. For
causal_projection, event_groups may contain the causal event and event_count is 0.
For an unsafe cluster, return only:
{"candidate_cluster_id":"exact ID", "decision":"reject", "reason":"brief reason"}
Never return an empty array when candidate_clusters are supplied."""

LEDGER_EVENT_ROLLUP_VERSION = 3
LEDGER_EVENT_MIN_CONFIDENCE = 0.80
LEDGER_EVENT_POSTPROCESS_VERSION = 1

_RELATION_SOURCE_RE = re.compile(
    r"\b(?:recommend(?:ed|ing|ation)?|suggest(?:ed|ing|ion)?|advi[cs](?:e|ed|ing)?|"
    r"told|asked|gave|given|sent|shared|borrow(?:ed|ing)?|lent|offer(?:ed|ing)?|"
    r"refer(?:red|ral)?|request(?:ed|ing)?|confirm(?:ed|ing|ation)?|"
    r"select(?:ed|ing|ion)?|chose|chosen|taking\s+your)\b",
    flags=re.IGNORECASE,
)

_EVENT_CANDIDATE_CATEGORIES = frozenset({"event", "travel"})

_EVENT_CAUSAL_RE = re.compile(
    r"\b(?:accident|crash|injur(?:y|ed)|danger(?:ous)?|scared|terrified|"
    r"frighten(?:ed|ing)?|awful|horrible|bad\s+start|went\s+badly|"
    r"disaster|traumatic|traumatized)\b",
    flags=re.IGNORECASE,
)

_EXPLICIT_OCCURRENCE_RE = re.compile(
    r"(?:\bhere['’]?s\s+(?:a\s+)?(?:pic|photo)\b|"
    r"\b(?:i|we|my\s+family|our\s+family)\s+"
    r"(?:(?:just|recently|actually|also)\s+)?"
    r"(?:went|visited|attended|camped|traveled|travelled|took|stayed|saw|did)\b)",
    flags=re.IGNORECASE,
)

_EVENT_LINK_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "during", "after", "before",
    "into", "that", "this", "their", "there", "where", "when", "what",
    "her", "his", "she", "they", "them", "was", "were", "are", "has",
    "had", "did", "went", "gone", "recently", "family", "kids", "children",
    "child", "trip", "event", "activity", "activities", "time", "day",
    "week", "weekend", "month", "year", "actual", "another",
})


def _json_array(raw: str) -> list[dict[str, Any]]:
    """Parse a model response while tolerating an optional Markdown fence."""
    value = str(raw or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start, end = value.find("["), value.rfind("]")
    if start < 0 or end < start:
        raise ValueError("ledger extraction response contains no JSON array")
    parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, list):
        raise ValueError("ledger extraction response must be a JSON array")
    return [item for item in parsed if isinstance(item, dict)]


def _message_actor(message: Message) -> str:
    speaker = str(message.speaker or "").strip()
    if speaker:
        return speaker
    role = str(message.role or "").strip().casefold()
    return {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
        "tool": "Tool",
    }.get(role, role.title() or "Unknown")


def _session_prompt(session: Session) -> str:
    lines: list[str] = []
    for message in session.messages:
        if not message.content:
            continue
        lines.append(
            f"[{message.message_id}] [source_date={message.timestamp}] "
            f"{_message_actor(message)}: {message.content}"
        )
    return "\n".join(lines)


def _conversation_source_rows(
    conversation: Conversation,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for session in conversation.sessions:
        for message in session.messages:
            source_id = str(message.message_id or "").strip()
            text = str(message.render_text() or "").strip()
            if not source_id or not text:
                continue
            rows.append({
                "source_id": source_id,
                "session_id": str(session.session_id or "").strip(),
                "source_date": str(message.timestamp or session.timestamp or "").strip(),
                "actor": _message_actor(message),
                "text": text,
            })
    return rows


def _relation_rollup_input(
    conversation: Conversation,
    atomic_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select relation-bearing construction evidence without benchmark labels."""
    source_rows = [
        row for row in _conversation_source_rows(conversation)
        if _RELATION_SOURCE_RE.search(row["text"])
    ]
    source_ids = {row["source_id"] for row in source_rows}
    facts = [
        fact for fact in atomic_facts
        if source_ids.intersection(
            str(source_id) for source_id in (fact.get("source_ids") or [])
        )
    ]
    return {
        "conversation_id": conversation.conversation_id,
        "actors": list(dict.fromkeys(row["actor"] for row in source_rows)),
        "source_messages": source_rows,
        "atomic_facts": facts,
    }


def _event_link_tokens(fact: dict[str, Any], actors: set[str]) -> set[str]:
    values = " ".join(str(value) for value in (fact.get("values") or []))
    # Exact extracted values are better event identity anchors than generic
    # statement verbs such as "went", "attended", or "last". Fall back to the
    # topic only when extraction supplied no values.
    rendered = (values or str(fact.get("topic") or ""))
    rendered = rendered.casefold().replace("roadtrip", "road trip")
    actor_tokens = {
        token
        for actor in actors
        for token in re.findall(r"\w+", actor.casefold())
    }
    return {
        token
        for token in re.findall(r"\w+", rendered)
        if len(token) > 2
        and token not in _EVENT_LINK_STOPWORDS
        and token not in actor_tokens
        and not token.isdigit()
    }


def _event_rollup_input(
    conversation: Conversation,
    atomic_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select event-bearing evidence, retaining co-extracted causal details."""
    all_rows = _conversation_source_rows(conversation)
    actors = {row["actor"] for row in all_rows}
    event_facts = [
        fact for fact in atomic_facts
        if str(fact.get("category") or "").casefold()
        in _EVENT_CANDIDATE_CATEGORIES
    ]
    token_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for index, fact in enumerate(event_facts):
        subject = str(fact.get("subject") or "").strip()
        sources = {
            str(value).strip() for value in (fact.get("source_ids") or [])
            if str(value).strip()
        }
        for token in _event_link_tokens(fact, actors):
            group = token_groups.setdefault((subject, token), {
                "fact_indexes": [], "source_ids": set(),
            })
            group["fact_indexes"].append(index)
            group["source_ids"].update(sources)
    candidate_clusters: list[dict[str, Any]] = []
    seed_ids: set[str] = set()
    for (subject, token), group in sorted(token_groups.items()):
        source_ids = sorted(group["source_ids"])
        if len(group["fact_indexes"]) < 2 or len(source_ids) < 2:
            continue
        candidate_clusters.append({
            "cluster_kind": "repeated_event",
            "subject": subject,
            "shared_anchor": token,
            "source_ids": source_ids,
        })
        seed_ids.update(source_ids)

    # Severe outcomes and explicit negative emotions support a conservative
    # near-term projection. Same-session grouping prevents unrelated bad events
    # across months from being fused into one causal chain.
    causal_groups: dict[tuple[str, str], list[str]] = {}
    for row in all_rows:
        if not _EVENT_CAUSAL_RE.search(row["text"]):
            continue
        group_key = (row["actor"], row["session_id"])
        causal_groups.setdefault(group_key, []).append(row["source_id"])
    for (subject, session_id), source_ids in sorted(causal_groups.items()):
        source_ids = list(dict.fromkeys(source_ids))
        if len(source_ids) < 2:
            continue
        candidate_clusters.append({
            "cluster_kind": "negative_outcome",
            "subject": subject,
            "shared_anchor": "severe negative outcome in one session",
            "session_id": session_id,
            "source_ids": source_ids,
        })
        seed_ids.update(source_ids)
    for index, cluster in enumerate(candidate_clusters, 1):
        cluster["candidate_cluster_id"] = f"event-cluster-{index:03d}"
    source_rows = [
        row for row in all_rows
        if row["source_id"] in seed_ids
    ]
    source_ids = {row["source_id"] for row in source_rows}
    facts = [
        fact for fact in atomic_facts
        if source_ids.intersection(
            str(source_id) for source_id in (fact.get("source_ids") or [])
        )
    ]
    return {
        "conversation_id": conversation.conversation_id,
        "actors": list(dict.fromkeys(row["actor"] for row in source_rows)),
        "source_messages": source_rows,
        "atomic_facts": facts,
        "candidate_clusters": candidate_clusters,
    }


def _validate_session_facts(
    raw_facts: list[dict[str, Any]], session: Session,
) -> list[dict[str, Any]]:
    """Keep only source-grounded, correctly attributed facts."""
    source_actors = {
        str(message.message_id): _message_actor(message)
        for message in session.messages
        if str(message.message_id).strip()
    }
    actors = set(source_actors.values())
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for row in raw_facts:
        subject = str(row.get("subject") or "").strip()
        statement = " ".join(str(row.get("statement") or "").split())
        if subject not in actors or not statement:
            continue
        source_ids = tuple(dict.fromkeys(
            str(value).strip()
            for value in (row.get("source_ids") or [])
            if str(value).strip()
            and source_actors.get(str(value).strip()) == subject
        ))
        if not source_ids:
            continue
        category = str(row.get("category") or "other").strip().lower()
        if category not in LEDGER_CATEGORIES:
            category = "other"
        topic = " ".join(str(row.get("topic") or category).split())
        values = list(dict.fromkeys(
            " ".join(str(value).split())
            for value in (row.get("values") or [])
            if " ".join(str(value).split())
        ))
        event_date = " ".join(str(row.get("event_date") or "").split())
        key = (subject, category, statement.casefold(), source_ids)
        if key in seen:
            continue
        seen.add(key)
        validated.append({
            "subject": subject,
            "category": category,
            "topic": topic,
            "statement": statement,
            "values": values,
            "event_date": event_date,
            "source_ids": list(source_ids),
        })
    return validated


class _LedgerClient:
    def __init__(self, config: dict[str, Any], *, max_tokens: int, retries: int) -> None:
        self.base_url = str(config["base_url"]).rstrip("/")
        self.api_key = str(config.get("api_key") or "")
        self.model = str(config["model"])
        self.timeout = int(config.get("timeout_seconds", 300))
        self.max_tokens = max(512, int(max_tokens))
        self.retries = max(0, int(retries))

    def _complete(
        self, system: str, user: str,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "enable_thinking": False,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
                content = str(result["choices"][0]["message"]["content"])
                usage = {
                    key: int((result.get("usage") or {}).get(key, 0))
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                return _json_array(content), usage
            except Exception:
                if attempt >= self.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    def extract(self, session: Session) -> tuple[list[dict[str, Any]], dict[str, int]]:
        raw, usage = self._complete(LEDGER_SYSTEM_PROMPT, _session_prompt(session))
        return _validate_session_facts(raw, session), usage

    def rollup(
        self, subject: str, facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        raw, usage = self._complete(
            LEDGER_ROLLUP_SYSTEM_PROMPT,
            json.dumps({"subject": subject, "atomic_facts": facts}, ensure_ascii=False),
        )
        return _validate_rollup_facts(raw, subject, facts), usage

    def reasoning_rollup(
        self, subject: str, facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        raw, usage = self._complete(
            LEDGER_REASONING_ROLLUP_SYSTEM_PROMPT,
            json.dumps({"subject": subject, "atomic_facts": facts}, ensure_ascii=False),
        )
        return _validate_rollup_facts(
            raw, subject, facts, fact_kind="reasoning_rollup"
        ), usage

    def relation_rollup(
        self, conversation: Conversation, payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        raw, usage = self._complete(
            LEDGER_RELATION_ROLLUP_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False),
        )
        return _validate_relation_rollup_facts(raw, conversation, payload), usage

    def event_rollup(
        self, conversation: Conversation, payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        raw, usage = self._complete(
            LEDGER_EVENT_ROLLUP_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False),
        )
        return _validate_event_rollup_facts(raw, conversation, payload), usage


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _deduplicate_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for fact in facts:
        key = (
            str(fact["subject"]),
            str(fact["category"]),
            str(fact["statement"]).casefold(),
            tuple(str(value) for value in fact["source_ids"]),
        )
        if key not in seen:
            result.append(fact)
            seen.add(key)
    return result


def _attach_source_dates(
    facts: list[dict[str, Any]], conversation: Conversation,
) -> list[dict[str, Any]]:
    """Attach message dates as provenance, never as inferred event dates."""
    source_dates = {
        str(message.message_id): str(message.timestamp).strip()
        for session in conversation.sessions
        for message in session.messages
        if str(message.message_id).strip() and str(message.timestamp).strip()
    }
    result: list[dict[str, Any]] = []
    for fact in facts:
        row = dict(fact)
        row["source_dates"] = list(dict.fromkeys(
            source_dates[source_id]
            for source_id in row.get("source_ids") or []
            if source_id in source_dates
        ))
        result.append(row)
    return result


def _validate_rollup_facts(
    raw_facts: list[dict[str, Any]],
    subject: str,
    atomic_facts: list[dict[str, Any]],
    *,
    fact_kind: str = "rollup",
) -> list[dict[str, Any]]:
    allowed_sources = {
        str(source_id)
        for fact in atomic_facts
        for source_id in (fact.get("source_ids") or [])
    }
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for row in raw_facts:
        if str(row.get("subject") or "").strip() != subject:
            continue
        statement = " ".join(str(row.get("statement") or "").split())
        source_ids = tuple(dict.fromkeys(
            str(value).strip()
            for value in (row.get("source_ids") or [])
            if str(value).strip() in allowed_sources
        ))
        # A rollup must actually combine evidence, not restate one atom.
        if not statement or len(source_ids) < 2:
            continue
        category = str(row.get("category") or "other").strip().lower()
        if category not in LEDGER_CATEGORIES:
            category = "other"
        topic = " ".join(str(row.get("topic") or category).split())
        values = list(dict.fromkeys(
            " ".join(str(value).split())
            for value in (row.get("values") or [])
            if " ".join(str(value).split())
        ))
        key = (category, statement.casefold(), source_ids)
        if key in seen:
            continue
        seen.add(key)
        validated.append({
            "subject": subject,
            "category": category,
            "topic": topic,
            "statement": statement,
            "values": values,
            "event_date": " ".join(str(row.get("event_date") or "").split()),
            "source_ids": list(source_ids),
            "fact_kind": fact_kind,
            "derived": bool(row.get("derived", True)),
            "derivation": " ".join(str(row.get("derivation") or "").split()),
        })
    return validated


def _row_confidence(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if 0.0 <= value <= 1.0 else 0.0


def _normalized_fact_fields(
    row: dict[str, Any], *, default_category: str = "other",
) -> tuple[str, str, str, list[str], str]:
    category = str(row.get("category") or default_category).strip().casefold()
    if category not in LEDGER_CATEGORIES:
        category = default_category
    topic = " ".join(str(row.get("topic") or category).split())
    statement = " ".join(str(row.get("statement") or "").split())
    values = list(dict.fromkeys(
        " ".join(str(value).split())
        for value in (row.get("values") or [])
        if " ".join(str(value).split())
    ))
    event_date = " ".join(str(row.get("event_date") or "").split())
    return category, topic, statement, values, event_date


def _validate_relation_rollup_facts(
    raw_facts: list[dict[str, Any]],
    conversation: Conversation,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    source_actors = {
        str(row.get("source_id") or "").strip(): str(row.get("actor") or "").strip()
        for row in (payload.get("source_messages") or [])
        if str(row.get("source_id") or "").strip()
        and str(row.get("actor") or "").strip()
    }
    actors = {
        _message_actor(message)
        for session in conversation.sessions
        for message in session.messages
    }
    relation_types = {
        "recommendation", "advice", "transfer", "referral", "shared_plan",
        "request_selection", "confirmation", "other",
    }
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for row in raw_facts:
        subject = str(row.get("subject") or "").strip()
        confidence = _row_confidence(row)
        category, topic, statement, values, event_date = _normalized_fact_fields(row)
        if subject not in actors or not statement or confidence < LEDGER_RELATION_MIN_CONFIDENCE:
            continue
        source_ids = tuple(dict.fromkeys(
            str(value).strip()
            for value in (row.get("source_ids") or [])
            if str(value).strip() in source_actors
        ))
        source_actor_set = {source_actors[source_id] for source_id in source_ids}
        # Both ends of a cross-actor link must be directly auditable.
        if len(source_ids) < 2 or len(source_actor_set) < 2:
            continue
        participants = tuple(dict.fromkeys(
            str(value).strip()
            for value in (row.get("participants") or [])
            if str(value).strip() in actors
        ))
        if subject not in participants or not source_actor_set.issubset(participants):
            continue
        relation_type = str(row.get("relation_type") or "other").strip().casefold()
        if relation_type not in relation_types:
            relation_type = "other"
        key = (subject, relation_type, statement.casefold(), source_ids)
        if key in seen:
            continue
        seen.add(key)
        validated.append({
            "subject": subject,
            "category": category,
            "topic": topic,
            "statement": statement,
            "values": values,
            "event_date": event_date,
            "source_ids": list(source_ids),
            "fact_kind": "relation_rollup",
            "participants": list(participants),
            "relation_type": relation_type,
            "confidence": confidence,
            "derived": True,
            "derivation": " ".join(str(row.get("derivation") or "").split()),
        })
    return validated


def _validate_event_rollup_facts(
    raw_facts: list[dict[str, Any]],
    conversation: Conversation,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    source_rows = {
        str(row.get("source_id") or "").strip(): row
        for row in (payload.get("source_messages") or [])
        if str(row.get("source_id") or "").strip()
    }
    candidate_clusters = {
        str(cluster.get("candidate_cluster_id") or "").strip(): cluster
        for cluster in (payload.get("candidate_clusters") or [])
        if str(cluster.get("candidate_cluster_id") or "").strip()
    }
    actors = {
        _message_actor(message)
        for session in conversation.sessions
        for message in session.messages
    }
    memory_kinds = {"event_identity", "event_count", "causal_projection"}
    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for row in raw_facts:
        if str(row.get("decision") or "").strip().casefold() != "accept":
            continue
        candidate_cluster_id = str(
            row.get("candidate_cluster_id") or ""
        ).strip()
        candidate_cluster = candidate_clusters.get(candidate_cluster_id)
        if candidate_cluster is None:
            continue
        subject = str(row.get("subject") or "").strip()
        confidence = _row_confidence(row)
        category, topic, statement, values, event_date = _normalized_fact_fields(
            row, default_category="event"
        )
        memory_kind = str(row.get("memory_kind") or "").strip().casefold()
        if (
            subject not in actors
            or subject != str(candidate_cluster.get("subject") or "").strip()
            or not statement
            or memory_kind not in memory_kinds
            or confidence < LEDGER_EVENT_MIN_CONFIDENCE
        ):
            continue
        source_ids = tuple(dict.fromkeys(
            str(value).strip()
            for value in (row.get("source_ids") or [])
            if str(value).strip() in source_rows
            and str(source_rows[str(value).strip()].get("actor") or "").strip()
            == subject
        ))
        if len(source_ids) < 2:
            continue
        cluster_sources = {
            str(value).strip()
            for value in (candidate_cluster.get("source_ids") or [])
            if str(value).strip()
        }
        if not set(source_ids).issubset(cluster_sources):
            continue

        event_groups: list[dict[str, Any]] = []
        group_seen: set[tuple[str, tuple[str, ...]]] = set()
        for group in row.get("event_groups") or []:
            if not isinstance(group, dict):
                continue
            group_ids = tuple(dict.fromkeys(
                str(value).strip()
                for value in (group.get("source_ids") or [])
                if str(value).strip() in source_ids
            ))
            event_key = " ".join(str(group.get("event_key") or "").split())
            if not event_key or not group_ids:
                continue
            group_key = (event_key.casefold(), group_ids)
            if group_key in group_seen:
                continue
            group_seen.add(group_key)
            event_groups.append({
                "event_key": event_key,
                "event_date": " ".join(
                    str(group.get("event_date") or "").split()
                ),
                "source_ids": list(group_ids),
            })

        try:
            event_count = int(row.get("event_count") or 0)
        except (TypeError, ValueError):
            event_count = 0
        projection_horizon = " ".join(
            str(row.get("projection_horizon") or "").split()
        ).casefold()
        grouped_sources = {
            source_id
            for group in event_groups
            for source_id in group["source_ids"]
        }
        if memory_kind == "event_count":
            if candidate_cluster.get("cluster_kind") != "repeated_event":
                continue
            if (
                len(event_groups) < 2
                or event_count != len(event_groups)
                or grouped_sources != set(source_ids)
            ):
                continue
            if str(event_count) not in values:
                values.append(str(event_count))
        elif memory_kind == "event_identity":
            if candidate_cluster.get("cluster_kind") != "repeated_event":
                continue
            if (
                len(event_groups) != 1
                or len(event_groups[0]["source_ids"]) < 2
                or grouped_sources != set(source_ids)
            ):
                continue
            event_count = 1
        else:
            if candidate_cluster.get("cluster_kind") != "negative_outcome":
                continue
            if projection_horizon != "near_term":
                continue
            negative_source_count = sum(
                1 for source_id in source_ids
                if _EVENT_CAUSAL_RE.search(
                    str(source_rows[source_id].get("text") or "")
                )
            )
            if negative_source_count < 2:
                continue
            event_count = 0

        key = (subject, memory_kind, statement.casefold(), source_ids)
        if key in seen:
            continue
        seen.add(key)
        validated.append({
            "subject": subject,
            "candidate_cluster_id": candidate_cluster_id,
            "category": category,
            "topic": topic,
            "statement": statement,
            "values": values,
            "event_date": event_date,
            "source_ids": list(source_ids),
            "fact_kind": "event_rollup",
            "memory_kind": memory_kind,
            "event_groups": event_groups,
            "event_count": event_count,
            "projection_horizon": projection_horizon,
            "confidence": confidence,
            "derived": True,
            "derivation": " ".join(str(row.get("derivation") or "").split()),
        })
    return validated


def _deterministic_pair_event_facts(
    payload: dict[str, Any],
    existing_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill exact two-occurrence clusters using strict, auditable text rules.

    This intentionally handles only the high-precision case: two source messages,
    different source dates, the same extracted anchor in both raw messages, and an
    explicit first-person occurrence (including an explicitly introduced photo).
    Larger clusters and ambiguous habits remain model-audited.
    """
    completed_clusters = {
        str(fact.get("candidate_cluster_id") or "").strip()
        for fact in existing_facts
        if str(fact.get("candidate_cluster_id") or "").strip()
    }
    source_rows = {
        str(row.get("source_id") or "").strip(): row
        for row in (payload.get("source_messages") or [])
        if str(row.get("source_id") or "").strip()
    }
    atomic_facts = list(payload.get("atomic_facts") or [])
    result: list[dict[str, Any]] = []
    for cluster in payload.get("candidate_clusters") or []:
        cluster_id = str(cluster.get("candidate_cluster_id") or "").strip()
        if (
            not cluster_id
            or cluster_id in completed_clusters
            or cluster.get("cluster_kind") != "repeated_event"
        ):
            continue
        source_ids = list(dict.fromkeys(
            str(value).strip()
            for value in (cluster.get("source_ids") or [])
            if str(value).strip() in source_rows
        ))
        if len(source_ids) != 2:
            continue
        anchor = " ".join(str(cluster.get("shared_anchor") or "").split())
        subject = " ".join(str(cluster.get("subject") or "").split())
        rows = [source_rows[source_id] for source_id in source_ids]
        source_dates = [str(row.get("source_date") or "").strip() for row in rows]
        if (
            not anchor
            or not subject
            or not all(source_dates)
            or len(set(source_dates)) != 2
            or not all(
                anchor.casefold() in str(row.get("text") or "").casefold()
                and _EXPLICIT_OCCURRENCE_RE.search(str(row.get("text") or ""))
                for row in rows
            )
        ):
            continue
        years = [
            match.group(1)
            for value in source_dates
            if (match := re.search(r"\b(20\d{2})\b", value))
        ]
        period = years[0] if len(years) == 2 and len(set(years)) == 1 else ""
        related_categories = {
            str(fact.get("category") or "").casefold()
            for fact in atomic_facts
            if set(source_ids).intersection(
                str(value) for value in (fact.get("source_ids") or [])
            )
        }
        category = "travel" if "travel" in related_categories else "event"
        period_phrase = f" in {period}" if period else ""
        result.append({
            "subject": subject,
            "candidate_cluster_id": cluster_id,
            "category": category,
            "topic": f"{period + ' ' if period else ''}{anchor} occurrences",
            "statement": (
                f"{subject} has two distinct documented {anchor} occurrences"
                f"{period_phrase}."
            ),
            "values": [anchor, "2"],
            "event_date": period,
            "source_ids": source_ids,
            "fact_kind": "event_rollup",
            "memory_kind": "event_count",
            "event_groups": [
                {
                    "event_key": f"{anchor} occurrence documented at {source_date}",
                    "event_date": "",
                    "source_ids": [source_id],
                }
                for source_id, source_date in zip(source_ids, source_dates)
            ],
            "event_count": 2,
            "projection_horizon": "",
            "confidence": 0.90,
            "derived": True,
            "derivation": (
                "Strict pair rule: both differently dated source messages explicitly "
                f"describe an actual {anchor} occurrence; no plan/habit-only source "
                "or repeated same-date description was counted."
            ),
            "construction_method": "deterministic_pair_audit",
        })
    return result


def build_ledger(
    conversations: list[Conversation],
    *,
    client: _LedgerClient,
    output: Path,
    concurrency: int,
    resume: bool,
    rollup: bool = True,
    audited_rollup: bool = False,
    reuse_base_checkpoint: bool = False,
) -> dict[str, Any]:
    checkpoint_path = output.with_name(output.name + ".checkpoint.json")
    checkpoint: dict[str, Any] = {
        "sessions": {}, "rollups": {}, "reasoning_rollups": {},
        "relation_rollups": {}, "event_rollups": {},
    }
    if resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed = dict(checkpoint.get("sessions") or {})
    work: list[tuple[str, Session]] = []
    for conversation in conversations:
        for session in conversation.sessions:
            key = f"{conversation.conversation_id}/{session.session_id}"
            if (
                key not in completed
                or (
                    not reuse_base_checkpoint
                    and int(completed[key].get("version", 0))
                    != LEDGER_EXTRACT_VERSION
                )
            ):
                work.append((key, session))

    def extract(item: tuple[str, Session]):
        key, session = item
        facts, usage = client.extract(session)
        return key, facts, usage

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(concurrency))
    ) as executor:
        futures = [executor.submit(extract, item) for item in work]
        for future in concurrent.futures.as_completed(futures):
            key, facts, usage = future.result()
            completed[key] = {
                "version": LEDGER_EXTRACT_VERSION,
                "facts": facts,
                "usage": usage,
            }
            checkpoint = {
                "schema_version": 2,
                "sessions": completed,
                "rollups": checkpoint.get("rollups") or {},
                "reasoning_rollups": checkpoint.get("reasoning_rollups") or {},
                "relation_rollups": checkpoint.get("relation_rollups") or {},
                "event_rollups": checkpoint.get("event_rollups") or {},
            }
            _atomic_write_json(checkpoint_path, checkpoint)

    conversation_rows: dict[str, Any] = {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rollups = dict(checkpoint.get("rollups") or {})
    reasoning_rollups = dict(checkpoint.get("reasoning_rollups") or {})
    relation_rollups = dict(checkpoint.get("relation_rollups") or {})
    event_rollups = dict(checkpoint.get("event_rollups") or {})
    for conversation in conversations:
        atomic_facts: list[dict[str, Any]] = []
        for session in conversation.sessions:
            key = f"{conversation.conversation_id}/{session.session_id}"
            row = completed[key]
            atomic_facts.extend(row.get("facts") or [])
            for token_key in usage:
                usage[token_key] += int((row.get("usage") or {}).get(token_key, 0))
        atomic_facts = _attach_source_dates(
            _deduplicate_facts(atomic_facts), conversation
        )
        actors = list(dict.fromkeys(
            _message_actor(message)
            for session in conversation.sessions
            for message in session.messages
        ))
        aggregate_facts: list[dict[str, Any]] = []
        reasoning_facts: list[dict[str, Any]] = []
        relation_facts: list[dict[str, Any]] = []
        event_facts: list[dict[str, Any]] = []
        if rollup:
            for subject in actors:
                key = f"{conversation.conversation_id}/{subject}"
                if (
                    key not in rollups
                    or (
                        not reuse_base_checkpoint
                        and (
                            int(rollups[key].get("version", 0))
                            != LEDGER_ROLLUP_VERSION
                            or int(rollups[key].get("extract_version", 0))
                            != LEDGER_EXTRACT_VERSION
                        )
                    )
                ):
                    subject_facts = [
                        fact for fact in atomic_facts
                        if fact.get("subject") == subject
                    ]
                    facts, rollup_usage = client.rollup(subject, subject_facts)
                    rollups[key] = {
                        "version": LEDGER_ROLLUP_VERSION,
                        "extract_version": LEDGER_EXTRACT_VERSION,
                        "facts": facts,
                        "usage": rollup_usage,
                    }
                    checkpoint = {
                        "schema_version": 2,
                        "sessions": completed,
                        "rollups": rollups,
                        "reasoning_rollups": reasoning_rollups,
                        "relation_rollups": relation_rollups,
                        "event_rollups": event_rollups,
                    }
                    _atomic_write_json(checkpoint_path, checkpoint)
                aggregate = rollups[key]
                aggregate_facts.extend(aggregate.get("facts") or [])
                for token_key in usage:
                    usage[token_key] += int(
                        (aggregate.get("usage") or {}).get(token_key, 0)
                    )
                if (
                    key not in reasoning_rollups
                    or (
                        not reuse_base_checkpoint
                        and (
                            int(reasoning_rollups[key].get("version", 0))
                            != LEDGER_REASONING_ROLLUP_VERSION
                            or int(reasoning_rollups[key].get("extract_version", 0))
                            != LEDGER_EXTRACT_VERSION
                        )
                    )
                ):
                    subject_facts = [
                        fact for fact in atomic_facts
                        if fact.get("subject") == subject
                    ]
                    facts, reasoning_usage = client.reasoning_rollup(
                        subject, subject_facts
                    )
                    reasoning_rollups[key] = {
                        "version": LEDGER_REASONING_ROLLUP_VERSION,
                        "extract_version": LEDGER_EXTRACT_VERSION,
                        "facts": facts,
                        "usage": reasoning_usage,
                    }
                    checkpoint = {
                        "schema_version": 2,
                        "sessions": completed,
                        "rollups": rollups,
                        "reasoning_rollups": reasoning_rollups,
                        "relation_rollups": relation_rollups,
                        "event_rollups": event_rollups,
                    }
                    _atomic_write_json(checkpoint_path, checkpoint)
                reasoning = reasoning_rollups[key]
                reasoning_facts.extend(reasoning.get("facts") or [])
                for token_key in usage:
                    usage[token_key] += int(
                        (reasoning.get("usage") or {}).get(token_key, 0)
                    )
        if audited_rollup:
            graph_key = str(conversation.conversation_id)
            if (
                graph_key not in relation_rollups
                or int(relation_rollups[graph_key].get("version", 0))
                != LEDGER_RELATION_ROLLUP_VERSION
                or int(relation_rollups[graph_key].get("extract_version", 0))
                != LEDGER_EXTRACT_VERSION
            ):
                relation_payload = _relation_rollup_input(
                    conversation, atomic_facts
                )
                relation_actors = {
                    str(row.get("actor") or "")
                    for row in relation_payload["source_messages"]
                }
                if (
                    len(relation_payload["source_messages"]) >= 2
                    and len(relation_actors) >= 2
                ):
                    facts, relation_usage = client.relation_rollup(
                        conversation, relation_payload
                    )
                else:
                    facts = []
                    relation_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                relation_rollups[graph_key] = {
                    "version": LEDGER_RELATION_ROLLUP_VERSION,
                    "extract_version": LEDGER_EXTRACT_VERSION,
                    "facts": facts,
                    "usage": relation_usage,
                    "candidate_source_count": len(
                        relation_payload["source_messages"]
                    ),
                    "candidate_fact_count": len(
                        relation_payload["atomic_facts"]
                    ),
                }
                checkpoint = {
                    "schema_version": 2,
                    "sessions": completed,
                    "rollups": rollups,
                    "reasoning_rollups": reasoning_rollups,
                    "relation_rollups": relation_rollups,
                    "event_rollups": event_rollups,
                }
                _atomic_write_json(checkpoint_path, checkpoint)
            relation = relation_rollups[graph_key]
            relation_facts.extend(relation.get("facts") or [])
            for token_key in usage:
                usage[token_key] += int(
                    (relation.get("usage") or {}).get(token_key, 0)
                )

            event_payload = _event_rollup_input(conversation, atomic_facts)
            if (
                graph_key not in event_rollups
                or int(event_rollups[graph_key].get("version", 0))
                != LEDGER_EVENT_ROLLUP_VERSION
                or int(event_rollups[graph_key].get("extract_version", 0))
                != LEDGER_EXTRACT_VERSION
            ):
                event_actor_counts: dict[str, int] = {}
                for row in event_payload["source_messages"]:
                    actor = str(row.get("actor") or "")
                    event_actor_counts[actor] = event_actor_counts.get(actor, 0) + 1
                if any(count >= 2 for count in event_actor_counts.values()):
                    facts, event_usage = client.event_rollup(
                        conversation, event_payload
                    )
                else:
                    facts = []
                    event_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                event_rollups[graph_key] = {
                    "version": LEDGER_EVENT_ROLLUP_VERSION,
                    "extract_version": LEDGER_EXTRACT_VERSION,
                    "facts": facts,
                    "usage": event_usage,
                    "candidate_source_count": len(event_payload["source_messages"]),
                    "candidate_fact_count": len(event_payload["atomic_facts"]),
                }
                checkpoint = {
                    "schema_version": 2,
                    "sessions": completed,
                    "rollups": rollups,
                    "reasoning_rollups": reasoning_rollups,
                    "relation_rollups": relation_rollups,
                    "event_rollups": event_rollups,
                }
                _atomic_write_json(checkpoint_path, checkpoint)
            event = event_rollups[graph_key]
            event_facts.extend(event.get("facts") or [])
            if (
                int(event.get("postprocess_version", 0))
                != LEDGER_EVENT_POSTPROCESS_VERSION
            ):
                event["deterministic_facts"] = _deterministic_pair_event_facts(
                    event_payload, list(event.get("facts") or [])
                )
                event["postprocess_version"] = LEDGER_EVENT_POSTPROCESS_VERSION
                event_rollups[graph_key] = event
                checkpoint = {
                    "schema_version": 2,
                    "sessions": completed,
                    "rollups": rollups,
                    "reasoning_rollups": reasoning_rollups,
                    "relation_rollups": relation_rollups,
                    "event_rollups": event_rollups,
                }
                _atomic_write_json(checkpoint_path, checkpoint)
            event_facts.extend(event.get("deterministic_facts") or [])
            for token_key in usage:
                usage[token_key] += int(
                    (event.get("usage") or {}).get(token_key, 0)
                )
        all_facts = _attach_source_dates(
            _deduplicate_facts(
                atomic_facts + aggregate_facts + reasoning_facts
                + relation_facts + event_facts
            ),
            conversation,
        )
        conversation_rows[conversation.conversation_id] = {
            "speakers": actors,
            "actors": actors,
            "atomic_fact_count": len(atomic_facts),
            "rollup_fact_count": len(aggregate_facts),
            "reasoning_rollup_fact_count": len(reasoning_facts),
            "relation_rollup_fact_count": len(relation_facts),
            "event_rollup_fact_count": len(event_facts),
            "facts": all_facts,
        }

    result = {
        "schema_version": 2,
        "kind": "tencentdb-adapter-lossless-ledger",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
        "multimodal_policy": "dataset text plus supplied caption/query only",
        "base_checkpoint_policy": (
            "reuse-pinned-existing" if reuse_base_checkpoint else "version-validated"
        ),
        "conversation_count": len(conversations),
        "session_count": sum(len(value.sessions) for value in conversations),
        "usage": usage,
        "conversations": conversation_rows,
    }
    _atomic_write_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a source-grounded lossless fact ledger for TencentDB evaluation"
    )
    parser.add_argument("--conversation", action="append", required=True)
    parser.add_argument("--dataset", default="locomo_refined")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rollup", action="store_true")
    parser.add_argument(
        "--audited-rollup",
        action="store_true",
        help="add confidence-gated cross-actor and audited-event construction",
    )
    parser.add_argument(
        "--reuse-base-checkpoint",
        action="store_true",
        help=(
            "pin existing session/ordinary-rollup checkpoint entries for a "
            "controlled augmentation ablation; requires --resume"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.reuse_base_checkpoint and not args.resume:
        raise SystemExit("--reuse-base-checkpoint requires --resume")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing ledger: {args.output}")
    raw_config = _load_config_data(args.config.resolve())
    config = _resolve_env(raw_config.get("memory_model") or {})
    if args.model:
        config["model"] = args.model
    catalog = PluginCatalog(
        root=ROOT,
        framework_profiles=PROFILES,
        dataset_profiles=DATASETS,
    )
    dataset = catalog.create_dataset(args.dataset, root=args.dataset_root)
    conversations: list[Conversation] = []
    for conversation_id in args.conversation:
        try:
            conversations.append(dataset.conversation(conversation_id))
        except KeyError as exc:
            raise SystemExit(f"unknown conversation: {conversation_id}") from exc
    result = build_ledger(
        conversations,
        client=_LedgerClient(
            config, max_tokens=args.max_tokens, retries=args.retries
        ),
        output=args.output.resolve(),
        concurrency=args.concurrency,
        resume=args.resume,
        rollup=not args.skip_rollup,
        audited_rollup=args.audited_rollup,
        reuse_base_checkpoint=args.reuse_base_checkpoint,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "conversation_count": result["conversation_count"],
        "session_count": result["session_count"],
        "fact_count": sum(
            len(row["facts"])
            for row in result["conversations"].values()
        ),
        "usage": result["usage"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
