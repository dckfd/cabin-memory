from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did",
    "do", "does", "during", "for", "from", "had", "has", "have", "how",
    "in", "interaction", "is", "it", "logged", "of", "on", "or", "that",
    "the", "their", "them", "they", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "would", "you", "your",
    # Common benchmark/cockpit question framing. Removing it makes coverage
    # measure the named action, entity, date, and slot rather than boilerplate.
    "asked", "assistant", "began", "car", "detail", "driver", "interaction",
    "reply", "said", "voice",
})
_ALIASES = {
    "children": "child", "kids": "child", "pets": "pet",
    "cats": "cat", "dogs": "dog", "destinations": "destination",
    "locations": "location", "preferences": "preference",
}
_COMPLEX = re.compile(
    r"\b(?:all|both|compare|every|how many|over time|timeline|what changed|"
    r"latest|most recent|before|after|first|last|currently|now|instead|"
    r"corrected|updated)\b|"
    r"(?:全部|所有|两者|比较|多少次|变化|最近|最新|之前|之后|最早|最后|"
    r"当前|现在|改成|纠正)",
    re.IGNORECASE,
)
_QUOTED = re.compile(
    r'"([^"\n]{2,})"|“([^”\n]{2,})”|'
    r"(?<![A-Za-z0-9])'([^'\n]{2,})'(?![A-Za-z0-9])"
)
_ISO_DATE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_CLOCK_TIME = re.compile(
    r"\b(?:(?:[01]?\d|2[0-3]):[0-5]\d|(?:1[0-2]|0?[1-9])\s*(?:a\.?m\.?|p\.?m\.?))\b",
    re.IGNORECASE,
)
_ZH_DATE = re.compile(
    r"(?:20\d{2}年)?(?:1[0-2]|0?[1-9])月(?:3[01]|[12]?\d)日"
)
_REPLY_QUERY = re.compile(
    r"\b(?:reply|replied|respond|responded|answer|answered|said next)\b|"
    r"(?:回复|回答|接着说了什么|随后说了什么)",
    re.IGNORECASE,
)


def _tokens(value: str) -> list[str]:
    result = []
    for raw in _TOKEN.findall(str(value or "")):
        token = raw.casefold()
        if token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 3 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        result.append(_ALIASES.get(token, token))
    return result


def _normalized(value: str) -> str:
    return "".join(_TOKEN.findall(str(value or "").casefold()))


def _quoted_values(value: str) -> tuple[str, ...]:
    return tuple(
        next(group for group in match.groups() if group is not None)
        for match in _QUOTED.finditer(str(value or ""))
    )


def _complex_query_text(value: str) -> str:
    """Ignore bounded time phrases and quoted slot values.

    A quoted ``"now"`` in a weather-memory question is the user's original
    forecast-period slot, not a request for the latest mutable memory state.
    Unquoted latest/current/correction operators remain complexity signals.
    """
    text = str(value or "")
    if "[normalized_query_time:" in text:
        text = re.sub(r"\[normalized_query_time:[^\]]*\]", "", text)
        text = re.sub(
            r"\b(?:the\s+)?day\s+before\s+yesterday\b|"
            r"\blast\s+(?:monday|tuesday|wednesday|thursday|friday|"
            r"saturday|sunday)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
    return _QUOTED.sub(" ", text)


@dataclass(frozen=True)
class AdaptiveDecision:
    route: str
    fallback: bool
    reason: str
    lexical_coverage: float
    score_margin: float
    quoted_anchor_match: bool
    complex_query: bool
    critical_slot_coverage: float = 1.0
    missing_critical_slots: tuple[str, ...] = ()
    dialogue_complete: bool = True

    def metadata(self) -> dict:
        return asdict(self)


def decide_l0_fast_path(
    query: str,
    content: str,
    *,
    top_score: float | None = None,
    second_score: float | None = None,
    min_coverage: float = 0.45,
    min_score_margin: float = 0.15,
) -> AdaptiveDecision:
    """Choose Top-1 or fallback without a model call or benchmark labels."""
    query_tokens = set(_tokens(query))
    content_tokens = set(_tokens(content))
    coverage = (
        len(query_tokens & content_tokens) / len(query_tokens)
        if query_tokens else 0.0
    )
    if top_score is None or second_score is None:
        margin = 0.0
    else:
        margin = max(0.0, (top_score - second_score) / max(abs(top_score), 1e-9))
    normalized_content = _normalized(content)
    anchors = [_normalized(value) for value in _quoted_values(query)]
    anchor_match = any(
        len(anchor) >= 3 and anchor in normalized_content for anchor in anchors
    )
    complex_query = bool(_COMPLEX.search(_complex_query_text(query)))

    if not content.strip():
        route, reason = "fallback", "empty_l0"
    elif complex_query:
        route, reason = "fallback", "complex_or_update_query"
    elif coverage >= min_coverage:
        route, reason = "fast", "high_lexical_coverage"
    elif anchor_match and coverage >= min_coverage / 2:
        route, reason = "fast", "quoted_anchor_match"
    elif coverage >= min_coverage / 2 and margin >= min_score_margin:
        route, reason = "fast", "moderate_coverage_with_margin"
    else:
        route, reason = "fallback", "low_confidence"
    return AdaptiveDecision(
        route=route,
        fallback=route == "fallback",
        reason=reason,
        lexical_coverage=round(coverage, 6),
        score_margin=round(margin, 6),
        quoted_anchor_match=anchor_match,
        complex_query=complex_query,
    )


@dataclass(frozen=True)
class CandidateRelevance:
    """Deterministic query/candidate signals used before adaptive routing."""

    score: float
    lexical_coverage: float
    critical_slot_coverage: float
    matched_critical_slots: tuple[str, ...]
    missing_critical_slots: tuple[str, ...]
    backend_rank: int

    def metadata(self) -> dict:
        return asdict(self)


def _critical_slots(value: str) -> tuple[str, ...]:
    slots: list[str] = []
    for raw in _quoted_values(value):
        normalized = _normalized(raw)
        if len(normalized) >= 3:
            slots.append(f"quote:{normalized}")
    for pattern, prefix in (
        (_ISO_DATE, "date"), (_CLOCK_TIME, "time"), (_ZH_DATE, "date")
    ):
        for raw in pattern.findall(str(value or "")):
            normalized = _normalized(raw)
            if normalized:
                slots.append(f"{prefix}:{normalized}")
    return tuple(dict.fromkeys(slots))


def _slot_matches(slots: tuple[str, ...], content: str) -> tuple[str, ...]:
    normalized_content = _normalized(content)
    return tuple(
        slot for slot in slots
        if slot.split(":", 1)[-1] in normalized_content
    )


def score_l0_candidate(
    query: str,
    content: str,
    *,
    backend_rank: int,
) -> CandidateRelevance:
    """Score one broad L0 candidate without another model invocation.

    Backend rank remains the primary signal when the query has no exact slots.
    Quoted utterance anchors and explicit dates/times can promote a lower dense
    result when it is the only candidate satisfying the user's hard constraint.
    """
    query_tokens = set(_tokens(query))
    content_tokens = set(_tokens(content))
    lexical = (
        len(query_tokens & content_tokens) / len(query_tokens)
        if query_tokens else 0.0
    )
    slots = _critical_slots(query)
    matched = _slot_matches(slots, content)
    missing = tuple(slot for slot in slots if slot not in set(matched))
    slot_coverage = len(matched) / len(slots) if slots else 1.0
    reciprocal_rank = 1.0 / (max(0, int(backend_rank)) + 1.0)
    if slots:
        score = 0.35 * reciprocal_rank + 0.25 * lexical + 0.40 * slot_coverage
    else:
        score = 0.70 * reciprocal_rank + 0.30 * lexical
    return CandidateRelevance(
        score=round(score, 8),
        lexical_coverage=round(lexical, 6),
        critical_slot_coverage=round(slot_coverage, 6),
        matched_critical_slots=matched,
        missing_critical_slots=missing,
        backend_rank=max(0, int(backend_rank)),
    )


def decide_l0_fast_path_v2(
    query: str,
    content: str,
    *,
    top_score: float | None = None,
    second_score: float | None = None,
    min_coverage: float = 0.45,
    min_score_margin: float = 0.15,
) -> AdaptiveDecision:
    """V2 fast-path gate with exact-slot and dialogue-completeness checks."""
    base = decide_l0_fast_path(
        query,
        content,
        top_score=top_score,
        second_score=second_score,
        min_coverage=min_coverage,
        min_score_margin=min_score_margin,
    )
    slots = _critical_slots(query)
    matched = _slot_matches(slots, content)
    missing = tuple(slot for slot in slots if slot not in set(matched))
    slot_coverage = len(matched) / len(slots) if slots else 1.0

    dialogue_complete = True
    if _REPLY_QUERY.search(str(query or "")):
        dialogue_lines = [
            line for line in str(content or "").splitlines()
            if ":" in line and not line.lstrip().startswith("[memory_episode ")
        ]
        dialogue_complete = len(dialogue_lines) >= 2 and not str(
            content or ""
        ).rstrip().endswith(("?", "？"))

    route, reason = base.route, base.reason
    if route == "fast" and missing:
        route, reason = "fallback", "missing_critical_slots"
    elif route == "fast" and not dialogue_complete:
        route, reason = "fallback", "incomplete_dialogue_pair"
    return AdaptiveDecision(
        route=route,
        fallback=route == "fallback",
        reason=reason,
        lexical_coverage=base.lexical_coverage,
        score_margin=base.score_margin,
        quoted_anchor_match=base.quoted_anchor_match,
        complex_query=base.complex_query,
        critical_slot_coverage=round(slot_coverage, 6),
        missing_critical_slots=missing,
        dialogue_complete=dialogue_complete,
    )
