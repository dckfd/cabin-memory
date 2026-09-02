from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .cockpit_episode import (
    EpisodeTurn,
    compile_navigation_episode,
    episode_from_dict,
)
from .temporal import resolve_temporal_query
from .structured_state import resolve_state_answer


_REPLY_INTENT_PATTERNS = (
    re.compile(
        r"\bwhat did (?:the )?(?:driver|user|passenger) repl(?:y|ied)\b"
        r".*\b(?:assistant|system) asked\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:司机|驾驶员|用户|乘客).*(?:回答|回复).*"
        r"(?:助手|车机|系统).*(?:问|询问|追问)"
    ),
)
_ANCHOR_PATTERNS = (
    re.compile(r'\bthat began\s+["“]([^"”]+)["”]', re.IGNORECASE),
    re.compile(r'(?:以|从)["“]([^"”]+)["”](?:开始|开头)'),
)
_SLOT_PATTERNS = (
    re.compile(
        r'asked for the\s+["“]([^"”]+)["”]\s+detail', re.IGNORECASE
    ),
    re.compile(r'(?:询问|追问).*?["“]([^"”]+)["”].*?(?:信息|槽位|内容)'),
)
_TIME_SLOT_QUESTION_PATTERN = re.compile(
    r'\bwhat time did the\s+'
    r'(driver|user|passenger|car assistant|assistant|system)\s+'
    r'(?:specify|report)\s+for the\s+["“]([^"”]+)["”]\s+'
    r'(?:reminder|event)\b',
    re.IGNORECASE,
)
_WEATHER_LOCATION_QUESTION_PATTERN = re.compile(
    r'\bwhich location did the\s+(driver|user|passenger)\s+'
    r'specify\s+for the forecast on\s+["“]([^"”]+)["”]',
    re.IGNORECASE,
)
_DRIVER_DESTINATION_QUESTION_PATTERN = re.compile(
    r'\bwhat destination did the\s+(driver|user|passenger)\s+'
    r'mention\s+in\s+(?:their|the)\s+["“]([^"”]+)["”]\s+request',
    re.IGNORECASE,
)
_ASSISTANT_DESTINATION_QUESTION_PATTERN = re.compile(
    r'\bwhich destination did the\s+'
    r'(car assistant|assistant|system)\s+select\s+for\s+the\s+'
    r'(?:driver|user|passenger)(?:\'s|’s)\s+["“]([^"”]+)["”]\s+request',
    re.IGNORECASE,
)
_ZH_DESTINATION_QUESTION_PATTERN = re.compile(
    r"(?:(?:我|司机|驾驶员|用户|乘客|车机|助手).{0,16})?"
    r"(?:导航(?:到|去)?|目的地|路线终点).{0,16}"
    r"(?:哪(?:里|儿)|什么地方|哪个(?:地方|地点|目的地)|去哪(?:里|儿)?)"
)
_QUOTED_ANCHOR_PATTERN = re.compile(r'["“]([^"”]+)["”]')
_CLOCK_TIME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])((?:an?\s+)?(?:half\s+)?(?:hour\s+before\s+)?"
    r"(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s*"
    r"(?:a\.?\s*m\.?|p\.?\s*m\.?))(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_BARE_CLOCK_TIME_PATTERN = re.compile(
    r"\bat\s+(1[0-2]|[1-9])\b(?!\s*(?:a\.?\s*m\.?|p\.?\s*m\.?))",
    re.IGNORECASE,
)
_WEATHER_LOCATION_END = (
    r"(?=\s+(?:today|tomorrow|tonight|now|right\s+now|"
    r"this\s+(?:week|weekend|sat|sun)|next\s+(?:week|few|\d|48)|"
    r"on\s+(?:mon|tue|wed|thu|fri|sat|sun)|"
    r"during\s+(?:the\s+)?(?:week|next)|"
    r"over\s+(?:the\s+)?(?:week|weekend)|anytime|going\s+to|"
    r"will\s+be|is\s+going|in\s+next)|[,?.!]|$)"
)
_WEATHER_LOCATION_PATTERNS = (
    re.compile(
        r"\b(?:in|for)\s+([a-z][a-z .'-]{1,40}?)"
        + _WEATHER_LOCATION_END,
        re.IGNORECASE,
    ),
    re.compile(
        r"\bif\s+([a-z][a-z .'-]{1,40}?)\s+(?:is|will|has)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:about|asking about)\s+([a-z][a-z .'-]{1,40}?)"
        + _WEATHER_LOCATION_END,
        re.IGNORECASE,
    ),
)
_POLITE_UTTERANCE_PATTERN = re.compile(
    r"^\s*(?:thanks?|thank\s+you|okay|ok|great|perfect|awesome|"
    r"sounds\s+great|alright|bye|no\s+problem)\b",
    re.IGNORECASE,
)
_NON_LOCATION_TERM_PATTERN = re.compile(
    r"\b(?:weather|forecast|rain|raining|snow|snowing|frost|cloudy|"
    r"overcast|hot|warm|cold|humid|temperature|blizzard|misty|foggy|"
    r"dew|drizzle|hail|stormy|windy|skies|outside|week|weekend|today|"
    r"tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday|day|days|information|all|whole)\b",
    re.IGNORECASE,
)
_NON_LOCATION_VALUES = frozenset({
    "it", "there", "the", "you", "this", "that", "now", "today",
    "tomorrow", "great", "a bunch", "the information",
    "for the information",
})
_MESSAGE_PATTERN = re.compile(
    r"^\[([^\]]+)\]\s+\[source_time=([^\]]+)\]\s+([^:]+):\s*(.*)$"
)
_MESSAGE_WITH_ID_PATTERN = re.compile(
    r"^\[([^\]]+)\]\s+([^:]+):\s*(.*)$"
)
_MESSAGE_PLAIN_PATTERN = re.compile(r"^([^:\[\]]+):\s*(.*)$")
_HIT_HEADER_PATTERN = re.compile(r"^\[(\d+)(?:\s+[^\]]*)?\]$")
_HEADER_TIME_PATTERN = re.compile(r"(?:^|\s)time=([^\s\]]+)")
_HEADER_SOURCES_PATTERN = re.compile(r"(?:^|\s)sources=([^\s\]]+)")


@dataclass(frozen=True)
class SlotAnswerCandidate:
    """A lossless answer read from one grounded clarification exchange."""

    value: str
    source_ids: tuple[str, ...]
    slot_label: str
    command_anchor: str
    confidence: float = 1.0
    reason: str = "anchored_clarification_reply"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ContextMessage:
    message_id: str
    session_key: str
    timestamp: str
    sequence: int
    speaker: str
    text: str


def extract_cockpit_answer(
    question: str,
    context: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    default_timezone: str = "UTC",
    retrieval_hits: Sequence[Mapping[str, Any]] = (),
) -> SlotAnswerCandidate | None:
    """Compile a high-confidence cockpit answer before model fallback.

    The compiler covers three common edge-device shapes: a clarification
    reply, a calendar clock-time slot, and a weather location slot. It uses
    only the question, retrieved context, and trusted request-time metadata.
    Gold answers and benchmark evidence IDs are never inputs.
    """
    state = resolve_state_answer(question, context, retrieval_hits, metadata)
    if state is not None:
        return SlotAnswerCandidate(
            value=str(state["value"]),
            source_ids=tuple(str(item) for item in state.get("source_ids", ())),
            slot_label=str(state.get("slot_label", "state")),
            command_anchor="structured-state",
            confidence=float(state.get("confidence", 0.9)),
            reason=str(state.get("reason", "structured_state_resolution")),
        )
    clarification = extract_clarification_reply(
        question,
        context,
        metadata,
        default_timezone=default_timezone,
    )
    if clarification is not None:
        return clarification
    calendar_time = _extract_calendar_time(
        question,
        context,
        metadata,
        default_timezone=default_timezone,
    )
    if calendar_time is not None:
        return calendar_time
    weather = _extract_weather_location(
        question,
        context,
        metadata,
        default_timezone=default_timezone,
    )
    if weather is not None:
        return weather
    return _extract_navigation_destination(
        question,
        context,
        metadata,
        default_timezone=default_timezone,
        retrieval_hits=retrieval_hits,
    )


def extract_clarification_reply(
    question: str,
    context: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    default_timezone: str = "UTC",
) -> SlotAnswerCandidate | None:
    """Read an explicit user reply without consulting gold or evidence IDs.

    The fast path deliberately requires four independent signals: a reply
    question intent, a quoted command anchor, a trusted date constraint, and
    an adjacent user -> assistant clarification -> user chain in one source
    session. Any ambiguity returns ``None`` so the normal answer model remains
    the fallback.
    """
    value = str(question or "")
    if not any(pattern.search(value) for pattern in _REPLY_INTENT_PATTERNS):
        return None
    command_anchor = _first_group(_ANCHOR_PATTERNS, value)
    slot_label = _first_group(_SLOT_PATTERNS, value)
    if not command_anchor or not slot_label:
        return None

    temporal = resolve_temporal_query(
        value, metadata or {}, default_timezone=default_timezone
    )
    target_dates = {
        span.start.date().isoformat()
        for span in temporal.spans
        if span.granularity in {
            "day", "morning", "noon", "afternoon", "evening", "night",
            "early_morning",
        }
    }
    if not target_dates:
        return None

    messages = _parse_context_messages(context)
    by_session: dict[str, list[_ContextMessage]] = {}
    for message in messages:
        if not any(message.timestamp.startswith(date) for date in target_dates):
            continue
        by_session.setdefault(message.session_key, []).append(message)

    anchor_key = _lexical_key(command_anchor)
    slot_key = _lexical_key(slot_label)
    candidates: list[SlotAnswerCandidate] = []
    for session_messages in by_session.values():
        ordered = sorted(
            session_messages,
            key=lambda item: (item.timestamp, item.sequence, item.message_id),
        )
        for index in range(len(ordered) - 2):
            command, clarification, reply = ordered[index:index + 3]
            if not _is_user_speaker(command.speaker):
                continue
            if not _lexical_key(command.text).startswith(anchor_key):
                continue
            if not _is_assistant_speaker(clarification.speaker):
                continue
            if slot_key not in _lexical_key(clarification.text):
                continue
            if not _is_user_speaker(reply.speaker) or not reply.text.strip():
                continue
            candidates.append(SlotAnswerCandidate(
                value=reply.text.strip(),
                source_ids=(
                    command.message_id,
                    clarification.message_id,
                    reply.message_id,
                ),
                slot_label=slot_label,
                command_anchor=command_anchor,
            ))

    # Multiple independently matching exchanges are not safe to resolve with
    # rules; leave them to the answer model instead of choosing by accident.
    unique = {
        (candidate.value, candidate.source_ids): candidate
        for candidate in candidates
    }
    return next(iter(unique.values())) if len(unique) == 1 else None


def _extract_calendar_time(
    question: str,
    context: str,
    metadata: Mapping[str, Any] | None,
    *,
    default_timezone: str,
) -> SlotAnswerCandidate | None:
    match = _TIME_SLOT_QUESTION_PATTERN.search(str(question or ""))
    if not match:
        return None
    target_speaker, event_anchor = match.groups()
    sessions = _target_sessions(
        question,
        context,
        metadata,
        default_timezone=default_timezone,
    )
    anchor_key = _lexical_key(event_anchor)
    wants_user = _is_user_speaker(target_speaker)
    candidates: dict[str, list[tuple[str, _ContextMessage, bool]]] = {}
    bare_candidates: dict[str, list[tuple[str, _ContextMessage, bool]]] = {}
    all_session_messages: list[_ContextMessage] = []
    for messages in sessions.values():
        if not any(anchor_key in _lexical_key(item.text) for item in messages):
            continue
        all_session_messages.extend(messages)
        for message in messages:
            is_target = (
                _is_user_speaker(message.speaker) if wants_user
                else _is_assistant_speaker(message.speaker)
            )
            if not is_target:
                continue
            anchored = anchor_key in _lexical_key(message.text)
            for clock_match in _CLOCK_TIME_PATTERN.finditer(message.text):
                raw = _clean_clock_value(clock_match.group(1))
                candidates.setdefault(_clock_key(raw), []).append(
                    (raw, message, anchored)
                )
            for bare_match in _BARE_CLOCK_TIME_PATTERN.finditer(message.text):
                raw = bare_match.group(1)
                bare_candidates.setdefault(raw, []).append(
                    (raw, message, anchored)
                )

    chosen = _choose_unique_anchored_candidate(candidates)
    if chosen is None and not candidates and len(bare_candidates) == 1:
        # A bare hour such as "at 3" is accepted only when both dialogue
        # roles repeat it, preventing dates and distances from masquerading
        # as clock slots.
        key = next(iter(bare_candidates))
        roles = {
            "user" if _is_user_speaker(message.speaker) else "assistant"
            for message in all_session_messages
            if _BARE_CLOCK_TIME_PATTERN.search(message.text)
            and any(
                item.group(1) == key
                for item in _BARE_CLOCK_TIME_PATTERN.finditer(message.text)
            )
        }
        if roles == {"user", "assistant"}:
            chosen = bare_candidates[key][-1]
    if chosen is None:
        return None
    value, message, _anchored = chosen
    return SlotAnswerCandidate(
        value=value,
        source_ids=(message.message_id,),
        slot_label="time",
        command_anchor=event_anchor,
        reason="grounded_calendar_time_slot",
    )


def _extract_weather_location(
    question: str,
    context: str,
    metadata: Mapping[str, Any] | None,
    *,
    default_timezone: str,
) -> SlotAnswerCandidate | None:
    match = _WEATHER_LOCATION_QUESTION_PATTERN.search(str(question or ""))
    if not match:
        return None
    _speaker, forecast_anchor = match.groups()
    sessions = _target_sessions(
        question,
        context,
        metadata,
        default_timezone=default_timezone,
    )
    ranked_candidates: list[
        tuple[int, int, int, str, _ContextMessage]
    ] = []
    for messages in sessions.values():
        candidate_rows: dict[str, list[tuple[str, _ContextMessage]]] = {}
        assistant_key = _lexical_key(" ".join(
            item.text for item in messages
            if _is_assistant_speaker(item.speaker)
        ))
        for message in messages:
            if not _is_user_speaker(message.speaker):
                continue
            for location in _weather_location_candidates(message.text):
                key = _lexical_key(location)
                if key:
                    candidate_rows.setdefault(key, []).append(
                        (location, message)
                    )
        for key, rows in candidate_rows.items():
            location, message = rows[-1]
            ranked_candidates.append((
                len(rows),
                int(key in assistant_key),
                message.sequence,
                location,
                message,
            ))
    if not ranked_candidates:
        return None
    ranked_candidates.sort(reverse=True, key=lambda item: item[:3])
    if (
        len(ranked_candidates) > 1
        and ranked_candidates[0][:3] == ranked_candidates[1][:3]
    ):
        return None
    _mentions, _echoed, _sequence, value, message = ranked_candidates[0]
    return SlotAnswerCandidate(
        value=value,
        source_ids=(message.message_id,),
        slot_label="location",
        command_anchor=forecast_anchor,
        reason="grounded_weather_location_slot",
    )


def _extract_navigation_destination(
    question: str,
    context: str,
    metadata: Mapping[str, Any] | None,
    *,
    default_timezone: str,
    retrieval_hits: Sequence[Mapping[str, Any]] = (),
) -> SlotAnswerCandidate | None:
    value = str(question or "")
    driver_match = _DRIVER_DESTINATION_QUESTION_PATTERN.search(value)
    assistant_match = _ASSISTANT_DESTINATION_QUESTION_PATTERN.search(value)
    chinese_match = _ZH_DESTINATION_QUESTION_PATTERN.search(value)
    if not driver_match and not assistant_match and not chinese_match:
        return None
    quoted = _QUOTED_ANCHOR_PATTERN.search(value)
    anchor = (
        (driver_match or assistant_match).group(2)
        if driver_match or assistant_match
        else quoted.group(1) if quoted else ""
    )
    driver_question = bool(
        driver_match
        or (
            chinese_match
            and re.search(r"(?:我|司机|驾驶员|用户|乘客)", value)
        )
    )
    typed = _typed_navigation_candidate(
        value,
        anchor,
        driver_question=driver_question,
        metadata=metadata,
        retrieval_hits=retrieval_hits,
        default_timezone=default_timezone,
    )
    if typed is not None:
        return typed
    sessions = _target_sessions(
        value,
        context,
        metadata,
        default_timezone=default_timezone,
    )
    candidates = []
    for messages in sessions.values():
        if anchor and not _navigation_anchor_matches(anchor, messages):
            continue
        episode = compile_navigation_episode([
            EpisodeTurn(
                message_id=message.message_id,
                speaker=message.speaker,
                text=message.text,
                timestamp=message.timestamp,
                sequence=message.sequence,
            )
            for message in messages
        ], domain="navigation")
        if episode is None or episode.confidence < 0.97:
            continue
        if driver_question and not any(
            transition.actor == "user"
            and transition.action in {"select", "structured_slot"}
            for transition in episode.transitions
        ):
            continue
        candidates.append(episode)

    unique = {
        (
            _lexical_key(episode.destination),
            _lexical_key(episode.address),
            episode.source_ids,
        ): episode
        for episode in candidates
    }
    if len(unique) != 1:
        return None
    episode = next(iter(unique.values()))
    return SlotAnswerCandidate(
        value=episode.destination,
        source_ids=episode.source_ids,
        slot_label="destination",
        command_anchor=anchor or "navigation",
        confidence=episode.confidence,
        reason="grounded_navigation_state",
    )


def _typed_navigation_candidate(
    question: str,
    anchor: str,
    *,
    driver_question: bool,
    metadata: Mapping[str, Any] | None,
    retrieval_hits: Sequence[Mapping[str, Any]],
    default_timezone: str,
) -> SlotAnswerCandidate | None:
    temporal = resolve_temporal_query(
        question, metadata or {}, default_timezone=default_timezone
    )
    target_dates = {
        span.start.date().isoformat()
        for span in temporal.spans
        if span.granularity in {
            "day", "morning", "noon", "afternoon", "evening", "night",
            "early_morning",
        }
    }
    if not target_dates:
        return None
    candidates = {}
    for hit in retrieval_hits:
        hit_metadata = hit.get("metadata") or {}
        if not isinstance(hit_metadata, Mapping):
            continue
        raw_episode = hit_metadata.get("typed_cockpit_episode")
        if not isinstance(raw_episode, Mapping):
            continue
        episode = episode_from_dict(raw_episode)
        if (
            episode is None
            or episode.confidence < 0.97
            or episode.state not in {"selected", "confirmed"}
        ):
            continue
        if not any(episode.mentioned_at.startswith(date) for date in target_dates):
            continue
        hit_source_ids = {
            str(item) for item in hit.get("source_ids") or [] if str(item)
        }
        if (
            not episode.source_ids
            or not hit_source_ids
            or not set(episode.source_ids).issubset(hit_source_ids)
        ):
            continue
        if anchor and not _navigation_anchor_text_matches(
            anchor,
            " ".join((
                episode.request_text,
                episode.destination,
                *episode.aliases,
            )),
        ):
            continue
        if driver_question and not any(
            transition.actor == "user"
            and transition.action in {"select", "structured_slot"}
            for transition in episode.transitions
        ):
            continue
        candidates[(
            _lexical_key(episode.destination),
            _lexical_key(episode.address),
            episode.source_ids,
        )] = episode
    if len(candidates) != 1:
        return None
    episode = next(iter(candidates.values()))
    return SlotAnswerCandidate(
        value=episode.destination,
        source_ids=episode.source_ids,
        slot_label="destination",
        command_anchor=anchor or "navigation",
        confidence=episode.confidence,
        reason="grounded_typed_navigation_episode",
    )


def _navigation_anchor_matches(
    anchor: str, messages: list[_ContextMessage]
) -> bool:
    return _navigation_anchor_text_matches(
        anchor, " ".join(message.text for message in messages)
    )


def _navigation_anchor_text_matches(anchor: str, text: str) -> bool:
    anchor_key = _lexical_key(anchor)
    session_key = _lexical_key(text)
    if not anchor_key or not session_key:
        return False
    if anchor_key in session_key:
        return True
    possessive_anchor = _possessive_place_key(anchor_key)
    possessive_session = _possessive_place_key(session_key)
    if possessive_anchor and possessive_anchor in possessive_session:
        return True
    aliases = {
        "shopping center": ("shopping mall", "mall", "shopping"),
        "shopping mall": ("shopping center", "mall", "shopping"),
        "coffee shop": ("coffee", "cafe", "tea house"),
        "parking lot": ("parking garage", "parking"),
        "parking garage": ("parking lot", "parking"),
        "rest stop": ("hotel", "lodge", "rest"),
        "chinese restaurant": ("chinese food", "chinese"),
        "coffee": ("coffee shop", "cafe"),
        "grocery store": ("groceries", "grocery", "supermarket"),
        "parking garage": ("parking lot", "parking", "park"),
        "机场": ("航站楼", "航空港"),
        "充电站": ("充电桩", "充电"),
        "公司": ("单位", "办公室", "上班的地方"),
        "家": ("我家", "家里", "回家"),
    }
    folded = anchor_key.rstrip("s")
    if folded and folded in session_key:
        return True
    return any(
        _lexical_key(alias) in session_key
        for source, values in aliases.items()
        if source in anchor_key
        for alias in values
    )


def _possessive_place_key(value: str) -> str:
    # ASR and text normalization commonly alternate among ``friend's house``,
    # ``friends house`` and ``friend s house``.  Normalize only before a place
    # noun; ordinary POI plurals remain untouched.
    place = r"(?:house|home|office|work|place|address)"
    result = re.sub(
        rf"\b([a-z0-9]+)\s+s\s+(?={place}\b)", r"\1 ", value
    )
    return re.sub(
        rf"\b([a-z0-9]{{3,}})s\s+(?={place}\b)", r"\1 ", result
    )


def _first_group(patterns: tuple[re.Pattern[str], ...], value: str) -> str:
    for pattern in patterns:
        match = pattern.search(value)
        if match:
            return " ".join(match.group(1).split())
    return ""


def _target_sessions(
    question: str,
    context: str,
    metadata: Mapping[str, Any] | None,
    *,
    default_timezone: str,
) -> dict[str, list[_ContextMessage]]:
    temporal = resolve_temporal_query(
        str(question or ""),
        metadata or {},
        default_timezone=default_timezone,
    )
    target_dates = {
        span.start.date().isoformat()
        for span in temporal.spans
        if span.granularity in {
            "day", "morning", "noon", "afternoon", "evening", "night",
            "early_morning",
        }
    }
    if not target_dates:
        return {}
    sessions: dict[str, list[_ContextMessage]] = {}
    for message in _parse_context_messages(context):
        if any(message.timestamp.startswith(date) for date in target_dates):
            sessions.setdefault(message.session_key, []).append(message)
    return sessions


def _clean_clock_value(value: str) -> str:
    result = " ".join(str(value or "").strip(" ,.!?").split())
    result = re.sub(r"^an?\s+", "", result, flags=re.IGNORECASE)
    return result


def _clock_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(re.findall(r"[a-z0-9]+", normalized))


def _choose_unique_anchored_candidate(
    candidates: dict[str, list[tuple[str, _ContextMessage, bool]]],
) -> tuple[str, _ContextMessage, bool] | None:
    if len(candidates) == 1:
        return next(iter(candidates.values()))[-1]
    anchored = [
        rows for rows in candidates.values()
        if any(item[2] for item in rows)
    ]
    if len(anchored) != 1:
        return None
    anchored_rows = [item for item in anchored[0] if item[2]]
    return anchored_rows[-1] if anchored_rows else anchored[0][-1]


def _clean_weather_location(value: str) -> str:
    result = str(value or "").strip(" ,.!?")
    result = re.sub(
        r"^(?:i(?: am|'m)? (?:currently )?in\s+)",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"^(?:for|in)\s+", "", result, flags=re.IGNORECASE)
    result = re.sub(
        r"\b(?:please|thank\s+you|thanks?|car)\b",
        " ",
        result,
        flags=re.IGNORECASE,
    )
    result = " ".join(result.strip(" ,.!?").split())
    result = re.sub(
        r"\s+(?:right\s+now|now|today|tomorrow|"
        r"this\s+(?:week|weekend)|next\s+\d+\s+days?)$",
        "",
        result,
        flags=re.IGNORECASE,
    )
    return " ".join(result.strip(" ,.!?").split())


def _weather_location_candidates(value: str) -> tuple[str, ...]:
    text = str(value or "")
    if _POLITE_UTTERANCE_PATTERN.search(text):
        return ()
    raw_candidates: list[str] = []
    for pattern in _WEATHER_LOCATION_PATTERNS:
        raw_candidates.extend(
            match.group(1) for match in pattern.finditer(text)
        )
    first_fragment = re.split(r"[,?.!]", text, maxsplit=1)[0]
    raw_candidates.extend((first_fragment, text))

    result: list[str] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        candidate = _clean_weather_location(raw)
        key = _lexical_key(candidate)
        if (
            not key
            or key in seen
            or candidate.casefold() in _NON_LOCATION_VALUES
            or not 1 <= len(candidate.split()) <= 4
            or not 3 <= len(candidate) <= 40
            or _NON_LOCATION_TERM_PATTERN.search(candidate)
        ):
            continue
        result.append(candidate)
        seen.add(key)
    return tuple(result)


def _lexical_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(
        r"[a-z0-9]+|[\u3400-\u9fff]+", normalized
    ))


def _is_user_speaker(value: str) -> bool:
    key = _lexical_key(value)
    return key in {"driver", "user", "passenger", "司机", "驾驶员", "用户", "乘客"}


def _is_assistant_speaker(value: str) -> bool:
    key = _lexical_key(value)
    return key in {
        "assistant", "car assistant", "system", "助手", "车机", "系统",
    }


def _source_session_key(message_id: str) -> str:
    key = re.sub(r"T\d+$", "", str(message_id), flags=re.IGNORECASE)
    return key or str(message_id)


def _parse_context_messages(context: str) -> tuple[_ContextMessage, ...]:
    messages: dict[str, _ContextMessage] = {}
    header_time = ""
    header_sources: list[str] = []
    sequence = 0
    for raw_line in str(context or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HIT_HEADER_PATTERN.match(line):
            time_match = _HEADER_TIME_PATTERN.search(line)
            source_match = _HEADER_SOURCES_PATTERN.search(line)
            header_time = time_match.group(1) if time_match else ""
            header_sources = (
                source_match.group(1).split(",") if source_match else []
            )
            continue

        parsed = _MESSAGE_PATTERN.match(line)
        if parsed:
            message_id, timestamp, speaker, text = parsed.groups()
        else:
            with_id = _MESSAGE_WITH_ID_PATTERN.match(line)
            if with_id:
                message_id, speaker, text = with_id.groups()
                timestamp = header_time
            else:
                plain = _MESSAGE_PLAIN_PATTERN.match(line)
                if not plain:
                    continue
                speaker, text = plain.groups()
                message_id = next(
                    (
                        source for source in header_sources
                        if re.search(r"T\d+$", source, re.IGNORECASE)
                        and source not in messages
                    ),
                    f"context-line-{sequence}",
                )
                timestamp = header_time
        messages.setdefault(message_id, _ContextMessage(
            message_id=message_id,
            session_key=_source_session_key(message_id),
            timestamp=timestamp,
            sequence=sequence,
            speaker=speaker.strip(),
            text=text.strip(),
        ))
        sequence += 1
    return tuple(messages.values())
