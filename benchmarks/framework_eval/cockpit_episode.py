from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence


_ADDRESS_PATTERN = re.compile(
    r"\b(\d{1,6}(?!\s*(?:miles?|mi\.?|kilomet(?:er|re)s?|km)\b)\s+"
    r"(?:[A-Za-z][A-Za-z.'-]*\s+){0,5}"
    r"(?:Ave(?:nue)?|Rd|Road|St(?:reet)?|Dr(?:ive)?|Way|Ct|Court|"
    r"Ln|Lane|Pl(?:ace)?|Walk|Blvd|Boulevard|Real))\b"
    r"(?!\s+(?:block|closure|construction|work)\b)",
    re.IGNORECASE,
)
_PROPER_NAME_PATTERN = re.compile(
    r"\b((?:The\s+)?[A-Z][A-Za-z0-9.'&-]*"
    r"(?:\s+(?:(?:and|of|the|my)\s+)?[A-Z][A-Za-z0-9.'&-]*){0,5}"
    r"(?:\s+(?:house|garage|hotel|cafe|coffee|market|restaurant))?)\b"
)
_NAVIGATION_ACTION = re.compile(
    r"\b(?:set(?:ting)?|plot(?:ting)?|display(?:ing)?|send(?:ing|s|t)?|"
    r"navigate|navigating|navigation|head(?:ing)?|tak(?:e|ing)|go(?:ing)?|"
    r"route|directions?|gps)\b|"
    r"(?:开始导航|正在导航|导航到|导航去|带我去|路线|目的地|地图规划)",
    re.IGNORECASE,
)
_ACTION_TARGET = re.compile(
    r"\b(?:set(?:ting)?|plot(?:ting)?|navigat(?:e|ing)|head(?:ing)?|"
    r"tak(?:e|ing)|go(?:ing)?|route|directions?|gps)"
    r"(?:\s+(?:the|a|an|your|our|fastest|quickest|shortest|best|only))*"
    r"(?:\s+(?:route|navigation|directions?|gps))*\s+(?:to|for|at)\s+"
    r"(.+?)(?=[,;.!?]|\s+(?:now|with|through|but|because|which|that|"
    r"and\s+(?:we|there|it))\b|$)",
    re.IGNORECASE,
)
_USER_SELECTION = re.compile(
    r"\b(?:let(?:'s| us)\s+go(?:\s+to)?|go\s+to|take\s+me\s+to|"
    r"i(?:'d| would)?\s+like(?:\s+to\s+(?:try|go\s+to))?|"
    r"i\s+(?:want|choose|pick|prefer)|"
    r"address\s+of|details?\s+for|"
    r"what\s+is\s+(?:the\s+)?address\s+of)\b",
    re.IGNORECASE,
)
_USER_CONFIRMATION = re.compile(
    r"\b(?:yes|sure|okay|ok|sounds?\s+(?:good|great)|that\s+will\s+work|"
    r"let(?:'s| us)\s+go|go\s+there|navigate\s+there|set\s+(?:the\s+)?gps|"
    r"set\s+(?:the\s+)?(?:route|navigation))\b|"
    r"(?:好的|好吧|可以|行|就这个|就去那里|开始导航|导航吧)",
    re.IGNORECASE,
)
_ASSISTANT_SELECTION = re.compile(
    r"\b(?:suggest|recommend|vote\s+for|pick(?:ed|ing)?|select(?:ed|ing)?|"
    r"closest|nearest|quickest|fastest|only\s+(?:one|route)|"
    r"route\s+to|way\s+to|en\s*route\s+to)\b",
    re.IGNORECASE,
)
_ASSISTANT_CONFIRMATION = re.compile(
    r"\b(?:setting|set|plotting|navigating|navigation\s+is\s+set|"
    r"gps\s+(?:is\s+)?set|heading|taking\s+you|sent\s+(?:the\s+)?route|"
    r"route\s+(?:is\s+)?(?:on|set)|displaying\s+directions)\b|"
    r"(?:已开始导航|开始为您导航|正在导航|路线已规划|目的地已设置)",
    re.IGNORECASE,
)
_NEGATIVE_RESULT = re.compile(
    r"\b(?:no\s+traffic|no\s+route|do\s+not\s+have|don't\s+have|"
    r"cannot|can't|unable|not\s+available|no\s+.+\s+nearby)\b|"
    r"(?:无法|不能|没有找到|未找到|不可用|路线不存在)",
    re.IGNORECASE,
)
_HOME_PATTERN = re.compile(
    r"\b(?:home|my\s+home|our\s+home|where\s+(?:do|did)\s+i\s+live)\b|"
    r"(?:回家|我家|家里)",
    re.IGNORECASE,
)
_WORK_PATTERN = re.compile(
    r"\b(?:work|my\s+work|office|my\s+office)\b|"
    r"(?:公司|单位|办公室|上班的地方)",
    re.IGNORECASE,
)

_NAME_STOPWORDS = frozenset({
    "a", "all", "an", "and", "anytime", "assistant", "be", "car",
    "certainly", "could", "day", "driver", "drive", "gps", "glad",
    "away", "have", "help", "here's", "i'm", "im", "info", "it's",
    "its", "m", "mile", "miles", "navigate", "need", "safe", "sure",
    "that's", "there's", "unfortunately", "what's",
    "i", "it", "no", "ok", "okay", "please", "route", "setting",
    "the", "then", "there", "this", "traffic", "we", "would", "yes",
    "your", "you", "chinese", "restaurant", "restaurants", "hospital",
    "hospitals", "parking", "garage", "mall", "shopping", "center",
    "coffee", "cafe", "hotel", "hotels", "gas", "station", "store",
    "food", "pizza", "place", "nearest", "closest", "quickest",
    "fastest", "address", "directions", "navigation", "road",
    "now", "there", "here", "i've", "ive", "marked", "sending", "sent",
    "you're", "youre", "which", "let's", "lets", "just", "thank",
    "thanks", "enjoy", "plotting", "navigating", "directing", "displaying",
    "taking", "give", "pick", "set", "find", "show", "sounds", "great",
    "perfect", "alright", "actually", "what", "where", "how", "could",
    "go", "going", "around", "get", "reach", "work", "welcome", "drive",
    "is", "are", "do", "does", "did", "make", "me", "my", "oh",
    "since", "that", "those", "these", "by", "block", "nearby", "avoid",
    "we're", "were",
})


@dataclass(frozen=True)
class EpisodeTurn:
    message_id: str
    speaker: str
    text: str
    timestamp: str = ""
    sequence: int = 0
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeTransition:
    action: str
    value: str
    source_id: str
    actor: str
    sequence: int


@dataclass(frozen=True)
class TypedCockpitEpisode:
    scene: str
    intent: str
    state: str
    destination: str
    address: str
    aliases: tuple[str, ...]
    source_ids: tuple[str, ...]
    confidence: float
    selection_actor: str
    mentioned_at: str
    request_text: str
    transitions: tuple[EpisodeTransition, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene": self.scene,
            "intent": self.intent,
            "state": self.state,
            "slots": {
                "destination": self.destination,
                **({"address": self.address} if self.address else {}),
            },
            "aliases": list(self.aliases),
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "selection_actor": self.selection_actor,
            "mentioned_at": self.mentioned_at,
            "request_text": self.request_text,
            "event_time": None,
            "supersedes": [
                asdict(item) for item in self.transitions[:-1]
            ],
            "transitions": [asdict(item) for item in self.transitions],
        }


@dataclass
class _EntityState:
    value: str
    aliases: list[str]
    source_ids: list[str]
    mentions: int = 0
    last_sequence: int = -1
    address: str = ""


@dataclass(frozen=True)
class _StructuredNavigationSlots:
    destination: str = ""
    address: str = ""
    state: str = ""
    destination_confidence: float = 0.0
    destination_signal: bool = False
    destination_ambiguous: bool = False


def compile_navigation_episode(
    turns: Sequence[EpisodeTurn],
    *,
    intent: str = "",
    domain: str = "",
    structured_slot_min_confidence: float = 0.85,
    structured_slot_min_margin: float = 0.08,
) -> TypedCockpitEpisode | None:
    """Compile one source-grounded navigation state without an LLM.

    Structured NLU/tool slots are preferred when present. Text fallback is a
    conservative state machine: assistant proposals remain tentative until a
    user selection, an address/route association, or a navigation outcome
    resolves them. No question, gold answer, or evidence ID is an input.
    """
    ordered = sorted(turns, key=lambda item: (item.sequence, item.message_id))
    if not ordered or not _looks_navigation(ordered, intent=intent, domain=domain):
        return None

    entities: dict[str, _EntityState] = {}
    current_key = ""
    current_state = "proposed"
    selection_actor = ""
    transitions: list[EpisodeTransition] = []
    last_address = ""
    structured_confidence = 0.0
    unresolved_structured_destination = False

    for turn in ordered:
        actor = _actor(turn.speaker)
        structured = _structured_navigation_slots(
            turn.metadata or {},
            min_confidence=structured_slot_min_confidence,
            min_margin=structured_slot_min_margin,
        )
        structured_destination = structured.destination
        structured_address = structured.address
        structured_state = structured.state
        if structured_state == "cancelled":
            if current_key:
                current_state = "cancelled"
                _append_transition(
                    transitions, "cancel", entities[current_key].value,
                    turn, actor or "system",
                )
            unresolved_structured_destination = False
            continue
        if structured.destination_signal and not structured_destination:
            # Explicit but low-confidence/conflicting NLU must not be silently
            # replaced by a surface regex over the same noisy ASR text.
            unresolved_structured_destination = True
            continue
        names = list(_candidate_names(turn.text))
        addresses = list(_addresses(turn.text))
        if structured_destination:
            names.append(structured_destination)
        if structured_address:
            addresses.append(structured_address)
        if addresses:
            last_address = addresses[-1]

        resolved_names: list[str] = []
        for name in names:
            existing = _resolve_entity(entities, name)
            # Text-only user turns may select or shorten a destination that
            # the assistant already proposed, but they cannot create a new
            # proper-name entity merely because a sentence starts with "Do",
            # "Oh", or another capitalized discourse word. Structured NLU
            # slots remain authoritative and may introduce a new value.
            if (
                actor == "user"
                and not existing
                and _key(name) != _key(structured_destination)
            ):
                continue
            key = existing or _resolve_or_add_entity(
                entities, name, turn.message_id, turn.sequence
            )
            if existing:
                key = _resolve_or_add_entity(
                    entities, name, turn.message_id, turn.sequence
                )
            if key and key not in resolved_names:
                resolved_names.append(key)

        if addresses and current_key:
            entities[current_key].address = addresses[-1]

        explicit_targets = _action_targets(turn.text)
        target_keys = []
        for value in explicit_targets:
            if _ADDRESS_PATTERN.fullmatch(value.strip()):
                continue
            key = _resolve_entity(entities, value)
            if not key:
                continue
            _prefer_explicit_target_surface(entities[key], value)
            target_keys.append(key)

        if structured_destination:
            key = _resolve_entity(entities, structured_destination)
            if key:
                current_key = key
                current_state = structured_state or "confirmed"
                selection_actor = actor or "system"
                structured_confidence = structured.destination_confidence
                unresolved_structured_destination = False
                if structured_address:
                    entities[key].address = structured_address
                _append_transition(
                    transitions, "structured_slot", entities[key].value,
                    turn, selection_actor,
                )
                continue

        if structured_state and current_key:
            current_state = structured_state
            _append_transition(
                transitions,
                "confirm" if structured_state == "confirmed" else "select",
                entities[current_key].value,
                turn,
                actor or "system",
            )
            continue

        if actor == "user":
            selected = _user_selected_entity(
                turn.text, resolved_names, entities
            )
            if selected:
                current_key = selected
                current_state = "selected"
                selection_actor = "user"
                _append_transition(
                    transitions, "select", entities[selected].value,
                    turn, "user",
                )
            elif current_key and _USER_CONFIRMATION.search(turn.text):
                current_state = "selected"
                selection_actor = selection_actor or "user"
                _append_transition(
                    transitions, "confirm", entities[current_key].value,
                    turn, "user",
                )
            continue

        if actor != "assistant":
            continue

        if target_keys and not _NEGATIVE_RESULT.search(turn.text):
            selected = target_keys[-1]
            current_key = selected
            current_state = (
                "proposed"
                if "?" in turn.text and not _ASSISTANT_CONFIRMATION.search(
                    turn.text
                )
                else "confirmed"
            )
            selection_actor = "assistant"
            if addresses:
                entities[selected].address = addresses[-1]
            _append_transition(
                transitions,
                "propose" if current_state == "proposed" else "navigate",
                entities[selected].value,
                turn, "assistant",
            )
            continue

        associated = _assistant_associated_entity(
            turn.text, resolved_names, entities
        )
        if associated:
            current_key = associated
            current_state = (
                "confirmed"
                if _ASSISTANT_CONFIRMATION.search(turn.text)
                else "selected"
            )
            selection_actor = "assistant"
            if addresses:
                entities[associated].address = addresses[-1]
            _append_transition(
                transitions,
                "navigate" if current_state == "confirmed" else "associate",
                entities[associated].value,
                turn,
                "assistant",
            )
            continue

        if len(resolved_names) == 1 and not _NEGATIVE_RESULT.search(turn.text):
            current_key = resolved_names[0]
            current_state = "proposed"
            selection_actor = "assistant"
            if addresses:
                entities[current_key].address = addresses[-1]
            _append_transition(
                transitions, "propose", entities[current_key].value,
                turn, "assistant",
            )
        elif (
            current_key
            and _ASSISTANT_CONFIRMATION.search(turn.text)
            and not _NEGATIVE_RESULT.search(turn.text)
        ):
            current_state = "confirmed"
            _append_transition(
                transitions, "confirm", entities[current_key].value,
                turn, "assistant",
            )

    alias = _navigation_alias(ordered)
    if alias:
        explicit_alternative = bool(
            current_key
            and _key(entities[current_key].value) not in {_key(alias), ""}
            and any(
                transition.action in {"select", "navigate", "associate"}
                and _key(transition.value) != _key(alias)
                for transition in transitions
            )
        )
        if not explicit_alternative:
            alias_key = _resolve_or_add_entity(
                entities, alias, ordered[0].message_id, ordered[0].sequence
            )
            if alias_key:
                alias_pattern = (
                    _HOME_PATTERN if alias == "home" else _WORK_PATTERN
                )
                for turn in ordered:
                    if (
                        alias_pattern.search(turn.text)
                        or _addresses(turn.text)
                        or _ASSISTANT_CONFIRMATION.search(turn.text)
                    ) and turn.message_id not in entities[alias_key].source_ids:
                        entities[alias_key].source_ids.append(turn.message_id)
                if last_address:
                    entities[alias_key].address = last_address
                current_key = alias_key
                current_state = (
                    "confirmed" if any(
                        _NAVIGATION_ACTION.search(turn.text)
                        for turn in ordered
                    ) else "selected"
                )
                selection_actor = selection_actor or "user"
                _append_transition(
                    transitions, "alias", alias, ordered[0], "user"
                )

    if not current_key or unresolved_structured_destination:
        return None
    entity = entities[current_key]
    source_ids = tuple(dict.fromkeys((
        *entity.source_ids,
        *(
            transition.source_id for transition in transitions
            if transition.value == entity.value and transition.source_id
        ),
    )))
    confidence = _episode_confidence(
        current_state,
        transitions,
        entity,
        selection_actor,
        structured_confidence=structured_confidence,
    )
    if confidence < 0.90:
        return None
    return TypedCockpitEpisode(
        scene="navigation",
        intent=str(intent or "navigation.set_destination"),
        state=current_state,
        destination=entity.value,
        address=entity.address,
        aliases=tuple(dict.fromkeys(entity.aliases)),
        source_ids=source_ids,
        confidence=confidence,
        selection_actor=selection_actor,
        mentioned_at=next((turn.timestamp for turn in ordered if turn.timestamp), ""),
        request_text=next((
            turn.text for turn in ordered
            if _actor(turn.speaker) == "user" and turn.text.strip()
        ), ""),
        transitions=tuple(transitions),
    )


def episode_from_dict(value: Mapping[str, Any]) -> TypedCockpitEpisode | None:
    """Read an adapter-produced episode defensively at answer time."""
    if str(value.get("scene") or "").casefold() != "navigation":
        return None
    slots = value.get("slots") or {}
    if not isinstance(slots, Mapping):
        return None
    destination = str(slots.get("destination") or "").strip()
    if not destination:
        return None
    raw_transitions = value.get("transitions") or []
    transitions: list[EpisodeTransition] = []
    if isinstance(raw_transitions, list):
        for item in raw_transitions:
            if not isinstance(item, Mapping):
                continue
            transitions.append(EpisodeTransition(
                action=str(item.get("action") or ""),
                value=str(item.get("value") or ""),
                source_id=str(item.get("source_id") or ""),
                actor=str(item.get("actor") or ""),
                sequence=int(item.get("sequence") or 0),
            ))
    return TypedCockpitEpisode(
        scene="navigation",
        intent=str(value.get("intent") or "navigation.set_destination"),
        state=str(value.get("state") or ""),
        destination=destination,
        address=str(slots.get("address") or ""),
        aliases=tuple(str(item) for item in value.get("aliases") or []),
        source_ids=tuple(str(item) for item in value.get("source_ids") or []),
        confidence=float(value.get("confidence") or 0.0),
        selection_actor=str(value.get("selection_actor") or ""),
        mentioned_at=str(value.get("mentioned_at") or ""),
        request_text=str(value.get("request_text") or ""),
        transitions=tuple(transitions),
        schema_version=int(value.get("schema_version") or 1),
    )


def _looks_navigation(
    turns: Sequence[EpisodeTurn], *, intent: str, domain: str
) -> bool:
    labels = f"{intent} {domain}".casefold()
    if re.search(r"navigation|navigate|route|maps?|location information", labels):
        return True
    text = "\n".join(turn.text for turn in turns)
    return bool(re.search(
        r"\b(?:navigate|navigation|directions?|route|gps|address|nearest|"
        r"closest|miles?\s+away|traffic)\b|"
        r"(?:导航|路线|目的地|地址|带我去|怎么走|地图)",
        text,
        re.IGNORECASE,
    ))


_DESTINATION_SLOT_KEYS = frozenset({
    "destination", "poi", "target poi", "target place", "place",
    "目的地", "目标地点", "地点", "兴趣点", "终点",
})
_ADDRESS_SLOT_KEYS = frozenset({
    "address", "destination address", "poi address",
    "地址", "详细地址", "目的地地址",
})
_STATE_SLOT_KEYS = frozenset({
    "state", "status", "navigation state", "状态", "导航状态",
})
_SLOT_NAME_KEYS = frozenset({
    "name", "slot", "slot name", "type", "label",
    "名称", "槽位", "槽位名", "类型",
})
_SLOT_VALUE_KEYS = frozenset({
    "value", "text", "normalized value", "canonical value",
    "值", "文本", "标准值", "归一化值",
})
_CONFIDENCE_KEYS = frozenset({
    "confidence", "score", "probability", "置信度", "置信分数",
})


def _structured_navigation_slots(
    metadata: Mapping[str, Any],
    *,
    min_confidence: float,
    min_margin: float,
) -> _StructuredNavigationSlots:
    destinations: list[tuple[str, float]] = []
    addresses: list[tuple[str, float]] = []
    states: list[tuple[str, float]] = []
    destination_signal = False
    queue: list[Mapping[str, Any]] = [metadata]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))

        descriptor = next((
            str(raw_value)
            for raw_key, raw_value in current.items()
            if _key(str(raw_key)) in _SLOT_NAME_KEYS
            and not isinstance(raw_value, (Mapping, list, tuple))
        ), "")
        descriptor_key = _key(descriptor)
        if descriptor_key in (
            _DESTINATION_SLOT_KEYS | _ADDRESS_SLOT_KEYS | _STATE_SLOT_KEYS
        ):
            raw_value = next((
                value for raw_key, value in current.items()
                if _key(str(raw_key)) in _SLOT_VALUE_KEYS
            ), "")
            candidates = _structured_slot_values(raw_value, current)
            if descriptor_key in _DESTINATION_SLOT_KEYS:
                destination_signal = True
                destinations.extend(candidates)
            elif descriptor_key in _ADDRESS_SLOT_KEYS:
                addresses.extend(candidates)
            else:
                states.extend(candidates)

        for raw_key, raw_value in current.items():
            key = _key(str(raw_key))
            if key in _DESTINATION_SLOT_KEYS:
                destination_signal = True
                destinations.extend(_structured_slot_values(raw_value, current))
            elif key in _ADDRESS_SLOT_KEYS:
                addresses.extend(_structured_slot_values(raw_value, current))
            elif key in _STATE_SLOT_KEYS:
                states.extend(_structured_slot_values(raw_value, current))
            if isinstance(raw_value, Mapping):
                queue.append(raw_value)
            elif isinstance(raw_value, (list, tuple)):
                queue.extend(
                    item for item in raw_value if isinstance(item, Mapping)
                )

    destination, destination_confidence, ambiguous = _choose_structured_value(
        destinations,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    address, _address_confidence, _address_ambiguous = _choose_structured_value(
        addresses,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    raw_state, _state_confidence, _state_ambiguous = _choose_structured_value(
        states,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )
    return _StructuredNavigationSlots(
        destination=destination,
        address=address,
        state=_canonical_navigation_state(raw_state),
        destination_confidence=destination_confidence,
        destination_signal=destination_signal,
        destination_ambiguous=ambiguous,
    )


def _structured_slot_values(
    raw_value: Any,
    parent: Mapping[str, Any],
) -> list[tuple[str, float]]:
    if isinstance(raw_value, Mapping):
        confidence = _mapping_confidence(
            raw_value, default=_mapping_confidence(parent, default=1.0)
        )
        return [
            (" ".join(str(value).strip().split()), confidence)
            for key, value in raw_value.items()
            if _key(str(key)) in _SLOT_VALUE_KEYS
            and not isinstance(value, (Mapping, list, tuple))
            and str(value or "").strip()
        ]
    if isinstance(raw_value, (list, tuple)):
        result: list[tuple[str, float]] = []
        for item in raw_value:
            result.extend(_structured_slot_values(item, parent))
        return result
    value = " ".join(str(raw_value or "").strip().split())
    return [(
        value, _mapping_confidence(parent, default=1.0)
    )] if value else []


def _mapping_confidence(
    value: Mapping[str, Any], *, default: float
) -> float:
    for raw_key, raw_value in value.items():
        if _key(str(raw_key)) not in _CONFIDENCE_KEYS:
            continue
        try:
            confidence = float(raw_value)
        except (TypeError, ValueError):
            continue
        if 1.0 < confidence <= 100.0:
            confidence /= 100.0
        return min(1.0, max(0.0, confidence))
    return default


def _choose_structured_value(
    candidates: Sequence[tuple[str, float]],
    *,
    min_confidence: float,
    min_margin: float,
) -> tuple[str, float, bool]:
    grouped: dict[str, tuple[str, float]] = {}
    for value, confidence in candidates:
        key = _key(value)
        if not key:
            continue
        previous = grouped.get(key)
        if previous is None or confidence >= previous[1]:
            grouped[key] = (value, confidence)
    ranked = sorted(
        (item for item in grouped.values() if item[1] >= min_confidence),
        key=lambda item: (item[1], len(item[0])),
        reverse=True,
    )
    if not ranked:
        return "", 0.0, False
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < min_margin:
        return "", 0.0, True
    return ranked[0][0], ranked[0][1], False


def _canonical_navigation_state(value: str) -> str:
    key = _key(value)
    if key in {
        "confirmed", "started", "active", "set", "navigating",
        "确认", "已确认", "已开始", "已设置", "导航中", "正在导航",
    }:
        return "confirmed"
    if key in {"selected", "选择", "已选择"}:
        return "selected"
    if key in {"proposed", "pending", "候选", "待确认"}:
        return "proposed"
    if key in {
        "cancelled", "canceled", "removed", "expired",
        "取消", "已取消", "撤销", "已撤销",
    }:
        return "cancelled"
    return ""


def _candidate_names(text: str) -> tuple[str, ...]:
    result: list[str] = []
    raw_text = str(text or "")
    # A street suffix is capitalized like a POI. Mask complete addresses
    # before proper-name detection so "434 Arastradero Rd" cannot replace a
    # previously selected Ravenswood destination with "Arastradero Rd".
    name_text = _ADDRESS_PATTERN.sub(" ", raw_text)
    for match in _PROPER_NAME_PATTERN.finditer(name_text):
        value = _clean_name(match.group(1))
        and_parts = re.split(
            r"\s+and\s+", value, maxsplit=1, flags=re.IGNORECASE
        )
        if len(and_parts) == 2 and (
            len(and_parts[0].split()) >= 2
            or len(and_parts[1].split()) >= 2
            or re.search(r"\bbetween\b[^,.!?]*" + re.escape(value), raw_text,
                         re.IGNORECASE)
            or re.search(re.escape(value) + r"[^,.!?]*\bwhich\b", raw_text,
                         re.IGNORECASE)
        ):
            result.extend(
                part for part in and_parts
                if _valid_name(part)
            )
        elif _valid_name(value):
            result.append(value)
    for target in _action_targets(raw_text):
        value = _clean_name(target)
        if _valid_name(value) and not _ADDRESS_PATTERN.fullmatch(value):
            result.append(value)
    # Lower-case names are recoverable from relation phrases, common in ASR
    # transcripts where capitalization is absent.
    relation_patterns = (
        re.compile(
            r"(?:route|way|directions?|navigation)\s+to\s+"
            r"([a-z][a-z0-9.'& -]{1,45}?)(?=[,;.!?]|\s+(?:has|is|with)\b|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:address\s+(?:of|for)|go\s+to|try)\s+"
            r"([a-z][a-z0-9.'& -]{1,45}?)(?=[,;.!?]|$)",
            re.IGNORECASE,
        ),
    )
    for pattern in relation_patterns:
        for match in pattern.finditer(name_text):
            value = _clean_name(match.group(1))
            if _valid_name(value):
                result.append(value)
    return tuple(dict.fromkeys(result))


def _action_targets(text: str) -> tuple[str, ...]:
    return tuple(
        _clean_target(match.group(1))
        for match in _ACTION_TARGET.finditer(str(text or ""))
        if _clean_target(match.group(1))
    )


def _addresses(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        " ".join(match.group(1).strip(" ,.!?").split())
        for match in _ADDRESS_PATTERN.finditer(str(text or ""))
    ))


def _clean_target(value: str) -> str:
    result = " ".join(str(value or "").strip(" ,.!?").split())
    address = _ADDRESS_PATTERN.search(result)
    if address:
        # ``heading to Midtown Shopping Center at 338 Alester Ave`` carries
        # both a POI and its address.  Preserve the explicit POI target; an
        # address-only command still returns the address so it can bind to the
        # already selected entity without inventing a new destination.
        prefix = re.sub(
            r"\s+(?:located\s+)?at$", "", result[:address.start()].rstrip(),
            flags=re.IGNORECASE,
        ).strip(" ,.!?")
        if prefix:
            result = prefix
        else:
            return " ".join(address.group(1).split())
    result = re.sub(
        r"^(?:the\s+)?(?:fastest|quickest|shortest|best|only)\s+"
        r"(?:route\s+)?(?:to\s+)?",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.split(
        r"\s+(?:is|are|has|have|will|was|were|with|through|because|but)\b",
        result,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    result = re.sub(
        r"\s+(?:then|now|please|instead|there|at)$",
        "",
        result,
        flags=re.IGNORECASE,
    )
    return result.strip(" ,.!?")


def _clean_name(value: str) -> str:
    result = " ".join(str(value or "").strip(" ,.!?").split())
    # The permissive proper-name matcher intentionally supports dotted names
    # such as ``P.F. Changs``.  It can therefore span a sentence boundary in
    # ``Jing Jing. Need more info?``.  Trim only a trailing discourse clause;
    # a real continuation such as ``P.F. Changs`` remains intact because its
    # first token is not a stopword.
    sentence_parts = re.split(r"(?<=[.!?])\s+", result)
    if len(sentence_parts) > 1:
        continuation = re.findall(
            r"[A-Za-z0-9']+", sentence_parts[1].casefold()
        )
        if continuation and continuation[0] in _NAME_STOPWORDS:
            result = sentence_parts[0].rstrip(" ,.!?")
    result = re.sub(
        r"^(?:and|or|then|okay|ok|alright|the\s+address\s+(?:of|for)|"
        r"the\s+(?:route|way)\s+to|let(?:'s| us)\s+go\s+(?:to|at))\s+",
        "",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"(?:'s|s')\s+address$", "", result, flags=re.IGNORECASE)
    result = re.sub(
        r"\s+(?:then|now|please|instead|there|at)$",
        "",
        result,
        flags=re.IGNORECASE,
    )
    return result.strip(" ,.!?")


def _valid_name(value: str) -> bool:
    if not value or _ADDRESS_PATTERN.fullmatch(value):
        return False
    cjk = "".join(re.findall(r"[\u3400-\u9fff]+", value))
    if cjk and not re.search(r"[A-Za-z0-9]", value):
        return 1 <= len(cjk) <= 32
    words = re.findall(r"[A-Za-z0-9']+", value.casefold())
    if not 1 <= len(words) <= 7 or len(value) > 64:
        return False
    meaningful = [word for word in words if word not in _NAME_STOPWORDS]
    if not meaningful:
        return False
    if all(word.isdigit() for word in meaningful):
        return False
    return True


def _resolve_or_add_entity(
    entities: dict[str, _EntityState],
    value: str,
    source_id: str,
    sequence: int,
) -> str:
    cleaned = _clean_name(value)
    if not _valid_name(cleaned):
        return ""
    resolved = _resolve_entity(entities, cleaned)
    if resolved:
        entity = entities[resolved]
        if cleaned not in entity.aliases:
            entity.aliases.append(cleaned)
        if _is_surface_correction(entity.value, cleaned):
            entity.value = cleaned
    else:
        resolved = _key(cleaned)
        entities[resolved] = _EntityState(cleaned, [cleaned], [])
        entity = entities[resolved]
    entity.mentions += 1
    entity.last_sequence = max(entity.last_sequence, sequence)
    if source_id and source_id not in entity.source_ids:
        entity.source_ids.append(source_id)
    return resolved


def _resolve_entity(entities: Mapping[str, _EntityState], value: str) -> str:
    target = _key(_clean_name(value))
    if not target:
        return ""
    if target in entities:
        return target
    matches = [
        key for key in entities
        if _keys_compatible(target, key)
    ]
    return matches[0] if len(matches) == 1 else ""


def _prefer_explicit_target_surface(entity: _EntityState, value: str) -> None:
    """Repair a broad conjunction parse using a later explicit route target."""
    cleaned = _clean_name(value)
    current_key = _key(entity.value)
    target_key = _key(cleaned)
    if (
        not target_key
        or target_key == current_key
        or " and " not in current_key
        or " and " in target_key
        or len(target_key.split()) < 2
        or not _keys_compatible(current_key, target_key)
    ):
        return
    if cleaned not in entity.aliases:
        entity.aliases.append(cleaned)
    entity.value = cleaned


def _keys_compatible(left: str, right: str) -> bool:
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return False
    if left == right:
        return True
    # Common ASR/transcript surface drift should not split one otherwise
    # identical entity, but this deliberately does not invent new names.
    substitutions = {"coffee": "cafe", "centre": "center"}
    normalized_left = [substitutions.get(token, token) for token in left_tokens]
    normalized_right = [substitutions.get(token, token) for token in right_tokens]
    if normalized_left == normalized_right:
        return True

    def contiguous_subset(shorter: list[str], longer: list[str]) -> bool:
        width = len(shorter)
        return width > 0 and any(
            longer[index:index + width] == shorter
            for index in range(len(longer) - width + 1)
        )

    return (
        contiguous_subset(normalized_left, normalized_right)
        or contiguous_subset(normalized_right, normalized_left)
    )


def _is_surface_correction(previous: str, current: str) -> bool:
    previous_key = _key(previous)
    current_key = _key(current)
    if not previous_key or not current_key or previous_key == current_key:
        return False
    substitutions = {"coffee": "cafe", "centre": "center"}
    normalized_previous = [
        substitutions.get(token, token) for token in previous_key.split()
    ]
    normalized_current = [
        substitutions.get(token, token) for token in current_key.split()
    ]
    return normalized_previous == normalized_current


def _user_selected_entity(
    text: str,
    resolved_names: Sequence[str],
    entities: Mapping[str, _EntityState],
) -> str:
    if not resolved_names:
        return ""
    folded = str(text or "").casefold()
    explicit = bool(
        _USER_SELECTION.search(text)
        or _NAVIGATION_ACTION.search(text)
        or re.search(r"\b(?:then|instead|rather|faster|closer)\b", folded)
    )
    if not explicit and len(re.findall(r"[a-z0-9']+", folded)) <= 5:
        explicit = any(
            _key(entities[key].value) in _key(text)
            for key in resolved_names
        )
    if not explicit:
        return ""
    # Prefer the last named value: corrections and "X then" place the final
    # destination after an earlier rejected alternative.
    return resolved_names[-1]


def _assistant_associated_entity(
    text: str,
    resolved_names: Sequence[str],
    entities: Mapping[str, _EntityState],
) -> str:
    if not resolved_names:
        return ""
    if len(resolved_names) == 1 and (
        _ASSISTANT_SELECTION.search(text)
        or _ASSISTANT_CONFIRMATION.search(text)
        or _ADDRESS_PATTERN.search(text)
    ):
        return resolved_names[0]
    if len(resolved_names) > 1:
        for raw_target in _action_targets(text):
            target = _resolve_entity(entities, raw_target)
            if target:
                return target
        recommendation = re.search(
            r"(?:suggest|recommend|vote\s+for|pick(?:ed)?|select(?:ed)?)"
            r"[^.!?]{0,45}?([A-Z][A-Za-z0-9.'&-]*(?:\s+[A-Z][A-Za-z0-9.'&-]*){0,4})",
            text,
            re.IGNORECASE,
        )
        if recommendation:
            target = _resolve_entity(entities, recommendation.group(1))
            if target:
                return target
    return ""


def _navigation_alias(turns: Sequence[EpisodeTurn]) -> str:
    user_text = "\n".join(
        turn.text for turn in turns if _actor(turn.speaker) == "user"
    )
    assistant_text = "\n".join(
        turn.text for turn in turns if _actor(turn.speaker) == "assistant"
    )
    if _HOME_PATTERN.search(user_text) and _HOME_PATTERN.search(assistant_text):
        return "home"
    if _WORK_PATTERN.search(user_text) and _WORK_PATTERN.search(assistant_text):
        return "work"
    return ""


def _episode_confidence(
    state: str,
    transitions: Sequence[EpisodeTransition],
    entity: _EntityState,
    selection_actor: str,
    *,
    structured_confidence: float = 0.0,
) -> float:
    relevant = [item for item in transitions if item.value == entity.value]
    actions = {item.action for item in relevant}
    assistant_introduced = any(
        item.actor == "assistant"
        and item.action in {"propose", "associate", "navigate"}
        for item in relevant
    )
    user_accepted = any(
        item.actor == "user" and item.action in {"select", "confirm"}
        for item in relevant
    )
    assistant_resolved = any(
        item.actor == "assistant"
        and item.action in {"navigate", "confirm"}
        for item in relevant
    )
    if "structured_slot" in actions:
        confidence = structured_confidence or 1.0
        if assistant_resolved and state == "confirmed":
            confidence = min(0.999, confidence + 0.07)
        return confidence
    if "select" in actions and ("navigate" in actions or "confirm" in actions):
        return 0.995
    if assistant_introduced and user_accepted and assistant_resolved:
        return 0.995
    if assistant_introduced and user_accepted:
        return 0.99
    if "associate" in actions and assistant_resolved:
        return 0.98
    if "navigate" in actions:
        return 0.99
    if "alias" in actions and state in {"selected", "confirmed"}:
        return 0.99
    if "associate" in actions and (entity.address or entity.mentions >= 2):
        return 0.98
    if state in {"selected", "confirmed"} and entity.mentions >= 2:
        return 0.97
    if selection_actor == "assistant" and entity.mentions >= 2:
        return 0.95
    return 0.90


def _append_transition(
    transitions: list[EpisodeTransition],
    action: str,
    value: str,
    turn: EpisodeTurn,
    actor: str,
) -> None:
    item = EpisodeTransition(
        action=action,
        value=value,
        source_id=turn.message_id,
        actor=actor,
        sequence=turn.sequence,
    )
    if not transitions or transitions[-1] != item:
        transitions.append(item)


def _actor(value: str) -> str:
    folded = _key(value)
    if folded in {"driver", "user", "passenger", "司机", "驾驶员", "用户", "乘客"}:
        return "user"
    if folded in {"assistant", "car assistant", "system", "助手", "车机", "系统"}:
        return "assistant"
    return ""


def _key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(
        r"[a-z0-9]+|[\u3400-\u9fff]+", normalized
    ))
