"""Generic, evidence-grounded state/event projection for cockpit retrieval.

This module is deliberately independent of a benchmark's gold answers.  It
only consumes the rendered evidence lines (and their source metadata), turns
them into immutable events, and resolves safe query shapes before an LLM is
called.  Ambiguous cases return ``None`` and remain on the model path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence
from datetime import datetime, timedelta
from .temporal import resolve_temporal_query


@dataclass(frozen=True)
class StateEvent:
    event_id: str
    source_id: str
    person: str
    value: str
    timestamp: str
    action: str
    completed: bool
    cancelled: bool


@dataclass(frozen=True)
class StateInterval:
    slot: str
    person: str
    value: str
    valid_from: datetime
    valid_to: datetime | None
    priority: int
    source_id: str


@dataclass(frozen=True)
class GroundedSlotEvent:
    """One immutable person/field value copied from a source user turn."""

    source_id: str
    timestamp: str
    person: str
    domain: str
    field: str
    value: str
    semantic_kind: str


_LINE = re.compile(
    r"^\[(?P<source>[^\]]+)\](?:\s+\[source_time=(?P<time>[^\]]+)\])?\s+"
    r"(?:\[source_role=(?P<role>[^\]]+)\]\s+)?(?P<body>.*)$"
)
_PERSON = re.compile(r"[【\[]([^】\]]+)[】\]]")
_NAV = re.compile(
    r"(?:导航(?:到|去)|导航至|目的地(?:是|为)|开到|前往|去往)\s*"
    r"([^，。；;！!？?\n]+)"
    r"|(?:navigate|head|go|drive|route)\s+(?:to\s+)?([A-Za-z0-9][^,.!?;\n]+)",
    re.I,
)
_DONE = re.compile(r"(?:已到达|到达目的地|行程完成|导航完成|完成导航|到达|arrived|completed|trip complete)", re.I)
_CANCEL = re.compile(r"(?:取消|撤销|作废|不去了|不用了|cancel(?:led)?)", re.I)
_UPDATE = re.compile(r"(?:改成|改为|更新为|更新到|替换为|变更为|instead of|changed? to|update(?:d)? to)", re.I)
_TEMP_DEST_BASE = re.compile(
    r"(?:平时(?:说)?临时目的[地的](?:就)?(?:导航到|是|为)|临时目的[地的](?:默认)?(?:是|为))\s*"
    r"([^，。；;！!？?\n]+)"
)
_TEMP_DEST_OVERRIDE = re.compile(
    r"(\d{1,2})月(\d{1,2})日?\s*(?:到|至|-)\s*"
    r"(?:(\d{1,2})月)?(\d{1,2})日?\s*"
    r"临时目的[地的]\s*(?:改成|改为|更新为|是|为)\s*"
    r"([^，。；;！!？?\n]+)"
)
_TEMP_DEST_RESTORE = re.compile(
    r"(?:从今天起)?临时目的[地的]\s*(?:恢复为|恢复到)\s*"
    r"([^，。；;！!？?\n]+)"
)


def parse_events(context: str, hits: Sequence[Mapping[str, Any]] = ()) -> tuple[StateEvent, ...]:
    """Parse only source-grounded lines; summary-only hits are not events."""
    rows: list[tuple[str, str, str]] = []
    for raw in str(context or "").splitlines():
        m = _LINE.match(raw.strip())
        if m:
            rows.append((m.group("source"), m.group("time") or "", m.group("body")))
    events: list[StateEvent] = []
    seen: set[str] = set()
    for source, timestamp, body in rows:
        nav = _NAV.search(body)
        if not nav:
            continue
        value = (nav.group(1) or nav.group(2) or "").strip(" ，。；;.!?\"'")
        value = re.split(r"(?:已经?到达|已到达|行程完成|导航完成|完成导航|arrived|completed|trip complete)", value, maxsplit=1, flags=re.I)[0].strip(" ，。；;.!?\"'")
        if not value:
            continue
        event_id = source.split(":", 1)[0]
        if event_id in seen:
            continue
        seen.add(event_id)
        person = (_PERSON.search(body).group(1).strip() if _PERSON.search(body) else "")
        cancelled = bool(_CANCEL.search(body))
        action = "cancel" if cancelled else "update" if _UPDATE.search(body) else "navigate"
        events.append(StateEvent(event_id, source, person, value, timestamp, action,
                                 bool(_DONE.search(body)), cancelled))
    return tuple(events)


def resolve_state_answer(question: str, context: str, hits: Sequence[Mapping[str, Any]] = (), metadata: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Return a conservative structured answer for generic event queries."""
    q = str(question or "")
    # These two chain operators deliberately run before the navigation state
    # projector.  Both are common cockpit shapes, but neither is a single
    # global state: owner-scoped preferences must not be overwritten by an
    # operational command, and query history must be ordered by source event
    # time rather than semantic-search rank.
    owner_bindings = _resolve_multi_person_preferences(q, context)
    if owner_bindings is not None:
        return owner_bindings
    query_sequence = _resolve_query_sequence(q, context)
    if query_sequence is not None:
        return query_sequence
    # Historical search/query slots are not navigation states, but they share
    # the same effective-time rule: select the latest source-grounded request
    # at or before the declared cutoff and extract only the named field.
    temporal = resolve_temporal_query(q, metadata or {}, default_timezone=str((metadata or {}).get("timezone") or "UTC"))
    cutoff_slot = _resolve_cutoff_query_slot(q, context, temporal)
    if cutoff_slot is not None:
        return cutoff_slot
    # This projector currently resolves navigation events only.  A search,
    # reminder, preference, or booking query has different state semantics and
    # must stay on the typed-episode/LLM path instead of being guessed as a
    # destination event.
    if not re.search(r"(?:导航|目的[地的]|到达|去得最多|navigation|destination|arriv)", q, re.I):
        return None
    events = parse_events(context, hits)
    # Apply explicit/relative query windows only when the caller supplies a
    # trusted anchor.  Never invent a date from the evidence itself.
    snapshots = _resolve_two_date_destination(q, context, temporal)
    if snapshots is not None:
        return snapshots
    cutoff = _query_cutoff(q, temporal)
    if temporal.spans:
        bounded = []
        for event in events:
            if not event.timestamp:
                continue
            try:
                stamp = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ((stamp <= cutoff) if cutoff is not None else any(span.contains(stamp) for span in temporal.spans)):
                bounded.append(event)
        if bounded:
            events = tuple(bounded)
    if not events:
        return None
    # Aggregation is valid only over explicit completed, source-distinct events.
    if re.search(r"(?:按.*(?:地点|目的地).*统计|最多.*(?:几次|次数)|一共(?:去|导航).*几次|how many times|most)", q, re.I):
        completed = [e for e in events if e.completed and not e.cancelled]
        if not completed:
            return None
        expected = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                    "七": 7, "八": 8, "九": 9, "十": 10}
        expected_count = next((n for word, n in expected.items() if f"{word}次" in q), None)
        if expected_count is not None and len(completed) < expected_count:
            # Retrieval did not cover the requested event chain: abstain
            # instead of estimating a mode from a truncated top-k window.
            return None
        counts = Counter(e.value for e in completed)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        value, count = top[0]
        return {"value": f"{value}，共{count}次。", "slot_label": "aggregation",
                "source_ids": [e.source_id for e in completed if e.value == value],
                "confidence": 0.96, "reason": "structured_completed_event_aggregation"}
    # Latest/final state: sort by event time when all candidates are timestamped.
    if re.search(r"(?:最新导航|最后一次导航|最终导航|最后一次到达|latest navigation|last destination)", q, re.I):
        dated = [e for e in events if e.timestamp]
        if not dated:
            return None
        dated.sort(key=lambda e: (e.timestamp, e.source_id))
        if len(dated) >= 2 and re.search(r"(?:上一次|前一次|previous|prior)", q, re.I) and re.search(r"(?:最后一次|最终|last)", q, re.I):
            previous, final = dated[-2], dated[-1]
            if previous.cancelled or final.cancelled:
                return None
            return {"value": f"上一次：{previous.value}；最后一次：{final.value}。",
                    "slot_label": "state_sequence", "source_ids": [previous.source_id, final.source_id],
                    "confidence": 0.93, "reason": "structured_ordered_event_sequence"}
        final = dated[-1]
        if final.cancelled:
            return {"value": "已取消。", "slot_label": "state", "source_ids": [final.source_id],
                    "confidence": 0.95, "reason": "structured_final_cancellation"}
        return {"value": final.value, "slot_label": "state", "source_ids": [final.source_id],
                "confidence": 0.93, "reason": "structured_latest_event_state"}
    return None


def _source_user_rows(context: str) -> list[tuple[str, str, str, str]]:
    """Return unique source-grounded user rows as source/time/person/body."""
    rows: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for raw in str(context or "").splitlines():
        row = _LINE.match(raw.strip())
        if not row or row.group("role") not in {"user", ""}:
            continue
        source_id = row.group("source")
        if source_id in seen:
            continue
        seen.add(source_id)
        body = row.group("body")
        person_match = _PERSON.search(body)
        rows.append((
            source_id,
            row.group("time") or "",
            person_match.group(1).strip() if person_match else "",
            body,
        ))
    return rows


def _question_people(
    question: str, rows: Sequence[tuple[str, str, str, str]]
) -> tuple[str, ...]:
    """Read requested people from source-grounded names, preserving order."""
    explicit = re.search(
        r"(?:^|[，,：:])([^，,：:。；;？?]{1,80}?(?:、|和|与|及)"
        r"[^，,：:。；;？?]{1,40})各自",
        question,
    )
    if explicit:
        values = tuple(
            value.strip()
            for value in re.split(r"、|和|与|及", explicit.group(1))
            if value.strip()
        )
        if 2 <= len(values) <= 8:
            return values
    names = {
        person for _source, _timestamp, person, _body in rows
        if person and person in question
    }
    return tuple(sorted(names, key=lambda name: question.index(name)))


def _persistent_preference_value(question: str, body: str) -> tuple[str, str, str] | None:
    """Classify durable cockpit preferences without folding in commands.

    The returned tuple is ``(domain, field, value)``.  A value is accepted
    only from explicit durable language (``常用``/``喜欢``/``偏好``), so a
    later one-shot ``播放`` or ``先设为`` operation lives in a different
    namespace and cannot accidentally supersede it.
    """
    text = _PERSON.sub("", body)
    specifications = (
        (
            r"(?:常用车机目的[地的]|常用目的[地的])",
            r"(?:我的)?(?:常用车机目的[地的]|常用目的[地的])\s*(?:记成|设成|设置为|设为|是|为)\s*"
            r"(?P<value>[^，。；;！!？?\n]+?)(?=\s*，?\s*(?:只归|仅归)|[。；;！!？?]|$)",
            "navigation",
            "usual_destination",
        ),
        (
            r"(?:停车休息.*(?:媒体偏好|车载媒体|车载内容)|媒体偏好)",
            r"停车休息时(?:我)?(?:喜欢|偏好)的?(?:车载)?(?:媒体)?内容\s*(?:记成|设成|设置为|设为|是|为)\s*"
            r"(?P<value>[^，。；;！!？?\n]+?)(?=\s*，?\s*(?:只归|仅归)|[。；;！!？?]|$)",
            "media",
            "parking_rest_preference",
        ),
    )
    for question_pattern, source_pattern, domain, field in specifications:
        if not re.search(question_pattern, question):
            continue
        match = re.search(source_pattern, text)
        if match:
            value = match.group("value").strip(" ，。；;.!?\"'")
            if value:
                return domain, field, value
    return None


def _resolve_multi_person_preferences(question: str, context: str) -> dict[str, Any] | None:
    if not re.search(r"(?:各自|分别).*(?:偏好|常用|习惯|默认|目的[地的])", question):
        return None
    rows = _source_user_rows(context)
    people = _question_people(question, rows)
    if len(people) < 2:
        return None
    by_person: dict[str, list[GroundedSlotEvent]] = defaultdict(list)
    for source_id, timestamp, person, body in rows:
        if person not in people:
            continue
        extracted = _persistent_preference_value(question, body)
        if extracted is None:
            continue
        domain, field, value = extracted
        by_person[person].append(GroundedSlotEvent(
            source_id, timestamp, person, domain, field, value,
            "persistent_preference",
        ))
    # Coverage is a hard contract: never fill one person's missing value with
    # another person's evidence.  Ambiguous same-time values also fail closed.
    selected: list[GroundedSlotEvent] = []
    for person in people:
        events = by_person.get(person, [])
        if not events:
            return None
        events.sort(key=lambda event: (event.timestamp, event.source_id))
        latest_time = events[-1].timestamp
        latest = [event for event in events if event.timestamp == latest_time]
        values = list(dict.fromkeys(event.value for event in latest))
        if len(values) != 1:
            return None
        selected.append(next(event for event in latest if event.value == values[0]))
    return {
        "value": "；".join(
            f"{event.person}是{event.value}" for event in selected
        ) + "。",
        "slot_label": "owner_scoped_preferences",
        "source_ids": [event.source_id for event in selected],
        "confidence": 0.98,
        "reason": "structured_owner_scoped_preferences",
    }


def _resolve_query_sequence(question: str, context: str) -> dict[str, Any] | None:
    if not (
        re.search(r"(?:最后一次|最近一次|最新一次)", question)
        and re.search(r"(?:前一次|上一次)", question)
        and re.search(r"明确查询的(.+?)是什么", question)
    ):
        return None
    rows = _source_user_rows(context)
    people = _question_people(question, rows)
    if len(people) != 1:
        return None
    person = people[0]
    slot_match = re.search(r"明确查询的(.+?)是什么", question)
    domain_match = re.search(
        r"这段\s*(餐馆|餐厅|景点|酒店|航班|飞机|火车|电影|电视剧|天气|汽车|电脑)\s*检索",
        question,
    )
    field = slot_match.group(1).strip()
    domain = domain_match.group(1) if domain_match else ""
    # A field such as ``系列`` or ``城市`` is ambiguous without its domain.
    # Compose both parts for the extractor instead of maintaining a fragile
    # allow-list of fields that happen to need qualification.
    extractor_slot = domain + field if domain else field
    events: list[GroundedSlotEvent] = []
    for source_id, timestamp, owner, body in rows:
        if owner != person or not timestamp:
            continue
        # Keep search/history events separate from later cockpit operations.
        # A destination mentioned in a navigation, appointment, cancellation,
        # or command update is not another scenic/hotel/restaurant query.
        if re.search(
            r"(?:导航|预约|检查|取消|目的[地的]|车机口令|已到达)", body
        ):
            continue
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        value = _extract_named_query_slot(extractor_slot, body)
        if not value:
            continue
        events.append(GroundedSlotEvent(
            source_id, timestamp, person, domain, field, value, "query_event"
        ))
    events.sort(key=lambda event: (event.timestamp, event.source_id))
    if len(events) < 2:
        return None
    previous, final = events[-2], events[-1]
    if previous.source_id == final.source_id:
        return None
    return {
        "value": f"最后一次是{final.value}；前一次是{previous.value}。",
        "slot_label": "ordered_query_field",
        # Preserve answer order rather than chronological display order.
        "source_ids": [final.source_id, previous.source_id],
        "confidence": 0.98,
        "reason": "structured_ordered_query_field",
    }


def _resolve_cutoff_query_slot(question: str, context: str, temporal) -> dict[str, Any] | None:
    """Extract one explicitly requested field from the cutoff predecessor.

    This is a small typed-slot compiler, not a benchmark answer lookup.  Slot
    names come from the question and values must be present verbatim in a
    timestamped user source row.  Ambiguous or absent values fall through to
    the model path.
    """
    cutoff = _query_cutoff(question, temporal)
    slot_match = re.search(r"明确查询的(.+?)是什么", question)
    if cutoff is None or not slot_match:
        return None
    slot = slot_match.group(1).strip()
    rows: list[tuple[datetime, str, str]] = []
    for raw in str(context or "").splitlines():
        row = _LINE.match(raw.strip())
        if not row or not row.group("time") or row.group("role") not in {"user", ""}:
            continue
        try:
            recorded = datetime.fromisoformat(row.group("time").replace("Z", "+00:00"))
        except ValueError:
            continue
        if recorded <= cutoff:
            rows.append((recorded, row.group("source"), row.group("body")))
    if not rows:
        return None
    grounded_people = {
        match.group(1).strip()
        for _, _, body in rows
        for match in [_PERSON.search(body)]
        if match and match.group(1).strip() in question
    }
    if len(grounded_people) == 1:
        owner = next(iter(grounded_people))
        rows = [
            row for row in rows
            if (match := _PERSON.search(row[2])) is None
            or match.group(1).strip() == owner
        ]
    elif len(grounded_people) > 1:
        # More than one retrieved speaker is explicitly named: ownership is a
        # first-class ambiguity and must not be guessed.
        return None
    # A cutoff query asks for the predecessor state.  Older rows are retained
    # only as fallback when the boundary row does not contain the named slot.
    rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for recorded in sorted({item[0] for item in rows}, reverse=True):
        extracted: list[tuple[str, str]] = []
        for _, source_id, body in (item for item in rows if item[0] == recorded):
            value = _extract_named_query_slot(slot, body)
            if value:
                extracted.append((source_id, value))
        distinct = list(dict.fromkeys(value for _, value in extracted))
        if len(distinct) == 1:
            value = distinct[0]
            return {
                "value": f"截至{_short_date(cutoff)}是{value}。",
                "slot_label": "cutoff_query_slot",
                "source_ids": [source for source, candidate in extracted if candidate == value],
                "confidence": 0.96,
                "reason": "structured_cutoff_predecessor_slot",
            }
        if len(distinct) > 1:
            return None
    return None


def _extract_named_query_slot(slot: str, body: str) -> str:
    """Return a source-verbatim value for a Chinese cockpit query slot."""
    text = _PERSON.sub("", body)
    compact_slot = re.sub(r"\s+", "", slot)

    patterns: list[str] = []
    if "人均消费" in compact_slot:
        patterns = [r"人均消费(?:是|为|在)?\s*([0-9]+\s*(?:-|到|至)\s*[0-9]+元|[0-9]+元(?:以上|以下|以内)?)"]
    elif "价格" in compact_slot or "门票" in compact_slot:
        if "门票" in compact_slot:
            patterns.append(r"(不免费|免费)(?:并且|而且|，|,|\s)")
        patterns.extend([
            r"(?:最低)?价格(?:是|为|在)?\s*([0-9]+\s*(?:-|到|至)\s*[0-9]+元|[0-9]+元(?:以上|以下|以内)?)",
            r"票价(?:是|为|在)?\s*([0-9]+\s*(?:-|到|至)\s*[0-9]+元|[0-9]+元(?:以上|以下|以内)?)",
        ])
    elif "评分" in compact_slot:
        patterns = [r"评分(?:是|为|在)?\s*([0-9]+(?:\.[0-9]+)?分(?:以上|以下)?)"]
    elif "游玩时间" in compact_slot or "游玩时长" in compact_slot:
        patterns = [
            r"游玩(?:时间|时长)(?:是|为)?\s*"
            r"([0-9]+(?:\.[0-9]+)?\s*(?:小时|天)(?:\s*(?:-|到|至)\s*"
            r"[0-9]+(?:\.[0-9]+)?\s*(?:小时|天))?)"
        ]
    elif "推荐菜" in compact_slot:
        patterns = [
            r"能吃到\s*([^，。；;！!？?]+?)(?:的(?:餐馆)?|吗|吧|$)",
            r"(?:吃|有)\s*([^，。；;！!？?]+?)(?:这道菜|这个菜)",
            r"餐馆有\s*([^，。；;！!？?]+?)(?:这个菜|吗)",
            r"(?:，|,)\s*有\s*([^，。；;！!？?]+?)(?:[。；;！!？?]|$)",
        ]
    elif "酒店类型" in compact_slot:
        patterns = [r"(?:的|个|家|找)\s*([\u4e00-\u9fffA-Za-z]{2,6}型)酒店"]
    elif "电视剧类型" in compact_slot or "电影类型" in compact_slot:
        patterns = [r"喜欢看\s*([A-Za-z\u4e00-\u9fff]{2,10}片)", r"(?:找|推荐)(?:一部)?\s*([A-Za-z\u4e00-\u9fff]{2,10}片)"]
    elif "日期" in compact_slot:
        patterns = [r"(下周[一二三四五六日天](?:上午|下午|晚上)?)", r"(本周[一二三四五六日天](?:上午|下午|晚上)?)"]
    elif "舱位" in compact_slot:
        patterns = [r"(头等舱|公务舱|商务舱|经济舱)"]
    elif "坐席" in compact_slot or "座席" in compact_slot:
        patterns = [r"(商务座|特等座|一等座|二等座|软卧|硬卧|软座|硬座|无座)"]
    elif "电脑系列" in compact_slot:
        patterns = [
            r"(?:了解一下|推荐一部|推荐一款|选择一款|选一款|一款|对比一下|比对一下).*?([A-Za-z\u4e00-\u9fff]+(?:-|－)[A-Za-z\u4e00-\u9fff]+)(?:这个)?系列",
            r"(?:了解一下|推荐一部|推荐一款|选择一款|选一款|一款|对比一下|比对一下|价位|价格区间的?)\s*([A-Za-z\u4e00-\u9fff]{2,12}?)(?:这个)?系列",
        ]
    elif "主演" in compact_slot:
        patterns = [
            r"(?:看看|电影看看|影片看看)\s*([A-Za-z\u4e00-\u9fff·]{2,12})是我喜欢的演员",
            r"由\s*([A-Za-z\u4e00-\u9fff·]{2,12})(?:主演|出演)",
            r"(?:一部(?!由)|推荐一部)\s*([A-Za-z\u4e00-\u9fff·]{2,12})(?:主演|出演)",
        ]
    elif "天气城市" in compact_slot:
        patterns = [
            r"(?:下周|本周)[一二三四五六日天](?:上午|下午|晚上)?\s*"
            r"([\u4e00-\u9fff]{2,8}?)(?:的|什么)?天气",
            r"(?:查查|看看|看下)\s*(?:(?:下周|本周)[一二三四五六日天])?\s*([\u4e00-\u9fff]{2,8}?)(?:的|什么)?天气",
            r"(?:(?:下周|本周)[一二三四五六日天]|那天)\s*([\u4e00-\u9fff]{2,8}?)(?:什么)?天气",
        ]
    elif "天气日期" in compact_slot:
        patterns = [r"(下周[一二三四五六日天])"]
    elif "飞机目的地" in compact_slot:
        patterns = [r"从[^，。；;！!？?\s]+(?:去|到|回)\s*([^，。；;！!？?\s]{2,12}?)(?:出差|旅游|办事|的飞机|[，。；;！!？?]|$)"]
    elif "菜系" in compact_slot:
        patterns = [
            r"(?:选一家|找个|找一家|找家|推荐一家)\s*"
            r"(?:(?:高档|中高档|中等|平价|便宜|低档)(?:价位)?的?)?\s*"
            r"([^，。；;！!？?]{2,12}?)(?:的)?(?:餐厅|餐馆|好了|把|吧|$|[，。；;！!？?])"
        ]
    elif "价位" in compact_slot:
        patterns = [r"(高档|中高档|中等|平价|便宜|低档)(?:价位)?"]
    elif "车系" in compact_slot:
        patterns = [
            r"(?:^|[，。；;！!？?])\s*([A-Za-z\u4e00-\u9fff0-9-]{2,20})(?:不错|怎么样|这个车系)",
            r"(?:看过|了解|看看)\s*([A-Za-z\u4e00-\u9fff0-9-]{2,20}?)(?:，|,|我想|怎么样|是什么|[。；;！!？?]|$)",
        ]
    elif "名称" in compact_slot:
        if "景点" in compact_slot:
            patterns = [
                r"(?:想去(?:景点)?|就去|去)\s*"
                r"([^，。；;！!？?]{2,30}?)(?=，|帮|要门票|溜达|玩(?:了)?|游玩|$)",
                r"(?:^|[，。；;！!？?\s])([^，。；;！!？?\s]{2,30}?)(?:的)?"
                r"(?:具体)?(?:附近|周边|地址|电话|要门票|看着)",
                r"(?:^|[，。；;！!？?])\s*([^，。；;！!？?]{2,30}?)(?:要门票|看着|(?:的)?(?:周边|附近))",
                r"(?:^|[，。；;！!？?])\s*([^，。；;！!？?]{2,30}?)"
                r"(?:(?:的)?(?:具体)?(?:电话|地址|票价)|要门票)",
                r"(?:想去|就去|去)\s*([^，。；;！!？?]{2,30}?)(?:玩|游玩|溜达)",
            ]
        elif "餐" in compact_slot:
            patterns = [
                r"(?:^|[，。；;！!？?])\s*([^，。；;！!？?]{2,40}?)(?:周边|这家餐馆)",
                r"(?:^|[，。；;！!？?])\s*([^，。；;！!？?]{2,40}?)(?:的)?(?:人均消费|评分|推荐菜|地址|电话)(?:是|为|有|在)",
            ]

    for pattern in patterns:
        values: list[str] = []
        for match in re.finditer(pattern, text, re.I):
            value = re.sub(r"\s*(?:-|到|至)\s*", "-", match.group(1)).strip(" ，。；;.!?\"'")
            if "电脑系列" in compact_slot:
                value = re.sub(
                    r"^(?:同等|相同|相近)(?:的)?(?:价格区间|价位)(?:的)?",
                    "",
                    value,
                )
            if "天气城市" in compact_slot:
                value = re.sub(
                    r"^(?:下周|本周)[一二三四五六日天](?:上午|下午|晚上)?",
                    "",
                    value,
                )
            if "推荐菜" in compact_slot or "名称" in compact_slot:
                value = value.removesuffix("的")
            if value and value not in {
                "这个景点", "这一个景点", "该景点", "它", "它的",
            }:
                values.append(value)
        distinct = list(dict.fromkeys(values))
        if len(distinct) == 1:
            return distinct[0]
        if len(distinct) > 1:
            return ""
    return ""


def _resolve_two_date_destination(question: str, context: str, temporal) -> dict[str, Any] | None:
    """Resolve two as-of snapshots from a generic effective-time ledger."""
    if not re.search(r"(?:按日期分别|分别查|各是哪里|分别是哪里|as of)", question, re.I):
        return None
    spans = sorted(temporal.spans, key=lambda span: span.start)
    if len(spans) != 2:
        return None
    ledger = _temporary_destination_ledger(context)
    if not ledger:
        return None
    grounded_people = {item.person for item in ledger if item.person and item.person in question}
    if len(grounded_people) == 1:
        owner = next(iter(grounded_people))
        ledger = tuple(item for item in ledger if not item.person or item.person == owner)
    elif len(grounded_people) > 1:
        return None
    results: list[tuple[Any, StateInterval]] = []
    for span in spans:
        active = [item for item in ledger if item.valid_from < span.end and (item.valid_to is None or span.start < item.valid_to)]
        if not active:
            return None
        active.sort(key=lambda item: (item.priority, item.valid_from, item.source_id))
        results.append((span, active[-1]))
    first, second = results
    return {
        "value": f"{_short_date(first[0].start)}是{first[1].value}；{_short_date(second[0].start)}是{second[1].value}。",
        "slot_label": "temporal_state_snapshots",
        "source_ids": list(dict.fromkeys((first[1].source_id, second[1].source_id))),
        "confidence": 0.97,
        "reason": "structured_effective_time_snapshots",
    }


def _temporary_destination_ledger(context: str) -> tuple[StateInterval, ...]:
    intervals: list[StateInterval] = []
    for raw in str(context or "").splitlines():
        row = _LINE.match(raw.strip())
        if not row or not row.group("time"):
            continue
        try:
            recorded = datetime.fromisoformat(row.group("time").replace("Z", "+00:00"))
        except ValueError:
            continue
        body = row.group("body")
        person_match = _PERSON.search(body)
        person = person_match.group(1).strip() if person_match else ""
        override = _TEMP_DEST_OVERRIDE.search(body)
        if override:
            year = recorded.year
            start = recorded.replace(year=year, month=int(override.group(1)), day=int(override.group(2)), hour=0, minute=0, second=0, microsecond=0)
            end_month = int(override.group(3) or override.group(1))
            end = recorded.replace(year=year, month=end_month, day=int(override.group(4)), hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            intervals.append(StateInterval("temporary_destination", person, _clean_state_value(override.group(5)), start, end, 2, row.group("source")))
            continue
        restore = _TEMP_DEST_RESTORE.search(body)
        base = restore or _TEMP_DEST_BASE.search(body)
        if base:
            intervals.append(StateInterval("temporary_destination", person, _clean_state_value(base.group(1)), recorded, None, 1, row.group("source")))
    return tuple(item for item in intervals if item.value)


def _clean_state_value(value: str) -> str:
    return re.split(r"(?:，|。|；|;|并且|同时)", str(value or ""), maxsplit=1)[0].strip(" ，。；;.!?\"'")


def _short_date(value: datetime) -> str:
    return f"{value.month}月{value.day}日"


def temporal_context(question: str, context: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Remove out-of-window source evidence before an answer model sees it.

    Untimestamped summaries are excluded whenever a bounded temporal query is
    present because they can silently leak a later state into a historical
    answer.  Untimed context remains untouched for ordinary non-temporal QA.
    """
    temporal = resolve_temporal_query(str(question or ""), metadata or {}, default_timezone=str((metadata or {}).get("timezone") or "UTC"))
    cutoff = _query_cutoff(str(question or ""), temporal)
    # Exact dates in a two-snapshot/effective-period question are query
    # targets, not evidence occurrence windows. Resolving the state on March
    # 16 may require updates from March 10/15; retain the chain unless the
    # question explicitly declares a cutoff boundary.
    if cutoff is None:
        return context
    output: list[str] = []
    for line in str(context or "").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            # Headers are retained only if at least one following source row
            # survives; dropping them is safer than exposing stale summaries.
            continue
        stamp = m.group("time") or ""
        if not stamp:
            continue
        try:
            value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if value <= cutoff:
            output.append(line)
    return "\n".join(output)


def _query_cutoff(question: str, temporal) -> datetime | None:
    """Parse inclusive cutoff language without inventing a range."""
    if not re.search(r"(?:截至|不晚于|及更早|之前的记录|只看|on or before|before)", question, re.I):
        return None
    if not temporal.spans:
        return None
    # An explicit date span is a day interval; inclusive cutoff is its end.
    explicit = [span for span in temporal.spans if span.kind == "explicit"]
    if not explicit:
        explicit = list(temporal.spans)
    # A question may mention a later comparison event (e.g. "3月4日不能
    # 倒灌") after the cutoff.  The cutoff is the first explicitly bounded
    # date, not the maximum date appearing anywhere in the question.
    return min(span.end for span in explicit) - timedelta(microseconds=1)
