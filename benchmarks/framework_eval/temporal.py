from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_ANCHOR_KEYS = (
    "query_time",
    "question_time",
    "request_time",
    "asked_at",
    "captured_at",
    "question_date",
)
_NESTED_CONTEXT_KEYS = ("request_context", "query_context", "temporal_context")
_TIMEZONE_KEYS = ("timezone", "time_zone", "tz")

_DAY_PARTS: dict[str, tuple[int, int]] = {
    "凌晨": (0, 6),
    "早上": (6, 12),
    "早晨": (6, 12),
    "上午": (6, 12),
    "中午": (11, 14),
    "下午": (12, 18),
    "傍晚": (17, 20),
    "晚上": (18, 24),
    "夜里": (18, 24),
    "夜间": (18, 24),
    "morning": (6, 12),
    "noon": (11, 14),
    "afternoon": (12, 18),
    "evening": (17, 20),
    "night": (18, 24),
}

_CHINESE_DAY_OFFSETS = {
    "大前天": -3,
    "前天": -2,
    "昨天": -1,
    "昨儿": -1,
    "昨儿个": -1,
    "今天": 0,
    "今儿": 0,
    "今儿个": 0,
    "明天": 1,
    "明儿": 1,
    "明儿个": 1,
    "后天": 2,
    "大后天": 3,
}

_CHINESE_COMPOUNDS = {
    "昨晚": (-1, "晚上"),
    "昨夜": (-1, "晚上"),
    "昨早": (-1, "早上"),
    "今早": (0, "早上"),
    "今晚": (0, "晚上"),
    "明早": (1, "早上"),
    "明晚": (1, "晚上"),
    "前晚": (-2, "晚上"),
    "前夜": (-2, "晚上"),
}

_WEEKDAY_INDEX = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

_ENGLISH_SMALL_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass(frozen=True)
class TemporalSpan:
    """One normalized half-open interval while preserving its raw wording."""

    raw: str
    start: datetime
    end: datetime
    granularity: str
    kind: str = "relative"
    confidence: float = 1.0

    def iso_interval(self) -> str:
        return f"{self.start.isoformat()}/{self.end.isoformat()}"

    def contains(self, value: datetime) -> bool:
        comparable = _coerce_for_comparison(value, self.start)
        return self.start <= comparable < self.end

    def overlaps(self, other: TemporalSpan) -> bool:
        other_start = _coerce_for_comparison(other.start, self.start)
        other_end = _coerce_for_comparison(other.end, self.start)
        return self.start < other_end and other_start < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "granularity": self.granularity,
            "kind": self.kind,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TemporalQuery:
    """Normalized temporal intent anchored to trusted request metadata."""

    anchor: datetime | None
    timezone_name: str
    anchor_source: str
    spans: tuple[TemporalSpan, ...] = ()
    operators: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.spans or self.operators)

    @property
    def relative(self) -> bool:
        return any(span.kind == "relative" for span in self.spans)

    def retrieval_text(self, text: str) -> str:
        """Add compact absolute keys only when a relative intent was resolved."""
        if not self.relative:
            return text
        mappings = "; ".join(
            f"{span.raw}={span.start.date().isoformat()} ({span.granularity})"
            for span in self.spans
            if span.kind == "relative"
        )
        return f"{text}\n[normalized_query_time: {mappings}]"

    def answer_text(self, text: str) -> str:
        """Inject one compact request-time envelope for time-sensitive QA."""
        if not self.relative or self.anchor is None:
            return text
        mappings = "; ".join(
            f'"{span.raw}"={span.iso_interval()}'
            for span in self.spans
            if span.kind == "relative"
        )
        return (
            f"[request_time={self.anchor.isoformat()}; "
            f"timezone={self.timezone_name}]\n"
            f"[resolved_query_time: {mappings}]\n{text}"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "anchor": self.anchor.isoformat() if self.anchor else "",
            "timezone": self.timezone_name,
            "anchor_source": self.anchor_source,
            "spans": [span.to_dict() for span in self.spans],
            "operators": list(self.operators),
        }


def resolve_temporal_query(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    default_timezone: str = "UTC",
) -> TemporalQuery:
    """Resolve explicit and relative time expressions without an LLM call.

    Relative expressions are resolved only when a trusted query/source anchor
    is present.  Ambiguous words such as ``recently`` remain unbounded and are
    represented as ordering operators rather than invented calendar ranges.
    """
    flattened = _flatten_temporal_metadata(metadata or {})
    timezone_name, tzinfo = _metadata_timezone(flattened, default_timezone)
    anchor, anchor_source = _metadata_anchor(flattened, tzinfo)
    value = str(text or "")
    matches: list[tuple[int, int, TemporalSpan]] = []
    quoted_ranges = _quoted_content_ranges(value)

    def available(start: int, end: int) -> bool:
        return not any(start < known_end and known_start < end
                       for known_start, known_end, _ in matches)

    def add(
        start_index: int,
        end_index: int,
        start: datetime,
        end: datetime,
        granularity: str,
        *,
        kind: str = "relative",
        confidence: float = 1.0,
    ) -> None:
        # Relative words inside a quoted utterance are content anchors, not
        # request-time constraints.  For example, in ``last Tuesday ...
        # forecast on "today"`` only ``last Tuesday`` may be resolved against
        # the current question time; historical ``today`` belonged to the
        # original utterance and would otherwise create a false second date.
        if kind == "relative" and _range_is_quoted(
            start_index, end_index, quoted_ranges
        ):
            return
        if end <= start or not available(start_index, end_index):
            return
        matches.append((start_index, end_index, TemporalSpan(
            raw=value[start_index:end_index],
            start=start,
            end=end,
            granularity=granularity,
            kind=kind,
            confidence=confidence,
        )))

    # Explicit dates do not require a query-time anchor.
    for match in re.finditer(
        r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)", value
    ):
        explicit = _safe_datetime(
            int(match.group(1)), int(match.group(2)), int(match.group(3)),
            tzinfo=tzinfo,
        )
        if explicit is not None:
            add(
                match.start(), match.end(), explicit,
                explicit + timedelta(days=1), "day", kind="explicit",
            )

    for match in re.finditer(
        r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})[日号]"
        r"(?:\s*(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里|夜间))?",
        value,
    ):
        if match.group(1) is None and anchor is None:
            continue
        year = int(match.group(1) or anchor.year)
        explicit = _safe_datetime(
            year, int(match.group(2)), int(match.group(3)), tzinfo=tzinfo
        )
        if explicit is None:
            continue
        part = match.group(4)
        start, end = _day_interval(explicit, part)
        add(
            match.start(), match.end(), start, end,
            _part_granularity(part), kind="explicit",
        )

    if anchor is not None:
        anchor_day = anchor.replace(hour=0, minute=0, second=0, microsecond=0)

        compound_pattern = "|".join(
            sorted(map(re.escape, _CHINESE_COMPOUNDS), key=len, reverse=True)
        )
        for match in re.finditer(compound_pattern, value):
            offset, part = _CHINESE_COMPOUNDS[match.group(0)]
            start, end = _day_interval(
                anchor_day + timedelta(days=offset), part
            )
            add(
                match.start(), match.end(), start, end,
                _part_granularity(part),
            )

        day_pattern = "|".join(
            sorted(map(re.escape, _CHINESE_DAY_OFFSETS), key=len, reverse=True)
        )
        for match in re.finditer(
            rf"({day_pattern})(?:\s*(凌晨|早上|早晨|上午|中午|下午|"
            rf"傍晚|晚上|夜里|夜间))?",
            value,
        ):
            phrase, part = match.group(1), match.group(2)
            start, end = _day_interval(
                anchor_day + timedelta(days=_CHINESE_DAY_OFFSETS[phrase]),
                part,
            )
            add(
                match.start(), match.end(), start, end,
                _part_granularity(part),
            )

        for match in re.finditer(r"([一二两三四五六七八九十\d]+)天前", value):
            days = _small_chinese_number(match.group(1))
            if days is None:
                continue
            start = anchor_day - timedelta(days=days)
            add(match.start(), match.end(), start, start + timedelta(days=1), "day")

        previous_monday = anchor_day - timedelta(days=anchor_day.weekday() + 7)
        for match in re.finditer(r"上周([一二三四五六日天])", value):
            day = previous_monday + timedelta(days=_WEEKDAY_INDEX[match.group(1)])
            add(match.start(), match.end(), day, day + timedelta(days=1), "day")
        for match in re.finditer(r"上周末", value):
            saturday = previous_monday + timedelta(days=5)
            add(
                match.start(), match.end(), saturday,
                saturday + timedelta(days=2), "weekend",
            )
        for match in re.finditer(r"上周(?![一二三四五六日天末])", value):
            add(
                match.start(), match.end(), previous_monday,
                previous_monday + timedelta(days=7), "week",
            )

        for match in re.finditer(r"刚才|刚刚|方才", value):
            add(
                match.start(), match.end(), anchor - timedelta(minutes=10),
                anchor, "recent_window", confidence=0.8,
            )

        folded = value.casefold()
        english_patterns: tuple[
            tuple[str, int, str | None, str], ...
        ] = (
            (r"\bday before yesterday\b", -2, None, "day"),
            (r"\blast night\b", -1, "night", "night"),
            (r"\btonight\b", 0, "night", "night"),
            (r"\bthis morning\b", 0, "morning", "morning"),
        )
        for pattern, offset, part, granularity in english_patterns:
            for match in re.finditer(pattern, folded, re.IGNORECASE):
                start, end = _day_interval(
                    anchor_day + timedelta(days=offset), part
                )
                add(match.start(), match.end(), start, end, granularity)

        for match in re.finditer(
            r"\b(yesterday|today|tomorrow)"
            r"(?:\s+(morning|noon|afternoon|evening|night))?\b",
            folded,
            re.IGNORECASE,
        ):
            offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[
                match.group(1).casefold()
            ]
            part = match.group(2).casefold() if match.group(2) else None
            start, end = _day_interval(anchor_day + timedelta(days=offset), part)
            add(
                match.start(), match.end(), start, end,
                _part_granularity(part),
            )

        weekdays = (
            r"monday|mon|tuesday|tues?|wednesday|wed|thursday|thurs?|"
            r"friday|fri|saturday|sat|sunday|sun"
        )
        for match in re.finditer(
            rf"\blast\s+({weekdays})\b", folded, re.IGNORECASE
        ):
            key = match.group(1).casefold().rstrip(".")
            weekday = _WEEKDAY_INDEX[key]
            days_back = (anchor_day.weekday() - weekday) % 7 or 7
            day = anchor_day - timedelta(days=days_back)
            add(match.start(), match.end(), day, day + timedelta(days=1), "day")

        for match in re.finditer(r"\b(?:this\s+)?past\s+weekend\b", folded):
            days_since_sunday = (anchor_day.weekday() - 6) % 7 or 7
            sunday = anchor_day - timedelta(days=days_since_sunday)
            saturday = sunday - timedelta(days=1)
            add(
                match.start(), match.end(), saturday,
                sunday + timedelta(days=1), "weekend",
            )

        for match in re.finditer(r"\blast week\b", folded):
            this_monday = anchor_day - timedelta(days=anchor_day.weekday())
            previous = this_monday - timedelta(days=7)
            add(match.start(), match.end(), previous, this_monday, "week")

        for match in re.finditer(r"\blast year\b", folded):
            start = anchor_day.replace(year=anchor_day.year - 1, month=1, day=1)
            end = start.replace(year=start.year + 1)
            add(match.start(), match.end(), start, end, "year")

        number = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
        for match in re.finditer(rf"\b({number})\s+days?\s+ago\b", folded):
            days = _english_number(match.group(1))
            start = anchor_day - timedelta(days=days)
            add(match.start(), match.end(), start, start + timedelta(days=1), "day")

        for match in re.finditer(rf"\b({number})\s+weeks?\s+ago\b", folded):
            weeks = _english_number(match.group(1))
            start = anchor_day - timedelta(days=7 * weeks)
            end = start + timedelta(days=1)
            add(match.start(), match.end(), start, end, "week")

    operators: list[str] = []
    folded = value.casefold()
    if _has_unquoted_match(
        r"截至|不晚于|及更早|之前的记录|只看|\bon or before\b|\bbefore\b",
        folded,
        quoted_ranges,
    ):
        operators.append("cutoff")
    if _has_unquoted_match(
        r"按日期分别|分别查|各是哪里|各自(?:是|为)|\bas of\b",
        folded,
        quoted_ranges,
    ):
        operators.append("as_of")
    if _has_unquoted_match(
        r"上次|最近一次|最新一次|\blast time\b|\bmost recent\b",
        folded,
        quoted_ranges,
    ):
        operators.append("latest")
    if _has_unquoted_match(
        r"第一次|最早一次|\bfirst time\b|\bearliest\b",
        folded,
        quoted_ranges,
    ):
        operators.append("earliest")

    spans = tuple(
        span for _, _, span in sorted(matches, key=lambda item: item[0])
    )
    return TemporalQuery(
        anchor=anchor,
        timezone_name=timezone_name,
        anchor_source=anchor_source,
        spans=spans,
        operators=tuple(dict.fromkeys(operators)),
    )


def humanize_temporal_span(span: TemporalSpan) -> str:
    if span.granularity == "year":
        return str(span.start.year)
    if (
        span.start.hour == 0
        and span.start.minute == 0
        and span.end == span.start + timedelta(days=1)
    ):
        return _human_date(span.start)
    if span.start.hour == 0 and span.end.hour == 0:
        return f"{_human_date(span.start)} to {_human_date(span.end - timedelta(days=1))}"
    if span.start.date() == (span.end - timedelta(microseconds=1)).date():
        return (
            f"{_human_date(span.start)} "
            f"{span.start.strftime('%H:%M')} to {span.end.strftime('%H:%M')}"
        )
    return (
        f"{_human_date(span.start)} {span.start.strftime('%H:%M')} to "
        f"{_human_date(span.end)} {span.end.strftime('%H:%M')}"
    )


def parse_temporal_timestamp(
    value: str | datetime, *, default_timezone: str = "UTC"
) -> datetime | None:
    _, tzinfo = _resolve_timezone(default_timezone, "UTC")
    return _parse_datetime(value, tzinfo)


def _quoted_content_ranges(value: str) -> tuple[tuple[int, int], ...]:
    """Return half-open content ranges for paired non-apostrophe quotes."""
    ranges: list[tuple[int, int]] = []
    quote_pairs = (("\"", "\""), ("“", "”"), ("「", "」"), ("『", "』"))
    for opening, closing in quote_pairs:
        cursor = 0
        while cursor < len(value):
            start = value.find(opening, cursor)
            if start < 0:
                break
            end = value.find(closing, start + len(opening))
            if end < 0:
                break
            ranges.append((start + len(opening), end))
            cursor = end + len(closing)
    return tuple(sorted(ranges))


def _range_is_quoted(
    start: int, end: int, quoted_ranges: tuple[tuple[int, int], ...]
) -> bool:
    return any(start >= quoted_start and end <= quoted_end
               for quoted_start, quoted_end in quoted_ranges)


def _has_unquoted_match(
    pattern: str,
    value: str,
    quoted_ranges: tuple[tuple[int, int], ...],
) -> bool:
    return any(
        not _range_is_quoted(match.start(), match.end(), quoted_ranges)
        for match in re.finditer(pattern, value)
    )


def _flatten_temporal_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key in _NESTED_CONTEXT_KEYS:
        value = metadata.get(key)
        if isinstance(value, Mapping):
            flattened.update(value)
    flattened.update(metadata)
    return flattened


def _metadata_timezone(
    metadata: Mapping[str, Any], default_timezone: str
) -> tuple[str, timezone | ZoneInfo]:
    requested = next((
        str(metadata.get(key) or "").strip()
        for key in _TIMEZONE_KEYS
        if str(metadata.get(key) or "").strip()
    ), "")
    if not requested:
        # Preserve an explicit UTC offset carried by the trusted anchor. This
        # matters around midnight: coercing +08:00 to UTC before resolving
        # "昨天" can move the anchor onto the previous calendar day.
        for key in _ANCHOR_KEYS:
            raw = str(metadata.get(key) or "").strip()
            if not raw:
                continue
            try:
                aware = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if aware.tzinfo is None or aware.utcoffset() is None:
                continue
            offset = aware.strftime("%z")
            requested = (
                "UTC" if offset == "+0000"
                else f"{offset[:3]}:{offset[3:]}"
            )
            break
    requested = requested or default_timezone
    return _resolve_timezone(requested, default_timezone)


def _resolve_timezone(
    requested: str, fallback: str
) -> tuple[str, timezone | ZoneInfo]:
    value = str(requested or fallback or "UTC").strip()
    if value.upper() in {"UTC", "Z"}:
        return "UTC", timezone.utc
    offset = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", value)
    if offset:
        minutes = int(offset.group(2)) * 60 + int(offset.group(3))
        if offset.group(1) == "-":
            minutes *= -1
        return value, timezone(timedelta(minutes=minutes))
    try:
        return value, ZoneInfo(value)
    except ZoneInfoNotFoundError:
        if value != fallback:
            return _resolve_timezone(fallback, "UTC")
        return "UTC", timezone.utc


def _metadata_anchor(
    metadata: Mapping[str, Any], tzinfo: timezone | ZoneInfo
) -> tuple[datetime | None, str]:
    for key in _ANCHOR_KEYS:
        raw = metadata.get(key)
        if raw in (None, ""):
            continue
        parsed = _parse_datetime(raw, tzinfo)
        if parsed is not None:
            return parsed.astimezone(tzinfo), key
    return None, ""


def _parse_datetime(
    value: Any, tzinfo: timezone | ZoneInfo
) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000.0
        try:
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        cleaned = re.sub(r"\s*\([A-Za-z]{3,9}\)\s*", " ", raw).strip()
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
            for pattern in (
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y/%m/%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y年%m月%d日 %H:%M",
                "%Y年%m月%d日",
            ):
                try:
                    parsed = datetime.strptime(cleaned, pattern)
                    break
                except ValueError:
                    continue
            if parsed is None:
                return None
        if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", cleaned):
            parsed = parsed.replace(hour=12)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tzinfo)
    return parsed.astimezone(tzinfo)


def _safe_datetime(
    year: int, month: int, day: int, *, tzinfo: timezone | ZoneInfo
) -> datetime | None:
    try:
        return datetime(year, month, day, tzinfo=tzinfo)
    except ValueError:
        return None


def _day_interval(
    day: datetime, part: str | None
) -> tuple[datetime, datetime]:
    base = day.replace(hour=0, minute=0, second=0, microsecond=0)
    if not part:
        return base, base + timedelta(days=1)
    start_hour, end_hour = _DAY_PARTS[part.casefold()]
    start = base + timedelta(hours=start_hour)
    end = (
        base + timedelta(days=1)
        if end_hour == 24
        else base + timedelta(hours=end_hour)
    )
    return start, end


def _part_granularity(part: str | None) -> str:
    if not part:
        return "day"
    folded = part.casefold()
    if folded in {"早上", "早晨", "上午", "morning"}:
        return "morning"
    if folded in {"中午", "noon"}:
        return "noon"
    if folded in {"下午", "afternoon"}:
        return "afternoon"
    if folded in {"傍晚", "evening"}:
        return "evening"
    if folded in {"晚上", "夜里", "夜间", "night"}:
        return "night"
    return "early_morning"


def _small_chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if len(value) == 2 and value[0] == "十" and value[1] in digits:
        return 10 + digits[value[1]]
    if len(value) == 2 and value[0] in digits and value[1] == "十":
        return digits[value[0]] * 10
    if len(value) == 1:
        return digits.get(value)
    return None


def _english_number(value: str) -> int:
    return int(value) if value.isdigit() else _ENGLISH_SMALL_NUMBERS[value]


def _coerce_for_comparison(value: datetime, target: datetime) -> datetime:
    if value.tzinfo is None and target.tzinfo is not None:
        return value.replace(tzinfo=target.tzinfo)
    if value.tzinfo is not None and target.tzinfo is None:
        return value.replace(tzinfo=None)
    return value.astimezone(target.tzinfo) if target.tzinfo else value


def _human_date(value: datetime) -> str:
    return f"{value.day} {value.strftime('%B')} {value.year}"
