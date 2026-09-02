export interface CockpitTemporalResolution {
  expression: string;
  localDate: string;
  dayPart: "day" | "morning" | "noon" | "afternoon" | "evening";
  localRange: string;
}

export interface CockpitTemporalQuery {
  active: boolean;
  timezone: string;
  requestTime: string;
  requestLocalDate: string;
  resolutions: CockpitTemporalResolution[];
  retrievalText: string;
  evidenceEnvelope: string;
}

interface RelativePattern {
  regex: RegExp;
  dayDelta: number;
}

interface MatchSpan {
  start: number;
  end: number;
}

const RELATIVE_PATTERNS: RelativePattern[] = [
  { regex: /前天(?:早上|上午|中午|下午|晚上|夜里|夜间)?/g, dayDelta: -2 },
  { regex: /昨(?:天(?:早上|上午|中午|下午|晚上|夜里|夜间)?|晚)/g, dayDelta: -1 },
  { regex: /今天(?:早上|上午|中午|下午|晚上|夜里|夜间)?/g, dayDelta: 0 },
  { regex: /明天(?:早上|上午|中午|下午|晚上|夜里|夜间)?/g, dayDelta: 1 },
  { regex: /后天(?:早上|上午|中午|下午|晚上|夜里|夜间)?/g, dayDelta: 2 },
  { regex: /\bday before yesterday(?:\s+(?:morning|noon|afternoon|evening|night))?\b/gi, dayDelta: -2 },
  { regex: /\byesterday(?:\s+(?:morning|noon|afternoon|evening|night))?\b/gi, dayDelta: -1 },
  { regex: /\blast night\b/gi, dayDelta: -1 },
  { regex: /\btoday(?:\s+(?:morning|noon|afternoon|evening|night))?\b/gi, dayDelta: 0 },
  { regex: /\bthis (?:morning|afternoon|evening)\b/gi, dayDelta: 0 },
  { regex: /\bday after tomorrow(?:\s+(?:morning|noon|afternoon|evening|night))?\b/gi, dayDelta: 2 },
  { regex: /\btomorrow(?:\s+(?:morning|noon|afternoon|evening|night))?\b/gi, dayDelta: 1 },
];

const PART_RANGES = {
  day: "00:00–24:00",
  morning: "06:00–12:00",
  noon: "11:00–14:00",
  afternoon: "12:00–18:00",
  evening: "18:00–24:00",
} as const;

function safeTimezone(value: string): string {
  const timezone = value.trim() || "UTC";
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: timezone }).format(0);
    return timezone;
  } catch {
    return "UTC";
  }
}

function localCalendarDate(referenceTime: Date, timezone: string): Date {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(referenceTime);
  const values = Object.fromEntries(
    parts.filter((part) => part.type !== "literal")
      .map((part) => [part.type, Number(part.value)]),
  );
  return new Date(Date.UTC(values.year, values.month - 1, values.day));
}

function shiftDate(base: Date, days: number): string {
  const shifted = new Date(base.getTime());
  shifted.setUTCDate(shifted.getUTCDate() + days);
  return shifted.toISOString().slice(0, 10);
}

function dayPart(expression: string): CockpitTemporalResolution["dayPart"] {
  const folded = expression.toLowerCase();
  if (/早上|上午|morning/.test(folded)) return "morning";
  if (/中午|noon/.test(folded)) return "noon";
  if (/下午|afternoon/.test(folded)) return "afternoon";
  if (/晚上|夜里|夜间|昨晚|晚$|evening|night/.test(folded)) return "evening";
  return "day";
}

/**
 * Resolve only explicit, bounded relative-day phrases. Expressions such as
 * “last time” remain unresolved because they require memory retrieval rather
 * than calendar arithmetic.
 */
export function normalizeCockpitTemporalQuery(
  text: string,
  options: { requestTime?: Date; timezone?: string } = {},
): CockpitTemporalQuery {
  const suppliedRequestTime = options.requestTime;
  const requestTime = suppliedRequestTime && Number.isFinite(suppliedRequestTime.getTime())
    ? suppliedRequestTime
    : new Date();
  const timezone = safeTimezone(options.timezone ?? "UTC");
  const base = localCalendarDate(requestTime, timezone);
  const requestLocalDate = shiftDate(base, 0);
  const matchedSpans: MatchSpan[] = [];
  const matchedResolutions: Array<{
    sourceIndex: number;
    resolution: CockpitTemporalResolution;
  }> = [];

  for (const pattern of RELATIVE_PATTERNS) {
    pattern.regex.lastIndex = 0;
    for (const match of text.matchAll(pattern.regex)) {
      const expression = match[0];
      const start = match.index ?? 0;
      const end = start + expression.length;
      // Longer phrases are listed before their suffixes. Reject any overlap so
      // "tomorrow" is not resolved a second time inside "day after tomorrow".
      if (matchedSpans.some((span) => start < span.end && end > span.start)) continue;
      matchedSpans.push({ start, end });
      const part = dayPart(expression);
      const localDate = shiftDate(base, pattern.dayDelta);
      matchedResolutions.push({
        sourceIndex: start,
        resolution: {
          expression,
          localDate,
          dayPart: part,
          localRange: `${localDate} ${PART_RANGES[part]}`,
        },
      });
    }
  }
  const resolutions = matchedResolutions
    .sort((left, right) => left.sourceIndex - right.sourceIndex)
    .map((item) => item.resolution);

  const requestTimeIso = requestTime.toISOString();
  if (resolutions.length === 0) {
    return {
      active: false,
      timezone,
      requestTime: requestTimeIso,
      requestLocalDate,
      resolutions,
      retrievalText: text,
      evidenceEnvelope: "",
    };
  }

  const normalized = resolutions.map((item) =>
    `${JSON.stringify(item.expression)}=${item.localRange} (${item.dayPart})`
  ).join("; ");
  const retrievalHint = [
    `normalized_query_time: ${normalized}`,
    `request_local_date: ${requestLocalDate}`,
    `timezone: ${timezone}`,
  ].join("; ");
  const evidenceEnvelope = [
    `<memory-query-time request_time=${JSON.stringify(requestTimeIso)} timezone=${JSON.stringify(timezone)}>`,
    "Relative expressions are normalized against this request, not memory ingestion time:",
    ...resolutions.map((item) =>
      `- ${JSON.stringify(item.expression)} → ${item.localRange} (${item.dayPart})`
    ),
    "Use mentioned_at for when something was said and activity time for when the event occurred.",
    "</memory-query-time>",
  ].join("\n");

  return {
    active: true,
    timezone,
    requestTime: requestTimeIso,
    requestLocalDate,
    resolutions,
    retrievalText: `${text}\n[${retrievalHint}]`,
    evidenceEnvelope,
  };
}
