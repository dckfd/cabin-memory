import {
  compileChineseCockpitSemantics,
  rejectsChineseCollapsedFinalValueQuery,
} from "./cockpit-chinese-semantics.js";

export interface CockpitQueryResolution {
  expression: string;
  localDate: string;
  dayPart: "day" | "morning" | "noon" | "afternoon" | "evening";
  localRange: string;
  dayDelta: number;
}

export interface PreparedCockpitQuery {
  shouldSearchMemory: boolean;
  shouldInject: boolean;
  reasons: string[];
  timezone: string;
  requestTime: string;
  requestLocalDate: string;
  resolutions: CockpitQueryResolution[];
  retrievalText: string;
  timeEnvelope: string;
}

interface RelativePattern {
  regex: RegExp;
  dayDelta: number;
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

// Attribute schemas are compiled independently from benchmark entities. The
// query side intentionally accepts colloquial identifier names (工牌数字、出入
// 卡、出入证); evidence matching remains narrower in the retrieval layer so an
// office address or a generic music category cannot satisfy the requested field.
const CHINESE_PERSONAL_ATTRIBUTE_QUERY = /(?:喜欢谁唱歌|谁(?:的歌|唱(?:的)?歌|(?:来)?演唱)|哪位(?:歌手|艺人|演唱者|主唱)|偏爱谁演唱)|(?:歌手(?:姓名|名字)?|艺人(?:姓名|名字)?|演唱者(?:姓名|名字)?|主唱(?:姓名|名字)?|具体歌手|谁(?:的歌|唱(?:的)?歌|(?:来)?演唱)|喜欢谁(?:唱歌|演唱)|偏爱谁演唱|门禁(?:卡号|编号|号码)?|门卡(?:编号|号码|卡号|ID|标识)?|工牌(?:编号|号码|ID|标识)?|通行(?:卡|证)(?:号|编号|号码|卡号|ID|标识)?|出入(?:卡|证)(?:号|编号|号码|卡号|ID|标识)?|办公(?:证件|凭证)(?:号|编号|号码|ID|标识)?|(?:公司|工作|办公(?:楼)?)(?:门禁卡|门卡|工牌|通行卡|通行证|出入卡|出入证|证件|凭证).{0,10}(?:号|编号|号码|卡号|ID|标识|数字|值)?|卡号).{0,48}(?:哪|谁|什么|多少|具体|有没有|能否|能不能|是否|名字|姓名|编号|号码|ID|标识|数字|准确|最爱|最常|偏爱|不要猜|不要编|没保存|未保存|已记录|未知|据实|查询|查一下|帮我查|请核实|请核对|核实|核对|无法确定|证据|覆盖|字段|缺失|拒答|吗|呢)|(?:哪|谁|什么|多少|具体|有没有|能否|能不能|是否|最爱|最常|偏爱|仅凭|只查|查询|查一下|帮我查|请核实|请核对|核实|核对|已记录|记录了|准确|未保存|未知).{0,48}(?:歌手(?:姓名|名字)?|艺人(?:姓名|名字)?|演唱者(?:姓名|名字)?|主唱(?:姓名|名字)?|具体歌手|谁(?:的歌|唱(?:的)?歌|(?:来)?演唱)|喜欢谁(?:唱歌|演唱)|偏爱谁演唱|门禁(?:卡号|编号|号码)?|门卡(?:编号|号码|卡号|ID|标识)?|工牌(?:编号|号码|ID|标识)?|通行(?:卡|证)(?:号|编号|号码|卡号|ID|标识)?|出入(?:卡|证)(?:号|编号|号码|卡号|ID|标识)?|办公(?:证件|凭证)(?:号|编号|号码|ID|标识)?|(?:公司|工作|办公(?:楼)?)(?:门禁卡|门卡|工牌|通行卡|通行证|出入卡|出入证|证件|凭证).{0,10}(?:号|编号|号码|卡号|ID|标识|数字|值)?|卡号)/iu;
const CHINESE_PERSONAL_IDENTIFIER_FIELD = /(?:(?:门禁卡|门卡|工牌|通行卡|通行证|出入卡|出入证|办公(?:证件|凭证)|(?:公司|工作|办公(?:楼)?)(?:门禁卡|门卡|工牌|通行卡|通行证|出入卡|出入证|证件|凭证)).{0,10}(?:号|编号|号码|卡号|ID|标识|数字|值)|卡号)/iu;
const CHINESE_EXPLICIT_FIELD_REQUEST = /查询|查|返回|给出|读取|找出|提供|核实|核对|多少|什么|准确|具体|已有记录|已记录|未保存|未知|不能|不要|不得|不可|拒答|无法确定/iu;
const CHINESE_COLLOQUIAL_OFFICE_CARD = /办公卡.{0,12}(?:编号|号码|卡号|ID|标识|序号|数字|值)/iu;
const CHINESE_EMPLOYEE_ACCESS_CARD = /(?:员工|单位|工作|公司|办公(?:楼)?)(?:用|的)?(?:门禁卡|门卡|工牌|通行卡|通行证|出入卡|出入证|出入凭证|证件|凭证|身份卡|卡)(?:的)?.{0,12}(?:号|编号|号码|卡号|ID|标识|序号|数字|值)/iu;
const CHINESE_SINGER_FIELD_QUERY = /(?:喜欢|偏爱|常听|爱听|最爱|最常听)(?:的)?.{0,10}(?:谁唱|谁演唱|哪位唱|哪名唱|歌手(?:人名|姓名|名字)?)|(?:谁唱|谁演唱).{0,10}(?:喜欢|偏爱|常听)|(?:唱的人|演唱的人|演唱者是谁)|(?:喜欢|偏爱)(?:的)?歌手(?:人名|姓名|名字)?/iu;

const MEMORY_CUES: Array<{ reason: string; regex: RegExp }> = [
  {
    reason: "history-reference",
    regex: /上次|上回|之前|以前|曾经|刚才|刚刚|最近|过去|历史|那次|那回|先前|记得|还记得|回忆|查一下记录|找一下记录|记录中|记录里|已有记录|现存记录|现有对话|已有对话|车载对话|记忆中|记忆里|已存记忆|存储记忆|车机记忆|现有记忆|历史记忆|现有历史|已经保存|已保存|已记录|记录了|明确存过|存过|留存|历史没保存|只依据.{0,16}(?:记忆|记录)|(?:记忆|记录|证据).{0,20}(?:没覆盖|未覆盖|不足|缺失)|请核实.{0,24}(?:记忆|记录|证据|已存|存过)/iu,
  },
  {
    reason: "elliptical-reference",
    regex: /照旧|照之前|跟之前一样|和上次一样|还是(?:那个|那里|原来|之前)|继续(?:播放|导航|刚才|上次)|恢复(?:播放|导航)?|同一个|老地方/iu,
  },
  {
    reason: "profile-reference",
    regex: /(?:我的|我)(?:偏好|习惯|常用|常去|喜欢|不喜欢|默认|通常)|(?:偏好|习惯|常用|默认)(?:是什么|哪个|哪条)?/iu,
  },
  {
    reason: "profile-reference",
    regex: /(?:目前|当前|现在).{0,32}(?:规则|偏好|约束|策略|别名|资料)|(?:规则|偏好|约束|策略|别名).{0,16}(?:是什么|如何|是否|仍然|优先|分别|各自)/iu,
  },
  {
    reason: "history-reference",
    regex: /\b(?:remember|previously|before|last time|earlier|recently|history|we discussed|i told you)\b/iu,
  },
  {
    reason: "elliptical-reference",
    regex: /\b(?:same as before|same one|that one|there again|as usual|continue|resume|usual route)\b/iu,
  },
  {
    reason: "profile-reference",
    regex: /\b(?:my preferences?|my usual|my favorite|what do i usually|where do i usually)\b/iu,
  },
  {
    reason: "profile-reference",
    regex: /\b(?:current|present|existing)\b.{0,32}\b(?:profile|policy|rule|preferences?|constraints?)\b|\b(?:profile|policy|rule|constraints?)\b.{0,24}\b(?:currently|now|after)\b/iu,
  },
  {
    // Explicit subject/attribute questions are historical even without words
    // such as "remember" or "previously".  Missing this shape made unknown
    // fields look like current-world questions and bypassed the recall/refusal
    // contract entirely (for example, "Which school does Emma attend?").
    reason: "subject-attribute-query",
    regex: /(?:驾驶员|副驾|乘客).{0,36}(?:哪(?:个|家|一)?|什么|多少|是否|几次|偏好|使用|参加|就读)|\b(?:what|which|where|when|how)\b.{0,64}\b(?:does|did|is|are|was|were|has|have)\s+(?:[A-Z][a-z]+|the\s+driver|driver|passenger)\b/iu,
  },
  {
    // Stored-but-possibly-unknown personal attributes must still enter the
    // bounded recall path so absence becomes an explicit abstention. These
    // are reusable attribute schemas, not benchmark people or answer values.
    reason: "subject-attribute-query",
    regex: CHINESE_PERSONAL_ATTRIBUTE_QUERY,
  },
  {
    // “办公卡” is a natural spoken shortening of an office access card. Keep
    // it in the attribute schema so a missing identifier fails closed instead
    // of bypassing recall merely because “门禁/门卡” was omitted.
    reason: "subject-attribute-query",
    regex: CHINESE_COLLOQUIAL_OFFICE_CARD,
  },
  {
    // “员工通行卡” is another productive spoken compound. It denotes the
    // same identifier field as a work access card, not a generic workplace.
    reason: "subject-attribute-query",
    regex: CHINESE_EMPLOYEE_ACCESS_CARD,
  },
  {
    // Spoken questions often omit the noun “歌手” and ask “喜欢谁唱”. The
    // requested field is still a named performer and must fail closed when a
    // memory contains only a genre.
    reason: "subject-attribute-query",
    regex: CHINESE_SINGER_FIELD_QUERY,
  },
  {
    reason: "aggregation-frequency",
    // Chinese “最多/最高” is ambiguous with a scalar cap (for example,
    // “音量最多开到多少”). Chinese event aggregation is compiled with
    // domain and event-scope constraints below; this cue remains English-only.
    regex: /\b(?:most often|most frequent(?:ly)?|how many times|frequency|across .{0,40} recorded)\b/iu,
  },
  {
    reason: "latest-final-update",
    regex: /截至|最新|最终|终版|末版|定版|生效版本|有效版本|现用|最后(?:一次|安排|改到|状态|结果|版本|生效|确认|定稿)|改到|定在|(?:策略|规则|配置|设置|偏好|约束).{0,12}(?:定稿|敲定|最终确认)(?:后|以后)?|(?:定稿|敲定|最终确认)(?:后|以后).{0,48}(?:现在|当前|目前|分别|各自|是什么)|(?:这|上述|以上)[一二两三四五六七八九十\d]+项.{0,16}(?:现在|当前|目前).{0,8}(?:分别|各自)?(?:是什么|如何)|(?:还|仍然)(?:有效|生效|保留|需要|存在|优先|首要)|(?:取消|撤销|撤掉|作废|纠正|更正|恢复|改回|不再).{0,32}(?:了吗|以后|之后|现在|当前|最终|有效|生效)?|(?:目前|当前|现在|现用).{0,32}(?:规则|约束|策略|别名|资料|地址)|\b(?:as of|finally|latest|most recent|ultimately)\b|\b(?:still\s+(?:valid|active|effective|preferred|primary|the\s+priority)|cancel(?:led|ed|ation)?|revok(?:e|ed)|no longer|revert(?:ed)?|restore(?:d)?)\b|\bcurrent\b.{0,32}\b(?:profile|policy|rule|preferences?|constraints?|address)\b|\bafter\b.{0,24}\b(?:update|change|correction|cancellation|revocation)\b/iu,
  },
  {
    reason: "multi-time-comparison",
    regex: /\d{1,2}月\d{1,2}日.{0,100}(?:和|与|以及|对比|相比).{0,50}\d{1,2}月\d{1,2}日|\b(?:on\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b.{0,100}\b(?:and|versus|vs\.?)\s+(?:on\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b/iu,
  },
  {
    reason: "cross-session-synthesis",
    regex: /跨会话|不同会话|多个会话|综合.{0,24}(?:两个人|两人|多人|双方).{0,40}(?:要求|偏好|条件|意见)|\b(?:across|from)\s+(?:different|multiple)\s+(?:sessions?|conversations?)\b|\bcombine.{0,40}(?:both|two|multiple).{0,30}(?:people|drivers?|passengers?).{0,30}(?:requirements?|preferences?|constraints?)\b/iu,
  },
  {
    reason: "cross-session-synthesis",
    regex: /比较.{0,40}(?:驾驶员|副驾|乘客|两个人|两人)|(?:驾驶员|副驾|乘客).{0,40}(?:分别|各自)|\bcompare\b.{0,60}\b(?:with|versus|vs\.?|when)\b|\bbest matches?\b.{0,48}\b(?:constraints?|preferences?|requirements?)\b/iu,
  },
];

const CHINESE_STATE_NOUN = /规则|策略|配置|设置|偏好|约束|状态|安排|预约|别名|地址|地点|路线|高架|收费(?:道路|路段|路|段)|档位|腰托|座椅|温度|音量|音乐|媒体|音频|音响|通知|告警|门禁|门卡|工牌|出入(?:卡|证)|办公(?:证件|凭证)|歌手|主唱|站点|补能点|车位|住处/iu;
const CHINESE_STATE_QUESTION = /是什么|是多少|哪(?:个|位|里|家|条|项)|几(?:点|档|次|度)?|多少|怎么(?:选|走|设置)?|如何|能不能|可不可以|是否|要不要|应当|应该|首要|优先|各自|分别|指向|指哪里|解析到|安排在/iu;
const CHINESE_STATE_MARKER = /现在|如今|当前|目前|现行|现用|现有|生效版本|有效版本|最终|终版|末版|定版|定稿|敲定|(?:确)?定下来|更新|调整|纠正|更正|修改|改动|变更|替换|改期|取消|恢复|当时|那天|彼时|此前|之前|原来|旧(?:的|版)?|平时|通常|默认|仍然|还(?:是|要|在)/iu;

/**
 * Compile common Chinese memory-question syntax into semantic recall risks.
 * This deliberately combines broad linguistic features (time points, state
 * transitions, conditions and person attributes) instead of enumerating
 * benchmark entities or expected answers.
 */
function additionalChineseMemoryReasons(text: string): string[] {
  if (!/[\u3400-\u9fff]/u.test(text)) return [];
  const reasons: string[] = [];
  const semantics = compileChineseCockpitSemantics(text);
  const eventFrequency = semantics.intents.includes("event-frequency");
  const absoluteDates = [...text.matchAll(/(?:\d{4}年)?\d{1,2}月\d{1,2}日/gu)];

  // Any question containing two explicit calendar points is a comparison,
  // regardless of whether they are joined by “和”, “到了”, a comma or a pause.
  if (absoluteDates.length >= 2 || semantics.intents.includes("two-date-state")) {
    reasons.push("multi-time-comparison");
  }

  if (eventFrequency) reasons.push("aggregation-frequency");
  if (semantics.intents.includes("latest-state")) reasons.push("latest-final-update");
  if (semantics.intents.includes("multi-person-state")) reasons.push("cross-session-synthesis");

  const asksState = CHINESE_STATE_NOUN.test(text) && CHINESE_STATE_QUESTION.test(text);
  const rejectsCollapsedFinalValue = rejectsChineseCollapsedFinalValueQuery(text);
  const transition = /(?:更新|调整|纠正|更正|修改|改动|变更|替换|定稿|定版|敲定|(?:确)?定下来|最终确认|改期|取消|撤销|撤掉|恢复|改回|最终版|终版|末版|现行|现用|生效版本)(?:完成|好了|以后|之后|后)?/iu.test(text);
  const conditionalPriority = semantics.intents.includes("conditional-priority");
  const cutoffSnapshot = absoluteDates.length >= 1
    && semantics.intents.includes("cutoff-state");
  if (!eventFrequency
    && ((asksState && ((transition && !rejectsCollapsedFinalValue) || cutoffSnapshot)) || conditionalPriority)) {
    reasons.push("latest-final-update");
  }

  const multiOccupant = semantics.intents.includes("multi-person-state");
  if (multiOccupant) reasons.push("cross-session-synthesis");

  const personalAttribute = CHINESE_PERSONAL_ATTRIBUTE_QUERY.test(text)
    || CHINESE_COLLOQUIAL_OFFICE_CARD.test(text)
    || CHINESE_EMPLOYEE_ACCESS_CARD.test(text)
    || CHINESE_SINGER_FIELD_QUERY.test(text)
    || (CHINESE_PERSONAL_IDENTIFIER_FIELD.test(text) && CHINESE_EXPLICIT_FIELD_REQUEST.test(text))
    || /[\u3400-\u9fff]{1,4}.{0,12}(?:最喜欢|最偏爱|最爱听|最常听|偏好|习惯|常用|温度|住址|地址|档位).{0,16}(?:哪|谁|什么|多少|几|哪里|是否|是|有没有|能否|吗|呢)/iu.test(text);
  if (personalAttribute) reasons.push("subject-attribute-query");

  if (!eventFrequency
    && asksState && CHINESE_STATE_MARKER.test(text) && !multiOccupant && !rejectsCollapsedFinalValue) {
    // `chinese-state-query` is useful diagnostic metadata, while
    // `latest-final-update` is the executable retrieval risk.  Emitting only
    // the former made natural forms such as “方案定下来后，如今……” fall back
    // to a shallow top-k lookup with no update-chain gate.
    reasons.push("chinese-state-query", "latest-final-update");
  }
  if (semantics.intents.includes("latest-state")) reasons.push("chinese-state-query");
  return reasons;
}

const PART_RANGES = {
  day: "00:00–24:00",
  morning: "06:00–12:00",
  noon: "11:00–14:00",
  afternoon: "12:00–18:00",
  evening: "18:00–24:00",
} as const;

function safeTimezone(value: string | undefined, fallback: string): string {
  for (const candidate of [value?.trim(), fallback.trim(), "UTC"]) {
    if (!candidate) continue;
    try {
      new Intl.DateTimeFormat("en-US", { timeZone: candidate }).format(0);
      return candidate;
    } catch {
      // Try the configured fallback, then UTC.
    }
  }
  return "UTC";
}

function safeRequestTime(value: string | Date | undefined): Date {
  const parsed = value instanceof Date ? value : value ? new Date(value) : new Date();
  return Number.isFinite(parsed.getTime()) ? parsed : new Date();
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

function dayPart(expression: string): CockpitQueryResolution["dayPart"] {
  const folded = expression.toLowerCase();
  if (/早上|上午|morning/.test(folded)) return "morning";
  if (/中午|noon/.test(folded)) return "noon";
  if (/下午|afternoon/.test(folded)) return "afternoon";
  if (/晚上|夜里|夜间|昨晚|晚$|evening|night/.test(folded)) return "evening";
  return "day";
}

function resolveRelativeTime(text: string, base: Date): CockpitQueryResolution[] {
  const occupied: Array<{ start: number; end: number }> = [];
  const matches: Array<{ index: number; value: CockpitQueryResolution }> = [];
  for (const pattern of RELATIVE_PATTERNS) {
    pattern.regex.lastIndex = 0;
    for (const match of text.matchAll(pattern.regex)) {
      const expression = match[0];
      const start = match.index ?? 0;
      const end = start + expression.length;
      if (occupied.some((span) => start < span.end && end > span.start)) continue;
      occupied.push({ start, end });
      const part = dayPart(expression);
      const localDate = shiftDate(base, pattern.dayDelta);
      matches.push({
        index: start,
        value: {
          expression,
          localDate,
          dayPart: part,
          localRange: `${localDate} ${PART_RANGES[part]}`,
          dayDelta: pattern.dayDelta,
        },
      });
    }
  }
  return matches.sort((a, b) => a.index - b.index).map((item) => item.value);
}

/**
 * Decide whether a cockpit turn needs active L1 recall and normalize bounded
 * relative dates. This is deterministic and adds no model call.
 */
export function prepareCockpitQuery(
  text: string,
  options: {
    requestTime?: string | Date;
    timezone?: string;
    fallbackTimezone?: string;
  } = {},
): PreparedCockpitQuery {
  const requestTime = safeRequestTime(options.requestTime);
  const timezone = safeTimezone(options.timezone, options.fallbackTimezone ?? "Asia/Shanghai");
  const base = localCalendarDate(requestTime, timezone);
  const requestLocalDate = shiftDate(base, 0);
  const resolutions = resolveRelativeTime(text, base);
  const rejectsCollapsedFinalValue = rejectsChineseCollapsedFinalValueQuery(text);
  const eventFrequency = compileChineseCockpitSemantics(text).intents.includes("event-frequency");
  const reasons = MEMORY_CUES
    .filter((cue) => cue.regex.test(text)
      && !(rejectsCollapsedFinalValue && cue.reason === "latest-final-update")
      && !(eventFrequency && cue.reason === "latest-final-update"))
    .map((cue) => cue.reason);
  reasons.push(...additionalChineseMemoryReasons(text));
  if (resolutions.some((item) => item.dayDelta <= 0)) reasons.push("past-or-current-relative-time");
  const uniqueReasons = [...new Set(reasons)];
  const shouldSearchMemory = uniqueReasons.length > 0;
  const shouldInject = shouldSearchMemory || resolutions.length > 0;
  const normalized = resolutions.map((item) =>
    `${JSON.stringify(item.expression)}=${item.localRange}`
  ).join("; ");
  const retrievalText = normalized
    ? `${text}\n[normalized_query_time: ${normalized}; timezone: ${timezone}]`
    : text;
  const requestTimeIso = requestTime.toISOString();
  const timeEnvelope = resolutions.length === 0 ? "" : [
    `<memory-query-time request_time=${JSON.stringify(requestTimeIso)} timezone=${JSON.stringify(timezone)}>`,
    "相对时间以本轮请求时间为基准，不以记忆入库时间为基准：",
    ...resolutions.map((item) => `- ${JSON.stringify(item.expression)} → ${item.localRange}`),
    "回答历史事实时区分事件发生时间与 mentioned_at（被说出/观察到的时间）。",
    "</memory-query-time>",
  ].join("\n");

  return {
    shouldSearchMemory,
    shouldInject,
    reasons: uniqueReasons,
    timezone,
    requestTime: requestTimeIso,
    requestLocalDate,
    resolutions,
    retrievalText,
    timeEnvelope,
  };
}
