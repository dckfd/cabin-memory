/**
 * Conservative source-side coverage obligations for Chinese cockpit text.
 *
 * These rules never extract a fact value. They only identify a small set of
 * independently queryable slot classes that a Flash extraction pass must
 * cover. The model remains responsible for reading the exact value, operator,
 * person, time and episode from the source text. A missed obligation triggers
 * a bounded, single-source directed Flash compilation before reconciliation;
 * these rules never create or authorize a factual memory themselves.
 */

import type { ConversationMessage } from "../conversation/l0-recorder.js";
import type { ExtractedMemory } from "./l1-writer.js";

export interface CockpitSourceCoverageObligation {
  id: string;
  sourceMessageId: string;
  domain: "navigation" | "selection" | "schedule";
  slot: string;
  constraintTarget?: "ticket" | "per_capita" | "room" | "generic";
  /** Conservative lower bound only; repeated/corrected mentions may collapse. */
  requiredFactCount: number;
  /** High-confidence number of independently bound subjects for a shared predicate. */
  requiredSubjectCount?: number;
  /** High-confidence number of explicit condition bindings for a shared predicate. */
  requiredConditionCount?: number;
  /** High-confidence number of explicit seat-zone bindings for a shared predicate. */
  requiredSeatZoneCount?: number;
  /** High-confidence number of distinct event/effective-time bindings. */
  requiredTemporalCount?: number;
  /** Every governed fact is one independently updateable member of a named state map. */
  requiresStateQualifier?: boolean;
  /** High-confidence number of distinct source-named state-map members. */
  requiredStateQualifierCount?: number;
  /** Multiple structural evidence groups must map to distinct atomic facts. */
  requiresDistinctEvidenceBindings: boolean;
  /** Source offsets are against NFKC-normalized content and never carry values. */
  evidenceGroups: CockpitCoverageEvidenceGroup[];
  /** Forces two blind source-only set compilations even when the lower bound is met. */
  requiresSetAudit: boolean;
  reason: string;
}

export interface CockpitCoverageEvidenceGroup {
  id: string;
  start: number;
  end: number;
  /** Source-level structural event anchor shared across slots from one event. */
  eventAnchor: string;
  /** Exact NFKC source label for a named state-map member; never a fact value. */
  stateQualifier?: string;
  /** Exact NFKC structured speaker/person bound to this source span. */
  subject?: string;
}

export interface CockpitPriorStructuredState {
  record_id?: string;
  metadata: Record<string, unknown>;
}

const NUMBER = String.raw`\d+(?:\.\d+)?`;
const RANGE = String.raw`${NUMBER}(?:\s*[-—~～至到]\s*${NUMBER})?`;
const COMPARATOR = String.raw`(?:以上|以下|以内|之内|至少|至多|不低于|不高于|起|左右|上下)?`;
const PRICE_GAP = (competingTargets: string) =>
  String.raw`(?:(?!(?:${competingTargets}))[^，,。！？!?\n、；;]){0,24}`;

const RATING_AFTER = new RegExp(
  String.raw`评分(?:为|是|设置(?:为|在)?|设为|要(?:在)?|需(?:要)?(?:在)?|要求(?:在)?|最好(?:在)?|达到|至少|不低于|不高于)?\s*${RANGE}\s*分?\s*${COMPARATOR}`,
  "u",
);
const RATING_BEFORE = new RegExp(String.raw`${RANGE}\s*分\s*${COMPARATOR}\s*(?:的)?评分`, "u");
const TICKET_PRICE = new RegExp(
  String.raw`(?:门票|票价)(?:价格|预算|费用)?${PRICE_GAP("人均|每晚|房价|住宿预算|酒店预算")}(?:免费|免票|不要钱|${RANGE}\s*元?\s*${COMPARATOR})`,
  "u",
);
const PER_CAPITA_PRICE = new RegExp(
  String.raw`人均(?:消费|价格|预算|花费|费用)?${PRICE_GAP("门票|票价|每晚|房价|住宿预算|酒店预算")}(?:免费|${RANGE}\s*元\s*${COMPARATOR})`,
  "u",
);
const ROOM_PRICE = new RegExp(
  String.raw`(?:房价|每晚|住宿预算|酒店预算)${PRICE_GAP("门票|票价|人均")}(?:免费|${RANGE}\s*元\s*${COMPARATOR})`,
  "u",
);
const GENERIC_PRICE = new RegExp(
  String.raw`(?:预算|价格|费用|花费)[^，,。！？!?\n、；;]{0,24}(?:免费|${RANGE}\s*元\s*${COMPARATOR})`,
  "u",
);
const DURATION_AFTER = new RegExp(
  String.raw`(?:游玩|停留|观看|用餐|行程|车程|路程)(?:时间|时长)?[^，。！？\n]{0,24}${RANGE}\s*(?:分钟|小时|天)\s*${COMPARATOR}`,
  "u",
);
const DURATION_BEFORE = new RegExp(
  String.raw`${RANGE}\s*(?:分钟|小时|天)\s*${COMPARATOR}[^，。！？\n]{0,16}(?:游玩|停留|观看|用餐|行程|车程|路程)`,
  "u",
);
const SELECTION_INTENT = /(?:推荐|帮我找|(?:要|想|需(?:要)?)找|找个|找一|搜一下|搜索|筛选|选择|想看|想听|想吃|看看|(?:给|替)[^，。！？\n]{1,24}(?:找|推荐))/u;
const CATEGORY_NOUN = /(?:景点|餐馆|饭店|酒店|民宿|电影|电视剧|剧集|歌曲|音乐|节目|充电站|补能点|停车场)/u;
const LOCATION_CONSTRAINT = /(?:周边|附近|就近|离[^，。！？\n]{1,24}(?:公里|千米|米)(?:以内|以下|之内|不超过)?)/u;
const RANKING_POLICY = /(?:(?:按|按照)[^，,；;。！？!?\n]{1,80}(?:排序|优先)|(?:排序|优先级)[^，,；;。！？!?\n]{1,80}(?:按|按照))/u;
const MEDIA_NOUN = /(?:电影|电视剧|剧集|歌曲|音乐|节目|纪录片|动画片)/u;
const RELEASE_PERIOD = /(?:(?:19|20)\d0年代|上世纪(?:[一二三四五六七八九十\d]+)年代|近\s*\d+\s*年)/u;
const APPOINTMENT_ACTIVITY = /(?:预约|改约|约好|车辆检查|检查|年检|保养|维修|体检|复诊|会议|面试|办理|看医生|看病)/u;
const APPOINTMENT_TRANSITION = /(?:(?:检查|保养|年检|维修|体检|复诊)[^。！？\n]{0,32}(?:安排|预约|改约|取消)|(?:安排|预约|改约|取消)[^。！？\n]{0,32}(?:检查|保养|年检|维修|体检|复诊))/u;
const EXPLICIT_TIME = /(?:\d{1,2}月\d{1,2}日|今天|明天|后天|本周|下周|周[一二三四五六日天]|上午|下午|晚上|早上|中午|\d{1,2}(?:[:：]\d{2}|点))/u;
const APPOINTMENT_ACTION_ANCHOR = /(?:预约|改约|约好|安排|取消|计划|打算|准备|定在|订在|排在)/u;
// Detect only an explicit travel phrase whose object has a high-precision
// place suffix. This deliberately rejects phrases such as “去做车辆检查”:
// the rule creates a slot obligation, never a rule-authored place value.
const APPOINTMENT_DESTINATION = /(?:去|到|前往|导航到|开车去)[^，。！？\n]{0,64}?(?:中心|医院|诊所|酒店|宾馆|博物院|博物馆|展馆|场馆|公司|大厦|园区|公园|广场|商场|超市|门店|车行|维修厂|服务站|服务区|停车场|充电站|机场|火车站|高铁站|汽车站|学校|小区|社区|村|镇|县|区|市|路|街|巷|弄|号|北京|上海|天津|重庆)(?:内|里|处|门口|附近)?(?=(?:做|进行|参加|办理|看|检查|保养|维修|年检|体检|复诊|开会|面试|的(?:车辆)?(?:检查|保养|维修|年检|体检|复诊|会议|面试)|[，。！？\n]|$))/u;

// A named destination map has multiple independently updateable members that
// share navigation.destination (for example, a meal place and a rendezvous
// place). Detect only the source structure: a bounded source label ending in
// 地点/目的地 followed by an explicit assignment or update operator. The rule
// neither captures nor emits the value to the right of the operator.
const NAMED_DESTINATION_STATE = /(?:^|[，,。！？!?；;\n:：、】])\s*(?:(?:嗯|呃|那个|请|先|再|然后)\s*)*(?:把|将)?\s*([\p{L}\p{N}._\-/ ]{1,16}(?:的)?(?:目的地|地点))\s*(设置为|设为|更正为|改成|改为|仍然是|仍是|还是|定为|为|是)/gu;
const GENERIC_DESTINATION_STATE_LABELS = new Set([
  "这个地点",
  "那个地点",
  "此地点",
  "当前位置",
  "这个目的地",
  "那个目的地",
  "此目的地",
  "导航目的地",
]);
const DEICTIC_DESTINATION_STATE_LABEL = /^(?:这|那|此|当前|现在|刚才|刚刚|之前|先前|原来|上次|旧)(?:个|次|来的?|有的?|的)?(?:地点|目的地)$/u;

interface CoverageSpan {
  start: number;
  end: number;
}

interface NamedDestinationStateCoverage {
  spans: CoverageSpan[];
  qualifierCount: number;
  qualifiersBySpan: ReadonlyMap<string, string>;
  subjectsBySpan: ReadonlyMap<string, string>;
}

// A generic “X地点” phrase is not necessarily a navigation state: it may be
// an ordinary event/property such as a meeting, workplace or accident site.
// The deterministic obligation activates only for an explicit saved-command
// map with parallel members, or for an exact member already present in live
// structured context. Ordinary model extraction remains available outside
// this high-precision coverage path.
const NAMED_DESTINATION_MAP_CONTEXT = /(?:(?:车机|导航|路线|行程|出行)(?:的)?(?:口令|快捷(?:项|键)?|预设(?:表|项)?|收藏(?:表|项)?|地点表|目的地表|状态表|配置表)|(?:地点|目的地)(?:口令|快捷(?:项|键)?|预设(?:表|项)?|收藏(?:表|项)?|表|状态表|配置表)|(?:记住|保存(?:一下)?)[^，,。！？!?；;\n]{0,20}(?:为|成)(?:(?:车机|导航|路线|行程|出行)(?:的)?)?(?:口令|快捷(?:项|键)?|预设(?:表|项)?|收藏(?:表|项)?))/u;
const DESTINATION_REPRESENTATION_LABEL = /^(?:默认|常用|首选|已保存|保存的|临时|当前|现在|本次|上次|上一个|原|原来|原定|旧|最终|最新|导航)(?:的)?(?:地点|目的地)$/u;

const CLAUSE_BOUNDARIES = new Set(["，", ",", "。", "！", "!", "？", "?", "；", ";", "\n"]);
const CORRECTION_OR_REITERATION = /(?:不对|不是这个|改成|改为|更正|纠正|算了|记住|再说一遍|我是说)/u;
const CONDITIONAL_SCOPE = /(?:默认|如果|若是|若|当.+?时|(?:带|载|电量|油量|天气|路况)[^，。！？\n]{0,16}时|否则|除非)/u;
const DATE_TOKEN = /(?:\d{1,2}月\d{1,2}日|今天|明天|后天|本周|下周|周[一二三四五六日天])/u;
const INFORMATION_QUERY = /(?:请问|想知道|帮我看看|(?:吗|呢)\s*$|怎么办|怎么|如何|是否|是不是|有没有|能否|可否|多少|什么|哪(?:个|些|里|儿)?)/u;
const EXPLICIT_QUERY_PRIORITY = /(?:请问|想知道|是否|是不是|有没有|能否|可否|怎么办|怎么|如何|多少|什么|哪(?:个|些|里|儿)?)/u;
const DIRECT_STATE_CHANGE_REQUEST = /^(?:(?:请|麻烦|帮我|替我|给我|把|将|直接)\s*)+[^，,。！？!?；;\n]{0,64}(?:设置|设为|调到|改成|改为|定为|限制|记住|保存|安排|预约|改约|取消|导航|播放|打开|关闭|找|推荐|搜一下|搜索|筛选|选择)/u;
const DIRECT_PERSON_SELECTION_REQUEST = /^(?:(?:请|麻烦)\s*)?(?:给|替)\s*(?:[\p{Script=Han}]{2,8}|我|你|他|她)\s*(?:找|推荐|搜一下|搜索|筛选|选择)/u;
const QUERY_ONLY_PREFIX = /^(?:请问|想知道|帮我看看|麻烦看看|查一下|查询一下)$/u;
const TRAILING_CONFIRMATION_QUERY = /^(?:对吗|是吗|没错吧|是不是|可以吗|行吗)$/u;
const PERSON_TOKEN_REJECT = /(?:评分|门票|票价|预算|价格|费用|景点|餐馆|酒店|电影|歌曲|距离|充电|排序|优先|要求|设置|推荐|找|元|分|小时|分钟|默认|否则|条件|满足|成人|儿童|老人|学生|婴儿|大人|小孩|\d)/u;
const MAJOR_BOUNDARY = /[。！？!?；;\n]/u;
const RATING_ELLIPTICAL = new RegExp(
  String.raw`(?:，|,|、|否则|以及|或者|或)\s*(?:[\p{Script=Han}]{1,8}\s*)?${RANGE}\s*分\s*${COMPARATOR}`,
  "u",
);
const TICKET_ELLIPTICAL = new RegExp(
  String.raw`(?:，|,|、|否则|以及|或者|或)\s*(?:(?:成人|儿童|老人|学生|婴儿|大人|小孩)\s*)?(?:免费|免票|不要钱|${RANGE}\s*元\s*${COMPARATOR})`,
  "u",
);
const CONDITION_AXIS = /(?:晴天|阴天|雨天|雪天|雾天|下雨时?|下雪时?|白天|夜间|早高峰|晚高峰|工作日|周末|节假日|高温|低温|拥堵|畅通|电量(?:低|高|不足|充足)时?|油量(?:低|高|不足|充足)时?|带孩子时?|独自时?|多人时?)/u;
const SEAT_ZONE_AXIS = /(?:主驾|驾驶位|副驾|副驾驶|前排|后排|后座|左后座|右后座|左侧|右侧)/u;
const NON_PERSON_BINDING = /(?:晴天|阴天|雨天|雪天|雾天|白天|夜间|高峰|工作日|周末|节假日|高温|低温|拥堵|畅通|电量|油量|天气|路况|副驾|驾驶位|前排|后排|后座|左侧|右侧)/u;

function findCoverageSpans(text: string, patterns: RegExp[]): CoverageSpan[] {
  const unique = new Map<string, CoverageSpan>();
  for (const pattern of patterns) {
    const flags = [...new Set(`${pattern.flags.replace(/[gy]/gu, "")}gu`.split(""))].join("");
    const matcher = new RegExp(pattern.source, flags);
    for (const match of text.matchAll(matcher)) {
      if (match.index === undefined || match[0].length === 0) continue;
      const span = { start: match.index, end: match.index + match[0].length };
      unique.set(`${span.start}:${span.end}`, span);
    }
  }
  return [...unique.values()].sort((left, right) => left.start - right.start || left.end - right.end);
}

function mergeTimeSpansWithinClause(text: string, spans: CoverageSpan[]): CoverageSpan[] {
  const merged: CoverageSpan[] = [];
  for (const span of spans) {
    const previous = merged.at(-1);
    if (previous) {
      const between = text.slice(previous.end, span.start);
      const previousText = text.slice(previous.start, previous.end);
      const currentText = text.slice(span.start, span.end);
      const startsAnotherDatedEvent = DATE_TOKEN.test(previousText)
        && DATE_TOKEN.test(currentText)
        && !/^\s*(?:至|到|-|—|~|～)\s*$/u.test(between);
      const explicitEventSeparator = /(?:、|以及)/u.test(between);
      if (between.length <= 12
        && !startsAnotherDatedEvent
        && !explicitEventSeparator
        && ![...between].some((character) => CLAUSE_BOUNDARIES.has(character))) {
        previous.end = span.end;
        continue;
      }
    }
    merged.push({ ...span });
  }
  return merged;
}

function clauseBounds(text: string, position: number): CoverageSpan {
  let start = position;
  while (start > 0 && !CLAUSE_BOUNDARIES.has(text[start - 1])) start -= 1;
  let end = position;
  while (end < text.length && !CLAUSE_BOUNDARIES.has(text[end])) end += 1;
  if (end < text.length) end += 1;
  return { start, end };
}

/**
 * True only when an utterance is wholly informational. This is deliberately
 * speech-act-only: it never reads a fact value. A direct mutation command such
 * as “请把温度设为24度，可以吗” remains a command, while explicit capability,
 * confirmation and fact questions remain non-persistent even with polite words.
 */
export function isCockpitPureInformationalQuery(text: string): boolean {
  const normalized = text.normalize("NFKC")
    .replace(/^\s*(?:\[[^\]\r\n]{1,256}\]\s*)*/u, "")
    .replace(/^\s*【[^】\r\n]{1,64}】\s*/u, "")
    .trim();
  if (!normalized) return false;
  const hasQuestionPunctuation = /[？?]/u.test(normalized);
  const segments = normalized
    .split(/(?:以及|另外|同时|[，,、。！？!?；;\n])+/u)
    .map((segment) => segment.trim())
    .filter(Boolean);
  const hasQuerySignal = hasQuestionPunctuation
    || segments.some((segment) => INFORMATION_QUERY.test(segment));
  if (!hasQuerySignal) return false;
  if ((DIRECT_STATE_CHANGE_REQUEST.test(normalized)
      || DIRECT_PERSON_SELECTION_REQUEST.test(normalized))
    && !EXPLICIT_QUERY_PRIORITY.test(normalized)) return false;
  if (segments.length <= 1) return true;
  if (hasQuestionPunctuation
    && !segments.some((segment) => INFORMATION_QUERY.test(segment))) return true;

  const substantive = segments.filter((segment) => !QUERY_ONLY_PREFIX.test(segment));
  if (substantive.length === 0) return true;
  if (substantive.every((segment) => INFORMATION_QUERY.test(segment))) return true;
  return TRAILING_CONFIRMATION_QUERY.test(substantive.at(-1) ?? "");
}

function isAssertiveCoverageSpan(text: string, span: CoverageSpan): boolean {
  const localBounds = clauseBounds(text, span.start);
  const localClause = text.slice(localBounds.start, localBounds.end);
  if (isCockpitPureInformationalQuery(localClause)) return false;
  const bounds = majorSentenceBounds(text, span.start);
  const sentenceEnd = bounds.end < text.length ? bounds.end + 1 : bounds.end;
  const clause = text.slice(bounds.start, sentenceEnd);
  // A polite prefix ("请问" / "帮我看看") does not turn a factual question
  // into an asserted preference. The deterministic detector only creates hard
  // obligations for non-question clauses; the general model path remains free
  // to answer or classify ambiguous speech acts without persisting them.
  return !isCockpitPureInformationalQuery(clause);
}

function filterAssertiveCoverageSpans(text: string, spans: CoverageSpan[]): CoverageSpan[] {
  return spans.filter((span) => isAssertiveCoverageSpan(text, span));
}

function precedingBaseInMajorSegment(
  text: string,
  span: CoverageSpan,
  baseSpans: CoverageSpan[],
): boolean {
  return baseSpans.some((base) => {
    if (base.end > span.start || span.start - base.end > 80) return false;
    return !MAJOR_BOUNDARY.test(text.slice(base.end, span.start));
  });
}

function appendEllipticalSpans(
  text: string,
  baseSpans: CoverageSpan[],
  pattern: RegExp,
): CoverageSpan[] {
  if (baseSpans.length === 0) return [];
  const additions = findCoverageSpans(text, [pattern]).filter((span) =>
    precedingBaseInMajorSegment(text, span, baseSpans)
      && !baseSpans.some((base) => spanOverlaps(base, span))
  );
  return [...baseSpans, ...additions]
    .filter((entry, index, all) =>
      all.findIndex((candidate) => candidate.start === entry.start && candidate.end === entry.end) === index
    )
    .sort((left, right) => left.start - right.start || left.end - right.end);
}

function coverageScopeText(text: string, spans: CoverageSpan[]): string {
  if (spans.length === 0) return "";
  const start = Math.min(...spans.map((span) => clauseBounds(text, span.start).start));
  const end = Math.max(...spans.map((span) => clauseBounds(text, span.start).end));
  return text.slice(start, end);
}

function coverageScopeBounds(text: string, spans: CoverageSpan[]): CoverageSpan[] {
  return spans.map((span) => structuralSegmentBounds(text, span.start))
    .filter((entry, index, all) =>
      all.findIndex((candidate) => candidate.start === entry.start && candidate.end === entry.end) === index
    );
}

function spanOverlaps(left: CoverageSpan, right: CoverageSpan): boolean {
  return left.start < right.end && right.start < left.end;
}

function spansWithinCoverageScope(
  text: string,
  ownerSpans: CoverageSpan[],
  pattern: RegExp,
): CoverageSpan[] {
  const bounds = coverageScopeBounds(text, ownerSpans);
  return findCoverageSpans(text, [pattern]).filter((candidate) =>
    bounds.some((scope) => candidate.start >= scope.start && candidate.end <= scope.end)
  );
}

function temporalAxisSpans(text: string, ownerSpans: CoverageSpan[]): CoverageSpan[] {
  const bounds = coverageScopeBounds(text, ownerSpans);
  return mergeTimeSpansWithinClause(text, findCoverageSpans(text, [EXPLICIT_TIME]))
    .filter((candidate) => bounds.some((scope) =>
      candidate.start >= scope.start && candidate.end <= scope.end
    ));
}

function structuralSegmentBounds(text: string, position: number): CoverageSpan {
  const separators = /(?:以及|[，,、。！？!?；;\n])/gu;
  let start = 0;
  let end = text.length;
  for (const match of text.matchAll(separators)) {
    if (match.index === undefined) continue;
    const separatorStart = match.index;
    const separatorEnd = separatorStart + match[0].length;
    if (separatorEnd <= position) {
      start = separatorEnd;
      continue;
    }
    end = separatorStart;
    break;
  }
  return { start, end };
}

function majorSentenceBounds(text: string, position: number): CoverageSpan {
  const majorBoundaries = new Set(["。", "！", "!", "？", "?", "；", ";", "\n"]);
  let start = position;
  while (start > 0 && !majorBoundaries.has(text[start - 1])) start -= 1;
  let end = position;
  while (end < text.length && !majorBoundaries.has(text[end])) end += 1;
  return { start, end };
}

/**
 * Derive one source-level event anchor without reading or authoring a fact
 * value. Explicit temporal occurrences are the strongest anchors. Otherwise
 * use the nearest structural segment. All slot obligations call this same
 * function, so content/time/destination facts from one source event converge.
 */
function sourceEventAnchor(text: string, span: CoverageSpan): string {
  const temporal = mergeTimeSpansWithinClause(text, findCoverageSpans(text, [EXPLICIT_TIME]));
  const containing = temporal.find((candidate) => spanOverlaps(candidate, span));
  if (containing) return `temporal:${containing.start}:${containing.end}`;
  const sentence = majorSentenceBounds(text, span.start);
  const inSentence = temporal.filter((candidate) =>
    candidate.start >= sentence.start && candidate.end <= sentence.end
  );
  const preceding = inSentence.filter((candidate) => candidate.end <= span.start).at(-1);
  if (preceding) return `temporal:${preceding.start}:${preceding.end}`;
  const following = inSentence.find((candidate) => candidate.start >= span.end);
  if (following) return `temporal:${following.start}:${following.end}`;
  const segment = structuralSegmentBounds(text, span.start);
  return `segment:${segment.start}:${segment.end}`;
}

function distinctAxisCount(
  text: string,
  ownerSpans: CoverageSpan[],
  pattern: RegExp,
): number | undefined {
  const count = new Set(spansWithinCoverageScope(text, ownerSpans, pattern).map((span) =>
    text.slice(span.start, span.end).normalize("NFKC").trim().toLocaleLowerCase()
  )).size;
  return count >= 2 ? count : undefined;
}

function isHighConfidenceParticipant(label: string): boolean {
  const normalized = label.normalize("NFKC").trim();
  return /^(?:[\p{Script=Han}]{2,8}|我|你|他|她)$/u.test(normalized)
    && !/[和与、]/u.test(normalized)
    && !PERSON_TOKEN_REJECT.test(normalized)
    && !NON_PERSON_BINDING.test(normalized);
}

function participantLabelsInClause(clause: string): string[] {
  const labels = new Set<string>();
  const add = (label: string | undefined): void => {
    if (!label || !isHighConfidenceParticipant(label)) return;
    labels.add(label.normalize("NFKC").trim().toLocaleLowerCase());
  };

  const tagged = clause.match(/^\s*[【\[]\s*([^】\]]{1,16})\s*[】\]]/u);
  add(tagged?.[1]);

  const directive = /(?:^|以及|同时|另外|[，,、；;])\s*(?:(?:请|麻烦|帮忙)\s*)?(?:给|替)\s*([\p{Script=Han}]{2,8}|我|你|他|她)\s*(?=(?:找|推荐|搜一下|搜索|筛选|选择|设置|设为|调到|改成|改为|定为|限制|记住|保存|安排|预约|改约|取消|导航|播放|打开|关闭))/gu;
  for (const match of clause.matchAll(directive)) add(match[1]);

  const leading = clause.match(
    /^\s*(?:[【\[]\s*)?([\p{Script=Han}]{2,8}|我|你|他|她)(?:\s*[】\]])?\s*(?=(?:要求|评分|门票|票价|预算|价格|费用|\d+(?:\.\d+)?\s*分))/u,
  );
  add(leading?.[1]);

  const marker = clause.search(/(?:都|分别|各自)/u);
  if (marker >= 0) {
    let prefix = clause.slice(0, marker)
      .replace(/[【】\[\]]/gu, "")
      .replace(/^\s*(?:请|麻烦|帮我)?\s*(?:给|替)?\s*/u, "")
      .trim();
    const lastDirective = Math.max(prefix.lastIndexOf("给"), prefix.lastIndexOf("替"));
    if (lastDirective >= 0) prefix = prefix.slice(lastDirective + 1).trim();
    const participants = prefix.split(/(?:、|和|与)/u).map((entry) => entry.trim());
    if (participants.length >= 2 && participants.every(isHighConfidenceParticipant)) {
      for (const participant of participants) add(participant);
    }
  }
  return [...labels];
}

function requiredSubjectCount(text: string, spans: CoverageSpan[]): number | undefined {
  const participants = new Set(spans.flatMap((span) => {
    const bounds = clauseBounds(text, span.start);
    return participantLabelsInClause(text.slice(bounds.start, bounds.end));
  }));
  return participants.size >= 2 ? participants.size : undefined;
}

function selectionCategorySpans(text: string): CoverageSpan[] {
  const selected: CoverageSpan[] = [];
  let clauseStart = 0;
  for (let index = 0; index <= text.length; index += 1) {
    if (index < text.length && !CLAUSE_BOUNDARIES.has(text[index])) continue;
    const clause = text.slice(clauseStart, index);
    if (SELECTION_INTENT.test(clause)) {
      for (const local of findCoverageSpans(clause, [CATEGORY_NOUN])) {
        const category = {
          start: clauseStart + local.start,
          end: clauseStart + local.end,
        };
        // A noun used as a location anchor ("酒店周边的景点") is not itself
        // the selected category. It remains available to the location compiler.
        if (!/^(?:周边|附近)/u.test(text.slice(category.end, category.end + 4))) {
          selected.push(category);
        }
      }
    }
    clauseStart = index + 1;
  }
  return selected;
}

function countPatternMatches(text: string, pattern: RegExp): number {
  return findCoverageSpans(text, [pattern]).length;
}

interface NamedDestinationStateCandidate {
  span: CoverageSpan;
  qualifier: string;
  segment: CoverageSpan;
}

function namedDestinationStateCandidates(text: string): NamedDestinationStateCandidate[] {
  const candidates: NamedDestinationStateCandidate[] = [];
  const matcher = new RegExp(NAMED_DESTINATION_STATE.source, NAMED_DESTINATION_STATE.flags);
  for (const match of text.matchAll(matcher)) {
    if (match.index === undefined || !match[1]) continue;
    const qualifier = match[1].normalize("NFKC").trim();
    if (!qualifier
      || GENERIC_DESTINATION_STATE_LABELS.has(qualifier)
      || DEICTIC_DESTINATION_STATE_LABEL.test(qualifier)
      || DESTINATION_REPRESENTATION_LABEL.test(qualifier)) continue;
    const localStart = match[0].indexOf(match[1]);
    if (localStart < 0) continue;
    const span = {
      start: match.index + localStart,
      end: match.index + match[0].length,
    };
    if (!isAssertiveCoverageSpan(text, span)) continue;
    candidates.push({
      span,
      qualifier,
      segment: majorSentenceBounds(text, span.start),
    });
  }
  return candidates;
}

const GENERIC_NAMED_STATE_SUBJECTS = new Set(["user", "用户", "我", "本人", "用户本人"]);
const UNKNOWN_NAMED_STATE_SCOPES = new Set([
  "unknown",
  "unspecified",
  "none",
  "null",
  "n/a",
  "na",
  "未知",
  "不明",
  "未指定",
]);

interface LiveNamedDestinationPrior {
  qualifier: string;
  subject: string;
  baseIdentity: string;
}

function canonicalNamedStateSubject(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.normalize("NFKC").trim();
  if (!normalized) return undefined;
  return GENERIC_NAMED_STATE_SUBJECTS.has(normalized.toLocaleLowerCase())
    ? "user"
    : normalized;
}

function canonicalNamedStateScope(value: unknown): string {
  if (typeof value !== "string") return "<unspecified>";
  const normalized = value.normalize("NFKC").trim();
  if (!normalized || UNKNOWN_NAMED_STATE_SCOPES.has(normalized.toLocaleLowerCase())) {
    return "<unspecified>";
  }
  return normalized;
}

function leadingNamedStateSpeaker(text: string): string | undefined {
  const payload = text.replace(/^\s*(?:\[[^\]\r\n]{1,256}\]\s*)*/u, "");
  const match = /^[^【】\r\n]{0,32}【([^】\r\n]{1,64})】/u.exec(payload);
  return canonicalNamedStateSubject(match?.[1]);
}

function namedStateCandidateSubject(
  message: ConversationMessage,
  candidate: NamedDestinationStateCandidate,
): string {
  const text = message.content.normalize("NFKC");
  const prefix = text.slice(candidate.segment.start, candidate.span.start);
  const localSpeakers = [...prefix.matchAll(/(?:^|[，,、:：])\s*【([^】\r\n]{1,64})】/gu)]
    .map((match) => canonicalNamedStateSubject(match[1]))
    .filter((value): value is string => Boolean(value));
  return localSpeakers.at(-1) ?? leadingNamedStateSpeaker(text) ?? "user";
}

function liveNamedDestinationPriors(
  priors: ReadonlyArray<CockpitPriorStructuredState>,
): LiveNamedDestinationPrior[] {
  const supersededRecordIds = new Set(priors.flatMap((entry) => {
    const supersedes = entry.metadata.supersedes;
    return Array.isArray(supersedes)
      ? supersedes.filter((value): value is string => typeof value === "string" && value.length > 0)
      : [];
  }));
  return priors.flatMap((entry) => {
    const metadata = entry.metadata;
    const relation = typeof metadata.relation === "string"
      ? metadata.relation.normalize("NFKC").trim().toLocaleLowerCase()
      : "";
    const actionStatus = typeof metadata.action_status === "string"
      ? metadata.action_status.normalize("NFKC").trim().toLocaleLowerCase()
      : "";
    if ((entry.record_id && supersededRecordIds.has(entry.record_id))
      || relation === "cancelled" || relation === "negated"
      || actionStatus === "cancelled"
      || metadata.domain !== "navigation" || metadata.slot !== "destination"
      || typeof metadata.state_qualifier !== "string") return [];
    const qualifier = metadata.state_qualifier.normalize("NFKC").trim();
    const subject = canonicalNamedStateSubject(metadata.subject);
    if (!qualifier || !subject) return [];
    return [{
      qualifier,
      subject,
      // Multiple materializations of one state (for example persona plus an
      // atomic representation) collapse here. Distinct vehicle/seat scopes do
      // not, so an unscoped utterance cannot choose one by record order.
      baseIdentity: JSON.stringify([
        subject,
        canonicalNamedStateScope(metadata.occupant_scope),
        canonicalNamedStateScope(metadata.vehicle_scope),
        canonicalNamedStateScope(metadata.seat_zone),
      ]),
    }];
  });
}

function priorAuthorizesNamedDestinationCandidate(
  message: ConversationMessage,
  candidate: NamedDestinationStateCandidate,
  priors: ReadonlyArray<LiveNamedDestinationPrior>,
): boolean {
  const subject = namedStateCandidateSubject(message, candidate);
  const compatibleBaseIdentities = new Set(priors.flatMap((prior) =>
    prior.subject === subject && prior.qualifier === candidate.qualifier
      ? [prior.baseIdentity]
      : []
  ));
  return compatibleBaseIdentities.size === 1;
}

function namedDestinationStateCoverage(
  message: ConversationMessage,
  livePriors: ReadonlyArray<LiveNamedDestinationPrior>,
): NamedDestinationStateCoverage {
  const text = message.content.normalize("NFKC");
  const candidates = namedDestinationStateCandidates(text);
  const candidatesBySegment = new Map<string, NamedDestinationStateCandidate[]>();
  for (const candidate of candidates) {
    const key = `${candidate.segment.start}:${candidate.segment.end}`
      + `:${namedStateCandidateSubject(message, candidate)}`;
    candidatesBySegment.set(key, [...(candidatesBySegment.get(key) ?? []), candidate]);
  }
  const accepted = candidates.filter((entry) => {
    const segmentCandidates = candidatesBySegment.get(
      `${entry.segment.start}:${entry.segment.end}`
        + `:${namedStateCandidateSubject(message, entry)}`,
    ) ?? [];
    const hasParallelSavedMap = new Set(segmentCandidates.map((candidate) => candidate.qualifier)).size >= 2
      && NAMED_DESTINATION_MAP_CONTEXT.test(text.slice(entry.segment.start, entry.segment.end));
    return hasParallelSavedMap
      || priorAuthorizesNamedDestinationCandidate(message, entry, livePriors);
  });
  const qualifiersBySpan = new Map(accepted.map((entry) => [
    `${entry.span.start}:${entry.span.end}`,
    entry.qualifier,
  ]));
  const subjectsBySpan = new Map(accepted.map((entry) => [
    `${entry.span.start}:${entry.span.end}`,
    namedStateCandidateSubject(message, entry),
  ]));
  return {
    spans: accepted.map((entry) => entry.span),
    qualifierCount: new Set(accepted.map((entry) => entry.qualifier)).size,
    qualifiersBySpan,
    subjectsBySpan,
  };
}

/**
 * Return only source labels authorized by the same high-precision map gate
 * used to create hard coverage obligations. A structural `X地点是...` match by
 * itself is deliberately insufficient evidence for a named navigation state.
 */
export function extractCockpitAuthorizedNamedDestinationStateQualifiers(
  message: ConversationMessage,
  priorStructuredMemories: ReadonlyArray<CockpitPriorStructuredState> = [],
  proposedSubject?: unknown,
): string[] {
  const coverage = namedDestinationStateCoverage(
    message,
    liveNamedDestinationPriors(priorStructuredMemories),
  );
  const subject = proposedSubject === undefined
    ? undefined
    : canonicalNamedStateSubject(proposedSubject);
  if (proposedSubject !== undefined && subject === undefined) return [];
  return [...new Set([...coverage.qualifiersBySpan.entries()].flatMap(([span, qualifier]) =>
    subject === undefined || coverage.subjectsBySpan.get(span) === subject
      ? [qualifier]
      : []
  ))];
}

function conservativeFactLowerBound(
  text: string,
  spans: CoverageSpan[],
  bindingCounts: Array<number | undefined> = [],
): number {
  if (spans.length === 0) return 1;
  let lowerBound = spans.length;
  // Corrections and emphatic repetitions may mention a slot repeatedly while
  // expressing only one final fact. Keep occurrence count as a lower bound,
  // never as a rule-authored semantic interpretation.
  if (CORRECTION_OR_REITERATION.test(coverageScopeText(text, spans))) lowerBound = 1;
  const subjectCount = requiredSubjectCount(text, spans);
  if (subjectCount !== undefined) lowerBound = Math.max(lowerBound, subjectCount);
  for (const count of bindingCounts) {
    if (count !== undefined) lowerBound = Math.max(lowerBound, count);
  }
  return Math.max(1, lowerBound);
}

function addObligation(
  obligations: CockpitSourceCoverageObligation[],
  message: ConversationMessage,
  normalizedText: string,
  domain: CockpitSourceCoverageObligation["domain"],
  slot: string,
  reason: string,
  evidenceSpans: CoverageSpan[],
  constraintTarget?: CockpitSourceCoverageObligation["constraintTarget"],
  options: {
    requiresStateQualifier?: boolean;
    requiredStateQualifierCount?: number;
    stateQualifiersBySpan?: ReadonlyMap<string, string>;
    subjectsBySpan?: ReadonlyMap<string, string>;
  } = {},
): void {
  const qualifier = constraintTarget ? `:${constraintTarget}` : "";
  const id = `coverage:${message.id}:${domain}:${slot}${qualifier}`;
  const baseSpans = evidenceSpans.length > 0
    ? evidenceSpans
    : [{ start: 0, end: normalizedText.length }];
  const scopedText = coverageScopeText(normalizedText, baseSpans);
  const isCorrectionOrReiteration = CORRECTION_OR_REITERATION.test(scopedText);
  const temporalSpans = temporalAxisSpans(normalizedText, baseSpans);
  const conditionSpans = spansWithinCoverageScope(normalizedText, baseSpans, CONDITION_AXIS);
  const seatZoneSpans = spansWithinCoverageScope(normalizedText, baseSpans, SEAT_ZONE_AXIS);
  const temporalCount = !isCorrectionOrReiteration && temporalSpans.length >= 2
    ? temporalSpans.length
    : undefined;
  const conditionCount = !isCorrectionOrReiteration
    ? distinctAxisCount(normalizedText, baseSpans, CONDITION_AXIS)
    : undefined;
  const seatZoneCount = !isCorrectionOrReiteration
    ? distinctAxisCount(normalizedText, baseSpans, SEAT_ZONE_AXIS)
    : undefined;
  let normalizedSpans = baseSpans;
  let evidenceUsesExplicitAxis = false;
  if (baseSpans.length === 1 && temporalCount !== undefined) {
    normalizedSpans = temporalSpans;
    evidenceUsesExplicitAxis = true;
  } else if (baseSpans.length === 1 && conditionCount !== undefined) {
    normalizedSpans = conditionSpans;
    evidenceUsesExplicitAxis = true;
  } else if (baseSpans.length === 1 && seatZoneCount !== undefined) {
    normalizedSpans = seatZoneSpans;
    evidenceUsesExplicitAxis = true;
  }
  const evidenceGroups = normalizedSpans.map((span, index) => {
    const stateQualifier = options.stateQualifiersBySpan?.get(`${span.start}:${span.end}`);
    const subject = options.subjectsBySpan?.get(`${span.start}:${span.end}`);
    return {
      id: `${id}:evidence:${index + 1}`,
      start: span.start,
      end: span.end,
      eventAnchor: evidenceUsesExplicitAxis
        ? `axis:${span.start}:${span.end}`
        : sourceEventAnchor(normalizedText, span),
      ...(stateQualifier ? { stateQualifier } : {}),
      ...(subject ? { subject } : {}),
    };
  });
  const subjectCount = requiredSubjectCount(normalizedText, baseSpans);
  const requiredFactCount = conservativeFactLowerBound(normalizedText, normalizedSpans, [
    subjectCount,
    conditionCount,
    seatZoneCount,
    temporalCount,
    options.requiredStateQualifierCount,
  ]);
  const requiresDistinctEvidenceBindings = normalizedSpans.length > 1
    && requiredFactCount >= normalizedSpans.length;
  const requiresSetAudit = normalizedSpans.length > 1
    || subjectCount !== undefined
    || CONDITIONAL_SCOPE.test(scopedText)
    || countPatternMatches(scopedText, DATE_TOKEN) > 1
    || options.requiresStateQualifier === true;
  const existing = obligations.find((entry) => entry.id === id);
  if (existing) {
    const merged = [...existing.evidenceGroups, ...evidenceGroups]
      .filter((entry, index, all) =>
        all.findIndex((candidate) => candidate.start === entry.start && candidate.end === entry.end) === index
      )
      .sort((left, right) => left.start - right.start || left.end - right.end)
      .map((entry, index) => ({ ...entry, id: `${id}:evidence:${index + 1}` }));
    existing.evidenceGroups = merged;
    existing.requiredFactCount = conservativeFactLowerBound(normalizedText, merged, [
      subjectCount,
      conditionCount,
      seatZoneCount,
      temporalCount,
      options.requiredStateQualifierCount,
    ]);
    existing.requiredSubjectCount = requiredSubjectCount(normalizedText, merged);
    existing.requiredConditionCount = conditionCount;
    existing.requiredSeatZoneCount = seatZoneCount;
    existing.requiredTemporalCount = temporalCount;
    existing.requiresStateQualifier = existing.requiresStateQualifier
      || options.requiresStateQualifier;
    existing.requiredStateQualifierCount = Math.max(
      existing.requiredStateQualifierCount ?? 0,
      options.requiredStateQualifierCount ?? 0,
    ) || undefined;
    existing.requiresDistinctEvidenceBindings = merged.length > 1
      && existing.requiredFactCount >= merged.length;
    existing.requiresSetAudit = existing.requiresSetAudit || requiresSetAudit || merged.length > 1;
    return;
  }
  obligations.push({
    id,
    sourceMessageId: message.id,
    domain,
    slot,
    ...(constraintTarget ? { constraintTarget } : {}),
    requiredFactCount,
    ...(subjectCount !== undefined ? { requiredSubjectCount: subjectCount } : {}),
    ...(conditionCount !== undefined ? { requiredConditionCount: conditionCount } : {}),
    ...(seatZoneCount !== undefined ? { requiredSeatZoneCount: seatZoneCount } : {}),
    ...(temporalCount !== undefined ? { requiredTemporalCount: temporalCount } : {}),
    ...(options.requiresStateQualifier ? { requiresStateQualifier: true } : {}),
    ...(options.requiredStateQualifierCount !== undefined
      ? { requiredStateQualifierCount: options.requiredStateQualifierCount }
      : {}),
    requiresDistinctEvidenceBindings,
    evidenceGroups,
    requiresSetAudit,
    reason,
  });
}

/** Detect only high-precision slot classes; never derive or normalize values. */
export function detectCockpitSourceCoverageObligations(
  messages: ConversationMessage[],
  priorStructuredMemories: ReadonlyArray<CockpitPriorStructuredState> = [],
): CockpitSourceCoverageObligation[] {
  const obligations: CockpitSourceCoverageObligation[] = [];
  const priorNamedDestinationStates = liveNamedDestinationPriors(priorStructuredMemories);
  for (const message of messages) {
    if (message.role !== "user") continue;
    const text = message.content.normalize("NFKC");
    const namedDestination = namedDestinationStateCoverage(message, priorNamedDestinationStates);
    if (namedDestination.spans.length > 0 && namedDestination.qualifierCount > 0) {
      addObligation(
        obligations,
        message,
        text,
        "navigation",
        "destination",
        "explicit_named_destination_state",
        namedDestination.spans,
        undefined,
        {
          requiresStateQualifier: true,
          requiredStateQualifierCount: namedDestination.qualifierCount,
          stateQualifiersBySpan: namedDestination.qualifiersBySpan,
          subjectsBySpan: namedDestination.subjectsBySpan,
        },
      );
    }
    const hasSelectionIntent = SELECTION_INTENT.test(text);
    const ratingBaseSpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [RATING_AFTER, RATING_BEFORE]),
    );
    const ratingSpans = appendEllipticalSpans(text, ratingBaseSpans, RATING_ELLIPTICAL);

    if (ratingSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "rating_constraint",
        "explicit_rating_criterion",
        ratingSpans,
      );
    }

    const ticketBaseSpans = filterAssertiveCoverageSpans(text, findCoverageSpans(text, [TICKET_PRICE]));
    const ticketPriceSpans = appendEllipticalSpans(text, ticketBaseSpans, TICKET_ELLIPTICAL);
    if (ticketPriceSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "price_constraint",
        "explicit_ticket_price_criterion",
        ticketPriceSpans,
        "ticket",
      );
    }
    const perCapitaPriceSpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [PER_CAPITA_PRICE]),
    );
    if (perCapitaPriceSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "price_constraint",
        "explicit_per_capita_price_criterion",
        perCapitaPriceSpans,
        "per_capita",
      );
    }
    const roomPriceSpans = filterAssertiveCoverageSpans(text, findCoverageSpans(text, [ROOM_PRICE]));
    if (roomPriceSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "price_constraint",
        "explicit_room_price_criterion",
        roomPriceSpans,
        "room",
      );
    }
    const specificPriceSpans = [
      ...ticketPriceSpans,
      ...perCapitaPriceSpans,
      ...roomPriceSpans,
    ];
    const genericPriceSpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [GENERIC_PRICE]),
    ).filter((genericSpan) =>
      !specificPriceSpans.some((specificSpan) => spanOverlaps(genericSpan, specificSpan))
    );
    if (genericPriceSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "price_constraint",
        "explicit_generic_price_criterion",
        genericPriceSpans,
        "generic",
      );
    }

    const durationSpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [DURATION_AFTER, DURATION_BEFORE]),
    );
    if (durationSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "duration_constraint",
        "explicit_duration_criterion",
        durationSpans,
      );
    }
    const categorySpans = filterAssertiveCoverageSpans(text, selectionCategorySpans(text));
    if (hasSelectionIntent && categorySpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "category_constraint",
        "explicit_selection_category",
        categorySpans,
      );
    }
    const locationSpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [LOCATION_CONSTRAINT]),
    );
    if (hasSelectionIntent && locationSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "location_constraint",
        "explicit_selection_location",
        locationSpans,
      );
    }
    const rankingSpans = filterAssertiveCoverageSpans(text, findCoverageSpans(text, [RANKING_POLICY]));
    if (rankingSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "ranking_policy",
        "explicit_ranking_policy",
        rankingSpans,
      );
    }
    const releasePeriodSpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [RELEASE_PERIOD]),
    );
    if (hasSelectionIntent && MEDIA_NOUN.test(text) && releasePeriodSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "selection",
        "release_period_constraint",
        "explicit_release_period",
        releasePeriodSpans,
      );
    }

    const activitySpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [APPOINTMENT_ACTIVITY]),
    ).filter((span) => {
      const activityText = text.slice(span.start, span.end);
      if (/^(?:检查|维修)$/u.test(activityText)
        && /^(?:中心|站|厂|店|院|所)/u.test(text.slice(span.end, span.end + 3))) {
        return false;
      }
      const bounds = clauseBounds(text, span.start);
      const clause = text.slice(bounds.start, bounds.end);
      return APPOINTMENT_ACTION_ANCHOR.test(clause)
        || (EXPLICIT_TIME.test(clause) && /(?:做|去|进行|参加|办理|看|检查|保养|维修|体检|复诊|开会|面试)/u.test(clause));
    });
    const transitionSpans = filterAssertiveCoverageSpans(
      text,
      findCoverageSpans(text, [APPOINTMENT_TRANSITION]),
    );
    if (activitySpans.length > 0 || transitionSpans.length > 0) {
      addObligation(
        obligations,
        message,
        text,
        "schedule",
        "appointment_content",
        "explicit_appointment_activity",
        activitySpans.length > 0 ? activitySpans : transitionSpans,
      );
      const appointmentOwnerSpans = activitySpans.length > 0 ? activitySpans : transitionSpans;
      const ownerEventAnchors = new Set(appointmentOwnerSpans.map((span) =>
        sourceEventAnchor(text, span)
      ));
      const timeSpans = mergeTimeSpansWithinClause(text, findCoverageSpans(text, [EXPLICIT_TIME]))
        .filter((span) => ownerEventAnchors.has(sourceEventAnchor(text, span)));
      if (timeSpans.length > 0) {
        addObligation(
          obligations,
          message,
          text,
          "schedule",
          "appointment_time",
          "explicit_appointment_time",
          timeSpans,
        );
      }
      const destinationSpans = findCoverageSpans(text, [APPOINTMENT_DESTINATION]);
      if (destinationSpans.length > 0) {
        addObligation(
          obligations,
          message,
          text,
          "navigation",
          "destination",
          "explicit_appointment_destination",
          destinationSpans,
        );
      }
    }
  }
  return obligations;
}

/** Build a non-persistable structural scaffold for the reconciliation ledger. */
export function sourceCoverageObligationToMemory(
  obligation: CockpitSourceCoverageObligation,
): ExtractedMemory {
  return {
    content: `source coverage obligation: ${obligation.domain}.${obligation.slot}`,
    type: "episodic",
    priority: 0,
    scene_name: "cockpit-source-coverage",
    source_message_ids: [obligation.sourceMessageId],
      metadata: {
      domain: obligation.domain,
      slot: obligation.slot,
      ...(obligation.constraintTarget ? { constraint_target: obligation.constraintTarget } : {}),
      coverage_required_fact_count: obligation.requiredFactCount,
      ...(obligation.requiredSubjectCount !== undefined
        ? { coverage_required_subject_count: obligation.requiredSubjectCount }
        : {}),
      ...(obligation.requiredConditionCount !== undefined
        ? { coverage_required_condition_count: obligation.requiredConditionCount }
        : {}),
      ...(obligation.requiredSeatZoneCount !== undefined
        ? { coverage_required_seat_zone_count: obligation.requiredSeatZoneCount }
        : {}),
      ...(obligation.requiredTemporalCount !== undefined
        ? { coverage_required_temporal_count: obligation.requiredTemporalCount }
        : {}),
      ...(obligation.requiresStateQualifier
        ? { coverage_requires_state_qualifier: true }
        : {}),
      ...(obligation.requiredStateQualifierCount !== undefined
        ? { coverage_required_state_qualifier_count: obligation.requiredStateQualifierCount }
        : {}),
      ...(obligation.requiresStateQualifier
        ? {
            coverage_required_state_qualifiers: [...new Set(obligation.evidenceGroups
              .map((group) => group.stateQualifier)
              .filter((qualifier): qualifier is string => Boolean(qualifier)))],
          }
        : {}),
      coverage_requires_distinct_evidence_bindings: obligation.requiresDistinctEvidenceBindings,
      coverage_evidence_group_ids: obligation.evidenceGroups.map((group) => group.id),
      coverage_event_anchors: [...new Set(obligation.evidenceGroups.map((group) => group.eventAnchor))],
      source_message_ids: [obligation.sourceMessageId],
      construction_quality: {
        status: "partial",
        score: 0,
        issues: ["source_coverage_obligation"],
        repairs: [],
        source_count: 1,
        user_source_count: 1,
      },
    },
  };
}
