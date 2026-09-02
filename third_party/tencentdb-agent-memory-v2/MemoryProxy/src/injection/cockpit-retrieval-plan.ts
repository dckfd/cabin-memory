import type { PreparedCockpitQuery } from "./cockpit-query.js";
import {
  compileChineseCockpitSemantics,
  extractChinesePersonTargets,
  extractChineseStateTargets,
  isChineseCutoffStateQuery,
  isChineseConditionalPriorityQuery,
} from "./cockpit-chinese-semantics.js";

export type CockpitRiskKind =
  | "aggregation-frequency"
  | "latest-final-update"
  | "multi-time-comparison"
  | "cross-session-synthesis"
  | "subject-attribute-query";

export interface CockpitRetrievalQuery {
  text: string;
  purpose: "original" | CockpitRiskKind | "date-point";
  targetDate?: string;
}

export interface CockpitRetrievalPlan {
  originalQuery: string;
  highRisk: boolean;
  risks: CockpitRiskKind[];
  queries: CockpitRetrievalQuery[];
  searchL0: boolean;
  perQueryLimit: number;
  maxEvidence: number;
  minEvidence: number;
  requiredDates: string[];
  cutoffDate?: string;
}

export interface CockpitEvidence {
  id: string;
  source: "l1" | "l0";
  content: string;
  score?: number;
  timestamp?: string;
  /** Event/validity time extracted from evidence, never the ingestion clock. */
  eventTime?: string;
  /** When the source assertion/update was observed; distinct from event time. */
  observedAt?: string;
  validFrom?: string;
  validTo?: string;
  type?: string;
  role?: string;
  sessionId?: string;
  version?: number;
  metadata?: Record<string, unknown>;
  sourceMessageIds?: string[];
  matchedTargets?: string[];
  matchedPurposes?: CockpitRetrievalQuery["purpose"][];
  /** True when this evidence preserves the ordered neighboring source turns. */
  isSessionPacket?: boolean;
}

export interface CockpitEvidenceAssessment {
  sufficient: boolean;
  reasons: string[];
  distinctEvidence: number;
  timelinePoints: number;
  coveredDates: string[];
  provenanceGroups: number;
  ownershipGroups: number;
}

export interface CockpitFinalState {
  evidence: CockpitEvidence;
  evidenceChain?: CockpitEvidence[];
  eventTime?: string;
  relation: "asserted" | "cancelled" | "negated" | "updated";
  /** Deterministically projected values for every requested final-state slot. */
  facts: Array<{ label: string; value: string }>;
}

const RISK_ORDER: CockpitRiskKind[] = [
  "aggregation-frequency",
  "latest-final-update",
  "multi-time-comparison",
  "cross-session-synthesis",
  "subject-attribute-query",
];

const ENGLISH_DATE = /\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b/giu;
const DAY_FIRST_ENGLISH_DATE = /\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:,?\s+\d{4})?\b/giu;
const CHINESE_DATE = /(?:\d{4}年)?\d{1,2}月\d{1,2}日/gu;
const ISO_DATE = /\b\d{4}-\d{2}-\d{2}\b/gu;

export function buildCockpitRetrievalPlan(
  query: string,
  prepared: PreparedCockpitQuery,
): CockpitRetrievalPlan {
  const chineseSemantics = compileChineseCockpitSemantics(query);
  const risks = RISK_ORDER.filter((risk) => prepared.reasons.includes(risk));
  const highRisk = risks.length > 0;
  if (!highRisk) {
    return {
      originalQuery: query,
      highRisk: false,
      risks: [],
      queries: [{ text: prepared.retrievalText, purpose: "original" }],
      searchL0: false,
      perQueryLimit: 5,
      maxEvidence: 3,
      minEvidence: 1,
      requiredDates: [],
    };
  }

  const requiredDates = risks.includes("multi-time-comparison")
    ? extractDateMentions(query).slice(0, 2)
    : [];
  const maxQueries = chineseSemantics.intents.includes("multi-person-state")
    ? 5
    : risks.includes("aggregation-frequency")
      ? 4
      : risks.includes("multi-time-comparison") || risks.includes("cross-session-synthesis")
        ? 4
        : 3;
  const queries: CockpitRetrievalQuery[] = [
    { text: prepared.retrievalText, purpose: "original" },
  ];

  // A single embedding for a comparison question commonly returns facts for
  // only the lexically dominant person. Split explicit people into bounded,
  // deterministic searches before the general synthesis expansion so every
  // named subject gets an independent chance to contribute direct evidence.
  if (risks.includes("cross-session-synthesis")) {
    const subjects = extractExplicitPersonTargets(query);
    if (subjects.length >= 2) {
      for (const subject of subjects) {
        if (queries.length >= maxQueries) break;
        const compiledPerson = chineseSemantics.people.find((person) => person.name === subject);
        const scope = containsCjk(query)
          ? `只检索 ${subject} 本人的相关事实，保留人物、属性和值的归属`
          : `retrieve facts specifically about ${subject}; preserve person-to-attribute ownership`;
        // Keep the semantic-search text short and shaped like a first-person
        // preference event. Repeated synonyms and retrieval-scope prose dilute
        // exact owner tokens in the L0 hybrid ranker.
        const roleSchema = compiledPerson?.role === "driver"
          ? "我是驾驶员时"
          : compiledPerson?.role === "front-passenger"
            ? "我是副驾时"
            : compiledPerson?.role === "rear-passenger"
              ? "我是后排乘客时"
              : "乘员";
        queries.push({
          text: containsCjk(query)
            ? `[${subject}] ${roleSchema} 座舱空调偏好`
            : `${subject}\n${query}\n[retrieval_scope: ${scope}]`,
          purpose: "cross-session-synthesis",
        });
      }
    }
  }

  // Natural Chinese wording and stored event prose often share a domain but
  // no exact surface phrase. Add at most two value-free canonical searches so
  // retrieval is driven by the compiled intent, not by benchmark paraphrases.
  for (const canonical of chineseSemantics.canonicalRetrievalQueries) {
    if (queries.length >= maxQueries) break;
    const purpose = risks.includes("aggregation-frequency")
      ? "aggregation-frequency"
      : risks.includes("latest-final-update")
        ? "latest-final-update"
        : "original";
    queries.push({
      text: `${canonical}\n[retrieval_scope: 中文座舱语义规范化检索；保留人物、事件时间、更新关系和字段归属]`,
      purpose,
    });
  }

  const namedTargets = extractCockpitNamedTargets(query);
  for (const date of requiredDates) {
    const scope = containsCjk(query)
      ? `围绕${namedTargets.length > 0 ? `「${namedTargets.join("、")}」` : "问题目标"}检索在 ${date} 当天有效的状态，保留常态定义、临时有效期、结束恢复和后续改回事件`
      : `retrieve evidence specifically valid or stated for ${date}; preserve event time, validity intervals, and later updates`;
    const targetPrefix = namedTargets.length > 0 ? `${namedTargets.join(" ")} ${date}\n` : "";
    queries.push({
      text: `${targetPrefix}${query}\n[retrieval_scope: ${scope}]`,
      purpose: "date-point",
      targetDate: date,
    });
  }

  const zh = containsCjk(query);
  const asksForRecommendationRationale = /(?:为什么|为何|最匹配|最合适|推荐|哪家)|\b(?:why|best\s+match|recommend(?:ed|ation)?)\b/iu.test(query);
  const expansions: Array<{ risk: CockpitRiskKind; instruction: string }> = [
    {
      risk: "aggregation-frequency",
      instruction: zh
        ? "检索每一个不同的相关事件、第一次/第二次/第三次记录、发生日期和地点，返回统计所需的完整事件链"
        : "retrieve every distinct matching event, occurrence, date, and place needed for counting; include first, second, and later records",
    },
    {
      risk: "latest-final-update",
      instruction: zh
        ? "检索完整变更时间线，包括初始状态、改期、更正、取消和最终更新，并保留事件时间"
        : "retrieve the complete change timeline: initial state, reschedules, corrections, cancellations, and final update with event times",
    },
    {
      risk: "multi-time-comparison",
      instruction: zh
        ? `检索${namedTargets.length > 0 ? `「${namedTargets.join("、")}」` : "该目标"}的完整生效链：常态定义、临时有效期起止、结束后恢复或改回`
        : `retrieve the target's complete validity chain: baseline definition, temporary interval boundaries, and the restoration or replacement after that interval`,
    },
    {
      risk: "cross-session-synthesis",
      instruction: asksForRecommendationRationale
        ? zh
          ? "检索所有当前约束、最终选择的餐厅或地点、候选方案，以及为何匹配或不匹配每条约束的明确理由"
          : "retrieve all current constraints, the selected restaurant or place, alternatives, and the explicit rationale explaining why each option matches or violates those constraints"
        : zh
          ? "分别检索每个人和每个会话中的相关要求，保留冲突、来源和会话归属"
          : "retrieve relevant requirements from each distinct person and conversation; preserve conflicts and provenance",
    },
  ];
  for (const expansion of expansions) {
    if (!risks.includes(expansion.risk) || queries.length >= maxQueries) continue;
    queries.push({
      text: `${query}\n[retrieval_scope: ${expansion.instruction}]`,
      purpose: expansion.risk,
    });
  }

  const subjectAttributeOnly = risks.length === 1 && risks[0] === "subject-attribute-query";
  return {
    originalQuery: query,
    highRisk: true,
    risks,
    queries: dedupeQueries(queries).slice(0, maxQueries),
    searchL0: true,
    perQueryLimit: subjectAttributeOnly ? 8 : 12,
    maxEvidence: subjectAttributeOnly ? 3 : 6,
    minEvidence: minimumEvidence(query, risks),
    requiredDates,
    cutoffDate: extractCutoffDate(query, prepared.requestLocalDate),
  };
}

function extractExplicitPersonTargets(query: string): string[] {
  const targets = new Set<string>();
  for (const person of extractChinesePersonTargets(query)) targets.add(person.name);
  for (const match of query.matchAll(/\b([A-Z][a-z]{2,})['’]s\b/g)) targets.add(match[1]);
  for (const match of query.matchAll(/\b(?:when|with|and)\s+([A-Z][a-z]{2,})\b/g)) {
    if (!/^(?:The|What|Which|When|Where|Compare|Across|Give|Based|Please|Could|My)$/u.test(match[1])) {
      targets.add(match[1]);
    }
  }
  for (const match of query.matchAll(/\b([A-Z][a-z]{2,})\s+(?:and|with|versus|vs\.?)\s+([A-Z][a-z]{2,})\b/g)) {
    targets.add(match[1]);
    targets.add(match[2]);
  }
  return [...targets].slice(0, 3);
}

export function mergeCockpitEvidence(
  evidence: CockpitEvidence[],
  plan: CockpitRetrievalPlan,
): CockpitEvidence[] {
  // A structured L1 row may summarize the exact L0 message returned by the
  // supplementary search.  Drop only that direct duplicate when the L1
  // contract is authoritative; retain raw L0 whenever construction was
  // partial so it can still support a conservative fallback.
  const structuredSourceIds = new Set(
    evidence
      .filter((item) => isStructuredCockpitEvidence(item) && isAuthoritativeEvidence(item))
      .flatMap((item) => item.sourceMessageIds ?? metadataStringArray(item, "source_message_ids")),
  );
  const byContent = new Map<string, CockpitEvidence>();
  for (const item of evidence) {
    if (item.source === "l0" && structuredSourceIds.has(item.id)) continue;
    const episodeKey = metadataString(item, "episode_key");
    const stateKey = metadataString(item, "state_key");
    // Text-only dedup used to collapse two genuinely repeated events with the
    // same wording, while episode-only dedup would collapse the atomic slots
    // extracted from one multi-slot turn.  The typed pair preserves both axes.
    const key = isStructuredCockpitEvidence(item)
      ? `structured:${episodeKey ?? item.id}:${stateKey ?? item.id}`
      : normalizeEvidence(item.content) || `${item.source}:${item.id}`;
    const existing = byContent.get(key);
    if (!existing) {
      byContent.set(key, {
        ...item,
        eventTime: extractEvidenceEventTime(item),
        observedAt: extractEvidenceObservedAt(item),
        validFrom: metadataString(item, "valid_from"),
        validTo: metadataString(item, "valid_to"),
        matchedTargets: [...new Set(item.matchedTargets ?? [])],
        matchedPurposes: [...new Set(item.matchedPurposes ?? [])],
      });
      continue;
    }
    const scores = [existing.score, item.score].filter((score): score is number =>
      typeof score === "number" && Number.isFinite(score)
    );
    existing.score = scores.length > 0 ? Math.max(...scores) : undefined;
    // When L1 and L0 contain the same text, retain the source L0 identity and
    // lineage. High-risk ownership/session expansion must not lose the direct
    // record merely because an equivalent atomic summary arrived first.
    if (existing.source === "l1" && item.source === "l0") {
      existing.source = "l0";
      existing.id = item.id;
      existing.sessionId = item.sessionId;
      existing.role = item.role;
      existing.timestamp = item.timestamp ?? existing.timestamp;
      existing.eventTime = extractEvidenceEventTime(item);
    }
    if (!existing.timestamp && item.timestamp) existing.timestamp = item.timestamp;
    if (!existing.sessionId && item.sessionId) existing.sessionId = item.sessionId;
    if (!existing.metadata && item.metadata) existing.metadata = item.metadata;
    if (item.version !== undefined) existing.version = Math.max(existing.version ?? item.version, item.version);
    if (!existing.observedAt) existing.observedAt = extractEvidenceObservedAt(item);
    if (!existing.validFrom) existing.validFrom = metadataString(item, "valid_from");
    if (!existing.validTo) existing.validTo = metadataString(item, "valid_to");
    existing.sourceMessageIds = [...new Set([
      ...(existing.sourceMessageIds ?? []),
      ...(item.sourceMessageIds ?? []),
    ])];
    existing.matchedTargets = [...new Set([
      ...(existing.matchedTargets ?? []),
      ...(item.matchedTargets ?? []),
    ])];
    existing.matchedPurposes = [...new Set([
      ...(existing.matchedPurposes ?? []),
      ...(item.matchedPurposes ?? []),
    ])];
    existing.isSessionPacket = existing.isSessionPacket || item.isSessionPacket;
  }

  let merged = [...byContent.values()]
    .filter((item) => isAtOrBeforeCutoff(item, plan.cutoffDate));
  if (!plan.highRisk) {
    return merged.sort((a, b) => compareScore(b.score, a.score)).slice(0, plan.maxEvidence);
  }
  merged = coalesceL0EventMessages(merged);

  // Backend scores across L1 and L0 are not calibrated. More importantly,
  // truncating by ingestion time or alternating sources can evict an already
  // retrieved update chain. Rank by query-topic evidence first, reserve one
  // slot per event/state transition, and only then fill duplicate/supporting
  // representations.
  const ranked = merged.map((item) => ({
    item,
    topic: evidenceTopicOverlap(item, plan.originalQuery),
    rank: evidenceRelevance(item, plan),
  })).filter(({ topic, item }) =>
    (topic > 0
      || hasExplicitDateTargetMatch(item, plan)
      || hasRiskQuerySupport(item, plan)
      || (plan.risks.includes("latest-final-update") && hasUpdateRelation(item)))
      && evidenceSharesDomainTopic(item, plan.originalQuery)
      && isRiskRelevant(item, plan)
      && (!plan.risks.includes("subject-attribute-query")
        || evidenceMatchesRequestedAttribute(item, plan.originalQuery))
  )
    .sort((a, b) => b.rank - a.rank || compareScore(b.item.score, a.item.score));

  const selected: CockpitEvidence[] = [];
  const selectedIds = new Set<string>();
  const groups = new Set<string>();

  // Comparison answers must preserve ownership. Reserve one direct L0 fact
  // per explicitly named person before generic ranking can fill the budget.
  if (plan.risks.includes("cross-session-synthesis")) {
    const subjects = extractExplicitPersonTargets(plan.originalQuery);
    if (subjects.length >= 2) {
      for (const subject of subjects) {
        const matchesSubject = ({ item }: (typeof ranked)[number]) =>
          evidenceMatchesOwner(item, subject)
          && !selectedIds.has(`${item.source}:${item.id}`);
        const candidate = ranked.find((entry) => entry.item.source === "l0" && matchesSubject(entry) && !entry.item.isSessionPacket)
          ?? ranked.find((entry) => entry.item.source === "l0" && matchesSubject(entry))
          ?? ranked.find((entry) => matchesSubject(entry) && !entry.item.isSessionPacket)
          ?? ranked.find(matchesSubject);
        if (!candidate) continue;
        const key = `${candidate.item.source}:${candidate.item.id}`;
        if (selectedIds.has(key)) continue;
        selected.push(candidate.item);
        selectedIds.add(key);
        groups.add(evidenceGroupKey(candidate.item));
      }
    }
  }

  // A multi-slot final-state question can retrieve the authoritative update
  // yet lose it to several higher-scored partial/history rows. Reserve the
  // smallest set of candidates that covers the greatest number of still
  // uncovered slots before generic ranking. One snapshot may satisfy several
  // slots, so this is a bounded set-cover step rather than one row per label.
  if (plan.risks.includes("latest-final-update")) {
    const targets = extractCockpitNamedTargets(plan.originalQuery);
    const uncovered = new Set(targets);
    while (uncovered.size > 0 && selected.length < plan.maxEvidence) {
      let best: (typeof ranked)[number] | undefined;
      let bestCoverage = 0;
      for (const candidate of ranked) {
        const key = `${candidate.item.source}:${candidate.item.id}`;
        if (selectedIds.has(key)) continue;
        const coverage = [...uncovered].filter((target) =>
          evidenceMatchesNamedTarget(candidate.item, target)
        ).length;
        if (coverage > bestCoverage) {
          best = candidate;
          bestCoverage = coverage;
        }
      }
      if (!best || bestCoverage === 0) break;
      const key = `${best.item.source}:${best.item.id}`;
      selected.push(best.item);
      selectedIds.add(key);
      groups.add(evidenceGroupKey(best.item));
      for (const target of [...uncovered]) {
        if (evidenceMatchesNamedTarget(best.item, target)) uncovered.delete(target);
      }
    }
  }
  for (const candidate of ranked) {
    const group = evidenceGroupKey(candidate.item);
    if (groups.has(group)) continue;
    selected.push(candidate.item);
    selectedIds.add(`${candidate.item.source}:${candidate.item.id}`);
    groups.add(group);
    if (selected.length >= plan.maxEvidence) break;
  }
  // A frequency answer must see one representation per event, never a second
  // user/assistant confirmation of the same event. Other risks may still use
  // supporting L1/L0 representations after their distinct chain is secured.
  if (!plan.risks.includes("aggregation-frequency")) {
    for (const candidate of ranked) {
      if (selected.length >= plan.maxEvidence) break;
      const key = `${candidate.item.source}:${candidate.item.id}`;
      if (selectedIds.has(key)) continue;
      selected.push(candidate.item);
      selectedIds.add(key);
    }
  }

  if (plan.risks.includes("latest-final-update") || plan.risks.includes("multi-time-comparison")) {
      selected.sort((a, b) => compareTimestamp(a.observedAt ?? a.eventTime, b.observedAt ?? b.eventTime) || compareScore(b.score, a.score));
  }
  return selected;
}

export function assessCockpitEvidence(
  plan: CockpitRetrievalPlan,
  evidence: CockpitEvidence[],
): CockpitEvidenceAssessment {
  const usableEvidence = evidence.filter(isAuthoritativeEvidence);
  if (!plan.highRisk) {
    const sufficient = usableEvidence.length >= plan.minEvidence;
    return {
      sufficient,
      reasons: sufficient
        ? []
        : evidence.some(isStructuredCockpitEvidence)
          ? ["structured_construction_contract_incomplete"]
          : ["no_matching_evidence"],
      distinctEvidence: usableEvidence.length,
      timelinePoints: countTimelinePoints(usableEvidence),
      coveredDates: [],
      provenanceGroups: countProvenanceGroups(usableEvidence),
      ownershipGroups: countOwnershipGroups(usableEvidence),
    };
  }

  const reasons: string[] = [];
  const timelinePoints = countTimelinePoints(usableEvidence, plan);
  const dateTargetEvidence = plan.risks.includes("multi-time-comparison")
    ? usableEvidence.filter((item) => evidenceMatchesRequestedDateTarget(item, plan))
    : usableEvidence;
  const coveredDates = plan.requiredDates.filter((date) =>
    dateTargetEvidence.some((item) => evidenceCoversDate(item, date))
  );
  const provenanceGroups = countProvenanceGroups(usableEvidence);
  const ownershipGroups = countOwnershipGroups(usableEvidence);
  const completeStructuredSnapshot = hasCompleteStructuredStateSnapshot(plan, usableEvidence);
  const completeFinalSnapshot = plan.risks.includes("latest-final-update") && (
    usableEvidence.some((item) =>
      isAuthoritativeFinalSnapshot(plan.originalQuery, item.content)
      || (isChineseConditionalPriorityQuery(plan.originalQuery)
        && hasCompleteConditionalPrioritySnapshot(item.content))
    )
    || completeStructuredSnapshot
  );

  if (plan.risks.includes("aggregation-frequency") && timelinePoints < plan.minEvidence) {
    reasons.push(`aggregation_evidence_${timelinePoints}_of_${plan.minEvidence}`);
  }
  if (plan.risks.includes("latest-final-update") && timelinePoints < 2 && !completeFinalSnapshot) {
    reasons.push(`timeline_points_${timelinePoints}_of_2`);
  }
  if (plan.risks.includes("latest-final-update")
    && !usableEvidence.some(hasUpdateRelation)
    && (!completeStructuredSnapshot || requiresExplicitStateTransition(plan.originalQuery))) {
    reasons.push("final_update_relation_missing");
  }
  if (plan.risks.includes("latest-final-update")) {
    const namedTargets = extractCockpitNamedTargets(plan.originalQuery);
    if (namedTargets.length >= 2) {
      // An authoritative, arity-checked ordered snapshot covers the complete
      // requested slot list even when the compact value clause does not repeat
      // every label ("三项当前映射为 A、B、C"). Keep this aligned with final-state
      // projection; accepting it here still requires an exact target/value
      // cardinality and an explicit final/update marker.
      const hasCompleteOrderedSnapshot = usableEvidence.some((item) =>
        isAuthoritativeFinalSnapshot(plan.originalQuery, item.content)
      );
      const coveredTargets = hasCompleteOrderedSnapshot
        ? namedTargets
        : namedTargets.filter((target) =>
            usableEvidence.some((item) => evidenceMatchesNamedTarget(item, target))
          );
      if (coveredTargets.length < namedTargets.length) {
        reasons.push(`named_target_coverage_${coveredTargets.length}_of_${namedTargets.length}`);
      }
    }
  }
  if (plan.risks.includes("multi-time-comparison")) {
    if (plan.requiredDates.length < 2) reasons.push("two_time_points_not_identified");
    for (const date of plan.requiredDates) {
      if (!coveredDates.includes(date)) reasons.push(`missing_date:${date}`);
    }
    if (plan.requiredDates.length >= 2 && !hasTransitionBetweenDates(dateTargetEvidence, plan.requiredDates)) {
      reasons.push("between_dates_transition_missing");
    }
  }
  if (plan.risks.includes("cross-session-synthesis")) {
    const explicitOwners = extractExplicitPersonTargets(plan.originalQuery);
    const coveredOwners = explicitOwners.filter((owner) =>
      usableEvidence.some((item) => evidenceMatchesOwner(item, owner))
    );
    if (explicitOwners.length >= 2) {
      if (coveredOwners.length < explicitOwners.length) {
        reasons.push(`named_owner_coverage_${coveredOwners.length}_of_${explicitOwners.length}`);
      }
    } else if (requiresMultipleOwners(plan.originalQuery) && ownershipGroups < 2) {
      reasons.push(`ownership_coverage_${ownershipGroups}_of_2`);
    }
    if (requiresMultipleSessions(plan.originalQuery) && provenanceGroups < 2) {
      reasons.push(`cross_session_provenance_${provenanceGroups}_of_2`);
    } else if (provenanceGroups < 2 && explicitOwners.length < 2) {
      reasons.push(`cross_session_provenance_${provenanceGroups}_of_2`);
    }
  }
  if (usableEvidence.length === 0) reasons.push("no_matching_evidence");
  const structured = evidence.filter(isStructuredCockpitEvidence);
  if (structured.length > 0 && structured.every((item) => !isAuthoritativeEvidence(item))) {
    reasons.push("structured_construction_contract_incomplete");
  }

  return {
    sufficient: reasons.length === 0,
    reasons: [...new Set(reasons)],
    distinctEvidence: usableEvidence.length,
    timelinePoints,
    coveredDates,
    provenanceGroups,
    ownershipGroups,
  };
}

function requiresMultipleOwners(query: string): boolean {
  return extractExplicitPersonTargets(query).length >= 2
    || /(?:两个人|两人|双方|多人|每个人|分别|各自)|\b(?:both|two|multiple)\s+(?:people|persons?|drivers?|passengers?|occupants?)\b/iu.test(query);
}

function requiresMultipleSessions(query: string): boolean {
  return /(?:跨会话|不同会话|多个会话)|\b(?:across|from)\s+(?:different|multiple)\s+(?:sessions?|conversations?)\b/iu.test(query);
}

export function extractCockpitNamedTargets(query: string): string[] {
  const targets = new Set<string>();
  const chineseStateTargets = extractChineseStateTargets(query);
  for (const target of chineseStateTargets) targets.add(target.label);
  for (const match of query.matchAll(/[‘“]([^’”]{1,32})[’”]/gu)) targets.add(match[1].trim());
  for (const match of query.matchAll(/'([^'\n]{1,32})'/gu)) targets.add(match[1].trim());
  // Appointment lifecycle facets (active slot, cancelled slot, replacement)
  // are one transition contract, not independent entity names. Feeding them
  // through the generic alias coverage gate creates false fourth/partial
  // targets and suppresses the dedicated cancellation projection.
  const lifecycleFacetList = /年检|年审|车检|车辆检查|车辆检验|检修|保养|预约/u.test(query)
    && /取消|撤销|撤掉|撤档|撤项|删除|作废|现存项|追加项|新建|替代|后续|重新(?:预约|排期)|新安排/u.test(query);

  // Chinese cockpit questions frequently name several state slots without
  // quoting them: "路线选择、高架通行和收费道路这三项".  Treat those slots exactly
  // like quoted aliases so coverage and the final answer contract are checked
  // per field instead of accepting one generic timeline record.
  const enumerated = query.match(
    /(?:^|[：:，,;；。])([^：:，,;；。？?]{2,96}?)(?:这|上述|以上)[一二两三四五六七八九十\d]+项/iu,
  )?.[1];
  if (enumerated && !lifecycleFacetList) {
    addChineseEnumeratedTargets(targets, enumerated);
  }

  // Natural spoken Chinese often puts the cardinality after the list
  // (“路线选择、高架、收费路三条规则”), or uses “各自/分别” without saying
  // “这三项”. Compile both forms into the same per-slot answer contract.
  const counted = query.match(
    /(?:^|[：:，,;；。])([^：:，,;；。？?]{2,96}?)[一二两三四五六七八九十\d]+(?:条|个|项)(?:规则|设置|别名|映射|状态|配置)/iu,
  )?.[1];
  // Cardinality before the list (“现有三项映射是什么，依次回答 A、B、C”)
  // must not turn the sentence prefix into a fourth target. A genuine
  // count-after-list form necessarily contains a list delimiter or connector.
  if (counted && !lifecycleFacetList && /[、，,]|(?:以及|并且|和|与|及)/u.test(counted)) {
    addChineseEnumeratedTargets(targets, counted);
  }
  const distributed = query.match(
    /(?:^|[：:，,;；。])([^：:，,;；。？?]{2,96}?)(?:目前|现在|如今|当前)?(?:各自|分别)(?:解析到|指向|对应到|是|为|在哪里|是什么)/iu,
  )?.[1];
  if (distributed && !lifecycleFacetList
    && /[、，,]|(?:以及|并且|和|与|及)/u.test(distributed)) {
    addChineseEnumeratedTargets(targets, distributed);
  }

  // The same semantic slot is often repeated with different spoken surfaces
  // ("家人住所" vs "亲友住处", "最高音量" vs "音量上限"). Collapse those
  // surfaces onto the label already chosen by the Chinese compiler so a
  // three-field request cannot become a false four-field coverage contract.
  return [...new Set([...targets]
    .filter(Boolean)
    .map((target) => canonicalChineseTargetSurface(target, query, chineseStateTargets)))]
    .slice(0, 4);
}

function canonicalChineseTargetSurface(
  target: string,
  query: string,
  compiled: ReturnType<typeof extractChineseStateTargets>,
): string {
  const label = (key: (typeof compiled)[number]["key"]): string | undefined =>
    compiled.find((item) => item.key === key)?.label;
  // Conditional charging policies have three stable feature slots. Spoken
  // Chinese names those slots productively ("远近/就近", "设施/休息室",
  // "口碑/评价/评分"). Normalize only inside a compiled low-energy priority
  // request so the generic named-target gate checks evidence semantics rather
  // than requiring the memory to repeat the user's exact surface wording.
  if (isChineseConditionalPriorityQuery(query)) {
    if (/距离|远近|就近/u.test(target)) return "距离";
    if (/休息室|休息设施|配套设施|设施/u.test(target)) return "休息室";
    if (/评分|评价|口碑|高分/u.test(target)) return "评分";
  }
  if (/固定车位|固定(?:停车|泊车)(?:位|点|处|地点|位置)|(?:惯用|常用|惯常|固定|常去|老)(?:停车|泊车)(?:位|点|处|地点|位置|地方|车位)|常用?停车位|常用车位|惯用车位|惯常车位|停车位|停车位置|停车老(?:位置|地方|地点)|常停(?:点|地点|位置)|常停车(?:的)?(?:地方|地点|位置|点|处)|停车处|(?:停车|泊车)(?:惯用|常用|惯常|固定|常去|老)(?:处|点|地点|位置|地方|车位)|泊车点|车位/u.test(target)) return label("parking-alias") ?? target;
  if (/亲友住处|亲友住所|亲友住址|亲友家中|家人住所|家人住处|家人住址|家人家中|家里人(?:住所|住址|住处|家)|亲友家|亲人家|亲属家|亲戚家|亲属住址|亲属住所|亲属住处|亲戚(?:住所|住址|住处)|亲人住处|家属(?:住所|住址|住处|家)/u.test(target)) return label("relative-home-alias") ?? target;
  if (/诊所|就诊地点|看诊(?:处|地点|位置)|看门诊(?:的)?(?:地方|地点|位置)|门诊(?:地址|位置|地点|点|处)|诊疗(?:点|地点|处)|就医(?:点|地点)|看病(?:点|地点)|挂号(?:处|地点|位置)/u.test(target)) return label("clinic-alias") ?? target;
  if (/曲风|音乐(?:类型|类别|内容|门类)?|播放(?:类型|类别|种类|门类)|(?:新|老|原|旧|现行|当前|现在|现用|先前|原先|被换)(?:内容)?(?:类型|门类)|内容(?:类型|门类|状态|效力|是否有效|还算数)?|被换内容状态|(?:旧|老|原先|以前|先前|原来)(?:播放|音乐|音频|媒体)?(?:类型|类别|门类|曲风|内容)(?:状态|效力|是否有效|还算数|去留)?/u.test(target)
    && /听歌|听啥|放啥|音乐|媒体|音频|音响|播放|夜驾|夜听|夜间|夜里|声音/u.test(query)) return label("music-type") ?? target;
  // Keep answer-contract target normalization aligned with the semantic
  // compiler.  In a comma-separated spoken list, productive noun-first
  // surfaces such as “声音最高值” are added by the generic enumerator after
  // the compiler has already emitted the canonical volume slot.  Missing one
  // of those surfaces here therefore creates a false extra target and makes a
  // complete two-slot media update fail as coverage 2/3.
  if (/(?:音量|声量|声音)(?:上限|限制|封顶|天花板)|(?:最大|最高)(?:音量|声量)|(?:音量|声量|声音)(?:的)?(?:最高|最大)(?:值|档(?:位)?|格(?:位|数)?|上限|限制|多少|几(?:档|格))|(?:音量|声量|声音).{0,8}(?:最多|不能超过|不超过|几格)|最多几格|(?:音量)?(?:最高|最大|封顶)(?:档|档位|格|格位|格数)|(?:开|调)(?:到|至|成)?几格|几格(?:为止|封顶|最多)/u.test(target)) return label("volume-limit") ?? target;
  if (/路线|路径|选路|择路/u.test(target)) return label("route") ?? target;
  if (/高架/u.test(target)) return label("elevated-road") ?? target;
  if (/收费(?:道路|路段|路|段)/u.test(target)) return label("toll-road") ?? target;
  if (/会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰/u.test(target)) return label("meeting-point") ?? target;
  return target;
}

function addChineseEnumeratedTargets(targets: Set<string>, value: string): void {
  for (const raw of value.split(/[、，,]|(?:以及|并且|和|与|及)|(?:”“|’‘)/u)) {
    // Calendar points are answer labels for temporal comparison, not state
    // slots. Treating “10月15日、10月19日的会合点” as two target names broke
    // the date-to-value projection.
    if (/(?:20\d{2}年)?\d{1,2}月\d{1,2}日/u.test(raw)) continue;
    const target = raw
      .replace(/^[‘’“”'"\s]+|[‘’“”'"\s]+$/gu, "")
      .replace(/^(?:请说明|请给出|请列出|需要说明|只需|仅需|只要|只列|仅列|定稿后|更新完后|如今|现在|当前|目前|现行|现有)/u, "")
      .trim();
    // Cardinality/umbrella phrases describe the following slots; they are
    // not a fourth answer slot themselves (for example “两类道路” or “这三个
    // 值”). The semantic compiler already expands these into canonical fields.
    if (/^(?:(?:这|上述|以上))?[一二两三四五六七八九十\d]+(?:类|种|项|个|条)?(?:(?:特殊|例外|指定)?道路(?:处置|政策|约束|规则|状态)?|规则|设置|别名|映射|状态|配置|值|地点|称呼)$/u.test(target)) continue;
    if (target.length >= 2 && target.length <= 24) targets.add(target);
  }
}

function evidenceMatchesTarget(content: string, target: string): boolean {
  const foldedContent = normalizeEvidence(content);
  const foldedTarget = normalizeEvidence(target);
  if (foldedContent.includes(foldedTarget)) return true;
  if (!containsCjk(target)) {
    const terms = target.toLowerCase().match(/[a-z][a-z'-]{2,}/gu) ?? [];
    return terms.length > 0 && terms.every((term) => foldedContent.includes(term));
  }
  if (/(?:路线(?:选择|依据|规则|原则)?|路径(?:选择|依据|规则|原则)?|(?:选路|择路)(?:准则|原则|标准|规则)?)/u.test(target) && /路线.{0,16}选择|选择.{0,16}路线|(?:最终|临时)通勤.{0,20}路线/u.test(content)) return true;
  if (/高架/u.test(target) && /高架/u.test(content)) return true;
  if (/收费(?:道路|路段|路|段)/u.test(target) && /收费(?:道路|路段|路|段)/u.test(content)) return true;
  if (/^(?:距离|远近|就近)$/u.test(target) && /距离|远近|就近/u.test(content)) return true;
  if (/^(?:休息室|休息设施|配套设施|设施)$/u.test(target) && /休息室|休息设施|配套设施|设施/u.test(content)) return true;
  if (/^(?:评分|评价|口碑|高分)$/u.test(target) && /评分|评价|口碑|高分/u.test(content)) return true;
  if (/曲风|音乐(?:类型|类别)?|内容类型|(?:新|原|旧|被换)(?:内容)?类型|被换内容状态/u.test(target) && /播放|音乐|曲风|媒体/u.test(content)) return true;
  if (/音量|最大音量|最高音量|(?:音量)?(?:最高|最大|封顶)(?:档|档位|格|格位|格数)/u.test(target) && /音量(?:上限|限制|最高|最大)?/u.test(content)) return true;
  if (/固定车位/u.test(target) && /固定车位/u.test(content)) return true;
  if (/亲友住处/u.test(target) && /亲友住处/u.test(content)) return true;
  if (/诊所/u.test(target) && /诊所/u.test(content)) return true;
  const compact = target.replace(/[\s\p{P}\p{S}]+/gu, "");
  const grams = new Set<string>();
  for (let index = 0; index + 1 < compact.length; index++) grams.add(compact.slice(index, index + 2));
  const hits = [...grams].filter((gram) => foldedContent.includes(gram)).length;
  return grams.size === 1 ? hits === 1 : hits >= 2;
}

/**
 * Project explicitly named Chinese/English state slots from one authoritative
 * snapshot. This supports both per-slot clauses and an arity-checked ordered
 * list such as “三项当前映射为 A、B、C”. The ordered form is accepted only
 * when its value count exactly equals the requested target count.
 */
export function projectCockpitNamedTargetValues(
  targets: string[],
  content: string,
): Array<{ target: string; value: string }> {
  const cleaned = content
    .replace(/\[memory_episode[^\]]*\]/giu, " ")
    .replace(/\[(?:source_time|source_role|resolved_relative_time)[^\]]*\]/giu, " ");
  const values = new Map<string, string>();
  for (const target of targets) {
    const escaped = target.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const quoted = `[‘’“”'"]?${escaped}[‘’“”'"]?`;
    const directPatterns = [
      new RegExp(`${quoted}\\s*(?:当前|目前|如今|现行|临时|仍然?|继续)?\\s*(?:指|是|为|指定为|设定为|变更为|对应(?:到)?|关联到|改成|改为|解析到|指向|恢复到)\\s*([^，。；;\\n\\[‘’“”'"]{1,96})`, "iu"),
      new RegExp(`${quoted}\\s*(?:resolves?\\s+to|means?|points?\\s+to|is|changed?\\s+to)\\s+([^.!?;\\n\\[‘’“”'"]{1,96})`, "iu"),
    ];
    const reversePatterns = [
      new RegExp(`([^，。；;\\n\\[‘’“”'"]{1,96})\\s*(?:只叫|称作|称为)\\s*${quoted}`, "iu"),
      new RegExp(`([^.!?;\\n\\[‘’“”'"]{1,96})\\s+(?:is\\s+)?(?:called|named)\\s+${quoted}`, "iu"),
    ];
    const direct = directPatterns.map((pattern) => cleaned.match(pattern)?.[1]).find(Boolean);
    const reverse = reversePatterns.map((pattern) => cleaned.match(pattern)?.[1]).find(Boolean);
    const value = cleanProjectedValue(direct ?? reverse ?? "");
    if (value) values.set(target, value);
  }

  // Media updates describe values by transition rather than by the user's
  // surface labels (“曲风/音乐类别”, “最高音量”). Project those synonymous
  // slots explicitly from the authoritative update record.
  const music = cleaned.match(/(?:不再播放|停止播放|停用)\s*[^，。；;\n]{1,48}[，,；;]\s*(?:改为|改成|切换为|换成)\s*([^，。；;\n]{1,48})/u)?.[1]
    ?? cleaned.match(/(?:夜间|夜里).{0,20}(?:播放|改为|改成)\s*([^，。；;\n]{1,48})/u)?.[1];
  const volume = cleaned.match(/音量(?:上限|限制|最高|最大)?\s*(?:仍然?|继续)?\s*(?:保持|设为|设置为|为|是|不超过|不能超过)?\s*(\d+(?:\.\d+)?)/u)?.[1];
  for (const target of targets) {
    if (!values.has(target) && /曲风|音乐(?:类型|类别)?/u.test(target) && music) {
      values.set(target, cleanProjectedValue(music));
    }
    if (!values.has(target) && /音量|最大音量|最高音量/u.test(target) && volume) {
      values.set(target, cleanProjectedValue(volume));
    }
  }

  const ordered = cleaned.match(
    /(?:[一二两三四五六七八九十\d]+项|这些|上述|以上|各项)(?:当前|如今|现行|最终)?(?:的)?(?:映射|对应|解析)?(?:分别|依次)?(?:为|是|到)[：:]?\s*([^。；;\n]{3,240})/iu,
  )?.[1];
  if (ordered) {
    const items = ordered.split(/[、，,]|(?:以及|并且|和|与)/u)
      .map(cleanProjectedValue)
      .filter(Boolean);
    if (items.length === targets.length) {
      targets.forEach((target, index) => {
        if (!values.has(target)) values.set(target, items[index]);
      });
    }
  }
  return targets.flatMap((target) => {
    const value = values.get(target);
    return value ? [{ target, value }] : [];
  });
}

function cleanProjectedValue(value: string): string {
  return value
    .replace(/^(?:以后|当前|目前|如今|the\s+alias\s+)/iu, "")
    .replace(/\s+/gu, " ")
    .replace(/[：:=,，;；。.!?？]+$/gu, "")
    .trim();
}

function extractFinalStateFacts(
  query: string,
  content: string,
): Array<{ label: string; value: string }> {
  const targets = extractCockpitNamedTargets(query);
  if (targets.length < 2) return [];
  const cleaned = content
    .replace(/\[memory_episode[^\]]*\]/giu, " ")
    .replace(/\[(?:source_time|source_role|resolved_relative_time)[^\]]*\]/giu, " ");
  const clauses = cleaned
    .split(/[\n。；;]+/u)
    .map((value) => value.replace(/^(?:\[[^\]]{1,64}\]\s*)+/u, "").trim())
    .filter(Boolean);
  const projected = projectCockpitNamedTargetValues(targets, cleaned);
  if (projected.length === targets.length) {
    return projected.map((item) => ({ label: item.target, value: item.value }));
  }
  const facts: Array<{ label: string; value: string }> = projected
    .map((item) => ({ label: item.target, value: item.value }));
  for (const target of targets) {
    if (facts.some((item) => item.label === target)) continue;
    const candidates = clauses
      .map((clause) => ({
        clause,
        exact: normalizeEvidence(clause).includes(normalizeEvidence(target)) ? 1 : 0,
        matches: evidenceMatchesTarget(clause, target) ? 1 : 0,
      }))
      .filter((candidate) => candidate.matches > 0)
      .sort((left, right) => right.exact - left.exact || right.clause.length - left.clause.length);
    const value = candidates[0]?.clause
      .replace(/^(?:用户(?:（[^）]{0,24}）|\([^)]{0,24}\))?\s*)?(?:于|在)\s*20\d{2}年\d{1,2}月\d{1,2}日(?:早上|上午|中午|下午|晚上)?\s*/u, "")
      .replace(/^(?:更新[：:]?|定稿[：:]?|最终确认[：:]?)/u, "")
      .trim();
    if (value) facts.push({ label: target, value: value.slice(0, 180) });
  }
  return facts;
}

function isAuthoritativeFinalSnapshot(query: string, content: string): boolean {
  if (!/(?:最终|定稿|敲定|最终确认|统一更新|更新(?:夜间)?媒体)|\b(?:final(?:ized)?|authoritative final|final confirmation)\b/iu.test(content)) {
    return false;
  }
  const targets = extractCockpitNamedTargets(query);
  return targets.length >= 2 && extractFinalStateFacts(query, content).length === targets.length;
}

function hasCompleteConditionalPrioritySnapshot(content: string): boolean {
  return /(?:平时|通常|默认).{0,48}(?:优先|首要)/u.test(content)
    && /(?:当|如果)?\s*(?:续航|电量|电池余量).{0,24}\d+(?:\.\d+)?\s*%?.{0,36}(?:优先|首要)/u.test(content)
    && /不再(?:是)?(?:首要|优先)|降为次要/u.test(content);
}

function countOwnershipGroups(evidence: CockpitEvidence[]): number {
  const owners = new Set<string>();
  const ignoredChinese = new Set(["用户", "车辆", "餐厅", "规则", "当前", "最终"]);
  for (const item of evidence) {
    const structuredOwner = metadataString(item, "subject") ?? metadataString(item, "occupant_scope");
    if (structuredOwner) owners.add(`structured:${structuredOwner.toLocaleLowerCase()}`);
    for (const match of item.content.matchAll(/\[([\u3400-\u9fff]{1,4})\]/gu)) {
      if (!ignoredChinese.has(match[1])) owners.add(`person:${match[1]}`);
    }
    for (const match of item.content.matchAll(/(?:驾驶员|副驾|乘客)\s*([\u3400-\u9fff]{1,3})?/gu)) {
      owners.add(match[1] ? `person:${match[1]}` : `role:${match[0].trim()}`);
    }
    for (const match of item.content.matchAll(/([\u3400-\u9fff]{1,3})(?=(?:要求|偏好|希望|需要|选择|不吃|不能吃|必须))/gu)) {
      if (!ignoredChinese.has(match[1])) owners.add(`person:${match[1]}`);
    }
    for (const match of item.content.matchAll(/\b([A-Z][a-z]{2,})(?:'s|’s|\s+(?:requires?|prefers?|wants?|needs?|chose|selected))\b/g)) {
      owners.add(`person:${match[1]}`);
    }
  }
  return owners.size;
}

export function extractDateMentions(text: string): string[] {
  const matches = [
    ...(text.match(CHINESE_DATE) ?? []),
    ...(text.match(ENGLISH_DATE) ?? []),
    ...(text.match(DAY_FIRST_ENGLISH_DATE) ?? []),
    ...(text.match(ISO_DATE) ?? []),
  ];
  return [...new Set(matches.map((value) => value.trim().toLowerCase()))];
}

export function resolveCockpitDatePoints(
  plan: CockpitRetrievalPlan,
  evidence: CockpitEvidence[],
): Array<{ date: string; evidence: CockpitEvidence; basis: "validity-interval" | "latest-effective" }> {
  if (!plan.risks.includes("multi-time-comparison")) return [];
  const targetEvidence = evidence.filter((item) =>
    isAuthoritativeEvidence(item) && evidenceMatchesRequestedDateTarget(item, plan)
  );
  const resolved: Array<{ date: string; evidence: CockpitEvidence; basis: "validity-interval" | "latest-effective" }> = [];
  for (const date of plan.requiredDates) {
    const target = parseLooseDate(date);
    if (target === undefined) continue;
    const intervalMatches = targetEvidence.filter((item) => {
      if (!isAuthoritativeEvidence(item)) return false;
      const validFrom = parseLooseDate(item.validFrom ?? metadataString(item, "valid_from") ?? "");
      const validTo = parseLooseDate(item.validTo ?? metadataString(item, "valid_to") ?? "");
      if (validFrom !== undefined && target >= validFrom && (validTo === undefined || target <= validTo)) return true;
      return extractValidityRanges(item.content).some(({ start, end }) => target >= start && target <= end);
    });
    const interval = latestAtOrBefore(intervalMatches, target);
    if (interval) {
      resolved.push({ date, evidence: interval, basis: "validity-interval" });
      continue;
    }
    const latest = latestAtOrBefore(targetEvidence, target);
    if (latest) resolved.push({ date, evidence: latest, basis: "latest-effective" });
  }
  return resolved;
}

export function resolveCockpitFinalState(
  plan: CockpitRetrievalPlan,
  evidence: CockpitEvidence[],
): CockpitFinalState | undefined {
  if (!plan.risks.includes("latest-final-update")) return undefined;
  const authoritative = evidence.filter(isAuthoritativeEvidence);
  const updates = authoritative.filter(hasUpdateRelation);
  const targets = extractCockpitNamedTargets(plan.originalQuery);

  // New cockpit-state-v1 memories are atomic by slot. Assemble the current
  // view from the newest authoritative record for each requested slot; the
  // legacy prose projection below remains the compatibility path.
  const structuredUpdates = authoritative.filter((item) =>
    isStructuredCockpitEvidence(item) && isStructuredCurrentStateCandidate(item)
  );
  if (structuredUpdates.length > 0) {
    const selected: CockpitEvidence[] = [];
    const facts: Array<{ label: string; value: string }> = [];
    const requested = targets.length > 0 ? targets : [""];
    for (const target of requested) {
      const candidates = structuredUpdates.filter((item) =>
        target === "" || evidenceMatchesNamedTarget(item, target)
      );
      const latest = [...candidates].sort(compareNewestEvidence)[0];
      if (!latest) continue;
      const relation = structuredRelation(latest);
      const value = relation === "cancelled"
        ? "已取消"
        : relation === "negated"
          ? "已否定"
          : getCockpitStructuredValue(latest);
      if (target && value !== undefined) facts.push({ label: target, value });
      selected.push(latest);
    }
    const uniqueSelected = [...new Map(selected.map((item) => [`${item.source}:${item.id}`, item])).values()];
    if (uniqueSelected.length > 0 && (targets.length === 0 || facts.length === targets.length)) {
      const representative = [...uniqueSelected].sort(compareNewestEvidence)[0];
      return {
        evidence: representative,
        evidenceChain: uniqueSelected,
        eventTime: latestEvidenceIsoTime(representative),
        relation: structuredRelation(representative) ?? "asserted",
        facts,
      };
    }
  }

  const completeTargetUpdates = targets.length >= 2
    ? updates.filter((item) => extractFinalStateFacts(plan.originalQuery, item.content).length === targets.length)
    : [];
  // Prefer the newest complete requested-field snapshot when one exists. A
  // newer event from a neighboring domain must not steal the final pointer
  // merely because it has a later timestamp.
  const candidates = completeTargetUpdates.length > 0 ? completeTargetUpdates : updates;
  const selected = [...candidates].sort((left, right) =>
    latestEvidenceTime(right) - latestEvidenceTime(left)
    || extractFinalStateFacts(plan.originalQuery, right.content).length
      - extractFinalStateFacts(plan.originalQuery, left.content).length
    || Number(right.isSessionPacket) - Number(left.isSessionPacket)
    || compareScore(right.score, left.score)
  )[0];
  if (!selected) return undefined;
  const content = selected.content
    .replace(/\[memory_episode[^\]]*\]/giu, " ")
    .replace(/\[(?:source_time|source_role|resolved_relative_time)[^\]]*\]/giu, " ");
  const relation = /(?:cancel(?:led|ed|ation)?|revok(?:e|ed)|void(?:ed)?|取消|撤销|作废|无效)/iu.test(content)
    ? "cancelled"
    : /(?:no longer|not anymore|cease(?:d)?|stop(?:ped)?|不再|不要了|停止|否定)/iu.test(content)
      ? "negated"
      : "updated";
  const eventTime = latestEvidenceIsoTime(selected);
  return {
    evidence: selected,
    eventTime,
    relation,
    facts: extractFinalStateFacts(plan.originalQuery, selected.content),
  };
}

function compareNewestEvidence(left: CockpitEvidence, right: CockpitEvidence): number {
  return latestEvidenceTime(right) - latestEvidenceTime(left)
    || (right.version ?? 0) - (left.version ?? 0)
    || compareScore(right.score, left.score);
}

export function summarizeCockpitEventFrequency(
  evidence: CockpitEvidence[],
): Array<{ label: string; count: number }> {
  const counts = new Map<string, { label: string; count: number }>();
  for (const item of evidence.filter(isAuthoritativeEvidence)) {
    const structuredLabel = metadataString(item, "target")
      ?? (typeof metadataValue(item, "value") === "string" ? metadataValue(item, "value") as string : undefined);
    const labels = structuredLabel && isStructuredCockpitEvidence(item)
      ? [structuredLabel]
      : extractOccurrenceLabels(item.content);
    for (const label of new Set(labels)) {
      const key = label.toLocaleLowerCase();
      const prior = counts.get(key);
      counts.set(key, { label: prior?.label ?? label, count: (prior?.count ?? 0) + 1 });
    }
  }
  return [...counts.values()].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

function latestAtOrBefore(evidence: CockpitEvidence[], target: number): CockpitEvidence | undefined {
  return evidence.map((item) => ({
    item,
    time: parseLooseDate(item.eventTime ?? extractEvidenceEventTime(item) ?? ""),
  }))
    .filter((entry): entry is { item: CockpitEvidence; time: number } =>
      entry.time !== undefined && entry.time <= target
    )
    .sort((a, b) => b.time - a.time)[0]?.item;
}

function extractOccurrenceLabels(content: string): string[] {
  const labels: string[] = [];
  for (const match of content.matchAll(/(?:又?去了?|本次在|在|到|选择|优先选择)\s*([\u3400-\u9fffA-Za-z0-9·' -]{2,28}?(?:超充站|充电站|充能站|能源点|补能中心|补能港))/gu)) {
    labels.push(match[1].trim());
  }
  for (const match of content.matchAll(/(?:本次在|在|于|到|前往)\s*([\u3400-\u9fffA-Za-z0-9·' -]{2,28}?(?:超充站|充电站|充能站|能源点|补能中心|补能港))\s*完成(?:了)?(?:车辆)?(?:充电|补能)/gu)) {
    labels.push(match[1].trim());
  }
  for (const match of content.matchAll(/\b(?:visited|went back to|charged at|at)\s+([A-Z][A-Za-z0-9' -]{1,48}?(?:Charging Hub|Fast Charge|Charging Station))\b/g)) {
    labels.push(match[1].trim());
  }
  return labels;
}

function minimumEvidence(query: string, risks: CockpitRiskKind[]): number {
  if (risks.includes("aggregation-frequency")) {
    const chinese = query.match(/([\d一二两三四五六七八九十]+)\s*(?:个|条|次|笔|回|单|趟|件|项|宗)(?:[^\s，。？?]{0,20})?(?:事件|流水|明细|记录|行程|访问|补能|补电|充电|充能|充完|补完)/u);
    const english = query.match(/\b(\d+|one|two|three|four|five|six)\s+(?:distinct\s+)?(?:recorded\s+)?(?:[a-z-]+\s+){0,2}(?:events?|records?|trips?|visits?|occurrences?)\b/iu);
    const explicit = chinese
      ? parseSmallNumber(chinese[1])
      : english
        ? parseEnglishSmallNumber(english[1])
        : undefined;
    return Math.min(Math.max(explicit ?? 2, 2), 6);
  }
  if (risks.includes("latest-final-update") || risks.includes("cross-session-synthesis")) return 2;
  return 1;
}

function parseEnglishSmallNumber(value: string): number | undefined {
  if (/^\d+$/u.test(value)) return Number(value);
  return ({ one: 1, two: 2, three: 3, four: 4, five: 5, six: 6 } as Record<string, number>)[value.toLowerCase()];
}

function parseSmallNumber(value: string): number | undefined {
  if (/^\d+$/.test(value)) return Number(value);
  const digits: Record<string, number> = {
    一: 1, 二: 2, 两: 2, 三: 3, 四: 4, 五: 5,
    六: 6, 七: 7, 八: 8, 九: 9, 十: 10,
  };
  if (value === "十") return 10;
  if (value.includes("十")) {
    const [left, right] = value.split("十");
    return (digits[left] ?? 1) * 10 + (digits[right] ?? 0);
  }
  return digits[value];
}

function countTimelinePoints(evidence: CockpitEvidence[], plan?: CockpitRetrievalPlan): number {
  const dates = new Set<string>();
  for (const item of evidence) {
    if (plan?.risks.includes("aggregation-frequency") && !hasOccurrenceRelation(item)) continue;
    for (const date of evidenceEventKeys(item)) dates.add(date);
    if (plan?.risks.includes("latest-final-update")) {
      for (const superseded of metadataStringArray(item, "supersedes")) {
        dates.add(`superseded:${superseded}`);
      }
    }
  }
  // When event dates are present they are a safer unit than record count:
  // L1 summaries and L0 raw messages can describe the same event twice.
  // Same-day multiple events may conservatively abstain, which is preferable
  // to inflating a frequency from duplicate representations.
  return dates.size > 0 ? dates.size : evidence.length;
}

function evidenceCoversDate(item: CockpitEvidence, date: string): boolean {
  if (!isAuthoritativeEvidence(item)) return false;
  const folded = `${item.eventTime ?? ""} ${item.content}`.toLowerCase();
  if (folded.includes(date)) return true;
  const target = canonicalMonthDay(date);
  if (target !== "" && extractDateMentions(folded).some((candidate) => canonicalMonthDay(candidate) === target)) return true;
  const targetTime = parseLooseDate(date);
  if (targetTime === undefined) return false;
  const validFrom = parseLooseDate(item.validFrom ?? metadataString(item, "valid_from") ?? "");
  const validTo = parseLooseDate(item.validTo ?? metadataString(item, "valid_to") ?? "");
  if (validFrom !== undefined && validTo !== undefined && targetTime >= validFrom && targetTime <= validTo) return true;
  if (validFrom !== undefined && validTo === undefined && targetTime >= validFrom) return true;
  if (extractValidityRanges(item.content).some(({ start, end }) => targetTime >= start && targetTime <= end)) return true;
  const eventTime = parseLooseDate(item.eventTime ?? "");
  return eventTime !== undefined
    && targetTime - eventTime >= 0
    && targetTime - eventTime <= 86_400_000
    && hasUpdateRelation(item);
}

const ENGLISH_STOP_WORDS = new Set([
  "about", "across", "after", "again", "before", "could", "does", "each", "from",
  "have", "into", "latest", "most", "often", "respectively", "should", "their", "there",
  "these", "this", "through", "what", "when", "where", "which", "with", "would", "finally",
  "january", "february", "march", "april", "june", "july", "august", "september", "october",
  "november", "december", "recorded", "records", "times",
]);

// Topic families are intentionally small and domain-level. They prevent a
// shared driver name or generic update verb from admitting unrelated events
// (for example a restaurant update into a commute timeline), without relying
// on benchmark IDs or exact answers.
const DOMAIN_TOPICS: RegExp[] = [
  /\b(?:route|commute|navigate|navigation|destination|airport|terminal|tunnel|toll)\b|高架|收费|路线|选路|择路|通勤|导航|机场|航站楼/iu,
  /\b(?:meeting|rendezvous|meeting point|rendezvous point)\b|(?:会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰)(?:点|地|处|地址|地点|位置|口令|标签|映射)|(?:会合|会面|会晤|会客|见面|碰面|集合|碰头|相约|约见)(?:事项|安排|约定).{0,28}(?:有效)?(?:点|地|处|地址|地点|位置|地方|去处)|(?:会合|会面|会晤|会师|会客|见面|碰面|集合|碰头|接头|相约|约见|约碰)(?:在|到|于)?(?:哪(?:里|儿)?|何处)/iu,
  /\b(?:restaurant|dinner|meal|food|dairy|lactose|vegan|vegetarian|italian|japanese|peanut)\b|餐厅|晚餐|用餐|素食|花生|奶制品|意大利|日料|生鱼/iu,
  /\b(?:charg(?:e|ed|ing)|battery|station)\b|充电|补能|补电|充能|充完|补完|补过能|充上电|补给电量|电量|充电站|超充站|能源点|补能中心|补能港/iu,
  /\b(?:maintenance|inspection|appointment|service)\b|保养|检修|检查|年检|年审|车检|车辆检验|检验排期|预约|轮胎|改期|时段/iu,
  /\b(?:alias|office|work)\b|老地方|爸妈家|别名|简称|固定车位|常停车位|常用车位|惯用车位|惯常车位|(?:惯用|常用|惯常|固定|常去|老)(?:停车|泊车)(?:位|点|处|地点|位置|地方|车位)|停车老(?:位置|地方|地点)|固定停车处|常停(?:点|地点|位置)|常停车(?:的)?(?:地方|地点|位置)|停车处|(?:停车|泊车)(?:惯用|常用|惯常|固定|常去|老)(?:处|点|地点|位置|地方|车位)|泊车点|车位|亲友住处|亲友住所|亲友住址|亲友家中|亲人家|亲属家|亲戚家|家里人(?:住所|住址|住处|家)|家人住所|家人住址|亲属住址|亲属住所|亲属住处|亲戚(?:住所|住址|住处)|亲人住处|家属(?:住所|住址|住处|家)|诊所|就诊地点|看诊(?:处|地点|位置)|看门诊(?:的)?(?:地方|地点|位置)|门诊(?:位置|地点|点|处)|诊疗(?:点|地点|处)|就医(?:点|地点)|看病点|挂号(?:处|地点|位置)|映射|绑定|公司|办公|证件|凭证|康平路/iu,
  /\b(?:address|home)\b|住址|家庭地址|住宅地址/iu,
  /\b(?:temperature|climate|air conditioning|a\/?c)\b|空调|温度|几度|多少度|\d+(?:\.\d+)?\s*度|\d+(?:\.\d+)?\s*°[cf]/iu,
  /\b(?:audio|music|jazz|audiobooks?|podcast|volume)\b|媒体|音频|音响|声音方案|播放|听什么|听啥|放啥|音乐|曲风|音量|声量|几格|格位|播客|有声书/iu,
  /\b(?:seat|lumbar|backrest)\b|座椅|座位|腰托|腰撑|腰部支撑|靠背|(?:长途|远途).{0,12}支撑|支撑(?:值|档位|设置|结果)/iu,
  /\b(?:school|academy|college|university)\b|学校|学院|就读/iu,
  /\b(?:membership|member card|payment|pay(?:ment|ing)?)\b|会员卡|付款|支付/iu,
  /\b(?:singer|artist|favorite musician|access card|badge number)\b|歌手|艺人|演唱者|主唱|谁(?:的歌|演唱)|门禁|门卡|工牌|通行(?:卡|证)|出入(?:卡|证|凭证)|办公(?:卡|证件|凭证)|(?:员工|单位|工作|公司|办公(?:楼)?)(?:用|的)?(?:通行卡|通行证|门卡|出入卡|出入证|出入凭证|证件|凭证|身份卡)|卡号/iu,
];

const REQUESTED_ATTRIBUTE_TOPICS: Array<{ query: RegExp; evidence: RegExp }> = [
  { query: /\b(?:school|academy|college|university)\b|学校|学院|就读/iu, evidence: /\b(?:school|academy|college|university)\b|学校|学院|就读/iu },
  {
    // A seat list often supplies ownership context for another requested
    // field ("座位依次主驾、副驾、后排，温度分别多少"). Only treat seating
    // as the requested attribute when the wording actually asks for it.
    query: /\bseat(?:ing)?(?:\s+position)?\b|座椅(?:位置|偏好|设置)|座位(?:位置|偏好|设置|在哪|是哪|是什么|如何|怎么)|坐(?:在)?哪(?:里|儿|个座位)/iu,
    evidence: /\bseat(?:ing)?(?:\s+position)?\b|座椅|座位|主驾|副驾|驾驶席|前排|后排|后座/iu,
  },
  {
    query: /\b(?:temperature|climate|air conditioning|a\/?c)\b|空调(?:温度|偏好|设置|调到|调成)|温度|多少度|几度/iu,
    evidence: /\b(?:temperature|climate|air conditioning|a\/?c)\b|空调|温度|\d+(?:\.\d+)?\s*(?:度|℃|°[cf])/iu,
  },
  { query: /\b(?:membership|member card|payment|pay(?:ment|ing)?)\b|会员卡|付款|支付/iu, evidence: /\b(?:membership|member card|payment|pay(?:ment|ing)?)\b|会员卡|付款|支付/iu },
  { query: /\bpodcasts?\b|播客/iu, evidence: /\bpodcasts?\b|播客/iu },
  {
    query: /\b(?:singer|artist|favorite musician)\b|歌手(?:人名|姓名|名字)?|艺人(?:姓名|名字)?|演唱者(?:姓名|名字)?|主唱(?:姓名|名字)?|具体歌手|谁(?:的歌|唱(?:的)?歌|(?:来)?演唱)|最爱听谁|最常听哪|(?:喜欢|偏爱|常听|爱听|最爱|最常听)(?:的)?.{0,10}(?:谁唱|谁演唱|哪位唱|哪名唱|歌手(?:人名|姓名|名字)?)|喜欢谁(?:唱歌|演唱)|偏爱谁演唱/iu,
    // A genre or broad music preference is not evidence for a named performer.
    evidence: /\b(?:singer|artist|favorite musician)\b|歌手|艺人|演唱者|主唱|由[^，。；;\n]{1,20}(?:演唱|主唱)|最爱听[^，。；;\n]{1,20}(?:唱歌|演唱)|最常听[^，。；;\n]{1,20}(?:歌手|艺人|主唱)/iu,
  },
  {
    query: /\b(?:access card|badge number)\b|门禁(?:卡号|编号|号码)?|(?:门卡|工牌|通行卡|通行证|出入卡|出入证|出入凭证|办公(?:证件|凭证)|员工(?:通行卡|通行证|门卡|出入卡|出入证|出入凭证|证件)).{0,12}(?:编号|号码|卡号|标识|序号|数字|准确值|\bid\b)|(?:员工|单位|公司|工作|办公(?:楼)?)(?:用|的)?(?:门禁卡|门卡|工牌|通行卡|通行证|出入卡|出入证|出入凭证|证件|凭证|身份卡|卡)(?:的)?.{0,10}(?:号|编号|号码|标识|序号|数字|值|\bid\b)|卡号/iu,
    // A generic mention of an office, access control, or a card is not
    // evidence for the requested identifier. Keep the evidence side narrower
    // than the query side so same-person but wrong-field memories fail closed.
    evidence: /\b(?:badge number|access card.{0,16}(?:number|id))\b|门禁(?:卡号|编号|号码)|(?:门卡|工牌|通行卡|通行证|出入卡|出入证|出入凭证|办公(?:证件|凭证)|员工(?:通行卡|通行证|门卡|出入卡|出入证|出入凭证|证件)).{0,12}(?:编号|号码|卡号|标识|序号|数字|准确值|\bid\b)|(?:员工|单位|公司|工作|办公(?:楼)?)(?:用|的)?(?:门禁卡|门卡|工牌|通行卡|通行证|出入卡|出入证|出入凭证|证件|凭证|身份卡|卡)(?:的)?.{0,10}(?:号|编号|号码|标识|序号|数字|值|\bid\b)|卡号/iu,
  },
];

function matchesRequestedAttribute(content: string, query: string): boolean {
  const requested = REQUESTED_ATTRIBUTE_TOPICS.filter((topic) => topic.query.test(query));
  if (requested.length === 0) return true;
  const explicitlyMissing = /(?:未|没有|没|并未|无法).{0,16}(?:记录|保存|提供|写明|覆盖|确定)|(?:字段|信息).{0,12}(?:缺失|未覆盖|不存在)/iu.test(content);
  return !explicitlyMissing && requested.every((topic) => topic.evidence.test(content));
}

function containsCjk(value: string): boolean {
  return /[\u3400-\u9fff]/u.test(value);
}

/**
 * Build a bounded retrieval view from the original prose plus the small set
 * of typed fields that the cockpit-state-v1 contract validates.  Arbitrary
 * metadata is deliberately excluded: ranking must not become a second prompt
 * parser or admit unrelated backend annotations.
 */
function evidenceSearchText(item: CockpitEvidence): string {
  if (!isStructuredCockpitEvidence(item)) return item.content;
  const typed = [
    "domain",
    "slot",
    "state_key",
    "target",
    "subject",
    "occupant_scope",
    "vehicle_scope",
    "seat_zone",
    "record_kind",
    "action_status",
    "relation",
  ].map((key) => metadataString(item, key));
  const value = getCockpitStructuredValue(item);
  return [item.content, ...typed, value]
    .filter((entry): entry is string => Boolean(entry))
    .map((entry) => entry.slice(0, 256))
    .join(" ");
}

function topicOverlap(content: string, query: string): number {
  const foldedContent = content.toLowerCase();
  const englishTerms = [...new Set((query.toLowerCase().match(/[a-z][a-z'-]{2,}/gu) ?? [])
    .map((term) => term.replace(/^[-']+|[-']+$/gu, "")))]
    .filter((term) => !ENGLISH_STOP_WORDS.has(term));
  let overlap = englishTerms.filter((term) => foldedContent.includes(term)).length;

  const queryCjk = (query.match(/[\u3400-\u9fff]{2,}/gu) ?? []).join("");
  const grams = new Set<string>();
  for (let i = 0; i + 1 < queryCjk.length; i++) grams.add(queryCjk.slice(i, i + 2));
  const generic = /^(?:驾驶|驶员|事件|记录|第一|一周|三个|最常|哪个|共去|去了|几次|中驾|的三|个充|站共)$/u;
  overlap += [...grams].filter((gram) => !generic.test(gram) && foldedContent.includes(gram)).length;
  return overlap;
}

function sharesDomainTopic(content: string, query: string): boolean {
  const queryTopics = DOMAIN_TOPICS.filter((topic) => topic.test(query));
  // Unknown cockpit subjects retain lexical routing; known subjects require
  // at least one shared semantic family so person names alone cannot match.
  return queryTopics.length === 0 || queryTopics.some((topic) => topic.test(content));
}

function hasRecognizedSharedDomainTopic(content: string, query: string): boolean {
  const queryTopics = DOMAIN_TOPICS.filter((topic) => topic.test(query));
  return queryTopics.length > 0 && queryTopics.some((topic) => topic.test(content));
}

function evidenceSharesDomainTopic(item: CockpitEvidence, query: string): boolean {
  return sharesDomainTopic(evidenceSearchText(item), query);
}

function evidenceTopicOverlap(item: CockpitEvidence, query: string): number {
  const evidenceText = evidenceSearchText(item);
  const lexical = topicOverlap(evidenceText, query);
  // English typed domains such as `commute` and `climate` should remain
  // recallable from Chinese queries even when compact L1 prose uses a very
  // different paraphrase. This bonus applies only to a recognized shared
  // domain family, never to an unknown-domain wildcard.
  return lexical + (isStructuredCockpitEvidence(item)
    && hasRecognizedSharedDomainTopic(evidenceText, query) ? 1 : 0);
}

function evidenceMatchesRequestedAttribute(item: CockpitEvidence, query: string): boolean {
  return matchesRequestedAttribute(evidenceSearchText(item), query);
}

function evidenceRelevance(item: CockpitEvidence, plan: CockpitRetrievalPlan): number {
  const evidenceText = evidenceSearchText(item);
  let rank = evidenceTopicOverlap(item, plan.originalQuery) * 4;
  const folded = evidenceText.toLowerCase();
  // High-risk answers favor source turns over equivalent summaries so event
  // identity, speaker ownership, and corrections survive final selection.
  if (item.source === "l0") rank += 5;
  if (/\[source_role=user\]/iu.test(item.content)) rank += 4;
  else if (item.role === "user") rank += 1;
  if (item.matchedPurposes?.some((purpose) => purpose !== "original")) rank += 1;
  if (evidenceSharesDomainTopic(item, plan.originalQuery)) rank += 8;
  if (isStructuredCockpitEvidence(item)) {
    rank += isAuthoritativeEvidence(item) ? 6 : -100;
  }

  if (plan.risks.includes("aggregation-frequency")) {
    if (hasOccurrenceRelation(item)) rank += 6;
    if (/persona|profile|偏好|通常|usually|prefer/iu.test(`${item.type ?? ""} ${folded}`)) rank -= 7;
    if (/导航|navigate|route to|电量|battery|距离|distance/iu.test(folded)) rank -= 5;
  }
  if (plan.risks.includes("latest-final-update")) {
    if (hasUpdateRelation(item)) rank += 5;
    if (/schedul|appointment|inspection|book(?:ed)?|安排|预约|原定/iu.test(folded)) rank += 3;
  }
  if (plan.risks.includes("multi-time-comparison")) {
    if (/alias|resolve|destination|temporary|normal|work|别名|有效|临时|恢复|改回/iu.test(folded)) rank += 4;
    if (hasUpdateRelation(item)) rank += 2;
  }
  if (plan.risks.includes("cross-session-synthesis") && item.sessionId) rank += 2;
  if (item.isSessionPacket) rank += 10;
  return rank;
}

export function hasOccurrenceRelation(item: CockpitEvidence): boolean {
  if (isStructuredCockpitEvidence(item)) {
    const recordKind = metadataString(item, "record_kind");
    const status = metadataString(item, "action_status");
    const relation = metadataString(item, "relation");
    if (recordKind === "event"
      && ["executed", "verified", "completed"].includes(status ?? "")
      && relation !== "cancelled"
      && relation !== "negated") return true;
  }
  return /(?:visited|went back|charged|charge(?:d|ing)? at|logged|occurrence|event|first time|second time|third time|去了|去过|访问|充了电|在.{0,20}充电|完成(?:了)?(?:车辆)?(?:充电|补能)|到.{0,28}(?:能源点|补能中心|补能港|充电站|超充站).{0,12}补能|已(?:记录|登记)(?:这次|本次|第二次|第三次)?.{0,48}(?:充电|补能|充能|能源点|补能中心|补能港|充电站|超充站)|事件)/iu.test(item.content)
    && !/(?:导航|navigate|route to|偏好|通常|usually|prefer)/iu.test(item.content);
}

function isRiskRelevant(item: CockpitEvidence, plan: CockpitRetrievalPlan): boolean {
  if (plan.risks.includes("aggregation-frequency") && !hasOccurrenceRelation(item)) return false;
  if (plan.risks.includes("latest-final-update")) {
    // A complete update chain needs both the prior state and its mutation.
    // Domain-topic gating above removes unrelated state-bearing records.
    return evidenceSharesDomainTopic(item, plan.originalQuery);
  }
  if (plan.risks.includes("multi-time-comparison")) {
    return hasExplicitDateTargetMatch(item, plan)
      || evidenceSharesDomainTopic(item, plan.originalQuery);
  }
  return true;
}

function hasUpdateRelation(item: CockpitEvidence): boolean {
  const relation = metadataString(item, "relation");
  if (relation && ["updated", "cancelled", "negated"].includes(relation)) return true;
  if (metadataStringArray(item, "supersedes").length > 0) return true;
  const content = item.content
    .replace(/\[memory_episode[^\]]*\]/giu, " ")
    .replace(/\[(?:source_time|source_role|resolved_relative_time)[^\]]*\]/giu, " ");
  const withoutNegatedChange = content.replace(/(?:do not change|without making (?:any )?changes?|不要修改|不要更改|只读)/giu, " ");
  return /(?:move(?:d)?|reschedul|cancel|revok(?:e|ed)|void(?:ed)?|no longer|not anymore|change(?:d)?|correct(?:ed|ion)?|replace(?:d)?|back to|now resolves?|finally|updated?|becomes?\s+(?:the\s+)?priority|priority\s+instead|temporary.{0,30}(?:finish|end)|(?:for|during)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)|改到|改为|改期|取消|撤销|作废|无效|不再|不要了|停止|更正|替代|恢复|改回|现在|最终|定稿|敲定|更新)/iu.test(withoutNegatedChange);
}

function requiresExplicitStateTransition(query: string): boolean {
  const semantics = compileChineseCockpitSemantics(query);
  if (semantics.intents.includes("correction-state")
    || semantics.intents.includes("final-cancellation")) return true;
  const withoutNegatedChange = query.replace(
    /(?:do not change|without making (?:any )?changes?|不要修改|不要更改|无需修改|并非(?:修改|更新))/giu,
    " ",
  );
  return /(?:\b(?:correct(?:ion|ed)?|cancel(?:lation|led)?|reschedul(?:e|ed|ing)|replace(?:d|ment)?|revoke(?:d)?|void(?:ed)?|no longer|back to|after (?:the )?(?:change|update))\b|更正|纠正|纠偏|改正|改口|改为|改成|改到|改期|替代|撤销|取消|作废|不再|停止|恢复|改回|更新后|调整后|修改后|变更后|改完后|多次变化后|定稿后|敲定后)/iu.test(withoutNegatedChange);
}

function hasExplicitDateTargetMatch(item: CockpitEvidence, plan: CockpitRetrievalPlan): boolean {
  return evidenceMatchesRequestedDateTarget(item, plan)
    && plan.requiredDates.some((date) => evidenceCoversDate(item, date));
}

function evidenceMatchesRequestedDateTarget(item: CockpitEvidence, plan: CockpitRetrievalPlan): boolean {
  const targets = extractCockpitNamedTargets(plan.originalQuery);
  return targets.length === 0 || targets.some((target) => evidenceMatchesNamedTarget(item, target));
}

function hasRiskQuerySupport(item: CockpitEvidence, plan: CockpitRetrievalPlan): boolean {
  const matchedRisk = item.matchedPurposes?.some((purpose) =>
    purpose !== "original"
    && purpose !== "date-point"
    && plan.risks.includes(purpose)
  ) === true;
  if (!matchedRisk) return false;
  // Frequency expansions still need an explicit occurrence; the other risk
  // expansions may intentionally retrieve semantically linked evidence whose
  // wording does not repeat the original question (for example a restaurant
  // rationale containing only its name and dietary properties).
  return !plan.risks.includes("aggregation-frequency") || hasOccurrenceRelation(item);
}

function evidenceGroupKey(item: CockpitEvidence): string {
  const episodeKey = metadataString(item, "episode_key");
  if (episodeKey) return `episode:${episodeKey}`;
  if (item.source === "l1") {
    const natural = extractDateMentions(item.content)[0];
    if (natural) return `event:${canonicalMonthDay(natural) || natural}`;
  }
  const eventKeys = evidenceEventKeys(item);
  if (eventKeys.length > 0) return `event:${canonicalMonthDay(eventKeys[0]) || eventKeys[0]}`;
  const benchmark = item.content.match(/\[([^\]\s:]*-s\d+):\d+\]/iu)?.[1];
  if (benchmark) return `benchmark:${benchmark}`;
  if (item.sessionId) return `session:${item.sessionId}`;
  return `${item.source}:${item.id}`;
}

function coalesceL0EventMessages(evidence: CockpitEvidence[]): CockpitEvidence[] {
  const groups = new Map<string, CockpitEvidence[]>();
  const order: string[] = [];
  for (const item of evidence) {
    const benchmarkSession = item.content.match(/\[([^\]\s:]+-s\d+):\d+\]/iu)?.[1];
    const owner = benchmarkSession ?? item.sessionId;
    const eventTime = item.eventTime;
    const key = item.source === "l0" && owner && eventTime
      ? `l0-event:${owner}:${eventTime}`
      : `single:${item.source}:${item.id}`;
    if (!groups.has(key)) order.push(key);
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }

  return order.map((key) => {
    const items = groups.get(key) ?? [];
    if (items.length <= 1) return items[0];
    const sorted = [...items].sort((a, b) => benchmarkMessageOrder(a.content) - benchmarkMessageOrder(b.content));
    const contents: string[] = [];
    const seen = new Set<string>();
    for (const item of sorted) {
      const normalized = normalizeEvidence(item.content);
      if (seen.has(normalized)) continue;
      seen.add(normalized);
      contents.push(item.content);
    }
    const scores = items.map((item) => item.score)
      .filter((score): score is number => typeof score === "number" && Number.isFinite(score));
    return {
      ...sorted[0],
      id: sorted.map((item) => item.id).join("+"),
      content: contents.join("\n"),
      score: scores.length > 0 ? Math.max(...scores) : undefined,
      role: sorted.some((item) => item.role === "user") ? "user" : sorted[0].role,
      matchedTargets: [...new Set(sorted.flatMap((item) => item.matchedTargets ?? []))],
      matchedPurposes: [...new Set(sorted.flatMap((item) => item.matchedPurposes ?? []))],
    };
  });
}

function benchmarkMessageOrder(content: string): number {
  const value = content.match(/\[[^\]\s:]+-s\d+:(\d+)\]/iu)?.[1];
  return value ? Number(value) : Number.MAX_SAFE_INTEGER;
}

function evidenceEventKeys(item: CockpitEvidence): string[] {
  const episodeKey = metadataString(item, "episode_key");
  if (episodeKey) return [`episode:${episodeKey}`];
  const explicit = item.content.matchAll(/\[source_time=([^\]\s]+)\]/giu);
  const keys = new Set<string>();
  for (const match of explicit) {
    const calendar = isoCalendarPrefix(match[1]);
    if (calendar) keys.add(calendar);
    else {
      const parsed = new Date(match[1]);
      if (Number.isFinite(parsed.getTime())) keys.add(parsed.toISOString().slice(0, 10));
    }
  }
  if (keys.size > 0) return [...keys];

  const eventTime = item.eventTime;
  if (eventTime) {
    const calendar = isoCalendarPrefix(eventTime);
    if (calendar) keys.add(calendar);
    else {
      const parsed = new Date(eventTime);
      if (Number.isFinite(parsed.getTime())) keys.add(parsed.toISOString().slice(0, 10));
    }
  }
  if (keys.size > 0) return [...keys];
  for (const date of extractDateMentions(item.content)) keys.add(canonicalMonthDay(date) || date);
  return [...keys];
}

function extractEvidenceEventTime(item: CockpitEvidence): string | undefined {
  const structuredEventTime = metadataString(item, "activity_start_time")
    ?? metadataString(item, "valid_from")
    ?? metadataString(item, "mentioned_at");
  if (structuredEventTime && Number.isFinite(new Date(structuredEventTime).getTime())) return structuredEventTime;
  const sourceTime = item.content.match(/\[source_time=([^\]\s]+)\]/iu)?.[1];
  // Preserve the calendar date and offset as written. Converting +08:00 to
  // UTC before date reasoning can move an early-morning event to the previous
  // day and falsely fail a Chinese validity/cutoff gate.
  if (sourceTime && Number.isFinite(new Date(sourceTime).getTime())) return sourceTime;
  const narrated = extractNarratedEventCalendarDate(item.content);
  if (narrated) return narrated;
  const embeddedIso = item.content.match(/\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z?)?/u)?.[0];
  if (embeddedIso && Number.isFinite(new Date(embeddedIso).getTime())) return new Date(embeddedIso).toISOString();
  // Unit and legacy records may expose the actual event clock only in the
  // timestamp field. Raw benchmark records above always take source_time.
  return item.timestamp;
}

function isoCalendarPrefix(value: string): string | undefined {
  return value.match(/^(20\d{2}-\d{2}-\d{2})/u)?.[1];
}

function extractNarratedEventCalendarDate(content: string): string | undefined {
  const chinese = content.match(
    /(?:^|[\n。；;])\s*(?:用户(?:（[^）]{0,24}）|\([^)]{0,24}\))?\s*)?(?:于|在)\s*(20\d{2}年\d{1,2}月\d{1,2}日)/u,
  )?.[1];
  if (!chinese) return undefined;
  const parts = chinese.match(/(20\d{2})年(\d{1,2})月(\d{1,2})日/u);
  if (!parts) return undefined;
  return `${parts[1]}-${String(Number(parts[2])).padStart(2, "0")}-${String(Number(parts[3])).padStart(2, "0")}`;
}

function latestEvidenceIsoTime(item: CockpitEvidence): string | undefined {
  const values: number[] = [];
  const observedAt = item.observedAt ?? extractEvidenceObservedAt(item);
  if (observedAt) {
    const value = new Date(observedAt).getTime();
    if (Number.isFinite(value)) values.push(value);
  }
  for (const match of item.content.matchAll(/\[source_time=([^\]\s]+)\]/giu)) {
    const value = new Date(match[1]).getTime();
    if (Number.isFinite(value)) values.push(value);
  }
  const fallback = new Date(item.eventTime ?? item.timestamp ?? "").getTime();
  if (Number.isFinite(fallback)) values.push(fallback);
  if (values.length === 0) return undefined;
  return new Date(Math.max(...values)).toISOString();
}

function latestEvidenceTime(item: CockpitEvidence): number {
  const value = latestEvidenceIsoTime(item);
  return value ? new Date(value).getTime() : Number.NEGATIVE_INFINITY;
}

function extractCutoffDate(query: string, referenceDate: string): string | undefined {
  const explicit = query.match(/(?:\bas of\b|截至|截止)\s*((?:\d{4}年)?\d{1,2}月\d{1,2}日|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)/iu)?.[1];
  const dates = extractDateMentions(query);
  const naturalChineseCutoff = dates.length === 1 && isChineseCutoffStateQuery(query)
    ? dates[0]
    : undefined;
  const value = explicit ?? naturalChineseCutoff;
  if (!value) return undefined;
  const parts = monthDayParts(value);
  if (!parts) return undefined;
  const explicitYear = value.match(/\b(20\d{2})\b|^(20\d{2})年/u);
  const year = Number(explicitYear?.[1] ?? explicitYear?.[2] ?? referenceDate.slice(0, 4));
  return `${year}-${String(parts.month).padStart(2, "0")}-${String(parts.day).padStart(2, "0")}`;
}

function isAtOrBeforeCutoff(item: CockpitEvidence, cutoff: string | undefined): boolean {
  if (!cutoff) return true;
  // Only provenance event time is safe for cutoff filtering. Ingestion clocks
  // and appointment target dates must never be mistaken for observation time.
  const structuredObserved = item.observedAt ?? metadataString(item, "mentioned_at");
  const structuredDate = structuredObserved ? isoCalendarPrefix(structuredObserved) : undefined;
  if (structuredDate) return structuredDate <= cutoff;
  const sourceTime = item.content.match(/\[source_time=([^\]\s]+)\]/iu)?.[1];
  const eventDate = sourceTime
    ? isoCalendarPrefix(sourceTime)
    : extractNarratedEventCalendarDate(item.content);
  if (!eventDate) return true;
  return eventDate <= cutoff;
}

function extractValidityRanges(content: string): Array<{ start: number; end: number }> {
  const ranges: Array<{ start: number; end: number }> = [];
  const english = content.matchAll(/(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:through|to|[-–])\s*(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+)?(\d{1,2})(?:st|nd|rd|th)?/giu);
  for (const match of english) {
    const month = monthNumber(match[1]);
    if (month > 0) ranges.push({ start: Date.UTC(2000, month - 1, Number(match[2])), end: Date.UTC(2000, month - 1, Number(match[3])) });
  }
  const chinese = content.matchAll(/(\d{1,2})月(\d{1,2})日?\s*(?:至|到|[-–])\s*(?:(\d{1,2})月)?(\d{1,2})日/gu);
  for (const match of chinese) {
    const startMonth = Number(match[1]);
    const endMonth = Number(match[3] ?? match[1]);
    ranges.push({ start: Date.UTC(2000, startMonth - 1, Number(match[2])), end: Date.UTC(2000, endMonth - 1, Number(match[4])) });
  }
  return ranges;
}

function hasTransitionBetweenDates(evidence: CockpitEvidence[], requiredDates: string[]): boolean {
  const points = requiredDates.slice(0, 2).map(parseLooseDate);
  if (points.some((point) => point === undefined)) return false;
  const [left, right] = points as number[];
  const start = Math.min(left, right);
  const end = Math.max(left, right);
  return evidence.some((item) => {
    const position = parseLooseDate(item.eventTime ?? "");
    return position !== undefined && position > start && position <= end
      && (hasUpdateRelation(item) || directlyMentionsDate(item, requiredDates[1]));
  });
}

function directlyMentionsDate(item: CockpitEvidence, date: string): boolean {
  const target = canonicalMonthDay(date);
  return target !== "" && extractDateMentions(item.content)
    .some((candidate) => canonicalMonthDay(candidate) === target);
}

function parseLooseDate(value: string): number | undefined {
  const parts = monthDayParts(value);
  return parts ? Date.UTC(2000, parts.month - 1, parts.day) : undefined;
}

function monthDayParts(value: string): { month: number; day: number } | undefined {
  const chinese = value.match(/(\d{1,2})月(\d{1,2})日/u);
  if (chinese) return { month: Number(chinese[1]), day: Number(chinese[2]) };
  const iso = value.match(/\d{4}-(\d{2})-(\d{2})/u);
  if (iso) return { month: Number(iso[1]), day: Number(iso[2]) };
  const monthFirst = value.toLowerCase().match(/(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})/u);
  if (monthFirst) return { month: monthNumber(monthFirst[1]), day: Number(monthFirst[2]) };
  const dayFirst = value.toLowerCase().match(/(\d{1,2})\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)/u);
  return dayFirst ? { month: monthNumber(dayFirst[2]), day: Number(dayFirst[1]) } : undefined;
}

function monthNumber(value: string): number {
  const months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
  return months.findIndex((month) => value.toLowerCase().startsWith(month)) + 1;
}

function canonicalMonthDay(value: string): string {
  const chinese = value.match(/(\d{1,2})月(\d{1,2})日/u);
  if (chinese) return `${Number(chinese[1])}-${Number(chinese[2])}`;
  const iso = value.match(/\d{4}-(\d{2})-(\d{2})/u);
  if (iso) return `${Number(iso[1])}-${Number(iso[2])}`;
  const english = value.toLowerCase().match(/(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})/u);
  if (!english) return "";
  const monthNames = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
  const month = monthNames.findIndex((name) => english[1].startsWith(name)) + 1;
  return month > 0 ? `${month}-${Number(english[2])}` : "";
}

function countProvenanceGroups(evidence: CockpitEvidence[]): number {
  const groups = new Set<string>();
  for (const item of evidence) {
    if (item.sessionId) groups.add(item.sessionId);
    for (const sessionId of metadataStringArray(item, "source_session_ids")) groups.add(sessionId);
    const sourceSession = metadataString(item, "source_session_id");
    if (sourceSession) groups.add(sourceSession);
    const explicit = item.content.matchAll(/\bsession(?:_id)?\s*[:=]\s*["']?([\w.-]+)/giu);
    for (const match of explicit) groups.add(match[1]);
    const benchmark = item.content.matchAll(/\[([^\]\s:]*-s\d+):\d+\]/giu);
    for (const match of benchmark) groups.add(match[1]);
  }
  return groups.size;
}

function metadataValue(item: CockpitEvidence, key: string): unknown {
  return item.metadata && typeof item.metadata === "object" ? item.metadata[key] : undefined;
}

function metadataString(item: CockpitEvidence, key: string): string | undefined {
  const value = metadataValue(item, key);
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
}

function metadataStringArray(item: CockpitEvidence, key: string): string[] {
  const value = metadataValue(item, key);
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((entry): entry is string => typeof entry === "string" && entry.length > 0))];
}

function isStructuredCockpitEvidence(item: CockpitEvidence): boolean {
  return metadataString(item, "schema_version") === "cockpit-state-v1";
}

function isAuthoritativeEvidence(item: CockpitEvidence): boolean {
  if (!isStructuredCockpitEvidence(item)) return true;
  const quality = metadataValue(item, "construction_quality");
  return Boolean(quality)
    && typeof quality === "object"
    && !Array.isArray(quality)
    && (quality as Record<string, unknown>).status === "complete";
}

/** Public read-only gate used by the injector before exposing evidence to the answer model. */
export function isCockpitEvidenceAuthoritative(item: CockpitEvidence): boolean {
  return isAuthoritativeEvidence(item);
}

function evidenceMatchesNamedTarget(item: CockpitEvidence, target: string): boolean {
  if (evidenceMatchesTarget(item.content, target)) return true;
  const needle = normalizeEvidence(target);
  if (!needle) return false;
  return ["slot", "state_key", "target", "subject", "occupant_scope", "seat_zone"]
    .map((key) => metadataString(item, key))
    .filter((value): value is string => Boolean(value))
    .some((value) => normalizeEvidence(value).includes(needle) || needle.includes(normalizeEvidence(value)));
}

function evidenceMatchesOwner(item: CockpitEvidence, owner: string): boolean {
  const normalizedOwner = normalizeEvidence(owner);
  return item.content.includes(owner)
    || ["subject", "occupant_scope"]
      .map((key) => metadataString(item, key))
      .filter((value): value is string => Boolean(value))
      .some((value) => normalizeEvidence(value) === normalizedOwner);
}

function extractEvidenceObservedAt(item: CockpitEvidence): string | undefined {
  const structured = metadataString(item, "mentioned_at");
  if (structured && Number.isFinite(new Date(structured).getTime())) return structured;
  const sourceTime = item.content.match(/\[source_time=([^\]\s]+)\]/iu)?.[1];
  if (sourceTime && Number.isFinite(new Date(sourceTime).getTime())) return sourceTime;
  return item.timestamp;
}

export function getCockpitStructuredValue(item: CockpitEvidence): string | undefined {
  const value = metadataValue(item, "value") ?? metadataValue(item, "target");
  if (typeof value === "string") return value.trim() || undefined;
  if (typeof value === "number" || typeof value === "boolean") {
    const unit = metadataString(item, "unit");
    return `${String(value)}${unit ?? ""}`;
  }
  if (Array.isArray(value)) {
    const parts = value.filter((entry) => ["string", "number", "boolean"].includes(typeof entry)).map(String);
    return parts.length > 0 ? parts.join("、") : undefined;
  }
  return undefined;
}

function structuredRelation(item: CockpitEvidence): CockpitFinalState["relation"] | undefined {
  const relation = metadataString(item, "relation");
  if (relation === "asserted" || relation === "cancelled" || relation === "negated" || relation === "updated") return relation;
  // A structured row selected from the update set can carry asserted after a
  // legacy dedup response; supersedes still proves that it is the new state.
  if (metadataStringArray(item, "supersedes").length > 0) return "updated";
  return undefined;
}

function isStructuredCurrentStateCandidate(item: CockpitEvidence): boolean {
  if (!isStructuredCockpitEvidence(item) || !isAuthoritativeEvidence(item)) return false;
  const relation = metadataString(item, "relation");
  if (relation === "updated" || relation === "cancelled" || relation === "negated") return true;
  const recordKind = metadataString(item, "record_kind");
  const status = metadataString(item, "action_status");
  if (recordKind === "state_assertion" && !status) return true;
  return ["selected", "confirmed", "executed", "verified", "completed"].includes(status ?? "");
}

function hasCompleteStructuredStateSnapshot(
  plan: CockpitRetrievalPlan,
  evidence: CockpitEvidence[],
): boolean {
  const candidates = evidence.filter(isStructuredCurrentStateCandidate);
  if (candidates.length === 0) return false;
  const targets = extractCockpitNamedTargets(plan.originalQuery);
  if (targets.length === 0) return true;
  return targets.every((target) => candidates.some((item) =>
    evidenceMatchesNamedTarget(item, target) && getCockpitStructuredValue(item) !== undefined
  ));
}

function normalizeEvidence(value: string): string {
  return value.toLowerCase().replace(/\s+/gu, " ").trim();
}

function dedupeQueries(queries: CockpitRetrievalQuery[]): CockpitRetrievalQuery[] {
  const seen = new Set<string>();
  return queries.filter((item) => {
    const key = item.text.trim();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function compareScore(a: number | undefined, b: number | undefined): number {
  return (a ?? -Infinity) - (b ?? -Infinity);
}

function compareTimestamp(a: string | undefined, b: string | undefined): number {
  if (!a && !b) return 0;
  if (!a) return 1;
  if (!b) return -1;
  const aTime = new Date(a).getTime();
  const bTime = new Date(b).getTime();
  if (Number.isFinite(aTime) && Number.isFinite(bTime)) return aTime - bTime;
  return a.localeCompare(b);
}
