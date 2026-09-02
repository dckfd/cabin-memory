/**
 * Deterministic post-extraction contract for smart-cockpit memories.
 *
 * The LLM proposes semantic fields; this module binds them to actual L0
 * evidence, stamps construction provenance, and records contract defects.
 * It never invents a person, value, scope, or execution result from prose.
 */

import { createHash } from "node:crypto";

import type { ConversationMessage } from "../conversation/l0-recorder.js";
import type { ExtractedMemory } from "./l1-writer.js";
import {
  canonicalCockpitConstraintTarget,
  canonicalCockpitDomain,
  canonicalCockpitSceneClass,
  canonicalCockpitSlot,
  canonicalControlledCockpitDomain,
  controlledCockpitSlotOwners,
} from "./cockpit-ontology.js";
import {
  extractCockpitAuthorizedNamedDestinationStateQualifiers,
  isCockpitPureInformationalQuery,
} from "./cockpit-source-coverage.js";
import {
  isControlledCockpitEventTimeSlot,
  strictZonedIsoInstant,
} from "./cockpit-temporal.js";

export const COCKPIT_STATE_SCHEMA_VERSION = "cockpit-state-v1" as const;

export type CockpitStateRelation = "asserted" | "updated" | "cancelled" | "negated";
export type CockpitConstructionQualityStatus = "complete" | "partial" | "invalid";

export interface CockpitConstructionQuality {
  status: CockpitConstructionQualityStatus;
  score: number;
  issues: string[];
  /** Deterministic, evidence-preserving normalizations; these do not lower status. */
  repairs: string[];
  source_count: number;
  user_source_count: number;
}

export interface CockpitKnownLineage {
  /** Persisted record ID when the lineage comes from an earlier session. */
  recordId?: string;
  /** Persisted/batch memory class; production callers always provide both. */
  type?: ExtractedMemory["type"];
  scene_name?: string;
  /** Compact structured metadata; prose is deliberately unnecessary here. */
  metadata: Record<string, unknown>;
}

export interface NormalizeCockpitMemoryParams {
  memory: ExtractedMemory;
  /** Only the messages the extraction prompt permits as factual sources. */
  sourceMessages: ConversationMessage[];
  sessionId?: string;
  constructionModel?: string;
  /** Exact prior/batch identities against which supersedes edges are checked. */
  knownLineage?: CockpitKnownLineage[];
  /** Exact directed-coverage spans used to scope appointment end grounding. */
  sourceEvidenceSpans?: Array<{ start: number; end: number }>;
}

export type CockpitDedupAction = "store" | "update" | "merge" | "skip";

const SAFE_KEY_PART = /[^\p{L}\p{N}._-]+/gu;
const UNKNOWN_SCOPE_SENTINELS = new Set([
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
const GENERIC_FIRST_PERSON_SUBJECTS = new Set(["user", "用户", "我", "本人", "用户本人"]);
const GENERIC_STATE_QUALIFIERS = new Set([
  "state",
  "slot",
  "value",
  "destination",
  "地点",
  "目的地",
  "位置",
  "导航目的地",
  "这个地点",
  "那个地点",
  "这个目的地",
  "那个目的地",
]);
const PLACEHOLDER_FACT_VALUES = new Set([
  "unknown",
  "unresolved",
  "unspecified",
  "none",
  "null",
  "n/a",
  "na",
  "未知",
  "待定",
  "未定",
  "未确定",
  "未解析",
]);

const CLOCK_TOKEN = String.raw`(?:[01]?\d|2[0-3])(?:[:：][0-5]\d|点(?:[0-5]?\d分?)?)`;
const DATE_TOKEN = String.raw`\d{1,2}月\d{1,2}日`;
const APPOINTMENT_TIME_RANGE = new RegExp(
  String.raw`(?:${CLOCK_TOKEN}|${DATE_TOKEN})[^，。！？\n]{0,10}(?:至|到|[-—~～])[^，。！？\n]{0,10}(?:${CLOCK_TOKEN}|${DATE_TOKEN})`,
  "u",
);
const APPOINTMENT_DURATION = /(?:持续|时长|耗时|为期|预计(?:需要|用时)?|安排(?:为|了)?)[^，。！？\n]{0,8}(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)\s*(?:分钟|小时|天)/u;
const APPOINTMENT_END_AFTER_MARKER = new RegExp(
  String.raw`(?:结束(?:时间)?|截至|截止|有效(?:期)?(?:到|至)|到期(?:时间)?)[^，。！？\n]{0,12}(?:${CLOCK_TOKEN}|${DATE_TOKEN})`,
  "u",
);
const APPOINTMENT_END_BEFORE_MARKER = new RegExp(
  String.raw`(?:${CLOCK_TOKEN}|${DATE_TOKEN})[^，。！？\n]{0,6}(?:结束|截止|到期)`,
  "u",
);
function stringValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.map(stringValue).filter((entry): entry is string => Boolean(entry)))];
}

function lineageEventTimeMs(value: unknown): number | undefined {
  if (typeof value !== "string" && typeof value !== "number") return undefined;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : undefined;
}

function lineageMatchesMemoryClass(
  entry: CockpitKnownLineage,
  memory: ExtractedMemory,
  memoryMetadata: Record<string, unknown>,
): boolean {
  const type = stringValue(entry.type ?? entry.metadata.type);
  const scene = canonicalCockpitSceneClass(
    entry.scene_name ?? entry.metadata.scene_name,
    entry.metadata.domain,
    entry.metadata.slot,
  );
  const memoryScene = canonicalCockpitSceneClass(
    memory.scene_name,
    memoryMetadata.domain,
    memoryMetadata.slot,
  );
  return (!type || type === memory.type) && (!scene || scene === memoryScene);
}

function lineageIsCausallyAvailable(
  entry: CockpitKnownLineage,
  memoryMetadata: Record<string, unknown>,
): boolean {
  const priorRaw = entry.metadata.mentioned_at;
  const currentRaw = memoryMetadata.mentioned_at;
  if (priorRaw === undefined && currentRaw === undefined) return true;
  const priorTime = lineageEventTimeMs(priorRaw);
  const currentTime = lineageEventTimeMs(currentRaw);
  return priorTime !== undefined && currentTime !== undefined && priorTime <= currentTime;
}

function lineageConstraintTargetMatches(
  entry: CockpitKnownLineage,
  memoryMetadata: Record<string, unknown>,
): boolean {
  const prior = normalizeKeyPart(entry.metadata.constraint_target);
  const current = normalizeKeyPart(memoryMetadata.constraint_target);
  return prior === current;
}

function controlledLineageStateIdentity(
  metadata: Record<string, unknown>,
): string | undefined {
  const slot = canonicalCockpitSlot(metadata.slot);
  const owners = controlledCockpitSlotOwners(slot);
  if (!slot || !owners || owners.length === 0) return undefined;
  const domain = canonicalControlledCockpitDomain(
    canonicalCockpitDomain(metadata.domain),
    slot,
  );
  return domain && owners.includes(domain) ? `${domain}|${slot}` : undefined;
}

/**
 * A record ID may repair representation-only domain/slot drift, but it may
 * never turn one recognized cockpit state into another.  Projecting only the
 * current row must therefore be outside the controlled ontology, and every
 * explicit person/scope field it did provide must already agree with the
 * predecessor. Missing scope may be completed by the exact record, but an
 * explicit conflicting scope is never overwritten.
 */
function lineageRecordIdentityMatches(
  entryMetadata: Record<string, unknown>,
  memoryMetadata: Record<string, unknown>,
  preliminaryStateKey: string | undefined,
  lineageStateKey: string | undefined,
): boolean {
  if (!lineageStateKey) return false;
  const priorQualifier = canonicalStateQualifier(entryMetadata.state_qualifier);
  const currentQualifier = canonicalStateQualifier(memoryMetadata.state_qualifier);
  // A matching serialized key cannot override conflicting structured axes.
  // Check the full qualifier before accepting even an exact key match.
  if (priorQualifier !== currentQualifier) return false;
  if (preliminaryStateKey === lineageStateKey) return true;
  const priorControlledIdentity = controlledLineageStateIdentity(entryMetadata);
  const currentControlledIdentity = controlledLineageStateIdentity(memoryMetadata);
  const scopesMatch = ["subject", "occupant_scope", "vehicle_scope", "seat_zone"].every((key) => {
    const proposed = normalizeScopeKeyPart(memoryMetadata[key]);
    return proposed === undefined
      || proposed === normalizeScopeKeyPart(entryMetadata[key]);
  });
  // A model-selected predecessor may never choose a named-map member by
  // itself. Missing qualifiers are completed earlier only from an exact source
  // label or one unique compatible live prior; conflicts fail closed here.
  if (priorQualifier !== currentQualifier) return false;
  // Permit an exact predecessor to bridge only the serialized state-key
  // version (legacy readable suffix -> collision-safe q2 digest). All factual
  // identity axes, including the full qualifier, must already agree.
  if (priorControlledIdentity
    && currentControlledIdentity === priorControlledIdentity
    && scopesMatch) {
    const priorBaseState = buildStateKey({ ...entryMetadata, state_qualifier: undefined });
    const currentBaseState = buildStateKey({ ...memoryMetadata, state_qualifier: undefined });
    return Boolean(priorBaseState && priorBaseState === currentBaseState);
  }
  if (!priorControlledIdentity || currentControlledIdentity) return false;
  if (!canonicalCockpitDomain(memoryMetadata.domain)
    || !canonicalCockpitSlot(memoryMetadata.slot)) return false;
  return scopesMatch;
}

function normalizedStringArray(value: unknown): { values: string[]; repairedScalar: boolean; dropped: boolean } {
  if (Array.isArray(value)) {
    const values = stringArray(value);
    return { values, repairedScalar: false, dropped: values.length !== value.length };
  }
  const scalar = stringValue(value);
  return {
    values: scalar ? [scalar] : [],
    repairedScalar: Boolean(scalar),
    dropped: value !== undefined && !scalar,
  };
}

function normalizeKeyPart(value: unknown): string | undefined {
  const text = stringValue(value)?.normalize("NFKC").toLocaleLowerCase();
  if (!text) return undefined;
  const normalized = text.replace(SAFE_KEY_PART, "-").replace(/^-+|-+$/gu, "");
  return normalized || undefined;
}

function canonicalStateQualifier(value: unknown): string | undefined {
  const text = stringValue(value)?.normalize("NFKC").trim();
  if (!text || Array.from(text).length > 64 || /[\p{Cc}\p{Cs}\r\n]/u.test(text)) return undefined;
  const comparable = text.toLocaleLowerCase();
  if (GENERIC_STATE_QUALIFIERS.has(comparable)) return undefined;
  // Keep the identity label compact and source-verifiable. Punctuation used
  // inside an address or sentence is not a safe state-map key.
  if (!/^[\p{L}\p{N} ._\-/]+$/u.test(text)) return undefined;
  return text;
}

function stateQualifierKeyPart(value: unknown): string | undefined {
  const qualifier = canonicalStateQualifier(value);
  if (!qualifier) return undefined;
  // Preserve NFKC-equivalence while retaining every other character boundary.
  // A full digest avoids the deterministic slash/space collisions introduced
  // by lossy key slugs. The human-readable label remains in metadata.
  const canonical = qualifier.normalize("NFKC").trim();
  return `q2-${createHash("sha256").update(canonical, "utf8").digest("hex")}`;
}

function lineageScopesMatch(
  entryMetadata: Record<string, unknown>,
  memoryMetadata: Record<string, unknown>,
): boolean {
  return ["subject", "occupant_scope", "vehicle_scope", "seat_zone"].every((key) => {
    const proposed = normalizeScopeKeyPart(memoryMetadata[key]);
    return proposed === undefined
      || proposed === normalizeScopeKeyPart(entryMetadata[key]);
  });
}

function liveKnownLineage(
  knownLineage: CockpitKnownLineage[] | undefined,
): CockpitKnownLineage[] {
  if (!knownLineage) return [];
  const supersededRecordIds = new Set(knownLineage.flatMap((entry) =>
    stringArray(entry.metadata.supersedes)
  ));
  return knownLineage.filter((entry) => {
    if (entry.recordId && supersededRecordIds.has(entry.recordId)) return false;
    const relation = stringValue(entry.metadata.relation)?.toLocaleLowerCase();
    const status = stringValue(entry.metadata.action_status)?.toLocaleLowerCase();
    return relation !== "cancelled" && relation !== "negated" && status !== "cancelled";
  });
}

function compatibleLiveLineageQualifiers(
  knownLineage: CockpitKnownLineage[] | undefined,
  memoryMetadata: Record<string, unknown>,
): Set<string> {
  const currentIdentity = controlledLineageStateIdentity(memoryMetadata);
  if (!currentIdentity) return new Set();
  const currentBaseState = buildStateKey({ ...memoryMetadata, state_qualifier: undefined });
  if (!currentBaseState) return new Set();
  return new Set(liveKnownLineage(knownLineage).flatMap((entry) => {
    if (controlledLineageStateIdentity(entry.metadata) !== currentIdentity
      || !lineageScopesMatch(entry.metadata, memoryMetadata)
      || !lineageConstraintTargetMatches(entry, memoryMetadata)
      || !lineageIsCausallyAvailable(entry, memoryMetadata)
      || buildStateKey({ ...entry.metadata, state_qualifier: undefined }) !== currentBaseState) {
      return [];
    }
    const qualifier = canonicalStateQualifier(entry.metadata.state_qualifier);
    return qualifier ? [qualifier] : [];
  }));
}

function exactSupersededLineageSupportsQualifier(
  knownLineage: CockpitKnownLineage[] | undefined,
  supersedes: string[],
  qualifier: string,
  memoryMetadata: Record<string, unknown>,
): boolean {
  if (!knownLineage || supersedes.length === 0) return false;
  const currentIdentity = controlledLineageStateIdentity(memoryMetadata);
  if (!currentIdentity) return false;
  const matches = knownLineage.filter((entry) =>
    Boolean(entry.recordId && supersedes.includes(entry.recordId))
      && canonicalStateQualifier(entry.metadata.state_qualifier) === qualifier
      && controlledLineageStateIdentity(entry.metadata) === currentIdentity
      && lineageConstraintTargetMatches(entry, memoryMetadata)
  );
  const compatibleQualifiers = compatibleLiveLineageQualifiers(knownLineage, memoryMetadata);
  return matches.length === 1
    && compatibleQualifiers.size === 1
    && compatibleQualifiers.has(qualifier);
}

function recoverMissingStateQualifier(params: {
  knownLineage: CockpitKnownLineage[] | undefined;
  supersedes: string[];
  memoryMetadata: Record<string, unknown>;
  sourceQualifiers: ReadonlySet<string>;
}): string | undefined {
  const { knownLineage, supersedes, memoryMetadata, sourceQualifiers } = params;
  if (!knownLineage || supersedes.length === 0) return undefined;
  const exactReferencedQualifiers = new Set(knownLineage.flatMap((entry) => {
    if (!entry.recordId || !supersedes.includes(entry.recordId)
      || !lineageScopesMatch(entry.metadata, memoryMetadata)
      || !lineageConstraintTargetMatches(entry, memoryMetadata)
      || !lineageIsCausallyAvailable(entry, memoryMetadata)) return [];
    const qualifier = canonicalStateQualifier(entry.metadata.state_qualifier);
    return qualifier ? [qualifier] : [];
  }));
  if (exactReferencedQualifiers.size !== 1) return undefined;
  const qualifier = [...exactReferencedQualifiers][0];
  if (sourceQualifiers.has(qualifier)) return qualifier;
  const compatibleQualifiers = compatibleLiveLineageQualifiers(knownLineage, memoryMetadata);
  return compatibleQualifiers.size === 1 && compatibleQualifiers.has(qualifier)
    ? qualifier
    : undefined;
}

function meaningfulScopeValue(value: unknown): string | undefined {
  const text = stringValue(value);
  if (!text || UNKNOWN_SCOPE_SENTINELS.has(text.normalize("NFKC").toLocaleLowerCase())) return undefined;
  return text;
}

function normalizeScopeKeyPart(value: unknown): string | undefined {
  return normalizeKeyPart(meaningfulScopeValue(value));
}

function normalizeRelation(metadata: Record<string, unknown>): CockpitStateRelation {
  const explicit = stringValue(metadata.relation)?.toLocaleLowerCase();
  if (explicit === "updated" || explicit === "cancelled" || explicit === "negated" || explicit === "asserted") {
    return explicit;
  }
  const status = stringValue(metadata.action_status)?.toLocaleLowerCase();
  if (status === "cancelled") return "cancelled";
  // Lifecycle status describes the outcome of this event, not whether it
  // replaces another memory. Only an explicit relation or a verified
  // supersedes edge may turn an assertion into an update.
  if (stringArray(metadata.supersedes).length > 0) return "updated";
  return "asserted";
}

function sourceObservedAt(messages: ConversationMessage[], ids: string[]): string | undefined {
  const allowed = new Set(ids);
  const timestamps = messages
    .filter((message) => allowed.has(message.id) && Number.isFinite(message.timestamp))
    .map((message) => message.timestamp);
  if (timestamps.length === 0) return undefined;
  return new Date(Math.max(...timestamps)).toISOString();
}

function boundedAppointmentEvidenceText(
  text: string,
  spans: Array<{ start: number; end: number }>,
): string[] {
  const normalized = text.normalize("NFKC");
  return spans.flatMap((span) => {
    if (!Number.isInteger(span.start) || !Number.isInteger(span.end)
      || span.start < 0 || span.end <= span.start || span.end > normalized.length) return [];
    let start = span.start;
    let end = span.end;
    while (start > 0 && !/[，,。！？!?；;\n、]/u.test(normalized[start - 1])) start -= 1;
    while (end < normalized.length && !/[，,。！？!?；;\n、]/u.test(normalized[end])) end += 1;
    const segment = normalized.slice(start, end);
    const relativeStart = span.start - start;
    const relativeEnd = span.end - start;
    const conjunctions = [...segment.matchAll(/(?:以及|和|与)(?=\s*(?:\d{1,2}月\d{1,2}日|今天|明天|后天|本周|下周|周[一二三四五六日天]))/gu)];
    let localStart = 0;
    let localEnd = segment.length;
    for (const conjunction of conjunctions) {
      const boundaryStart = conjunction.index ?? 0;
      const boundaryEnd = boundaryStart + conjunction[0].length;
      if (boundaryEnd <= relativeStart) localStart = Math.max(localStart, boundaryEnd);
      if (boundaryStart >= relativeEnd) localEnd = Math.min(localEnd, boundaryStart);
    }
    return [segment.slice(localStart, localEnd)];
  });
}

function hasExplicitAppointmentEndEvidence(
  messages: ConversationMessage[],
  ids: string[],
  sourceEvidenceSpans?: Array<{ start: number; end: number }>,
): boolean {
  const allowed = new Set(ids);
  return messages.some((message) => {
    if (!allowed.has(message.id) || message.role !== "user") return false;
    const evidenceTexts = sourceEvidenceSpans && ids.length === 1
      ? boundedAppointmentEvidenceText(message.content, sourceEvidenceSpans)
      : [message.content.normalize("NFKC")];
    return evidenceTexts.some((text) =>
      APPOINTMENT_TIME_RANGE.test(text)
      || APPOINTMENT_DURATION.test(text)
      || APPOINTMENT_END_AFTER_MARKER.test(text)
      || APPOINTMENT_END_BEFORE_MARKER.test(text)
    );
  });
}

function isExplicitToolEvidence(message: ConversationMessage): boolean {
  return /(?:\[|\{|\b)["']?source_role["']?\s*[:=]\s*["']?tool["']?(?:\]|\}|\b)/iu.test(message.content)
    || /["']?type["']?\s*:\s*["']tool_result["']/iu.test(message.content);
}

function explicitSpeakerIdentity(message: ConversationMessage): string | undefined {
  if (message.role !== "user") return undefined;
  // Treat only a near-leading transcript envelope as identity evidence. A
  // short ASR filler may precede the label; later names in natural-language
  // content are never promoted by this rule.
  const payload = message.content.replace(/^\s*(?:\[[^\]\r\n]{1,256}\]\s*)*/u, "");
  const match = /^[^【】\r\n]{0,32}【([^】\r\n]{1,64})】/u.exec(payload);
  const identity = match?.[1]?.trim();
  if (!identity || GENERIC_FIRST_PERSON_SUBJECTS.has(identity.normalize("NFKC").toLocaleLowerCase())) {
    return undefined;
  }
  return identity;
}

function buildStateKey(metadata: Record<string, unknown>): string | undefined {
  const domain = normalizeKeyPart(metadata.domain);
  const slot = normalizeKeyPart(metadata.slot);
  if (!domain || !slot) return stringValue(metadata.state_key);
  const subject = normalizeScopeKeyPart(metadata.subject)
    ?? normalizeScopeKeyPart(metadata.occupant_scope)
    ?? "unspecified-subject";
  const vehicle = normalizeScopeKeyPart(metadata.vehicle_scope) ?? "unspecified-vehicle";
  const seat = normalizeScopeKeyPart(metadata.seat_zone) ?? "unspecified-zone";
  // A price constraint is a small keyed state map: ticket, per-capita and
  // room budgets may coexist in one planning episode. Keep the legacy
  // five-part state-key shape while qualifying only this controlled slot.
  const constraintTarget = domain === "selection" && slot === "price_constraint"
    ? normalizeKeyPart(metadata.constraint_target)
    : undefined;
  const stateQualifier = stateQualifierKeyPart(metadata.state_qualifier);
  const slotQualifiers = [constraintTarget, stateQualifier]
    .filter((value): value is string => Boolean(value));
  const slotIdentity = slotQualifiers.length > 0
    ? `${slot}@${slotQualifiers.join("@")}`
    : slot;
  return `${domain}|${subject}|${vehicle}|${seat}|${slotIdentity}`;
}

function stableQualifiedStateEpisodeKey(
  memory: ExtractedMemory,
  metadata: Record<string, unknown>,
): string | undefined {
  const stateQualifier = canonicalStateQualifier(metadata.state_qualifier);
  const stateKey = buildStateKey(metadata);
  const isPersistentStateClass = memory.type === "persona" || memory.type === "instruction";
  const hasEventPartition = [
    "valid_from",
    "valid_to",
    "activity_start_time",
    "activity_end_time",
    "condition",
    "trigger",
  ].some((key) => {
    const value = metadata[key];
    return value !== undefined && value !== null
      && (typeof value !== "string" || value.trim().length > 0);
  });
  const hasTransientStatus = Boolean(stringValue(metadata.action_status));
  if (stateQualifier && stateKey
    && isPersistentStateClass
    && metadata.relation === "asserted"
    && stringArray(metadata.supersedes).length === 0
    && metadata.record_kind !== "event"
    && !hasEventPartition
    && !hasTransientStatus) {
    const digest = createHash("sha256").update(stateKey).digest("hex");
    return `qualified-state:${digest}`;
  }
  return undefined;
}

/** True only for a contract-derived, eligibility-checked stable state episode. */
export function hasCanonicalCockpitQualifiedStateEpisode(memory: ExtractedMemory): boolean {
  const metadata = memory.metadata as Record<string, unknown>;
  const expected = stableQualifiedStateEpisodeKey(memory, metadata);
  return expected !== undefined && stringValue(metadata.episode_key) === expected;
}

function buildEpisodeKey(
  memory: ExtractedMemory,
  metadata: Record<string, unknown>,
  sessionId: string | undefined,
  sourceIds: string[],
  verifiedLineageEpisodeKey?: string,
): string | undefined {
  const stableEpisodeKey = stableQualifiedStateEpisodeKey(memory, metadata);
  if (stableEpisodeKey) return stableEpisodeKey;
  const explicit = stringValue(metadata.episode_key);
  if (explicit && (!explicit.startsWith("qualified-state:")
    || explicit === verifiedLineageEpisodeKey)) return explicit;
  if (!sessionId || sourceIds.length === 0) return undefined;
  return `${sessionId}|${sourceIds.join("+")}`;
}

/**
 * Normalize one Flash-produced cockpit memory without silently repairing
 * semantic facts. Defects remain queryable in construction_quality and are
 * used by the Proxy's evidence-completeness gate.
 */
export function normalizeCockpitExtractedMemory(params: NormalizeCockpitMemoryParams): ExtractedMemory {
  const { memory, sourceMessages, sessionId, constructionModel, knownLineage, sourceEvidenceSpans } = params;
  const metadata = { ...(memory.metadata as Record<string, unknown>) };
  const issues: string[] = [];
  const repairs: string[] = [];

  const normalizedSupersedes = normalizedStringArray(metadata.supersedes);
  if (normalizedSupersedes.values.length > 0) metadata.supersedes = normalizedSupersedes.values;
  else if (metadata.supersedes !== undefined) delete metadata.supersedes;
  if (normalizedSupersedes.repairedScalar) repairs.push("normalized_scalar_supersedes");
  if (normalizedSupersedes.dropped) repairs.push("removed_invalid_supersedes_entry");

  const knownById = new Map(sourceMessages.map((message) => [message.id, message]));
  const proposedIds = [...new Set([
    ...memory.source_message_ids,
    ...stringArray(metadata.source_message_ids),
  ])];
  const invalidSourceIds = proposedIds.filter((id) => !knownById.has(id));
  const unsupportedRoleSourceIds = proposedIds.filter((id) => {
    const source = knownById.get(id);
    return source !== undefined && source.role !== "user" && !isExplicitToolEvidence(source);
  });
  // Current L0 stores user/assistant turns. Ordinary assistant text may resolve
  // a reference, but only user turns or explicitly tagged tool results are
  // factual sources.
  const sourceIds = proposedIds.filter((id) => {
    const source = knownById.get(id);
    return source?.role === "user" || Boolean(source && isExplicitToolEvidence(source));
  });
  const userSourceIds = sourceIds.filter((id) => knownById.get(id)?.role === "user");
  const toolSourceIds = sourceIds.filter((id) => {
    const source = knownById.get(id);
    return Boolean(source && isExplicitToolEvidence(source));
  });
  const sourceRoles = [
    ...(userSourceIds.length > 0 ? ["user"] : []),
    ...(toolSourceIds.length > 0 ? ["tool"] : []),
  ];
  const observedAt = sourceObservedAt(sourceMessages, sourceIds);

  if (invalidSourceIds.length > 0) {
    if (userSourceIds.length > 0) repairs.push("removed_unknown_source_message_id");
    else issues.push("unknown_source_message_id");
  }
  if (unsupportedRoleSourceIds.length > 0) {
    if (userSourceIds.length > 0) repairs.push("removed_unsupported_source_role");
    else issues.push("unsupported_source_role");
  }
  if (userSourceIds.length === 0) issues.push("missing_user_source");
  if (userSourceIds.length > 0 && userSourceIds.every((id) => {
    const source = knownById.get(id);
    return Boolean(source && isCockpitPureInformationalQuery(source.content));
  })) {
    // Flash may turn a question about a value into an asserted preference.
    // A second model agreeing with that hallucination is not independent
    // evidence, so the deterministic construction contract fails it closed.
    issues.push("informational_query_source");
  }

  const subject = stringValue(metadata.subject)?.normalize("NFKC").toLocaleLowerCase();
  if (subject && GENERIC_FIRST_PERSON_SUBJECTS.has(subject)) {
    const sourceSpeakers = [...new Set(userSourceIds
      .map((id) => knownById.get(id))
      .filter((message): message is ConversationMessage => message !== undefined)
      .map(explicitSpeakerIdentity)
      .filter((identity): identity is string => Boolean(identity)))];
    if (sourceSpeakers.length === 1) {
      metadata.subject = sourceSpeakers[0];
      repairs.push("bound_first_person_subject_to_explicit_speaker");
    } else if (sourceSpeakers.length > 1) {
      issues.push("ambiguous_source_speaker");
    }
  }

  metadata.schema_version = COCKPIT_STATE_SCHEMA_VERSION;
  metadata.record_kind = memory.type === "episodic" ? "event" : "state_assertion";
  metadata.relation = normalizeRelation(metadata);
  const proposedDomain = stringValue(metadata.domain);
  const proposedSlot = stringValue(metadata.slot);
  const canonicalSlot = canonicalCockpitSlot(proposedSlot);
  const canonicalDomain = canonicalControlledCockpitDomain(
    canonicalCockpitDomain(proposedDomain),
    canonicalSlot,
  );
  if (canonicalSlot) {
    metadata.slot = canonicalSlot;
    if (proposedSlot !== canonicalSlot) repairs.push("canonicalized_slot_token");
  }
  if (canonicalDomain) {
    metadata.domain = canonicalDomain;
    if (proposedDomain !== canonicalDomain) repairs.push("canonicalized_controlled_slot_domain");
  }
  const normalizedSceneName = canonicalCockpitSceneClass(
    memory.scene_name,
    metadata.domain,
    metadata.slot,
  ) ?? memory.scene_name;
  if (normalizedSceneName !== memory.scene_name) {
    repairs.push("canonicalized_scene_name_to_controlled_domain");
  }
  const proposedConstraintTarget = stringValue(metadata.constraint_target);
  if (metadata.domain === "selection" && metadata.slot === "price_constraint") {
    const canonicalConstraintTarget = canonicalCockpitConstraintTarget(proposedConstraintTarget);
    if (canonicalConstraintTarget) {
      metadata.constraint_target = canonicalConstraintTarget;
      if (proposedConstraintTarget !== canonicalConstraintTarget) {
        repairs.push("canonicalized_constraint_target");
      }
    } else {
      issues.push("missing_constraint_target");
    }
  } else if (metadata.constraint_target !== undefined) {
    delete metadata.constraint_target;
    repairs.push("removed_constraint_target_from_non_price_slot");
  }
  if (metadata.domain === "schedule"
    && (metadata.slot === "appointment_time" || metadata.slot === "appointment_content")
    && (metadata.valid_to !== undefined || metadata.activity_end_time !== undefined)
    && !hasExplicitAppointmentEndEvidence(sourceMessages, sourceIds, sourceEvidenceSpans)) {
    // A point appointment time does not imply a one-hour validity interval.
    // Keep model-parsed end times only when the factual sources explicitly
    // contain an end, range, or duration expression.
    delete metadata.valid_to;
    delete metadata.activity_end_time;
    repairs.push("removed_ungrounded_appointment_valid_to");
    repairs.push("removed_ungrounded_appointment_activity_end_time");
  }
  metadata.source_message_ids = sourceIds;
  metadata.evidence_roles = sourceRoles;
  if (sessionId) {
    const proposedSourceSessionId = stringValue(metadata.source_session_id);
    const proposedSourceSessionIds = stringArray(metadata.source_session_ids);
    if ((proposedSourceSessionId && proposedSourceSessionId !== sessionId)
      || proposedSourceSessionIds.some((id) => id !== sessionId)) {
      repairs.push("bound_source_session_to_evidence");
    }
    metadata.source_session_id = sessionId;
    metadata.source_session_ids = [sessionId];
  }
  const proposedMentionedAt = stringValue(metadata.mentioned_at);
  if (observedAt) {
    metadata.mentioned_at = observedAt;
    if (proposedMentionedAt) {
      const proposedMs = new Date(proposedMentionedAt).getTime();
      if (!Number.isFinite(proposedMs) || proposedMs !== new Date(observedAt).getTime()) {
        repairs.push("bound_mentioned_at_to_evidence");
      }
    }
  }
  if (constructionModel) {
    const proposedModel = stringValue(metadata.construction_model);
    if (proposedModel && proposedModel !== constructionModel) repairs.push("corrected_construction_model");
    metadata.construction_model = constructionModel;
  }
  metadata.construction_stage = "l1";

  // `target` is a legacy scalar used by current consumers. Keep it aligned
  // with the typed `value` only when one side is explicitly available.
  if (metadata.value === undefined && metadata.target !== undefined) metadata.value = metadata.target;
  if (metadata.target === undefined && ["string", "number", "boolean"].includes(typeof metadata.value)) {
    metadata.target = metadata.value;
  }
  const scalarValue = stringValue(metadata.value ?? metadata.target);
  if (scalarValue
    && isControlledCockpitEventTimeSlot(metadata.domain, metadata.slot)
    && strictZonedIsoInstant(scalarValue) !== undefined
    && !stringValue(metadata.activity_start_time)) {
    // This is a representation-only projection of an already accepted fact:
    // it copies the exact zoned timestamp instead of parsing source prose or
    // inventing a date. Retrieval can therefore use the structured event-time
    // axis even when Flash placed the ISO value only in the slot value.
    metadata.activity_start_time = scalarValue;
    repairs.push("projected_iso_time_value_to_activity_start_time");
  }
  const factValue = metadata.value ?? metadata.target;
  if (typeof factValue === "string"
    && PLACEHOLDER_FACT_VALUES.has(factValue.normalize("NFKC").trim().toLocaleLowerCase())) {
    issues.push("placeholder_value");
  }

  const proposedStateQualifier = stringValue(metadata.state_qualifier);
  const namedStatePriorContext = (knownLineage ?? []).map((entry) => ({
    record_id: entry.recordId,
    metadata: entry.metadata,
  }));
  const namedStateSubjectScope = metadata.subject ?? metadata.occupant_scope;
  const sourceStateQualifiers = new Set(namedStateSubjectScope === undefined ? [] : userSourceIds.flatMap((id) => {
    const source = knownById.get(id);
    return source
      ? extractCockpitAuthorizedNamedDestinationStateQualifiers(
        source,
        namedStatePriorContext,
        namedStateSubjectScope,
      )
      : [];
  }));
  if (metadata.state_qualifier === undefined) {
    const recoveredQualifier = recoverMissingStateQualifier({
      knownLineage,
      supersedes: stringArray(metadata.supersedes),
      memoryMetadata: metadata,
      sourceQualifiers: sourceStateQualifiers,
    });
    if (recoveredQualifier) {
      metadata.state_qualifier = recoveredQualifier;
      repairs.push("reused_unambiguous_superseded_state_qualifier");
    }
  }
  if (metadata.state_qualifier !== undefined) {
    const canonicalQualifier = canonicalStateQualifier(metadata.state_qualifier);
    const qualifierMatchesFactValue = canonicalQualifier !== undefined
      && typeof factValue === "string"
      && canonicalQualifier.toLocaleLowerCase()
        === factValue.normalize("NFKC").trim().toLocaleLowerCase();
    const sourceSupportsQualifier = canonicalQualifier !== undefined
      && sourceStateQualifiers.has(canonicalQualifier);
    const lineageSupportsQualifier = canonicalQualifier !== undefined
      && exactSupersededLineageSupportsQualifier(
        knownLineage,
        stringArray(metadata.supersedes),
        canonicalQualifier,
        metadata,
      );
    if (!canonicalQualifier || qualifierMatchesFactValue) {
      delete metadata.state_qualifier;
      issues.push("invalid_state_qualifier");
    } else if (!sourceSupportsQualifier && !lineageSupportsQualifier) {
      delete metadata.state_qualifier;
      issues.push("unverified_state_qualifier");
    } else {
      metadata.state_qualifier = canonicalQualifier;
      if (proposedStateQualifier !== canonicalQualifier) repairs.push("canonicalized_state_qualifier");
    }
  }

  const terminalStatus = stringValue(metadata.action_status)?.toLocaleLowerCase();
  if ((terminalStatus === "executed" || terminalStatus === "verified" || terminalStatus === "completed")
    && toolSourceIds.length === 0) {
    metadata.action_status = "requested";
    repairs.push("downgraded_action_without_tool_evidence");
  }

  const proposedStateKey = stringValue(metadata.state_key);
  const proposedEpisodeKey = stringValue(metadata.episode_key);
  const preliminaryStateKey = buildStateKey(metadata);
  const relation = metadata.relation as CockpitStateRelation;
  const supersedes = stringArray(metadata.supersedes);

  // A transition may deterministically reuse identity only when its explicit
  // supersedes reference resolves to authoritative prior/batch structured
  // metadata. This repairs model vocabulary drift without inferring a fact.
  let reusedLineageStateKey: string | undefined;
  let verifiedLineageEpisodeKey: string | undefined;
  if (knownLineage !== undefined && supersedes.length > 0) {
    const resolved: Array<{ metadata: Record<string, unknown>; stateKey?: string; episodeKey?: string }> = [];
    let hasUnverifiedReference = false;
    const lineageEntries = knownLineage.map((entry) => ({
      entry,
      stateKey: stringValue(entry.metadata.state_key) ?? buildStateKey(entry.metadata),
      episodeKey: stringValue(entry.metadata.episode_key),
    }));
    for (const reference of supersedes) {
      // Exact record IDs take precedence over an episode alias with the same
      // string. This prevents one identifier collision from widening a single
      // predecessor edge into an episode-wide match set.
      const referencesRecordId = lineageEntries.some(({ entry }) =>
        entry.recordId === reference
      );
      const matches = lineageEntries
        .filter(({ entry, episodeKey }) =>
          (referencesRecordId ? entry.recordId === reference : episodeKey === reference)
            && lineageMatchesMemoryClass(entry, memory, metadata)
            && lineageConstraintTargetMatches(entry, metadata)
            && lineageIsCausallyAvailable(entry, metadata)
        )
        .filter((match) => referencesRecordId
          ? lineageRecordIdentityMatches(
            match.entry.metadata,
            metadata,
            preliminaryStateKey,
            match.stateKey,
          )
          : Boolean(preliminaryStateKey && match.stateKey === preliminaryStateKey));
      // Episode aliases must identify one exact state; record IDs must also be
      // unique in the bounded lineage context. Ambiguity is never repaired by
      // array order.
      if (matches.length !== 1) {
        hasUnverifiedReference = true;
        continue;
      }
      resolved.push({
        metadata: matches[0].entry.metadata,
        stateKey: matches[0].stateKey,
        episodeKey: matches[0].episodeKey,
      });
    }
    if (hasUnverifiedReference) issues.push("unverified_supersedes");

    if (!hasUnverifiedReference && resolved.length > 0) {
      const resolvedStateKeys = [...new Set(resolved.map((entry) => entry.stateKey).filter((key): key is string => Boolean(key)))];
      const resolvedEpisodeKeys = [...new Set(resolved.map((entry) => entry.episodeKey).filter((key): key is string => Boolean(key)))];
      if (resolvedStateKeys.length === 1) {
        const stateKey = resolvedStateKeys[0];
        const identity = resolved.find((entry) => entry.stateKey === stateKey)?.metadata;
        let reusedIdentity = metadata.state_key !== stateKey;
        metadata.state_key = stateKey;
        reusedLineageStateKey = stateKey;
        if (identity) {
          for (const key of [
            "domain",
            "slot",
            "subject",
            "occupant_scope",
            "vehicle_scope",
            "seat_zone",
            "constraint_target",
            "state_qualifier",
          ] as const) {
            if (identity[key] !== undefined && metadata[key] !== identity[key]) {
              metadata[key] = identity[key];
              reusedIdentity = true;
            }
          }
        }
        if (reusedIdentity) repairs.push("reused_superseded_state_identity");
      } else if (resolvedStateKeys.length > 1) {
        issues.push("ambiguous_transition_state");
      }
      if (resolvedEpisodeKeys.length === 1 && resolvedStateKeys.length === 1) {
        if (metadata.episode_key !== resolvedEpisodeKeys[0]) repairs.push("reused_superseded_episode_key");
        metadata.episode_key = resolvedEpisodeKeys[0];
        verifiedLineageEpisodeKey = resolvedEpisodeKeys[0];
      } else if (resolvedEpisodeKeys.length > 1) {
        issues.push("ambiguous_transition_episode");
      }
    }
  }

  const stateKey = reusedLineageStateKey ?? buildStateKey(metadata);
  if (stateKey) {
    metadata.state_key = stateKey;
    if (!proposedStateKey) repairs.push("derived_state_key");
    else if (proposedStateKey !== stateKey) repairs.push("canonicalized_state_key");
  }
  const stableEpisodeKey = stableQualifiedStateEpisodeKey(memory, metadata);
  const episodeKey = buildEpisodeKey(
    memory,
    metadata,
    sessionId,
    sourceIds,
    verifiedLineageEpisodeKey,
  );
  if (episodeKey) {
    metadata.episode_key = episodeKey;
    if (!proposedEpisodeKey) repairs.push("derived_episode_key");
    else if (proposedEpisodeKey !== episodeKey && episodeKey !== verifiedLineageEpisodeKey) {
      repairs.push(stableEpisodeKey === episodeKey
        ? "canonicalized_qualified_state_episode_key"
        : "replaced_ineligible_qualified_state_episode_key");
    }
  } else if (proposedEpisodeKey?.startsWith("qualified-state:")) {
    delete metadata.episode_key;
    repairs.push("removed_ineligible_qualified_state_episode_key");
  }

  if (!stringValue(metadata.domain)) issues.push("missing_domain");
  if (!stringValue(metadata.slot)) issues.push("missing_slot");
  if (metadata.value === undefined && metadata.target === undefined) issues.push("missing_value");
  if (!stringValue(metadata.state_key)) issues.push("missing_state_key");
  if (![metadata.subject, metadata.occupant_scope, metadata.vehicle_scope, metadata.seat_zone]
    .some((value) => Boolean(meaningfulScopeValue(value)))) issues.push("missing_scope");
  if (memory.type === "episodic" && !stringValue(metadata.action_status)) issues.push("missing_action_status");
  if ((relation === "updated" || relation === "cancelled" || relation === "negated")
    && stringArray(metadata.supersedes).length === 0) {
    issues.push("missing_supersedes");
  }

  const uniqueIssues = [...new Set(issues)];
  const uniqueRepairs = [...new Set(repairs)];
  const invalid = uniqueIssues.includes("missing_user_source");
  const quality: CockpitConstructionQuality = {
    status: invalid ? "invalid" : uniqueIssues.length > 0 ? "partial" : "complete",
    score: Math.max(0, 100
      - (invalid ? 50 : 0)
      - uniqueIssues.filter((issue) => issue !== "missing_user_source" && issue !== "unknown_source_message_id").length * 10),
    issues: uniqueIssues,
    repairs: uniqueRepairs,
    source_count: sourceIds.length,
    user_source_count: userSourceIds.length,
  };
  metadata.construction_quality = quality;

  return {
    ...memory,
    scene_name: normalizedSceneName,
    source_message_ids: sourceIds,
    metadata,
  };
}

/** Add a fail-closed batch-level issue without discarding per-record evidence. */
export function addCockpitConstructionIssue(
  memory: ExtractedMemory,
  issue: string,
): ExtractedMemory {
  const metadata = { ...(memory.metadata as Record<string, unknown>) };
  const rawQuality = metadata.construction_quality;
  const current = rawQuality && typeof rawQuality === "object" && !Array.isArray(rawQuality)
    ? rawQuality as Record<string, unknown>
    : {};
  const issues = [...new Set([...stringArray(current.issues), issue])];
  const invalid = current.status === "invalid" || issues.includes("missing_user_source");
  const computedScore = Math.max(0, 100
    - (invalid ? 50 : 0)
    - issues.filter((entry) => entry !== "missing_user_source" && entry !== "unknown_source_message_id").length * 10);
  metadata.construction_quality = {
    ...current,
    status: invalid ? "invalid" : "partial",
    score: typeof current.score === "number" && Number.isFinite(current.score)
      ? Math.min(current.score, computedScore)
      : computedScore,
    issues,
    repairs: stringArray(current.repairs),
    source_count: typeof current.source_count === "number"
      ? current.source_count
      : memory.source_message_ids.length,
    user_source_count: typeof current.user_source_count === "number"
      ? current.user_source_count
      : memory.source_message_ids.length,
  } satisfies CockpitConstructionQuality;
  return { ...memory, metadata };
}

/**
 * Complete lineage fields that only become knowable after TencentDB's
 * original dedup decision. `update` is a semantic replacement; `merge` is a
 * physical consolidation and must not masquerade as a state transition.
 */
export function finalizeCockpitMetadataAfterDedup(
  value: Record<string, unknown>,
  action: CockpitDedupAction,
  targetIds: string[],
): Record<string, unknown> {
  if (value.schema_version !== COCKPIT_STATE_SCHEMA_VERSION || targetIds.length === 0) return value;
  const metadata = { ...value };
  if (action === "update") {
    metadata.supersedes = [...new Set([...stringArray(metadata.supersedes), ...targetIds])];
    const relation = stringValue(metadata.relation);
    if (relation !== "cancelled" && relation !== "negated") metadata.relation = "updated";
  } else if (action === "merge") {
    metadata.merged_from_record_ids = [
      ...new Set([...stringArray(metadata.merged_from_record_ids), ...targetIds]),
    ];
  }

  const quality = metadata.construction_quality;
  if (quality && typeof quality === "object" && !Array.isArray(quality)) {
    const current = quality as Record<string, unknown>;
    const issues = stringArray(current.issues).filter((issue) =>
      !(issue === "missing_supersedes" && action === "update" && targetIds.length > 0)
    );
    const invalid = issues.includes("missing_user_source") || issues.includes("unknown_source_message_id");
    metadata.construction_quality = {
      ...current,
      status: invalid ? "invalid" : issues.length > 0 ? "partial" : "complete",
      score: Math.max(0, 100
        - (invalid ? 50 : 0)
      - issues.filter((issue) => issue !== "missing_user_source" && issue !== "unknown_source_message_id").length * 10),
      issues,
      repairs: stringArray(current.repairs),
      source_count: typeof current.source_count === "number"
        ? current.source_count
        : stringArray(metadata.source_message_ids).length,
      user_source_count: typeof current.user_source_count === "number"
        ? current.user_source_count
        : stringArray(metadata.source_message_ids).length,
    } satisfies CockpitConstructionQuality;
  }
  return metadata;
}
