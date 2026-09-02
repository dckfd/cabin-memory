/**
 * L1 Memory Extractor: extracts structured memories from L0 conversation messages
 * using a single LLM call with JSON-mode structured output.
 *
 * v3: Aligned with Kenty's prompt — scene segmentation + memory extraction in one call,
 * followed by batch conflict detection.
 *
 * Pipeline:
 * 1. Read recent messages from L0 (split into background + new)
 * 2. Call LLM to extract scene-segmented memories
 * 3. Batch conflict detection against existing records
 * 4. Write to L1 JSONL files
 */

import { createHash } from "node:crypto";

import type { ConversationMessage } from "../conversation/l0-recorder.js";
import {
  COCKPIT_ATOMIC_COMPILER_SYSTEM_PROMPT,
  COCKPIT_COVERAGE_FACT_COMPILER_SYSTEM_PROMPT,
  COCKPIT_CONSTRUCTION_RECONCILER_SYSTEM_PROMPT,
  formatCockpitAtomicCompilerPrompt,
  formatCockpitCoverageFactCompilerPrompt,
  formatCockpitConstructionReconciliationPrompt,
  formatExtractionPrompt,
  getExtractMemoriesSystemPrompt,
  type MemoryPromptMode,
} from "../prompts/l1-extraction.js";
import { batchDedup } from "./l1-dedup.js";
import { writeMemory, generateMemoryId } from "./l1-writer.js";
import type { ExtractedMemory, MemoryRecord, MemoryType, DedupDecision } from "./l1-writer.js";
import {
  commitL1PersistencePlan,
  createL1PersistenceBatchId,
  createL1PersistenceRecordId,
  prepareL1PersistencePlan,
  resumeL1PersistenceTransaction,
  type L1PersistenceScope,
} from "./l1-persistence-transaction.js";
import {
  hasCanonicalCockpitQualifiedStateEpisode,
  normalizeCockpitExtractedMemory,
  type CockpitKnownLineage,
} from "./cockpit-memory-contract.js";
import {
  cockpitCoverageObligationCovers,
  gateCockpitConstructionReconciliation,
  type CockpitReconciliationDiagnostic,
} from "./cockpit-construction-reconciliation.js";
import { assembleCockpitConstructionReconciliation } from "./cockpit-construction-assembler.js";
import {
  detectCockpitSourceCoverageObligations,
  sourceCoverageObligationToMemory,
  type CockpitSourceCoverageObligation,
} from "./cockpit-source-coverage.js";
import {
  loadCockpitPriorMemoryContext,
  type CockpitPriorMemoryContext,
} from "./cockpit-prior-context.js";
import { CleanContextRunner } from "../../utils/clean-context-runner.js";
import { sanitizeJsonForParse, shouldExtractL1 } from "../../utils/sanitize.js";
import type { IMemoryStore } from "../store/types.js";
import type { EmbeddingService } from "../store/embedding.js";
import { report } from "../report/reporter.js";
import { metricProducer } from "../report/kafka-metric-producer.js";
import { reportL1LatencyMetrics } from "../report/metric-tracking-l1-latency.js";
import type { LLMRunner, Logger, TraceContext } from "../types.js";
import { buildTraceParams } from "../types.js";
import type { StorageAdapter } from "../storage/adapter.js";

const TAG = "[memory-tdai][l1-extractor]";

function throwIfAborted(signal: AbortSignal | undefined, context: string): void {
  if (!signal?.aborted) return;
  if (signal.reason instanceof Error) throw signal.reason;
  throw new Error(`${context}: aborted`);
}
const DEFAULT_L1_TIMEOUT_MS = 180_000;
const MAX_COCKPIT_CONSTRUCTION_ATTEMPTS = 2;
const MAX_COCKPIT_RECONCILIATION_GATE_ATTEMPTS = 3;
const MAX_COCKPIT_DIRECTED_CALLS_PER_EXTRACTION = 16;

class L1StructuredOutputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "L1StructuredOutputError";
  }
}

class L1ReconciliationGateError extends L1StructuredOutputError {
  constructor(message: string) {
    super(message);
    this.name = "L1ReconciliationGateError";
  }
}

function cockpitStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((entry): entry is string =>
    typeof entry === "string" && entry.trim().length > 0
  ))];
}

function sameCockpitStringSet(left: string[], right: string[]): boolean {
  const normalizedLeft = [...new Set(left)].sort();
  const normalizedRight = [...new Set(right)].sort();
  return normalizedLeft.length === normalizedRight.length
    && normalizedLeft.every((value, index) => value === normalizedRight[index]);
}

function comparableCockpitValue(value: unknown): string {
  if (value === undefined) return "<undefined>";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function canonicalCockpitFactValue(value: unknown): unknown {
  if (value === undefined) return "<undefined>";
  if (Array.isArray(value)) return value.map((entry) => canonicalCockpitFactValue(entry));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonicalCockpitFactValue(entry)]));
  }
  return value;
}

/** Type- and episode-neutral factual identity used for cardinality only. */
function cockpitFactualIdentityFingerprint(candidate: ExtractedMemory): string {
  const metadata = candidate.metadata as Record<string, unknown>;
  const supersedes = cockpitStringArray(metadata.supersedes).sort();
  return JSON.stringify(canonicalCockpitFactValue({
    domain: metadata.domain,
    slot: metadata.slot,
    state_key: metadata.state_key,
    subject: metadata.subject,
    occupant_scope: metadata.occupant_scope,
    vehicle_scope: metadata.vehicle_scope,
    seat_zone: metadata.seat_zone,
    constraint_target: metadata.constraint_target,
    state_qualifier: metadata.state_qualifier,
    value: metadata.value,
    target: metadata.target,
    unit: metadata.unit,
    condition: metadata.condition,
    trigger: metadata.trigger,
    valid_from: metadata.valid_from,
    valid_to: metadata.valid_to,
    activity_start_time: metadata.activity_start_time,
    activity_end_time: metadata.activity_end_time,
    timezone: metadata.timezone,
    time_precision: metadata.time_precision,
    temporal_status: metadata.temporal_status,
    relation: metadata.relation,
    supersedes,
    action_status: metadata.action_status,
  }));
}

/**
 * Semantic receipt used for the second blind-pass comparison.  Flash is free
 * to choose a different rendering/type/episode label for the same source
 * fact; those are canonicalized by the deterministic assembler afterwards.
 * Comparing them here made an otherwise valid set fail closed merely because
 * two independent generations chose different episode keys.  Facts, relation
 * edges and evidence bindings remain exact, so this does not admit omissions,
 * cross-slot values, or fabricated transitions.
 */
function cockpitBlindRepresentationFingerprint(candidate: ExtractedMemory): string {
  return JSON.stringify(canonicalCockpitFactValue({
    fact: cockpitFactualIdentityFingerprint(candidate),
  }));
}

function cockpitSourceBoundFactualFingerprint(candidate: ExtractedMemory): string {
  const metadata = candidate.metadata as Record<string, unknown>;
  return JSON.stringify({
    fact: cockpitFactualIdentityFingerprint(candidate),
    top_source_message_ids: [...new Set(candidate.source_message_ids)].sort(),
    metadata_source_message_ids: cockpitStringArray(metadata.source_message_ids).sort(),
  });
}

function cockpitCoverageCandidateMatchesObligation(
  obligation: CockpitSourceCoverageObligation,
  candidate: ExtractedMemory,
): boolean {
  const scaffold = sourceCoverageObligationToMemory(obligation);
  const metadata = candidate.metadata as Record<string, unknown>;
  const metadataSources = cockpitStringArray(metadata.source_message_ids);
  return isCompleteCockpitProposal(candidate)
    // Coverage is intentionally type-neutral. A persistent preference may be
    // an instruction while a one-off request may be episodic.
    // `scene_name` is a legacy, model-authored grouping label. The cockpit
    // state envelope is governed by canonical metadata domain/slot and exact
    // source bindings; a free-text scene label must not trigger a redundant
    // paid compiler pass for an otherwise authoritative atomic fact.
    && candidate.source_message_ids.length === 1
    && candidate.source_message_ids[0] === obligation.sourceMessageId
    && metadataSources.length === 1
    && metadataSources[0] === obligation.sourceMessageId
    && cockpitCoverageObligationCovers(scaffold, candidate)
    && comparableCockpitValue(metadata.constraint_target)
      === comparableCockpitValue(obligation.constraintTarget)
    && (!obligation.requiresStateQualifier
      || (typeof metadata.state_qualifier === "string"
        && metadata.state_qualifier.trim().length > 0))
    && (metadata.value !== undefined || metadata.target !== undefined);
}

function cockpitCoverageDistinctCandidateCount(
  obligation: CockpitSourceCoverageObligation,
  candidates: ExtractedMemory[],
): number {
  return new Set(candidates
    .filter((candidate) => cockpitCoverageCandidateMatchesObligation(obligation, candidate))
    .map((candidate) => cockpitFactualIdentityFingerprint(candidate))).size;
}

/**
 * A general or directed atomic candidate may discharge a deterministic source
 * obligation only when it is already a complete, single-event fact. Exact
 * source cardinality prevents one aggregate row from collapsing two same-slot
 * events; exact target equality keeps independently keyed price constraints
 * separate.
 */
function cockpitCoverageObligationHasIndependentCandidate(
  obligation: CockpitSourceCoverageObligation,
  candidates: ExtractedMemory[],
): boolean {
  const matching = candidates.filter((candidate) =>
    cockpitCoverageCandidateMatchesObligation(obligation, candidate)
  );
  if (cockpitCoverageDistinctCandidateCount(obligation, matching) < obligation.requiredFactCount) {
    return false;
  }
  return cockpitCoverageBindingAxesAreComplete(obligation, matching);
}

function cockpitDistinctBindingCount(
  candidates: ExtractedMemory[],
  select: (metadata: Record<string, unknown>) => unknown,
): number {
  return new Set(candidates.flatMap((candidate) => {
    const value = select(candidate.metadata as Record<string, unknown>);
    if (value === undefined || value === null
      || (typeof value === "string" && value.trim().length === 0)) return [];
    return [comparableCockpitValue(value)];
  })).size;
}

function cockpitCoverageBindingAxesAreComplete(
  obligation: CockpitSourceCoverageObligation,
  candidates: ExtractedMemory[],
): boolean {
  if (obligation.requiredSubjectCount !== undefined
    && cockpitDistinctBindingCount(candidates, (metadata) => metadata.subject)
      < obligation.requiredSubjectCount) {
    return false;
  }
  if (obligation.requiredConditionCount !== undefined
    && cockpitDistinctBindingCount(candidates, (metadata) => metadata.condition)
      < obligation.requiredConditionCount) {
    return false;
  }
  if (obligation.requiredSeatZoneCount !== undefined
    && cockpitDistinctBindingCount(candidates, (metadata) => metadata.seat_zone)
      < obligation.requiredSeatZoneCount) {
    return false;
  }
  if (obligation.requiredTemporalCount !== undefined) {
    const temporalBindings = candidates.flatMap((candidate) => {
      const metadata = candidate.metadata as Record<string, unknown>;
      const value = metadata.activity_start_time
        ?? metadata.valid_from
        ?? (metadata.slot === "appointment_time" ? metadata.value : undefined);
      if (value === undefined || value === null
        || (typeof value === "string" && value.trim().length === 0)) return [];
      return [comparableCockpitValue(value)];
    });
    if (temporalBindings.length < obligation.requiredTemporalCount) return false;
    // Two simultaneous schedule events may intentionally share the same wall
    // clock value; their source-event anchors and factual payload distinguish
    // them. Effective-date constraints outside schedule must carry distinct
    // temporal metadata rather than duplicating one dated fact.
    if (obligation.domain !== "schedule"
      && new Set(temporalBindings).size < obligation.requiredTemporalCount) {
      return false;
    }
  }
  if (obligation.requiredStateQualifierCount !== undefined
    && cockpitDistinctBindingCount(candidates, (metadata) => metadata.state_qualifier)
      < obligation.requiredStateQualifierCount) {
    return false;
  }
  return true;
}

interface CockpitDirectedCoverageCandidate {
  memory: ExtractedMemory;
  evidenceGroupIds: string[];
}

type CockpitCoverageEvidenceBinding =
  | {
    ok: true;
    groupIds: string[];
    source: "group_ids" | "legacy_spans" | "sole_group";
  }
  | {
    ok: false;
    reason:
      | "missing_binding"
      | "invalid_group_ids_shape"
      | "unknown_group_id"
      | "invalid_spans_shape"
      | "invalid_span_coordinates"
      | "span_quote_mismatch"
      | "unknown_span"
      | "conflicting_binding_forms"
      | "missing_state_qualifier"
      | "state_qualifier_outside_bound_evidence"
      | "evidence_subject_mismatch";
  };

function cockpitCoverageEvidenceBinding(
  rawMetadata: Record<string, unknown>,
  obligation: CockpitSourceCoverageObligation,
  normalizedSourceContent: string,
  allowSoleGroupBinding: boolean,
): CockpitCoverageEvidenceBinding {
  const knownGroupIds = new Set(obligation.evidenceGroups.map((group) => group.id));
  const hasGroupIds = Object.prototype.hasOwnProperty.call(
    rawMetadata,
    "coverage_evidence_group_ids",
  );
  const hasSpans = Object.prototype.hasOwnProperty.call(
    rawMetadata,
    "coverage_evidence_spans",
  );

  let idsFromGroupField: string[] | undefined;
  if (hasGroupIds) {
    const rawGroupIds = rawMetadata.coverage_evidence_group_ids;
    if (!Array.isArray(rawGroupIds) || rawGroupIds.length === 0
      || rawGroupIds.some((entry) => typeof entry !== "string" || entry.trim().length === 0)) {
      return { ok: false, reason: "invalid_group_ids_shape" };
    }
    idsFromGroupField = [...new Set(rawGroupIds)].sort();
    if (idsFromGroupField.some((id) => !knownGroupIds.has(id))) {
      return { ok: false, reason: "unknown_group_id" };
    }
  }

  let idsFromSpans: string[] | undefined;
  const rawSpans = rawMetadata.coverage_evidence_spans;
  if (hasSpans) {
    if (!Array.isArray(rawSpans) || rawSpans.length === 0) {
      return { ok: false, reason: "invalid_spans_shape" };
    }
    const groupIds = new Set<string>();
    for (const rawSpan of rawSpans) {
      if (!rawSpan || typeof rawSpan !== "object" || Array.isArray(rawSpan)) {
        return { ok: false, reason: "invalid_spans_shape" };
      }
      const span = rawSpan as Record<string, unknown>;
      const start = span.start;
      const end = span.end;
      const quote = span.quote;
      if (typeof start !== "number" || typeof end !== "number"
        || !Number.isInteger(start) || !Number.isInteger(end)
        || start < 0 || end <= start || end > normalizedSourceContent.length) {
        return { ok: false, reason: "invalid_span_coordinates" };
      }
      if (typeof quote !== "string" || normalizedSourceContent.slice(start, end) !== quote) {
        return { ok: false, reason: "span_quote_mismatch" };
      }
      const group = obligation.evidenceGroups.find((entry) =>
        entry.start === start && entry.end === end
      );
      if (!group) return { ok: false, reason: "unknown_span" };
      groupIds.add(group.id);
    }
    idsFromSpans = [...groupIds].sort();
  }

  if (idsFromGroupField && idsFromSpans
    && !sameCockpitStringSet(idsFromGroupField, idsFromSpans)) {
    return { ok: false, reason: "conflicting_binding_forms" };
  }
  const verifyStateQualifier = (groupIds: string[]): CockpitCoverageEvidenceBinding | undefined => {
    if (!obligation.requiresStateQualifier) return undefined;
    const qualifier = typeof rawMetadata.state_qualifier === "string"
      ? rawMetadata.state_qualifier.normalize("NFKC").trim()
      : "";
    if (!qualifier) return { ok: false, reason: "missing_state_qualifier" };
    const expectedQualifiers = groupIds.map((id) =>
      obligation.evidenceGroups.find((entry) => entry.id === id)?.stateQualifier
        ?.normalize("NFKC").trim()
    );
    const qualifierIsLocallyBound = expectedQualifiers.length > 0
      && expectedQualifiers.every((expected) => expected === qualifier);
    const qualifierHasOneExactBinding = new Set(expectedQualifiers).size === 1;
    if (!qualifierIsLocallyBound || !qualifierHasOneExactBinding) {
      return { ok: false, reason: "state_qualifier_outside_bound_evidence" };
    }
    const qualifierOccursInEveryBoundGroup = groupIds.every((id) => {
      const group = obligation.evidenceGroups.find((entry) => entry.id === id);
      return Boolean(group
        && normalizedSourceContent.slice(group.start, group.end).includes(qualifier));
    });
    return qualifierOccursInEveryBoundGroup
      ? undefined
      : { ok: false, reason: "state_qualifier_outside_bound_evidence" };
  };
  const verifyEvidenceSubject = (groupIds: string[]): CockpitCoverageEvidenceBinding | undefined => {
    const boundGroups = groupIds.map((id) => obligation.evidenceGroups.find((entry) => entry.id === id));
    const subjects = boundGroups.map((group) => group?.subject).filter(
      (value): value is string => typeof value === "string" && value.length > 0,
    );
    // Only named-state groups carry this axis. Once present, every bound group
    // must belong to the same exact NFKC subject as the candidate.
    if (subjects.length === 0) return undefined;
    const candidateSubject = typeof rawMetadata.subject === "string"
      ? rawMetadata.subject.normalize("NFKC").trim()
      : "";
    if (!candidateSubject || subjects.some((subject) => subject !== candidateSubject)
      || boundGroups.some((group) => !group?.subject)) {
      return { ok: false, reason: "evidence_subject_mismatch" };
    }
    return undefined;
  };
  if (idsFromGroupField) {
    const qualifierDefect = verifyStateQualifier(idsFromGroupField);
    if (qualifierDefect) return qualifierDefect;
    const subjectDefect = verifyEvidenceSubject(idsFromGroupField);
    if (subjectDefect) return subjectDefect;
    return { ok: true, groupIds: idsFromGroupField, source: "group_ids" };
  }
  if (idsFromSpans) {
    const qualifierDefect = verifyStateQualifier(idsFromSpans);
    if (qualifierDefect) return qualifierDefect;
    const subjectDefect = verifyEvidenceSubject(idsFromSpans);
    if (subjectDefect) return subjectDefect;
    return { ok: true, groupIds: idsFromSpans, source: "legacy_spans" };
  }
  if (allowSoleGroupBinding && obligation.evidenceGroups.length === 1) {
    const soleGroupIds = [obligation.evidenceGroups[0].id];
    const qualifierDefect = verifyStateQualifier(soleGroupIds);
    if (qualifierDefect) return qualifierDefect;
    const subjectDefect = verifyEvidenceSubject(soleGroupIds);
    if (subjectDefect) return subjectDefect;
    return {
      ok: true,
      groupIds: soleGroupIds,
      source: "sole_group",
    };
  }
  return { ok: false, reason: "missing_binding" };
}

function appendCockpitConstructionRepair(
  memory: ExtractedMemory,
  repair: string,
): ExtractedMemory {
  const metadata = { ...(memory.metadata as Record<string, unknown>) };
  const rawQuality = metadata.construction_quality;
  if (!rawQuality || typeof rawQuality !== "object" || Array.isArray(rawQuality)) return memory;
  const constructionQuality = { ...(rawQuality as Record<string, unknown>) };
  constructionQuality.repairs = [...new Set([
    ...cockpitStringArray(constructionQuality.repairs),
    repair,
  ])];
  metadata.construction_quality = constructionQuality;
  return { ...memory, metadata };
}

function collapseCockpitDirectedCoverageSet(
  candidates: CockpitDirectedCoverageCandidate[],
): CockpitDirectedCoverageCandidate[] {
  const byFingerprint = new Map<string, CockpitDirectedCoverageCandidate>();
  for (const candidate of candidates) {
    const fingerprint = JSON.stringify({
      fact: cockpitFactualIdentityFingerprint(candidate.memory),
      evidence_group_ids: [...candidate.evidenceGroupIds].sort(),
    });
    const existing = byFingerprint.get(fingerprint);
    if (existing) {
      if (cockpitBlindRepresentationFingerprint(existing.memory)
        !== cockpitBlindRepresentationFingerprint(candidate.memory)) {
        throw new L1StructuredOutputError(
          "coverage fact compiler returned conflicting representations of one factual binding",
        );
      }
      existing.evidenceGroupIds = [...new Set([
        ...existing.evidenceGroupIds,
        ...candidate.evidenceGroupIds,
      ])].sort();
      continue;
    }
    byFingerprint.set(fingerprint, {
      memory: candidate.memory,
      evidenceGroupIds: [...candidate.evidenceGroupIds].sort(),
    });
  }
  return [...byFingerprint.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([, candidate]) => candidate);
}

function cockpitDirectedCoverageSetReceipt(
  candidates: CockpitDirectedCoverageCandidate[],
): string[] {
  return candidates.map((candidate) => JSON.stringify({
    fact: cockpitBlindRepresentationFingerprint(candidate.memory),
    evidence_group_ids: [...candidate.evidenceGroupIds].sort(),
  })).sort();
}

function cockpitCoverageSourceEventAnchors(
  obligation: CockpitSourceCoverageObligation,
  evidenceGroupIds: string[],
): string[] {
  return [...new Set(evidenceGroupIds.flatMap((id) => {
    const group = obligation.evidenceGroups.find((entry) => entry.id === id);
    return group?.eventAnchor ? [group.eventAnchor] : [];
  }))].sort();
}

function cockpitDirectedCoverageSetIsComplete(
  obligation: CockpitSourceCoverageObligation,
  candidates: CockpitDirectedCoverageCandidate[],
): boolean {
  const factualCount = new Set(candidates.map((candidate) => JSON.stringify({
    fact: cockpitFactualIdentityFingerprint(candidate.memory),
    // A source-derived event anchor is admissible identity evidence. Unlike a
    // model-authored episode label or raw evidence-group ID, it is computed by
    // the same source parser across slots. This preserves two simultaneous
    // same-value events while temporal/condition/seat gates still reject a
    // copied undated fact assigned to multiple structural axes.
    source_event_anchors: cockpitCoverageSourceEventAnchors(
      obligation,
      candidate.evidenceGroupIds,
    ),
  }))).size;
  if (factualCount < obligation.requiredFactCount) return false;
  const coveredGroups = new Set(candidates.flatMap((candidate) => candidate.evidenceGroupIds));
  if (!obligation.evidenceGroups.every((group) => coveredGroups.has(group.id))) return false;
  if (obligation.requiresDistinctEvidenceBindings) {
    if (candidates.some((candidate) => candidate.evidenceGroupIds.length !== 1)) return false;
    const independentlyBoundGroups = new Set(candidates.map((candidate) => candidate.evidenceGroupIds[0]));
    if (!obligation.evidenceGroups.every((group) => independentlyBoundGroups.has(group.id))) return false;
  }
  return cockpitCoverageBindingAxesAreComplete(
    obligation,
    candidates.map((candidate) => candidate.memory),
  );
}

function canonicalizeCockpitDirectedEpisode(
  candidate: ExtractedMemory,
  obligation: CockpitSourceCoverageObligation,
  evidenceGroupIds: string[],
): ExtractedMemory {
  const metadata = { ...(candidate.metadata as Record<string, unknown>) };
  const relation = metadata.relation;
  if (relation !== "asserted" || cockpitStringArray(metadata.supersedes).length > 0) {
    return candidate;
  }
  // Only a persistent persona/instruction state receives this contract-owned
  // cross-session identity. Qualified episodic, dated, conditional or
  // otherwise event-partitioned rows must retain a source-event episode.
  if (hasCanonicalCockpitQualifiedStateEpisode(candidate)) {
    return candidate;
  }
  const eventAnchors = cockpitCoverageSourceEventAnchors(obligation, evidenceGroupIds);
  if (eventAnchors.length === 0) {
    throw new L1StructuredOutputError(
      `coverage fact compiler evidence lacks a source event anchor for ${obligation.id}`,
    );
  }
  const digest = createHash("sha256").update(JSON.stringify(canonicalCockpitFactValue({
    source_message_id: obligation.sourceMessageId,
    source_event_anchors: eventAnchors,
    subject: metadata.subject,
    occupant_scope: metadata.occupant_scope,
    vehicle_scope: metadata.vehicle_scope,
    seat_zone: metadata.seat_zone,
  }))).digest("hex").slice(0, 24);
  const deterministicEpisode = `coverage-event:${digest}`;
  if (metadata.episode_key !== deterministicEpisode) {
    metadata.episode_key = deterministicEpisode;
    const quality = metadata.construction_quality && typeof metadata.construction_quality === "object"
      && !Array.isArray(metadata.construction_quality)
      ? { ...(metadata.construction_quality as Record<string, unknown>) }
      : undefined;
    if (quality) {
      quality.repairs = [...new Set([
        ...cockpitStringArray(quality.repairs),
        "canonicalized_directed_episode_key",
      ])];
      metadata.construction_quality = quality;
    }
  }
  return { ...candidate, metadata };
}

function cockpitCandidateFallsWithinCoverageScope(
  obligation: CockpitSourceCoverageObligation,
  candidate: ExtractedMemory,
): boolean {
  const metadata = candidate.metadata as Record<string, unknown>;
  return metadata.domain === obligation.domain
    && metadata.slot === obligation.slot
    && comparableCockpitValue(metadata.constraint_target)
      === comparableCockpitValue(obligation.constraintTarget)
    && sameCockpitStringSet(candidate.source_message_ids, [obligation.sourceMessageId])
    && sameCockpitStringSet(cockpitStringArray(metadata.source_message_ids), [obligation.sourceMessageId]);
}

function cockpitCandidateTouchesCoverageSource(
  obligation: CockpitSourceCoverageObligation,
  candidate: ExtractedMemory,
): boolean {
  const metadata = candidate.metadata as Record<string, unknown>;
  const sourceIds = new Set([
    ...candidate.source_message_ids,
    ...cockpitStringArray(metadata.source_message_ids),
  ]);
  return sourceIds.has(obligation.sourceMessageId)
    && metadata.domain === obligation.domain
    && metadata.slot === obligation.slot
    && comparableCockpitValue(metadata.constraint_target)
      === comparableCockpitValue(obligation.constraintTarget);
}

function withoutCockpitModelCandidateReferences(memory: ExtractedMemory): ExtractedMemory {
  const metadata = { ...(memory.metadata as Record<string, unknown>) };
  delete metadata.input_candidate_ids;
  delete metadata.canonicalized_input_candidate_ids;
  delete metadata.coverage_evidence_spans;
  delete metadata.coverage_evidence_group_ids;
  delete metadata.coverage_required_fact_count;
  delete metadata.coverage_required_subject_count;
  delete metadata.coverage_required_condition_count;
  delete metadata.coverage_required_seat_zone_count;
  delete metadata.coverage_required_temporal_count;
  delete metadata.coverage_requires_state_qualifier;
  delete metadata.coverage_required_state_qualifier_count;
  delete metadata.coverage_required_state_qualifiers;
  delete metadata.coverage_requires_distinct_evidence_bindings;
  delete metadata.coverage_event_anchors;
  return {
    ...memory,
    source_message_ids: [...memory.source_message_ids],
    metadata,
  };
}

/** Dedicated env override > parsed gateway/plugin config > legacy default. */
function resolveL1TimeoutMs(configured?: number): number {
  const envValue = Number(process.env.TDAI_L1_TIMEOUT_MS);
  if (Number.isFinite(envValue) && envValue > 0) return Math.floor(envValue);
  // parseConfig always materializes a 120s generic LLM default. Do not let
  // that accidentally shorten L1's historical 180s layer-specific budget.
  return Number.isFinite(configured) && Number(configured) > 0
    ? Math.max(DEFAULT_L1_TIMEOUT_MS, Math.floor(Number(configured)))
    : DEFAULT_L1_TIMEOUT_MS;
}

// ============================
// Types
// ============================

/** A scene segment with its extracted memories (LLM output) */
interface SceneSegment {
  scene_name: string;
  message_ids: string[];
  memories: Array<{
    content: string;
    type: string;
    priority: number;
    source_message_ids: string[];
    metadata: Record<string, unknown>;
  }>;
}

export interface L1ExtractionResult {
  /** Whether extraction succeeded */
  success: boolean;
  /** Number of memories extracted */
  extractedCount: number;
  /** Number of memories actually stored (after dedup) */
  storedCount: number;
  /** The memory records that were stored */
  records: MemoryRecord[];
  /** Scene names detected during extraction */
  sceneNames: string[];
  /** Last scene name (for continuity in next extraction) */
  lastSceneName?: string;
  /** Deterministic post-extraction contract audit (cockpit mode only). */
  constructionQuality?: {
    model?: string;
    complete: number;
    partial: number;
    invalid: number;
    averageScore: number;
  };
}

// ============================
// Core function
// ============================

/**
 * Run the full L1 extraction pipeline on conversation messages.
 *
 * @param messages - Filtered conversation messages (from L0 or directly from hook)
 * @param sessionKey - The session key
 * @param baseDir - Base data directory (~/.openclaw/memory-tdai/)
 * @param config - OpenClaw config (for LLM access)
 * @param options - Extraction options
 * @param logger - Optional logger
 */
export async function extractL1Memories(params: {
  messages: ConversationMessage[];
  sessionKey: string;
  sessionId?: string;
  taskId?: string;
  teamId?: string;
  userId?: string;
  agentId?: string;
  baseDir: string;
  config: unknown;
  options?: {
    /** Max new messages to send in one extraction call */
    maxMessagesPerExtraction?: number;
    /** Max background messages for context */
    maxBackgroundMessages?: number;
    /** Enable conflict detection */
    enableDedup?: boolean;
    /** Max memories extracted per call */
    maxMemoriesPerSession?: number;
    /** LLM model override */
    model?: string;
    /** Actual configured construction model, recorded as provenance. */
    constructionModel?: string;
    /** Previous scene name for continuity */
    previousSceneName?: string;
    /** Prompt family for L1 extraction (default: chat). */
    promptMode?: MemoryPromptMode;
    /** Vector store for cosine similarity candidate recall */
    vectorStore?: IMemoryStore;
    /** Embedding service for computing query vectors */
    embeddingService?: EmbeddingService;
    /** Top-K candidates for conflict recall (default: 5) */
    conflictRecallTopK?: number;
    /** Override embedding timeout for capture-path calls (milliseconds) */
    embeddingTimeoutMs?: number;
    /** L1 LLM timeout from the resolved runtime config (milliseconds). */
    llmTimeoutMs?: number;
    /** Distributed-lock cancellation propagated through model and persistence work. */
    abortSignal?: AbortSignal;
    /** Enable durable receipt-backed, replayable persistence (cockpit service path). */
    strictPersistence?: boolean;
    /**
     * Host-neutral LLM runner. When provided, used instead of creating
     * a CleanContextRunner (decouples from OpenClaw runtime).
     */
    llmRunner?: LLMRunner;
  };
  logger?: Logger;
  /** Plugin instance ID for metric reporting (optional — metrics skipped if absent) */
  instanceId?: string;
  /**
   * StorageAdapter for L1 JSONL writes.
   * - service mode: must be provided (CosStorageBackend) — JSONL is the source of
   *   truth for backup/recovery; without storage, writes silently fall back to local
   *   pod fs and are lost on pod restart (CR-2 root cause, fixed 2026-05-19).
   * - standalone mode: caller usually provides LocalStorageBackend; if absent,
   *   writeMemory falls back to fs at `{baseDir}/records/{date}.jsonl`.
   */
  storage?: StorageAdapter;
}): Promise<L1ExtractionResult> {
  const { messages, sessionKey, sessionId, taskId, teamId, userId, agentId, baseDir, config, logger, instanceId: metricInstanceId, storage } = params;
  const options = params.options ?? {};
  throwIfAborted(options.abortSignal, "L1 extraction before start");
  const maxNewMessages = options.maxMessagesPerExtraction ?? 10;
  const maxBgMessages = options.maxBackgroundMessages ?? 5;
  const enableDedup = options.enableDedup ?? true;
  const maxMemoriesPerSession = options.maxMemoriesPerSession ?? 10;

  if (messages.length === 0) {
    logger?.debug?.(`${TAG} No messages to extract from`);
    return { success: true, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
  }

  const l1StartMs = Date.now();

  // Quality gate: filter messages through L1 extraction rules (length, symbols,
  // prompt injection, etc.) before sending to the LLM. L0 deliberately captures
  // everything; the strict filtering happens here at L1 stage.
  const qualifiedMessages = messages.filter((m) => shouldExtractL1(m.content));
  if (qualifiedMessages.length < messages.length) {
    logger?.debug?.(
      `${TAG} L1 quality filter: ${messages.length} → ${qualifiedMessages.length} messages ` +
      `(${messages.length - qualifiedMessages.length} filtered out)`,
    );
  }

  if (qualifiedMessages.length === 0) {
    logger?.debug?.(`${TAG} All messages filtered out by L1 quality gate`);
    return { success: true, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
  }

  // Split messages into background (older) + new (recent)
  const newMessages = qualifiedMessages.slice(-maxNewMessages);
  const bgEndIdx = qualifiedMessages.length - newMessages.length;
  const backgroundMessages = bgEndIdx > 0
    ? qualifiedMessages.slice(Math.max(0, bgEndIdx - maxBgMessages), bgEndIdx)
    : [];

  const persistenceScope: L1PersistenceScope = {
    sessionKey,
    sessionId,
    taskId,
    teamId,
    userId,
    agentId,
  };
  const strictPersistence = options.strictPersistence === true;
  const persistenceBatchId = strictPersistence
    ? createL1PersistenceBatchId(newMessages, persistenceScope)
    : undefined;

  if (strictPersistence) {
    if (!storage || !options.vectorStore || !persistenceBatchId) {
      logger?.error?.(
        `${TAG} Strict L1 persistence requires both StorageAdapter and IMemoryStore; failing closed`,
      );
      return { success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
    }
    try {
      const resumed = await resumeL1PersistenceTransaction({
        batchId: persistenceBatchId,
        storage,
        vectorStore: options.vectorStore,
        embeddingService: options.embeddingService,
        logger,
        abortSignal: options.abortSignal,
      });
      if (resumed) {
        logger?.info?.(`${TAG} Resumed durable L1 transaction ${persistenceBatchId} before model construction`);
        return {
          success: true,
          extractedCount: resumed.extractedCount,
          storedCount: resumed.records.length,
          records: resumed.records,
          sceneNames: resumed.sceneNames,
          lastSceneName: resumed.lastSceneName,
          constructionQuality: resumed.constructionQuality,
        };
      }
    } catch (error) {
      throwIfAborted(options.abortSignal, "L1 extraction transaction replay");
      logger?.error?.(
        `${TAG} Durable L1 transaction replay failed closed: `
        + `${error instanceof Error ? error.message : String(error)}`,
      );
      return { success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
    }
  }

  logger?.debug?.(`${TAG} Extracting from ${newMessages.length} new messages (+ ${backgroundMessages.length} background) [${qualifiedMessages.length} qualified from ${messages.length} input]`);

  const priorStructuredMemories = options.promptMode === "cockpit"
    ? await loadCockpitPriorMemoryContext({
        vectorStore: options.vectorStore,
        teamId,
        userId,
        agentId,
        taskId,
        currentSessionId: sessionId,
        // The prompt is shared by every proposal in this extraction batch.
        // Bound prior context to the earliest source event so an older row in
        // the batch cannot see a predecessor that is still in its future.
        currentEventTimeMs: newMessages.reduce<number | undefined>((earliest, message) => {
          if (!Number.isFinite(message.timestamp)) return earliest;
          return earliest === undefined ? message.timestamp : Math.min(earliest, message.timestamp);
        }, undefined),
        logger,
      })
    : undefined;
  throwIfAborted(options.abortSignal, "L1 extraction after prior-memory query");

  // Step 1: LLM extraction (scene segmentation + memory extraction)
  let scenes: SceneSegment[];
  let primaryContentRiskUnavailable = false;
  try {
    scenes = await callLlmExtraction({
      newMessages,
      backgroundMessages,
      previousSceneName: options.previousSceneName,
      config,
      logger,
      model: options.model,
      promptMode: options.promptMode,
      priorStructuredMemories,
      traceContext: { teamId, userId, agentId, sessionId },
      llmRunner: options.llmRunner,
      timeoutMs: resolveL1TimeoutMs(options.llmTimeoutMs),
      abortSignal: options.abortSignal,
    });
    logger?.debug?.(`${TAG} LLM detected ${scenes.length} scene(s)`);
  } catch (err) {
    throwIfAborted(options.abortSignal, "L1 primary construction");
    if (options.promptMode === "cockpit" && isProviderContentRiskError(err)) {
      // Cockpit construction has an independent source compiler and a final
      // deterministic reconciliation gate. Let that independent path run when
      // the provider rejects both primary prompt variants, but never treat an
      // empty/failed independent result as success (see the fail-closed checks
      // below). This keeps the current source text unchanged.
      primaryContentRiskUnavailable = true;
      scenes = [];
      logger?.warn?.(
        `${TAG} Primary cockpit extractor unavailable after bounded content-risk fallback; `
        + "continuing with independent source compiler under fail-closed reconciliation",
      );
    } else {
      logger?.error(`${TAG} LLM extraction failed: ${err instanceof Error ? err.message : String(err)}`);
      return { success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
    }
  }

  // Flatten all memories across scenes
  const extractedProposals: ExtractedMemory[] = [];
  const sceneNames: string[] = [];

  for (const scene of scenes) {
    sceneNames.push(scene.scene_name);
    for (const mem of scene.memories) {
      const memType = normalizeType(mem.type);
      if (!memType) {
        logger?.warn?.(`${TAG} Skipping memory with invalid type "${mem.type}"`);
        continue;
      }
      const extractedMemory: ExtractedMemory = {
        content: mem.content,
        type: memType,
        priority: typeof mem.priority === "number" ? mem.priority : 50,
        source_message_ids: Array.isArray(mem.source_message_ids) ? mem.source_message_ids : [],
        metadata: mem.metadata ?? {},
        scene_name: scene.scene_name,
      };
      extractedProposals.push(extractedMemory);
    }
  }

  const priorKnownLineage: CockpitKnownLineage[] = options.promptMode === "cockpit"
    ? (priorStructuredMemories ?? []).map((context) => ({
        recordId: context.record_id,
        type: normalizeType(context.type) ?? undefined,
        scene_name: context.scene_name,
        metadata: context.metadata,
      }))
    : [];
  const batchAssertionLineage: CockpitKnownLineage[] = options.promptMode === "cockpit"
    ? extractedProposals
        .filter((memory) => !isCockpitTransitionProposal(memory))
        .map((memory) => normalizeCockpitExtractedMemory({
          memory,
          sourceMessages: newMessages,
          sessionId,
          constructionModel: options.constructionModel ?? options.model,
          knownLineage: priorKnownLineage,
        }))
        .filter(isCompleteCockpitProposal)
        .map((memory) => ({
          type: memory.type,
          scene_name: memory.scene_name,
          metadata: memory.metadata as Record<string, unknown>,
        }))
    : [];
  // Only assertions may provide an in-batch identity. The contract performs
  // a second per-output time check, so a transition at t1 cannot bind an
  // assertion at t2 merely because both arrived in one extraction call.
  const knownLineage: CockpitKnownLineage[] = [
    ...priorKnownLineage,
    ...batchAssertionLineage,
  ];

  let allExtracted = options.promptMode === "cockpit"
    ? extractedProposals.map((memory) => normalizeCockpitExtractedMemory({
        memory,
        sourceMessages: newMessages,
        sessionId,
        constructionModel: options.constructionModel ?? options.model,
        knownLineage,
      }))
    : extractedProposals;

  if (options.promptMode === "cockpit") {
    const rejectedPrimaryInformationQueries = allExtracted.filter((memory) =>
      hasCockpitConstructionIssue(memory, "informational_query_source")
    ).length;
    if (rejectedPrimaryInformationQueries > 0) {
      logger?.info(
        `${TAG} Cockpit speech-act gate rejected ${rejectedPrimaryInformationQueries} `
        + "primary candidate(s) grounded only in pure informational queries",
      );
      allExtracted = allExtracted.filter((memory) =>
        !hasCockpitConstructionIssue(memory, "informational_query_source")
      );
    }
    const compilerModel = options.constructionModel ?? options.model;
    const sourceCoverageObligations = detectCockpitSourceCoverageObligations(
      newMessages,
      priorStructuredMemories ?? [],
    );
    const compilerSourceMessages = primaryContentRiskUnavailable
      ? minimizeCockpitProviderMessages(newMessages)
      : newMessages;
    const compilerPriorMemories = primaryContentRiskUnavailable
      ? []
      : priorStructuredMemories ?? [];
    logger?.info(
      `${TAG} Cockpit deterministic source coverage: ${JSON.stringify(sourceCoverageObligations.map((entry) => ({
        id: entry.id,
        source_message_id: entry.sourceMessageId,
        domain: entry.domain,
        slot: entry.slot,
        constraint_target: entry.constraintTarget,
        requires_state_qualifier: entry.requiresStateQualifier,
        required_state_qualifier_count: entry.requiredStateQualifierCount,
        reason: entry.reason,
      })))}`,
    );
    let compilerNormalized: ExtractedMemory[] = [];
    let compilerCompleted = false;
    let constructionAttempt = 0;
    let directedCoverageCallCount = 0;
    const directedCompilerCandidates = new WeakSet<ExtractedMemory>();
    while (constructionAttempt < MAX_COCKPIT_RECONCILIATION_GATE_ATTEMPTS) {
      constructionAttempt += 1;
      compilerNormalized = [];
      compilerCompleted = false;
      try {
        throwIfAborted(options.abortSignal, "L1 cockpit transaction attempt");
        const compilerScenes = await callCockpitAtomicCompiler({
          newMessages: compilerSourceMessages,
          priorStructuredMemories: compilerPriorMemories,
          maxMemories: maxMemoriesPerSession,
          sourceCoverageObligations,
          config,
          logger,
          model: options.model,
          llmRunner: options.llmRunner,
          traceContext: { teamId, userId, agentId, sessionId },
          timeoutMs: resolveL1TimeoutMs(options.llmTimeoutMs),
          transactionAttempt: constructionAttempt,
          abortSignal: options.abortSignal,
        });
        const compilerProposals = compilerScenes.flatMap((scene) => scene.memories.flatMap((memory) => {
          const type = normalizeType(memory.type);
          if (!type) return [];
          return [{
            content: memory.content,
            type,
            priority: typeof memory.priority === "number" ? memory.priority : 50,
            source_message_ids: Array.isArray(memory.source_message_ids) ? memory.source_message_ids : [],
            metadata: memory.metadata ?? {},
            scene_name: scene.scene_name,
          } satisfies ExtractedMemory];
        }));
        if (compilerProposals.length > maxMemoriesPerSession) {
          throw new Error(`atomic compiler returned ${compilerProposals.length} memories; maximum is ${maxMemoriesPerSession}`);
        }
        compilerNormalized = compilerProposals.map((memory) => normalizeCockpitExtractedMemory({
          memory,
          sourceMessages: newMessages,
          sessionId,
          constructionModel: compilerModel,
          knownLineage,
        }));
        const rejectedAtomicInformationQueries = compilerNormalized.filter((memory) =>
          hasCockpitConstructionIssue(memory, "informational_query_source")
        ).length;
        if (rejectedAtomicInformationQueries > 0) {
          logger?.info(
            `${TAG} Cockpit speech-act gate rejected ${rejectedAtomicInformationQueries} `
            + "atomic candidate(s) grounded only in pure informational queries",
          );
          compilerNormalized = compilerNormalized.filter((memory) =>
            !hasCockpitConstructionIssue(memory, "informational_query_source")
          );
        }
        if (primaryContentRiskUnavailable && compilerNormalized.length === 0) {
          throw new Error("independent atomic compiler returned no memories after primary content-risk rejection");
        }
        const generalCompilerCount = compilerNormalized.length;
        const directedCoverageIds: string[] = [];
        for (const obligation of sourceCoverageObligations) {
        const mixedSourceCandidates = [...allExtracted, ...compilerNormalized].filter((candidate) =>
          cockpitCandidateTouchesCoverageSource(obligation, candidate)
          && !cockpitCandidateFallsWithinCoverageScope(obligation, candidate)
        );
        if (mixedSourceCandidates.length > 0) {
          throw new L1StructuredOutputError(
            `coverage source scope contains ${mixedSourceCandidates.length} mixed-source atomic candidates for ${obligation.id}`,
          );
        }
        const lowerBoundAlreadyMet = cockpitCoverageObligationHasIndependentCandidate(
          obligation,
          compilerNormalized,
        );
        if (lowerBoundAlreadyMet && !obligation.requiresSetAudit) continue;
        const matchingSourceMessages = newMessages.filter((message) =>
          message.role === "user" && message.id === obligation.sourceMessageId
        );
        if (matchingSourceMessages.length !== 1) {
          throw new L1StructuredOutputError(
            `coverage fact compiler expected one exact user source for ${obligation.id}; found ${matchingSourceMessages.length}`,
          );
        }
        const sourceMessage = matchingSourceMessages[0];
        const normalizedSourceContent = sourceMessage.content.normalize("NFKC");
        const coverageSetMax = Math.min(
          maxMemoriesPerSession,
          Math.max(8, obligation.requiredFactCount + 4),
        );
        const compileDirectedSet = async (
          verificationPass: boolean,
        ): Promise<CockpitDirectedCoverageCandidate[]> => {
          const directedScenes = await callCockpitCoverageFactCompiler({
            sourceMessage,
            priorStructuredMemories: compilerPriorMemories,
            obligation,
            maxMemories: coverageSetMax,
            verificationPass,
            config,
            logger,
            model: options.model,
            llmRunner: options.llmRunner,
            traceContext: { teamId, userId, agentId, sessionId },
            timeoutMs: resolveL1TimeoutMs(options.llmTimeoutMs),
            transactionAttempt: constructionAttempt,
            beforeProviderCall: () => {
              throwIfAborted(options.abortSignal, "L1 directed coverage provider call");
              if (directedCoverageCallCount >= MAX_COCKPIT_DIRECTED_CALLS_PER_EXTRACTION) {
                // A provider-attempt budget is an operational limit, not a
                // malformed model response. Do not restart the full atomic
                // transaction and incur another paid call after it is spent.
                throw new Error(
                  `coverage fact compiler provider call budget exceeded (${MAX_COCKPIT_DIRECTED_CALLS_PER_EXTRACTION})`,
                );
              }
              directedCoverageCallCount += 1;
            },
            abortSignal: options.abortSignal,
          });
          const directedMemoryCount = directedScenes.reduce(
            (count, scene) => count + scene.memories.length,
            0,
          );
          if (directedScenes.length !== 1
            || directedMemoryCount < 1
            || directedMemoryCount > coverageSetMax
            || directedScenes[0].scene_name !== obligation.domain
            || directedScenes[0].message_ids.length !== 1
            || !sameCockpitStringSet(directedScenes[0].message_ids, [obligation.sourceMessageId])) {
            throw new L1StructuredOutputError(
              `coverage fact compiler returned an invalid single-source set envelope for ${obligation.id}`,
            );
          }
          const normalizedSet: CockpitDirectedCoverageCandidate[] = [];
          for (const directedRaw of directedScenes[0].memories) {
            const directedType = normalizeType(directedRaw.type);
            if (directedType !== "instruction" && directedType !== "episodic") {
              throw new L1StructuredOutputError(
                `coverage fact compiler returned invalid type for ${obligation.id}`,
              );
            }
            const rawMetadata = directedRaw.metadata && typeof directedRaw.metadata === "object"
              && !Array.isArray(directedRaw.metadata)
              ? directedRaw.metadata as Record<string, unknown>
              : {};
            const evidenceBinding = cockpitCoverageEvidenceBinding(
              rawMetadata,
              obligation,
              normalizedSourceContent,
              directedMemoryCount === 1
                && obligation.requiredFactCount === 1
                && !obligation.requiresDistinctEvidenceBindings,
            );
            if (!evidenceBinding.ok) {
              throw new L1StructuredOutputError(
                `coverage fact compiler returned invalid evidence binding (${evidenceBinding.reason}) `
                + `for ${obligation.id}`,
              );
            }
            const evidenceGroupIds = evidenceBinding.groupIds;
            const normalizedDirectedCandidate = normalizeCockpitExtractedMemory({
              memory: {
                content: directedRaw.content,
                type: directedType,
                priority: typeof directedRaw.priority === "number" ? directedRaw.priority : 50,
                source_message_ids: Array.isArray(directedRaw.source_message_ids)
                  ? directedRaw.source_message_ids
                  : [],
                metadata: rawMetadata,
                scene_name: directedScenes[0].scene_name,
              },
              sourceMessages: [sourceMessage],
              sessionId,
              constructionModel: compilerModel,
              // A directed candidate remains blind to the primary/general
              // drafts and all other current-batch events. Persisted lineage
              // is the only authority for an update/cancellation edge.
              knownLineage: priorKnownLineage,
              sourceEvidenceSpans: obligation.evidenceGroups
                .filter((group) => evidenceGroupIds.includes(group.id))
                .map((group) => ({ start: group.start, end: group.end })),
            });
            const directedCandidate = withoutCockpitModelCandidateReferences(
              canonicalizeCockpitDirectedEpisode(
                evidenceBinding.source === "sole_group"
                  ? appendCockpitConstructionRepair(
                    normalizedDirectedCandidate,
                    "deterministically_bound_sole_coverage_group",
                  )
                  : normalizedDirectedCandidate,
                obligation,
                evidenceGroupIds,
              ),
            );
            if (!cockpitCoverageCandidateMatchesObligation(obligation, directedCandidate)) {
              throw new L1StructuredOutputError(
                `coverage fact compiler returned an incomplete or cross-slot fact for ${obligation.id}`,
              );
            }
            normalizedSet.push({ memory: directedCandidate, evidenceGroupIds });
          }
          const collapsed = collapseCockpitDirectedCoverageSet(normalizedSet);
          if (!cockpitDirectedCoverageSetIsComplete(obligation, collapsed)) {
            throw new L1StructuredOutputError(
              `coverage fact compiler returned an incomplete or duplicate fact set for ${obligation.id}`,
            );
          }
          return collapsed;
        };

        const directedSet = await compileDirectedSet(false);
        const needsBlindVerification = obligation.requiresSetAudit
          || directedSet.length > obligation.requiredFactCount;
        if (needsBlindVerification) {
          const verificationSet = await compileDirectedSet(true);
          if (!sameCockpitStringSet(
            cockpitDirectedCoverageSetReceipt(directedSet),
            cockpitDirectedCoverageSetReceipt(verificationSet),
          )) {
            throw new L1StructuredOutputError(
              `blind coverage fact sets disagree for ${obligation.id}`,
            );
          }
        }
        if (needsBlindVerification) {
          // A verified source-only set is authoritative for this exact source,
          // domain, slot and target. Remove the general subset so a hallucinated
          // or stale extra cannot contaminate a complete blind receipt.
          compilerNormalized = compilerNormalized.filter((candidate) =>
            !cockpitCandidateFallsWithinCoverageScope(obligation, candidate)
          );
          allExtracted = allExtracted.filter((candidate) =>
            !cockpitCandidateFallsWithinCoverageScope(obligation, candidate)
          );
        }
        const directedMemories = directedSet.map((candidate) => candidate.memory);
        const directedFingerprints = new Set(directedMemories.map((candidate) =>
          cockpitSourceBoundFactualFingerprint(candidate)
        ));
        // A general candidate may describe the same fact while violating the
        // exact source/domain/slot envelope (or remaining construction-partial).
        // Such a row must not suppress the independently verified source-only
        // representation merely because their semantic fingerprints match.
        // Replace only that same-fact, non-authoritative representation; keep
        // unrelated general candidates available to reconciliation.
        compilerNormalized = compilerNormalized.filter((candidate) =>
          !directedFingerprints.has(cockpitSourceBoundFactualFingerprint(candidate))
          || cockpitCoverageCandidateMatchesObligation(obligation, candidate)
        );
        const existingAuthoritativeFingerprints = new Set(compilerNormalized
          .filter((candidate) => cockpitCoverageCandidateMatchesObligation(obligation, candidate))
          .map((candidate) => cockpitSourceBoundFactualFingerprint(candidate)));
        const newDirectedCandidates = directedMemories
          .filter((candidate) => !existingAuthoritativeFingerprints.has(
            cockpitSourceBoundFactualFingerprint(candidate),
          ));
        if (compilerNormalized.length + newDirectedCandidates.length > maxMemoriesPerSession) {
          throw new L1StructuredOutputError(
            `coverage fact compiler cannot add ${obligation.id}; atomic candidate limit ${maxMemoriesPerSession} reached`,
          );
        }
        for (const candidate of newDirectedCandidates) directedCompilerCandidates.add(candidate);
        compilerNormalized.push(...newDirectedCandidates);
        if (!needsBlindVerification
          && !cockpitCoverageObligationHasIndependentCandidate(obligation, compilerNormalized)) {
          throw new L1StructuredOutputError(
            `coverage fact compiler did not satisfy the fact-set lower bound for ${obligation.id}`,
          );
        }
        directedCoverageIds.push(
          `${obligation.id}:facts=${directedSet.length}:verified=${needsBlindVerification}`,
        );
      }
      compilerCompleted = true;
      logger?.info(
        `${TAG} Cockpit atomic compiler audit: general=${generalCompilerCount}, `
        + `directed=${directedCoverageIds.length}, directed_coverage_ids=${JSON.stringify(directedCoverageIds)}, `
        + `rows=${JSON.stringify(compilerNormalized.map((memory, index) => {
          const metadata = memory.metadata as Record<string, unknown>;
          const quality = metadata.construction_quality && typeof metadata.construction_quality === "object"
            ? metadata.construction_quality as Record<string, unknown>
            : {};
          return {
            domain: metadata.domain,
            slot: metadata.slot,
            relation: metadata.relation,
            quality_status: quality.status,
            quality_issues: quality.issues,
            source_count: Array.isArray(metadata.source_message_ids) ? metadata.source_message_ids.length : 0,
            compiler_path: directedCompilerCandidates.has(memory) ? "coverage_directed" : "general",
          };
        }))}`,
      );

      if (allExtracted.length === 0
        && compilerNormalized.length === 0
        && sourceCoverageObligations.length === 0) {
        logger?.info(`${TAG} Cockpit construction harness: both independent passes returned no memory`);
      } else {
        const reconciliationKnownLineage: CockpitKnownLineage[] = [
          ...(priorStructuredMemories ?? []).map((context) => ({
            recordId: context.record_id,
            type: normalizeType(context.type) ?? undefined,
            scene_name: context.scene_name,
            metadata: context.metadata,
          })),
          ...[...allExtracted, ...compilerNormalized]
            .filter((memory) =>
              !isCockpitTransitionProposal(memory) && isCompleteCockpitProposal(memory)
            )
            .map((memory) => ({
              type: memory.type,
              scene_name: memory.scene_name,
              metadata: memory.metadata as Record<string, unknown>,
            })),
        ];
        const reconciliationInputs = [
          ...allExtracted.map((memory, index) => ({ id: `primary:${index}`, memory })),
          ...compilerNormalized.map((memory, index) => ({ id: `atomic:${index}`, memory })),
          ...sourceCoverageObligations.map((obligation) => ({
            id: obligation.id,
            memory: sourceCoverageObligationToMemory(obligation),
          })),
        ];
        const normalizeReconciledScenes = (candidateScenes: SceneSegment[]): ExtractedMemory[] => {
          const proposals = candidateScenes.flatMap((scene) => scene.memories.flatMap((memory) => {
            const type = normalizeType(memory.type);
            if (!type) return [];
            return [{
              content: memory.content,
              type,
              priority: typeof memory.priority === "number" ? memory.priority : 50,
              source_message_ids: Array.isArray(memory.source_message_ids) ? memory.source_message_ids : [],
              metadata: memory.metadata ?? {},
              scene_name: scene.scene_name,
            } satisfies ExtractedMemory];
          }));
          if (proposals.length > maxMemoriesPerSession) {
            throw new Error(
              `construction reconciliation returned ${proposals.length} memories; expected 0..${maxMemoriesPerSession}`,
            );
          }
          return proposals.map((memory) => normalizeCockpitExtractedMemory({
            memory,
            sourceMessages: newMessages,
            sessionId,
            constructionModel: compilerModel,
            knownLineage: reconciliationKnownLineage,
          }));
        };
        const runReconciliation = (repairFeedback?: {
          issues: string[];
          uncoveredCandidateIds: string[];
          previousMemories: ExtractedMemory[];
          diagnostics: CockpitReconciliationDiagnostic[];
        }) => callCockpitConstructionReconciliation({
          newMessages: compilerSourceMessages,
          priorStructuredMemories: compilerPriorMemories,
          primaryMemories: allExtracted,
          atomicMemories: compilerNormalized,
          sourceCoverageObligations,
          maxMemories: maxMemoriesPerSession,
          config,
          logger,
          model: options.model,
          llmRunner: options.llmRunner,
          traceContext: { teamId, userId, agentId, sessionId },
          timeoutMs: resolveL1TimeoutMs(options.llmTimeoutMs),
          repairFeedback,
          transactionAttempt: constructionAttempt,
          abortSignal: options.abortSignal,
        });

        let reconciledNormalized = normalizeReconciledScenes(await runReconciliation());
        let assembly = assembleCockpitConstructionReconciliation({
          inputs: reconciliationInputs,
          reconciled: reconciledNormalized,
          maxMemories: maxMemoriesPerSession,
          priorMemories: priorStructuredMemories ?? [],
        });
        reconciledNormalized = assembly.memories;
        let resolvedCandidateIds = assembly.resolvedCandidateIds;
        logger?.info(
          `${TAG} Cockpit deterministic assembly audit: final=${reconciledNormalized.length}, `
          + `resolved_candidates=${resolvedCandidateIds.length}, repairs=${JSON.stringify(assembly.repairCounts)}`,
        );
        let gate = gateCockpitConstructionReconciliation({
          inputs: reconciliationInputs,
          reconciled: reconciledNormalized,
          maxMemories: maxMemoriesPerSession,
          priorMemories: priorStructuredMemories ?? [],
          resolvedCandidateIds,
        });
        logger?.info(
          `${TAG} Cockpit construction reconciliation rows: ${JSON.stringify(reconciledNormalized.map((memory) => {
            const metadata = memory.metadata as Record<string, unknown>;
            const quality = metadata.construction_quality && typeof metadata.construction_quality === "object"
              ? metadata.construction_quality as Record<string, unknown>
              : {};
            return {
              domain: metadata.domain,
              slot: metadata.slot,
              relation: metadata.relation,
              input_candidate_ids: metadata.input_candidate_ids,
              quality_status: quality.status,
              quality_issues: quality.issues,
            };
          }))}`,
        );
        logger?.info(
          `${TAG} Cockpit construction reconciliation audit: final=${reconciledNormalized.length}, `
          + `required=${gate.requiredCandidateIds.length}, covered=${gate.coveredCandidateIds.length}, `
          + `uncovered=${JSON.stringify(gate.uncoveredCandidateIds)}, issues=${JSON.stringify(gate.issues)}`,
        );
        if (!gate.accepted) {
          logger?.warn?.(
            `${TAG} Cockpit construction reconciliation repair requested: issues=${JSON.stringify(gate.issues)}, `
            + `uncovered=${JSON.stringify(gate.uncoveredCandidateIds)}, `
            + `diagnostics=${JSON.stringify(gate.diagnostics)}`,
          );
          const previousMemories = reconciledNormalized;
          reconciledNormalized = normalizeReconciledScenes(await runReconciliation({
            issues: gate.issues,
            uncoveredCandidateIds: gate.uncoveredCandidateIds,
            previousMemories,
            diagnostics: gate.diagnostics,
          }));
          assembly = assembleCockpitConstructionReconciliation({
            inputs: reconciliationInputs,
            reconciled: reconciledNormalized,
            maxMemories: maxMemoriesPerSession,
            priorMemories: priorStructuredMemories ?? [],
          });
          reconciledNormalized = assembly.memories;
          resolvedCandidateIds = assembly.resolvedCandidateIds;
          logger?.info(
            `${TAG} Cockpit deterministic repair assembly audit: final=${reconciledNormalized.length}, `
            + `resolved_candidates=${resolvedCandidateIds.length}, repairs=${JSON.stringify(assembly.repairCounts)}`,
          );
          gate = gateCockpitConstructionReconciliation({
            inputs: reconciliationInputs,
            reconciled: reconciledNormalized,
            maxMemories: maxMemoriesPerSession,
            priorMemories: priorStructuredMemories ?? [],
            resolvedCandidateIds,
          });
          logger?.info(
            `${TAG} Cockpit construction reconciliation repair rows: ${JSON.stringify(reconciledNormalized.map((memory) => {
              const metadata = memory.metadata as Record<string, unknown>;
              const quality = metadata.construction_quality && typeof metadata.construction_quality === "object"
                ? metadata.construction_quality as Record<string, unknown>
                : {};
              return {
                domain: metadata.domain,
                slot: metadata.slot,
                relation: metadata.relation,
                input_candidate_ids: metadata.input_candidate_ids,
                quality_status: quality.status,
                quality_issues: quality.issues,
              };
            }))}`,
          );
          logger?.info(
            `${TAG} Cockpit construction reconciliation repair audit: final=${reconciledNormalized.length}, `
            + `required=${gate.requiredCandidateIds.length}, covered=${gate.coveredCandidateIds.length}, `
            + `uncovered=${JSON.stringify(gate.uncoveredCandidateIds)}, issues=${JSON.stringify(gate.issues)}`,
          );
          if (!gate.accepted) {
            throw new L1ReconciliationGateError(
              `construction reconciliation gate rejected repaired output: ${gate.issues.join(",")}`,
            );
          }
        }
        allExtracted = reconciledNormalized.map((memory) => stampCockpitConstructionHarness(
          memory,
          compilerModel,
          "passed",
          "passed",
          primaryContentRiskUnavailable ? "content_risk_unavailable" : "passed",
          constructionAttempt,
        ));
      }
      logger?.info(
        `${TAG} Cockpit construction harness passed: primary=${extractedProposals.length}, `
        + `atomic=${compilerNormalized.length}, final=${allExtracted.length}, `
        + `primary_status=${primaryContentRiskUnavailable ? "content_risk_unavailable" : "passed"}, `
        + `transaction_attempt=${constructionAttempt}`,
      );
        break;
      } catch (error) {
        throwIfAborted(options.abortSignal, "L1 cockpit construction transaction");
        const failedStage = compilerCompleted ? "reconciliation" : "atomic compiler";
        const maximumAttempts = error instanceof L1ReconciliationGateError
          ? MAX_COCKPIT_RECONCILIATION_GATE_ATTEMPTS
          : MAX_COCKPIT_CONSTRUCTION_ATTEMPTS;
        if (error instanceof L1StructuredOutputError
          && constructionAttempt < maximumAttempts) {
          logger?.warn?.(
            `${TAG} Cockpit construction ${failedStage} returned invalid structured output; `
            + `discarding the entire attempt and restarting the independent construction transaction `
            + `within the bounded retry budget `
            + `(attempt=${constructionAttempt}/${maximumAttempts})`,
          );
          continue;
        }
        logger?.error(
          `${TAG} Cockpit construction failed closed at ${failedStage}`
          + `${primaryContentRiskUnavailable ? " after primary content-risk rejection" : ""}: `
          + `${error instanceof Error ? error.message : String(error)}`,
        );
        // Cockpit memories are committed only after the primary extractor,
        // independent atomic compiler and reconciliation gate all complete. A
        // partial provider failure must never persist primary-only candidates or
        // advance the L1 checkpoint, otherwise later retrieval cannot distinguish
        // a complete event chain from an interrupted construction transaction.
        return { success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
      }
    }
  }

  logger?.debug?.(`${TAG} Total extracted memories: ${allExtracted.length} across ${scenes.length} scene(s)`);

  if (allExtracted.length === 0) {
    // ── 评测指标：L1 提取率（提取为空的情况） ──
    if (metricInstanceId) {
      try {
        const l0Count = messages.length;
        metricProducer.send({ metric: "l0_input_count", instanceId: metricInstanceId, value: l0Count, source: "core" });
        metricProducer.send({ metric: "l1_extracted_count", instanceId: metricInstanceId, value: 0, source: "core" });
        if (l0Count > 0) {
          metricProducer.send({ metric: "l1_extraction_rate", instanceId: metricInstanceId, value: 0, source: "core" });
        }
      } catch {
        // 静默忽略，不影响业务逻辑
      }
    }
    return {
      success: true,
      extractedCount: 0,
      storedCount: 0,
      records: [],
      sceneNames,
      lastSceneName: sceneNames[sceneNames.length - 1],
    };
  }

  // Limit per session
  let extracted = allExtracted;
  if (extracted.length > maxMemoriesPerSession) {
    logger?.debug?.(`${TAG} Limiting from ${extracted.length} to ${maxMemoriesPerSession} memories per session`);
    extracted = extracted.slice(0, maxMemoriesPerSession);
  }

  // Strict cockpit writes use deterministic per-source IDs so a receipt can
  // replay a partial backend attempt without creating duplicate VDB rows.
  const memoriesWithIds = extracted.map((m) => ({
    ...m,
    record_id: strictPersistence && persistenceBatchId
      ? createL1PersistenceRecordId(persistenceBatchId, m)
      : generateMemoryId(),
  }));

  const constructionQuality = options.promptMode === "cockpit"
    ? summarizeCockpitConstructionQuality(extracted, options.constructionModel ?? options.model)
    : undefined;

  // Step 2: Batch Conflict Detection. Persistence is deliberately outside
  // this try/catch: a sink failure must never be mistaken for a dedup failure
  // and retried through the compatibility "store all" path.
  let decisions: DedupDecision[] = memoriesWithIds.map((memory) => ({
    record_id: memory.record_id,
    action: "store",
    target_ids: [],
  }));
  let dedupLatencyMs: number | null = null;

  if (enableDedup) {
    try {
      const dedupStartMs = Date.now();
      decisions = await batchDedup({
        memories: memoriesWithIds,
        config,
        logger,
        model: options.model,
        promptMode: options.promptMode,
        vectorStore: options.vectorStore,
        embeddingService: options.embeddingService,
        conflictRecallTopK: options.conflictRecallTopK,
        embeddingTimeoutMs: options.embeddingTimeoutMs,
        llmRunner: options.llmRunner,
        traceContext: { teamId, userId, agentId, sessionId },
        abortSignal: options.abortSignal,
        ...(teamId || userId || agentId || sessionId || taskId ? { filter: { teamId, userId, agentId, sessionId, taskId } } : {}),
      });
      dedupLatencyMs = Date.now() - dedupStartMs;

      // ── 评测指标：去重决策分布 ──
      if (metricInstanceId) {
        try {
          const dedupCounts = { store: 0, update: 0, merge: 0, skip: 0 };
          for (const d of decisions) {
            if (d.action in dedupCounts) {
              dedupCounts[d.action as keyof typeof dedupCounts]++;
            }
          }
          metricProducer.send({ metric: "l1_dedup_store_count", instanceId: metricInstanceId, value: dedupCounts.store, source: "core" });
          metricProducer.send({ metric: "l1_dedup_update_count", instanceId: metricInstanceId, value: dedupCounts.update, source: "core" });
          metricProducer.send({ metric: "l1_dedup_merge_count", instanceId: metricInstanceId, value: dedupCounts.merge, source: "core" });
          metricProducer.send({ metric: "l1_dedup_skip_count", instanceId: metricInstanceId, value: dedupCounts.skip, source: "core" });
        } catch {
          // 静默忽略，不影响业务逻辑
        }
      }
    } catch (err) {
      throwIfAborted(options.abortSignal, "L1 deduplication");
      logger?.warn?.(`${TAG} Batch dedup failed, storing all as new: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  let storedRecords: MemoryRecord[];
  if (strictPersistence) {
    if (!storage || !options.vectorStore || !persistenceBatchId) {
      // Guard repeated here for type narrowing and defense against future
      // control-flow changes around the pre-construction replay check.
      return { success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
    }
    try {
      const plan = await prepareL1PersistencePlan({
        batchId: persistenceBatchId,
        scope: persistenceScope,
        memoriesWithIds,
        decisions,
        baseDir,
        vectorStore: options.vectorStore,
        outcome: {
          extractedCount: extracted.length,
          sceneNames,
          lastSceneName: sceneNames[sceneNames.length - 1],
          constructionQuality,
        },
        logger,
        abortSignal: options.abortSignal,
      });
      const committed = await commitL1PersistencePlan({
        plan,
        storage,
        vectorStore: options.vectorStore,
        embeddingService: options.embeddingService,
        logger,
        abortSignal: options.abortSignal,
      });
      storedRecords = committed.records;
    } catch (error) {
      throwIfAborted(options.abortSignal, "L1 strict persistence");
      logger?.error?.(
        `${TAG} Strict L1 persistence failed closed for ${persistenceBatchId}: `
        + `${error instanceof Error ? error.message : String(error)}`,
      );
      return { success: false, extractedCount: 0, storedCount: 0, records: [], sceneNames: [] };
    }
  } else if (enableDedup) {
    storedRecords = await applyDecisions({
      memoriesWithIds,
      decisions,
      baseDir,
      sessionKey,
      sessionId,
      taskId,
      teamId,
      userId,
      agentId,
      logger,
      vectorStore: options.vectorStore,
      embeddingService: options.embeddingService,
      storage,
    });
  } else {
    storedRecords = await storeAllDirectly(memoriesWithIds, baseDir, sessionKey, sessionId, taskId, teamId, userId, agentId, logger, options.vectorStore, options.embeddingService, storage);
  }

  logger?.info(`${TAG} Extraction complete: extracted=${extracted.length}, stored=${storedRecords.length}`);

  // ── l1_extraction metric ──
  if (metricInstanceId && logger) {
    // Build type distribution of stored memories
    const memoriesByType: Record<string, number> = {};
    for (const r of storedRecords) {
      memoriesByType[r.type] = (memoriesByType[r.type] ?? 0) + 1;
    }
    report("l1_extraction", {
      sessionKey,
      inputMessageCount: messages.length,
      memoriesExtracted: extracted.length,
      memoriesStored: storedRecords.length,
      memoriesStoredContent: storedRecords.map((r) => ({
        content: r.content,
        type: r.type,
        scene: r.scene_name ?? null,
        constructionQuality: (r.metadata as Record<string, unknown>).construction_quality ?? null,
        constructionModel: (r.metadata as Record<string, unknown>).construction_model ?? null,
      })),
      memoriesByType,
      constructionQuality,
      totalDurationMs: Date.now() - l1StartMs,
      success: true,
      error: null,
    });
  }

  // ── 评测指标：L1 提取率 ──
  if (metricInstanceId) {
    try {
      const l0Count = messages.length;
      const l1Count = extracted.length;
      metricProducer.send({ metric: "l0_input_count", instanceId: metricInstanceId, value: l0Count, source: "core" });
      metricProducer.send({ metric: "l1_extracted_count", instanceId: metricInstanceId, value: l1Count, source: "core" });
      if (l0Count > 0) {
        metricProducer.send({ metric: "l1_extraction_rate", instanceId: metricInstanceId, value: l1Count / l0Count, source: "core" });
      }
    } catch {
      // 静默忽略，不影响业务逻辑
    }
  }

  // ── 评测指标：L1 延迟 ──
  try {
    reportL1LatencyMetrics({
      instanceId: metricInstanceId ?? "",
      extractionLatencyMs: Date.now() - l1StartMs,
      dedupLatencyMs,
      hasError: false,
    });
  } catch {
    // 静默忽略
  }

  return {
    success: true,
    extractedCount: extracted.length,
    storedCount: storedRecords.length,
    records: storedRecords,
    sceneNames,
    lastSceneName: sceneNames[sceneNames.length - 1],
    constructionQuality,
  };
}

// ============================
// LLM call
// ============================

/**
 * Call LLM to extract scene-segmented memories from conversation messages.
 */
async function callLlmExtraction(params: {
  newMessages: ConversationMessage[];
  backgroundMessages: ConversationMessage[];
  previousSceneName?: string;
  config: unknown;
  logger?: Logger;
  model?: string;
  promptMode?: MemoryPromptMode;
  priorStructuredMemories?: CockpitPriorMemoryContext[];
  /** Host-neutral LLM runner — when provided, used instead of CleanContextRunner. */
  llmRunner?: LLMRunner;
  /** langfuse 上报身份四元组（team/user/agent/session）。 */
  traceContext?: TraceContext;
  /** Resolved per-call timeout (milliseconds). */
  timeoutMs: number;
  abortSignal?: AbortSignal;
}): Promise<SceneSegment[]> {
  const { newMessages, backgroundMessages, previousSceneName, config, logger, model, promptMode = "chat", priorStructuredMemories, llmRunner, traceContext, timeoutMs, abortSignal } = params;

  const systemPrompt = getExtractMemoriesSystemPrompt(promptMode);
  const userPrompt = formatExtractionPrompt({
    newMessages,
    backgroundMessages,
    previousSceneName,
    ...(promptMode === "cockpit" ? { priorStructuredMemories: priorStructuredMemories ?? [] } : {}),
  });
  const minimizedMessages = promptMode === "cockpit"
    ? minimizeCockpitProviderMessages(newMessages)
    : newMessages;
  const contextMinimizedPrompt = promptMode === "cockpit"
    ? formatExtractionPrompt({
        newMessages: minimizedMessages,
        backgroundMessages: [],
        previousSceneName,
        priorStructuredMemories: [],
      })
    : undefined;

  // [l1-debug] ENTRY — what are we about to ask the LLM to extract?
  logger?.debug?.(
    `${TAG} [l1-debug] ENTRY taskId=l1-extraction, promptMode=${promptMode}, newMsgs=${newMessages.length}, bgMsgs=${backgroundMessages.length}, userPromptLen=${userPrompt.length}, sysPromptLen=${systemPrompt.length}, model=${model ?? "(default)"}, previousSceneName=${previousSceneName ? JSON.stringify(previousSceneName) : "(none)"}, runnerKind=${llmRunner ? "llmRunner" : "CleanContextRunner"}`,
  );

  // langfuse trace 语义：让此次 L1 抽取在 UI 有稳定 name / 顶级 user/session 列
  // / 可筛选 tags。避免所有记忆抽取都显示为 Unnamed trace。
  const traceParams = buildTraceParams("memory.l1-extract", traceContext);
  const result = await runLlmTaskWithOptionalPriorFallback({
    prompt: userPrompt,
    contextMinimizedPrompt,
    systemPrompt,
    taskId: "l1-extraction",
    config,
    logger,
    model,
    llmRunner,
    timeoutMs,
    traceParams,
    abortSignal,
  });

  return parseExtractionResult(result, logger);
}

async function callCockpitAtomicCompiler(params: {
  newMessages: ConversationMessage[];
  priorStructuredMemories: CockpitPriorMemoryContext[];
  maxMemories: number;
  sourceCoverageObligations: CockpitSourceCoverageObligation[];
  config: unknown;
  logger?: Logger;
  model?: string;
  llmRunner?: LLMRunner;
  traceContext?: TraceContext;
  timeoutMs: number;
  transactionAttempt?: number;
  abortSignal?: AbortSignal;
}): Promise<SceneSegment[]> {
  const {
    newMessages,
    priorStructuredMemories,
    maxMemories,
    sourceCoverageObligations,
    config,
    logger,
    model,
    llmRunner,
    traceContext,
    timeoutMs,
    transactionAttempt = 1,
    abortSignal,
  } = params;
  const taskId = transactionAttempt > 1
    ? "l1-cockpit-atomic-compiler-transaction-retry"
    : "l1-cockpit-atomic-compiler";
  const prompt = formatCockpitAtomicCompilerPrompt({
    newMessages,
    priorStructuredMemories,
    maxMemories,
    sourceCoverageObligations,
  });
  logger?.debug?.(
    `${TAG} [l1-debug] ENTRY taskId=${taskId}, `
    + `newMsgs=${newMessages.length}, prior=${priorStructuredMemories.length}, `
    + `draft=hidden, promptLen=${prompt.length}, model=${model ?? "(default)"}`,
  );
  const traceParams = buildTraceParams("memory.l1-atomic-compiler", traceContext);
  const contextMinimizedPrompt = formatCockpitAtomicCompilerPrompt({
    newMessages: minimizeCockpitProviderMessages(newMessages),
    priorStructuredMemories: [],
    maxMemories,
    sourceCoverageObligations,
  });
  const result = await runLlmTaskWithOptionalPriorFallback({
    prompt,
    contextMinimizedPrompt,
    systemPrompt: COCKPIT_ATOMIC_COMPILER_SYSTEM_PROMPT,
    taskId,
    config,
    logger,
    model,
    llmRunner,
    timeoutMs,
    traceParams,
    abortSignal,
  });
  return parseExtractionResult(result, logger);
}

async function callCockpitCoverageFactCompiler(params: {
  sourceMessage: ConversationMessage;
  priorStructuredMemories: CockpitPriorMemoryContext[];
  obligation: CockpitSourceCoverageObligation;
  maxMemories: number;
  verificationPass: boolean;
  config: unknown;
  logger?: Logger;
  model?: string;
  llmRunner?: LLMRunner;
  traceContext?: TraceContext;
  timeoutMs: number;
  transactionAttempt?: number;
  beforeProviderCall?: () => void;
  abortSignal?: AbortSignal;
}): Promise<SceneSegment[]> {
  const {
    sourceMessage,
    priorStructuredMemories,
    obligation,
    maxMemories,
    verificationPass,
    config,
    logger,
    model,
    llmRunner,
    traceContext,
    timeoutMs,
    transactionAttempt = 1,
    beforeProviderCall,
    abortSignal,
  } = params;
  const taskId = verificationPass
    ? transactionAttempt > 1
      ? "l1-cockpit-coverage-fact-verifier-transaction-retry"
      : "l1-cockpit-coverage-fact-verifier"
    : transactionAttempt > 1
      ? "l1-cockpit-coverage-fact-compiler-transaction-retry"
      : "l1-cockpit-coverage-fact-compiler";
  const prompt = formatCockpitCoverageFactCompilerPrompt({
    sourceMessage,
    priorStructuredMemories,
    obligation,
    maxMemories,
  });
  logger?.debug?.(
    `${TAG} [l1-debug] ENTRY taskId=${taskId}, source=${sourceMessage.id}, `
    + `coverage=${obligation.id}, prior=${priorStructuredMemories.length}, `
    + `promptLen=${prompt.length}, model=${model ?? "(default)"}`,
  );
  const traceParams = buildTraceParams(
    verificationPass
      ? "memory.l1-coverage-fact-verifier"
      : "memory.l1-coverage-fact-compiler",
    traceContext,
  );
  const contextMinimizedPrompt = formatCockpitCoverageFactCompilerPrompt({
    // Evidence offsets are defined against the exact original source bytes.
    // The bounded fallback removes only optional prior context; it must never
    // strip a transport envelope and silently change the coordinate system.
    sourceMessage,
    priorStructuredMemories: [],
    obligation,
    maxMemories,
  });
  const result = await runLlmTaskWithOptionalPriorFallback({
    prompt,
    contextMinimizedPrompt,
    systemPrompt: COCKPIT_COVERAGE_FACT_COMPILER_SYSTEM_PROMPT,
    taskId,
    config,
    logger,
    model,
    llmRunner,
    timeoutMs,
    traceParams,
    beforeProviderCall,
    abortSignal,
    // Keep one runner invocation equal to one provider request so the global
    // directed-call budget cannot be bypassed by an adapter-level length retry.
    retryOnLength: false,
  });
  return parseExtractionResult(result, logger);
}

async function callCockpitConstructionReconciliation(params: {
  newMessages: ConversationMessage[];
  priorStructuredMemories: CockpitPriorMemoryContext[];
  primaryMemories: ExtractedMemory[];
  atomicMemories: ExtractedMemory[];
  sourceCoverageObligations: CockpitSourceCoverageObligation[];
  maxMemories: number;
  config: unknown;
  logger?: Logger;
  model?: string;
  llmRunner?: LLMRunner;
  traceContext?: TraceContext;
  timeoutMs: number;
  transactionAttempt?: number;
  abortSignal?: AbortSignal;
  repairFeedback?: {
    issues: string[];
    uncoveredCandidateIds: string[];
    previousMemories: ExtractedMemory[];
    diagnostics: CockpitReconciliationDiagnostic[];
  };
}): Promise<SceneSegment[]> {
  const {
    newMessages,
    priorStructuredMemories,
    primaryMemories,
    atomicMemories,
    sourceCoverageObligations,
    maxMemories,
    config,
    logger,
    model,
    llmRunner,
    traceContext,
    timeoutMs,
    transactionAttempt = 1,
    repairFeedback,
    abortSignal,
  } = params;
  const taskId = transactionAttempt > 1
    ? "l1-cockpit-construction-reconcile-transaction-retry"
    : "l1-cockpit-construction-reconcile";
  const prompt = formatCockpitConstructionReconciliationPrompt({
    newMessages,
    priorStructuredMemories,
    primaryMemories,
    atomicMemories,
    sourceCoverageObligations,
    maxMemories,
    repairFeedback,
  });
  logger?.debug?.(
    `${TAG} [l1-debug] ENTRY taskId=${taskId}, `
    + `newMsgs=${newMessages.length}, prior=${priorStructuredMemories.length}, `
    + `primary=${primaryMemories.length}, atomic=${atomicMemories.length}, `
    + `repair=${repairFeedback ? "yes" : "no"}, promptLen=${prompt.length}, model=${model ?? "(default)"}`,
  );
  const traceParams = buildTraceParams("memory.l1-construction-reconcile", traceContext);
  const contextMinimizedPrompt = formatCockpitConstructionReconciliationPrompt({
    newMessages: minimizeCockpitProviderMessages(newMessages),
    priorStructuredMemories: [],
    primaryMemories,
    atomicMemories,
    sourceCoverageObligations,
    maxMemories,
    repairFeedback,
  });
  const result = await runLlmTaskWithOptionalPriorFallback({
    prompt,
    contextMinimizedPrompt,
    systemPrompt: COCKPIT_CONSTRUCTION_RECONCILER_SYSTEM_PROMPT,
    taskId,
    config,
    logger,
    model,
    llmRunner,
    timeoutMs,
    traceParams,
    abortSignal,
  });
  return parseExtractionResult(result, logger);
}

const COCKPIT_TRANSPORT_ENVELOPE = /^\s*\[[^\]\r\n]{1,200}\]\s*\[source_time=[^\]\r\n]{1,80}\]\s*\[source_role=(?:user|assistant|tool)\]\s*/u;

/**
 * Build a source-evidence-only provider view without changing any factual
 * user text. Ordinary assistant acknowledgements are not evidence under the
 * cockpit contract. The repeated benchmark transport envelope is also
 * redundant because id, role and timestamp remain first-class fields in the
 * prompt. Named-speaker labels and the full user utterance are preserved.
 */
function minimizeCockpitProviderMessages(messages: ConversationMessage[]): ConversationMessage[] {
  const userMessages = messages.filter((message) => message.role === "user");
  const evidenceMessages = userMessages.length > 0 ? userMessages : messages;
  return evidenceMessages.map((message) => {
    const stripped = message.content.replace(COCKPIT_TRANSPORT_ENVELOPE, "");
    return {
      ...message,
      content: stripped.trim().length > 0 ? stripped : message.content,
    };
  });
}

/**
 * DeepSeek can reject an otherwise benign construction request when optional
 * historical records are present. Those records are useful for lineage, but
 * the current source messages remain the only evidence for new facts. Retry
 * exactly once with only that optional prior block removed, and only for the
 * provider's explicit content-risk error. Authentication, quota, timeout and
 * all other failures remain fail-closed.
 */
async function runLlmTaskWithOptionalPriorFallback(params: {
  prompt: string;
  contextMinimizedPrompt?: string;
  systemPrompt: string;
  taskId: string;
  config: unknown;
  logger?: Logger;
  model?: string;
  llmRunner?: LLMRunner;
  timeoutMs: number;
  traceParams: ReturnType<typeof buildTraceParams>;
  /** Called immediately before every real provider attempt, including fallback. */
  beforeProviderCall?: () => void;
  /** Directed compilers disable adapter-internal replay for an exact call budget. */
  retryOnLength?: boolean;
  abortSignal?: AbortSignal;
}): Promise<string> {
  const {
    prompt,
    contextMinimizedPrompt,
    systemPrompt,
    taskId,
    config,
    logger,
    model,
    llmRunner,
    timeoutMs,
    traceParams,
    beforeProviderCall,
    retryOnLength = true,
    abortSignal,
  } = params;
  const runOnce = async (activePrompt: string, activeTaskId: string): Promise<string> => {
    throwIfAborted(abortSignal, `${activeTaskId} before provider call`);
    beforeProviderCall?.();
    let result: string;
    if (llmRunner) {
      result = await llmRunner.run({
        prompt: activePrompt,
        systemPrompt,
        taskId: activeTaskId,
        timeoutMs,
        thinkingMode: "disabled",
        retryOnLength,
        abortSignal,
        ...traceParams,
      });
    } else {
      const runner = new CleanContextRunner({
        config,
        modelRef: model,
        enableTools: false,
        logger,
      });
      result = await runner.run({
        prompt: activePrompt,
        systemPrompt,
        taskId: activeTaskId,
        timeoutMs,
        thinkingMode: "disabled",
        retryOnLength,
        abortSignal,
        ...traceParams,
      });
    }
    throwIfAborted(abortSignal, `${activeTaskId} after provider call`);
    return result;
  };

  try {
    return await runOnce(prompt, taskId);
  } catch (error) {
    if (!contextMinimizedPrompt
      || contextMinimizedPrompt === prompt
      || !isProviderContentRiskError(error)) throw error;
    const fallbackTaskId = `${taskId}-context-minimized`;
    logger?.warn?.(
      `${TAG} Provider content-risk rejected construction context for taskId=${taskId}; `
      + `retrying once with optional prior/background/assistant transport context omitted `
      + `(promptLen=${prompt.length}->${contextMinimizedPrompt.length})`,
    );
    logger?.debug?.(
      `${TAG} [l1-debug] ENTRY taskId=${fallbackTaskId}, `
      + `promptLen=${contextMinimizedPrompt.length}, sysPromptLen=${systemPrompt.length}, `
      + `model=${model ?? "(default)"}, fallback=context-minimized`,
    );
    return runOnce(contextMinimizedPrompt, fallbackTaskId);
  }
}

function isProviderContentRiskError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /\bcontent[\s_-]+exists[\s_-]+risk\b/iu.test(message);
}

/**
 * Parse the LLM's JSON response into SceneSegment array.
 * Expected format: [{scene_name, message_ids, memories: [...]}]
 */
function parseExtractionResult(raw: string, logger?: Logger): SceneSegment[] {
  try {
    // Strip markdown code block wrappers if present
    let cleaned = raw.trim();
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
    }

    // Try to extract JSON array
    const arrayMatch = cleaned.match(/\[[\s\S]*\]/);
    if (!arrayMatch) {
      logger?.warn?.(`${TAG} No JSON array found in extraction response`);
      // [l1-debug] NO_JSON — dump the full raw so we can see what the LLM actually said
      const rawPreview = raw.slice(0, 2048);
      logger?.warn?.(
        `${TAG} [l1-debug] NO_JSON taskId=l1-extraction, rawLen=${raw.length}, cleanedLen=${cleaned.length}, rawFull=${JSON.stringify(rawPreview)}${raw.length > 2048 ? `…(+${raw.length - 2048})` : ""}`,
      );
      throw new Error("L1 extraction response did not contain a JSON array");
    }

    // Sanitize control characters inside JSON string literals that LLM may produce.
    // Some weaker OpenAI-compatible models occasionally emit bare identifiers for
    // numeric fields (e.g. `"priority": sheet`). Repair only known safe fields and
    // retry once so one bad scalar does not drop the whole extraction result.
    const sanitized = sanitizeJsonForParse(arrayMatch[0]);
    let parsed: unknown[];
    try {
      parsed = JSON.parse(sanitized) as unknown[];
    } catch (err) {
      const repaired = repairExtractionJson(sanitized);
      if (repaired === sanitized) throw err;
      parsed = JSON.parse(repaired) as unknown[];
      logger?.warn?.(`${TAG} Repaired non-strict extraction JSON: ${err instanceof Error ? err.message : String(err)}`);
    }

    if (!Array.isArray(parsed)) {
      throw new Error("L1 extraction response is not an array");
    }

    const scenes: SceneSegment[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const s = item as Record<string, unknown>;

      scenes.push({
        scene_name: typeof s.scene_name === "string" ? s.scene_name : "未知情境",
        message_ids: Array.isArray(s.message_ids) ? s.message_ids.map(String) : [],
        memories: Array.isArray(s.memories)
          ? (s.memories as Array<Record<string, unknown>>)
              .filter((m) => m && typeof m === "object" && typeof m.content === "string" && (m.content as string).length > 0)
              .map((m) => ({
                content: String(m.content),
                type: String(m.type ?? "episodic"),
                priority: typeof m.priority === "number" ? m.priority : 50,
                source_message_ids: Array.isArray(m.source_message_ids) ? m.source_message_ids.map(String) : [],
                metadata: (m.metadata && typeof m.metadata === "object" ? m.metadata : {}) as Record<string, unknown>,
              }))
          : [],
      });
    }

    if (parsed.length > 0 && scenes.length === 0) {
      throw new Error("L1 extraction response contained no valid scene objects");
    }
    return scenes;
  } catch (err) {
    logger?.warn?.(`${TAG} Failed to parse extraction result: ${err instanceof Error ? err.message : String(err)}`);
    const rawPreview = raw.slice(0, 2048);
    logger?.warn?.(
      `${TAG} [l1-debug] PARSE_FAIL rawLen=${raw.length}, rawFull=${JSON.stringify(rawPreview)}${raw.length > 2048 ? `…(+${raw.length - 2048})` : ""}`,
    );
    throw err instanceof L1StructuredOutputError
      ? err
      : new L1StructuredOutputError(err instanceof Error ? err.message : String(err));
  }
}

function repairExtractionJson(json: string): string {
  return json
    .replace(
      /("priority"\s*:\s*)(?!-?\d+(?:\.\d+)?\s*[,}]|"[^"\\]*(?:\\.[^"\\]*)*"\s*[,}])([\s\S]*?)(?=,\s*"(?:content|type|priority|source_message_ids|metadata)"\s*:|[}\]])/g,
      (_m, prefix: string) => `${prefix}50`,
    )
    .replace(/,\s*([}\]])/g, "$1");
}

// ============================
// Write helpers
// ============================

/**
 * Apply batch dedup decisions — write memories according to their decisions.
 */
async function applyDecisions(params: {
  memoriesWithIds: Array<ExtractedMemory & { record_id: string }>;
  decisions: DedupDecision[];
  baseDir: string;
  sessionKey: string;
  sessionId?: string;
  taskId?: string;
  teamId?: string;
  userId?: string;
  agentId?: string;
  logger?: Logger;
  vectorStore?: IMemoryStore;
  embeddingService?: EmbeddingService;
  storage?: StorageAdapter;
}): Promise<MemoryRecord[]> {
  const { memoriesWithIds, decisions, baseDir, sessionKey, sessionId, taskId, teamId, userId, agentId, logger, vectorStore, embeddingService, storage } = params;
  const storedRecords: MemoryRecord[] = [];

  // Build a map from record_id → decision
  const decisionMap = new Map<string, DedupDecision>();
  for (const d of decisions) {
    decisionMap.set(d.record_id, d);
  }

  for (const memoryWithId of memoriesWithIds) {
    const decision = decisionMap.get(memoryWithId.record_id) ?? {
      record_id: memoryWithId.record_id,
      action: "store" as const,
      target_ids: [],
    };

    try {
      const record = await writeMemory({
        memory: memoryWithId,
        decision,
        baseDir,
        sessionKey,
        sessionId,
        taskId,
        teamId,
        userId,
        agentId,
        logger,
        vectorStore,
        embeddingService,
        storage,
      });

      if (record) {
        storedRecords.push(record);
      }
    } catch (err) {
      logger?.warn?.(
        `${TAG} Write failed for memory "${memoryWithId.content.slice(0, 50)}...": ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  return storedRecords;
}

/**
 * Store all memories directly (no dedup).
 */
async function storeAllDirectly(
  memoriesWithIds: Array<ExtractedMemory & { record_id: string }>,
  baseDir: string,
  sessionKey: string,
  sessionId: string | undefined,
  taskId: string | undefined,
  teamId?: string,
  userId?: string,
  agentId?: string,
  logger?: Logger,
  vectorStore?: IMemoryStore,
  embeddingService?: EmbeddingService,
  storage?: StorageAdapter,
): Promise<MemoryRecord[]> {
  const storedRecords: MemoryRecord[] = [];

  for (const memoryWithId of memoriesWithIds) {
    try {
      const record = await writeMemory({
        memory: memoryWithId,
        decision: {
          record_id: memoryWithId.record_id,
          action: "store",
          target_ids: [],
        },
        baseDir,
        sessionKey,
        sessionId,
        taskId,
        teamId,
        userId,
        agentId,
        logger,
        vectorStore,
        embeddingService,
        storage,
      });
      if (record) {
        storedRecords.push(record);
      }
    } catch (err) {
      logger?.warn?.(
        `${TAG} Write failed for memory "${memoryWithId.content.slice(0, 50)}...": ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  return storedRecords;
}

// ============================
// Helpers
// ============================

const VALID_TYPES: MemoryType[] = ["persona", "episodic", "instruction", "work_fact", "work_task", "work_method", "work_artifact"];

function stampCockpitConstructionHarness(
  memory: ExtractedMemory,
  model: string | undefined,
  compilerStatus: "passed" | "failed",
  reconciliationStatus: "passed" | "failed" | "skipped",
  primaryStatus: "passed" | "content_risk_unavailable" = "passed",
  transactionAttempt = 1,
): ExtractedMemory {
  const overallPassed = compilerStatus === "passed" && reconciliationStatus === "passed";
  return {
    ...memory,
    metadata: {
      ...(memory.metadata as Record<string, unknown>),
      construction_compiler_status: compilerStatus,
      ...(model ? { construction_compiler_model: model } : {}),
      construction_primary_status: primaryStatus,
      construction_transaction_attempt: transactionAttempt,
      construction_reconciliation_status: reconciliationStatus,
      ...(model ? { construction_reconciliation_model: model } : {}),
      construction_assembler_status: overallPassed ? "passed" : "failed",
      construction_assembler_version: "cockpit-deterministic-v1",
      construction_review_mode: "independent_source_compiler_model_reconciled_deterministic_assembled",
      // Preserve the RC50/RC51 metadata keys for downstream compatibility.
      construction_review_status: overallPassed ? "passed" : "failed",
      ...(model ? { construction_review_model: model } : {}),
    },
  };
}

function isCockpitTransitionProposal(memory: ExtractedMemory): boolean {
  const metadata = memory.metadata as Record<string, unknown>;
  const relation = typeof metadata.relation === "string"
    ? metadata.relation.trim().toLocaleLowerCase()
    : "";
  const status = typeof metadata.action_status === "string"
    ? metadata.action_status.trim().toLocaleLowerCase()
    : "";
  const supersedes = metadata.supersedes;
  return relation === "updated"
    || relation === "cancelled"
    || relation === "negated"
    || status === "cancelled"
    || (typeof supersedes === "string" && supersedes.trim().length > 0)
    || (Array.isArray(supersedes) && supersedes.length > 0);
}

function isCompleteCockpitProposal(memory: ExtractedMemory): boolean {
  const metadata = memory.metadata as Record<string, unknown>;
  const quality = metadata.construction_quality;
  return Boolean(quality
    && typeof quality === "object"
    && !Array.isArray(quality)
    && (quality as Record<string, unknown>).status === "complete");
}

function hasCockpitConstructionIssue(memory: ExtractedMemory, issue: string): boolean {
  const metadata = memory.metadata as Record<string, unknown>;
  const quality = metadata.construction_quality;
  if (!quality || typeof quality !== "object" || Array.isArray(quality)) return false;
  const issues = (quality as Record<string, unknown>).issues;
  return Array.isArray(issues) && issues.includes(issue);
}

function normalizeType(raw: string): MemoryType | null {
  const lower = raw.toLowerCase().trim();
  if (VALID_TYPES.includes(lower as MemoryType)) {
    return lower as MemoryType;
  }
  // Handle legacy type names
  if (lower === "episode") return "episodic";
  if (lower === "instruct") return "instruction";
  if (lower === "preference") return "persona"; // fold preference into persona
  return null;
}

function summarizeCockpitConstructionQuality(
  memories: ExtractedMemory[],
  model?: string,
): NonNullable<L1ExtractionResult["constructionQuality"]> {
  const summary = { model, complete: 0, partial: 0, invalid: 0, averageScore: 0 };
  let totalScore = 0;
  for (const memory of memories) {
    const quality = (memory.metadata as Record<string, unknown>).construction_quality;
    const value = quality && typeof quality === "object"
      ? quality as Record<string, unknown>
      : {};
    const status = value.status;
    if (status === "complete" || status === "partial" || status === "invalid") summary[status] += 1;
    else summary.invalid += 1;
    totalScore += typeof value.score === "number" && Number.isFinite(value.score) ? value.score : 0;
  }
  summary.averageScore = memories.length > 0 ? totalScore / memories.length : 0;
  return summary;
}
