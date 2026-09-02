/**
 * Deterministic gate for the model-based cockpit construction assembler.
 *
 * Semantic canonicalization stays in the Flash harness, while this module
 * verifies that the assembler did not silently drop a source-only atomic or
 * deterministic source-coverage obligation, invent proposal IDs, cross
 * domains/sources, or return duplicate final state identities. Primary
 * proposals remain advisory because schema-complete does not imply
 * semantically authoritative. It does not alter TencentDB persistence or recall.
 */

import type { ExtractedMemory } from "./l1-writer.js";
import {
  canonicalCockpitSceneClass,
  isValidControlledCockpitOntology,
} from "./cockpit-ontology.js";

export interface CockpitReconciliationInput {
  id: string;
  memory: ExtractedMemory;
}

export interface CockpitReconciliationPriorMemory {
  record_id: string;
  /** Present on real cockpit prior contexts; optional for legacy callers. */
  type?: string;
  scene_name?: string;
  session_id?: string;
  metadata: Record<string, unknown>;
}

export interface CockpitReconciliationGateResult {
  accepted: boolean;
  issues: string[];
  requiredCandidateIds: string[];
  coveredCandidateIds: string[];
  uncoveredCandidateIds: string[];
  diagnostics: CockpitReconciliationDiagnostic[];
}

export interface CockpitReconciliationDiagnostic {
  rowIndex: number;
  issue: string;
  stateKey?: string;
  episodeKey?: string;
  relation?: string;
  qualityIssues: string[];
  matchingLivePriorRecordIds: string[];
  livePriorTargets: Array<{
    recordId: string;
    domain?: string;
    slot?: string;
    stateKey?: string;
    episodeKey?: string;
  }>;
}

// Deterministic assembler rewrites may combine two independently extracted
// facts into one gate-ready row (for example, an appointment activity plus its
// co-episodic destination). The resulting value intentionally differs from
// every individual model proposal, so an ordinary factual-candidate check
// cannot prove it. Keep that proof as an object-identity capability: only this
// process's assembler can register the exact returned object, model metadata
// and serialized repair strings cannot forge membership, and a copied or
// deserialized row fails closed.
const cockpitAssemblerVerifiedFactualRewrites = new WeakMap<ExtractedMemory, string>();

function cockpitAssemblerVerifiedFingerprint(memory: ExtractedMemory): string | undefined {
  try {
    return JSON.stringify({
      content: memory.content,
      type: memory.type,
      priority: memory.priority,
      scene_name: memory.scene_name,
      source_message_ids: memory.source_message_ids,
      metadata: memory.metadata,
    });
  } catch {
    return undefined;
  }
}

/** @internal Called only by the deterministic cockpit construction assembler. */
export function registerCockpitAssemblerVerifiedFactualRewrite(
  memory: ExtractedMemory,
): void {
  const fingerprint = cockpitAssemblerVerifiedFingerprint(memory);
  if (fingerprint !== undefined) {
    cockpitAssemblerVerifiedFactualRewrites.set(memory, fingerprint);
  }
}

function hasCockpitAssemblerVerifiedFactualRewrite(memory: ExtractedMemory): boolean {
  const registered = cockpitAssemblerVerifiedFactualRewrites.get(memory);
  return registered !== undefined
    && registered === cockpitAssemblerVerifiedFingerprint(memory);
}

function metadataOf(memory: ExtractedMemory): Record<string, unknown> {
  return memory.metadata as Record<string, unknown>;
}

function stringValue(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed || undefined;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((entry): entry is string =>
    typeof entry === "string" && entry.trim().length > 0
  ))];
}

function qualityStatus(memory: ExtractedMemory): string | undefined {
  const quality = metadataOf(memory).construction_quality;
  return quality && typeof quality === "object" && !Array.isArray(quality)
    ? stringValue((quality as Record<string, unknown>).status)
    : undefined;
}

function qualityIssues(memory: ExtractedMemory): string[] {
  const quality = metadataOf(memory).construction_quality;
  return quality && typeof quality === "object" && !Array.isArray(quality)
    ? stringArray((quality as Record<string, unknown>).issues)
    : [];
}

function qualityRepairs(memory: ExtractedMemory): string[] {
  const quality = metadataOf(memory).construction_quality;
  return quality && typeof quality === "object" && !Array.isArray(quality)
    ? stringArray((quality as Record<string, unknown>).repairs)
    : [];
}

function hasSharedSource(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const rightIds = new Set(right.source_message_ids);
  return left.source_message_ids.some((id) => rightIds.has(id));
}

function hasExactSourceSet(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const normalize = (values: string[]): string => JSON.stringify([...new Set(values)].sort());
  return normalize(left.source_message_ids) === normalize(right.source_message_ids)
    && normalize(stringArray(metadataOf(left).source_message_ids))
      === normalize(stringArray(metadataOf(right).source_message_ids));
}

function isSingleEventCoverageObligation(input: CockpitReconciliationInput): boolean {
  if (!input.id.startsWith("coverage:")) return false;
  const metadata = metadataOf(input.memory);
  const boundedCount = (key: string): boolean => {
    const value = metadata[key];
    return value === undefined || (typeof value === "number" && value <= 1);
  };
  return metadata.coverage_required_fact_count === 1
    && metadata.coverage_requires_distinct_evidence_bindings !== true
    && stringArray(metadata.coverage_evidence_group_ids).length === 1
    && stringArray(metadata.coverage_event_anchors).length === 1
    && [
      "coverage_required_subject_count",
      "coverage_required_condition_count",
      "coverage_required_seat_zone_count",
      "coverage_required_temporal_count",
      "coverage_required_state_qualifier_count",
    ].every(boundedCount);
}

/**
 * Detect two incompatible complete representations claiming the same exact
 * single source event. This is an ambiguity detector, not a deduplicator: it
 * deliberately ignores raw value/target and model-authored episode aliases,
 * but requires every independently meaningful binding and time axis to agree.
 * The caller fails the transaction closed instead of choosing either value.
 */
function conflictingSingleEventCoverageAlias(
  left: ExtractedMemory,
  right: ExtractedMemory,
): boolean {
  if (qualityStatus(left) !== "complete"
    || qualityStatus(right) !== "complete"
    || left.type !== right.type
    || !hasExactSourceSet(left, right)) return false;
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  const leftEpisode = stringValue(leftMetadata.episode_key);
  const rightEpisode = stringValue(rightMetadata.episode_key);
  if (!leftEpisode || !rightEpisode || leftEpisode === rightEpisode) return false;
  return [
    "domain",
    "slot",
    "state_key",
    "unit",
    "constraint_target",
    "state_qualifier",
    "relation",
    "action_status",
    "record_kind",
    "subject",
    "occupant_scope",
    "vehicle_scope",
    "seat_zone",
    "valid_from",
    "valid_to",
    "activity_start_time",
    "activity_end_time",
    "condition",
    "trigger",
    "timezone",
    "time_precision",
    "temporal_status",
    "mentioned_at",
    "source_session_id",
  ].every((key) => comparableValue(leftMetadata[key]) === comparableValue(rightMetadata[key]))
    && comparableValue(stringArray(leftMetadata.supersedes).sort())
      === comparableValue(stringArray(rightMetadata.supersedes).sort());
}

function canonicalBindingValue(key: string, value: unknown): string | undefined {
  const normalized = stringValue(value)?.normalize("NFKC");
  if (!normalized) return undefined;
  const folded = normalized.toLocaleLowerCase();
  if (["unknown", "unspecified", "none", "null", "未知", "未指定"].includes(folded)) return undefined;
  // A state qualifier is a source-bound identity label, not a vocabulary
  // token. Preserve case after NFKC exactly as the evidence binder does.
  const text = key === "state_qualifier" ? normalized : folded;
  if (key === "subject" && ["user", "用户", "我", "本人", "用户本人"].includes(text)) return "user";
  if (key === "seat_zone") {
    if (["driver", "driver-seat", "驾驶位", "驾驶员", "主驾", "主驾驶"].includes(text)) return "driver";
    if (["front-passenger", "front-passenger-seat", "副驾", "副驾驶", "前排乘客位"].includes(text)) {
      return "front-passenger";
    }
  }
  return text;
}

export function preservesCockpitEvidenceBinding(input: ExtractedMemory, output: ExtractedMemory): boolean {
  const inputMetadata = metadataOf(input);
  const outputMetadata = metadataOf(output);
  for (const key of ["subject", "occupant_scope", "vehicle_scope", "seat_zone"] as const) {
    const expected = canonicalBindingValue(key, inputMetadata[key]);
    if (expected && canonicalBindingValue(key, outputMetadata[key]) !== expected) return false;
  }
  for (const key of ["valid_from", "valid_to"] as const) {
    const expected = stringValue(inputMetadata[key]);
    if (expected && stringValue(outputMetadata[key]) !== expected) return false;
  }
  const expectedConstraintTarget = canonicalBindingValue(
    "constraint_target",
    inputMetadata.constraint_target,
  );
  if (expectedConstraintTarget
    && canonicalBindingValue("constraint_target", outputMetadata.constraint_target) !== expectedConstraintTarget) {
    return false;
  }
  const expectedStateQualifier = canonicalBindingValue(
    "state_qualifier",
    inputMetadata.state_qualifier,
  );
  if (expectedStateQualifier
    && canonicalBindingValue("state_qualifier", outputMetadata.state_qualifier)
      !== expectedStateQualifier) {
    return false;
  }
  return true;
}

export function cockpitFinalIdentity(memory: ExtractedMemory): string | undefined {
  const metadata = metadataOf(memory);
  const stateKey = stringValue(metadata.state_key);
  const episodeKey = stringValue(metadata.episode_key);
  if (!stateKey || !episodeKey) return undefined;
  const identityComponent = (value: unknown): string => {
    if (value === undefined) return "";
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  };
  return [
    episodeKey,
    stateKey,
    stringValue(metadata.valid_from) ?? "",
    stringValue(metadata.valid_to) ?? "",
    identityComponent(metadata.condition),
    identityComponent(metadata.trigger),
    stringValue(metadata.constraint_target) ?? "",
    stringValue(metadata.state_qualifier) ?? "",
  ].join("\u0000");
}

function eventTimeMs(value: unknown): number | undefined {
  if (typeof value !== "string" && typeof value !== "number") return undefined;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : undefined;
}

/**
 * A persisted predecessor may affect only an event at or after its own source
 * event time. When either side carries an event-time claim, both timestamps
 * must be parseable; the no-timestamp fallback is reserved for legacy unit
 * callers predating the cockpit contract.
 */
export function cockpitPriorIsCausallyAvailable(
  memory: ExtractedMemory,
  prior: CockpitReconciliationPriorMemory,
): boolean {
  const memoryRaw = metadataOf(memory).mentioned_at;
  const priorRaw = prior.metadata.mentioned_at;
  if (memoryRaw === undefined && priorRaw === undefined) return true;
  const memoryTime = eventTimeMs(memoryRaw);
  const priorTime = eventTimeMs(priorRaw);
  return memoryTime !== undefined && priorTime !== undefined && priorTime <= memoryTime;
}

function priorClassesAreCompatible(
  superseder: CockpitReconciliationPriorMemory,
  target: CockpitReconciliationPriorMemory,
): boolean {
  const supersederType = stringValue(superseder.type ?? superseder.metadata.type);
  const targetType = stringValue(target.type ?? target.metadata.type);
  const supersederScene = canonicalCockpitSceneClass(
    superseder.scene_name ?? superseder.metadata.scene_name,
    superseder.metadata.domain,
    superseder.metadata.slot,
  );
  const targetScene = canonicalCockpitSceneClass(
    target.scene_name ?? target.metadata.scene_name,
    target.metadata.domain,
    target.metadata.slot,
  );
  return (!supersederType || !targetType || supersederType === targetType)
    && (!supersederScene || !targetScene || supersederScene === targetScene);
}

function persistedSupersessionEdgeIsValid(
  superseder: CockpitReconciliationPriorMemory,
  target: CockpitReconciliationPriorMemory,
): boolean {
  const relation = stringValue(superseder.metadata.relation);
  const supersederState = stringValue(superseder.metadata.state_key);
  const targetState = stringValue(target.metadata.state_key);
  const supersederEpisode = stringValue(superseder.metadata.episode_key);
  const targetEpisode = stringValue(target.metadata.episode_key);
  if (relation !== "updated" && relation !== "cancelled" && relation !== "negated") return false;
  if (!supersederState || !targetState || !supersederEpisode || !targetEpisode
    || !priorClassesAreCompatible(superseder, target)
    || supersederState !== targetState
    || supersederEpisode !== targetEpisode
    || comparableValue(superseder.metadata.constraint_target)
      !== comparableValue(target.metadata.constraint_target)
    || comparableValue(superseder.metadata.state_qualifier)
      !== comparableValue(target.metadata.state_qualifier)) return false;
  const supersederRaw = superseder.metadata.mentioned_at;
  const targetRaw = target.metadata.mentioned_at;
  if (supersederRaw === undefined && targetRaw === undefined) return true;
  const supersederTime = eventTimeMs(supersederRaw);
  const targetTime = eventTimeMs(targetRaw);
  return supersederTime !== undefined && targetTime !== undefined && targetTime <= supersederTime;
}

/**
 * Derive the active prior set only from class/state/episode-coherent persisted
 * transition edges. A corrupt cross-class `supersedes` reference must not hide
 * the authoritative target on every subsequent construction request.
 */
export function cockpitLivePriorMemories(
  memories: CockpitReconciliationPriorMemory[],
): CockpitReconciliationPriorMemory[] {
  const byRecordId = new Map(memories.map((memory) => [memory.record_id, memory]));
  const superseded = new Set<string>();
  for (const superseder of memories) {
    for (const reference of stringArray(superseder.metadata.supersedes)) {
      const target = byRecordId.get(reference);
      if (target && persistedSupersessionEdgeIsValid(superseder, target)) superseded.add(reference);
    }
  }
  return memories.filter((memory) => {
    if (superseded.has(memory.record_id)) return false;
    const relation = stringValue(memory.metadata.relation);
    const actionStatus = stringValue(memory.metadata.action_status);
    return relation !== "cancelled" && relation !== "negated" && actionStatus !== "cancelled";
  });
}

function matchingLivePrior(
  memory: ExtractedMemory,
  priors: CockpitReconciliationPriorMemory[],
): CockpitReconciliationPriorMemory[] {
  const metadata = metadataOf(memory);
  const stateKey = stringValue(metadata.state_key);
  const episodeKey = stringValue(metadata.episode_key);
  if (!stateKey || !episodeKey) return [];
  return priors.filter((prior) =>
    cockpitPriorMatchesTransitionIdentity(prior, memory)
      && stringValue(prior.metadata.state_key) === stateKey
      && stringValue(prior.metadata.episode_key) === episodeKey
  );
}

/**
 * Match the persisted memory class whenever the prior context carries it.
 * Production cockpit prior contexts always provide both fields; the optional
 * fallback preserves compatibility for legacy in-process callers that only
 * supplied metadata before the typed prior contract was introduced.
 */
export function cockpitPriorMatchesMemoryClass(
  prior: CockpitReconciliationPriorMemory,
  memory: ExtractedMemory,
): boolean {
  const priorType = stringValue(prior.type ?? prior.metadata.type);
  const priorScene = canonicalCockpitSceneClass(
    prior.scene_name ?? prior.metadata.scene_name,
    prior.metadata.domain,
    prior.metadata.slot,
  );
  const memoryMetadata = metadataOf(memory);
  const memoryScene = canonicalCockpitSceneClass(
    memory.scene_name,
    memoryMetadata.domain,
    memoryMetadata.slot,
  );
  return (!priorType || priorType === memory.type)
    && (!priorScene || priorScene === memoryScene);
}

/** Exact state/class match, including the keyed price-constraint target. */
export function cockpitPriorMatchesMemoryState(
  prior: CockpitReconciliationPriorMemory,
  memory: ExtractedMemory,
): boolean {
  const metadata = metadataOf(memory);
  return cockpitPriorMatchesMemoryClass(prior, memory)
    && cockpitPriorIsCausallyAvailable(memory, prior)
    && stringValue(prior.metadata.state_key) === stringValue(metadata.state_key)
    && comparableValue(prior.metadata.constraint_target)
      === comparableValue(metadata.constraint_target);
}

/** Exact predecessor identity consumed by ordinary update/cancel transitions. */
export function cockpitPriorMatchesTransitionIdentity(
  prior: CockpitReconciliationPriorMemory,
  memory: ExtractedMemory,
): boolean {
  return cockpitPriorMatchesMemoryState(prior, memory)
    && stringValue(prior.metadata.episode_key)
      === stringValue(metadataOf(memory).episode_key);
}

function hasValidControlledOntology(memory: ExtractedMemory): boolean {
  const metadata = metadataOf(memory);
  return isValidControlledCockpitOntology(metadata.domain, metadata.slot);
}

export function cockpitCandidateStructurallyCovers(
  input: ExtractedMemory,
  output: ExtractedMemory,
  explicitlyCanonicalized = false,
): boolean {
  const inputMetadata = metadataOf(input);
  const outputMetadata = metadataOf(output);
  const inputDomain = stringValue(inputMetadata.domain);
  const inputSlot = stringValue(inputMetadata.slot);
  const outputDomain = stringValue(outputMetadata.domain);
  const outputSlot = stringValue(outputMetadata.slot);
  return Boolean(inputDomain && inputSlot
    && hasSharedSource(input, output)
    && preservesCockpitEvidenceBinding(input, output)
    && ((outputDomain === inputDomain && outputSlot === inputSlot) || explicitlyCanonicalized));
}

/** Exact slot/source/scope check used by the bounded directed coverage fact compiler. */
export function cockpitCoverageObligationCovers(
  input: ExtractedMemory,
  output: ExtractedMemory,
): boolean {
  return cockpitCandidateStructurallyCovers(input, output, false);
}

/**
 * Atomic candidates carry transition semantics in addition to source and slot
 * structure. A cancelled row must never consume an updated obligation merely
 * because both rows mention the same source and slot. An asserted candidate
 * may become updated later when it is deterministically matched to a live
 * predecessor; all explicit transitions otherwise remain relation-exact.
 */
export function cockpitAtomicCandidateCovers(
  input: ExtractedMemory,
  output: ExtractedMemory,
  explicitlyCanonicalized = false,
): boolean {
  if (input.type !== output.type
    || input.scene_name !== output.scene_name
    || !cockpitCandidateStructurallyCovers(input, output, explicitlyCanonicalized)) return false;
  const inputRelation = stringValue(metadataOf(input).relation);
  const outputRelation = stringValue(metadataOf(output).relation);
  if (!inputRelation) return true;
  if (inputRelation === "asserted") {
    return outputRelation === "asserted" || outputRelation === "updated";
  }
  if (inputRelation === "updated" || inputRelation === "cancelled" || inputRelation === "negated") {
    return outputRelation === inputRelation;
  }
  return false;
}

function candidateCovers(
  candidateId: string,
  input: ExtractedMemory,
  output: ExtractedMemory,
  explicitlyCanonicalized: boolean,
): boolean {
  if (candidateId.startsWith("atomic:")) {
    return cockpitAtomicCandidateCovers(input, output, explicitlyCanonicalized);
  }
  // Source-coverage scaffolds intentionally use their own scene marker and
  // are verified by their governed slot/evidence bindings. Primary proposals,
  // unlike coverage scaffolds, must remain in their exact memory class.
  if (candidateId.startsWith("primary:")) {
    return cockpitCandidateFactuallyAnchors(input, output);
  }
  if (candidateId.startsWith("coverage:")) {
    // A coverage ID already encodes its governed domain and slot. Its hard
    // omission obligation remains structural, but a model-authored
    // canonicalization marker can never move it to another state.
    const inputMetadata = metadataOf(input);
    const outputMetadata = metadataOf(output);
    if (inputMetadata.coverage_requires_state_qualifier === true
      && !canonicalBindingValue("state_qualifier", outputMetadata.state_qualifier)) {
      return false;
    }
    const expectedQualifiers = stringArray(inputMetadata.coverage_required_state_qualifiers)
      .map((value) => canonicalBindingValue("state_qualifier", value))
      .filter((value): value is string => Boolean(value));
    if (expectedQualifiers.length > 0
      && !expectedQualifiers.includes(
        canonicalBindingValue("state_qualifier", outputMetadata.state_qualifier) ?? "",
      )) return false;
    return cockpitCandidateStructurallyCovers(input, output, false);
  }
  return false;
}

function comparableValue(value: unknown): string {
  if (value === undefined) return "<undefined>";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * Persisted facts that define whether a live cockpit state is unchanged.
 * Relation and supersedes are lineage, while source/session/time fields are
 * evidence provenance handled separately by `cockpitHasNewEvidence`.
 */
export const COCKPIT_PERSISTED_STATE_FACT_KEYS = [
  "domain",
  "slot",
  "state_key",
  "episode_key",
  "value",
  "target",
  "unit",
  "valid_from",
  "valid_to",
  "activity_start_time",
  "activity_end_time",
  "condition",
  "trigger",
  "constraint_target",
  "state_qualifier",
  "action_status",
  "record_kind",
  "subject",
  "occupant_scope",
  "vehicle_scope",
  "seat_zone",
  "timezone",
  "time_precision",
  "temporal_status",
] as const;

function isGenericMissingScopeFactualRepair(
  input: ExtractedMemory,
  output: ExtractedMemory,
): boolean {
  if (!qualityIssues(input).includes("missing_scope")) return false;
  const inputMetadata = metadataOf(input);
  const outputMetadata = metadataOf(output);
  if (["subject", "occupant_scope", "vehicle_scope", "seat_zone"].some((key) => {
    const inputBinding = canonicalBindingValue(key, inputMetadata[key]);
    const outputBinding = canonicalBindingValue(key, outputMetadata[key]);
    return inputBinding !== undefined
      || (key !== "subject" && outputBinding !== undefined)
      || (key === "subject" && outputBinding !== "user");
  })) return false;
  const inputState = stringValue(inputMetadata.state_key)?.split("|");
  const outputState = stringValue(outputMetadata.state_key)?.split("|");
  return Boolean(inputState && outputState
    && inputState.length === 5
    && outputState.length === 5
    && inputState[0] === outputState[0]
    && inputState[1] === "unspecified-subject"
    && outputState[1] === "user"
    && inputState.slice(2).every((part, index) => part === outputState[index + 2]));
}

function hasCompatibleCandidateProvenance(
  input: ExtractedMemory,
  output: ExtractedMemory,
): boolean {
  if (input.source_message_ids.length === 0) return false;
  const outputSources = new Set(output.source_message_ids);
  if (!input.source_message_ids.every((source) => outputSources.has(source))) return false;
  const inputMetadata = metadataOf(input);
  const outputMetadata = metadataOf(output);
  if (comparableValue(inputMetadata.mentioned_at)
      !== comparableValue(outputMetadata.mentioned_at)
    || comparableValue(inputMetadata.source_session_id)
      !== comparableValue(outputMetadata.source_session_id)) return false;
  const outputSessions = new Set(stringArray(outputMetadata.source_session_ids));
  return stringArray(inputMetadata.source_session_ids)
    .every((session) => outputSessions.has(session));
}

/**
 * A final cockpit row needs at least one factual proposal anchor. Structural
 * source/domain/slot overlap is intentionally insufficient: it cannot grant a
 * reconciler permission to invent value, episode, unit, time, condition, or
 * any other persisted state fact. Deterministic asserted-to-updated lineage
 * promotion and the narrowly checked generic missing-scope repair are the only
 * non-exact transformations admitted here.
 */
export function cockpitCandidateFactuallyAnchors(
  input: ExtractedMemory,
  output: ExtractedMemory,
  verifiedLivePriors: CockpitReconciliationPriorMemory[] = [],
): boolean {
  const inputStatus = qualityStatus(input);
  if ((inputStatus !== "complete" && inputStatus !== "partial")
    || input.type !== output.type
    || input.scene_name !== output.scene_name
    || !cockpitCandidateStructurallyCovers(input, output, false)
    || !hasCompatibleCandidateProvenance(input, output)) return false;
  const inputMetadata = metadataOf(input);
  const outputMetadata = metadataOf(output);
  if (inputMetadata.value === undefined && inputMetadata.target === undefined) return false;

  const genericScopeRepair = inputStatus === "partial"
    && isGenericMissingScopeFactualRepair(input, output);
  const inputRelation = stringValue(inputMetadata.relation);
  const outputSupersedes = stringArray(outputMetadata.supersedes);
  const verifiedUniquePredecessor = verifiedLivePriors.length === 1
    && outputSupersedes.length === 1
    && outputSupersedes[0] === verifiedLivePriors[0].record_id
    && stringValue(outputMetadata.episode_key)
      === stringValue(verifiedLivePriors[0].metadata.episode_key);
  const missingPredecessorRepair = inputStatus === "partial"
    && qualityIssues(input).includes("missing_supersedes")
    && (qualityRepairs(output).includes("repaired_missing_supersedes_from_unique_live_prior")
      || verifiedUniquePredecessor);
  if (!COCKPIT_PERSISTED_STATE_FACT_KEYS.every((key) => {
    if (genericScopeRepair && (key === "state_key" || key === "subject")) return true;
    if (missingPredecessorRepair && key === "episode_key") return true;
    return comparableValue(inputMetadata[key]) === comparableValue(outputMetadata[key]);
  })) return false;

  const outputRelation = stringValue(outputMetadata.relation);
  const exactRelation = inputRelation === outputRelation;
  const promotedAssertion = inputStatus === "complete"
    && inputRelation === "asserted"
    && outputRelation === "updated";
  if (!exactRelation && !promotedAssertion) return false;
  const inputSupersedes = stringArray(inputMetadata.supersedes);
  return promotedAssertion || missingPredecessorRepair
    ? inputSupersedes.length === 0 && outputSupersedes.length > 0
    : JSON.stringify([...inputSupersedes].sort())
      === JSON.stringify([...outputSupersedes].sort());
}

export function cockpitSameStructuredState(
  memory: ExtractedMemory,
  prior: CockpitReconciliationPriorMemory,
): boolean {
  const metadata = metadataOf(memory);
  return cockpitPriorMatchesMemoryClass(prior, memory)
    && COCKPIT_PERSISTED_STATE_FACT_KEYS.every((key) =>
      comparableValue(metadata[key]) === comparableValue(prior.metadata[key])
    );
}

export function cockpitHasNewEvidence(
  memory: ExtractedMemory,
  priors: CockpitReconciliationPriorMemory[],
): boolean {
  const priorSourceIds = new Set(priors.flatMap((prior) =>
    stringArray(prior.metadata.source_message_ids)
  ));
  const priorSessionIds = new Set(priors.flatMap((prior) =>
    [
      ...stringArray(prior.metadata.source_session_ids),
      stringValue(prior.metadata.source_session_id),
      stringValue(prior.session_id),
    ].filter((value): value is string => Boolean(value))
  ));
  const priorMentionedAt = new Set(priors
    .map((prior) => prior.metadata.mentioned_at)
    .filter((value) => value !== undefined)
    .map(comparableValue));
  const metadata = metadataOf(memory);
  const sourceIds = [...new Set([
    ...memory.source_message_ids,
    ...stringArray(metadata.source_message_ids),
  ])];
  const sessionIds = [
    ...stringArray(metadata.source_session_ids),
    stringValue(metadata.source_session_id),
  ].filter((value): value is string => Boolean(value));
  const mentionedAt = metadata.mentioned_at;
  return sourceIds.some((id) => !priorSourceIds.has(id))
    || sessionIds.some((id) => !priorSessionIds.has(id))
    || (mentionedAt !== undefined && !priorMentionedAt.has(comparableValue(mentionedAt)));
}

function diagnosticLivePriors(
  memory: ExtractedMemory,
  exactMatches: CockpitReconciliationPriorMemory[],
  allLivePriors: CockpitReconciliationPriorMemory[],
): CockpitReconciliationPriorMemory[] {
  const referencedIds = new Set(stringArray(metadataOf(memory).supersedes));
  const referenced = allLivePriors.filter((prior) => referencedIds.has(prior.record_id));
  return [...new Map([...exactMatches, ...referenced].map((prior) => [prior.record_id, prior])).values()];
}

function compactPriorTargets(priors: CockpitReconciliationPriorMemory[]): CockpitReconciliationDiagnostic["livePriorTargets"] {
  return priors.map((prior) => ({
    recordId: prior.record_id,
    domain: stringValue(prior.metadata.domain),
    slot: stringValue(prior.metadata.slot),
    stateKey: stringValue(prior.metadata.state_key),
    episodeKey: stringValue(prior.metadata.episode_key),
  }));
}

/**
 * Verify reconciliation coverage after every returned memory has already
 * passed `normalizeCockpitExtractedMemory`.
 */
export function gateCockpitConstructionReconciliation(params: {
  inputs: CockpitReconciliationInput[];
  reconciled: ExtractedMemory[];
  maxMemories: number;
  priorMemories?: CockpitReconciliationPriorMemory[];
  /** Candidate obligations deterministically consumed by assembly or exact live-state no-ops. */
  resolvedCandidateIds?: string[];
}): CockpitReconciliationGateResult {
  const issues: string[] = [];
  const inputById = new Map(params.inputs.map((input) => [input.id, input]));
  const requiredCandidateIds = params.inputs
    .filter((input) => input.id.startsWith("coverage:")
      || (input.id.startsWith("atomic:")
        && ["complete", "partial"].includes(qualityStatus(input.memory) ?? "")))
    .map((input) => input.id);
  const structurallyCovered = new Set<string>();
  for (const resolvedId of params.resolvedCandidateIds ?? []) {
    if (inputById.has(resolvedId)) structurallyCovered.add(resolvedId);
    else issues.push("reconciliation_unknown_resolved_candidate_id");
  }
  const identities = new Set<string>();
  const livePriors = cockpitLivePriorMemories(params.priorMemories ?? []);
  const diagnostics: CockpitReconciliationDiagnostic[] = [];

  if (params.reconciled.length > params.maxMemories) issues.push("reconciliation_exceeds_max_memories");
  for (const [rowIndex, memory] of params.reconciled.entries()) {
    if (qualityStatus(memory) !== "complete") {
      issues.push("reconciliation_contains_incomplete_memory");
      const targets = diagnosticLivePriors(memory, matchingLivePrior(memory, livePriors), livePriors);
      const quality = metadataOf(memory).construction_quality;
      diagnostics.push({
        rowIndex,
        issue: "reconciliation_contains_incomplete_memory",
        stateKey: stringValue(metadataOf(memory).state_key),
        episodeKey: stringValue(metadataOf(memory).episode_key),
        relation: stringValue(metadataOf(memory).relation),
        qualityIssues: quality && typeof quality === "object" && !Array.isArray(quality)
          ? stringArray((quality as Record<string, unknown>).issues)
          : [],
        matchingLivePriorRecordIds: targets.map((prior) => prior.record_id),
        livePriorTargets: compactPriorTargets(targets),
      });
    }

    const identity = cockpitFinalIdentity(memory);
    if (!identity) issues.push("reconciliation_missing_final_identity");
    else if (identities.has(identity)) issues.push("reconciliation_duplicate_final_identity");
    else identities.add(identity);

    const metadata = metadataOf(memory);
    const relation = stringValue(metadata.relation);
    const matchingPriors = matchingLivePrior(memory, livePriors);
    if (!hasValidControlledOntology(memory)) {
      issues.push("reconciliation_invalid_controlled_ontology");
      diagnostics.push({
        rowIndex,
        issue: "reconciliation_invalid_controlled_ontology",
        stateKey: stringValue(metadata.state_key),
        episodeKey: stringValue(metadata.episode_key),
        relation,
        qualityIssues: [],
        matchingLivePriorRecordIds: [],
        livePriorTargets: [],
      });
    }
    if (relation === "asserted" && matchingPriors.length > 0) {
      const sameState = matchingPriors.every((prior) =>
        cockpitSameStructuredState(memory, prior)
      );
      // A repeated state backed by a new source/session is evidence enrichment,
      // not a duplicate write. Keep it for TencentDB's original dedup merger so
      // aggregation and event lineage retain the new observation.
      if (!sameState || !cockpitHasNewEvidence(memory, matchingPriors)) {
        const issue = sameState
          ? "reconciliation_reasserts_unchanged_live_prior"
          : "reconciliation_asserts_over_existing_live_prior";
        issues.push(issue);
        diagnostics.push({
          rowIndex,
          issue,
          stateKey: stringValue(metadata.state_key),
          episodeKey: stringValue(metadata.episode_key),
          relation,
          qualityIssues: [],
          matchingLivePriorRecordIds: matchingPriors.map((prior) => prior.record_id),
          livePriorTargets: compactPriorTargets(matchingPriors),
        });
      }
    }
    if (relation === "updated" || relation === "cancelled" || relation === "negated") {
      const supersedes = new Set(stringArray(metadata.supersedes));
      if (matchingPriors.length > 0
        && !matchingPriors.every((prior) => supersedes.has(prior.record_id))) {
        issues.push("reconciliation_transition_misses_live_prior");
        diagnostics.push({
          rowIndex,
          issue: "reconciliation_transition_misses_live_prior",
          stateKey: stringValue(metadata.state_key),
          episodeKey: stringValue(metadata.episode_key),
          relation,
          qualityIssues: [],
          matchingLivePriorRecordIds: matchingPriors.map((prior) => prior.record_id),
          livePriorTargets: compactPriorTargets(matchingPriors),
        });
      }
      const allLiveById = new Map(livePriors.map((prior) => [prior.record_id, prior]));
      if ([...supersedes].some((reference) => !allLiveById.has(reference))) {
        issues.push("reconciliation_transition_supersedes_non_live_prior");
        diagnostics.push({
          rowIndex,
          issue: "reconciliation_transition_supersedes_non_live_prior",
          stateKey: stringValue(metadata.state_key),
          episodeKey: stringValue(metadata.episode_key),
          relation,
          qualityIssues: [],
          matchingLivePriorRecordIds: matchingPriors.map((prior) => prior.record_id),
          livePriorTargets: compactPriorTargets(matchingPriors),
        });
      }
      const matchingIds = new Set(matchingPriors.map((prior) => prior.record_id));
      if ([...supersedes].some((reference) =>
        allLiveById.has(reference) && !matchingIds.has(reference)
      )) {
        issues.push("reconciliation_transition_supersedes_non_matching_live_prior");
        diagnostics.push({
          rowIndex,
          issue: "reconciliation_transition_supersedes_non_matching_live_prior",
          stateKey: stringValue(metadata.state_key),
          episodeKey: stringValue(metadata.episode_key),
          relation,
          qualityIssues: [],
          matchingLivePriorRecordIds: matchingPriors.map((prior) => prior.record_id),
          livePriorTargets: compactPriorTargets(diagnosticLivePriors(
            memory,
            matchingPriors,
            livePriors,
          )),
        });
      }
    }

    const references = stringArray(metadata.input_candidate_ids);
    const canonicalizedReferences = new Set(stringArray(metadata.canonicalized_input_candidate_ids));
    if (references.length === 0) issues.push("reconciliation_missing_input_candidate_ids");
    for (const reference of canonicalizedReferences) {
      if (!references.includes(reference)) issues.push("reconciliation_invalid_canonicalized_candidate_id");
    }
    for (const reference of references) {
      const input = inputById.get(reference);
      if (!input) {
        issues.push("reconciliation_unknown_input_candidate_id");
        continue;
      }
      const explicitlyCanonicalized = canonicalizedReferences.has(reference);
      if (candidateCovers(reference, input.memory, memory, explicitlyCanonicalized)) {
        structurallyCovered.add(reference);
      }
    }
    const hasFactualCandidateAnchor = references.some((reference) => {
      const input = inputById.get(reference);
      return Boolean(input && cockpitCandidateFactuallyAnchors(
        input.memory,
        memory,
        matchingPriors,
      ));
    });
    if (!hasFactualCandidateAnchor
      && !hasCockpitAssemblerVerifiedFactualRewrite(memory)) {
      issues.push("reconciliation_row_without_factual_candidate_anchor");
      diagnostics.push({
        rowIndex,
        issue: "reconciliation_row_without_factual_candidate_anchor",
        stateKey: stringValue(metadata.state_key),
        episodeKey: stringValue(metadata.episode_key),
        relation,
        qualityIssues: qualityIssues(memory),
        matchingLivePriorRecordIds: matchingPriors.map((prior) => prior.record_id),
        livePriorTargets: compactPriorTargets(matchingPriors),
      });
    }
  }

  // A one-group source obligation cannot authorize two otherwise identical
  // state/event envelopes merely because independent model paths chose
  // different raw values or episode labels. Preserve both candidates in the
  // repair feedback and fail closed; never collapse by state_key globally,
  // because genuinely distinct multi-event instructions may share it.
  for (const input of params.inputs.filter(isSingleEventCoverageObligation)) {
    const governed = params.reconciled
      .map((memory, rowIndex) => ({ memory, rowIndex }))
      .filter(({ memory }) => {
        const references = stringArray(metadataOf(memory).input_candidate_ids);
        return references.includes(input.id)
          && candidateCovers(input.id, input.memory, memory, false);
      });
    const conflictingRows = new Set<number>();
    for (let leftIndex = 0; leftIndex < governed.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < governed.length; rightIndex += 1) {
        if (!conflictingSingleEventCoverageAlias(
          governed[leftIndex].memory,
          governed[rightIndex].memory,
        )) continue;
        conflictingRows.add(governed[leftIndex].rowIndex);
        conflictingRows.add(governed[rightIndex].rowIndex);
      }
    }
    if (conflictingRows.size === 0) continue;
    issues.push("reconciliation_conflicting_single_event_coverage_alias");
    for (const rowIndex of conflictingRows) {
      const memory = params.reconciled[rowIndex];
      diagnostics.push({
        rowIndex,
        issue: "reconciliation_conflicting_single_event_coverage_alias",
        stateKey: stringValue(metadataOf(memory).state_key),
        episodeKey: stringValue(metadataOf(memory).episode_key),
        relation: stringValue(metadataOf(memory).relation),
        qualityIssues: qualityIssues(memory),
        matchingLivePriorRecordIds: [],
        livePriorTargets: [],
      });
    }
  }

  const uncoveredCandidateIds: string[] = [];
  for (const requiredId of requiredCandidateIds) {
    if (!structurallyCovered.has(requiredId)) {
      uncoveredCandidateIds.push(requiredId);
      issues.push(requiredId.startsWith("coverage:")
        ? "reconciliation_uncovered_source_coverage_obligation"
        : "reconciliation_uncovered_atomic_candidate");
    }
  }
  // An empty result is valid only when every atomic obligation was consumed
  // as an exact live-state no-op. Primary-only proposals remain advisory, but
  // an otherwise empty reconciliation must not silently discard all input.
  if (params.reconciled.length === 0
    && (uncoveredCandidateIds.length > 0
      || (requiredCandidateIds.length === 0
        && params.inputs.length > 0
        && structurallyCovered.size === 0))) {
    issues.push("reconciliation_empty_with_input_candidates");
  }

  const uniqueIssues = [...new Set(issues)];
  return {
    accepted: uniqueIssues.length === 0,
    issues: uniqueIssues,
    requiredCandidateIds,
    coveredCandidateIds: [...structurallyCovered],
    uncoveredCandidateIds,
    diagnostics,
  };
}
