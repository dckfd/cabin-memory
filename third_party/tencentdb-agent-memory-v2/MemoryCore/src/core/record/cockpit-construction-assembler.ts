/**
 * Deterministic final assembly for model-reconciled cockpit memories.
 *
 * Flash remains responsible for semantic extraction and reconciliation. This
 * layer only enforces relations already knowable from structured candidates,
 * exact source bindings, timestamps, and persisted live identities. It never
 * creates a value, person, place, date, or scope from free text.
 */

import type { ExtractedMemory } from "./l1-writer.js";
import {
  cockpitAtomicCandidateCovers,
  cockpitCandidateStructurallyCovers,
  cockpitFinalIdentity,
  cockpitHasNewEvidence,
  cockpitLivePriorMemories,
  cockpitPriorMatchesMemoryState,
  cockpitPriorMatchesTransitionIdentity,
  cockpitSameStructuredState,
  COCKPIT_PERSISTED_STATE_FACT_KEYS,
  registerCockpitAssemblerVerifiedFactualRewrite,
  type CockpitReconciliationInput,
  type CockpitReconciliationPriorMemory,
} from "./cockpit-construction-reconciliation.js";
import {
  canonicalCockpitSceneClass,
  controlledCockpitSlotOwners,
} from "./cockpit-ontology.js";
import {
  isControlledCockpitEventTimeSlot,
  strictZonedIsoInstant,
} from "./cockpit-temporal.js";

export interface CockpitConstructionAssemblyResult {
  memories: ExtractedMemory[];
  /** Candidate obligations deterministically consumed by coalescence or exact live-state no-ops. */
  resolvedCandidateIds: string[];
  repairCounts: Record<string, number>;
}

const VERIFIED_FACTUAL_REWRITE = Symbol("cockpit-assembler-verified-factual-rewrite");
type AssemblerVerifiedMemory = ExtractedMemory & {
  [VERIFIED_FACTUAL_REWRITE]?: true;
};

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

function hasBlockingQualityIssue(memory: ExtractedMemory): boolean {
  return qualityIssues(memory).includes("ambiguous_semantic_binding_alias");
}

function candidateCoversOutput(
  candidate: CockpitReconciliationInput,
  output: ExtractedMemory,
  explicitlyCanonicalized = false,
  inputById?: Map<string, CockpitReconciliationInput>,
): boolean {
  if (candidate.id.startsWith("primary:")) {
    // Primary proposals are advisory semantic drafts. They may anchor a row
    // only when the complete persisted fact and event provenance are exact;
    // a singleton structural match is not permission to rewrite value,
    // episode, time, unit, condition, or any other state fact.
    return hasFactualPrimaryAssemblyEntrySupport(candidate, output);
  }
  if (candidate.id.startsWith("coverage:")) {
    // Coverage IDs already name their governed domain/slot. Never trust a
    // model-authored canonicalization flag to move that obligation elsewhere.
    return cockpitCandidateStructurallyCovers(
      candidate.memory,
      output,
      false,
    );
  }
  if (!candidate.id.startsWith("atomic:")) return false;
  if (qualityStatus(candidate.memory) !== "complete") {
    return partialAtomicCandidateCoversOutput(
      candidate.memory,
      output,
      explicitlyCanonicalized,
    );
  }
  return completeAtomicCandidateCoversOutput(
    candidate.memory,
    output,
    explicitlyCanonicalized,
    inputById,
  );
}

function referencedCandidateCoversOutput(
  reference: string,
  output: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const candidate = inputById.get(reference);
  if (!candidate) return false;
  const canonicalized = stringArray(metadataOf(output).canonicalized_input_candidate_ids)
    .includes(reference);
  return candidateCoversOutput(candidate, output, canonicalized, inputById);
}

function cloneMemory(memory: ExtractedMemory): ExtractedMemory {
  const metadata = metadataOf(memory);
  const quality = metadata.construction_quality;
  return {
    ...memory,
    source_message_ids: [...memory.source_message_ids],
    metadata: {
      ...metadata,
      ...(Array.isArray(metadata.source_message_ids)
        ? { source_message_ids: [...metadata.source_message_ids] }
        : {}),
      ...(Array.isArray(metadata.source_session_ids)
        ? { source_session_ids: [...metadata.source_session_ids] }
        : {}),
      ...(Array.isArray(metadata.evidence_roles)
        ? { evidence_roles: [...metadata.evidence_roles] }
        : {}),
      ...(Array.isArray(metadata.supersedes)
        ? { supersedes: [...metadata.supersedes] }
        : {}),
      ...(Array.isArray(metadata.input_candidate_ids)
        ? { input_candidate_ids: [...metadata.input_candidate_ids] }
        : {}),
      ...(Array.isArray(metadata.canonicalized_input_candidate_ids)
        ? { canonicalized_input_candidate_ids: [...metadata.canonicalized_input_candidate_ids] }
        : {}),
      ...(quality && typeof quality === "object" && !Array.isArray(quality)
        ? {
          construction_quality: {
            ...(quality as Record<string, unknown>),
            issues: stringArray((quality as Record<string, unknown>).issues),
            repairs: stringArray((quality as Record<string, unknown>).repairs),
          },
        }
        : {}),
    },
  };
}

function sameStrings(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameStringSet(left: string[], right: string[]): boolean {
  return sameStrings([...new Set(left)].sort(), [...new Set(right)].sort());
}

function bump(repairs: Map<string, number>, repair: string): void {
  repairs.set(repair, (repairs.get(repair) ?? 0) + 1);
}

function addQualityRepair(memory: ExtractedMemory, repair: string): ExtractedMemory {
  const cloned = cloneMemory(memory);
  const metadata = metadataOf(cloned);
  const rawQuality = metadata.construction_quality;
  if (!rawQuality || typeof rawQuality !== "object" || Array.isArray(rawQuality)) return cloned;
  const quality = rawQuality as Record<string, unknown>;
  metadata.construction_quality = {
    ...quality,
    repairs: [...new Set([...stringArray(quality.repairs), repair])],
    source_count: cloned.source_message_ids.length,
  };
  return cloned;
}

function addVerifiedFactualRewriteRepair(
  memory: ExtractedMemory,
  repair: string,
): ExtractedMemory {
  const repaired = addQualityRepair(memory, repair) as AssemblerVerifiedMemory;
  repaired[VERIFIED_FACTUAL_REWRITE] = true;
  return repaired;
}

function addQualityIssue(
  memory: ExtractedMemory,
  issue: string,
  repair: string,
): { memory: ExtractedMemory; added: boolean } {
  const cloned = cloneMemory(memory);
  const metadata = metadataOf(cloned);
  const rawQuality = metadata.construction_quality;
  if (!rawQuality || typeof rawQuality !== "object" || Array.isArray(rawQuality)) {
    return { memory: cloned, added: false };
  }
  const quality = rawQuality as Record<string, unknown>;
  const issues = stringArray(quality.issues);
  const repairs = stringArray(quality.repairs);
  const added = !issues.includes(issue);
  metadata.construction_quality = {
    ...quality,
    status: "partial",
    score: Math.min(typeof quality.score === "number" ? quality.score : 100, 90),
    issues: [...new Set([...issues, issue])],
    repairs: [...new Set([...repairs, repair])],
    source_count: cloned.source_message_ids.length,
  };
  return { memory: cloned, added };
}

function bindCandidateIds(
  memory: ExtractedMemory,
  inputs: CockpitReconciliationInput[],
  inputById: Map<string, CockpitReconciliationInput>,
  repairs: Map<string, number>,
): ExtractedMemory {
  let output = cloneMemory(memory);
  const metadata = metadataOf(output);
  const explicit = stringArray(metadata.input_candidate_ids);
  const canonicalized = new Set(stringArray(metadata.canonicalized_input_candidate_ids));
  const coverageHasIndependentAtomicSupport = (
    coverage: CockpitReconciliationInput,
  ): boolean => inputs.some((input) =>
    input.id.startsWith("atomic:")
      && (qualityStatus(input.memory) === "complete"
        || qualityStatus(input.memory) === "partial")
      && sameStringSet(input.memory.source_message_ids, coverage.memory.source_message_ids)
      && candidateCoversOutput(coverage, input.memory, false, inputById)
      && candidateCoversOutput(input, output, false, inputById)
  );
  const retainedExplicit = explicit.filter((id) => {
    const input = inputById.get(id);
    if (!input || !candidateCoversOutput(
      input,
      output,
      canonicalized.has(id),
      inputById,
    )) return false;
    return !id.startsWith("coverage:") || coverageHasIndependentAtomicSupport(input);
  });
  const structurallyMatching = inputs
    .filter((input) => candidateCoversOutput(input, output, false, inputById));
  // Coverage scaffolds carry no factual value. Bind them independently from
  // proposal value ranking, and only when a single-source atomic candidate
  // both discharges that exact obligation and factually supports this output.
  // Partial atomics enter only through the assembler's narrow deterministic
  // repair predicates above. This prevents either a pure coverage row or a
  // cross-event aggregate from self-authorizing while preserving an
  // independently sourced fact.
  const supportedCoverage = structurallyMatching.filter((input) =>
    input.id.startsWith("coverage:") && coverageHasIndependentAtomicSupport(input)
  );
  const proposalMatching = structurallyMatching.filter((input) =>
    !input.id.startsWith("coverage:")
  );
  const outputIdentity = cockpitFinalIdentity(output);
  const identityMatching = outputIdentity
    ? proposalMatching.filter((input) => cockpitFinalIdentity(input.memory) === outputIdentity)
    : [];
  const inferencePool = identityMatching.length > 0 ? identityMatching : proposalMatching;
  const outputValue = comparableValue(metadata.value);
  const exactValueMatching = inferencePool.filter((input) =>
    comparableValue(metadataOf(input.memory).value) === outputValue
  );
  let inferredInputs: CockpitReconciliationInput[];
  if (exactValueMatching.length > 0) {
    inferredInputs = exactValueMatching;
  } else if (inferencePool.length === 1) {
    // A single structurally possible source can safely survive semantic
    // canonicalization by the reconciler.
    inferredInputs = inferencePool;
  } else {
    // Never guess across several distinct atomic obligations merely because
    // they share one source, domain, and slot. That would let one conditional
    // value consume a sibling value. Equivalent candidates may still co-bind.
    const signatures = new Set(inferencePool.map((input) => JSON.stringify({
      identity: cockpitFinalIdentity(input.memory),
      value: metadataOf(input.memory).value,
      unit: metadataOf(input.memory).unit,
      action_status: metadataOf(input.memory).action_status,
    })));
    inferredInputs = signatures.size === 1 ? inferencePool : [];
  }
  const inferred = inferredInputs.map((input) => input.id);
  const bound = [...new Set([
    ...retainedExplicit,
    ...inferred,
    ...supportedCoverage.map((input) => input.id),
  ])];
  if (!sameStrings(explicit, bound)) {
    output = addQualityRepair(output, "deterministically_bound_input_candidates");
    bump(repairs, "deterministically_bound_input_candidates");
  }
  const outputMetadata = metadataOf(output);
  outputMetadata.input_candidate_ids = bound;
  const retainedCanonicalized = stringArray(outputMetadata.canonicalized_input_candidate_ids)
    .filter((id) => bound.includes(id));
  if (retainedCanonicalized.length > 0) {
    outputMetadata.canonicalized_input_candidate_ids = retainedCanonicalized;
  } else {
    delete outputMetadata.canonicalized_input_candidate_ids;
  }
  return output;
}

function comparableValue(value: unknown): string {
  if (value === undefined) return "<undefined>";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

const REPRESENTATION_ONLY_SLOT_QUALIFIERS = new Set([
  "default",
  "preferred",
  "saved",
  "temporary",
  "usual",
]);

function isQualifiedFormOfControlledSlot(outputSlot: string, controlledSlot: string): boolean {
  const suffix = `_${controlledSlot}`;
  if (!outputSlot.endsWith(suffix)) return false;
  const qualifier = outputSlot.slice(0, -suffix.length);
  return qualifier.length > 0
    && qualifier.split("_").every((token) => REPRESENTATION_ONLY_SLOT_QUALIFIERS.has(token));
}

function hasExactCanonicalFactPayload(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  return left.type === right.type
    && [
      "value",
      "target",
      "unit",
      "valid_from",
      "valid_to",
      "condition",
      "trigger",
      "constraint_target",
      "state_qualifier",
      "action_status",
      "record_kind",
    ].every((key) => comparableValue(leftMetadata[key]) === comparableValue(rightMetadata[key]));
}

function isRecoverableCompositeTransitionMemory(memory: ExtractedMemory): boolean {
  const metadata = metadataOf(memory);
  const relation = stringValue(metadata.relation);
  return qualityStatus(memory) === "partial"
    && stringValue(metadata.domain) === "schedule"
    && stringValue(metadata.slot) === "status"
    && (relation === "updated" || relation === "cancelled" || relation === "negated");
}

function isRecoverableCompositeTransitionCandidate(
  candidate: CockpitReconciliationInput,
): boolean {
  return isRecoverableCompositeTransitionMemory(candidate.memory);
}

/**
 * Validate a reconciler's explicit claim that a model-composed slot is the
 * canonical form of an atomic obligation.
 *
 * A canonicalization is accepted only when one complete atomic candidate has
 * the same exact evidence, bindings, timestamp, relation, and factual payload;
 * its slot is controlled for the same domain; and the reconciled slot differs
 * only by representation-only qualifiers.  Conflicting controlled slots or
 * semantic qualifiers (for example `avoid_destination`) are never collapsed.
 * An unverified claim is dropped so the required atomic obligation is restored
 * later, rather than trusting model-authored `canonicalized_input_candidate_ids`.
 */
function canonicalizeReconciledSlotFromAtomicEvidence(
  memory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
  repairs: Map<string, number>,
): ExtractedMemory | undefined {
  const metadata = metadataOf(memory);
  const references = new Set(stringArray(metadata.input_candidate_ids));
  const canonicalized = stringArray(metadata.canonicalized_input_candidate_ids);
  const outputDomain = stringValue(metadata.domain);
  const outputSlot = stringValue(metadata.slot);
  if (!outputDomain || !outputSlot) return memory;

  const mismatchedAtomicClaims = canonicalized
    .filter((id) => id.startsWith("atomic:") && references.has(id))
    .map((id) => inputById.get(id))
    .filter((candidate): candidate is CockpitReconciliationInput => Boolean(candidate))
    // Only an explicit schedule.status transition may intentionally span
    // several concrete appointment edges. A partial candidate for a concrete
    // slot remains slot-exact: otherwise one cancelled appointment_time could
    // be claimed by appointment_content and silently disappear from a pure
    // multi-slot cancellation transaction.
    .filter((candidate) => !isRecoverableCompositeTransitionCandidate(candidate))
    .filter((candidate) => {
      const candidateMetadata = metadataOf(candidate.memory);
      return stringValue(candidateMetadata.domain) !== outputDomain
        || stringValue(candidateMetadata.slot) !== outputSlot;
    });
  if (mismatchedAtomicClaims.length === 0) return memory;

  const supported = mismatchedAtomicClaims.filter((candidate) => {
    const candidateMetadata = metadataOf(candidate.memory);
    const candidateDomain = stringValue(candidateMetadata.domain);
    const candidateSlot = stringValue(candidateMetadata.slot);
    const candidateStateKey = stringValue(candidateMetadata.state_key);
    const owners = controlledCockpitSlotOwners(candidateSlot);
    return qualityStatus(candidate.memory) === "complete"
      && Boolean(candidateDomain && candidateSlot && candidateStateKey)
      && candidateDomain === outputDomain
      && Boolean(owners?.includes(candidateDomain))
      && controlledCockpitSlotOwners(outputSlot) === undefined
      && isQualifiedFormOfControlledSlot(outputSlot, candidateSlot ?? "")
      && sameTransitionEvidence(candidate.memory, memory)
      && hasExactCanonicalFactPayload(candidate.memory, memory)
      && cockpitAtomicCandidateCovers(candidate.memory, memory, true);
  });
  const targetIdentities = new Set(supported.map((candidate) => {
    const candidateMetadata = metadataOf(candidate.memory);
    return JSON.stringify({
      domain: candidateMetadata.domain,
      slot: candidateMetadata.slot,
      state_key: candidateMetadata.state_key,
    });
  }));
  if (supported.length !== mismatchedAtomicClaims.length || targetIdentities.size !== 1) {
    bump(repairs, "dropped_unverified_atomic_slot_canonicalization");
    return undefined;
  }

  let output = cloneMemory(memory);
  const outputMetadata = metadataOf(output);
  const targetMetadata = metadataOf(supported[0].memory);
  outputMetadata.domain = targetMetadata.domain;
  outputMetadata.slot = targetMetadata.slot;
  outputMetadata.state_key = targetMetadata.state_key;
  const supportedIds = new Set(supported.map((candidate) => candidate.id));
  const retainedCanonicalized = stringArray(outputMetadata.canonicalized_input_candidate_ids)
    .filter((id) => !supportedIds.has(id));
  if (retainedCanonicalized.length > 0) {
    outputMetadata.canonicalized_input_candidate_ids = retainedCanonicalized;
  } else {
    delete outputMetadata.canonicalized_input_candidate_ids;
  }
  output = addQualityRepair(
    output,
    "canonicalized_reconciled_slot_from_unique_atomic_evidence",
  );
  bump(repairs, "canonicalized_reconciled_slot_from_unique_atomic_evidence");
  return output;
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

function uniqueLivePriorForMissingSupersedes(
  memory: ExtractedMemory,
  priors: CockpitReconciliationPriorMemory[],
): CockpitReconciliationPriorMemory | undefined {
  const metadata = metadataOf(memory);
  const stateKey = stringValue(metadata.state_key);
  if (!stateKey) return undefined;
  const stateMatches = priors.filter((prior) =>
    cockpitPriorMatchesMemoryState(prior, memory)
      && stringValue(prior.metadata.state_key) === stateKey
  );
  const episodeKey = stringValue(metadata.episode_key);
  const exactEpisodeMatches = episodeKey
    ? stateMatches.filter((prior) => stringValue(prior.metadata.episode_key) === episodeKey)
    : [];
  if (exactEpisodeMatches.length === 1) return exactEpisodeMatches[0];
  if (exactEpisodeMatches.length > 1) return undefined;
  return stateMatches.length === 1 ? stateMatches[0] : undefined;
}

const PARTIAL_TRANSITION_PRIOR_FACT_KEYS = [
  "domain",
  "slot",
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
  "record_kind",
  "subject",
  "occupant_scope",
  "vehicle_scope",
  "seat_zone",
  "timezone",
  "time_precision",
  "temporal_status",
] as const;

function partialTransitionFactsMatchPrior(
  memory: ExtractedMemory,
  prior: CockpitReconciliationPriorMemory,
): boolean {
  const metadata = metadataOf(memory);
  const relation = stringValue(metadata.relation);
  const lifecycleStatus = stringValue(metadata.domain) === "schedule"
    && stringValue(metadata.slot) === "status";
  return PARTIAL_TRANSITION_PRIOR_FACT_KEYS.every((key) => {
    // Updates are expected to change value. A schedule lifecycle cancellation
    // similarly changes status to `cancelled`; every other supplied fact must
    // remain an exact predecessor fact.
    if ((relation === "updated" || lifecycleStatus) && (key === "value" || key === "target")) {
      return true;
    }
    return metadata[key] === undefined
      || comparableValue(metadata[key]) === comparableValue(prior.metadata[key]);
  });
}

function restorePartialAtomicTransition(
  candidate: CockpitReconciliationInput,
  priors: CockpitReconciliationPriorMemory[],
  repairs: Map<string, number>,
): ExtractedMemory | undefined {
  const memory = candidate.memory;
  const metadata = metadataOf(memory);
  const relation = stringValue(metadata.relation);
  if (qualityStatus(memory) !== "partial"
    || !sameStrings(qualityIssues(memory), ["missing_supersedes"])
    || (relation !== "updated" && relation !== "cancelled" && relation !== "negated")
    || stringArray(metadata.supersedes).length > 0) {
    return undefined;
  }
  const prior = uniqueLivePriorForMissingSupersedes(memory, priors);
  const priorEpisodeKey = prior && stringValue(prior.metadata.episode_key);
  if (!prior || !priorEpisodeKey || !partialTransitionFactsMatchPrior(memory, prior)) return undefined;

  let restored = cloneMemory(memory);
  const restoredMetadata = metadataOf(restored);
  restoredMetadata.episode_key = priorEpisodeKey;
  restoredMetadata.supersedes = [prior.record_id];
  restoredMetadata.input_candidate_ids = [candidate.id];
  restored = addQualityRepair(restored, "repaired_missing_supersedes_from_unique_live_prior");
  const restoredQuality = metadataOf(restored).construction_quality as Record<string, unknown>;
  metadataOf(restored).construction_quality = {
    ...restoredQuality,
    status: "complete",
    score: 100,
    issues: [],
  };
  bump(repairs, "repaired_missing_supersedes_from_unique_live_prior");
  return restored;
}

function resolveAgainstLivePrior(
  memory: ExtractedMemory,
  priors: CockpitReconciliationPriorMemory[],
  inputById: Map<string, CockpitReconciliationInput>,
  repairs: Map<string, number>,
): { memory?: ExtractedMemory; resolvedCandidateIds: string[] } {
  let output = cloneMemory(memory);
  const matching = matchingLivePrior(output, priors);
  if (matching.length === 0) return { memory: output, resolvedCandidateIds: [] };
  const metadata = metadataOf(output);
  const relation = stringValue(metadata.relation);
  const inputCandidateIds = stringArray(metadata.input_candidate_ids);
  const priorStateSignatures = new Set(matching.map((prior) => JSON.stringify({
    type: stringValue(prior.type ?? prior.metadata.type),
    scene_name: canonicalCockpitSceneClass(
      prior.scene_name ?? prior.metadata.scene_name,
      prior.metadata.domain,
      prior.metadata.slot,
    ),
    facts: COCKPIT_PERSISTED_STATE_FACT_KEYS.map((key) => comparableValue(prior.metadata[key])),
  })));
  const liveStatesAreEquivalent = priorStateSignatures.size === 1;

  if (relation === "asserted"
    && matching.every((prior) => cockpitSameStructuredState(output, prior))) {
    if (cockpitHasNewEvidence(output, matching)) {
      output = addQualityRepair(output, "retained_new_evidence_for_live_state");
      bump(repairs, "retained_new_evidence_for_live_state");
      return { memory: output, resolvedCandidateIds: [] };
    }
    bump(repairs, "suppressed_exact_source_replay");
    return {
      resolvedCandidateIds: inputCandidateIds.filter((id) =>
        referencedCandidateCoversOutput(id, output, inputById)
      ),
    };
  }

  const liveIds = matching.map((prior) => prior.record_id);
  if (relation === "asserted") {
    // One final identity may temporarily have several live rows because the
    // original store retains evidence observations. Promote across all only
    // when those rows describe the same persisted state; conflicting live
    // values are an ambiguity for the public gate, not permission to end all.
    if (!liveStatesAreEquivalent) return { memory: output, resolvedCandidateIds: [] };
    output = addQualityRepair(output, "promoted_assertion_over_live_state_to_update");
    const updatedMetadata = metadataOf(output);
    updatedMetadata.relation = "updated";
    updatedMetadata.supersedes = liveIds;
    bump(repairs, "promoted_assertion_over_live_state_to_update");
  } else if (relation === "updated" || relation === "cancelled" || relation === "negated") {
    const current = stringArray(metadata.supersedes);
    const referencedLiveIds = matching
      .filter((prior) => current.some((reference) =>
        reference === prior.record_id
          || reference === stringValue(prior.metadata.episode_key)
      ))
      .map((prior) => prior.record_id);
    const desired = current.length > 0 && referencedLiveIds.length > 0
      ? referencedLiveIds
      : matching.length === 1 || liveStatesAreEquivalent
        ? liveIds
        : [];
    if (desired.length > 0 && !sameStrings(current, desired)) {
      output = addQualityRepair(output, "bound_transition_to_exact_live_predecessors");
      metadataOf(output).supersedes = desired;
      bump(repairs, "bound_transition_to_exact_live_predecessors");
    }
  }
  return { memory: output, resolvedCandidateIds: [] };
}

function mentionedAtMs(memory: ExtractedMemory): number {
  const raw = stringValue(metadataOf(memory).mentioned_at);
  const value = raw ? new Date(raw).getTime() : Number.NaN;
  return Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
}

function atomicReferenceCount(memory: ExtractedMemory): number {
  return stringArray(metadataOf(memory).input_candidate_ids)
    .filter((id) => id.startsWith("atomic:") || id.startsWith("coverage:")).length;
}

function exactAtomicReferenceFidelity(
  memory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): number {
  const metadata = metadataOf(memory);
  return stringArray(metadata.input_candidate_ids).filter((id) => {
    if (!id.startsWith("atomic:")) return false;
    const input = inputById.get(id);
    if (!input || !referencedCandidateCoversOutput(id, memory, inputById)) return false;
    const inputMetadata = metadataOf(input.memory);
    return stringValue(inputMetadata.relation) === stringValue(metadata.relation)
      && stringValue(inputMetadata.domain) === stringValue(metadata.domain)
      && stringValue(inputMetadata.slot) === stringValue(metadata.slot)
      && comparableValue(inputMetadata.value) === comparableValue(metadata.value);
  }).length;
}

function collapseCompositeTransitionShadowGroup(
  group: Array<{ memory: ExtractedMemory; index: number }>,
  inputById: Map<string, CockpitReconciliationInput>,
  repairs: Map<string, number>,
): { memory: ExtractedMemory; index: number } | undefined {
  const hasExactSpecificAtomicSupport = (memory: ExtractedMemory): boolean => {
    const metadata = metadataOf(memory);
    return stringArray(metadata.input_candidate_ids).some((id) => {
      if (!id.startsWith("atomic:")) return false;
      const candidate = inputById.get(id);
      if (!candidate) return false;
      const candidateMetadata = metadataOf(candidate.memory);
      return stringValue(candidateMetadata.domain) === stringValue(metadata.domain)
        && stringValue(candidateMetadata.slot) === stringValue(metadata.slot)
        && comparableValue(candidateMetadata.value) === comparableValue(metadata.value)
        && referencedCandidateCoversOutput(id, memory, inputById);
    });
  };
  const specific = group.filter(({ memory }) => hasExactSpecificAtomicSupport(memory));
  if (specific.length !== 1) return undefined;
  const shadows = group.filter((entry) => entry !== specific[0]);
  if (shadows.length === 0) return undefined;
  const specificMemory = specific[0].memory;
  const specificMetadata = metadataOf(specificMemory);
  const specificRelation = stringValue(specificMetadata.relation);
  if ((specificRelation !== "asserted" && specificRelation !== "updated")
    || qualityStatus(specificMemory) !== "complete") return undefined;

  const compositeIds = new Set<string>();
  const safeShadows = shadows.every(({ memory }) => {
    const metadata = metadataOf(memory);
    const relation = stringValue(metadata.relation);
    const atomicReferences = stringArray(metadata.input_candidate_ids)
      .filter((id) => id.startsWith("atomic:"));
    if ((relation !== "cancelled" && relation !== "negated")
      || memory.type !== specificMemory.type
      || memory.scene_name !== specificMemory.scene_name
      || comparableValue(metadata.value) !== comparableValue(specificMetadata.value)
      || !hasExactlySameSources(memory, specificMemory)
      || !sameStringSet(
        stringArray(metadata.supersedes),
        stringArray(specificMetadata.supersedes),
      )
      || !bindingDominates(memory, specificMemory)
      || !bindingDominates(specificMemory, memory)
      || atomicReferences.length === 0) return false;
    return atomicReferences.every((id) => {
      const candidate = inputById.get(id);
      if (!candidate
        || !isRecoverableCompositeTransitionCandidate(candidate)
        || !sameStrings(qualityIssues(candidate.memory), ["ambiguous_transition_state"])
        || !stringArray(metadata.canonicalized_input_candidate_ids).includes(id)
        || !hasExactlySameSources(candidate.memory, memory)
        || stringValue(metadataOf(candidate.memory).episode_key)
          !== stringValue(metadata.episode_key)
        || comparableValue(metadataOf(candidate.memory).relation)
          !== comparableValue(metadata.relation)
        || comparableValue(metadataOf(candidate.memory).mentioned_at)
          !== comparableValue(metadata.mentioned_at)
        || comparableValue(metadataOf(candidate.memory).source_session_id)
          !== comparableValue(metadata.source_session_id)
        || !bindingDominates(candidate.memory, memory)
        || !bindingDominates(memory, candidate.memory)) return false;
      compositeIds.add(id);
      return true;
    });
  });
  if (!safeShadows || compositeIds.size !== 1) return undefined;

  let collapsed = cloneMemory(specificMemory);
  const collapsedMetadata = metadataOf(collapsed);
  const allSources = [...new Set(group.flatMap(({ memory }) => memory.source_message_ids))];
  const allReferences = [...new Set(group.flatMap(({ memory }) =>
    stringArray(metadataOf(memory).input_candidate_ids)
  ))];
  const allCanonicalized = [...new Set(group.flatMap(({ memory }) =>
    stringArray(metadataOf(memory).canonicalized_input_candidate_ids)
  ))].filter((id) => allReferences.includes(id));
  collapsed.source_message_ids = allSources;
  collapsedMetadata.source_message_ids = allSources;
  collapsedMetadata.input_candidate_ids = allReferences;
  if (allCanonicalized.length > 0) {
    collapsedMetadata.canonicalized_input_candidate_ids = allCanonicalized;
  } else {
    delete collapsedMetadata.canonicalized_input_candidate_ids;
  }
  collapsed = addQualityRepair(
    collapsed,
    "collapsed_composite_transition_shadow_to_specific_atomic_edge",
  );
  bump(repairs, "collapsed_composite_transition_shadow_to_specific_atomic_edge");
  return { memory: collapsed, index: Math.min(...group.map((entry) => entry.index)) };
}

function coalesceDuplicateIdentities(
  memories: ExtractedMemory[],
  repairs: Map<string, number>,
  inputById: Map<string, CockpitReconciliationInput>,
): ExtractedMemory[] {
  const grouped = new Map<string, Array<{ memory: ExtractedMemory; index: number }>>();
  for (const [index, memory] of memories.entries()) {
    const identity = cockpitFinalIdentity(memory) ?? `__missing_identity__:${index}`;
    const group = grouped.get(identity) ?? [];
    group.push({ memory, index });
    grouped.set(identity, group);
  }

  const output: Array<{ memory: ExtractedMemory; index: number }> = [];
  for (const group of grouped.values()) {
    if (group.length === 1) {
      output.push(group[0]);
      continue;
    }
    // Never let ordinary duplicate ranking promote a deterministic ambiguity
    // blocker back to `complete`. Preserve the rows so both the partial-status
    // and duplicate-identity checks remain visible to the public gate.
    if (group.some(({ memory }) => hasBlockingQualityIssue(memory))) {
      const repair = "preserved_blocking_duplicate_final_identity";
      const alreadyMarked = group.every(({ memory }) => {
        const quality = metadataOf(memory).construction_quality;
        return quality && typeof quality === "object" && !Array.isArray(quality)
          && stringArray((quality as Record<string, unknown>).repairs).includes(repair);
      });
      if (!alreadyMarked) bump(repairs, repair);
      output.push(...group.map(({ memory, index }) => ({
        memory: addQualityRepair(memory, repair),
        index,
      })));
      continue;
    }
    const collapsedComposite = collapseCompositeTransitionShadowGroup(
      group,
      inputById,
      repairs,
    );
    if (collapsedComposite) {
      output.push(collapsedComposite);
      continue;
    }
    const first = group[0].memory;
    const firstMetadata = metadataOf(first);
    const exactValues = new Set(group.map(({ memory }) =>
      comparableValue(metadataOf(memory).value)
    ));
    const timestampValues = new Map<number, string>();
    let hasAmbiguousTimestampValue = false;
    if (exactValues.size > 1) {
      for (const { memory } of group) {
        const timestamp = mentionedAtMs(memory);
        const value = comparableValue(metadataOf(memory).value);
        if (!Number.isFinite(timestamp)) {
          hasAmbiguousTimestampValue = true;
          break;
        }
        const existing = timestampValues.get(timestamp);
        if (existing !== undefined && existing !== value) {
          hasAmbiguousTimestampValue = true;
          break;
        }
        timestampValues.set(timestamp, value);
      }
    }
    const incompatibleFacts = group.some(({ memory }) => {
      const metadata = metadataOf(memory);
      const target = metadata.target;
      return memory.type !== first.type
        || memory.scene_name !== first.scene_name
        || comparableValue(metadata.relation) !== comparableValue(firstMetadata.relation)
        || !sameStringSet(
          stringArray(metadata.supersedes),
          stringArray(firstMetadata.supersedes),
        )
        || (target !== undefined
          && comparableValue(target) !== comparableValue(metadata.value))
        || [
          "unit",
          "action_status",
          "record_kind",
          "activity_start_time",
          "activity_end_time",
          "timezone",
          "time_precision",
          "temporal_status",
        ].some((key) =>
          metadata[key] !== undefined
            && firstMetadata[key] !== undefined
            && comparableValue(metadata[key]) !== comparableValue(firstMetadata[key])
        )
        || !(bindingDominates(memory, first) || bindingDominates(first, memory));
    });
    if (incompatibleFacts || hasAmbiguousTimestampValue) {
      const repair = "preserved_conflicting_duplicate_final_identity";
      const alreadyMarked = group.every(({ memory }) => {
        const quality = metadataOf(memory).construction_quality;
        return quality && typeof quality === "object" && !Array.isArray(quality)
          && stringArray((quality as Record<string, unknown>).repairs).includes(repair);
      });
      if (!alreadyMarked) bump(repairs, repair);
      output.push(...group.map(({ memory, index }) => ({
        memory: addQualityRepair(memory, repair),
        index,
      })));
      continue;
    }
    const ranked = [...group].sort((left, right) =>
      exactAtomicReferenceFidelity(right.memory, inputById)
        - exactAtomicReferenceFidelity(left.memory, inputById)
        || mentionedAtMs(right.memory) - mentionedAtMs(left.memory)
        || atomicReferenceCount(right.memory) - atomicReferenceCount(left.memory)
        || right.memory.source_message_ids.length - left.memory.source_message_ids.length
        || left.index - right.index
    );
    let winner = cloneMemory(ranked[0].memory);
    const allSources = [...new Set(group.flatMap(({ memory }) => memory.source_message_ids))];
    const allReferences = [...new Set(group.flatMap(({ memory }) =>
      stringArray(metadataOf(memory).input_candidate_ids)
    ))];
    const allCanonicalized = [...new Set(group.flatMap(({ memory }) =>
      stringArray(metadataOf(memory).canonicalized_input_candidate_ids)
    ))].filter((id) => allReferences.includes(id));
    const allEvidenceRoles = [...new Set(group.flatMap(({ memory }) =>
      stringArray(metadataOf(memory).evidence_roles)
    ))];
    winner.source_message_ids = allSources;
    const metadata = metadataOf(winner);
    metadata.source_message_ids = allSources;
    metadata.input_candidate_ids = allReferences;
    if (allCanonicalized.length > 0) metadata.canonicalized_input_candidate_ids = allCanonicalized;
    else delete metadata.canonicalized_input_candidate_ids;
    if (allEvidenceRoles.length > 0) metadata.evidence_roles = allEvidenceRoles;
    if (!preservesEveryReferencedObligation(winner, inputById)) {
      const repair = "preserved_conflicting_duplicate_final_identity";
      const alreadyMarked = group.every(({ memory }) => {
        const quality = metadataOf(memory).construction_quality;
        return quality && typeof quality === "object" && !Array.isArray(quality)
          && stringArray((quality as Record<string, unknown>).repairs).includes(repair);
      });
      if (!alreadyMarked) bump(repairs, repair);
      output.push(...group.map(({ memory, index }) => ({
        memory: addQualityRepair(memory, repair),
        index,
      })));
      continue;
    }
    winner = addQualityRepair(winner, "coalesced_duplicate_final_identity");
    bump(repairs, "coalesced_duplicate_final_identity");
    output.push({ memory: winner, index: Math.min(...group.map((entry) => entry.index)) });
  }
  return output.sort((left, right) => left.index - right.index).map((entry) => entry.memory);
}

function candidateIsCovered(
  candidate: CockpitReconciliationInput,
  memories: ExtractedMemory[],
  resolved: Set<string>,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  if (resolved.has(candidate.id)) return true;
  return memories.some((memory) => {
    const metadata = metadataOf(memory);
    const references = stringArray(metadata.input_candidate_ids);
    if (!references.includes(candidate.id)) return false;
    const canonicalized = stringArray(metadata.canonicalized_input_candidate_ids).includes(candidate.id);
    return candidateCoversOutput(candidate, memory, canonicalized, inputById);
  });
}

function preservesEveryReferencedObligation(
  memory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const metadata = metadataOf(memory);
  const canonicalized = new Set(stringArray(metadata.canonicalized_input_candidate_ids));
  return stringArray(metadata.input_candidate_ids)
    .filter((reference) => reference.startsWith("atomic:") || reference.startsWith("coverage:"))
    .every((reference) => {
      const input = inputById.get(reference);
      return Boolean(input && candidateCoversOutput(
        input,
        memory,
        canonicalized.has(reference),
        inputById,
      ));
    });
}

/**
 * Collapse an explicit "cancel the old value, replace it with this value"
 * transaction to the replacement edge before duplicate final identities are
 * coalesced. This is deliberately evidence-bounded: the cancelled value must
 * equal a persisted live predecessor, both sides must share source evidence,
 * and every consumed transition must be backed by an exact atomic candidate.
 *
 * The discarded cancellation is not lost. Its atomic obligation is recorded
 * as deterministically resolved by the replacement edge; `processPriors`
 * subsequently promotes the asserted replacement to `updated` and binds the
 * exact predecessor record ID. A pure cancellation has no replacement row and
 * therefore never enters this path.
 */
function resolveAtomicCancelReplacePairs(
  memories: ExtractedMemory[],
  priors: CockpitReconciliationPriorMemory[],
  inputById: Map<string, CockpitReconciliationInput>,
  resolved: Set<string>,
  repairs: Map<string, number>,
): ExtractedMemory[] {
  const grouped = new Map<string, Array<{ memory: ExtractedMemory; index: number }>>();
  for (const [index, memory] of memories.entries()) {
    const identity = cockpitFinalIdentity(memory) ?? `__missing_identity__:${index}`;
    const group = grouped.get(identity) ?? [];
    group.push({ memory, index });
    grouped.set(identity, group);
  }

  const output: Array<{ memory: ExtractedMemory; index: number }> = [];
  for (const group of grouped.values()) {
    if (group.some(({ memory }) => hasBlockingQualityIssue(memory))) {
      output.push(...group);
      continue;
    }
    const replacements = group.filter(({ memory }) => {
      const relation = stringValue(metadataOf(memory).relation);
      return relation === "asserted" || relation === "updated";
    });
    const cancellations = group.filter(({ memory }) => {
      const relation = stringValue(metadataOf(memory).relation);
      return relation === "cancelled" || relation === "negated";
    });
    if (replacements.length !== 1 || cancellations.length === 0
      || replacements.length + cancellations.length !== group.length) {
      output.push(...group);
      continue;
    }

    const replacement = replacements[0].memory;
    const liveTargets = matchingLivePrior(replacement, priors);
    if (liveTargets.length !== 1
      || !hasExactReplacementAnchor(replacement, liveTargets[0], inputById)) {
      output.push(...group);
      continue;
    }

    const cancellationAtomicIds = new Set<string>();
    const safePair = cancellations.every(({ memory: cancellation }) => {
      if (cancellation.type !== replacement.type
        || cancellation.scene_name !== replacement.scene_name
        || !sameTransitionEvidence(cancellation, replacement)) return false;
      const cancellationValue = comparableValue(metadataOf(cancellation).value);
      const matchesPersistedValue = matchingLivePrior(cancellation, priors)
        .some((prior) => comparableValue(prior.metadata.value) === cancellationValue);
      if (!matchesPersistedValue) return false;
      const exactAtomicIds = stringArray(metadataOf(cancellation).input_candidate_ids)
        .filter((id) => {
          if (!id.startsWith("atomic:")) return false;
          const candidate = inputById.get(id);
          return Boolean(candidate
            && referencedCandidateCoversOutput(id, cancellation, inputById)
            && comparableValue(metadataOf(candidate.memory).value) === cancellationValue);
        });
      if (exactAtomicIds.length === 0) return false;
      exactAtomicIds.forEach((id) => cancellationAtomicIds.add(id));
      return true;
    });
    if (!safePair) {
      output.push(...group);
      continue;
    }

    let collapsed = cloneMemory(replacement);
    const collapsedMetadata = metadataOf(collapsed);
    const transferableCoverageIds = cancellations.flatMap(({ memory }) =>
      stringArray(metadataOf(memory).input_candidate_ids).filter((id) => {
        if (!id.startsWith("coverage:")) return false;
        const candidate = inputById.get(id);
        return Boolean(candidate && candidateCoversOutput(candidate, collapsed));
      })
    );
    collapsedMetadata.input_candidate_ids = [...new Set([
      ...stringArray(collapsedMetadata.input_candidate_ids)
        .filter((id) => !cancellationAtomicIds.has(id)),
      ...transferableCoverageIds,
    ])];
    const allSources = [...new Set([
      ...collapsed.source_message_ids,
      ...cancellations.flatMap(({ memory }) => memory.source_message_ids),
    ])];
    const allSourceSessions = [...new Set([
      ...stringArray(collapsedMetadata.source_session_ids),
      ...cancellations.flatMap(({ memory }) =>
        stringArray(metadataOf(memory).source_session_ids)
      ),
    ])];
    const allEvidenceRoles = [...new Set([
      ...stringArray(collapsedMetadata.evidence_roles),
      ...cancellations.flatMap(({ memory }) =>
        stringArray(metadataOf(memory).evidence_roles)
      ),
    ])];
    collapsed.source_message_ids = allSources;
    collapsedMetadata.source_message_ids = allSources;
    if (allSourceSessions.length > 0) collapsedMetadata.source_session_ids = allSourceSessions;
    if (allEvidenceRoles.length > 0) collapsedMetadata.evidence_roles = allEvidenceRoles;
    collapsed = addQualityRepair(collapsed, "resolved_atomic_cancel_replace_pair_to_live_update");
    cancellationAtomicIds.forEach((id) => resolved.add(id));
    cancellations.forEach(() => bump(repairs, "resolved_atomic_cancel_replace_pair_to_live_update"));
    output.push({
      memory: collapsed,
      index: Math.min(...group.map((entry) => entry.index)),
    });
  }

  return output.sort((left, right) => left.index - right.index).map((entry) => entry.memory);
}

function hasExactEventProvenance(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  return hasExactlySameSources(left, right)
    && comparableValue(leftMetadata.source_session_id)
      === comparableValue(rightMetadata.source_session_id)
    && sameStringSet(
      stringArray(leftMetadata.source_session_ids),
      stringArray(rightMetadata.source_session_ids),
    )
    && comparableValue(leftMetadata.mentioned_at) === comparableValue(rightMetadata.mentioned_at);
}

function sameTransitionEvidence(left: ExtractedMemory, right: ExtractedMemory): boolean {
  return hasExactEventProvenance(left, right)
    && bindingDominates(left, right)
    && bindingDominates(right, left);
}

const EXACT_ATOMIC_NON_VALUE_FACT_KEYS = [
  "domain",
  "slot",
  "state_key",
  "episode_key",
  "relation",
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
  "timezone",
  "time_precision",
  "temporal_status",
] as const;

function hasExactAtomicStructuredNonValueFacts(
  candidate: ExtractedMemory,
  output: ExtractedMemory,
): boolean {
  const candidateMetadata = metadataOf(candidate);
  const outputMetadata = metadataOf(output);
  return candidate.type === output.type
    && candidate.scene_name === output.scene_name
    && bindingDominates(candidate, output)
    && bindingDominates(output, candidate)
    && EXACT_ATOMIC_NON_VALUE_FACT_KEYS.every((key) =>
      comparableValue(candidateMetadata[key]) === comparableValue(outputMetadata[key])
    )
    && sameStringSet(
      stringArray(candidateMetadata.supersedes),
      stringArray(outputMetadata.supersedes),
    );
}

function hasAggregatedEventEvidence(
  candidate: ExtractedMemory,
  output: ExtractedMemory,
  requireStrictlyLater: boolean,
): boolean {
  const outputSources = new Set(output.source_message_ids);
  if (candidate.source_message_ids.length === 0
    || !candidate.source_message_ids.every((source) => outputSources.has(source))) return false;
  const candidateMetadata = metadataOf(candidate);
  const outputMetadata = metadataOf(output);
  const candidateSessions = stringArray(candidateMetadata.source_session_ids);
  const outputSessions = new Set(stringArray(outputMetadata.source_session_ids));
  if ((candidateSessions.length === 0) !== (outputSessions.size === 0)
    || !candidateSessions.every((session) => outputSessions.has(session))) return false;
  const candidateSession = stringValue(candidateMetadata.source_session_id);
  const outputSession = stringValue(outputMetadata.source_session_id);
  if (Boolean(candidateSession) !== Boolean(outputSession)) return false;
  if (candidateSession && candidateSession !== outputSession && !outputSessions.has(candidateSession)) {
    return false;
  }
  const candidateTime = mentionedAtMs(candidate);
  const outputTime = mentionedAtMs(output);
  if (!Number.isFinite(candidateTime) || !Number.isFinite(outputTime)) return false;
  return requireStrictlyLater ? outputTime > candidateTime : outputTime >= candidateTime;
}

function hasSelfConsistentValueTarget(memory: ExtractedMemory): boolean {
  const metadata = metadataOf(memory);
  return metadata.target === undefined
    || comparableValue(metadata.target) === comparableValue(metadata.value);
}

function hasExactReferencedLatestValueAnchor(
  output: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const outputMetadata = metadataOf(output);
  return stringArray(outputMetadata.input_candidate_ids).some((id) => {
    if (!id.startsWith("primary:") && !id.startsWith("atomic:")) return false;
    const anchor = inputById.get(id);
    if (!anchor || qualityStatus(anchor.memory) !== "complete") return false;
    const anchorMetadata = metadataOf(anchor.memory);
    return hasExactAtomicStructuredNonValueFacts(anchor.memory, output)
      && comparableValue(anchorMetadata.value) === comparableValue(outputMetadata.value)
      && comparableValue(anchorMetadata.target) === comparableValue(outputMetadata.target)
      && comparableValue(anchorMetadata.mentioned_at)
        === comparableValue(outputMetadata.mentioned_at)
      && hasAggregatedEventEvidence(anchor.memory, output, false);
  });
}

function isGenericMissingScopeStateKeyRepair(
  candidate: ExtractedMemory,
  output: ExtractedMemory,
): boolean {
  if (!qualityIssues(candidate).includes("missing_scope")
    || !bindingDominates(candidate, output)
    || !bindingDominates(output, candidate)) return false;
  const bindingKeys = ["subject", "occupant_scope", "vehicle_scope", "seat_zone"];
  if (bindingKeys.some((key) =>
    canonicalBindingValue(key, metadataOf(candidate)[key]) !== undefined
      || canonicalBindingValue(key, metadataOf(output)[key]) !== undefined
  )) return false;
  const candidateState = stringValue(metadataOf(candidate).state_key)?.split("|");
  const outputState = stringValue(metadataOf(output).state_key)?.split("|");
  if (!candidateState || !outputState || candidateState.length !== 5 || outputState.length !== 5) {
    return false;
  }
  return candidateState[0] === outputState[0]
    && candidateState[1] === "unspecified-subject"
    && new Set(["user", "用户", "我", "本人", "用户本人"]).has(outputState[1])
    && candidateState[2] === outputState[2]
    && candidateState[3] === outputState[3]
    && candidateState[4] === outputState[4];
}

function partialAtomicCandidateCoversOutput(
  candidate: ExtractedMemory,
  output: ExtractedMemory,
  explicitlyCanonicalized: boolean,
): boolean {
  if (!cockpitAtomicCandidateCovers(candidate, output, explicitlyCanonicalized)
    || candidate.type !== output.type
    || candidate.scene_name !== output.scene_name) return false;
  // The composite schedule-status transition is repaired only by its separate
  // exact multi-edge proof. Keep that established path structural here; every
  // other partial atomic must preserve all facts it actually supplied.
  if (explicitlyCanonicalized && isRecoverableCompositeTransitionMemory(candidate)) return true;
  const candidateMetadata = metadataOf(candidate);
  const outputMetadata = metadataOf(output);
  const outputSources = new Set(output.source_message_ids);
  if (candidate.source_message_ids.length === 0
    || !candidate.source_message_ids.every((source) => outputSources.has(source))
    || !bindingDominates(output, candidate)) return false;
  const relation = stringValue(candidateMetadata.relation);
  const repairsMissingPredecessor = qualityIssues(candidate).includes("missing_supersedes")
    && (relation === "updated" || relation === "cancelled" || relation === "negated")
    && stringArray(candidateMetadata.supersedes).length === 0
    && stringArray(outputMetadata.supersedes).length > 0;
  const repairsGenericMissingScope = isGenericMissingScopeStateKeyRepair(candidate, output);
  const factKeys = [
    ...EXACT_ATOMIC_NON_VALUE_FACT_KEYS,
    "value",
    "target",
    "mentioned_at",
    "source_session_id",
  ].filter((key) => !(
    (repairsMissingPredecessor && key === "episode_key")
      || (repairsGenericMissingScope && key === "state_key")
  ));
  if (!factKeys.every((key) => candidateMetadata[key] === undefined
    || comparableValue(candidateMetadata[key]) === comparableValue(outputMetadata[key]))) {
    return false;
  }
  const candidateSessions = stringArray(candidateMetadata.source_session_ids);
  const outputSessions = new Set(stringArray(outputMetadata.source_session_ids));
  if (!candidateSessions.every((session) => outputSessions.has(session))) return false;
  const candidateSupersedes = stringArray(candidateMetadata.supersedes);
  return candidateSupersedes.length === 0 || sameStringSet(
    candidateSupersedes,
    stringArray(outputMetadata.supersedes),
  );
}

/**
 * A complete atomic candidate is a factual obligation, not merely a slot
 * hint. Every reference must preserve its own exact event and structured fact.
 * The sole value-changing exception is the separately proven appointment
 * activity/location alias; it is recomputed from one exact destination edge.
 */
function completeAtomicCandidateCoversOutput(
  candidate: ExtractedMemory,
  output: ExtractedMemory,
  explicitlyCanonicalized: boolean,
  inputById?: Map<string, CockpitReconciliationInput>,
): boolean {
  if (!cockpitAtomicCandidateCovers(candidate, output, explicitlyCanonicalized)
    || !hasExactAtomicStructuredNonValueFacts(candidate, output)) return false;
  const candidateMetadata = metadataOf(candidate);
  const outputMetadata = metadataOf(output);
  if (comparableValue(candidateMetadata.value) === comparableValue(outputMetadata.value)
    && comparableValue(candidateMetadata.target) === comparableValue(outputMetadata.target)) {
    return hasExactEventProvenance(candidate, output)
      || hasAggregatedEventEvidence(candidate, output, false);
  }
  if (inputById
    && hasExactEventProvenance(candidate, output)
    && hasSelfConsistentValueTarget(candidate)
    && hasSelfConsistentValueTarget(output)
    && appointmentContainmentPair(candidate, output)?.coarse === candidate
    && hasUniqueExactDestinationProof(candidate, output, inputById)) return true;
  // Multiple exact observations of one final identity may be compacted to the
  // latest state only when the chosen row carries the complete older evidence
  // set and has a strictly later observed event time. Equal/unknown times can
  // never turn a conflicting sibling value into a supersession.
  return hasSelfConsistentValueTarget(candidate)
    && hasSelfConsistentValueTarget(output)
    && inputById !== undefined
    && hasAggregatedEventEvidence(candidate, output, true)
    && hasExactReferencedLatestValueAnchor(output, inputById);
}

const EXACT_PRIMARY_REPLACEMENT_FACT_KEYS = COCKPIT_PERSISTED_STATE_FACT_KEYS.filter(
  (key) => key !== "value" && key !== "target",
);

function hasPrimaryUpdatedReplacementSupport(
  replacement: ExtractedMemory,
  target: CockpitReconciliationPriorMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const replacementMetadata = metadataOf(replacement);
  if (!cockpitPriorMatchesMemoryState(target, replacement)) return false;
  return stringArray(replacementMetadata.input_candidate_ids).some((id) => {
    if (!id.startsWith("primary:")) return false;
    const candidate = inputById.get(id);
    if (!candidate || qualityStatus(candidate.memory) !== "complete") return false;
    const candidateMetadata = metadataOf(candidate.memory);
    if (stringValue(candidateMetadata.relation) !== "updated"
      || candidate.memory.type !== replacement.type
      || candidate.memory.scene_name !== replacement.scene_name
      || candidate.memory.source_message_ids.length === 0
      || !hasExactEventProvenance(candidate.memory, replacement)
      || !(bindingDominates(candidate.memory, replacement)
        && bindingDominates(replacement, candidate.memory))
      || !EXACT_PRIMARY_REPLACEMENT_FACT_KEYS.every((key) =>
        comparableValue(candidateMetadata[key]) === comparableValue(replacementMetadata[key])
      )
      || comparableValue(candidateMetadata.value) !== comparableValue(replacementMetadata.value)
      || comparableValue(candidateMetadata.target) !== comparableValue(replacementMetadata.target)) {
      return false;
    }
    const references = stringArray(candidateMetadata.supersedes);
    const targetEpisode = stringValue(target.metadata.episode_key);
    return references.length === 1
      && (references[0] === target.record_id || references[0] === targetEpisode);
  });
}

function hasExactPrimaryOutputSupport(
  candidate: CockpitReconciliationInput,
  output: ExtractedMemory,
): boolean {
  if (qualityStatus(candidate.memory) !== "complete"
    || candidate.memory.type !== output.type
    || candidate.memory.scene_name !== output.scene_name
    || candidate.memory.source_message_ids.length === 0
    || !hasExactEventProvenance(candidate.memory, output)
    || !(bindingDominates(candidate.memory, output)
      && bindingDominates(output, candidate.memory))) return false;
  const candidateMetadata = metadataOf(candidate.memory);
  const outputMetadata = metadataOf(output);
  return COCKPIT_PERSISTED_STATE_FACT_KEYS.every((key) =>
    comparableValue(candidateMetadata[key]) === comparableValue(outputMetadata[key])
  ) && comparableValue(candidateMetadata.relation) === comparableValue(outputMetadata.relation)
    && sameStringSet(
      stringArray(candidateMetadata.supersedes),
      stringArray(outputMetadata.supersedes),
    );
}

// A primary update may enter the assembler beside a decomposed atomic
// assertion for the new episode. Preserve that reference only long enough for
// the bounded cancel/replace transaction proof to validate it. The ordinary
// gate still rejects the relation/supersedes rewrite unless the assembler
// later registers the exact final row as a verified deterministic rewrite.
function hasFactualPrimaryAssemblyEntrySupport(
  candidate: CockpitReconciliationInput,
  output: ExtractedMemory,
): boolean {
  if (hasExactPrimaryOutputSupport(candidate, output)) return true;
  if (qualityStatus(candidate.memory) !== "complete"
    || candidate.memory.type !== output.type
    || candidate.memory.scene_name !== output.scene_name
    || candidate.memory.source_message_ids.length === 0
    || !hasExactEventProvenance(candidate.memory, output)
    || !(bindingDominates(candidate.memory, output)
      && bindingDominates(output, candidate.memory))) return false;
  const candidateMetadata = metadataOf(candidate.memory);
  const outputMetadata = metadataOf(output);
  if (!COCKPIT_PERSISTED_STATE_FACT_KEYS.every((key) =>
    comparableValue(candidateMetadata[key]) === comparableValue(outputMetadata[key])
  )) return false;
  const candidateSupersedes = stringArray(candidateMetadata.supersedes);
  return stringValue(candidateMetadata.relation) === "updated"
    && stringValue(outputMetadata.relation) === "asserted"
    && candidateSupersedes.length === 1
    && stringArray(outputMetadata.supersedes).length === 0;
}

function hasExactReplacementAnchor(
  replacement: ExtractedMemory,
  target: CockpitReconciliationPriorMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  return stringArray(metadataOf(replacement).input_candidate_ids).some((id) => {
    const candidate = inputById.get(id);
    if (!candidate || qualityStatus(candidate.memory) !== "complete") return false;
    if (id.startsWith("atomic:")) {
      return referencedCandidateCoversOutput(id, replacement, inputById);
    }
    if (!id.startsWith("primary:")) return false;
    return hasExactPrimaryOutputSupport(candidate, replacement)
      || hasPrimaryUpdatedReplacementSupport(replacement, target, inputById);
  });
}

interface CrossEpisodeCancelReplacePair {
  cancellation: ExtractedMemory;
  cancellationIndex: number;
  cancellationAtomicIds: string[];
  replacement: ExtractedMemory;
  replacementIndex: number;
  target: CockpitReconciliationPriorMemory;
}

const CONTROLLED_APPOINTMENT_MIGRATION_STATES = new Set([
  "schedule|appointment_time",
  "schedule|appointment_content",
  "navigation|destination",
]);

function isControlledAppointmentMigrationState(
  metadata: Record<string, unknown>,
): boolean {
  const domain = stringValue(metadata.domain);
  const slot = stringValue(metadata.slot);
  return Boolean(domain && slot
    && CONTROLLED_APPOINTMENT_MIGRATION_STATES.has(`${domain}|${slot}`));
}

function crossEpisodeTransactionKey(pair: CrossEpisodeCancelReplacePair): string | undefined {
  const cancellationMetadata = metadataOf(pair.cancellation);
  const replacementMetadata = metadataOf(pair.replacement);
  const oldEpisode = stringValue(cancellationMetadata.episode_key);
  const newEpisode = stringValue(replacementMetadata.episode_key);
  if (!oldEpisode || !newEpisode || oldEpisode === newEpisode) return undefined;
  return JSON.stringify({
    type: pair.cancellation.type,
    sources: [...new Set(pair.cancellation.source_message_ids)].sort(),
    source_session_id: cancellationMetadata.source_session_id,
    source_session_ids: stringArray(cancellationMetadata.source_session_ids).sort(),
    mentioned_at: cancellationMetadata.mentioned_at,
    subject: canonicalBindingValue("subject", cancellationMetadata.subject),
    occupant_scope: canonicalBindingValue("occupant_scope", cancellationMetadata.occupant_scope),
    vehicle_scope: canonicalBindingValue("vehicle_scope", cancellationMetadata.vehicle_scope),
    seat_zone: canonicalBindingValue("seat_zone", cancellationMetadata.seat_zone),
    old_episode: oldEpisode,
    new_episode: newEpisode,
  });
}

/**
 * Collapse a model's cross-episode decomposition of one explicit reschedule.
 *
 * Some providers represent "cancel the old appointment and move it" as a
 * cancelled old episode plus an asserted new episode, even while the primary
 * proposal marks each concrete slot as `updated`.  Episode identity alone
 * cannot safely distinguish that representation from an unrelated cancel and
 * create transaction.  This repair therefore requires all of the following:
 * exact shared evidence/time/person bindings, a unique persisted predecessor
 * whose old value is repeated by the cancellation, exact atomic cancellation
 * support, a primary `updated` proposal for every replacement, and at least
 * two distinct slots migrating between the same old/new episode pair.  Pure
 * cancellations, single-slot edits, ambiguous predecessors, mixed occupants,
 * and independent new appointments all fail closed.
 */
function resolveCrossEpisodeAtomicCancelReplaceTransactions(
  memories: ExtractedMemory[],
  priors: CockpitReconciliationPriorMemory[],
  inputById: Map<string, CockpitReconciliationInput>,
  resolved: Set<string>,
  repairs: Map<string, number>,
): ExtractedMemory[] {
  const pairs: CrossEpisodeCancelReplacePair[] = [];
  for (const [cancellationIndex, cancellation] of memories.entries()) {
    const cancellationMetadata = metadataOf(cancellation);
    const cancellationRelation = stringValue(cancellationMetadata.relation);
    if ((cancellationRelation !== "cancelled" && cancellationRelation !== "negated")
      || qualityStatus(cancellation) !== "complete") continue;
    const targets = matchingLivePrior(cancellation, priors).filter((prior) =>
      comparableValue(prior.metadata.value) === comparableValue(cancellationMetadata.value)
    );
    if (targets.length !== 1
      || !sameStrings(stringArray(cancellationMetadata.supersedes), [targets[0].record_id])) {
      continue;
    }
    const cancellationAtomicIds = stringArray(cancellationMetadata.input_candidate_ids)
      .filter((id) => {
        if (!id.startsWith("atomic:")) return false;
        const candidate = inputById.get(id);
        if (!candidate
          || !referencedCandidateCoversOutput(id, cancellation, inputById)) return false;
        const candidateMetadata = metadataOf(candidate.memory);
        return stringValue(candidateMetadata.relation) === cancellationRelation
          && comparableValue(candidateMetadata.value)
            === comparableValue(cancellationMetadata.value);
      });
    if (cancellationAtomicIds.length === 0) continue;

    const replacements = memories
      .map((replacement, replacementIndex) => ({ replacement, replacementIndex }))
      .filter(({ replacement, replacementIndex }) => {
        if (replacementIndex === cancellationIndex || qualityStatus(replacement) !== "complete") {
          return false;
        }
        const replacementMetadata = metadataOf(replacement);
        const replacementRelation = stringValue(replacementMetadata.relation);
        return (replacementRelation === "asserted" || replacementRelation === "updated")
          && replacement.type === cancellation.type
          && replacement.scene_name === cancellation.scene_name
          && stringValue(replacementMetadata.domain) === stringValue(cancellationMetadata.domain)
          && stringValue(replacementMetadata.slot) === stringValue(cancellationMetadata.slot)
          && stringValue(replacementMetadata.state_key) === stringValue(cancellationMetadata.state_key)
          && stringValue(replacementMetadata.episode_key)
            !== stringValue(cancellationMetadata.episode_key)
          && sameTransitionEvidence(cancellation, replacement)
          && hasPrimaryUpdatedReplacementSupport(replacement, targets[0], inputById);
      });
    if (replacements.length !== 1) continue;
    pairs.push({
      cancellation,
      cancellationIndex,
      cancellationAtomicIds,
      replacement: replacements[0].replacement,
      replacementIndex: replacements[0].replacementIndex,
      target: targets[0],
    });
  }

  const groups = new Map<string, CrossEpisodeCancelReplacePair[]>();
  for (const pair of pairs) {
    const key = crossEpisodeTransactionKey(pair);
    if (!key) continue;
    const group = groups.get(key) ?? [];
    group.push(pair);
    groups.set(key, group);
  }

  const consumed = new Set<number>();
  const collapsed: Array<{ memory: ExtractedMemory; index: number }> = [];
  for (const group of groups.values()) {
    const stateKeys = new Set(group.map((pair) => stringValue(
      metadataOf(pair.cancellation).state_key,
    )).filter((value): value is string => Boolean(value)));
    const uniqueCancellationIndexes = new Set(group.map((pair) => pair.cancellationIndex));
    const uniqueReplacementIndexes = new Set(group.map((pair) => pair.replacementIndex));
    const hasAppointmentState = group.some((pair) =>
      stringValue(metadataOf(pair.cancellation).domain) === "schedule"
        && ["appointment_time", "appointment_content"].includes(
          stringValue(metadataOf(pair.cancellation).slot) ?? "",
        )
    );
    const hasOnlyControlledAppointmentStates = group.every((pair) =>
      isControlledAppointmentMigrationState(metadataOf(pair.cancellation))
    );
    const hasExactControlledTransactionScenes = group.every((pair) =>
      pair.cancellation.scene_name === stringValue(metadataOf(pair.cancellation).domain)
    );
    if (group.length < 2
      || stateKeys.size !== group.length
      || uniqueCancellationIndexes.size !== group.length
      || uniqueReplacementIndexes.size !== group.length
      || !hasAppointmentState
      || !hasOnlyControlledAppointmentStates
      || !hasExactControlledTransactionScenes
      || !hasCoherentSpecificBindings(group.flatMap((pair) => [
        pair.cancellation,
        pair.replacement,
      ]))) {
      continue;
    }

    for (const pair of group) {
      if (consumed.has(pair.cancellationIndex) || consumed.has(pair.replacementIndex)) continue;
      let replacement = cloneMemory(pair.replacement);
      const replacementMetadata = metadataOf(replacement);
      const cancellationMetadata = metadataOf(pair.cancellation);
      const transferableCoverageIds = stringArray(cancellationMetadata.input_candidate_ids)
        .filter((id) => {
          if (!id.startsWith("coverage:")) return false;
          const candidate = inputById.get(id);
          return Boolean(candidate && candidateCoversOutput(candidate, replacement));
        });
      replacementMetadata.input_candidate_ids = [...new Set([
        ...stringArray(replacementMetadata.input_candidate_ids)
          .filter((id) => !pair.cancellationAtomicIds.includes(id)),
        ...transferableCoverageIds,
      ])];
      replacementMetadata.episode_key = stringValue(pair.target.metadata.episode_key);
      replacementMetadata.relation = "updated";
      replacementMetadata.supersedes = [pair.target.record_id];
      const allSources = [...new Set([
        ...replacement.source_message_ids,
        ...pair.cancellation.source_message_ids,
      ])];
      const allSourceSessions = [...new Set([
        ...stringArray(replacementMetadata.source_session_ids),
        ...stringArray(cancellationMetadata.source_session_ids),
      ])];
      const allEvidenceRoles = [...new Set([
        ...stringArray(replacementMetadata.evidence_roles),
        ...stringArray(cancellationMetadata.evidence_roles),
      ])];
      replacement.source_message_ids = allSources;
      replacementMetadata.source_message_ids = allSources;
      if (allSourceSessions.length > 0) replacementMetadata.source_session_ids = allSourceSessions;
      if (allEvidenceRoles.length > 0) replacementMetadata.evidence_roles = allEvidenceRoles;
      replacement = addVerifiedFactualRewriteRepair(
        replacement,
        "resolved_cross_episode_cancel_replace_transaction_to_live_update",
      );
      pair.cancellationAtomicIds.forEach((id) => resolved.add(id));
      bump(repairs, "resolved_cross_episode_cancel_replace_transaction_to_live_update");
      consumed.add(pair.cancellationIndex);
      consumed.add(pair.replacementIndex);
      collapsed.push({
        memory: replacement,
        index: Math.min(pair.cancellationIndex, pair.replacementIndex),
      });
    }
  }

  const untouched = memories
    .map((memory, index) => ({ memory, index }))
    .filter(({ index }) => !consumed.has(index));
  return [...untouched, ...collapsed]
    .sort((left, right) => left.index - right.index)
    .map((entry) => entry.memory);
}

function hasSharedSource(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const rightSources = new Set(right.source_message_ids);
  return left.source_message_ids.some((source) => rightSources.has(source));
}

function exactProposalClass(
  memory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): "primary" | "atomic" | undefined {
  const hasPrimary = hasExactReferencedProposalEvidence(memory, "primary:", inputById);
  const hasAtomic = hasExactReferencedProposalEvidence(memory, "atomic:", inputById);
  if (hasPrimary === hasAtomic) return undefined;
  return hasPrimary ? "primary" : "atomic";
}

function hasExactReferencedProposalEvidence(
  memory: ExtractedMemory,
  prefix: "primary:" | "atomic:",
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const metadata = metadataOf(memory);
  return stringArray(metadata.input_candidate_ids).some((id) => {
    if (!id.startsWith(prefix)) return false;
    const candidate = inputById.get(id);
    if (!candidate || qualityStatus(candidate.memory) !== "complete") return false;
    const candidateMetadata = metadataOf(candidate.memory);
    return candidate.memory.type === memory.type
      && candidate.memory.scene_name === memory.scene_name
      && sameTransitionEvidence(candidate.memory, memory)
      && [...EXACT_ATOMIC_NON_VALUE_FACT_KEYS, "value", "target"]
        .every((key) => comparableValue(candidateMetadata[key]) === comparableValue(metadata[key]))
      && sameStringSet(
        stringArray(candidateMetadata.supersedes),
        stringArray(metadata.supersedes),
      );
  });
}

function hasOneWayPrimaryToAtomicEvidence(
  left: ExtractedMemory,
  right: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const leftHasPrimary = hasExactReferencedProposalEvidence(left, "primary:", inputById);
  const rightHasPrimary = hasExactReferencedProposalEvidence(right, "primary:", inputById);
  const leftHasAtomic = hasExactReferencedProposalEvidence(left, "atomic:", inputById);
  const rightHasAtomic = hasExactReferencedProposalEvidence(right, "atomic:", inputById);
  return (leftHasPrimary && !rightHasPrimary && rightHasAtomic)
    || (rightHasPrimary && !leftHasPrimary && leftHasAtomic);
}

function canonicalBindingValue(key: string, value: unknown): string | undefined {
  const normalized = stringValue(value)?.normalize("NFKC");
  if (!normalized) return undefined;
  const folded = normalized.toLocaleLowerCase();
  if (["unknown", "unspecified", "none", "null", "未知", "未指定"].includes(folded)) {
    return undefined;
  }
  // Keep source-bound map labels case-sensitive after NFKC; other controlled
  // binding vocabularies retain their existing case-folded comparison.
  const text = key === "state_qualifier" ? normalized : folded;
  if (key === "subject" && ["user", "用户", "我", "本人", "用户本人"].includes(text)) {
    return undefined;
  }
  if (key === "seat_zone") {
    if (["driver", "driver-seat", "驾驶位", "驾驶员", "主驾", "主驾驶"].includes(text)) return "driver";
    if (["front-passenger", "front-passenger-seat", "副驾", "副驾驶", "前排乘客位"].includes(text)) {
      return "front-passenger";
    }
  }
  return text;
}

function bindingDominates(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  return ["subject", "occupant_scope", "vehicle_scope", "seat_zone"].every((key) => {
    const expected = canonicalBindingValue(key, rightMetadata[key]);
    return !expected || canonicalBindingValue(key, leftMetadata[key]) === expected;
  });
}

function normalizedFactText(value: unknown): string | undefined {
  const text = stringValue(value)?.normalize("NFKC").toLocaleLowerCase();
  if (!text) return undefined;
  return text.replace(/[\s，。！？、；：,.!?;:'"“”‘’（）()【】\[\]]+/gu, "");
}

const COEPISODIC_TEMPORAL_FACT_KEYS = [
  "valid_from",
  "valid_to",
  "activity_start_time",
  "activity_end_time",
  "timezone",
  "time_precision",
  "temporal_status",
] as const;

function coepisodicTemporalFactsAreCompatible(
  left: ExtractedMemory,
  right: ExtractedMemory,
): boolean {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  return COEPISODIC_TEMPORAL_FACT_KEYS.every((key) =>
    leftMetadata[key] === undefined
      || rightMetadata[key] === undefined
      || comparableValue(leftMetadata[key]) === comparableValue(rightMetadata[key])
  );
}

/**
 * A controlled cross-domain event may use one scene per governed slot
 * (`schedule` + `navigation`). Arbitrary scene drift is never coherent.
 */
function coepisodicScenesAreCompatible(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const leftDomain = stringValue(metadataOf(left).domain);
  const rightDomain = stringValue(metadataOf(right).domain);
  if (leftDomain === rightDomain) return left.scene_name === right.scene_name;
  return new Set([leftDomain, rightDomain]).size === 2
    && [leftDomain, rightDomain].every((domain) =>
      domain === "schedule" || domain === "navigation"
    )
    && left.scene_name === leftDomain
    && right.scene_name === rightDomain;
}

function hasExactReferencedProposalAnchor(
  memory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  return stringArray(metadataOf(memory).input_candidate_ids).some((id) => {
    const candidate = inputById.get(id);
    if (!candidate || qualityStatus(candidate.memory) !== "complete") return false;
    if (id.startsWith("atomic:")) {
      return referencedCandidateCoversOutput(id, memory, inputById);
    }
    return id.startsWith("primary:") && hasExactPrimaryOutputSupport(candidate, memory);
  });
}

// A destination model pass can occasionally drop a bounded place qualifier
// that another complete slot in the same structured event retained (for
// example, `博物院` versus `在博物院内做检查`). Only close that gap when the
// richer suffix is an exact substring of one complete co-episodic appointment
// value, the people/scope/source/relation bindings are identical, and the
// following token is an appointment action boundary. This never parses a new
// place from raw prose or joins evidence across people, sources, or episodes.
const COEPISODIC_DESTINATION_QUALIFIER = new RegExp(
  String.raw`^(地下停车场|地上停车场|地下车库|停车场|地库|园区内|院内|店内|馆内|楼内|站内|机场内|景区内|商场内|医院内|学校内|公司内|小区内|门口|入口|出口|正门|侧门|附近|周边|旁边|对面|路口|内|里)(?=$|[\s，。！？、；：,.!?;:]|做|进行|办理|参加|召开|完成|安排|预约|检查|保养|维修|检测|充电|用餐|就餐|取|接|送)`,
  "u",
);

function coepisodicDestinationExtension(
  destinationValue: unknown,
  appointmentValue: unknown,
): string | undefined {
  const destination = stringValue(destinationValue)?.normalize("NFKC");
  const appointment = stringValue(appointmentValue)?.normalize("NFKC");
  if (!destination || !appointment) return undefined;
  const occurrences: number[] = [];
  for (let from = 0; from < appointment.length;) {
    const index = appointment.indexOf(destination, from);
    if (index < 0) break;
    occurrences.push(index);
    from = index + destination.length;
  }
  if (occurrences.length !== 1) return undefined;
  const remainder = appointment.slice(occurrences[0] + destination.length);
  const qualifier = COEPISODIC_DESTINATION_QUALIFIER.exec(remainder)?.[1];
  return qualifier ? `${destination}${qualifier}` : undefined;
}

function hasDestinationQualifierClosureProof(
  output: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const outputMetadata = metadataOf(output);
  const outputValue = stringValue(outputMetadata.value);
  if (!outputValue || stringValue(outputMetadata.target) !== outputValue) return false;
  const coarseCandidates = stringArray(outputMetadata.input_candidate_ids).flatMap((id) => {
    if (!id.startsWith("atomic:")) return [];
    const candidate = inputById.get(id);
    if (!candidate || qualityStatus(candidate.memory) !== "complete") return [];
    const metadata = metadataOf(candidate.memory);
    return candidate.memory.type === output.type
      && candidate.memory.scene_name === output.scene_name
      && stringValue(metadata.domain) === "navigation"
      && stringValue(metadata.slot) === "destination"
      && hasExactEventProvenance(candidate.memory, output)
      && hasExactAtomicStructuredNonValueFacts(candidate.memory, output)
      ? [candidate.memory]
      : [];
  });
  const proofs: Array<[ExtractedMemory, ExtractedMemory]> = [];
  for (const coarse of coarseCandidates) {
    for (const candidate of inputById.values()) {
      const appointment = candidate.memory;
      const appointmentMetadata = metadataOf(appointment);
      if ((!candidate.id.startsWith("atomic:") && !candidate.id.startsWith("primary:"))
        || qualityStatus(appointment) !== "complete"
        || appointment.type !== output.type
        || !coepisodicScenesAreCompatible(output, appointment)
        || stringValue(appointmentMetadata.domain) !== "schedule"
        || stringValue(appointmentMetadata.slot) !== "appointment_content"
        || stringValue(appointmentMetadata.episode_key) !== stringValue(outputMetadata.episode_key)
        || stringValue(appointmentMetadata.relation) !== stringValue(outputMetadata.relation)
        || !hasExactEventProvenance(appointment, output)
        || !sameTransitionEvidence(appointment, output)
        || !coepisodicTemporalFactsAreCompatible(appointment, output)
        || coepisodicDestinationExtension(
          metadataOf(coarse).value,
          appointmentMetadata.value,
        ) !== outputValue) continue;
      proofs.push([coarse, appointment]);
    }
  }
  return proofs.length === 1;
}

function hasImmutableStructuredAnchor(
  memory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  return hasExactReferencedProposalAnchor(memory, inputById)
    || hasDestinationQualifierClosureProof(memory, inputById);
}

function closeCoepisodicDestinationQualifiers(
  memories: ExtractedMemory[],
  repairs: Map<string, number>,
  inputById: Map<string, CockpitReconciliationInput>,
): ExtractedMemory[] {
  return memories.map((memory) => {
    const metadata = metadataOf(memory);
    const relation = stringValue(metadata.relation);
    if (stringValue(metadata.domain) !== "navigation"
      || stringValue(metadata.slot) !== "destination"
      || (relation !== "asserted" && relation !== "updated")
      || qualityStatus(memory) !== "complete"
      || !hasExactReferencedProposalAnchor(memory, inputById)) return memory;
    const destination = stringValue(metadata.value);
    if (!destination || (metadata.target !== undefined && stringValue(metadata.target) !== destination)) {
      return memory;
    }
    const episode = stringValue(metadata.episode_key);
    if (!episode) return memory;
    const appointmentMatches = memories.filter((candidate) => {
      if (candidate === memory || qualityStatus(candidate) !== "complete") return false;
      const candidateMetadata = metadataOf(candidate);
      return candidate.type === memory.type
        && coepisodicScenesAreCompatible(memory, candidate)
        && stringValue(candidateMetadata.domain) === "schedule"
        && stringValue(candidateMetadata.slot) === "appointment_content"
        && stringValue(candidateMetadata.episode_key) === episode
        && stringValue(candidateMetadata.relation) === relation
        && hasExactEventProvenance(memory, candidate)
        && bindingDominates(memory, candidate)
        && bindingDominates(candidate, memory)
        && coepisodicTemporalFactsAreCompatible(memory, candidate)
        && hasExactReferencedProposalAnchor(candidate, inputById);
    });
    if (appointmentMatches.length !== 1) return memory;
    const extension = coepisodicDestinationExtension(
      destination,
      metadataOf(appointmentMatches[0]).value,
    );
    if (!extension || extension === destination || !memory.content.includes(destination)) return memory;
    let closed = cloneMemory(memory);
    const closedMetadata = metadataOf(closed);
    closedMetadata.value = extension;
    closedMetadata.target = extension;
    closed.content = closed.content.split(destination).join(extension);
    if (!hasDestinationQualifierClosureProof(closed, inputById)) return memory;
    closed = addVerifiedFactualRewriteRepair(
      closed,
      "closed_coepisodic_destination_qualifier_from_appointment_content",
    );
    bump(repairs, "closed_coepisodic_destination_qualifier_from_appointment_content");
    return closed;
  });
}

function hasExactlySameSources(left: ExtractedMemory, right: ExtractedMemory): boolean {
  const leftSources = [...new Set(left.source_message_ids)].sort();
  const rightSources = [...new Set(right.source_message_ids)].sort();
  return sameStrings(leftSources, rightSources);
}

function replaceTrailingStructuredValue(
  content: string,
  currentValue: string,
  replacement: string,
): string | undefined {
  const trailingWhitespace = content.match(/\s*$/u)?.[0] ?? "";
  const body = trailingWhitespace ? content.slice(0, -trailingWhitespace.length) : content;
  if (!body.endsWith(currentValue)) return undefined;
  return `${body.slice(0, -currentValue.length)}${replacement}${trailingWhitespace}`;
}

function boundDestinationValuesAreCompatible(
  destinationMemory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const metadata = metadataOf(destinationMemory);
  const destination = stringValue(metadata.value);
  if (!destination) return false;
  return stringArray(metadata.input_candidate_ids).every((id) => {
    const input = inputById.get(id);
    if (!input) return false;
    const inputMetadata = metadataOf(input.memory);
    if (stringValue(inputMetadata.domain) !== "navigation"
      || stringValue(inputMetadata.slot) !== "destination") return true;
    const proposed = stringValue(inputMetadata.value);
    if (!proposed || proposed === destination) return true;
    // A shorter proposal may differ only by the bounded place qualifier that
    // the preceding destination closure already proved. Unrelated bound
    // destination values remain ambiguous and block appointment enrichment.
    return coepisodicDestinationExtension(proposed, destination) === destination;
  });
}

// A complete appointment_content proposal can occasionally retain only the
// activity while a sibling navigation.destination proposal retains the exact
// location from the same source event. Close that structured evidence gap only
// when there is exactly one complete destination with identical source,
// episode, relation, person/scope and event time bindings. The assembler joins
// two already-modelled slot values; it does not parse or invent a place or an
// activity from source prose. Requiring the rendered memory to end in the old
// structured value also makes the content/value/target rewrite fail closed.
function closeCoepisodicAppointmentContentLocations(
  memories: ExtractedMemory[],
  repairs: Map<string, number>,
  inputById: Map<string, CockpitReconciliationInput>,
): ExtractedMemory[] {
  return memories.map((memory) => {
    const metadata = metadataOf(memory);
    const relation = stringValue(metadata.relation);
    if (stringValue(metadata.domain) !== "schedule"
      || stringValue(metadata.slot) !== "appointment_content"
      || (relation !== "asserted" && relation !== "updated")
      || qualityStatus(memory) !== "complete"
      || !hasExactReferencedProposalAnchor(memory, inputById)) return memory;
    const activity = stringValue(metadata.value);
    if (!activity || (metadata.target !== undefined && stringValue(metadata.target) !== activity)) {
      return memory;
    }
    const episode = stringValue(metadata.episode_key);
    if (!episode) return memory;
    const destinationMatches = memories.filter((candidate) => {
      if (candidate === memory || qualityStatus(candidate) !== "complete") return false;
      const candidateMetadata = metadataOf(candidate);
      const destination = stringValue(candidateMetadata.value);
      if (!destination
        || (candidateMetadata.target !== undefined
          && stringValue(candidateMetadata.target) !== destination)) return false;
      return stringValue(candidateMetadata.domain) === "navigation"
        && stringValue(candidateMetadata.slot) === "destination"
        && candidate.type === memory.type
        && coepisodicScenesAreCompatible(memory, candidate)
        && stringValue(candidateMetadata.episode_key) === episode
        && stringValue(candidateMetadata.relation) === relation
        && hasExactEventProvenance(memory, candidate)
        && bindingDominates(memory, candidate)
        && bindingDominates(candidate, memory)
        && coepisodicTemporalFactsAreCompatible(memory, candidate)
        && hasImmutableStructuredAnchor(candidate, inputById)
        && boundDestinationValuesAreCompatible(candidate, inputById);
    });
    if (destinationMatches.length !== 1) return memory;
    const destination = stringValue(metadataOf(destinationMatches[0]).value);
    const normalizedActivity = normalizedFactText(activity);
    const normalizedDestination = normalizedFactText(destination);
    if (!destination || !normalizedActivity || !normalizedDestination
      || normalizedActivity.includes(normalizedDestination)) return memory;
    const completed = `在${destination}进行${activity}`;
    const completedContent = replaceTrailingStructuredValue(memory.content, activity, completed);
    if (!completedContent) return memory;
    let closed = cloneMemory(memory);
    const closedMetadata = metadataOf(closed);
    closedMetadata.value = completed;
    closedMetadata.target = completed;
    closed.content = completedContent;
    const exactAtomicRewriteSupport = stringArray(closedMetadata.input_candidate_ids)
      .some((id) => id.startsWith("atomic:")
        && referencedCandidateCoversOutput(id, closed, inputById));
    if (!exactAtomicRewriteSupport) return memory;
    closed = addVerifiedFactualRewriteRepair(
      closed,
      "closed_coepisodic_appointment_content_location_from_destination",
    );
    bump(repairs, "closed_coepisodic_appointment_content_location_from_destination");
    return closed;
  });
}

const UNSAFE_APPOINTMENT_ALIAS_TOKENS = [
  "取消",
  "撤销",
  "不要",
  "不用",
  "无需",
  "不再",
  "不进行",
  "停止",
  "暂停",
  "禁止",
  "避免",
  "改为",
  "改成",
  "换成",
  "替换",
  "而不是",
  "以及",
  "并且",
  "同时",
  "然后",
  "另外",
  "还有",
  "而且",
  "再加",
  "cancel",
  "revoke",
  "donot",
  "dont",
  "stop",
  "pause",
  "avoid",
  "instead",
  "replace",
  "change",
];

const SAFE_APPOINTMENT_LOCATION_WRAPPER = new RegExp(
  String.raw`^(?:在|于|到|去|前往)?(?:地下停车场|地上停车场|地下车库|停车场|地库|园区内|院内|店内|馆内|楼内|站内|机场内|景区内|商场内|医院内|学校内|公司内|小区内|门口|入口|出口|正门|侧门|附近|周边|旁边|对面|路口|内|里)?(?:的|进行|做|办理|安排|预约|参加|召开|完成)?$`,
  "u",
);

const APPOINTMENT_LOCATION_SIGNAL = new RegExp(
  String.raw`(?:^在|^于|^到|^去|^前往|(?:地下停车场|地上停车场|地下车库|停车场|地库|园区内|院内|店内|馆内|楼内|站内|机场内|景区内|商场内|医院内|学校内|公司内|小区内|门口|入口|出口|正门|侧门|附近|周边|旁边|对面|路口|内|里)(?:的|进行|做|办理|安排|预约|参加|召开|完成)?$)`,
  "u",
);

function hasUnsafeAppointmentAliasToken(text: string): boolean {
  return UNSAFE_APPOINTMENT_ALIAS_TOKENS.some((token) => text.includes(token));
}

function appointmentContainmentPair(
  left: ExtractedMemory,
  right: ExtractedMemory,
): { rich: ExtractedMemory; coarse: ExtractedMemory; prefix: string } | undefined {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  if (stringValue(leftMetadata.domain) !== "schedule"
    || stringValue(leftMetadata.slot) !== "appointment_content"
    || stringValue(rightMetadata.domain) !== "schedule"
    || stringValue(rightMetadata.slot) !== "appointment_content") return undefined;
  const leftText = normalizedFactText(leftMetadata.value);
  const rightText = normalizedFactText(rightMetadata.value);
  if (!leftText || !rightText || Math.min([...leftText].length, [...rightText].length) < 4) {
    return undefined;
  }
  const rich = leftText.length >= rightText.length ? left : right;
  const coarse = rich === left ? right : left;
  const richText = rich === left ? leftText : rightText;
  const coarseText = rich === left ? rightText : leftText;
  if (richText === coarseText
    || !richText.endsWith(coarseText)
    || hasUnsafeAppointmentAliasToken(richText)) return undefined;
  return { rich, coarse, prefix: richText.slice(0, -coarseText.length) };
}

function compatibleCrossProposalValue(left: ExtractedMemory, right: ExtractedMemory): boolean {
  if (comparableValue(metadataOf(left).value) === comparableValue(metadataOf(right).value)) {
    return true;
  }
  const pair = appointmentContainmentPair(left, right);
  return Boolean(pair && APPOINTMENT_LOCATION_SIGNAL.test(pair.prefix));
}

function hasUniqueExactDestinationProof(
  left: ExtractedMemory,
  right: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const pair = appointmentContainmentPair(left, right);
  if (!pair) return false;
  const richMetadata = metadataOf(pair.rich);
  const allowedEpisodeKeys = new Set([left, right]
    .map((memory) => stringValue(metadataOf(memory).episode_key))
    .filter((value): value is string => Boolean(value)));
  if (allowedEpisodeKeys.size === 0) return false;
  const candidates = [...inputById.values()].filter((candidate) => {
    if (!candidate.id.startsWith("atomic:") || qualityStatus(candidate.memory) !== "complete") {
      return false;
    }
    const candidateMetadata = metadataOf(candidate.memory);
    if (!normalizedFactText(candidateMetadata.value)
      || candidate.memory.type !== pair.rich.type
      || pair.coarse.type !== pair.rich.type
      || !coepisodicScenesAreCompatible(candidate.memory, pair.rich)
      || !coepisodicScenesAreCompatible(candidate.memory, pair.coarse)
      || stringValue(candidateMetadata.domain) !== "navigation"
      || stringValue(candidateMetadata.slot) !== "destination"
      || !allowedEpisodeKeys.has(stringValue(candidateMetadata.episode_key) ?? "")
      || comparableValue(candidateMetadata.relation) !== comparableValue(richMetadata.relation)
      || !hasExactEventProvenance(candidate.memory, pair.rich)
      || !hasExactEventProvenance(candidate.memory, pair.coarse)
      || !bindingDominates(candidate.memory, pair.rich)
      || !bindingDominates(pair.rich, candidate.memory)
      || (candidateMetadata.target !== undefined
        && comparableValue(candidateMetadata.target)
          !== comparableValue(candidateMetadata.value))
      || !boundDestinationValuesAreCompatible(candidate.memory, inputById)) return false;
    return coepisodicTemporalFactsAreCompatible(candidate.memory, pair.rich)
      && coepisodicTemporalFactsAreCompatible(candidate.memory, pair.coarse);
  });
  if (candidates.length !== 1) return false;
  const destination = normalizedFactText(metadataOf(candidates[0].memory).value);
  if (!destination) return false;
  const occurrence = pair.prefix.indexOf(destination);
  if (occurrence < 0 || occurrence !== pair.prefix.lastIndexOf(destination)) return false;
  const before = pair.prefix.slice(0, occurrence);
  const after = pair.prefix.slice(occurrence + destination.length);
  const wrapper = `${before}${after}`;
  return SAFE_APPOINTMENT_LOCATION_WRAPPER.test(wrapper)
    && (APPOINTMENT_LOCATION_SIGNAL.test(before)
      || APPOINTMENT_LOCATION_SIGNAL.test(after)
      || APPOINTMENT_LOCATION_SIGNAL.test(destination));
}

function compatibleSemanticEventValue(
  left: ExtractedMemory,
  right: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  const selfConsistent = [leftMetadata, rightMetadata].every((metadata) =>
    metadata.target === undefined
      || comparableValue(metadata.target) === comparableValue(metadata.value)
  );
  if (!selfConsistent) return false;
  if (comparableValue(leftMetadata.value) === comparableValue(rightMetadata.value)) return true;

  if (exactControlledTemporalRepresentationAlias(left, right, inputById)) return true;

  // A same-episode containment alias is safe only across the independent
  // full-context and atomic proposals for the exact same evidence event. Two
  // atomic (or two primary) rows can be distinct user instructions even when
  // one happens to contain the other, so they remain separate.
  const leftClass = exactProposalClass(left, inputById);
  const rightClass = exactProposalClass(right, inputById);
  if (!leftClass || !rightClass || leftClass === rightClass
    || !hasExactEventProvenance(left, right)
    || !compatibleCrossProposalValue(left, right)
    || !hasUniqueExactDestinationProof(left, right, inputById)) return false;

  return true;
}

/**
 * Prove a narrow natural-language/ISO representation alias without parsing the
 * natural-language value. Both proposals must independently carry the exact
 * same structured event instant and every other persisted binding must agree.
 */
function exactControlledTemporalRepresentationAlias(
  left: ExtractedMemory,
  right: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  const leftClass = exactProposalClass(left, inputById);
  const rightClass = exactProposalClass(right, inputById);
  if (!leftClass || !rightClass || leftClass === rightClass
    || !hasExactEventProvenance(left, right)
    || !isControlledCockpitEventTimeSlot(leftMetadata.domain, leftMetadata.slot)
    || comparableValue(leftMetadata.domain) !== comparableValue(rightMetadata.domain)
    || comparableValue(leftMetadata.slot) !== comparableValue(rightMetadata.slot)
    || !(bindingDominates(left, right) && bindingDominates(right, left))) return false;
  const leftInstant = strictZonedIsoInstant(leftMetadata.activity_start_time);
  const rightInstant = strictZonedIsoInstant(rightMetadata.activity_start_time);
  if (leftInstant === undefined || rightInstant === undefined || leftInstant !== rightInstant) {
    return false;
  }
  const leftValueInstant = strictZonedIsoInstant(leftMetadata.value);
  const rightValueInstant = strictZonedIsoInstant(rightMetadata.value);
  // This path proves one natural-language/ISO representation pair. Exactly
  // one raw value must be the canonical zoned timestamp, and it must agree
  // with both independently supplied structured event-time anchors.
  if ((leftValueInstant === undefined) === (rightValueInstant === undefined)) return false;
  const canonicalValueInstant = leftValueInstant ?? rightValueInstant;
  if (canonicalValueInstant !== leftInstant) return false;
  if (!sameStringSet(
    stringArray(leftMetadata.source_session_ids),
    stringArray(rightMetadata.source_session_ids),
  )) return false;
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
    "valid_from",
    "valid_to",
    "activity_end_time",
    "condition",
    "trigger",
    "timezone",
    "time_precision",
    "temporal_status",
    "mentioned_at",
    "source_session_id",
  ].every((key) =>
    comparableValue(leftMetadata[key]) === comparableValue(rightMetadata[key])
  ) && sameStringSet(
    stringArray(leftMetadata.supersedes),
    stringArray(rightMetadata.supersedes),
  );
}

function compatibleActivityStartRepresentation(
  left: Record<string, unknown>,
  right: Record<string, unknown>,
): boolean {
  if (comparableValue(left.activity_start_time) === comparableValue(right.activity_start_time)) {
    return true;
  }
  const leftInstant = strictZonedIsoInstant(left.activity_start_time);
  const rightInstant = strictZonedIsoInstant(right.activity_start_time);
  return leftInstant !== undefined && rightInstant !== undefined && leftInstant === rightInstant;
}

function allReferencesHaveImmutableOriginSupport(
  row: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const metadata = metadataOf(row);
  const canonicalized = new Set(stringArray(metadata.canonicalized_input_candidate_ids));
  const references = stringArray(metadata.input_candidate_ids);
  return references.length > 0 && references.every((reference) => {
    const candidate = inputById.get(reference);
    if (!candidate) return false;
    if (reference.startsWith("atomic:")) {
      return qualityStatus(candidate.memory) === "complete"
        && referencedCandidateCoversOutput(reference, row, inputById);
    }
    if (reference.startsWith("primary:")) {
      return qualityStatus(candidate.memory) === "complete"
        && hasExactPrimaryOutputSupport(candidate, row);
    }
    return reference.startsWith("coverage:")
      && candidateCoversOutput(candidate, row, canonicalized.has(reference), inputById);
  });
}

function sameCrossProposalEpisodeAlias(
  left: ExtractedMemory,
  right: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  if (qualityStatus(left) !== "complete"
    || qualityStatus(right) !== "complete"
    || !hasExactEventProvenance(left, right)
    || left.type !== right.type
    || left.scene_name !== right.scene_name
    || !allReferencesHaveImmutableOriginSupport(left, inputById)
    || !allReferencesHaveImmutableOriginSupport(right, inputById)) return false;
  const leftClass = exactProposalClass(left, inputById);
  const rightClass = exactProposalClass(right, inputById);
  if (!leftClass || !rightClass || leftClass === rightClass) return false;
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  const leftEpisode = stringValue(leftMetadata.episode_key);
  const rightEpisode = stringValue(rightMetadata.episode_key);
  const leftStart = stringValue(leftMetadata.valid_from);
  const rightStart = stringValue(rightMetadata.valid_from);
  const temporalRepresentationAlias = exactControlledTemporalRepresentationAlias(
    left,
    right,
    inputById,
  );
  if (!leftEpisode || !rightEpisode || leftEpisode === rightEpisode) return false;
  if ((!leftStart || leftStart !== rightStart) && !temporalRepresentationAlias) return false;
  for (const key of [
    "domain",
    "slot",
    "state_key",
    "unit",
    "constraint_target",
    "state_qualifier",
    "relation",
    "action_status",
    "record_kind",
    "valid_to",
    "activity_end_time",
    "condition",
    "trigger",
    "timezone",
    "time_precision",
    "temporal_status",
    "mentioned_at",
    "source_session_id",
  ]) {
    if (comparableValue(leftMetadata[key]) !== comparableValue(rightMetadata[key])) return false;
  }
  if (!compatibleActivityStartRepresentation(leftMetadata, rightMetadata)) return false;
  if (!sameStringSet(
    stringArray(leftMetadata.supersedes),
    stringArray(rightMetadata.supersedes),
  ) || !compatibleSemanticEventValue(left, right, inputById)) return false;
  return bindingDominates(left, right) || bindingDominates(right, left);
}

function mergeCrossProposalEpisodeAlias(
  left: ExtractedMemory,
  right: ExtractedMemory,
  repairs: Map<string, number>,
  inputById: Map<string, CockpitReconciliationInput>,
): ExtractedMemory {
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  const leftClass = exactProposalClass(left, inputById);
  const primary = leftClass === "primary" ? left : right;
  const primaryEpisode = stringValue(metadataOf(primary).episode_key);
  const leftTemporalCanonical = isControlledCockpitEventTimeSlot(
    leftMetadata.domain,
    leftMetadata.slot,
  ) && strictZonedIsoInstant(leftMetadata.value) !== undefined ? 1 : 0;
  const rightTemporalCanonical = isControlledCockpitEventTimeSlot(
    rightMetadata.domain,
    rightMetadata.slot,
  ) && strictZonedIsoInstant(rightMetadata.value) !== undefined ? 1 : 0;
  const leftValueLength = [...(normalizedFactText(leftMetadata.value) ?? "")].length;
  const rightValueLength = [...(normalizedFactText(rightMetadata.value) ?? "")].length;
  const leftRank = [
    bindingSpecificity(left),
    leftTemporalCanonical,
    leftValueLength,
    leftClass === "primary" ? 1 : 0,
  ];
  const rightRank = [
    bindingSpecificity(right),
    rightTemporalCanonical,
    rightValueLength,
    exactProposalClass(right, inputById) === "primary" ? 1 : 0,
  ];
  const winner = compareRank(rightRank, leftRank) > 0 ? right : left;
  let merged = cloneMemory(winner);
  const other = winner === left ? right : left;
  const metadata = metadataOf(merged);
  const otherMetadata = metadataOf(other);
  const allSources = [...new Set([...merged.source_message_ids, ...other.source_message_ids])];
  const allReferences = [...new Set([
    ...stringArray(metadata.input_candidate_ids),
    ...stringArray(otherMetadata.input_candidate_ids),
  ])];
  const allCanonicalized = [...new Set([
    ...stringArray(metadata.canonicalized_input_candidate_ids),
    ...stringArray(otherMetadata.canonicalized_input_candidate_ids),
  ])].filter((id) => allReferences.includes(id));
  const allEvidenceRoles = [...new Set([
    ...stringArray(metadata.evidence_roles),
    ...stringArray(otherMetadata.evidence_roles),
  ])];
  const allSourceSessions = [...new Set([
    ...stringArray(metadata.source_session_ids),
    ...stringArray(otherMetadata.source_session_ids),
  ])];
  merged.source_message_ids = allSources;
  metadata.source_message_ids = allSources;
  metadata.input_candidate_ids = allReferences;
  if (primaryEpisode) metadata.episode_key = primaryEpisode;
  if (allCanonicalized.length > 0) metadata.canonicalized_input_candidate_ids = allCanonicalized;
  else delete metadata.canonicalized_input_candidate_ids;
  if (allEvidenceRoles.length > 0) metadata.evidence_roles = allEvidenceRoles;
  if (allSourceSessions.length > 0) metadata.source_session_ids = allSourceSessions;
  const repair = exactControlledTemporalRepresentationAlias(left, right, inputById)
    && comparableValue(leftMetadata.value) !== comparableValue(rightMetadata.value)
    ? "coalesced_cross_proposal_temporal_representation_alias"
    : "coalesced_cross_proposal_episode_alias";
  merged = addVerifiedFactualRewriteRepair(merged, repair);
  bump(repairs, repair);
  return merged;
}

function coalesceCrossProposalEpisodeAliases(
  memories: ExtractedMemory[],
  repairs: Map<string, number>,
  inputById: Map<string, CockpitReconciliationInput>,
): ExtractedMemory[] {
  const consumed = new Set<number>();
  const output: Array<{ memory: ExtractedMemory; index: number }> = [];
  for (const [index, memory] of memories.entries()) {
    if (consumed.has(index)) continue;
    const matches = memories
      .map((candidate, candidateIndex) => ({ candidate, candidateIndex }))
      .filter(({ candidate, candidateIndex }) => candidateIndex !== index
        && !consumed.has(candidateIndex)
        && sameCrossProposalEpisodeAlias(memory, candidate, inputById));
    if (matches.length !== 1) {
      output.push({ memory, index });
      continue;
    }
    const match = matches[0];
    const reciprocalMatches = memories
      .map((candidate, candidateIndex) => ({ candidate, candidateIndex }))
      .filter(({ candidate, candidateIndex }) => candidateIndex !== match.candidateIndex
        && !consumed.has(candidateIndex)
        && sameCrossProposalEpisodeAlias(match.candidate, candidate, inputById));
    if (reciprocalMatches.length !== 1 || reciprocalMatches[0].candidateIndex !== index) {
      output.push({ memory, index });
      continue;
    }
    consumed.add(index);
    consumed.add(match.candidateIndex);
    output.push({
      memory: mergeCrossProposalEpisodeAlias(memory, match.candidate, repairs, inputById),
      index: Math.min(index, match.candidateIndex),
    });
  }
  return output.sort((left, right) => left.index - right.index).map((entry) => entry.memory);
}

function compatibleSemanticStateKey(
  left: ExtractedMemory,
  right: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const leftStateKey = stringValue(metadataOf(left).state_key);
  const rightStateKey = stringValue(metadataOf(right).state_key);
  if (leftStateKey === rightStateKey) return Boolean(leftStateKey);
  if (!leftStateKey || !rightStateKey) return false;
  // `bindCandidateIds` may safely co-bind an exact primary+atomic duplicate
  // before this phase. Preserve the cross-proposal proof in that case: one
  // side must still contain primary evidence while the other is atomic-only.
  // Two atomic-only (or two primary-bearing) states can be independent user
  // instructions and may never alias merely through a generic binding.
  if (!hasExactEventProvenance(left, right)) return false;
  if (!hasOneWayPrimaryToAtomicEvidence(left, right, inputById)) return false;
  const leftDominates = bindingDominates(left, right);
  const rightDominates = bindingDominates(right, left);
  // A different state key is admissible only for a strict generic-to-specific
  // person/scope refinement. Equivalent or conflicting bindings cannot prove
  // that two differently keyed states are aliases.
  return leftDominates !== rightDominates;
}

function sameCompatibleSemanticEvent(
  left: ExtractedMemory,
  right: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  if (qualityStatus(left) !== "complete" || qualityStatus(right) !== "complete") return false;
  if (!hasSharedSource(left, right)) return false;
  if (left.type !== right.type || left.scene_name !== right.scene_name) return false;
  const leftMetadata = metadataOf(left);
  const rightMetadata = metadataOf(right);
  for (const key of [
    "domain",
    "slot",
    "episode_key",
    "unit",
    "constraint_target",
    "state_qualifier",
    "relation",
    "action_status",
    "record_kind",
  ]) {
    if (comparableValue(leftMetadata[key]) !== comparableValue(rightMetadata[key])) return false;
  }
  if (!compatibleSemanticStateKey(left, right, inputById)
    || !sameStringSet(
      stringArray(leftMetadata.supersedes),
      stringArray(rightMetadata.supersedes),
    )
    || !compatibleSemanticEventValue(left, right, inputById)) return false;
  for (const key of [
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
  ]) {
    const leftValue = leftMetadata[key];
    const rightValue = rightMetadata[key];
    if (leftValue !== undefined && rightValue !== undefined
      && comparableValue(leftValue) !== comparableValue(rightValue)) return false;
  }
  return true;
}

function bindingSpecificity(memory: ExtractedMemory): number {
  const metadata = metadataOf(memory);
  return ["subject", "occupant_scope", "vehicle_scope", "seat_zone"]
    .filter((key) => canonicalBindingValue(key, metadata[key]) !== undefined).length;
}

function temporalSpecificity(memory: ExtractedMemory): number {
  const metadata = metadataOf(memory);
  return [
    "valid_from",
    "valid_to",
    "activity_start_time",
    "activity_end_time",
    "condition",
    "trigger",
  ].filter((key) => metadata[key] !== undefined).length;
}

function semanticValueSpecificity(memory: ExtractedMemory): number {
  return [...(normalizedFactText(metadataOf(memory).value) ?? "")].length;
}

function compareRank(left: number[], right: number[]): number {
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const difference = (left[index] ?? 0) - (right[index] ?? 0);
    if (difference !== 0) return difference;
  }
  return 0;
}

function mergeCompatibleSemanticRows(
  left: ExtractedMemory,
  right: ExtractedMemory,
  repairs: Map<string, number>,
  inputById: Map<string, CockpitReconciliationInput>,
): ExtractedMemory | undefined {
  if (!sameCompatibleSemanticEvent(left, right, inputById)) return undefined;
  const leftDominates = bindingDominates(left, right);
  const rightDominates = bindingDominates(right, left);
  if (!leftDominates && !rightDominates) return undefined;

  let winner: ExtractedMemory;
  let other: ExtractedMemory;
  if (leftDominates !== rightDominates) {
    [winner, other] = leftDominates ? [left, right] : [right, left];
  } else {
    const leftRank = [
      bindingSpecificity(left),
      semanticValueSpecificity(left),
      temporalSpecificity(left),
      atomicReferenceCount(left),
      left.source_message_ids.length,
    ];
    const rightRank = [
      bindingSpecificity(right),
      semanticValueSpecificity(right),
      temporalSpecificity(right),
      atomicReferenceCount(right),
      right.source_message_ids.length,
    ];
    const useRight = compareRank(rightRank, leftRank) > 0;
    [winner, other] = useRight ? [right, left] : [left, right];
  }

  let merged = cloneMemory(winner);
  const metadata = metadataOf(merged);
  const otherMetadata = metadataOf(other);
  for (const key of [
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
  ]) {
    if (metadata[key] === undefined && otherMetadata[key] !== undefined) metadata[key] = otherMetadata[key];
  }
  const allSources = [...new Set([...merged.source_message_ids, ...other.source_message_ids])];
  const allReferences = [...new Set([
    ...stringArray(metadata.input_candidate_ids),
    ...stringArray(otherMetadata.input_candidate_ids),
  ])];
  const allCanonicalized = [...new Set([
    ...stringArray(metadata.canonicalized_input_candidate_ids),
    ...stringArray(otherMetadata.canonicalized_input_candidate_ids),
  ])].filter((id) => allReferences.includes(id));
  const allEvidenceRoles = [...new Set([
    ...stringArray(metadata.evidence_roles),
    ...stringArray(otherMetadata.evidence_roles),
  ])];
  const allSourceSessions = [...new Set([
    ...stringArray(metadata.source_session_ids),
    ...stringArray(otherMetadata.source_session_ids),
  ])];
  merged.source_message_ids = allSources;
  metadata.source_message_ids = allSources;
  metadata.input_candidate_ids = allReferences;
  if (allCanonicalized.length > 0) metadata.canonicalized_input_candidate_ids = allCanonicalized;
  else delete metadata.canonicalized_input_candidate_ids;
  if (allEvidenceRoles.length > 0) metadata.evidence_roles = allEvidenceRoles;
  if (allSourceSessions.length > 0) metadata.source_session_ids = allSourceSessions;
  // `sameCompatibleSemanticEvent` requires exact predecessor sets. Never
  // union two transition chains merely because their other fields align.
  const supersedes = stringArray(metadata.supersedes);
  if (supersedes.length > 0) metadata.supersedes = supersedes;
  const repair = comparableValue(metadataOf(left).value) === comparableValue(metadataOf(right).value)
    ? "coalesced_compatible_semantic_binding"
    : "coalesced_same_episode_cross_proposal_value_alias";
  merged = addVerifiedFactualRewriteRepair(merged, repair);
  if (!preservesEveryReferencedObligation(merged, inputById)) return undefined;
  bump(repairs, repair);
  return merged;
}

function coalesceCompatibleSemanticBindings(
  memories: ExtractedMemory[],
  repairs: Map<string, number>,
  inputById: Map<string, CockpitReconciliationInput>,
): ExtractedMemory[] {
  const compatible = (left: ExtractedMemory, right: ExtractedMemory): boolean =>
    sameCompatibleSemanticEvent(left, right, inputById)
      && (bindingDominates(left, right) || bindingDominates(right, left));
  const matches = memories.map((memory, index) => memories
    .map((candidate, candidateIndex) => ({ candidate, candidateIndex }))
    .filter(({ candidate, candidateIndex }) => candidateIndex !== index
      && compatible(memory, candidate))
    .map(({ candidateIndex }) => candidateIndex));

  // A generic subject can be a safe alias only when it has one unique concrete
  // owner. If the same event/value can bind to two different people or seat
  // scopes, retaining a complete generic row would let retrieval attribute it
  // arbitrarily even though no merge occurred. Mark the construction partial
  // so the public gate retries/fails closed instead of depending on row order.
  const bindingSignature = (memory: ExtractedMemory): string => JSON.stringify(
    ["subject", "occupant_scope", "vehicle_scope", "seat_zone"]
      .map((key) => canonicalBindingValue(key, metadataOf(memory)[key]) ?? null),
  );
  const ambiguousGeneric = new Set<number>();
  for (const [index, memory] of memories.entries()) {
    const strictDominators = matches[index]
      .map((candidateIndex) => memories[candidateIndex])
      .filter((candidate) => bindingDominates(candidate, memory)
        && !bindingDominates(memory, candidate));
    if (new Set(strictDominators.map(bindingSignature)).size > 1) {
      ambiguousGeneric.add(index);
    }
  }

  const consumed = new Set<number>();
  const output: Array<{ memory: ExtractedMemory; index: number }> = [];
  for (const [index, original] of memories.entries()) {
    if (consumed.has(index)) continue;
    let memory = original;
    if (ambiguousGeneric.has(index)) {
      const flagged = addQualityIssue(
        memory,
        "ambiguous_semantic_binding_alias",
        "preserved_ambiguous_semantic_binding_alias",
      );
      memory = flagged.memory;
      if (flagged.added) bump(repairs, "preserved_ambiguous_semantic_binding_alias");
    }
    const candidates = matches[index].filter((candidateIndex) => !consumed.has(candidateIndex));
    if (ambiguousGeneric.has(index) || candidates.length !== 1) {
      output.push({ memory, index });
      continue;
    }
    const candidateIndex = candidates[0];
    const reciprocal = matches[candidateIndex]
      .filter((reciprocalIndex) => !consumed.has(reciprocalIndex));
    if (ambiguousGeneric.has(candidateIndex)
      || reciprocal.length !== 1
      || reciprocal[0] !== index) {
      output.push({ memory, index });
      continue;
    }
    const merged = mergeCompatibleSemanticRows(
      memory,
      memories[candidateIndex],
      repairs,
      inputById,
    );
    if (!merged) {
      output.push({ memory, index });
      continue;
    }
    consumed.add(index);
    consumed.add(candidateIndex);
    output.push({ memory: merged, index: Math.min(index, candidateIndex) });
  }
  return output.sort((left, right) => left.index - right.index).map((entry) => entry.memory);
}

function processPriors(
  memories: ExtractedMemory[],
  priors: CockpitReconciliationPriorMemory[],
  inputById: Map<string, CockpitReconciliationInput>,
  resolved: Set<string>,
  repairs: Map<string, number>,
): ExtractedMemory[] {
  const output: ExtractedMemory[] = [];
  for (const memory of memories) {
    if (hasBlockingQualityIssue(memory)) {
      output.push(memory);
      continue;
    }
    const result = resolveAgainstLivePrior(memory, priors, inputById, repairs);
    result.resolvedCandidateIds.forEach((id) => resolved.add(id));
    if (result.memory) output.push(result.memory);
  }
  return output;
}

function referencedLivePriors(
  memory: ExtractedMemory,
  priors: CockpitReconciliationPriorMemory[],
): CockpitReconciliationPriorMemory[] | undefined {
  const references = stringArray(metadataOf(memory).supersedes);
  if (references.length === 0) return undefined;
  const targets = new Map<string, CockpitReconciliationPriorMemory>();
  for (const reference of references) {
    const matches = priors.filter((prior) =>
      prior.record_id === reference || stringValue(prior.metadata.episode_key) === reference
    );
    if (matches.length === 0) return undefined;
    matches.forEach((prior) => targets.set(prior.record_id, prior));
  }
  return [...targets.values()];
}

function hasSpecificAtomicTransitionSupport(
  memory: ExtractedMemory,
  target: CockpitReconciliationPriorMemory,
  compositeCandidateId: string,
  compositeMemory: ExtractedMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  const metadata = metadataOf(memory);
  const targetStateKey = stringValue(target.metadata.state_key);
  const targetEpisodeKey = stringValue(target.metadata.episode_key);
  const relation = stringValue(metadata.relation);
  if (qualityStatus(memory) !== "complete"
    || !cockpitPriorMatchesTransitionIdentity(target, memory)
    || !targetStateKey || !targetEpisodeKey
    || stringValue(metadata.state_key) !== targetStateKey
    || stringValue(metadata.episode_key) !== targetEpisodeKey
    || !sameStrings(stringArray(metadata.supersedes), [target.record_id])
    || (relation !== "updated" && relation !== "cancelled" && relation !== "negated")) {
    return false;
  }
  return stringArray(metadata.input_candidate_ids).some((id) => {
    if (id === compositeCandidateId || !id.startsWith("atomic:")) return false;
    const candidate = inputById.get(id);
    if (!candidate || candidate.memory.type !== compositeMemory.type) return false;
    const candidateMetadata = metadataOf(candidate.memory);
    const candidateRelation = stringValue(candidateMetadata.relation);
    const candidateQuality = qualityStatus(candidate.memory);
    const repairsMissingPredecessor = candidateQuality === "partial"
      && sameStrings(qualityIssues(candidate.memory), ["missing_supersedes"])
      && stringArray(candidateMetadata.supersedes).length === 0;
    if ((candidateQuality !== "complete" && !repairsMissingPredecessor)
      || candidate.memory.type !== memory.type
      || candidate.memory.scene_name !== memory.scene_name
      || !sameTransitionEvidence(candidate.memory, compositeMemory)
      || !hasExactEventProvenance(candidate.memory, memory)
      || !(bindingDominates(candidate.memory, memory)
        && bindingDominates(memory, candidate.memory))
      || !EXACT_ATOMIC_NON_VALUE_FACT_KEYS.every((key) =>
        key === "relation" || key === "episode_key"
          || comparableValue(candidateMetadata[key]) === comparableValue(metadata[key])
      )
      || comparableValue(candidateMetadata.value) !== comparableValue(metadata.value)
      || comparableValue(candidateMetadata.target) !== comparableValue(metadata.target)) {
      return false;
    }
    const relationIsProven = candidateRelation === relation
      || (candidateRelation === "asserted" && relation === "updated");
    if (!relationIsProven) return false;
    const candidateEpisode = stringValue(candidateMetadata.episode_key);
    const outputEpisode = stringValue(metadata.episode_key);
    const episodeIsProven = candidateEpisode === outputEpisode
      || (repairsMissingPredecessor
        && outputEpisode === stringValue(target.metadata.episode_key))
      || (candidateRelation === "asserted"
        && relation === "updated"
        && hasPrimaryUpdatedReplacementSupport(candidate.memory, target, inputById));
    if (!episodeIsProven) return false;
    const predecessors = stringArray(candidateMetadata.supersedes);
    const targetEpisodeReference = stringValue(target.metadata.episode_key);
    return predecessors.length === 0
      ? candidateRelation === "asserted" || repairsMissingPredecessor
      : predecessors.length === 1
        && (predecessors[0] === target.record_id
          || predecessors[0] === targetEpisodeReference);
  });
}

function isSpecificUpdatedReplacementSupport(
  candidate: CockpitReconciliationInput,
  memory: ExtractedMemory,
  target: CockpitReconciliationPriorMemory,
  inputById: Map<string, CockpitReconciliationInput>,
): boolean {
  return stringValue(metadataOf(memory).relation) === "updated"
    && hasExactCompositeEventEvidence(candidate.memory, memory)
    && hasSpecificAtomicTransitionSupport(
      memory,
      target,
      candidate.id,
      candidate.memory,
      inputById,
    );
}

function hasExactCompositeEventEvidence(
  composite: ExtractedMemory,
  specific: ExtractedMemory,
): boolean {
  const compositeMetadata = metadataOf(composite);
  const specificMetadata = metadataOf(specific);
  const episode = stringValue(compositeMetadata.episode_key);
  return Boolean(episode)
    && composite.type === specific.type
    && coepisodicScenesAreCompatible(composite, specific)
    && episode === stringValue(specificMetadata.episode_key)
    && sameTransitionEvidence(composite, specific);
}

function hasCoherentSpecificBindings(memories: ExtractedMemory[]): boolean {
  return ["subject", "occupant_scope", "vehicle_scope", "seat_zone"].every((key) => {
    const values = new Set(memories
      .map((memory) => canonicalBindingValue(key, metadataOf(memory)[key]))
      .filter((value): value is string => Boolean(value)));
    return values.size <= 1;
  });
}

function isControlledAppointmentCompositeCandidate(
  candidate: CockpitReconciliationInput,
): boolean {
  const metadata = metadataOf(candidate.memory);
  const issues = qualityIssues(candidate.memory);
  const supersedes = stringArray(metadata.supersedes);
  const hasSupportedDefectShape = (sameStrings(issues, ["missing_supersedes"])
      && supersedes.length === 0)
    || (sameStrings(issues, ["ambiguous_transition_state"])
      && supersedes.length > 0);
  return qualityStatus(candidate.memory) === "partial"
    && candidate.memory.scene_name === "schedule"
    && stringValue(metadata.domain) === "schedule"
    && stringValue(metadata.slot) === "status"
    && stringValue(metadata.value)?.toLocaleLowerCase() === "cancelled"
    && stringValue(metadata.relation) === "cancelled"
    && hasSupportedDefectShape;
}

/**
 * Infer the exact old episode represented by a redundant schedule-status
 * cancellation emitted alongside a multi-slot replacement transaction.
 *
 * The compiler is explicitly told to emit the concrete updated slots instead
 * of an umbrella `status=cancelled` row, but providers can still return both.
 * Treating every partial status row as optional would hide real cancellations,
 * so this path is intentionally narrow.  The status row is consumed only when
 * two or more independently compiled, exact-evidence `updated` edges already
 * supersede distinct live states in one unambiguous appointment episode.  A
 * pure cancellation, a single-slot edit, mixed people/scopes, or updates to
 * multiple episodes therefore remains uncovered and fails closed.
 */
function inferCompositeScheduleReplacementTargets(
  candidate: CockpitReconciliationInput,
  memories: ExtractedMemory[],
  priors: CockpitReconciliationPriorMemory[],
  inputById: Map<string, CockpitReconciliationInput>,
): CockpitReconciliationPriorMemory[] | undefined {
  const candidateMetadata = metadataOf(candidate.memory);
  if (!isControlledAppointmentCompositeCandidate(candidate)
    || !sameStrings(qualityIssues(candidate.memory), ["missing_supersedes"])) {
    return undefined;
  }

  // If a real status predecessor exists, this may be an explicit lifecycle
  // cancellation and must be restored/validated as its own transition.
  const candidateStateKey = stringValue(candidateMetadata.state_key);
  if (!candidateStateKey || priors.some((prior) =>
    stringValue(prior.metadata.state_key) === candidateStateKey
  )) {
    return undefined;
  }

  const supported = priors.filter((target) => memories.some((memory) =>
    isSpecificUpdatedReplacementSupport(candidate, memory, target, inputById)
  ));
  const supportingMemories = memories.filter((memory) => supported.some((target) =>
    isSpecificUpdatedReplacementSupport(candidate, memory, target, inputById)
  ));
  const episodeKeys = new Set(supported
    .map((target) => stringValue(target.metadata.episode_key))
    .filter((value): value is string => Boolean(value)));
  const stateKeys = new Set(supported
    .map((target) => stringValue(target.metadata.state_key))
    .filter((value): value is string => Boolean(value)));
  const hasAppointmentState = supported.some((target) =>
    stringValue(target.metadata.domain) === "schedule"
      && ["appointment_time", "appointment_content"].includes(
        stringValue(target.metadata.slot) ?? "",
      )
  );
  if (supported.length < 2
    || episodeKeys.size !== 1
    || stateKeys.size < 2
    || !hasAppointmentState
    || !supported.every((target) =>
      isControlledAppointmentMigrationState(target.metadata)
    )
    || !hasCoherentSpecificBindings([candidate.memory, ...supportingMemories])) {
    return undefined;
  }
  return supported;
}

function resolveCompositeTransitionObligations(
  memories: ExtractedMemory[],
  inputs: CockpitReconciliationInput[],
  inputById: Map<string, CockpitReconciliationInput>,
  priors: CockpitReconciliationPriorMemory[],
  resolved: Set<string>,
  repairs: Map<string, number>,
): ExtractedMemory[] {
  let output = memories;
  for (const candidate of inputs) {
    if (!candidate.id.startsWith("atomic:")
      || !isControlledAppointmentCompositeCandidate(candidate)) {
      continue;
    }
    const explicitlyAmbiguous = sameStrings(
      qualityIssues(candidate.memory),
      ["ambiguous_transition_state"],
    );
    const inferredReplacementTargets = explicitlyAmbiguous
      ? undefined
      : inferCompositeScheduleReplacementTargets(candidate, output, priors, inputById);
    const targets = explicitlyAmbiguous
      ? referencedLivePriors(candidate.memory, priors)
      : inferredReplacementTargets;
    const targetStateKeys = new Set((targets ?? [])
      .map((prior) => stringValue(prior.metadata.state_key))
      .filter((value): value is string => Boolean(value)));
    if (!targets
      || targets.length < 2
      || targetStateKeys.size !== targets.length
      || !targets.every((target) =>
        isControlledAppointmentMigrationState(target.metadata)
      )) continue;
    if (!targets.every((target) => output.some((memory) =>
      hasExactCompositeEventEvidence(candidate.memory, memory)
        && hasSpecificAtomicTransitionSupport(
          memory,
          target,
          candidate.id,
          candidate.memory,
          inputById,
        )
    ))) {
      continue;
    }

    resolved.add(candidate.id);
    output = output.map((memory) => {
      const metadata = metadataOf(memory);
      const references = stringArray(metadata.input_candidate_ids);
      const supportsInferredReplacement = Boolean(inferredReplacementTargets?.some((target) =>
        isSpecificUpdatedReplacementSupport(candidate, memory, target, inputById)
      ));
      if (!references.includes(candidate.id) && !supportsInferredReplacement) return memory;
      let cleaned = cloneMemory(memory);
      const cleanedMetadata = metadataOf(cleaned);
      cleanedMetadata.input_candidate_ids = references.filter((id) => id !== candidate.id);
      const canonicalized = stringArray(cleanedMetadata.canonicalized_input_candidate_ids)
        .filter((id) => id !== candidate.id);
      if (canonicalized.length > 0) cleanedMetadata.canonicalized_input_candidate_ids = canonicalized;
      else delete cleanedMetadata.canonicalized_input_candidate_ids;
      cleaned = addQualityRepair(cleaned, "resolved_composite_transition_by_specific_atomic_edges");
      bump(repairs, "resolved_composite_transition_by_specific_atomic_edges");
      return cleaned;
    });
  }
  return output;
}

/** Assemble a gate-ready result while preserving every complete atomic source obligation. */
export function assembleCockpitConstructionReconciliation(params: {
  inputs: CockpitReconciliationInput[];
  reconciled: ExtractedMemory[];
  maxMemories: number;
  priorMemories?: CockpitReconciliationPriorMemory[];
}): CockpitConstructionAssemblyResult {
  const repairs = new Map<string, number>();
  const inputById = new Map(params.inputs.map((input) => [input.id, input]));
  const livePriors = cockpitLivePriorMemories(params.priorMemories ?? []);
  const resolved = new Set<string>();

  let memories = params.reconciled
    .flatMap((memory) => {
      const canonicalized = canonicalizeReconciledSlotFromAtomicEvidence(
        memory,
        inputById,
        repairs,
      );
      return canonicalized ? [canonicalized] : [];
    })
    .filter((memory) => {
      const complete = qualityStatus(memory) === "complete";
      if (!complete) bump(repairs, "dropped_incomplete_reconciliation_row");
      return complete;
    })
    .map((memory) => bindCandidateIds(memory, params.inputs, inputById, repairs))
    .filter((memory) => {
      const bound = stringArray(metadataOf(memory).input_candidate_ids).length > 0;
      if (!bound) bump(repairs, "dropped_unbound_reconciliation_row");
      return bound;
    });
  // Run evidence-safe semantic coalescence before identity ranking. Otherwise
  // a reconciler that returns both a rich primary and a coarse atomic row with
  // the same identity can lose the richer value before the alias proof runs.
  memories = coalesceCompatibleSemanticBindings(memories, repairs, inputById);
  memories = coalesceDuplicateIdentities(memories, repairs, inputById);

  const required = params.inputs.filter((input) =>
    input.id.startsWith("atomic:") && qualityStatus(input.memory) === "complete"
  );
  for (const candidate of required) {
    if (candidateIsCovered(candidate, memories, resolved, inputById)) continue;
    let fallback = bindCandidateIds(candidate.memory, params.inputs, inputById, repairs);
    fallback = addQualityRepair(fallback, "restored_complete_atomic_obligation");
    bump(repairs, "restored_complete_atomic_obligation");
    memories.push(fallback);
  }

  const recoverablePartialTransitions = params.inputs.filter((input) =>
    input.id.startsWith("atomic:") && qualityStatus(input.memory) === "partial"
  );
  for (const candidate of recoverablePartialTransitions) {
    if (candidateIsCovered(candidate, memories, resolved, inputById)) continue;
    const restored = restorePartialAtomicTransition(candidate, livePriors, repairs);
    if (!restored) continue;
    let fallback = bindCandidateIds(restored, params.inputs, inputById, repairs);
    fallback = addQualityRepair(fallback, "restored_recoverable_partial_atomic_obligation");
    bump(repairs, "restored_recoverable_partial_atomic_obligation");
    memories.push(fallback);
  }

  memories = resolveAtomicCancelReplacePairs(memories, livePriors, inputById, resolved, repairs);
  memories = resolveCrossEpisodeAtomicCancelReplaceTransactions(
    memories,
    livePriors,
    inputById,
    resolved,
    repairs,
  );
  memories = coalesceCompatibleSemanticBindings(memories, repairs, inputById);
  memories = coalesceDuplicateIdentities(memories, repairs, inputById);
  memories = coalesceCrossProposalEpisodeAliases(memories, repairs, inputById);
  memories = coalesceDuplicateIdentities(memories, repairs, inputById);
  memories = closeCoepisodicDestinationQualifiers(memories, repairs, inputById);
  memories = closeCoepisodicAppointmentContentLocations(memories, repairs, inputById);
  memories = processPriors(memories, livePriors, inputById, resolved, repairs);
  memories = resolveCompositeTransitionObligations(
    memories,
    params.inputs,
    inputById,
    livePriors,
    resolved,
    repairs,
  );

  // Advisory rows are the first safe rows to omit if a model returned more
  // outputs than the registered contract permits. Atomic obligations remain.
  if (memories.length > params.maxMemories) {
    const mandatory = memories.filter((memory) =>
      atomicReferenceCount(memory) > 0 || hasBlockingQualityIssue(memory)
    );
    const advisory = memories.filter((memory) =>
      atomicReferenceCount(memory) === 0 && !hasBlockingQualityIssue(memory)
    );
    const retainedAdvisory = advisory.slice(0, Math.max(0, params.maxMemories - mandatory.length));
    const trimmed = [...mandatory, ...retainedAdvisory];
    if (trimmed.length < memories.length) {
      bump(repairs, "trimmed_advisory_rows_above_maximum");
      memories = trimmed;
    }
  }

  // Convert the private symbol into an object-identity capability immediately
  // before returning. Removing the symbol prevents it from leaking into later
  // object spreads or persistence while the WeakSet retains proof for this
  // exact gate invocation.
  for (const memory of memories) {
    const verified = memory as AssemblerVerifiedMemory;
    if (!verified[VERIFIED_FACTUAL_REWRITE]) continue;
    delete verified[VERIFIED_FACTUAL_REWRITE];
    registerCockpitAssemblerVerifiedFactualRewrite(memory);
  }

  return {
    memories,
    resolvedCandidateIds: [...resolved],
    repairCounts: Object.fromEntries([...repairs.entries()].sort(([left], [right]) =>
      left.localeCompare(right)
    )),
  };
}
