import { describe, expect, it } from "vitest";

import type { ExtractedMemory } from "./l1-writer.js";
import {
  gateCockpitConstructionReconciliation,
  preservesCockpitEvidenceBinding,
  registerCockpitAssemblerVerifiedFactualRewrite,
} from "./cockpit-construction-reconciliation.js";

function memory(params: {
  domain?: string;
  slot: string;
  episode?: string;
  source?: string;
  status?: "complete" | "partial";
  refs?: string[];
  canonicalizedRefs?: string[];
  validFrom?: string;
  seatZone?: string;
  relation?: "asserted" | "updated" | "cancelled" | "negated";
  value?: string;
  stateQualifier?: string;
}): ExtractedMemory {
  const qualifiedSlot = params.stateQualifier
    ? `${params.slot}@${params.stateQualifier}`
    : params.slot;
  return {
    content: `${params.slot} fact`,
    type: "episodic",
    priority: 70,
    scene_name: "cockpit",
      source_message_ids: [params.source ?? "u1"],
      metadata: {
      domain: params.domain ?? "navigation",
      slot: params.slot,
      value: params.value ?? "value",
      subject: "user",
      state_key: `${params.domain ?? "navigation"}|user|car|driver|${qualifiedSlot}`,
      episode_key: params.episode ?? "route-1",
      relation: params.relation ?? "asserted",
      state_qualifier: params.stateQualifier,
      valid_from: params.validFrom,
        seat_zone: params.seatZone,
        source_message_ids: [params.source ?? "u1"],
        input_candidate_ids: params.refs,
      canonicalized_input_candidate_ids: params.canonicalizedRefs,
      construction_quality: { status: params.status ?? "complete" },
    },
  };
}

describe("cockpit construction reconciliation gate", () => {
  it("does not let reconciliation drop a source-bound named-state qualifier", () => {
    const atomic = memory({ slot: "destination", stateQualifier: "早餐地点" });
    const output = memory({ slot: "destination", refs: ["atomic:0"] });

    expect(preservesCockpitEvidenceBinding(atomic, output)).toBe(false);
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: atomic }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toContain("atomic:0");
  });

  it("keeps source-bound state qualifiers case-sensitive after NFKC", () => {
    const upper = memory({ slot: "destination", stateQualifier: "A地点" });
    const lower = memory({ slot: "destination", stateQualifier: "a地点" });

    expect(preservesCockpitEvidenceBinding(upper, lower)).toBe(false);
    expect(preservesCockpitEvidenceBinding(lower, upper)).toBe(false);
  });

  it("accepts deduplicated outputs that account for every complete input", () => {
    const primary = memory({ slot: "destination" });
    const atomic = memory({ slot: "destination", episode: "route-alias" });
    const final = memory({ slot: "destination", refs: ["primary:0", "atomic:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [
        { id: "primary:0", memory: primary },
        { id: "atomic:0", memory: atomic },
      ],
      reconciled: [final],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(true);
    expect(result.requiredCandidateIds).toEqual(["atomic:0"]);
    expect(result.coveredCandidateIds).toEqual(["primary:0", "atomic:0"]);
  });

  it.each([
    {
      label: "value and episode",
      mutate: (output: ExtractedMemory) => {
        output.metadata.value = "上海";
        output.metadata.episode_key = "route-hallucinated";
      },
    },
    {
      label: "unit",
      mutate: (output: ExtractedMemory) => {
        output.metadata.unit = "km";
      },
    },
    {
      label: "event time",
      mutate: (output: ExtractedMemory) => {
        output.metadata.activity_start_time = "2026-04-01T12:00:00+08:00";
      },
    },
    {
      label: "condition",
      mutate: (output: ExtractedMemory) => {
        output.metadata.condition = "仅在下雨时";
      },
    },
  ])("rejects a primary-only row that rewrites its $label", ({ mutate }) => {
    const primary = memory({ slot: "destination", value: "北京", episode: "route-grounded" });
    const output = memory({
      slot: "destination",
      value: "北京",
      episode: "route-grounded",
      refs: ["primary:0"],
    });
    mutate(output);
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: primary }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_row_without_factual_candidate_anchor");
  });

  it("does not accept a model-written deterministic repair label as assembler proof", () => {
    const primary = memory({ slot: "destination", value: "北京" });
    const output = memory({
      slot: "destination",
      value: "上海",
      refs: ["primary:0"],
    });
    output.metadata.construction_quality = {
      status: "complete",
      repairs: ["closed_coepisodic_destination_qualifier_from_appointment_content"],
    };
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: primary }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_row_without_factual_candidate_anchor");
  });

  it("invalidates an assembler capability when its registered row is mutated", () => {
    const obligation = memory({ domain: "navigation", slot: "destination", status: "partial" });
    delete obligation.metadata.value;
    const output = memory({
      domain: "navigation",
      slot: "destination",
      value: "北京",
      refs: ["coverage:u1:navigation:destination"],
    });
    registerCockpitAssemblerVerifiedFactualRewrite(output);
    const registered = gateCockpitConstructionReconciliation({
      inputs: [{ id: "coverage:u1:navigation:destination", memory: obligation }],
      reconciled: [output],
      maxMemories: 10,
    });
    expect(registered.accepted).toBe(true);

    output.metadata.value = "上海";
    const mutated = gateCockpitConstructionReconciliation({
      inputs: [{ id: "coverage:u1:navigation:destination", memory: obligation }],
      reconciled: [output],
      maxMemories: 10,
    });
    expect(mutated.accepted).toBe(false);
    expect(mutated.issues).toContain("reconciliation_row_without_factual_candidate_anchor");
  });

  it.each([
    { id: "primary:0", status: "complete" as const },
    { id: "coverage:u1:climate:temperature", status: "partial" as const },
  ])("rejects a model-authored cross-slot canonicalization for $id", ({ id, status }) => {
    const candidate = memory({ domain: "climate", slot: "temperature", value: "22", status });
    candidate.scene_name = "climate";
    const output = memory({
      domain: "navigation",
      slot: "destination",
      value: "上海",
      refs: [id],
      canonicalizedRefs: [id],
    });
    output.scene_name = "climate";
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id, memory: candidate }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_row_without_factual_candidate_anchor");
  });

  it("does not force an uncovered schema-complete primary proposal into final state", () => {
    const misleadingPrimary = memory({ domain: "navigation", slot: "departure_time" });
    const atomic = memory({ domain: "navigation", slot: "pickup_time" });
    const final = memory({
      domain: "navigation",
      slot: "pickup_time",
      refs: ["atomic:0"],
    });
    const result = gateCockpitConstructionReconciliation({
      inputs: [
        { id: "primary:0", memory: misleadingPrimary },
        { id: "atomic:0", memory: atomic },
      ],
      reconciled: [final],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(true);
    expect(result.requiredCandidateIds).toEqual(["atomic:0"]);
    expect(result.uncoveredCandidateIds).toEqual([]);
    expect(result.coveredCandidateIds).toEqual(["atomic:0"]);
  });

  it("fails closed on conflicting representations of one exact coverage event", () => {
    const coverageId = "coverage:u1:schedule:appointment_time";
    const coverage = memory({
      domain: "schedule",
      slot: "appointment_time",
      status: "partial",
      episode: "coverage-scaffold",
    });
    coverage.scene_name = "cockpit-source-coverage";
    delete coverage.metadata.value;
    delete coverage.metadata.target;
    Object.assign(coverage.metadata, {
      coverage_required_fact_count: 1,
      coverage_requires_distinct_evidence_bindings: false,
      coverage_evidence_group_ids: [`${coverageId}:evidence:1`],
      coverage_event_anchors: ["clause:1"],
    });
    const natural = memory({
      domain: "schedule",
      slot: "appointment_time",
      episode: "coverage-event:one",
      value: "4月5日上午10点",
      refs: ["atomic:0", coverageId],
    });
    const iso = memory({
      domain: "schedule",
      slot: "appointment_time",
      episode: "semantic-vehicle-check",
      value: "2026-04-05T10:00:00+08:00",
      refs: ["atomic:1", coverageId],
    });
    const atomicNatural = memory({
      domain: "schedule",
      slot: "appointment_time",
      episode: "coverage-event:one",
      value: "4月5日上午10点",
    });
    const atomicIso = memory({
      domain: "schedule",
      slot: "appointment_time",
      episode: "semantic-vehicle-check",
      value: "2026-04-05T10:00:00+08:00",
    });

    const result = gateCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:0", memory: atomicNatural },
        { id: "atomic:1", memory: atomicIso },
        { id: coverageId, memory: coverage },
      ],
      reconciled: [natural, iso],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_conflicting_single_event_coverage_alias");
    expect(result.diagnostics.filter((entry) =>
      entry.issue === "reconciliation_conflicting_single_event_coverage_alias"
    ).map((entry) => entry.rowIndex)).toEqual([0, 1]);
  });

  it.each([1, 2])(
    "preserves distinct same-state events when structured event times prove separation (required=%i)",
    (requiredFactCount) => {
    const coverageId = "coverage:u1:schedule:appointment_time";
    const coverage = memory({
      domain: "schedule",
      slot: "appointment_time",
      status: "partial",
      episode: "coverage-scaffold",
    });
    coverage.scene_name = "cockpit-source-coverage";
    delete coverage.metadata.value;
    Object.assign(coverage.metadata, {
      coverage_required_fact_count: requiredFactCount,
      coverage_required_temporal_count: requiredFactCount,
      coverage_requires_distinct_evidence_bindings: requiredFactCount > 1,
      coverage_evidence_group_ids: Array.from(
        { length: requiredFactCount },
        (_, index) => `${coverageId}:evidence:${index + 1}`,
      ),
      coverage_event_anchors: Array.from(
        { length: requiredFactCount },
        (_, index) => `event:${index + 1}`,
      ),
    });
    const first = memory({
      domain: "schedule",
      slot: "appointment_time",
      episode: "event-one",
      value: "2026-04-05T10:00:00+08:00",
      refs: ["atomic:0", coverageId],
    });
    const second = memory({
      domain: "schedule",
      slot: "appointment_time",
      episode: "event-two",
      value: "2026-04-06T10:00:00+08:00",
      refs: ["atomic:1", coverageId],
    });
    first.metadata.activity_start_time = first.metadata.value;
    second.metadata.activity_start_time = second.metadata.value;
    const atomicFirst = structuredClone(first);
    const atomicSecond = structuredClone(second);
    delete atomicFirst.metadata.input_candidate_ids;
    delete atomicSecond.metadata.input_candidate_ids;

    const result = gateCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:0", memory: atomicFirst },
        { id: "atomic:1", memory: atomicSecond },
        { id: coverageId, memory: coverage },
      ],
      reconciled: [first, second],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(true);
    expect(result.issues).not.toContain("reconciliation_conflicting_single_event_coverage_alias");
    },
  );

  it("fails when a complete input is omitted or mapped across domains", () => {
    const input = memory({ domain: "notification", slot: "valid_period" });
    const wrong = memory({ domain: "reminder", slot: "reminder_content", refs: ["atomic:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: input }],
      reconciled: [wrong],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_uncovered_atomic_candidate");
  });

  it("does not let a cancelled row consume an updated atomic obligation", () => {
    const updatedAtomic = memory({
      domain: "schedule",
      slot: "appointment_time",
      relation: "updated",
      value: "2026-04-08T15:00:00+08:00",
    });
    updatedAtomic.metadata.supersedes = ["prior-time"];
    const cancelled = memory({
      domain: "schedule",
      slot: "appointment_time",
      relation: "cancelled",
      value: "2026-04-08T15:00:00+08:00",
      refs: ["atomic:0"],
    });
    cancelled.metadata.supersedes = ["prior-time"];

    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: updatedAtomic }],
      reconciled: [cancelled],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toEqual(["atomic:0"]);
    expect(result.issues).toContain("reconciliation_uncovered_atomic_candidate");
  });

  it("requires an explicit factual compilation stage for source coverage", () => {
    const obligation = memory({ domain: "selection", slot: "price_constraint", status: "partial" });
    obligation.metadata.constraint_target = "ticket";
    delete obligation.metadata.value;
    const missing = gateCockpitConstructionReconciliation({
      inputs: [{ id: "coverage:u1:selection:price_constraint:ticket", memory: obligation }],
      reconciled: [],
      maxMemories: 10,
    });

    expect(missing.accepted).toBe(false);
    expect(missing.requiredCandidateIds).toEqual(["coverage:u1:selection:price_constraint:ticket"]);
    expect(missing.issues).toContain("reconciliation_uncovered_source_coverage_obligation");

    const covered = memory({
      domain: "selection",
      slot: "price_constraint",
      refs: ["coverage:u1:selection:price_constraint:ticket"],
    });
    covered.metadata.constraint_target = "ticket";
    const accepted = gateCockpitConstructionReconciliation({
      inputs: [{ id: "coverage:u1:selection:price_constraint:ticket", memory: obligation }],
      reconciled: [covered],
      maxMemories: 10,
    });

    expect(accepted.accepted).toBe(false);
    expect(accepted.coveredCandidateIds).toContain("coverage:u1:selection:price_constraint:ticket");
    expect(accepted.issues).toContain("reconciliation_row_without_factual_candidate_anchor");
  });

  it.each([
    {
      label: "changes its occupant scope",
      mutate: (output: ExtractedMemory) => {
        output.metadata.occupant_scope = "front-right-passenger";
      },
    },
    {
      label: "drops its occupant scope",
      mutate: (output: ExtractedMemory) => {
        delete output.metadata.occupant_scope;
      },
    },
  ])("does not cover a scoped source obligation when output $label", ({ mutate }) => {
    const id = "coverage:u1:climate:temperature:rear-left";
    const obligation = memory({ domain: "climate", slot: "temperature", status: "partial" });
    obligation.metadata.occupant_scope = "rear-left-passenger";
    const output = memory({ domain: "climate", slot: "temperature", refs: [id] });
    output.metadata.occupant_scope = "rear-left-passenger";
    mutate(output);
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id, memory: obligation }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toEqual([id]);
    expect(result.issues).toContain("reconciliation_uncovered_source_coverage_obligation");
  });

  it.each([
    {
      label: "memory type",
      mutate: (output: ExtractedMemory) => {
        output.type = "semantic";
      },
    },
    {
      label: "scene",
      mutate: (output: ExtractedMemory) => {
        output.scene_name = "schedule";
      },
    },
  ])("does not cover a complete atomic candidate across a different $label", ({ mutate }) => {
    const atomic = memory({ domain: "navigation", slot: "destination" });
    const output = memory({
      domain: "navigation",
      slot: "destination",
      refs: ["atomic:0"],
    });
    mutate(output);
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: atomic }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toEqual(["atomic:0"]);
    expect(result.issues).toContain("reconciliation_uncovered_atomic_candidate");
  });

  it("does not let one price target consume another source obligation", () => {
    const obligation = memory({ domain: "selection", slot: "price_constraint", status: "partial" });
    obligation.metadata.constraint_target = "ticket";
    const wrongTarget = memory({
      domain: "selection",
      slot: "price_constraint",
      refs: ["coverage:u1:selection:price_constraint:ticket"],
    });
    wrongTarget.metadata.constraint_target = "per_capita";
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "coverage:u1:selection:price_constraint:ticket", memory: obligation }],
      reconciled: [wrongTarget],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toEqual(["coverage:u1:selection:price_constraint:ticket"]);
  });

  it("does not trust a model-authored controlled-domain canonicalization flag", () => {
    const input = memory({ domain: "schedule", slot: "arrival_time" });
    const canonical = memory({
      domain: "navigation",
      slot: "pickup_time",
      refs: ["atomic:0"],
      canonicalizedRefs: ["atomic:0"],
    });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: input }],
      reconciled: [canonical],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.coveredCandidateIds).toContain("atomic:0");
    expect(result.uncoveredCandidateIds).toEqual([]);
    expect(result.issues).toContain("reconciliation_row_without_factual_candidate_anchor");
  });

  it("does not count a candidate as covered after dropping its explicit seat scope", () => {
    const scoped = memory({ slot: "temperature", domain: "climate", seatZone: "副驾" });
    const weakened = memory({ slot: "temperature", domain: "climate", refs: ["atomic:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: scoped }],
      reconciled: [weakened],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toEqual(["atomic:0"]);
  });

  it("does not let arbitrary model output refine a generic user to a named person", () => {
    const generic = memory({ slot: "destination" });
    const named = memory({ slot: "destination", refs: ["atomic:0"] });
    named.metadata.subject = "冯遥";
    named.metadata.state_key = "navigation|冯遥|car|driver|destination";
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: generic }],
      reconciled: [named],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toEqual(["atomic:0"]);
  });

  it("does not weaken a named subject back to a generic user", () => {
    const named = memory({ slot: "destination" });
    named.metadata.subject = "冯遥";
    named.metadata.state_key = "navigation|冯遥|car|driver|destination";
    const generic = memory({ slot: "destination", refs: ["atomic:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: named }],
      reconciled: [generic],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.uncoveredCandidateIds).toEqual(["atomic:0"]);
  });

  it("rejects unknown references, incomplete rows, and duplicate final identities", () => {
    const input = memory({ slot: "pickup_time" });
    const first = memory({ slot: "pickup_time", status: "partial", refs: ["unknown:9"] });
    const duplicate = memory({ slot: "pickup_time", refs: ["primary:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: input }],
      reconciled: [first, duplicate],
      maxMemories: 1,
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toEqual(expect.arrayContaining([
      "reconciliation_exceeds_max_memories",
      "reconciliation_contains_incomplete_memory",
      "reconciliation_duplicate_final_identity",
      "reconciliation_unknown_input_candidate_id",
    ]));
  });

  it("rejects an asserted duplicate of the current live prior state", () => {
    const input = memory({ slot: "pickup_time" });
    const repeated = memory({ slot: "pickup_time", refs: ["primary:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: input }],
      reconciled: [repeated],
      maxMemories: 10,
      priorMemories: [{
        record_id: "prior-live",
        metadata: { ...(repeated.metadata as Record<string, unknown>) },
      }],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_reasserts_unchanged_live_prior");
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      rowIndex: 0,
      issue: "reconciliation_reasserts_unchanged_live_prior",
      matchingLivePriorRecordIds: ["prior-live"],
    }));
  });

  it("allows an unchanged live state when a new source enriches its evidence", () => {
    const prior = memory({ slot: "pickup_time", source: "u0" });
    const repeated = memory({ slot: "pickup_time", source: "u1", refs: ["atomic:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: repeated }],
      reconciled: [repeated],
      maxMemories: 10,
      priorMemories: [{
        record_id: "prior-live",
        metadata: prior.metadata as Record<string, unknown>,
      }],
    });

    expect(result.accepted).toBe(true);
    expect(result.issues).not.toContain("reconciliation_reasserts_unchanged_live_prior");
  });

  it("requires a transition to supersede the matching live prior record", () => {
    const input = memory({ slot: "destination" });
    const updated = memory({ slot: "destination", refs: ["primary:0"] });
    (updated.metadata as Record<string, unknown>).relation = "updated";
    (updated.metadata as Record<string, unknown>).supersedes = ["stale-record"];
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: input }],
      reconciled: [updated],
      maxMemories: 10,
      priorMemories: [{
        record_id: "prior-live",
        metadata: { ...(input.metadata as Record<string, unknown>) },
      }],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_transition_misses_live_prior");
    expect(result.issues).toContain("reconciliation_transition_supersedes_non_live_prior");
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      rowIndex: 0,
      matchingLivePriorRecordIds: ["prior-live"],
    }));
  });

  it("rejects a transition that supersedes a live prior from another memory class", () => {
    const updated = memory({ slot: "destination", relation: "updated", refs: ["atomic:0"] });
    updated.metadata.supersedes = ["wrong-class-live"];
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: updated }],
      reconciled: [updated],
      maxMemories: 10,
      priorMemories: [{
        record_id: "wrong-class-live",
        type: "instruction",
        scene_name: updated.scene_name,
        metadata: { ...(updated.metadata as Record<string, unknown>), relation: "asserted" },
      }],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain(
      "reconciliation_transition_supersedes_non_matching_live_prior",
    );
  });

  it("accepts a controlled transition whose legacy prior only drifted in scene label", () => {
    const updated = memory({
      slot: "destination",
      value: "上海南站",
      relation: "updated",
      refs: ["atomic:0"],
    });
    updated.scene_name = "navigation";
    updated.metadata.supersedes = ["prior-live"];
    const priorMetadata = {
      ...(updated.metadata as Record<string, unknown>),
      value: "虹桥火车站",
      relation: "asserted",
    };
    delete priorMetadata.supersedes;
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: structuredClone(updated) }],
      reconciled: [updated],
      maxMemories: 10,
      priorMemories: [{
        record_id: "prior-live",
        type: updated.type,
        scene_name: "legacy-free-route-label",
        metadata: priorMetadata,
      }],
    });

    expect(result.accepted).toBe(true);
    expect(result.issues).not.toContain(
      "reconciliation_transition_supersedes_non_matching_live_prior",
    );
  });

  it("rejects a transition that consumes a predecessor from its event-time future", () => {
    const updated = memory({ slot: "destination", relation: "updated", refs: ["atomic:0"] });
    updated.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
    updated.metadata.supersedes = ["future-live"];
    const priorMetadata = { ...(updated.metadata as Record<string, unknown>) };
    priorMetadata.relation = "asserted";
    priorMetadata.mentioned_at = "2026-04-01T10:00:00+08:00";
    delete priorMetadata.supersedes;
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: updated }],
      reconciled: [updated],
      maxMemories: 10,
      priorMemories: [{
        record_id: "future-live",
        type: updated.type,
        scene_name: updated.scene_name,
        metadata: priorMetadata,
      }],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain(
      "reconciliation_transition_supersedes_non_matching_live_prior",
    );
  });

  it("rejects a price transition whose predecessor has another constraint target", () => {
    const updated = memory({
      domain: "selection",
      slot: "price_constraint",
      relation: "updated",
      refs: ["atomic:0"],
    });
    updated.metadata.constraint_target = "per_capita";
    // Reproduce an upstream identity-reuse defect: the state key still names
    // the ticket constraint even though the structured target is per-capita.
    updated.metadata.state_key = "selection|user|car|driver|price_constraint@ticket";
    updated.metadata.supersedes = ["ticket-live"];
    const priorMetadata = { ...(updated.metadata as Record<string, unknown>) };
    priorMetadata.constraint_target = "ticket";
    priorMetadata.relation = "asserted";
    delete priorMetadata.supersedes;
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: updated }],
      reconciled: [updated],
      maxMemories: 10,
      priorMemories: [{
        record_id: "ticket-live",
        type: updated.type,
        scene_name: updated.scene_name,
        metadata: priorMetadata,
      }],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain(
      "reconciliation_transition_supersedes_non_matching_live_prior",
    );
  });

  it("does not let a persisted cross-class supersedes edge hide the correct live prior", () => {
    const old = memory({ slot: "destination", source: "u-old", value: "A" });
    const corrupt = memory({ slot: "destination", source: "u-corrupt", value: "B" });
    corrupt.metadata.relation = "updated";
    corrupt.metadata.supersedes = ["correct-live"];
    const next = memory({ slot: "destination", source: "u-next", value: "C", refs: ["atomic:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: next }],
      reconciled: [next],
      maxMemories: 10,
      priorMemories: [
        {
          record_id: "correct-live",
          type: old.type,
          scene_name: old.scene_name,
          metadata: old.metadata as Record<string, unknown>,
        },
        {
          record_id: "corrupt-cross-class-edge",
          type: "instruction",
          scene_name: corrupt.scene_name,
          metadata: corrupt.metadata as Record<string, unknown>,
        },
      ],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_asserts_over_existing_live_prior");
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      matchingLivePriorRecordIds: ["correct-live"],
    }));
  });

  it("rejects a transition that also points to an already superseded record", () => {
    const input = memory({ slot: "destination" });
    const updated = memory({ slot: "destination", refs: ["primary:0"] });
    (updated.metadata as Record<string, unknown>).relation = "cancelled";
    (updated.metadata as Record<string, unknown>).supersedes = ["current-record", "old-record"];
    const oldMetadata = { ...(input.metadata as Record<string, unknown>) };
    const currentMetadata = {
      ...(input.metadata as Record<string, unknown>),
      relation: "updated",
      supersedes: ["old-record"],
    };
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: input }],
      reconciled: [updated],
      maxMemories: 10,
      priorMemories: [
        { record_id: "current-record", metadata: currentMetadata },
        { record_id: "old-record", metadata: oldMetadata },
      ],
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_transition_supersedes_non_live_prior");
    expect(result.diagnostics).toContainEqual(expect.objectContaining({
      matchingLivePriorRecordIds: ["current-record"],
    }));
  });

  it("rejects a controlled slot attached to the wrong domain", () => {
    const wrong = memory({ domain: "notification", slot: "route_constraint", refs: ["atomic:0"] });
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: wrong }],
      reconciled: [wrong],
      maxMemories: 10,
    });

    expect(result.accepted).toBe(false);
    expect(result.issues).toContain("reconciliation_invalid_controlled_ontology");
  });

  it("exposes exact live targets for an ambiguous composite transition repair", () => {
    const destination = memory({ slot: "destination" });
    const pickupTime = memory({ slot: "pickup_time" });
    const composite = memory({ slot: "status", status: "partial", refs: ["primary:0"] });
    (composite.metadata as Record<string, unknown>).relation = "cancelled";
    (composite.metadata as Record<string, unknown>).supersedes = ["destination-live", "time-live"];
    (composite.metadata as Record<string, unknown>).construction_quality = {
      status: "partial",
      issues: ["ambiguous_transition_state"],
    };
    const result = gateCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: destination }],
      reconciled: [composite],
      maxMemories: 10,
      priorMemories: [
        { record_id: "destination-live", metadata: destination.metadata as Record<string, unknown> },
        { record_id: "time-live", metadata: pickupTime.metadata as Record<string, unknown> },
      ],
    });

    expect(result.accepted).toBe(false);
    expect(result.diagnostics[0]).toMatchObject({
      qualityIssues: ["ambiguous_transition_state"],
      matchingLivePriorRecordIds: ["destination-live", "time-live"],
      livePriorTargets: [
        { recordId: "destination-live", slot: "destination" },
        { recordId: "time-live", slot: "pickup_time" },
      ],
    });
  });
});
