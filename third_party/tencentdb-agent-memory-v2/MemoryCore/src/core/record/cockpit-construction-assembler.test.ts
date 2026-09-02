import { describe, expect, it } from "vitest";

import type { ConversationMessage } from "../conversation/l0-recorder.js";
import { assembleCockpitConstructionReconciliation } from "./cockpit-construction-assembler.js";
import { gateCockpitConstructionReconciliation } from "./cockpit-construction-reconciliation.js";
import { normalizeCockpitExtractedMemory } from "./cockpit-memory-contract.js";
import type { ExtractedMemory } from "./l1-writer.js";

function memory(params: {
  source?: string;
  value?: string;
  episode?: string;
  relation?: string;
  status?: "complete" | "partial";
  refs?: string[];
  mentionedAt?: string;
  stateQualifier?: string;
} = {}): ExtractedMemory {
  const source = params.source ?? "u1";
  return {
    content: `destination ${params.value ?? "A"}`,
    type: "episodic",
    priority: 70,
    scene_name: "navigation",
    source_message_ids: [source],
    metadata: {
      domain: "navigation",
      slot: "destination",
      value: params.value ?? "A",
      subject: "user",
      state_qualifier: params.stateQualifier,
      state_key: `navigation|user|car|driver|destination${params.stateQualifier ? `@${params.stateQualifier}` : ""}`,
      episode_key: params.episode ?? "route-1",
      relation: params.relation ?? "asserted",
      action_status: "requested",
      source_message_ids: [source],
      mentioned_at: params.mentionedAt,
      input_candidate_ids: params.refs,
      construction_quality: {
        status: params.status ?? "complete",
        score: params.status === "partial" ? 90 : 100,
        issues: params.status === "partial" ? ["missing_supersedes"] : [],
        repairs: [],
        source_count: 1,
        user_source_count: 1,
      },
    },
  };
}

function scheduleMemory(params: {
  slot: "appointment_time" | "appointment_content" | "status";
  value?: string;
  source?: string;
  episode?: string;
  relation?: "asserted" | "updated" | "cancelled" | "negated";
  status?: "complete" | "partial";
  issues?: string[];
  refs?: string[];
  supersedes?: string[];
}): ExtractedMemory {
  const result = memory({
    source: params.source ?? "u1",
    value: params.value ?? "车辆检查",
    episode: params.episode ?? "appointment-1",
    relation: params.relation,
    status: params.status,
    refs: params.refs,
  });
  result.content = `${params.slot} ${params.value ?? "车辆检查"}`;
  result.scene_name = "schedule";
  result.metadata.domain = "schedule";
  result.metadata.slot = params.slot;
  result.metadata.state_key = `schedule|user|car|driver|${params.slot}`;
  if (params.supersedes) result.metadata.supersedes = params.supersedes;
  if (params.issues) {
    result.metadata.construction_quality = {
      status: params.status ?? "partial",
      score: Math.max(0, 100 - params.issues.length * 10),
      issues: params.issues,
      repairs: [],
      source_count: 1,
      user_source_count: 1,
    };
  }
  return result;
}

function hasAmbiguousBindingIssue(memory: ExtractedMemory): boolean {
  const quality = memory.metadata.construction_quality;
  return Boolean(quality && typeof quality === "object" && !Array.isArray(quality)
    && Array.isArray((quality as Record<string, unknown>).issues)
    && ((quality as Record<string, unknown>).issues as unknown[])
      .includes("ambiguous_semantic_binding_alias"));
}

describe("cockpit deterministic construction assembler", () => {
  it("binds source coverage only through an exact single-source atomic fact", () => {
    const atomic = memory({ source: "u1", value: "A", refs: [] });
    const exactCoverage = memory({ source: "u1", status: "partial", refs: [] });
    delete exactCoverage.metadata.value;
    const otherEventCoverage = memory({ source: "u2", status: "partial", refs: [] });
    delete otherEventCoverage.metadata.value;
    const inputs = [
      { id: "atomic:0", memory: atomic },
      { id: "coverage:u1:navigation:destination", memory: exactCoverage },
      { id: "coverage:u2:navigation:destination", memory: otherEventCoverage },
    ];

    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.input_candidate_ids).toEqual(
      expect.arrayContaining(["atomic:0", "coverage:u1:navigation:destination"]),
    );
    expect(result.memories[0].metadata.input_candidate_ids).not.toContain(
      "coverage:u2:navigation:destination",
    );
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });
    expect(gate.accepted).toBe(false);
    expect(gate.uncoveredCandidateIds).toEqual(["coverage:u2:navigation:destination"]);
  });

  it("canonicalizes a qualified unknown slot from one exact controlled atomic obligation", () => {
    const primary = memory({ value: "北京人卫酒店", refs: ["primary:0"] });
    primary.metadata.slot = "default_temporary_destination";
    primary.metadata.state_key = "navigation|user|car|driver|default_temporary_destination";
    const atomic = memory({ value: "北京人卫酒店", refs: ["atomic:0"] });
    const reconciled = memory({
      value: "北京人卫酒店",
      refs: ["primary:0", "atomic:0"],
    });
    reconciled.metadata.slot = "default_temporary_destination";
    reconciled.metadata.state_key = "navigation|user|car|driver|default_temporary_destination";
    reconciled.metadata.canonicalized_input_candidate_ids = ["primary:0", "atomic:0"];

    const inputs = [
      { id: "primary:0", memory: primary },
      { id: "atomic:0", memory: atomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [reconciled],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata).toMatchObject({
      domain: "navigation",
      slot: "destination",
      state_key: "navigation|user|car|driver|destination",
      value: "北京人卫酒店",
      input_candidate_ids: ["atomic:0"],
      construction_quality: {
        status: "complete",
        repairs: expect.arrayContaining([
          "canonicalized_reconciled_slot_from_unique_atomic_evidence",
        ]),
      },
    });
    expect(result.repairCounts).toMatchObject({
      canonicalized_reconciled_slot_from_unique_atomic_evidence: 1,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not collapse an unverified semantic slot qualifier into a controlled atomic slot", () => {
    const atomic = memory({ value: "北京人卫酒店", refs: ["atomic:0"] });
    const reconciled = memory({ value: "北京人卫酒店", refs: ["atomic:0"] });
    reconciled.metadata.slot = "avoid_destination";
    reconciled.metadata.state_key = "navigation|user|car|driver|avoid_destination";
    reconciled.metadata.canonicalized_input_candidate_ids = ["atomic:0"];

    const inputs = [{ id: "atomic:0", memory: atomic }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [reconciled],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.slot).toBe("destination");
    expect(result.repairCounts).toMatchObject({
      dropped_unverified_atomic_slot_canonicalization: 1,
      restored_complete_atomic_obligation: 1,
    });
  });

  it("binds omitted proposal IDs from exact evidence and structure", () => {
    const primary = memory();
    const atomic = memory();
    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "primary:0", memory: primary },
        { id: "atomic:0", memory: atomic },
      ],
      reconciled: [memory()],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.input_candidate_ids).toEqual(["primary:0", "atomic:0"]);
    expect(gateCockpitConstructionReconciliation({
      inputs: [
        { id: "primary:0", memory: primary },
        { id: "atomic:0", memory: atomic },
      ],
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
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
  ])("does not bind a primary-only row that rewrites its $label", ({ mutate }) => {
    const primary = memory({ value: "北京", episode: "route-grounded" });
    const output = memory({
      value: "北京",
      episode: "route-grounded",
      refs: ["primary:0"],
    });
    mutate(output);
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "primary:0", memory: primary }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.memories).toEqual([]);
    expect(result.repairCounts.dropped_unbound_reconciliation_row).toBe(1);
  });

  it.each([
    { id: "primary:0", status: "complete" as const },
    { id: "coverage:u1:climate:temperature", status: "partial" as const },
  ])("does not trust a model-authored cross-slot canonicalization for $id", ({ id, status }) => {
    const candidate = memory({ value: "22" });
    candidate.scene_name = "climate";
    candidate.metadata.domain = "climate";
    candidate.metadata.slot = "temperature";
    candidate.metadata.state_key = "climate|user|car|driver|temperature";
    candidate.metadata.construction_quality = {
      status,
      score: status === "complete" ? 100 : 90,
      issues: status === "complete" ? [] : ["source_coverage_obligation"],
      repairs: [],
      source_count: 1,
      user_source_count: 1,
    };
    const output = memory({ value: "上海", refs: [id] });
    output.scene_name = "climate";
    output.metadata.canonicalized_input_candidate_ids = [id];
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id, memory: candidate }],
      reconciled: [output],
      maxMemories: 10,
    });

    expect(result.memories).toEqual([]);
    expect(result.repairCounts.dropped_unbound_reconciliation_row).toBe(1);
  });

  it("replaces an incomplete model row with the complete atomic obligation", () => {
    const atomic = memory();
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: atomic }],
      reconciled: [memory({ relation: "negated", status: "partial", refs: ["atomic:0"] })],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.repairCounts).toMatchObject({
      dropped_incomplete_reconciliation_row: 1,
      restored_complete_atomic_obligation: 1,
    });
    expect(result.memories[0].metadata.relation).toBe("asserted");
  });

  it("drops ungrounded advisory rows and restores distinct same-slot atomic obligations", () => {
    const normal = memory({ source: "u1", value: "facilities-first", episode: "route-normal" });
    const lowBattery = memory({ source: "u1", value: "distance-first", episode: "route-low-battery" });
    const unboundNormal = memory({ source: "u1", value: "facilities-first", episode: "route-normal" });
    const unboundLowBattery = memory({ source: "u1", value: "distance-first", episode: "route-low-battery" });
    unboundNormal.metadata.subject = "passenger";
    unboundLowBattery.metadata.subject = "passenger";

    const inputs = [
      { id: "atomic:0", memory: normal },
      { id: "atomic:1", memory: lowBattery },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [unboundNormal, unboundLowBattery],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((entry) => entry.metadata.input_candidate_ids)).toEqual([
      ["atomic:0"],
      ["atomic:1"],
    ]);
    expect(result.memories.map((entry) => entry.metadata.value)).toEqual([
      "facilities-first",
      "distance-first",
    ]);
    expect(result.repairCounts).toMatchObject({
      dropped_unbound_reconciliation_row: 2,
      restored_complete_atomic_obligation: 2,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("preserves two conditional variants of the same policy identity", () => {
    const normal = memory({ value: "facilities-rating-distance", refs: ["atomic:0"] });
    const lowBattery = memory({ value: "distance-rating-facilities", refs: ["atomic:1"] });
    for (const item of [normal, lowBattery]) {
      item.metadata.domain = "selection";
      item.metadata.slot = "ranking_policy";
      item.metadata.state_key = "selection|user|car|driver|ranking_policy";
    }
    normal.metadata.condition = "default";
    lowBattery.metadata.condition = "battery_level < 14%";

    const inputs = [
      { id: "atomic:0", memory: normal },
      { id: "atomic:1", memory: lowBattery },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [normal, lowBattery],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((entry) => entry.metadata.condition)).toEqual([
      "default",
      "battery_level < 14%",
    ]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("lets a precise atomic row suppress a coarser same-source advisory duplicate", () => {
    const primary = memory({ source: "u1", value: "A", episode: "route-precise" });
    const atomic = memory({ source: "u1", value: "A", episode: "route-precise" });
    primary.metadata.input_candidate_ids = ["primary:0"];
    primary.metadata.subject = "user";
    atomic.metadata.input_candidate_ids = ["atomic:0"];
    atomic.metadata.subject = "named-driver";
    atomic.metadata.state_key = "navigation|named-driver|car|driver|destination";
    atomic.metadata.valid_from = "2026-04-07T15:00:00+08:00";
    atomic.metadata.valid_to = "2026-04-07T16:00:00+08:00";

    const inputs = [
      { id: "primary:0", memory: primary },
      { id: "atomic:0", memory: atomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, atomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.input_candidate_ids).toEqual(expect.arrayContaining([
      "primary:0",
      "atomic:0",
    ]));
    expect(result.memories[0].metadata.subject).toBe("named-driver");
    expect(result.memories[0].metadata.state_key)
      .toBe("navigation|named-driver|car|driver|destination");
    expect(result.memories[0].metadata.valid_from).toBe("2026-04-07T15:00:00+08:00");
    expect(result.repairCounts).toMatchObject({ coalesced_compatible_semantic_binding: 1 });
  });

  it("keeps a non-covering generic atomic binding separate from a named primary", () => {
    const namedPrimary = memory({ source: "u1", value: "A", episode: "route-named" });
    const genericAtomic = memory({ source: "u1", value: "A", episode: "route-named" });
    namedPrimary.metadata.subject = "冯遥";
    namedPrimary.metadata.state_key = "navigation|冯遥|car|driver|destination";
    namedPrimary.metadata.input_candidate_ids = ["primary:0"];
    genericAtomic.metadata.input_candidate_ids = ["atomic:0"];
    genericAtomic.metadata.valid_from = "2026-04-07T15:00:00+08:00";
    genericAtomic.metadata.valid_to = "2026-04-07T16:00:00+08:00";

    const inputs = [
      { id: "primary:0", memory: namedPrimary },
      { id: "atomic:0", memory: genericAtomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [namedPrimary, genericAtomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((entry) => ({
      subject: entry.metadata.subject,
      stateKey: entry.metadata.state_key,
      refs: entry.metadata.input_candidate_ids,
    }))).toEqual([
      {
        subject: "冯遥",
        stateKey: "navigation|冯遥|car|driver|destination",
        refs: ["primary:0"],
      },
      {
        subject: "user",
        stateKey: "navigation|user|car|driver|destination",
        refs: ["atomic:0"],
      },
    ]);
    expect(result.resolvedCandidateIds).toEqual([]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not persist a coverage-only row without an independent atomic fact", () => {
    const namedPrimary = memory({ source: "u1", value: "A", refs: ["primary:0"] });
    namedPrimary.metadata.subject = "冯遥";
    namedPrimary.metadata.state_key = "navigation|冯遥|car|driver|destination";
    const genericCoverage = memory({ source: "u1", value: "A", refs: ["coverage:0"] });
    genericCoverage.metadata.valid_from = "2026-04-07T15:00:00+08:00";
    const inputs = [
      { id: "primary:0", memory: namedPrimary },
      { id: "coverage:0", memory: genericCoverage },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [namedPrimary, genericCoverage],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.resolvedCandidateIds).toEqual([]);
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });
    expect(gate.accepted).toBe(false);
    expect(gate.uncoveredCandidateIds).toEqual(["coverage:0"]);
  });

  it("does not coalesce differently keyed states with equivalent concrete bindings", () => {
    const first = memory({ source: "u1", value: "A", refs: ["primary:0"] });
    const second = memory({ source: "u1", value: "A", refs: ["atomic:0"] });
    for (const row of [first, second]) row.metadata.subject = "周宁";
    first.metadata.state_key = "navigation|周宁|car|driver|destination";
    second.metadata.state_key = "navigation|周宁|car|driver|alternate_destination";
    const inputs = [
      { id: "primary:0", memory: first },
      { id: "atomic:0", memory: second },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [first, second],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
    expect(result.resolvedCandidateIds).toEqual([]);
  });

  it("does not coalesce separately named members of the same state slot", () => {
    const breakfast = memory({
      source: "u1",
      value: "松林文化馆",
      refs: ["atomic:0"],
      stateQualifier: "早餐地点",
    });
    const returnTrip = memory({
      source: "u1",
      value: "松林文化馆",
      refs: ["atomic:1"],
      stateQualifier: "返程地点",
    });
    const inputs = [
      { id: "atomic:0", memory: breakfast },
      { id: "atomic:1", memory: returnTrip },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [breakfast, returnTrip],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((entry) => entry.metadata.state_qualifier).sort())
      .toEqual(["早餐地点", "返程地点"].sort());
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
  });

  it("does not authorize binding refinement from an unrelated primary reference", () => {
    const source = "u-wrong-primary-refinement";
    const episode = "wrong-primary-refinement";
    const wrongPrimary = memory({ source, value: "错误地点", episode });
    wrongPrimary.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const genericAtomic = memory({ source, value: "首都机场", episode });
    genericAtomic.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const mixedGeneric = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["primary:wrong", "atomic:generic"],
    });
    mixedGeneric.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const specificAtomic = memory({ source, value: "首都机场", episode });
    specificAtomic.metadata.occupant_scope = "rear-left-passenger";
    specificAtomic.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const specific = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["atomic:specific"],
    });
    specific.metadata.occupant_scope = "rear-left-passenger";
    specific.metadata.state_key = "navigation|user|car|rear-left-passenger|destination";
    const inputs = [
      { id: "primary:wrong", memory: wrongPrimary },
      { id: "atomic:generic", memory: genericAtomic },
      { id: "atomic:specific", memory: specificAtomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [mixedGeneric, specific],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
    expect(result.memories.map((row) => row.metadata.state_key)).toEqual([
      "navigation|user|car|unspecified-occupant|destination",
      "navigation|user|car|rear-left-passenger|destination",
    ]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not authorize binding refinement from wrong-value atomic evidence", () => {
    const source = "u-wrong-atomic-refinement";
    const episode = "wrong-atomic-refinement";
    const primary = memory({ source, value: "首都机场", episode, refs: ["primary:0"] });
    primary.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const atomic = memory({ source, value: "大兴机场", episode });
    atomic.metadata.occupant_scope = "rear-left-passenger";
    atomic.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const specific = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["atomic:0"],
    });
    specific.metadata.occupant_scope = "rear-left-passenger";
    specific.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const inputs = [
      { id: "primary:0", memory: primary },
      { id: "atomic:0", memory: atomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, specific],
      maxMemories: 10,
    });

    // The reconciler-authored specific row has neither the primary value nor
    // the atomic value and is now discarded instead of being retained as a
    // duplicate blocker. The two exact source proposals remain independent.
    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
    expect(result.memories.map((row) => row.metadata.state_key)).toEqual([
      "navigation|user|car|unspecified-occupant|destination",
      "navigation|user|car|rear-left-passenger|destination",
    ]);
    expect(result.memories.map((row) => row.metadata.value)).toEqual([
      "首都机场",
      "大兴机场",
    ]);
    expect(result.repairCounts).toMatchObject({
      restored_complete_atomic_obligation: 1,
      dropped_unbound_reconciliation_row: 1,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not let one exact atomic sibling launder a conflicting atomic obligation", () => {
    const source = "u-conflicting-atomic-siblings";
    const episode = "conflicting-atomic-siblings";
    const primary = memory({ source, value: "首都机场", episode, refs: ["primary:0"] });
    primary.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const goodAtomic = memory({ source, value: "首都机场", episode });
    const badAtomic = memory({ source, value: "大兴机场", episode });
    for (const row of [goodAtomic, badAtomic]) {
      row.metadata.occupant_scope = "rear-left-passenger";
      row.metadata.state_key =
        "navigation|user|car|rear-left-passenger|destination";
    }
    const specific = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["atomic:good", "atomic:bad"],
    });
    specific.metadata.occupant_scope = "rear-left-passenger";
    specific.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const inputs = [
      { id: "primary:0", memory: primary },
      { id: "atomic:good", memory: goodAtomic },
      { id: "atomic:bad", memory: badAtomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, specific],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((row) => row.metadata.value)).toEqual([
      "首都机场",
      "大兴机场",
    ]);
    expect(result.memories.find((row) => row.metadata.value === "首都机场")
      ?.metadata.input_candidate_ids).not.toContain("atomic:bad");
    expect(result.memories.find((row) => row.metadata.value === "大兴机场")
      ?.metadata.input_candidate_ids).toContain("atomic:bad");
    expect(result.repairCounts).toMatchObject({
      restored_complete_atomic_obligation: 1,
      preserved_conflicting_duplicate_final_identity: 1,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("never treats a rich appointment atomic as covered by a coarse output", () => {
    const source = "u-reverse-appointment-alias";
    const episode = "reverse-appointment-alias";
    const mentionedAt = "2026-03-16T01:00:00.000Z";
    const richAtomic = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内进行车辆检查",
      source,
      episode,
    });
    richAtomic.metadata.mentioned_at = mentionedAt;
    richAtomic.metadata.source_session_id = "session-reverse-alias";
    const coarseOutput = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    coarseOutput.content = "已记录车辆检查预约";
    coarseOutput.metadata.mentioned_at = mentionedAt;
    coarseOutput.metadata.source_session_id = "session-reverse-alias";
    const destinationAtomic = memory({ source, value: "故宫博物院内", episode });
    destinationAtomic.metadata.mentioned_at = mentionedAt;
    destinationAtomic.metadata.source_session_id = "session-reverse-alias";
    const inputs = [
      { id: "atomic:content", memory: richAtomic },
      { id: "atomic:destination", memory: destinationAtomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [coarseOutput],
      maxMemories: 10,
    });

    expect(result.memories.map((row) => row.metadata.value)).toEqual([
      "在故宫博物院内进行车辆检查",
      "故宫博物院内",
    ]);
    expect(result.memories.map((row) => row.metadata.value)).not.toContain("车辆检查");
    expect(result.repairCounts.restored_complete_atomic_obligation).toBe(2);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not erase a present atomic fact when a duplicate sibling omits it", () => {
    const source = "u-atomic-unit-siblings";
    const episode = "atomic-unit-siblings";
    const withoutUnit = memory({
      source,
      value: "120",
      episode,
      refs: ["atomic:without-unit"],
    });
    const withUnit = memory({
      source,
      value: "120",
      episode,
      refs: ["atomic:with-unit"],
    });
    withUnit.metadata.unit = "km";
    const inputs = [
      { id: "atomic:without-unit", memory: withoutUnit },
      { id: "atomic:with-unit", memory: withUnit },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [withoutUnit, withUnit],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((row) => row.metadata.unit)).toEqual([undefined, "km"]);
    expect(result.repairCounts.preserved_conflicting_duplicate_final_identity).toBe(1);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not enter an appointment alias path from an inexact primary ID", () => {
    const source = "u-inexact-primary-appointment-alias";
    const episode = "inexact-primary-appointment-alias";
    const wrongPrimary = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎更换",
      source,
      episode,
    });
    const rich = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内进行车辆检查",
      source,
      episode,
      refs: ["primary:wrong"],
    });
    const coarse = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    const destination = memory({
      source,
      value: "故宫博物院内",
      episode,
      refs: ["atomic:destination"],
    });
    const inputs = [
      { id: "primary:wrong", memory: wrongPrimary },
      { id: "atomic:content", memory: coarse },
      { id: "atomic:destination", memory: destination },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [rich, coarse, destination],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(3);
    expect(result.repairCounts.coalesced_same_episode_cross_proposal_value_alias)
      .toBeUndefined();
    expect(result.memories.map((row) => row.metadata.value)).toEqual([
      "在故宫博物院内进行车辆检查",
      "在故宫博物院内进行车辆检查",
      "故宫博物院内",
    ]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not authorize binding refinement from incomplete atomic evidence", () => {
    const source = "u-partial-atomic-refinement";
    const episode = "partial-atomic-refinement";
    const primary = memory({ source, value: "首都机场", episode, refs: ["primary:0"] });
    primary.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const atomic = memory({ source, value: "首都机场", episode, status: "partial" });
    atomic.metadata.occupant_scope = "rear-left-passenger";
    atomic.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const specific = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["atomic:0"],
    });
    specific.metadata.occupant_scope = "rear-left-passenger";
    specific.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const inputs = [
      { id: "primary:0", memory: primary },
      { id: "atomic:0", memory: atomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, specific],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
    expect(result.memories.map((row) => row.metadata.state_key)).toEqual([
      "navigation|user|car|unspecified-occupant|destination",
      "navigation|user|car|rear-left-passenger|destination",
    ]);
  });

  it("does not let a wrong-value partial atomic satisfy a complete output", () => {
    const source = "u-wrong-partial-atomic";
    const episode = "wrong-partial-atomic";
    const primary = memory({ source, value: "首都机场", episode, refs: ["primary:0"] });
    primary.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const partialAtomic = memory({
      source,
      value: "大兴机场",
      episode,
      status: "partial",
    });
    partialAtomic.metadata.occupant_scope = "rear-left-passenger";
    partialAtomic.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const specific = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["atomic:partial"],
    });
    specific.metadata.occupant_scope = "rear-left-passenger";
    specific.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const inputs = [
      { id: "primary:0", memory: primary },
      { id: "atomic:partial", memory: partialAtomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, specific],
      maxMemories: 10,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories.flatMap((row) => row.metadata.input_candidate_ids as string[]))
      .not.toContain("atomic:partial");
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
    expect(gate.accepted).toBe(false);
    expect(gate.uncoveredCandidateIds).toContain("atomic:partial");
  });

  it("allows only a generic user state-key repair for a missing-scope partial atomic", () => {
    const partialAtomic = memory({ status: "partial" });
    delete partialAtomic.metadata.subject;
    partialAtomic.metadata.state_key =
      "navigation|unspecified-subject|unspecified-vehicle|unspecified-zone|destination";
    partialAtomic.metadata.construction_quality = {
      status: "partial",
      score: 90,
      issues: ["missing_scope"],
      repairs: [],
      source_count: 1,
      user_source_count: 1,
    };
    const genericOutput = memory({ refs: ["atomic:partial"] });
    genericOutput.metadata.state_key =
      "navigation|user|unspecified-vehicle|unspecified-zone|destination";
    const inputs = [{ id: "atomic:partial", memory: partialAtomic }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [genericOutput],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.subject).toBe("user");
    expect(result.memories[0].metadata.input_candidate_ids).toContain("atomic:partial");
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not turn a missing-scope partial atomic into a specific occupant", () => {
    const partialAtomic = memory({ status: "partial" });
    delete partialAtomic.metadata.subject;
    partialAtomic.metadata.state_key =
      "navigation|unspecified-subject|unspecified-vehicle|unspecified-zone|destination";
    partialAtomic.metadata.construction_quality = {
      status: "partial",
      score: 90,
      issues: ["missing_scope"],
      repairs: [],
      source_count: 1,
      user_source_count: 1,
    };
    const specificOutput = memory({ refs: ["atomic:partial"] });
    delete specificOutput.metadata.subject;
    specificOutput.metadata.occupant_scope = "rear-left-passenger";
    specificOutput.metadata.state_key =
      "navigation|rear-left-passenger|unspecified-vehicle|unspecified-zone|destination";
    const inputs = [{ id: "atomic:partial", memory: partialAtomic }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [specificOutput],
      maxMemories: 10,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.memories).toEqual([]);
    expect(gate.accepted).toBe(false);
    expect(gate.uncoveredCandidateIds).toContain("atomic:partial");
  });

  it("does not resolve a missing-scope atomic through a specific cancel-replace pair", () => {
    const source = "u-missing-scope-cancel-replace";
    const episode = "missing-scope-cancel-replace";
    const prior = memory({ source: "u-prior", value: "首都机场", episode });
    prior.metadata.subject = "周宁";
    prior.metadata.occupant_scope = "rear-left-passenger";
    prior.metadata.state_key =
      "navigation|周宁|car|rear-left-passenger|destination";

    const primaryCancellation = memory({
      source,
      value: "首都机场",
      episode,
      relation: "cancelled",
      refs: ["primary:cancel"],
    });
    primaryCancellation.metadata.subject = "周宁";
    primaryCancellation.metadata.occupant_scope = "rear-left-passenger";
    primaryCancellation.metadata.state_key =
      "navigation|周宁|car|rear-left-passenger|destination";
    primaryCancellation.metadata.supersedes = ["prior-destination"];

    const partialCancellation = memory({
      source,
      value: "首都机场",
      episode,
      relation: "cancelled",
      status: "partial",
    });
    delete partialCancellation.metadata.subject;
    partialCancellation.metadata.state_key =
      "navigation|unspecified-subject|car|unspecified-zone|destination";
    partialCancellation.metadata.supersedes = ["prior-destination"];
    partialCancellation.metadata.construction_quality = {
      status: "partial",
      score: 90,
      issues: ["missing_scope"],
      repairs: [],
      source_count: 1,
      user_source_count: 1,
    };

    const assertedReplacement = memory({
      source,
      value: "大兴机场",
      episode,
      relation: "asserted",
      refs: ["atomic:replacement"],
    });
    assertedReplacement.metadata.subject = "周宁";
    assertedReplacement.metadata.occupant_scope = "rear-left-passenger";
    assertedReplacement.metadata.state_key =
      "navigation|周宁|car|rear-left-passenger|destination";

    const reconciledCancellation = memory({
      source,
      value: "首都机场",
      episode,
      relation: "cancelled",
      refs: ["primary:cancel", "atomic:cancel"],
    });
    reconciledCancellation.metadata.subject = "周宁";
    reconciledCancellation.metadata.occupant_scope = "rear-left-passenger";
    reconciledCancellation.metadata.state_key =
      "navigation|周宁|car|rear-left-passenger|destination";
    reconciledCancellation.metadata.supersedes = ["prior-destination"];

    const inputs = [
      { id: "primary:cancel", memory: primaryCancellation },
      { id: "atomic:cancel", memory: partialCancellation },
      { id: "atomic:replacement", memory: assertedReplacement },
    ];
    const priors = [{
      record_id: "prior-destination",
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [reconciledCancellation, assertedReplacement],
      maxMemories: 10,
      priorMemories: priors,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:cancel");
    expect(result.memories.flatMap((row) =>
      row.metadata.input_candidate_ids as string[]
    )).not.toContain("atomic:cancel");
    expect(gate.accepted).toBe(false);
    expect(gate.uncoveredCandidateIds).toContain("atomic:cancel");
  });

  it("requires identical source sets across a semantic binding refinement", () => {
    const episode = "overlapping-source-refinement";
    const primary = memory({ source: "u1", value: "首都机场", episode, refs: ["primary:0"] });
    primary.source_message_ids = ["u1", "u2"];
    primary.metadata.source_message_ids = ["u1", "u2"];
    primary.metadata.state_key =
      "navigation|user|car|unspecified-occupant|destination";
    const atomic = memory({ source: "u1", value: "首都机场", episode, refs: ["atomic:0"] });
    atomic.source_message_ids = ["u1", "u3"];
    atomic.metadata.source_message_ids = ["u1", "u3"];
    atomic.metadata.occupant_scope = "rear-left-passenger";
    atomic.metadata.state_key =
      "navigation|user|car|rear-left-passenger|destination";
    const inputs = [
      { id: "primary:0", memory: primary },
      { id: "atomic:0", memory: atomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, atomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
    expect(result.memories.map((row) => row.source_message_ids)).toEqual([
      ["u1", "u2"],
      ["u1", "u3"],
    ]);
  });

  it("does not fill missing event provenance while refining a person binding", () => {
    for (const [field, value] of [
      ["source_session_id", "session-2"],
      ["source_session_ids", ["session-2"]],
      ["mentioned_at", "2026-04-07T15:00:00+08:00"],
    ] as const) {
      const source = `u-missing-${field}`;
      const episode = `missing-${field}-refinement`;
      const primary = memory({ source, value: "首都机场", episode, refs: ["primary:0"] });
      primary.metadata.state_key =
        "navigation|user|car|unspecified-occupant|destination";
      const atomic = memory({ source, value: "首都机场", episode, refs: ["atomic:0"] });
      atomic.metadata.occupant_scope = "rear-left-passenger";
      atomic.metadata.state_key =
        "navigation|user|car|rear-left-passenger|destination";
      atomic.metadata[field] = value;
      const inputs = [
        { id: "primary:0", memory: primary },
        { id: "atomic:0", memory: atomic },
      ];
      const result = assembleCockpitConstructionReconciliation({
        inputs,
        reconciled: [primary, atomic],
        maxMemories: 10,
      });

      expect(result.memories, field).toHaveLength(2);
      expect(
        result.repairCounts.coalesced_compatible_semantic_binding,
        field,
      ).toBeUndefined();
    }
  });

  it("fails closed instead of assigning one generic proposal to either of two people", () => {
    const source = "u-two-people-same-value";
    const episode = "shared-destination-request";
    const generic = memory({ source, value: "首都机场", episode, refs: ["primary:0"] });
    const zhou = memory({ source, value: "首都机场", episode, refs: ["atomic:zhou"] });
    const li = memory({ source, value: "首都机场", episode, refs: ["atomic:li"] });
    generic.metadata.state_key = "navigation|unspecified-subject|car|driver|destination";
    for (const [row, person] of [[zhou, "周宁"], [li, "李然"]] as const) {
      row.metadata.subject = person;
      row.metadata.state_key = `navigation|${person}|car|driver|destination`;
    }
    const inputs = [
      { id: "primary:0", memory: generic },
      { id: "atomic:zhou", memory: zhou },
      { id: "atomic:li", memory: li },
    ];

    for (const reconciled of [
      [generic, zhou, li],
      [li, generic, zhou],
    ]) {
      const result = assembleCockpitConstructionReconciliation({
        inputs,
        reconciled,
        maxMemories: 10,
      });
      expect(result.memories).toHaveLength(3);
      const genericResult = result.memories.find((row) =>
        row.metadata.state_key === "navigation|unspecified-subject|car|driver|destination"
      );
      expect(genericResult?.metadata).toMatchObject({
        input_candidate_ids: ["primary:0"],
        construction_quality: {
          status: "partial",
          issues: expect.arrayContaining(["ambiguous_semantic_binding_alias"]),
          repairs: expect.arrayContaining(["preserved_ambiguous_semantic_binding_alias"]),
        },
      });
      expect(result.memories.find((row) => row.metadata.subject === "周宁")
        ?.metadata.input_candidate_ids).toEqual(["atomic:zhou"]);
      expect(result.memories.find((row) => row.metadata.subject === "李然")
        ?.metadata.input_candidate_ids).toEqual(["atomic:li"]);
      expect(result.repairCounts.preserved_ambiguous_semantic_binding_alias).toBe(1);
      const gate = gateCockpitConstructionReconciliation({
        inputs,
        reconciled: result.memories,
        maxMemories: 10,
        resolvedCandidateIds: result.resolvedCandidateIds,
      });
      expect(gate.accepted).toBe(false);
      expect(gate.issues).toContain("reconciliation_contains_incomplete_memory");
    }
  });

  it("does not let duplicate coalescence erase an ambiguous person-binding blocker", () => {
    const source = "u-ambiguous-duplicate";
    const episode = "ambiguous-duplicate-episode";
    const genericPrimary = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["primary:0"],
    });
    const genericAtomic = memory({
      source,
      value: "首都机场",
      episode,
      refs: ["atomic:generic"],
    });
    for (const row of [genericPrimary, genericAtomic]) {
      row.metadata.state_key = "navigation|unspecified-subject|car|driver|destination";
    }
    const zhou = memory({ source, value: "首都机场", episode, refs: ["atomic:zhou"] });
    const li = memory({ source, value: "首都机场", episode, refs: ["atomic:li"] });
    for (const [row, person] of [[zhou, "周宁"], [li, "李然"]] as const) {
      row.metadata.subject = person;
      row.metadata.state_key = `navigation|${person}|car|driver|destination`;
    }
    const inputs = [
      { id: "primary:0", memory: genericPrimary },
      { id: "atomic:generic", memory: genericAtomic },
      { id: "atomic:zhou", memory: zhou },
      { id: "atomic:li", memory: li },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [genericPrimary, genericAtomic, zhou, li],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(4);
    expect(result.repairCounts.preserved_blocking_duplicate_final_identity).toBe(1);
    expect(result.memories.some((row) =>
      (row.metadata.construction_quality as Record<string, unknown>).status === "partial"
        && ((row.metadata.construction_quality as Record<string, unknown>).issues as string[])
          .includes("ambiguous_semantic_binding_alias")
    )).toBe(true);
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });
    expect(gate.accepted).toBe(false);
    expect(gate.issues).toEqual(expect.arrayContaining([
      "reconciliation_contains_incomplete_memory",
      "reconciliation_duplicate_final_identity",
    ]));
  });

  it("retains an ambiguous advisory blocker even when atomic rows fill maxMemories", () => {
    const source = "u-ambiguous-capacity";
    const episode = "ambiguous-capacity-episode";
    const generic = memory({ source, value: "首都机场", episode, refs: ["primary:0"] });
    generic.metadata.state_key = "navigation|unspecified-subject|car|driver|destination";
    const zhou = memory({ source, value: "首都机场", episode, refs: ["atomic:zhou"] });
    const li = memory({ source, value: "首都机场", episode, refs: ["atomic:li"] });
    for (const [row, person] of [[zhou, "周宁"], [li, "李然"]] as const) {
      row.metadata.subject = person;
      row.metadata.state_key = `navigation|${person}|car|driver|destination`;
    }
    const inputs = [
      { id: "primary:0", memory: generic },
      { id: "atomic:zhou", memory: zhou },
      { id: "atomic:li", memory: li },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [generic, zhou, li],
      maxMemories: 2,
    });

    expect(result.memories).toHaveLength(3);
    expect(result.memories.some(hasAmbiguousBindingIssue)).toBe(true);
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 2,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });
    expect(gate.accepted).toBe(false);
    expect(gate.issues).toEqual(expect.arrayContaining([
      "reconciliation_contains_incomplete_memory",
      "reconciliation_exceeds_max_memories",
    ]));
  });

  it("does not erase an ambiguous binding blocker through a cross-episode alias", () => {
    const source = "u-ambiguous-cross-episode";
    const point = "2026-04-05T10:00:00+08:00";
    const genericAtomic = memory({
      source,
      value: "首都机场",
      episode: "episode-one",
      refs: ["atomic:generic"],
    });
    genericAtomic.metadata.state_key = "navigation|unspecified-subject|car|driver|destination";
    const zhou = memory({
      source,
      value: "首都机场",
      episode: "episode-one",
      refs: ["primary:zhou"],
    });
    const li = memory({
      source,
      value: "首都机场",
      episode: "episode-one",
      refs: ["primary:li"],
    });
    for (const [row, person] of [[zhou, "周宁"], [li, "李然"]] as const) {
      row.metadata.subject = person;
      row.metadata.state_key = `navigation|${person}|car|driver|destination`;
    }
    const genericOtherEpisode = memory({
      source,
      value: "首都机场",
      episode: "episode-two",
      refs: ["primary:other"],
    });
    genericOtherEpisode.metadata.state_key =
      "navigation|unspecified-subject|car|driver|destination";
    for (const row of [genericAtomic, zhou, li, genericOtherEpisode]) {
      row.metadata.valid_from = point;
      row.metadata.mentioned_at = point;
      row.metadata.source_session_id = "session-cross-episode";
    }
    const inputs = [
      { id: "atomic:generic", memory: genericAtomic },
      { id: "primary:zhou", memory: zhou },
      { id: "primary:li", memory: li },
      { id: "primary:other", memory: genericOtherEpisode },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [genericAtomic, zhou, li, genericOtherEpisode],
      maxMemories: 10,
    });

    expect(result.memories.some(hasAmbiguousBindingIssue)).toBe(true);
    expect(result.repairCounts.coalesced_cross_proposal_episode_alias).toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not consume an ambiguous cancelled binding as a cancel-replace pair", () => {
    const source = "u-ambiguous-cancel-replace";
    const episode = "ambiguous-cancel-replace";
    const prior = memory({ source: "u-old", value: "旧地点", episode });
    prior.metadata.state_key = "navigation|unspecified-subject|car|driver|destination";
    const genericCancellation = memory({
      source,
      value: "旧地点",
      episode,
      relation: "cancelled",
      refs: ["atomic:cancel"],
    });
    genericCancellation.metadata.state_key =
      "navigation|unspecified-subject|car|driver|destination";
    genericCancellation.metadata.supersedes = ["prior-generic"];
    const zhouCancellation = memory({
      source,
      value: "旧地点",
      episode,
      relation: "cancelled",
      refs: ["primary:zhou"],
    });
    const liCancellation = memory({
      source,
      value: "旧地点",
      episode,
      relation: "cancelled",
      refs: ["primary:li"],
    });
    for (const [row, person] of [
      [zhouCancellation, "周宁"],
      [liCancellation, "李然"],
    ] as const) {
      row.metadata.subject = person;
      row.metadata.state_key = `navigation|${person}|car|driver|destination`;
      row.metadata.supersedes = ["prior-generic"];
    }
    const replacement = memory({
      source,
      value: "新地点",
      episode,
      refs: ["atomic:replacement"],
    });
    replacement.metadata.state_key =
      "navigation|unspecified-subject|car|driver|destination";
    const inputs = [
      { id: "atomic:cancel", memory: genericCancellation },
      { id: "primary:zhou", memory: zhouCancellation },
      { id: "primary:li", memory: liCancellation },
      { id: "atomic:replacement", memory: replacement },
    ];
    const priors = [{
      record_id: "prior-generic",
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [
        genericCancellation,
        zhouCancellation,
        liCancellation,
        replacement,
      ],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories.some(hasAmbiguousBindingIssue)).toBe(true);
    expect(result.resolvedCandidateIds).not.toContain("atomic:cancel");
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("preserves conflicting predecessor chains for the gate to reject", () => {
    const first = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u1",
      episode: "episode-predecessor-conflict",
      relation: "updated",
      refs: ["atomic:0"],
      supersedes: ["prior-1"],
    });
    const second = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u2",
      episode: "episode-predecessor-conflict",
      relation: "updated",
      refs: ["atomic:1"],
      supersedes: ["prior-2"],
    });
    const mentionedAt = "2026-03-16T01:00:00.000Z";
    for (const row of [first, second]) row.metadata.mentioned_at = mentionedAt;
    const inputs = [
      { id: "atomic:0", memory: first },
      { id: "atomic:1", memory: second },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [first, second],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((entry) => entry.metadata.supersedes)).toEqual([
      ["prior-1"],
      ["prior-2"],
    ]);
    expect(result.repairCounts.preserved_conflicting_duplicate_final_identity).toBe(1);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("preserves incompatible temporal interpretations for the gate to reject", () => {
    const first = memory({ source: "u1", value: "A", refs: ["primary:0"] });
    const second = memory({ source: "u1", value: "A", refs: ["atomic:0"] });
    first.metadata.timezone = "Asia/Shanghai";
    second.metadata.timezone = "UTC";
    first.metadata.time_precision = "minute";
    second.metadata.time_precision = "day";
    const inputs = [
      { id: "primary:0", memory: first },
      { id: "atomic:0", memory: second },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [first, second],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_compatible_semantic_binding).toBeUndefined();
    expect(result.repairCounts.preserved_conflicting_duplicate_final_identity).toBe(1);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("coalesces one appointment represented by primary and atomic draft episode keys", () => {
    const source = "u-appointment";
    const point = "2026-04-05T10:00:00+08:00";
    const primaryTime = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "appointment-20260405-1000-zhouning",
      refs: ["primary:time", "coverage:time"],
    });
    const primaryContent = scheduleMemory({
      slot: "appointment_content",
      value: "北京市东城区故宫博物院内的车辆检查",
      source,
      episode: "appointment-20260405-1000-zhouning",
      refs: ["primary:content", "coverage:content"],
    });
    const atomicTime = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "episode-周宁-车辆检查-20260405",
      refs: ["atomic:time"],
    });
    const atomicContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode: "episode-周宁-车辆检查-20260405",
      refs: ["atomic:content"],
    });
    const destination = memory({
      source,
      value: "北京市东城区故宫博物院内",
      episode: "appointment-20260405-1000-zhouning",
      refs: ["atomic:destination", "coverage:destination"],
    });
    for (const item of [primaryTime, primaryContent, atomicTime, atomicContent]) {
      item.metadata.subject = "周宁";
      item.metadata.state_key = `schedule|周宁|car|driver|${item.metadata.slot as string}`;
      item.metadata.valid_from = point;
    }
    destination.metadata.subject = "周宁";
    destination.metadata.state_key = "navigation|周宁|car|driver|destination";
    destination.metadata.valid_from = point;
    const coverageTime = scheduleMemory({
      slot: "appointment_time",
      source,
      status: "partial",
      issues: ["source_coverage_obligation"],
    });
    const coverageContent = scheduleMemory({
      slot: "appointment_content",
      source,
      status: "partial",
      issues: ["source_coverage_obligation"],
    });
    const coverageDestination = memory({
      source,
      status: "partial",
      refs: [],
    });
    delete coverageTime.metadata.value;
    delete coverageContent.metadata.value;
    for (const item of [coverageTime, coverageContent, coverageDestination]) {
      delete item.metadata.subject;
      delete item.metadata.state_key;
      delete item.metadata.episode_key;
      delete item.metadata.action_status;
      delete item.metadata.value;
    }
    const inputs = [
      { id: "primary:time", memory: primaryTime },
      { id: "primary:content", memory: primaryContent },
      { id: "atomic:time", memory: atomicTime },
      { id: "atomic:content", memory: atomicContent },
      { id: "atomic:destination", memory: destination },
      { id: "coverage:time", memory: coverageTime },
      { id: "coverage:content", memory: coverageContent },
      { id: "coverage:destination", memory: coverageDestination },
    ];

    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primaryTime, primaryContent, destination],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(3);
    expect(result.memories.map((entry) => ({
      slot: entry.metadata.slot,
      value: entry.metadata.value,
      episode: entry.metadata.episode_key,
      refs: entry.metadata.input_candidate_ids,
    }))).toEqual([
      {
        slot: "appointment_time",
        value: point,
        episode: "appointment-20260405-1000-zhouning",
        refs: expect.arrayContaining(["primary:time", "atomic:time", "coverage:time"]),
      },
      {
        slot: "appointment_content",
        value: "北京市东城区故宫博物院内的车辆检查",
        episode: "appointment-20260405-1000-zhouning",
        refs: expect.arrayContaining(["primary:content", "atomic:content", "coverage:content"]),
      },
      {
        slot: "destination",
        value: "北京市东城区故宫博物院内",
        episode: "appointment-20260405-1000-zhouning",
        refs: expect.arrayContaining(["atomic:destination", "coverage:destination"]),
      },
    ]);
    expect(result.repairCounts).toMatchObject({
      coalesced_cross_proposal_episode_alias: 2,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not cross-episode alias non-equal appointment values without complete atomic destination proof", () => {
    const source = "u-cross-episode-no-destination";
    const point = "2026-04-05T10:00:00+08:00";
    const rich = scheduleMemory({
      slot: "appointment_content",
      value: "北京市东城区故宫博物院内的车辆检查",
      source,
      episode: "primary-episode",
      refs: ["primary:content"],
    });
    const coarse = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode: "atomic-episode",
      refs: ["atomic:content"],
    });
    for (const row of [rich, coarse]) {
      row.metadata.subject = "周宁";
      row.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
      row.metadata.target = row.metadata.value;
      row.metadata.valid_from = point;
    }
    const inputs = [
      { id: "primary:content", memory: rich },
      { id: "atomic:content", memory: coarse },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [rich, coarse],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_cross_proposal_episode_alias).toBeUndefined();
    expect(result.resolvedCandidateIds).toEqual([]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not cross-episode alias a rich appointment with an inconsistent target", () => {
    const source = "u-cross-episode-target-mismatch";
    const point = "2026-04-05T10:00:00+08:00";
    const rich = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内进行车辆检查",
      source,
      episode: "primary-episode",
      refs: ["primary:content"],
    });
    const coarse = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode: "atomic-episode",
      refs: ["atomic:content"],
    });
    rich.metadata.target = "车辆检查";
    coarse.metadata.target = coarse.metadata.value;
    const destination = memory({
      source,
      value: "故宫博物院",
      episode: "primary-episode",
      refs: ["atomic:destination"],
    });
    destination.metadata.target = destination.metadata.value;
    for (const row of [rich, coarse, destination]) row.metadata.valid_from = point;
    const inputs = [
      { id: "primary:content", memory: rich },
      { id: "atomic:content", memory: coarse },
      { id: "atomic:destination", memory: destination },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [rich, coarse, destination],
      maxMemories: 10,
    });

    expect(result.memories.filter((entry) =>
      entry.metadata.slot === "appointment_content"
    )).toHaveLength(2);
    expect(result.repairCounts.coalesced_cross_proposal_episode_alias).toBeUndefined();
  });

  it("closes a destination qualifier from one complete co-episodic appointment slot", () => {
    const source = "u-appointment-location";
    const episode = "vehicle-check-appointment";
    const destination = memory({
      source,
      value: "北京市东城区故宫博物院",
      episode,
      refs: ["atomic:destination"],
    });
    destination.content = "周宁车辆检查预约的行驶目的地为北京市东城区故宫博物院";
    destination.metadata.subject = "周宁";
    destination.metadata.state_key = "navigation|周宁|car|driver|destination";
    destination.metadata.target = "北京市东城区故宫博物院";
    const appointment = scheduleMemory({
      slot: "appointment_content",
      value: "在北京市东城区故宫博物院内做车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    appointment.metadata.subject = "周宁";
    appointment.metadata.state_key = "schedule|周宁|car|driver|appointment_content";

    const inputs = [
      { id: "atomic:destination", memory: destination },
      { id: "atomic:content", memory: appointment },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [destination, appointment],
      maxMemories: 10,
    });
    const closedDestination = result.memories.find((entry) => entry.metadata.slot === "destination");
    const retainedAppointment = result.memories.find((entry) => entry.metadata.slot === "appointment_content");

    expect(closedDestination).toMatchObject({
      content: "周宁车辆检查预约的行驶目的地为北京市东城区故宫博物院内",
      metadata: {
        value: "北京市东城区故宫博物院内",
        target: "北京市东城区故宫博物院内",
        construction_quality: {
          repairs: expect.arrayContaining([
            "closed_coepisodic_destination_qualifier_from_appointment_content",
          ]),
        },
      },
    });
    // The evidence-faithful activity wording is not rewritten merely to make
    // model phrasing deterministic.
    expect(retainedAppointment?.metadata.value).toBe("在北京市东城区故宫博物院内做车辆检查");
    expect(result.repairCounts).toMatchObject({
      closed_coepisodic_destination_qualifier_from_appointment_content: 1,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("coalesces a same-event coarse atomic appointment into the richer primary before closing its destination", () => {
    const source = "u-appointment-mixed-proposals";
    const episode = "episode-周宁-车辆检查-20260405";
    const mentionedAt = "2026-03-16T01:00:00.000Z";
    const detailed = scheduleMemory({
      slot: "appointment_content",
      value: "在北京市东城区故宫博物院内进行车辆检查",
      source,
      episode,
      refs: ["primary:content"],
    });
    detailed.content = "周宁车辆检查的预约内容为在北京市东城区故宫博物院内进行车辆检查";
    detailed.metadata.subject = "周宁";
    detailed.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
    detailed.metadata.target = "在北京市东城区故宫博物院内进行车辆检查";
    detailed.metadata.source_session_id = "session-appointment";
    detailed.metadata.mentioned_at = mentionedAt;

    const coarse = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    coarse.content = "周宁预约了车辆检查";
    coarse.metadata.subject = "周宁";
    coarse.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
    coarse.metadata.target = "车辆检查";
    coarse.metadata.source_session_id = "session-appointment";
    coarse.metadata.mentioned_at = mentionedAt;
    coarse.metadata.valid_from = mentionedAt;

    const destination = memory({
      source,
      value: "北京市东城区故宫博物院",
      episode,
      refs: ["atomic:destination"],
      mentionedAt,
    });
    destination.content = "周宁车辆检查的预约地点为北京市东城区故宫博物院";
    destination.metadata.subject = "周宁";
    destination.metadata.state_key = "navigation|周宁|car|driver|destination";
    destination.metadata.target = "北京市东城区故宫博物院";
    destination.metadata.source_session_id = "session-appointment";

    const inputs = [
      { id: "primary:content", memory: detailed },
      { id: "atomic:content", memory: coarse },
      { id: "atomic:destination", memory: destination },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      // This reproduces a valid reconciler response that selected the richer
      // primary row while leaving the complete coarse atomic obligation for
      // deterministic restoration.
      reconciled: [detailed, destination],
      maxMemories: 10,
    });
    const contents = result.memories.filter((entry) =>
      entry.metadata.slot === "appointment_content"
    );
    const retainedDestination = result.memories.find((entry) =>
      entry.metadata.slot === "destination"
    );

    expect(result.memories).toHaveLength(2);
    expect(contents).toHaveLength(1);
    expect(contents[0]).toMatchObject({
      content: detailed.content,
      metadata: {
        value: detailed.metadata.value,
        target: detailed.metadata.target,
        input_candidate_ids: expect.arrayContaining(["primary:content", "atomic:content"]),
        construction_quality: {
          repairs: expect.arrayContaining([
            "coalesced_same_episode_cross_proposal_value_alias",
          ]),
        },
      },
    });
    expect(retainedDestination?.metadata).toMatchObject({
      value: "北京市东城区故宫博物院内",
      target: "北京市东城区故宫博物院内",
    });
    expect(result.repairCounts).toMatchObject({
      coalesced_same_episode_cross_proposal_value_alias: 1,
      closed_coepisodic_destination_qualifier_from_appointment_content: 1,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("coalesces rich and coarse same-identity rows before identity ranking in either input order", () => {
    const source = "u-same-identity-order";
    const episode = "episode-order-invariant";
    const mentionedAt = "2026-03-16T01:00:00.000Z";
    const detailed = scheduleMemory({
      slot: "appointment_content",
      value: "在北京市东城区故宫博物院内进行车辆检查",
      source,
      episode,
      refs: ["primary:content"],
    });
    detailed.content = "周宁车辆检查的预约内容为在北京市东城区故宫博物院内进行车辆检查";
    detailed.metadata.subject = "周宁";
    detailed.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
    detailed.metadata.target = detailed.metadata.value;
    detailed.metadata.source_session_id = "session-order";
    detailed.metadata.mentioned_at = mentionedAt;
    detailed.metadata.valid_from = mentionedAt;
    const coarse = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    coarse.metadata.subject = "周宁";
    coarse.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
    coarse.metadata.target = coarse.metadata.value;
    coarse.metadata.source_session_id = "session-order";
    coarse.metadata.mentioned_at = mentionedAt;
    coarse.metadata.valid_from = mentionedAt;
    const destination = memory({
      source,
      value: "北京市东城区故宫博物院",
      episode,
      refs: ["atomic:destination"],
      mentionedAt,
    });
    destination.metadata.subject = "周宁";
    destination.metadata.state_key = "navigation|周宁|car|driver|destination";
    destination.metadata.target = destination.metadata.value;
    destination.metadata.source_session_id = "session-order";
    const inputs = [
      { id: "primary:content", memory: detailed },
      { id: "atomic:content", memory: coarse },
      { id: "atomic:destination", memory: destination },
    ];

    for (const reconciled of [
      [coarse, detailed, destination],
      [detailed, coarse, destination],
    ]) {
      const result = assembleCockpitConstructionReconciliation({
        inputs,
        reconciled,
        maxMemories: 10,
      });
      const contents = result.memories.filter((entry) =>
        entry.metadata.slot === "appointment_content"
      );
      expect(contents).toHaveLength(1);
      expect(contents[0]).toMatchObject({
        content: detailed.content,
        metadata: {
          value: detailed.metadata.value,
          input_candidate_ids: expect.arrayContaining(["primary:content", "atomic:content"]),
        },
      });
      expect(result.resolvedCandidateIds).toEqual([]);
      expect(gateCockpitConstructionReconciliation({
        inputs,
        reconciled: result.memories,
        maxMemories: 10,
        resolvedCandidateIds: result.resolvedCandidateIds,
      }).accepted).toBe(true);
    }
  });

  it("fails closed instead of aliasing a negated rich appointment to a positive activity", () => {
    const source = "u-negated-alias";
    const episode = "episode-negated-alias";
    const mentionedAt = "2026-03-16T01:00:00.000Z";
    const negated = scheduleMemory({
      slot: "appointment_content",
      value: "取消在北京市东城区故宫博物院内进行车辆检查",
      source,
      episode,
      refs: ["primary:content"],
    });
    const coarse = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    for (const row of [negated, coarse]) {
      row.metadata.subject = "周宁";
      row.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
      row.metadata.target = row.metadata.value;
      row.metadata.source_session_id = "session-negated";
      row.metadata.mentioned_at = mentionedAt;
      row.metadata.valid_from = mentionedAt;
    }
    const destination = memory({
      source,
      value: "北京市东城区故宫博物院",
      episode,
      refs: ["atomic:destination"],
      mentionedAt,
    });
    destination.metadata.subject = "周宁";
    destination.metadata.state_key = "navigation|周宁|car|driver|destination";
    destination.metadata.target = destination.metadata.value;
    destination.metadata.source_session_id = "session-negated";
    const inputs = [
      { id: "primary:content", memory: negated },
      { id: "atomic:content", memory: coarse },
      { id: "atomic:destination", memory: destination },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [negated, coarse, destination],
      maxMemories: 10,
    });

    expect(result.memories.filter((entry) =>
      entry.metadata.slot === "appointment_content"
    )).toHaveLength(2);
    expect(result.repairCounts.coalesced_same_episode_cross_proposal_value_alias)
      .toBeUndefined();
    expect(result.repairCounts.preserved_conflicting_duplicate_final_identity).toBe(1);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("requires a unique exact destination candidate before coalescing appointment values", () => {
    const source = "u-ambiguous-destination-alias";
    const episode = "episode-ambiguous-destination";
    const mentionedAt = "2026-03-16T01:00:00.000Z";
    const detailed = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内进行车辆检查",
      source,
      episode,
      refs: ["primary:content"],
    });
    const coarse = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    for (const row of [detailed, coarse]) {
      row.metadata.subject = "周宁";
      row.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
      row.metadata.target = row.metadata.value;
      row.metadata.source_session_id = "session-ambiguous";
      row.metadata.mentioned_at = mentionedAt;
      row.metadata.valid_from = mentionedAt;
    }
    const destinations = ["故宫博物院", "国家博物馆"].map((value, index) => {
      const destination = memory({
        source,
        value,
        episode,
        refs: [`atomic:destination-${index}`],
        mentionedAt,
      });
      destination.metadata.subject = "周宁";
      destination.metadata.state_key = "navigation|周宁|car|driver|destination";
      destination.metadata.target = value;
      destination.metadata.source_session_id = "session-ambiguous";
      return destination;
    });
    const inputs = [
      { id: "primary:content", memory: detailed },
      { id: "atomic:content", memory: coarse },
      ...destinations.map((memory, index) => ({ id: `atomic:destination-${index}`, memory })),
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [detailed, coarse, ...destinations],
      maxMemories: 10,
    });

    expect(result.memories.filter((entry) =>
      entry.metadata.slot === "appointment_content"
    )).toHaveLength(2);
    expect(result.repairCounts.coalesced_same_episode_cross_proposal_value_alias)
      .toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not merge containing appointment values from two same-class or different-time events", () => {
    const first = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u1",
      episode: "appointment-a",
      refs: ["atomic:first"],
    });
    const second = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内进行车辆检查",
      source: "u1",
      episode: "appointment-a",
      refs: ["atomic:second"],
    });
    first.metadata.target = first.metadata.value;
    second.metadata.target = second.metadata.value;
    first.metadata.mentioned_at = "2026-03-16T01:00:00.000Z";
    second.metadata.mentioned_at = "2026-03-16T01:01:00.000Z";
    first.metadata.valid_from = first.metadata.mentioned_at;
    second.metadata.valid_from = second.metadata.mentioned_at;

    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:first", memory: first },
        { id: "atomic:second", memory: second },
      ],
      reconciled: [first, second],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_same_episode_cross_proposal_value_alias)
      .toBeUndefined();
  });

  it("does not close a destination qualifier across episodes or people", () => {
    const destination = memory({
      source: "u1",
      value: "故宫博物院",
      episode: "appointment-a",
      refs: ["atomic:destination"],
    });
    destination.content = "目的地为故宫博物院";
    destination.metadata.target = "故宫博物院";
    destination.metadata.subject = "周宁";
    destination.metadata.state_key = "navigation|周宁|car|driver|destination";
    const wrongEpisode = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内做车辆检查",
      source: "u1",
      episode: "appointment-b",
      refs: ["atomic:content-a"],
    });
    wrongEpisode.metadata.subject = "周宁";
    wrongEpisode.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
    const wrongPerson = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内做车辆检查",
      source: "u1",
      episode: "appointment-a",
      refs: ["atomic:content-b"],
    });
    wrongPerson.metadata.subject = "李然";
    wrongPerson.metadata.state_key = "schedule|李然|car|driver|appointment_content";

    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:destination", memory: destination },
        { id: "atomic:content-a", memory: wrongEpisode },
        { id: "atomic:content-b", memory: wrongPerson },
      ],
      reconciled: [destination, wrongEpisode, wrongPerson],
      maxMemories: 10,
    });

    expect(result.memories.find((entry) => entry.metadata.slot === "destination")?.metadata.value)
      .toBe("故宫博物院");
    expect(result.repairCounts.closed_coepisodic_destination_qualifier_from_appointment_content)
      .toBeUndefined();
  });

  it("does not mistake the first character of a longer place name for a destination qualifier", () => {
    const destination = memory({
      source: "u1",
      value: "北京",
      episode: "trip-a",
      refs: ["atomic:destination"],
    });
    destination.content = "目的地为北京";
    destination.metadata.target = "北京";
    const appointment = scheduleMemory({
      slot: "appointment_content",
      value: "在北京内蒙古饭店参加会议",
      source: "u1",
      episode: "trip-a",
      refs: ["atomic:content"],
    });

    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:destination", memory: destination },
        { id: "atomic:content", memory: appointment },
      ],
      reconciled: [destination, appointment],
      maxMemories: 10,
    });

    expect(result.memories.find((entry) => entry.metadata.slot === "destination")?.metadata.value)
      .toBe("北京");
    expect(result.repairCounts.closed_coepisodic_destination_qualifier_from_appointment_content)
      .toBeUndefined();
  });

  it("closes a bare appointment activity from one exact co-episodic destination", () => {
    const source = "u-appointment-activity";
    const episode = "vehicle-check-appointment";
    const appointment = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    appointment.content = "周宁的车辆检查预约内容为车辆检查";
    appointment.metadata.subject = "周宁";
    appointment.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
    appointment.metadata.target = "车辆检查";
    appointment.metadata.source_session_id = "session-appointment";
    appointment.metadata.mentioned_at = "2026-03-16T01:00:00.000Z";
    const destination = memory({
      source,
      value: "北京市东城区故宫博物院内",
      episode,
      refs: ["atomic:destination"],
    });
    destination.content = "周宁车辆检查预约的行驶目的地为北京市东城区故宫博物院内";
    destination.metadata.subject = "周宁";
    destination.metadata.state_key = "navigation|周宁|car|driver|destination";
    destination.metadata.target = "北京市东城区故宫博物院内";
    destination.metadata.source_session_id = "session-appointment";
    destination.metadata.mentioned_at = "2026-03-16T01:00:00.000Z";

    const inputs = [
      { id: "atomic:content", memory: appointment },
      { id: "atomic:destination", memory: destination },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [appointment, destination],
      maxMemories: 10,
    });
    const closedAppointment = result.memories.find((entry) =>
      entry.metadata.slot === "appointment_content"
    );

    expect(closedAppointment).toMatchObject({
      content: "周宁的车辆检查预约内容为在北京市东城区故宫博物院内进行车辆检查",
      metadata: {
        value: "在北京市东城区故宫博物院内进行车辆检查",
        target: "在北京市东城区故宫博物院内进行车辆检查",
        construction_quality: {
          repairs: expect.arrayContaining([
            "closed_coepisodic_appointment_content_location_from_destination",
          ]),
        },
      },
    });
    expect(result.repairCounts).toMatchObject({
      closed_coepisodic_appointment_content_location_from_destination: 1,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not close appointment content across evidence, episodes, people, or ambiguity", () => {
    const appointment = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u1",
      episode: "appointment-a",
      refs: ["atomic:content"],
    });
    appointment.content = "预约内容为车辆检查";
    appointment.metadata.target = "车辆检查";
    appointment.metadata.subject = "周宁";
    appointment.metadata.state_key = "schedule|周宁|car|driver|appointment_content";
    appointment.metadata.source_session_id = "session-a";
    appointment.metadata.mentioned_at = "2026-03-16T01:00:00.000Z";
    const wrongEvidence = memory({
      source: "u2",
      value: "地点甲",
      episode: "appointment-a",
      refs: ["atomic:destination-a"],
    });
    const wrongEpisode = memory({
      source: "u1",
      value: "地点乙",
      episode: "appointment-b",
      refs: ["atomic:destination-b"],
    });
    const wrongPerson = memory({
      source: "u1",
      value: "地点丙",
      episode: "appointment-a",
      refs: ["atomic:destination-c"],
    });
    for (const destination of [wrongEvidence, wrongEpisode, wrongPerson]) {
      destination.metadata.target = destination.metadata.value;
      destination.metadata.source_session_id = "session-a";
      destination.metadata.mentioned_at = "2026-03-16T01:00:00.000Z";
    }
    wrongEvidence.metadata.subject = "周宁";
    wrongEvidence.metadata.state_key = "navigation|周宁|car|driver|destination";
    wrongEpisode.metadata.subject = "周宁";
    wrongEpisode.metadata.state_key = "navigation|周宁|car|driver|destination";
    wrongPerson.metadata.subject = "李然";
    wrongPerson.metadata.state_key = "navigation|李然|car|driver|destination";

    const inputs = [
      { id: "atomic:content", memory: appointment },
      { id: "atomic:destination-a", memory: wrongEvidence },
      { id: "atomic:destination-b", memory: wrongEpisode },
      { id: "atomic:destination-c", memory: wrongPerson },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [appointment, wrongEvidence, wrongEpisode, wrongPerson],
      maxMemories: 10,
    });

    expect(result.memories.find((entry) => entry.metadata.slot === "appointment_content")
      ?.metadata.value).toBe("车辆检查");
    expect(result.repairCounts.closed_coepisodic_appointment_content_location_from_destination)
      .toBeUndefined();
  });

  it("does not guess between two otherwise valid co-episodic destinations", () => {
    const appointment = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u1",
      episode: "appointment-a",
      refs: ["atomic:content"],
    });
    appointment.content = "预约内容为车辆检查";
    appointment.metadata.target = "车辆检查";
    appointment.metadata.source_session_id = "session-a";
    appointment.metadata.mentioned_at = "2026-03-16T01:00:00.000Z";
    const first = memory({
      source: "u1",
      value: "地点甲",
      episode: "appointment-a",
      refs: ["atomic:destination-a"],
    });
    const second = memory({
      source: "u1",
      value: "地点乙",
      episode: "appointment-a",
      refs: ["atomic:destination-b"],
    });
    for (const destination of [first, second]) {
      destination.metadata.target = destination.metadata.value;
      destination.metadata.source_session_id = "session-a";
      destination.metadata.mentioned_at = "2026-03-16T01:00:00.000Z";
    }

    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:content", memory: appointment },
        { id: "atomic:destination-a", memory: first },
        { id: "atomic:destination-b", memory: second },
      ],
      reconciled: [appointment, first, second],
      maxMemories: 10,
    });

    expect(result.memories.find((entry) => entry.metadata.slot === "appointment_content")
      ?.metadata.value).toBe("车辆检查");
    expect(result.repairCounts.closed_coepisodic_appointment_content_location_from_destination)
      .toBeUndefined();
  });

  it("does not guess an episode alias when proposal pairing is ambiguous", () => {
    const point = "2026-04-05T10:00:00+08:00";
    const firstPrimary = scheduleMemory({
      slot: "appointment_time",
      value: point,
      episode: "appointment-a",
      refs: ["primary:a"],
    });
    const secondPrimary = scheduleMemory({
      slot: "appointment_time",
      value: point,
      episode: "appointment-b",
      refs: ["primary:b"],
    });
    const atomic = scheduleMemory({
      slot: "appointment_time",
      value: point,
      episode: "appointment-atomic",
      refs: ["atomic:a"],
    });
    for (const item of [firstPrimary, secondPrimary, atomic]) item.metadata.valid_from = point;
    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "primary:a", memory: firstPrimary },
        { id: "primary:b", memory: secondPrimary },
        { id: "atomic:a", memory: atomic },
      ],
      reconciled: [firstPrimary, secondPrimary],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(3);
    expect(result.repairCounts).not.toHaveProperty("coalesced_cross_proposal_episode_alias");
  });

  it("does not merge distinct named people who share one value and source", () => {
    const first = memory({ source: "u1", value: "24", episode: "shared-climate" });
    const second = memory({ source: "u1", value: "24", episode: "shared-climate" });
    first.metadata.subject = "冯遥";
    first.metadata.state_key = "navigation|冯遥|car|driver|destination";
    first.metadata.input_candidate_ids = ["atomic:0"];
    second.metadata.subject = "尤遥";
    second.metadata.state_key = "navigation|尤遥|car|driver|destination";
    second.metadata.input_candidate_ids = ["atomic:1"];

    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:0", memory: first },
        { id: "atomic:1", memory: second },
      ],
      reconciled: [first, second],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((entry) => entry.metadata.subject)).toEqual(["冯遥", "尤遥"]);
  });

  it("rejects silent loss of a partial atomic source obligation", () => {
    const partialAtomic = memory({ status: "partial" });
    const gate = gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: partialAtomic }],
      reconciled: [],
      maxMemories: 10,
    });

    expect(gate.accepted).toBe(false);
    expect(gate.requiredCandidateIds).toEqual(["atomic:0"]);
    expect(gate.uncoveredCandidateIds).toEqual(["atomic:0"]);
    expect(gate.issues).toEqual(expect.arrayContaining([
      "reconciliation_empty_with_input_candidates",
      "reconciliation_uncovered_atomic_candidate",
    ]));
  });

  it("coalesces duplicate identities into the latest evidence-backed state", () => {
    const oldAtomic = memory({ source: "u1", value: "A", mentionedAt: "2026-01-01T00:00:00.000Z" });
    const newAtomic = memory({ source: "u2", value: "B", mentionedAt: "2026-01-01T00:01:00.000Z" });
    const result = assembleCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:0", memory: oldAtomic },
        { id: "atomic:1", memory: newAtomic },
      ],
      reconciled: [oldAtomic, newAtomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.value).toBe("B");
    expect(result.memories[0].source_message_ids).toEqual(["u1", "u2"]);
    expect(result.memories[0].metadata.input_candidate_ids).toEqual(["atomic:0", "atomic:1"]);
    expect(gateCockpitConstructionReconciliation({
      inputs: [
        { id: "atomic:0", memory: oldAtomic },
        { id: "atomic:1", memory: newAtomic },
      ],
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not accept a later changed value without an exact new-value candidate anchor", () => {
    const oldAtomic = memory({
      source: "u1",
      value: "A",
      mentionedAt: "2026-01-01T00:00:00.000Z",
    });
    const unsupportedNewOutput = memory({
      source: "u2",
      value: "B",
      refs: ["atomic:old"],
      mentionedAt: "2026-01-01T00:01:00.000Z",
    });
    unsupportedNewOutput.source_message_ids = ["u1", "u2"];
    unsupportedNewOutput.metadata.source_message_ids = ["u1", "u2"];
    const inputs = [{ id: "atomic:old", memory: oldAtomic }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [unsupportedNewOutput],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.value).toBe("A");
    expect(result.memories[0].source_message_ids).toEqual(["u1"]);
    expect(result.memories[0].metadata.input_candidate_ids).toContain("atomic:old");
    expect(result.repairCounts.restored_complete_atomic_obligation).toBe(1);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("suppresses an exact live-state reassertion while resolving its obligation", () => {
    const atomic = memory({ refs: ["atomic:0"] });
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: atomic }],
      reconciled: [atomic],
      maxMemories: 10,
      priorMemories: [{ record_id: "live-1", metadata: atomic.metadata as Record<string, unknown> }],
    });

    expect(result.memories).toEqual([]);
    expect(result.resolvedCandidateIds).toEqual(["atomic:0"]);
    expect(gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: atomic }],
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: [{ record_id: "live-1", metadata: atomic.metadata as Record<string, unknown> }],
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("retains an unchanged live state when it carries new evidence", () => {
    const prior = memory({ source: "u0", value: "A" });
    const repeated = memory({ source: "u1", value: "A", refs: ["atomic:0"] });
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: repeated }],
      reconciled: [repeated],
      maxMemories: 10,
      priorMemories: [{ record_id: "live-1", metadata: prior.metadata as Record<string, unknown> }],
    });

    expect(result.memories).toHaveLength(1);
    expect(result.resolvedCandidateIds).toEqual([]);
    expect(result.repairCounts).toMatchObject({ retained_new_evidence_for_live_state: 1 });
    expect(gateCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: repeated }],
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: [{ record_id: "live-1", metadata: prior.metadata as Record<string, unknown> }],
    }).accepted).toBe(true);
  });

  it.each([
    {
      label: "a new activity start time",
      mutate: (_prior: ExtractedMemory, repeated: ExtractedMemory) => {
        repeated.metadata.activity_start_time = "2026-04-01T09:30:00+08:00";
      },
    },
    {
      label: "a new target",
      mutate: (_prior: ExtractedMemory, repeated: ExtractedMemory) => {
        repeated.metadata.target = "A";
      },
    },
    {
      label: "a new temporal status",
      mutate: (_prior: ExtractedMemory, repeated: ExtractedMemory) => {
        repeated.metadata.temporal_status = "active";
      },
    },
    {
      label: "a different scalar source session",
      mutate: (prior: ExtractedMemory, repeated: ExtractedMemory) => {
        prior.metadata.source_session_id = "session-old";
        repeated.metadata.source_session_id = "session-new";
      },
    },
    {
      label: "a corrected event time",
      mutate: (prior: ExtractedMemory, repeated: ExtractedMemory) => {
        prior.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
        repeated.metadata.mentioned_at = "2026-04-01T09:01:00+08:00";
      },
    },
  ])("does not suppress a live-state reassertion carrying $label", ({ mutate }) => {
    const prior = memory({ source: "u-same", value: "A" });
    const repeated = memory({ source: "u-same", value: "A", refs: ["atomic:0"] });
    mutate(prior, repeated);
    const inputs = [{ id: "atomic:0", memory: repeated }];
    const priors = [{
      record_id: "live-1",
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [repeated],
      maxMemories: 10,
      priorMemories: priors,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.resolvedCandidateIds).toEqual([]);
    expect(gate.accepted).toBe(true);
  });

  it("promotes a changed assertion over an exact live identity to an update", () => {
    const prior = memory({ value: "A" });
    const changed = memory({ value: "B", refs: ["atomic:0"] });
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: changed }],
      reconciled: [changed],
      maxMemories: 10,
      priorMemories: [{ record_id: "live-1", metadata: prior.metadata as Record<string, unknown> }],
    });

    expect(result.memories[0].metadata.relation).toBe("updated");
    expect(result.memories[0].metadata.supersedes).toEqual(["live-1"]);
  });

  it("promotes across equivalent controlled live priors despite legacy scene drift", () => {
    const firstPrior = memory({ value: "A" });
    const secondPrior = structuredClone(firstPrior);
    const changed = memory({ value: "B", refs: ["atomic:0"] });
    const priors = [
      {
        record_id: "live-legacy-a",
        type: firstPrior.type,
        scene_name: "legacy-route-session-a",
        metadata: firstPrior.metadata as Record<string, unknown>,
      },
      {
        record_id: "live-legacy-b",
        type: secondPrior.type,
        scene_name: "legacy-route-session-b",
        metadata: secondPrior.metadata as Record<string, unknown>,
      },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: changed }],
      reconciled: [changed],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories[0].metadata.relation).toBe("updated");
    expect(result.memories[0].metadata.supersedes).toEqual([
      "live-legacy-a",
      "live-legacy-b",
    ]);
  });

  it("does not collapse uncontrolled live priors with different scene labels", () => {
    const firstPrior = memory({ value: "A" });
    const secondPrior = structuredClone(firstPrior);
    const changed = memory({ value: "B", refs: ["atomic:0"] });
    for (const row of [firstPrior, secondPrior, changed]) {
      row.metadata.domain = "uncontrolled-domain";
      row.metadata.slot = "uncontrolled-slot";
      row.metadata.state_key = "uncontrolled-domain|user|car|driver|uncontrolled-slot";
    }
    firstPrior.scene_name = "uncontrolled-scene-a";
    secondPrior.scene_name = "uncontrolled-scene-b";
    changed.scene_name = "uncontrolled-scene-a";
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: changed }],
      reconciled: [changed],
      maxMemories: 10,
      priorMemories: [
        {
          record_id: "uncontrolled-live-a",
          type: firstPrior.type,
          scene_name: firstPrior.scene_name,
          metadata: firstPrior.metadata as Record<string, unknown>,
        },
        {
          record_id: "uncontrolled-live-b",
          type: secondPrior.type,
          scene_name: secondPrior.scene_name,
          metadata: secondPrior.metadata as Record<string, unknown>,
        },
      ],
    });

    expect(result.memories[0].metadata.relation).toBe("updated");
    expect(result.memories[0].metadata.supersedes).toEqual(["uncontrolled-live-a"]);
  });

  it("rebinds a transition to the exact live predecessor set", () => {
    const prior = memory({ value: "A" });
    const cancelled = memory({ relation: "cancelled", refs: ["atomic:0"] });
    cancelled.metadata.supersedes = ["stale-id"];
    const result = assembleCockpitConstructionReconciliation({
      inputs: [{ id: "atomic:0", memory: cancelled }],
      reconciled: [cancelled],
      maxMemories: 10,
      priorMemories: [{ record_id: "live-1", metadata: prior.metadata as Record<string, unknown> }],
    });

    expect(result.memories[0].metadata.supersedes).toEqual(["live-1"]);
  });

  it("repairs a replacement transaction into specific updated atomic edges", () => {
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode: "vehicle-inspection-appointment",
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "在旧地点做车辆检查",
      source: "u-old",
      episode: "vehicle-inspection-appointment",
    });
    const composite = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-new",
      episode: "vehicle-inspection-appointment",
      relation: "cancelled",
      status: "partial",
      issues: ["ambiguous_transition_state"],
      supersedes: ["vehicle-inspection-appointment"],
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode: "replacement-draft",
      relation: "updated",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const updatedContent = scheduleMemory({
      slot: "appointment_content",
      value: "在新地点做车辆检查",
      source: "u-new",
      episode: "replacement-draft",
      relation: "updated",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const coverageTime = scheduleMemory({
      slot: "appointment_time",
      source: "u-new",
      status: "partial",
      issues: ["coverage_only"],
    });
    delete coverageTime.metadata.value;
    const coverageContent = scheduleMemory({
      slot: "appointment_content",
      source: "u-new",
      status: "partial",
      issues: ["coverage_only"],
    });
    delete coverageContent.metadata.value;

    const wrongCancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode: "vehicle-inspection-appointment",
      relation: "cancelled",
      refs: ["atomic:0", "atomic:1"],
      supersedes: ["prior-time"],
    });
    wrongCancelledTime.metadata.canonicalized_input_candidate_ids = ["atomic:0"];
    const wrongCancelledContent = scheduleMemory({
      slot: "appointment_content",
      value: "在新地点做车辆检查",
      source: "u-new",
      episode: "vehicle-inspection-appointment",
      relation: "cancelled",
      refs: ["atomic:0", "atomic:2", "coverage:content"],
      supersedes: ["prior-content"],
    });
    wrongCancelledContent.metadata.canonicalized_input_candidate_ids = ["atomic:0"];

    const inputs = [
      { id: "atomic:0", memory: composite },
      { id: "atomic:1", memory: updatedTime },
      { id: "atomic:2", memory: updatedContent },
      { id: "coverage:time", memory: coverageTime },
      { id: "coverage:content", memory: coverageContent },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [wrongCancelledTime, wrongCancelledContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((entry) => ({
      slot: entry.metadata.slot,
      value: entry.metadata.value,
      relation: entry.metadata.relation,
      episode: entry.metadata.episode_key,
      supersedes: entry.metadata.supersedes,
    }))).toEqual([
      {
        slot: "appointment_time",
        value: "2026-04-08T15:00:00+08:00",
        relation: "updated",
        episode: "vehicle-inspection-appointment",
        supersedes: ["prior-time"],
      },
      {
        slot: "appointment_content",
        value: "在新地点做车辆检查",
        relation: "updated",
        episode: "vehicle-inspection-appointment",
        supersedes: ["prior-content"],
      },
    ]);
    expect(result.resolvedCandidateIds).toEqual(["atomic:0"]);
    expect(result.memories.every((entry) =>
      !(entry.metadata.input_candidate_ids as string[]).includes("atomic:0")
    )).toBe(true);
    expect(result.repairCounts).toMatchObject({
      repaired_missing_supersedes_from_unique_live_prior: 2,
      restored_recoverable_partial_atomic_obligation: 2,
      resolved_composite_transition_by_specific_atomic_edges: 2,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("restores every concrete partial edge when a pure cancellation is cross-slot canonicalized", () => {
    const episode = "vehicle-inspection-appointment";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-old",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "在5号牛街清真超市内进行车辆检查",
      source: "u-old",
      episode,
    });
    const priorDestination = memory({
      value: "5号牛街清真超市内",
      source: "u-old",
      episode,
    });
    const cancelledContent = scheduleMemory({
      slot: "appointment_content",
      value: "在5号牛街清真超市内进行车辆检查",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const cancelledDestination = memory({
      value: "5号牛街清真超市内",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
    });

    const wrongContent = scheduleMemory({
      slot: "appointment_content",
      value: "在5号牛街清真超市内进行车辆检查",
      source: "u-new",
      episode,
      relation: "cancelled",
      refs: ["atomic:1"],
      supersedes: ["prior-content"],
    });
    wrongContent.metadata.canonicalized_input_candidate_ids = ["atomic:1"];
    const wrongDestination = memory({
      value: "5号牛街清真超市内",
      source: "u-new",
      episode,
      relation: "cancelled",
      refs: ["atomic:0", "atomic:2"],
    });
    wrongDestination.metadata.supersedes = ["prior-destination"];
    wrongDestination.metadata.canonicalized_input_candidate_ids = ["atomic:0"];

    const inputs = [
      { id: "atomic:0", memory: cancelledContent },
      { id: "atomic:1", memory: cancelledTime },
      { id: "atomic:2", memory: cancelledDestination },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
      { record_id: "prior-destination", metadata: priorDestination.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [wrongContent, wrongDestination],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories.map((entry) => ({
      domain: entry.metadata.domain,
      slot: entry.metadata.slot,
      relation: entry.metadata.relation,
      supersedes: entry.metadata.supersedes,
    }))).toEqual([
      {
        domain: "schedule",
        slot: "appointment_content",
        relation: "cancelled",
        supersedes: ["prior-content"],
      },
      {
        domain: "schedule",
        slot: "appointment_time",
        relation: "cancelled",
        supersedes: ["prior-time"],
      },
      {
        domain: "navigation",
        slot: "destination",
        relation: "cancelled",
        supersedes: ["prior-destination"],
      },
    ]);
    expect(result.repairCounts).toMatchObject({
      dropped_unverified_atomic_slot_canonicalization: 2,
      repaired_missing_supersedes_from_unique_live_prior: 3,
      restored_recoverable_partial_atomic_obligation: 3,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("consumes a redundant missing-supersedes status cancellation after an exact multi-slot replacement", () => {
    const episode = "vehicle-inspection-appointment";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内进行车辆检查",
      source: "u-old",
      episode,
    });
    const priorDestination = memory({
      value: "北京市东城区故宫博物院内",
      source: "u-old",
      episode,
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:0", "atomic:3"],
      supersedes: ["prior-time"],
    });
    const updatedContent = scheduleMemory({
      slot: "appointment_content",
      value: "在5号牛街清真超市内进行车辆检查",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:1"],
      supersedes: ["prior-content"],
    });
    const updatedDestination = memory({
      value: "5号牛街清真超市内",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:2"],
    });
    updatedDestination.metadata.supersedes = ["prior-destination"];
    const redundantStatus = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });

    const inputs = [
      { id: "atomic:0", memory: updatedTime },
      { id: "atomic:1", memory: updatedContent },
      { id: "atomic:2", memory: updatedDestination },
      { id: "atomic:3", memory: redundantStatus },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
      { record_id: "prior-destination", metadata: priorDestination.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [updatedTime, updatedContent, updatedDestination],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(3);
    expect(result.memories.every((entry) => entry.metadata.relation === "updated")).toBe(true);
    expect(result.resolvedCandidateIds).toEqual(["atomic:3"]);
    expect(result.memories.every((entry) =>
      !(entry.metadata.input_candidate_ids as string[]).includes("atomic:3")
    )).toBe(true);
    expect(result.repairCounts).toMatchObject({
      resolved_composite_transition_by_specific_atomic_edges: 3,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not consume a composite transition from merely overlapping source evidence", () => {
    const episode = "overlap-is-not-one-event";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode,
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:time"],
      supersedes: ["prior-time"],
    });
    const updatedContent = scheduleMemory({
      slot: "appointment_content",
      value: "在新地点进行车辆检查",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:content"],
      supersedes: ["prior-content"],
    });
    const composite = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    composite.source_message_ids = ["u-new", "u-unrelated"];
    composite.metadata.source_message_ids = ["u-new", "u-unrelated"];
    const inputs = [
      { id: "atomic:time", memory: updatedTime },
      { id: "atomic:content", memory: updatedContent },
      { id: "atomic:composite", memory: composite },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [updatedTime, updatedContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:composite");
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });
    expect(gate.accepted).toBe(false);
    expect(gate.uncoveredCandidateIds).toContain("atomic:composite");
  });

  it("does not let an expanded final source set launder narrower atomic evidence", () => {
    const episode = "atomic-source-laundering";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode,
    });
    const atomicTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:time"],
      supersedes: ["prior-time"],
    });
    const atomicContent = scheduleMemory({
      slot: "appointment_content",
      value: "在新地点进行车辆检查",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:content"],
      supersedes: ["prior-content"],
    });
    const finalTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:time"],
      supersedes: ["prior-time"],
    });
    const finalContent = scheduleMemory({
      slot: "appointment_content",
      value: "在新地点进行车辆检查",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:content"],
      supersedes: ["prior-content"],
    });
    for (const row of [finalTime, finalContent]) {
      row.source_message_ids = ["u-new", "u-unexplained"];
      row.metadata.source_message_ids = ["u-new", "u-unexplained"];
    }
    const composite = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    composite.source_message_ids = ["u-new", "u-unexplained"];
    composite.metadata.source_message_ids = ["u-new", "u-unexplained"];
    const inputs = [
      { id: "atomic:time", memory: atomicTime },
      { id: "atomic:content", memory: atomicContent },
      { id: "atomic:composite", memory: composite },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [finalTime, finalContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:composite");
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });
    expect(gate.accepted).toBe(false);
    expect(gate.uncoveredCandidateIds).toContain("atomic:composite");
  });

  it("does not consume a composite transition across session or event time", () => {
    for (const mismatch of ["source_session_id", "mentioned_at"] as const) {
      const episode = `composite-${mismatch}-mismatch`;
      const priorTime = scheduleMemory({
        slot: "appointment_time",
        value: "2026-04-05T10:00:00+08:00",
        source: "u-old",
        episode,
      });
      const priorContent = scheduleMemory({
        slot: "appointment_content",
        value: "车辆检查",
        source: "u-old",
        episode,
      });
      const updatedTime = scheduleMemory({
        slot: "appointment_time",
        value: "2026-04-08T15:00:00+08:00",
        source: "u-new",
        episode,
        relation: "updated",
        refs: ["atomic:time"],
        supersedes: ["prior-time"],
      });
      const updatedContent = scheduleMemory({
        slot: "appointment_content",
        value: "在新地点进行车辆检查",
        source: "u-new",
        episode,
        relation: "updated",
        refs: ["atomic:content"],
        supersedes: ["prior-content"],
      });
      const composite = scheduleMemory({
        slot: "status",
        value: "cancelled",
        source: "u-new",
        episode,
        relation: "cancelled",
        status: "partial",
        issues: ["missing_supersedes"],
      });
      for (const row of [composite, updatedTime, updatedContent]) {
        row.metadata.source_session_id = "session-a";
        row.metadata.mentioned_at = "2026-04-01T01:00:00.000Z";
      }
      if (mismatch === "source_session_id") {
        updatedTime.metadata.source_session_id = "session-b";
        updatedContent.metadata.source_session_id = "session-b";
      } else {
        updatedTime.metadata.mentioned_at = "2026-04-01T01:01:00.000Z";
        updatedContent.metadata.mentioned_at = "2026-04-01T01:01:00.000Z";
      }
      const inputs = [
        { id: "atomic:time", memory: updatedTime },
        { id: "atomic:content", memory: updatedContent },
        { id: "atomic:composite", memory: composite },
      ];
      const priors = [
        { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
        { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
      ];
      const result = assembleCockpitConstructionReconciliation({
        inputs,
        reconciled: [updatedTime, updatedContent],
        maxMemories: 10,
        priorMemories: priors,
      });

      expect(result.resolvedCandidateIds).not.toContain("atomic:composite");
      expect(gateCockpitConstructionReconciliation({
        inputs,
        reconciled: result.memories,
        maxMemories: 10,
        priorMemories: priors,
        resolvedCandidateIds: result.resolvedCandidateIds,
      }).accepted).toBe(false);
    }
  });

  it("does not consume a missing-supersedes status cancellation from only one updated edge", () => {
    const episode = "vehicle-inspection-appointment";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:0"],
      supersedes: ["prior-time"],
    });
    const unresolvedStatus = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const inputs = [
      { id: "atomic:0", memory: updatedTime },
      { id: "atomic:1", memory: unresolvedStatus },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [updatedTime],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).toEqual([]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not consume a missing-supersedes status cancellation across two episodes", () => {
    const firstPrior = scheduleMemory({
      slot: "appointment_time",
      source: "u-old",
      episode: "appointment-1",
    });
    const secondPrior = scheduleMemory({
      slot: "appointment_content",
      source: "u-old",
      episode: "appointment-2",
    });
    const firstUpdate = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode: "appointment-1",
      relation: "updated",
      refs: ["atomic:0"],
      supersedes: ["prior-1"],
    });
    const secondUpdate = scheduleMemory({
      slot: "appointment_content",
      value: "车辆复检",
      source: "u-new",
      episode: "appointment-2",
      relation: "updated",
      refs: ["atomic:1"],
      supersedes: ["prior-2"],
    });
    const unresolvedStatus = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-new",
      episode: "appointment-1",
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const inputs = [
      { id: "atomic:0", memory: firstUpdate },
      { id: "atomic:1", memory: secondUpdate },
      { id: "atomic:2", memory: unresolvedStatus },
    ];
    const priors = [
      { record_id: "prior-1", metadata: firstPrior.metadata as Record<string, unknown> },
      { record_id: "prior-2", metadata: secondPrior.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [firstUpdate, secondUpdate],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).toEqual([]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not consume a missing-supersedes status cancellation across two people", () => {
    const episode = "shared-appointment-key";
    const firstPrior = scheduleMemory({
      slot: "appointment_time",
      source: "u-old",
      episode,
    });
    firstPrior.metadata.subject = "周宁";
    firstPrior.metadata.state_key = "schedule|周宁|car|driver|appointment_time";
    const secondPrior = scheduleMemory({
      slot: "appointment_content",
      source: "u-old",
      episode,
    });
    secondPrior.metadata.subject = "李然";
    secondPrior.metadata.state_key = "schedule|李然|car|driver|appointment_content";
    const firstUpdate = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:0"],
      supersedes: ["prior-1"],
    });
    firstUpdate.metadata.subject = "周宁";
    firstUpdate.metadata.state_key = "schedule|周宁|car|driver|appointment_time";
    const secondUpdate = scheduleMemory({
      slot: "appointment_content",
      value: "车辆复检",
      source: "u-new",
      episode,
      relation: "updated",
      refs: ["atomic:1"],
      supersedes: ["prior-2"],
    });
    secondUpdate.metadata.subject = "李然";
    secondUpdate.metadata.state_key = "schedule|李然|car|driver|appointment_content";
    const unresolvedStatus = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-new",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const inputs = [
      { id: "atomic:0", memory: firstUpdate },
      { id: "atomic:1", memory: secondUpdate },
      { id: "atomic:2", memory: unresolvedStatus },
    ];
    const priors = [
      { record_id: "prior-1", metadata: firstPrior.metadata as Record<string, unknown> },
      { record_id: "prior-2", metadata: secondPrior.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [firstUpdate, secondUpdate],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).toEqual([]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("collapses complete cancel-plus-assert atomic pairs into live replacement edges", () => {
    const episode = "vehicle-inspection-appointment";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode,
    });
    const priorDestination = memory({
      value: "北京市东城区故宫博物院内",
      source: "u-old",
      episode,
    });

    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-new",
      episode,
      relation: "cancelled",
      refs: ["atomic:cancel-time", "coverage:time"],
      supersedes: ["prior-time"],
    });
    const assertedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode,
      relation: "asserted",
      refs: ["atomic:assert-time"],
    });
    const cancelledContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-new",
      episode,
      relation: "cancelled",
      refs: ["atomic:cancel-content", "coverage:content"],
      supersedes: ["prior-content"],
    });
    const assertedContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-new",
      episode,
      relation: "asserted",
      refs: ["atomic:assert-content"],
    });
    const cancelledDestination = memory({
      value: "北京市东城区故宫博物院内",
      source: "u-new",
      episode,
      relation: "cancelled",
      refs: ["atomic:cancel-destination"],
    });
    cancelledDestination.metadata.supersedes = ["prior-destination"];
    const assertedDestination = memory({
      value: "5号牛街清真超市内",
      source: "u-new",
      episode,
      relation: "asserted",
      refs: ["atomic:assert-destination"],
    });
    const coverageTime = scheduleMemory({
      slot: "appointment_time",
      source: "u-new",
      episode,
      status: "partial",
      issues: ["coverage_only"],
    });
    delete coverageTime.metadata.value;
    const coverageContent = scheduleMemory({
      slot: "appointment_content",
      source: "u-new",
      episode,
      status: "partial",
      issues: ["coverage_only"],
    });
    delete coverageContent.metadata.value;

    const inputs = [
      { id: "atomic:cancel-time", memory: cancelledTime },
      { id: "atomic:assert-time", memory: assertedTime },
      { id: "atomic:cancel-content", memory: cancelledContent },
      { id: "atomic:assert-content", memory: assertedContent },
      { id: "atomic:cancel-destination", memory: cancelledDestination },
      { id: "atomic:assert-destination", memory: assertedDestination },
      { id: "coverage:time", memory: coverageTime },
      { id: "coverage:content", memory: coverageContent },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
      { record_id: "prior-destination", metadata: priorDestination.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancelledTime, cancelledContent, cancelledDestination],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(3);
    expect(result.memories.map((entry) => ({
      slot: entry.metadata.slot,
      value: entry.metadata.value,
      relation: entry.metadata.relation,
      supersedes: entry.metadata.supersedes,
    }))).toEqual([
      {
        slot: "appointment_time",
        value: "2026-04-08T15:00:00+08:00",
        relation: "updated",
        supersedes: ["prior-time"],
      },
      {
        slot: "appointment_content",
        value: "在5号牛街清真超市内进行车辆检查",
        relation: "updated",
        supersedes: ["prior-content"],
      },
      {
        slot: "destination",
        value: "5号牛街清真超市内",
        relation: "updated",
        supersedes: ["prior-destination"],
      },
    ]);
    expect(result.resolvedCandidateIds).toEqual(expect.arrayContaining([
      "atomic:cancel-time",
      "atomic:cancel-content",
      "atomic:cancel-destination",
    ]));
    expect(result.memories.flatMap((entry) =>
      entry.metadata.input_candidate_ids as string[]
    )).toEqual(expect.arrayContaining(["coverage:time", "coverage:content"]));
    expect(result.repairCounts).toMatchObject({
      resolved_atomic_cancel_replace_pair_to_live_update: 3,
      closed_coepisodic_appointment_content_location_from_destination: 1,
      promoted_assertion_over_live_state_to_update: 3,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it.each([
    {
      label: "overlapping but non-identical source sets",
      expectedReplacementRelation: "updated",
      mutate: (cancellation: ExtractedMemory, replacement: ExtractedMemory) => {
        cancellation.source_message_ids = ["u-context", "u-cancel"];
        cancellation.metadata.source_message_ids = ["u-context", "u-cancel"];
        replacement.source_message_ids = ["u-context", "u-create"];
        replacement.metadata.source_message_ids = ["u-context", "u-create"];
      },
    },
    {
      label: "different source sessions",
      expectedReplacementRelation: "updated",
      mutate: (cancellation: ExtractedMemory, replacement: ExtractedMemory) => {
        cancellation.metadata.source_session_id = "session-cancel";
        cancellation.metadata.source_session_ids = ["session-cancel"];
        replacement.metadata.source_session_id = "session-create";
        replacement.metadata.source_session_ids = ["session-create"];
      },
    },
    {
      label: "different event times",
      expectedReplacementRelation: "asserted",
      mutate: (cancellation: ExtractedMemory, replacement: ExtractedMemory) => {
        cancellation.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
        replacement.metadata.mentioned_at = "2026-04-01T10:00:00+08:00";
      },
    },
    {
      label: "different memory type and scene",
      expectedReplacementRelation: "updated",
      mutate: (_cancellation: ExtractedMemory, replacement: ExtractedMemory) => {
        replacement.type = "semantic";
        replacement.scene_name = "schedule";
      },
    },
  ])("does not collapse distinct same-episode events with $label", ({
    mutate,
    expectedReplacementRelation,
  }) => {
    const episode = "same-identity-distinct-events";
    const prior = memory({ source: "u-prior", value: "首都机场", episode });
    const cancellation = memory({
      source: "u-event",
      value: "首都机场",
      episode,
      relation: "cancelled",
      refs: ["atomic:cancel"],
    });
    cancellation.metadata.supersedes = ["prior-destination"];
    const replacement = memory({
      source: "u-event",
      value: "大兴机场",
      episode,
      relation: "asserted",
      refs: ["atomic:replacement"],
    });
    mutate(cancellation, replacement);

    const inputs = [
      { id: "atomic:cancel", memory: cancellation },
      { id: "atomic:replacement", memory: replacement },
    ];
    const priors = [{
      record_id: "prior-destination",
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancellation],
      maxMemories: 10,
      priorMemories: priors,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:cancel");
    expect(result.memories).toHaveLength(2);
    expect(result.memories.map((row) => row.metadata.relation))
      .toEqual(expect.arrayContaining(["cancelled", expectedReplacementRelation]));
    expect(gate.accepted).toBe(false);
    expect(gate.issues).toContain("reconciliation_duplicate_final_identity");
  });

  it("does not invent a same-episode replacement from an unrelated primary value", () => {
    const source = "u-unanchored-replacement";
    const episode = "unanchored-replacement";
    const prior = memory({ source: "u-prior", value: "A", episode });
    const cancellation = memory({
      source,
      value: "A",
      episode,
      relation: "cancelled",
      refs: ["atomic:cancel"],
    });
    cancellation.metadata.supersedes = ["prior-a"];
    const unrelatedPrimary = memory({
      source,
      value: "C",
      episode,
      relation: "asserted",
      refs: ["primary:unrelated"],
    });
    const inventedReplacement = memory({
      source,
      value: "B",
      episode,
      relation: "asserted",
      refs: ["primary:unrelated"],
    });
    const inputs = [
      { id: "atomic:cancel", memory: cancellation },
      { id: "primary:unrelated", memory: unrelatedPrimary },
    ];
    const priors = [{
      record_id: "prior-a",
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancellation, inventedReplacement],
      maxMemories: 10,
      priorMemories: priors,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:cancel");
    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata).toMatchObject({
      value: "A",
      relation: "cancelled",
      input_candidate_ids: ["atomic:cancel"],
    });
    expect(gate.accepted).toBe(true);
  });

  it("does not collapse a replacement across conflicting live predecessors", () => {
    const source = "u-conflicting-live-priors";
    const episode = "conflicting-live-priors";
    const priorA = memory({ source: "u-prior-a", value: "A", episode });
    const priorC = memory({ source: "u-prior-c", value: "C", episode });
    const cancellation = memory({
      source,
      value: "A",
      episode,
      relation: "cancelled",
      refs: ["atomic:cancel"],
    });
    cancellation.metadata.supersedes = ["prior-a"];
    const replacement = memory({
      source,
      value: "B",
      episode,
      relation: "asserted",
      refs: ["atomic:replacement"],
    });
    const inputs = [
      { id: "atomic:cancel", memory: cancellation },
      { id: "atomic:replacement", memory: replacement },
    ];
    const priors = [
      { record_id: "prior-a", metadata: priorA.metadata as Record<string, unknown> },
      { record_id: "prior-c", metadata: priorC.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancellation],
      maxMemories: 10,
      priorMemories: priors,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:cancel");
    expect(result.memories).toHaveLength(2);
    expect(gate.accepted).toBe(false);
  });

  it("collapses a primary-confirmed multi-slot reschedule across model episode aliases", () => {
    const oldEpisode = "vehicle-inspection-2026-04-05";
    const newEpisode = "vehicle-inspection-2026-04-08";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode: oldEpisode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "在北京市东城区故宫博物院内进行车辆检查",
      source: "u-old",
      episode: oldEpisode,
    });
    const priorDestination = memory({
      value: "北京市东城区故宫博物院内",
      source: "u-old",
      episode: oldEpisode,
    });

    const primaryTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-move",
      episode: newEpisode,
      relation: "updated",
      supersedes: ["prior-time"],
    });
    const primaryContent = scheduleMemory({
      slot: "appointment_content",
      value: "在5号牛街清真超市内进行车辆检查",
      source: "u-move",
      episode: newEpisode,
      relation: "updated",
      supersedes: ["prior-content"],
    });
    const primaryDestination = memory({
      value: "5号牛街清真超市内",
      source: "u-move",
      episode: newEpisode,
      relation: "updated",
    });
    primaryDestination.metadata.supersedes = ["prior-destination"];

    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-move",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-time", "primary:time"],
      supersedes: ["prior-time"],
    });
    const assertedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-move",
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-time", "primary:time"],
    });
    const cancelledContent = scheduleMemory({
      slot: "appointment_content",
      value: "在北京市东城区故宫博物院内进行车辆检查",
      source: "u-move",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-content", "coverage:content"],
      supersedes: ["prior-content"],
    });
    const assertedContent = scheduleMemory({
      slot: "appointment_content",
      value: "在5号牛街清真超市内进行车辆检查",
      source: "u-move",
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-content", "primary:content"],
    });
    const cancelledDestination = memory({
      value: "北京市东城区故宫博物院内",
      source: "u-move",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-destination"],
    });
    cancelledDestination.metadata.supersedes = ["prior-destination"];
    const assertedDestination = memory({
      value: "5号牛街清真超市内",
      source: "u-move",
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-destination", "primary:destination"],
    });
    const coverageContent = scheduleMemory({
      slot: "appointment_content",
      source: "u-move",
      episode: oldEpisode,
      status: "partial",
      issues: ["coverage_only"],
    });
    delete coverageContent.metadata.value;

    const inputs = [
      { id: "primary:time", memory: primaryTime },
      { id: "primary:content", memory: primaryContent },
      { id: "primary:destination", memory: primaryDestination },
      { id: "atomic:cancel-time", memory: cancelledTime },
      { id: "atomic:assert-time", memory: assertedTime },
      { id: "atomic:cancel-content", memory: cancelledContent },
      { id: "atomic:assert-content", memory: assertedContent },
      { id: "atomic:cancel-destination", memory: cancelledDestination },
      { id: "atomic:assert-destination", memory: assertedDestination },
      { id: "coverage:content", memory: coverageContent },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
      { record_id: "prior-destination", metadata: priorDestination.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [
        cancelledTime,
        assertedTime,
        cancelledContent,
        assertedContent,
        cancelledDestination,
        assertedDestination,
      ],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(3);
    expect(result.memories.map((entry) => ({
      slot: entry.metadata.slot,
      value: entry.metadata.value,
      episode: entry.metadata.episode_key,
      relation: entry.metadata.relation,
      supersedes: entry.metadata.supersedes,
    }))).toEqual([
      {
        slot: "appointment_time",
        value: "2026-04-08T15:00:00+08:00",
        episode: oldEpisode,
        relation: "updated",
        supersedes: ["prior-time"],
      },
      {
        slot: "appointment_content",
        value: "在5号牛街清真超市内进行车辆检查",
        episode: oldEpisode,
        relation: "updated",
        supersedes: ["prior-content"],
      },
      {
        slot: "destination",
        value: "5号牛街清真超市内",
        episode: oldEpisode,
        relation: "updated",
        supersedes: ["prior-destination"],
      },
    ]);
    expect(result.resolvedCandidateIds).toEqual(expect.arrayContaining([
      "atomic:cancel-time",
      "atomic:cancel-content",
      "atomic:cancel-destination",
    ]));
    expect(result.repairCounts).toMatchObject({
      resolved_cross_episode_cancel_replace_transaction_to_live_update: 3,
    });
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not borrow primary updates from an unrelated new episode", () => {
    const oldEpisode = "appointment-old";
    const replacementEpisode = "appointment-independent";
    const unrelatedPrimaryEpisode = "appointment-other-update";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-prior",
      episode: oldEpisode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-prior",
      episode: oldEpisode,
    });
    const primaryTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode: unrelatedPrimaryEpisode,
      relation: "updated",
      supersedes: ["prior-time"],
    });
    const primaryContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-event",
      episode: unrelatedPrimaryEpisode,
      relation: "updated",
      supersedes: ["prior-content"],
    });
    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-event",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-time"],
      supersedes: ["prior-time"],
    });
    const assertedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode: replacementEpisode,
      relation: "asserted",
      refs: ["atomic:assert-time", "primary:time"],
    });
    const cancelledContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-event",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-content"],
      supersedes: ["prior-content"],
    });
    const assertedContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-event",
      episode: replacementEpisode,
      relation: "asserted",
      refs: ["atomic:assert-content", "primary:content"],
    });

    const inputs = [
      { id: "primary:time", memory: primaryTime },
      { id: "primary:content", memory: primaryContent },
      { id: "atomic:cancel-time", memory: cancelledTime },
      { id: "atomic:assert-time", memory: assertedTime },
      { id: "atomic:cancel-content", memory: cancelledContent },
      { id: "atomic:assert-content", memory: assertedContent },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancelledTime, assertedTime, cancelledContent, assertedContent],
      maxMemories: 10,
      priorMemories: priors,
    });
    const gate = gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    });

    expect(result.resolvedCandidateIds).not.toEqual(expect.arrayContaining([
      "atomic:cancel-time",
      "atomic:cancel-content",
    ]));
    expect(result.memories).toHaveLength(4);
    expect(new Set(result.memories.map((row) => row.metadata.episode_key)))
      .toEqual(new Set([oldEpisode, replacementEpisode]));
    expect(gate.accepted).toBe(true);
  });

  it("keeps cross-episode cancel and create rows separate without primary update evidence", () => {
    const oldEpisode = "appointment-old";
    const newEpisode = "appointment-independent";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode: oldEpisode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode: oldEpisode,
    });
    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-new",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-time"],
      supersedes: ["prior-time"],
    });
    const assertedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-time", "primary:time"],
    });
    const cancelledContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-new",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-content"],
      supersedes: ["prior-content"],
    });
    const assertedContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-new",
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-content", "primary:content"],
    });
    const assertedPrimaryTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode: newEpisode,
      relation: "asserted",
    });
    const assertedPrimaryContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-new",
      episode: newEpisode,
      relation: "asserted",
    });
    const inputs = [
      { id: "primary:time", memory: assertedPrimaryTime },
      { id: "primary:content", memory: assertedPrimaryContent },
      { id: "atomic:cancel-time", memory: cancelledTime },
      { id: "atomic:assert-time", memory: assertedTime },
      { id: "atomic:cancel-content", memory: cancelledContent },
      { id: "atomic:assert-content", memory: assertedContent },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancelledTime, assertedTime, cancelledContent, assertedContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(4);
    expect(result.memories.map((entry) => entry.metadata.relation)).toEqual([
      "cancelled",
      "asserted",
      "cancelled",
      "asserted",
    ]);
    expect(result.repairCounts).not.toHaveProperty(
      "resolved_cross_episode_cancel_replace_transaction_to_live_update",
    );
  });

  it("fails closed when a partial transition has multiple possible live predecessors", () => {
    const partial = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-new",
      episode: "replacement-draft",
      relation: "updated",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const firstPrior = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      episode: "appointment-1",
    });
    const secondPrior = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-06T10:00:00+08:00",
      episode: "appointment-2",
    });
    const inputs = [{ id: "atomic:0", memory: partial }];
    const priors = [
      { record_id: "prior-1", metadata: firstPrior.metadata as Record<string, unknown> },
      { record_id: "prior-2", metadata: secondPrior.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toEqual([]);
    expect(result.resolvedCandidateIds).toEqual([]);
    expect(result.repairCounts).not.toHaveProperty("repaired_missing_supersedes_from_unique_live_prior");
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });
});

describe("post-bind evidence integrity regressions", () => {
  it.each([
    {
      label: "different source session arrays",
      mutate: (destination: ExtractedMemory, appointment: ExtractedMemory) => {
        destination.metadata.source_session_ids = ["session-a"];
        appointment.metadata.source_session_ids = ["session-b"];
      },
    },
    {
      label: "an incoherent appointment scene",
      mutate: (_destination: ExtractedMemory, appointment: ExtractedMemory) => {
        appointment.scene_name = "notification";
      },
    },
    {
      label: "one shared arbitrary scene across controlled domains",
      mutate: (destination: ExtractedMemory, appointment: ExtractedMemory) => {
        destination.scene_name = "notification";
        appointment.scene_name = "notification";
      },
    },
  ])("does not close a destination qualifier across $label", ({ mutate }) => {
    const source = "u-post-bind-destination";
    const episode = "post-bind-destination";
    const mentionedAt = "2026-04-01T09:00:00+08:00";
    const destination = memory({
      source,
      value: "故宫博物院",
      episode,
      refs: ["atomic:destination"],
      mentionedAt,
    });
    destination.content = "目的地为故宫博物院";
    destination.metadata.target = "故宫博物院";
    destination.metadata.source_session_id = "session-shared";
    destination.metadata.source_session_ids = ["session-shared"];
    const appointment = scheduleMemory({
      slot: "appointment_content",
      value: "在故宫博物院内进行车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    appointment.metadata.target = appointment.metadata.value;
    appointment.metadata.mentioned_at = mentionedAt;
    appointment.metadata.source_session_id = "session-shared";
    appointment.metadata.source_session_ids = ["session-shared"];
    mutate(destination, appointment);
    const inputs = [
      { id: "atomic:destination", memory: destination },
      { id: "atomic:content", memory: appointment },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [destination, appointment],
      maxMemories: 10,
    });

    expect(result.memories.find((row) => row.metadata.slot === "destination")
      ?.metadata.value).toBe("故宫博物院");
    expect(result.repairCounts.closed_coepisodic_destination_qualifier_from_appointment_content)
      .toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it.each([
    {
      label: "different source session arrays",
      mutate: (appointment: ExtractedMemory, destination: ExtractedMemory) => {
        appointment.metadata.source_session_ids = ["session-a"];
        destination.metadata.source_session_ids = ["session-b"];
      },
    },
    {
      label: "an incoherent destination type and scene",
      mutate: (_appointment: ExtractedMemory, destination: ExtractedMemory) => {
        destination.type = "semantic";
        destination.scene_name = "notification";
      },
    },
    {
      label: "conflicting temporal facts",
      mutate: (appointment: ExtractedMemory, destination: ExtractedMemory) => {
        appointment.metadata.timezone = "Asia/Shanghai";
        destination.metadata.timezone = "UTC";
      },
    },
    {
      label: "one shared arbitrary scene across controlled domains",
      mutate: (appointment: ExtractedMemory, destination: ExtractedMemory) => {
        appointment.scene_name = "notification";
        destination.scene_name = "notification";
      },
    },
  ])("does not enrich appointment content across $label", ({ mutate }) => {
    const source = "u-post-bind-appointment";
    const episode = "post-bind-appointment";
    const mentionedAt = "2026-04-01T09:00:00+08:00";
    const appointment = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source,
      episode,
      refs: ["atomic:content"],
    });
    appointment.content = "预约内容为车辆检查";
    appointment.metadata.target = "车辆检查";
    appointment.metadata.mentioned_at = mentionedAt;
    appointment.metadata.source_session_id = "session-shared";
    appointment.metadata.source_session_ids = ["session-shared"];
    const destination = memory({
      source,
      value: "故宫博物院内",
      episode,
      refs: ["atomic:destination"],
      mentionedAt,
    });
    destination.metadata.target = destination.metadata.value;
    destination.metadata.source_session_id = "session-shared";
    destination.metadata.source_session_ids = ["session-shared"];
    mutate(appointment, destination);
    const inputs = [
      { id: "atomic:content", memory: appointment },
      { id: "atomic:destination", memory: destination },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [appointment, destination],
      maxMemories: 10,
    });

    expect(result.memories.find((row) => row.metadata.slot === "appointment_content")
      ?.metadata.value).toBe("车辆检查");
    expect(result.repairCounts.closed_coepisodic_appointment_content_location_from_destination)
      .toBeUndefined();
  });

  it("does not cross-proposal alias different source session arrays", () => {
    const source = "u-post-bind-episode-alias";
    const point = "2026-04-05T10:00:00+08:00";
    const primary = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "primary-episode",
      refs: ["primary:time"],
    });
    const atomic = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "atomic-episode",
      refs: ["atomic:time"],
    });
    for (const row of [primary, atomic]) {
      row.metadata.valid_from = point;
      row.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
      row.metadata.source_session_id = "session-shared";
    }
    primary.metadata.source_session_ids = ["session-primary"];
    atomic.metadata.source_session_ids = ["session-atomic"];
    const inputs = [
      { id: "primary:time", memory: primary },
      { id: "atomic:time", memory: atomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, atomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_cross_proposal_episode_alias).toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not resolve a composite transition across an incoherent scene", () => {
    const episode = "post-bind-composite-scene";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode,
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode,
      relation: "updated",
      refs: ["atomic:time"],
      supersedes: ["prior-time"],
    });
    const updatedContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-event",
      episode,
      relation: "updated",
      refs: ["atomic:content"],
      supersedes: ["prior-content"],
    });
    const composite = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-event",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["ambiguous_transition_state"],
      supersedes: [episode],
    });
    composite.scene_name = "notification";
    const inputs = [
      { id: "atomic:time", memory: updatedTime },
      { id: "atomic:content", memory: updatedContent },
      { id: "atomic:composite", memory: composite },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [updatedTime, updatedContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:composite");
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not resolve a composite edge carrying an unrelated live predecessor", () => {
    const episode = "post-bind-composite-lineage";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const unrelatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-06T10:00:00+08:00",
      source: "u-other",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode,
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode,
      relation: "updated",
      refs: ["atomic:time"],
      supersedes: ["prior-time", "unrelated-time"],
    });
    const partialTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode,
      relation: "updated",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const updatedContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-event",
      episode,
      relation: "updated",
      refs: ["atomic:content"],
      supersedes: ["prior-content"],
    });
    const composite = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-event",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["ambiguous_transition_state"],
      supersedes: ["prior-time", "prior-content"],
    });
    const inputs = [
      { id: "atomic:time", memory: partialTime },
      { id: "atomic:content", memory: updatedContent },
      { id: "atomic:composite", memory: composite },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "unrelated-time", metadata: unrelatedTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [updatedTime, updatedContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:composite");
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not collapse a cross-episode transaction with incoherent slot scenes", () => {
    const oldEpisode = "post-bind-cross-old";
    const newEpisode = "post-bind-cross-new";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode: oldEpisode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode: oldEpisode,
    });
    const primaryTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode: newEpisode,
      relation: "updated",
      supersedes: ["prior-time"],
    });
    const primaryContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-event",
      episode: newEpisode,
      relation: "updated",
      supersedes: ["prior-content"],
    });
    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-event",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-time"],
      supersedes: ["prior-time"],
    });
    const replacementTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode: newEpisode,
      refs: ["atomic:replace-time", "primary:time"],
    });
    const cancelledContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-event",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-content"],
      supersedes: ["prior-content"],
    });
    cancelledContent.scene_name = "navigation";
    const replacementContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-event",
      episode: newEpisode,
      refs: ["atomic:replace-content", "primary:content"],
    });
    const inputs = [
      { id: "primary:time", memory: primaryTime },
      { id: "primary:content", memory: primaryContent },
      { id: "atomic:cancel-time", memory: cancelledTime },
      { id: "atomic:replace-time", memory: replacementTime },
      { id: "atomic:cancel-content", memory: cancelledContent },
      { id: "atomic:replace-content", memory: replacementContent },
    ];
    const priors = [
      { record_id: "prior-time", metadata: priorTime.metadata as Record<string, unknown> },
      { record_id: "prior-content", metadata: priorContent.metadata as Record<string, unknown> },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancelledTime, replacementTime, cancelledContent, replacementContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(4);
    expect(result.resolvedCandidateIds).not.toEqual(expect.arrayContaining([
      "atomic:cancel-time",
      "atomic:cancel-content",
    ]));
    expect(result.repairCounts.resolved_cross_episode_cancel_replace_transaction_to_live_update)
      .toBeUndefined();
  });

  it.each([
    {
      label: "memory type",
      mutate: (row: ExtractedMemory) => { row.type = "semantic"; },
    },
    {
      label: "controlled class",
      mutate: (row: ExtractedMemory) => {
        row.scene_name = "schedule";
        row.metadata.domain = "schedule";
        row.metadata.slot = "appointment_content";
        row.metadata.state_key = "schedule|user|car|driver|appointment_content";
      },
    },
  ])("does not promote an assertion over a live prior with a different $label", ({ mutate }) => {
    const prior = memory({ value: "A", episode: "typed-live-prior" });
    const changed = memory({ value: "B", episode: "typed-live-prior", refs: ["atomic:changed"] });
    mutate(changed);
    const inputs = [{ id: "atomic:changed", memory: changed }];
    const priors = [{
      record_id: "prior-a",
      type: prior.type,
      scene_name: prior.scene_name,
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [changed],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories[0].metadata.relation).toBe("asserted");
    expect(result.memories[0].metadata.supersedes).toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not promote an earlier assertion over a future persisted prior", () => {
    const prior = memory({
      value: "A",
      episode: "future-live-prior",
      mentionedAt: "2026-04-01T10:00:00+08:00",
    });
    const earlier = memory({
      value: "B",
      episode: "future-live-prior",
      refs: ["atomic:earlier"],
      mentionedAt: "2026-04-01T09:00:00+08:00",
    });
    const inputs = [{ id: "atomic:earlier", memory: earlier }];
    const priors = [{
      record_id: "future-prior",
      type: prior.type,
      scene_name: prior.scene_name,
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [earlier],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.relation).toBe("asserted");
    expect(result.memories[0].metadata.supersedes).toBeUndefined();
  });

  it("strips an unsupported reference before a proven cross-proposal episode alias", () => {
    const source = "u-post-bind-reference-origin";
    const point = "2026-04-05T10:00:00+08:00";
    const primary = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "primary-reference-origin",
      refs: ["primary:time"],
    });
    const atomic = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "atomic-reference-origin",
      refs: ["atomic:time", "primary:unrelated"],
    });
    const unrelated = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-06T11:00:00+08:00",
      source,
      episode: "unrelated-reference-origin",
      refs: ["primary:unrelated"],
    });
    for (const row of [primary, atomic]) {
      row.metadata.valid_from = point;
      row.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
      row.metadata.source_session_id = "session-reference-origin";
      row.metadata.source_session_ids = ["session-reference-origin"];
    }
    const inputs = [
      { id: "primary:time", memory: primary },
      { id: "atomic:time", memory: atomic },
      { id: "primary:unrelated", memory: unrelated },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, atomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.input_candidate_ids).toEqual(
      expect.arrayContaining(["primary:time", "atomic:time"]),
    );
    expect(result.memories[0].metadata.input_candidate_ids).not.toContain("primary:unrelated");
    expect(result.repairCounts.coalesced_cross_proposal_episode_alias).toBe(1);
  });

  it("coalesces natural and ISO values only when independent proposals prove one event instant", () => {
    const source = "u-temporal-representation";
    const point = "2026-04-05T10:00:00+08:00";
    const primary = scheduleMemory({
      slot: "appointment_time",
      value: "4月5日上午10点",
      source,
      episode: "primary-natural-time",
      refs: ["primary:time"],
    });
    const atomic = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "atomic-iso-time",
      refs: ["atomic:time"],
    });
    for (const row of [primary, atomic]) {
      row.metadata.target = row.metadata.value;
      row.metadata.activity_start_time = point;
      row.metadata.valid_from = point;
      row.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
      row.metadata.source_session_id = "session-temporal-representation";
      row.metadata.source_session_ids = ["session-temporal-representation"];
    }
    const inputs = [
      { id: "primary:time", memory: structuredClone(primary) },
      { id: "atomic:time", memory: structuredClone(atomic) },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, atomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata.value).toBe(point);
    expect(result.memories[0].metadata.episode_key).toBe("primary-natural-time");
    expect(result.memories[0].metadata.input_candidate_ids).toEqual(
      expect.arrayContaining(["primary:time", "atomic:time"]),
    );
    expect(result.repairCounts.coalesced_cross_proposal_temporal_representation_alias).toBe(1);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it.each([
    { label: "different structured instants", primaryStart: "2026-04-05T11:00:00+08:00", classPrefix: "primary", isoValue: "2026-04-05T10:00:00+08:00" },
    { label: "missing structured proof", primaryStart: undefined, classPrefix: "primary", isoValue: "2026-04-05T10:00:00+08:00" },
    { label: "the same proposal class", primaryStart: "2026-04-05T10:00:00+08:00", classPrefix: "atomic", isoValue: "2026-04-05T10:00:00+08:00" },
    { label: "an invalid calendar ISO value", primaryStart: "2026-04-05T10:00:00+08:00", classPrefix: "primary", isoValue: "2026-02-30T10:00:00+08:00" },
    { label: "a timezone-free ISO value", primaryStart: "2026-04-05T10:00:00+08:00", classPrefix: "primary", isoValue: "2026-04-05T10:00:00" },
    { label: "an ISO value that disagrees with structured time", primaryStart: "2026-04-05T10:00:00+08:00", classPrefix: "primary", isoValue: "2026-04-05T11:00:00+08:00" },
    { label: "two non-ISO raw values", primaryStart: "2026-04-05T10:00:00+08:00", classPrefix: "primary", isoValue: "四月五日上午十点" },
  ])("does not alias natural and ISO values with $label", ({ primaryStart, classPrefix, isoValue }) => {
    const source = "u-temporal-representation-negative";
    const point = "2026-04-05T10:00:00+08:00";
    const primary = scheduleMemory({
      slot: "appointment_time",
      value: "4月5日上午10点",
      source,
      episode: "natural-time-negative",
      refs: [`${classPrefix}:natural`],
    });
    const atomic = scheduleMemory({
      slot: "appointment_time",
      value: isoValue,
      source,
      episode: "iso-time-negative",
      refs: ["atomic:iso"],
    });
    for (const row of [primary, atomic]) {
      row.metadata.target = row.metadata.value;
      row.metadata.valid_from = point;
      row.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
      row.metadata.source_session_id = "session-temporal-negative";
      row.metadata.source_session_ids = ["session-temporal-negative"];
    }
    if (primaryStart !== undefined) primary.metadata.activity_start_time = primaryStart;
    atomic.metadata.activity_start_time = point;
    const inputs = [
      { id: `${classPrefix}:natural`, memory: structuredClone(primary) },
      { id: "atomic:iso", memory: structuredClone(atomic) },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, atomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_cross_proposal_temporal_representation_alias)
      .toBeUndefined();
  });

  it("does not restore an ordinary partial cancellation with the wrong live value", () => {
    const prior = memory({ value: "A", episode: "wrong-cancel-value" });
    const cancelled = memory({
      value: "B",
      episode: "wrong-cancel-value",
      relation: "cancelled",
      status: "partial",
    });
    const inputs = [{ id: "atomic:cancel", memory: cancelled }];
    const priors = [{
      record_id: "prior-a",
      type: prior.type,
      scene_name: prior.scene_name,
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toEqual([]);
    expect(result.resolvedCandidateIds).toEqual([]);
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it.each([
    { key: "timezone", value: "Asia/Shanghai" },
    { key: "condition", value: "battery_level < 20%" },
  ])("does not cross-proposal alias when only one proposal carries $key", ({ key, value }) => {
    const source = "u-cross-proposal-obligation";
    const point = "2026-04-05T10:00:00+08:00";
    const primary = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "primary-obligation-episode",
      refs: ["primary:time"],
    });
    const atomic = scheduleMemory({
      slot: "appointment_time",
      value: point,
      source,
      episode: "atomic-obligation-episode",
      refs: ["atomic:time"],
    });
    for (const row of [primary, atomic]) {
      row.metadata.valid_from = point;
      row.metadata.mentioned_at = "2026-04-01T09:00:00+08:00";
      row.metadata.source_session_id = "session-shared";
      row.metadata.source_session_ids = ["session-shared"];
    }
    atomic.metadata[key] = value;
    const inputs = [
      { id: "primary:time", memory: primary },
      { id: "atomic:time", memory: atomic },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [primary, atomic],
      maxMemories: 10,
    });

    expect(result.memories).toHaveLength(2);
    expect(result.repairCounts.coalesced_cross_proposal_episode_alias).toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(true);
  });

  it("does not resolve a schedule composite through an unrelated climate state", () => {
    const episode = "appointment-with-unrelated-climate";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode,
      relation: "updated",
      refs: ["atomic:time"],
      supersedes: ["prior-time"],
    });
    const priorTemperature = memory({
      source: "u-old",
      value: "20",
      episode,
    });
    priorTemperature.scene_name = "climate";
    priorTemperature.metadata.domain = "climate";
    priorTemperature.metadata.slot = "temperature";
    priorTemperature.metadata.state_key = "climate|user|car|driver|temperature";
    const updatedTemperature = memory({
      source: "u-event",
      value: "22",
      episode,
      relation: "updated",
      refs: ["atomic:temperature"],
    });
    updatedTemperature.scene_name = "climate";
    updatedTemperature.metadata.domain = "climate";
    updatedTemperature.metadata.slot = "temperature";
    updatedTemperature.metadata.state_key = "climate|user|car|driver|temperature";
    updatedTemperature.metadata.supersedes = ["prior-temperature"];
    const composite = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-event",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["missing_supersedes"],
    });
    const inputs = [
      { id: "atomic:time", memory: updatedTime },
      { id: "atomic:temperature", memory: updatedTemperature },
      { id: "atomic:composite", memory: composite },
    ];
    const priors = [
      {
        record_id: "prior-time",
        type: priorTime.type,
        scene_name: priorTime.scene_name,
        metadata: priorTime.metadata as Record<string, unknown>,
      },
      {
        record_id: "prior-temperature",
        type: priorTemperature.type,
        scene_name: priorTemperature.scene_name,
        metadata: priorTemperature.metadata as Record<string, unknown>,
      },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [updatedTime, updatedTemperature],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:composite");
    expect(result.repairCounts.resolved_composite_transition_by_specific_atomic_edges)
      .toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not resolve an arbitrary partial state as an appointment composite", () => {
    const episode = "appointment-with-media-composite";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode,
    });
    const priorContent = scheduleMemory({
      slot: "appointment_content",
      value: "车辆检查",
      source: "u-old",
      episode,
    });
    const updatedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source: "u-event",
      episode,
      relation: "updated",
      refs: ["atomic:time"],
      supersedes: ["prior-time"],
    });
    const updatedContent = scheduleMemory({
      slot: "appointment_content",
      value: "轮胎检查",
      source: "u-event",
      episode,
      relation: "updated",
      refs: ["atomic:content"],
      supersedes: ["prior-content"],
    });
    const arbitraryComposite = scheduleMemory({
      slot: "status",
      value: "cancelled",
      source: "u-event",
      episode,
      relation: "cancelled",
      status: "partial",
      issues: ["ambiguous_transition_state"],
      supersedes: ["prior-time", "prior-content"],
    });
    arbitraryComposite.scene_name = "media";
    arbitraryComposite.metadata.domain = "media";
    arbitraryComposite.metadata.slot = "playback_status";
    arbitraryComposite.metadata.state_key = "media|user|car|driver|playback_status";
    const inputs = [
      { id: "atomic:time", memory: updatedTime },
      { id: "atomic:content", memory: updatedContent },
      { id: "atomic:arbitrary-composite", memory: arbitraryComposite },
    ];
    const priors = [
      {
        record_id: "prior-time",
        type: priorTime.type,
        scene_name: priorTime.scene_name,
        metadata: priorTime.metadata as Record<string, unknown>,
      },
      {
        record_id: "prior-content",
        type: priorContent.type,
        scene_name: priorContent.scene_name,
        metadata: priorContent.metadata as Record<string, unknown>,
      },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [updatedTime, updatedContent],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toContain("atomic:arbitrary-composite");
    expect(result.repairCounts.resolved_composite_transition_by_specific_atomic_edges)
      .toBeUndefined();
    expect(gateCockpitConstructionReconciliation({
      inputs,
      reconciled: result.memories,
      maxMemories: 10,
      priorMemories: priors,
      resolvedCandidateIds: result.resolvedCandidateIds,
    }).accepted).toBe(false);
  });

  it("does not fold a non-appointment navigation slot into a cross-episode reschedule", () => {
    const oldEpisode = "appointment-old-with-route-policy";
    const newEpisode = "appointment-new-with-route-policy";
    const source = "u-move";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode: oldEpisode,
    });
    const primaryTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source,
      episode: newEpisode,
      relation: "updated",
      supersedes: ["prior-time"],
    });
    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source,
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-time"],
      supersedes: ["prior-time"],
    });
    const assertedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source,
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-time", "primary:time"],
    });

    const priorPolicy = memory({
      source: "u-old",
      value: "avoid_tolls",
      episode: oldEpisode,
    });
    const primaryPolicy = memory({
      source,
      value: "avoid_highways",
      episode: newEpisode,
      relation: "updated",
    });
    primaryPolicy.metadata.supersedes = ["prior-policy"];
    const cancelledPolicy = memory({
      source,
      value: "avoid_tolls",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-policy"],
    });
    cancelledPolicy.metadata.supersedes = ["prior-policy"];
    const assertedPolicy = memory({
      source,
      value: "avoid_highways",
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-policy", "primary:policy"],
    });
    for (const row of [priorPolicy, primaryPolicy, cancelledPolicy, assertedPolicy]) {
      row.metadata.slot = "avoid_destination";
      row.metadata.state_key = "navigation|user|car|driver|avoid_destination";
    }
    const inputs = [
      { id: "primary:time", memory: primaryTime },
      { id: "primary:policy", memory: primaryPolicy },
      { id: "atomic:cancel-time", memory: cancelledTime },
      { id: "atomic:assert-time", memory: assertedTime },
      { id: "atomic:cancel-policy", memory: cancelledPolicy },
      { id: "atomic:assert-policy", memory: assertedPolicy },
    ];
    const priors = [
      {
        record_id: "prior-time",
        type: priorTime.type,
        scene_name: priorTime.scene_name,
        metadata: priorTime.metadata as Record<string, unknown>,
      },
      {
        record_id: "prior-policy",
        type: priorPolicy.type,
        scene_name: priorPolicy.scene_name,
        metadata: priorPolicy.metadata as Record<string, unknown>,
      },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [cancelledTime, assertedTime, cancelledPolicy, assertedPolicy],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toEqual(expect.arrayContaining([
      "atomic:cancel-time",
      "atomic:cancel-policy",
    ]));
    expect(result.repairCounts.resolved_cross_episode_cancel_replace_transaction_to_live_update)
      .toBeUndefined();
  });

  it("does not fold a controlled appointment migration through one shared arbitrary scene", () => {
    const oldEpisode = "appointment-old-shared-arbitrary-scene";
    const newEpisode = "appointment-new-shared-arbitrary-scene";
    const source = "u-move-shared-arbitrary-scene";
    const priorTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source: "u-old",
      episode: oldEpisode,
    });
    const primaryTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source,
      episode: newEpisode,
      relation: "updated",
      supersedes: ["prior-time"],
    });
    const cancelledTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-05T10:00:00+08:00",
      source,
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-time"],
      supersedes: ["prior-time"],
    });
    const assertedTime = scheduleMemory({
      slot: "appointment_time",
      value: "2026-04-08T15:00:00+08:00",
      source,
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-time", "primary:time"],
    });

    const priorDestination = memory({
      source: "u-old",
      value: "旧检测站",
      episode: oldEpisode,
    });
    const primaryDestination = memory({
      source,
      value: "新检测站",
      episode: newEpisode,
      relation: "updated",
    });
    primaryDestination.metadata.supersedes = ["prior-destination"];
    const cancelledDestination = memory({
      source,
      value: "旧检测站",
      episode: oldEpisode,
      relation: "cancelled",
      refs: ["atomic:cancel-destination"],
    });
    cancelledDestination.metadata.supersedes = ["prior-destination"];
    const assertedDestination = memory({
      source,
      value: "新检测站",
      episode: newEpisode,
      relation: "asserted",
      refs: ["atomic:assert-destination", "primary:destination"],
    });
    for (const row of [
      priorTime,
      primaryTime,
      cancelledTime,
      assertedTime,
      priorDestination,
      primaryDestination,
      cancelledDestination,
      assertedDestination,
    ]) {
      row.scene_name = "notification";
    }
    const inputs = [
      { id: "primary:time", memory: primaryTime },
      { id: "primary:destination", memory: primaryDestination },
      { id: "atomic:cancel-time", memory: cancelledTime },
      { id: "atomic:assert-time", memory: assertedTime },
      { id: "atomic:cancel-destination", memory: cancelledDestination },
      { id: "atomic:assert-destination", memory: assertedDestination },
    ];
    const priors = [
      {
        record_id: "prior-time",
        type: priorTime.type,
        scene_name: priorTime.scene_name,
        metadata: priorTime.metadata as Record<string, unknown>,
      },
      {
        record_id: "prior-destination",
        type: priorDestination.type,
        scene_name: priorDestination.scene_name,
        metadata: priorDestination.metadata as Record<string, unknown>,
      },
    ];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [
        cancelledTime,
        assertedTime,
        cancelledDestination,
        assertedDestination,
      ],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.resolvedCandidateIds).not.toEqual(expect.arrayContaining([
      "atomic:cancel-time",
      "atomic:cancel-destination",
    ]));
    expect(result.repairCounts.resolved_cross_episode_cancel_replace_transaction_to_live_update)
      .toBeUndefined();
  });

  it("still restores an ordinary partial cancellation with the exact live value", () => {
    const prior = memory({ value: "A", episode: "exact-cancel-value" });
    const cancelled = memory({
      value: "A",
      episode: "exact-cancel-value",
      relation: "cancelled",
      status: "partial",
    });
    const inputs = [{ id: "atomic:cancel", memory: cancelled }];
    const priors = [{
      record_id: "prior-a",
      type: prior.type,
      scene_name: prior.scene_name,
      metadata: prior.metadata as Record<string, unknown>,
    }];
    const result = assembleCockpitConstructionReconciliation({
      inputs,
      reconciled: [],
      maxMemories: 10,
      priorMemories: priors,
    });

    expect(result.memories).toHaveLength(1);
    expect(result.memories[0].metadata).toMatchObject({
      value: "A",
      relation: "cancelled",
      supersedes: ["prior-a"],
    });
  });
});

describe("controlled cockpit ontology normalization", () => {
  it("canonicalizes a governed slot to its unique schema owner", () => {
    const sourceMessages: ConversationMessage[] = [{
      id: "u1",
      role: "user",
      content: "避开高速",
      timestamp: 1,
    }];
    const proposed = memory();
    proposed.metadata.domain = "notification";
    proposed.metadata.slot = "route-constraint";
    proposed.metadata.value = "avoid_highway";
    const normalized = normalizeCockpitExtractedMemory({
      memory: proposed,
      sourceMessages,
      sessionId: "session-1",
      constructionModel: "deepseek-v4-flash",
    });

    expect(normalized.metadata.domain).toBe("navigation");
    expect(normalized.metadata.slot).toBe("route_constraint");
    expect((normalized.metadata.construction_quality as Record<string, unknown>).status).toBe("complete");
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .toEqual(expect.arrayContaining(["canonicalized_slot_token", "canonicalized_controlled_slot_domain"]));
  });
});
