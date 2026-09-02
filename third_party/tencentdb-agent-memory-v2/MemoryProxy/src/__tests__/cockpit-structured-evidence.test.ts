import { describe, expect, it } from "vitest";

import {
  assessCockpitEvidence,
  getCockpitStructuredValue,
  mergeCockpitEvidence,
  resolveCockpitDatePoints,
  resolveCockpitFinalState,
  summarizeCockpitEventFrequency,
  type CockpitEvidence,
  type CockpitRetrievalPlan,
} from "../injection/cockpit-retrieval-plan.js";

const completeQuality = { status: "complete", score: 100, issues: [], source_count: 1, user_source_count: 1 };

function plan(overrides: Partial<CockpitRetrievalPlan> = {}): CockpitRetrievalPlan {
  return {
    originalQuery: "座舱状态是什么？",
    highRisk: false,
    risks: [],
    queries: [{ text: "座舱状态", purpose: "original" }],
    searchL0: false,
    perQueryLimit: 8,
    maxEvidence: 8,
    minEvidence: 1,
    requiredDates: [],
    ...overrides,
  };
}

function structured(
  id: string,
  episodeKey: string,
  stateKey: string,
  metadata: Record<string, unknown> = {},
): CockpitEvidence {
  return {
    id,
    source: "l1",
    content: "驾驶员完成了相同的一次座舱操作。",
    timestamp: "2026-08-29T02:00:00.000Z",
    sourceMessageIds: [`source-${id}`],
    metadata: {
      schema_version: "cockpit-state-v1",
      record_kind: "event",
      domain: "climate",
      slot: "temperature",
      value: 22,
      subject: "driver",
      action_status: "completed",
      relation: "asserted",
      episode_key: episodeKey,
      state_key: stateKey,
      mentioned_at: "2026-08-29T10:00:00+08:00",
      source_session_ids: [episodeKey.split("|")[0]],
      construction_quality: completeQuality,
      ...metadata,
    },
  };
}

describe("cockpit typed evidence contract", () => {
  it("does not text-deduplicate repeated events or atomic slots from one event", () => {
    const repeated = mergeCockpitEvidence([
      structured("a", "s1|u1", "climate|driver|temperature"),
      structured("b", "s2|u2", "climate|driver|temperature"),
    ], plan());
    expect(repeated).toHaveLength(2);

    const atomicSlots = mergeCockpitEvidence([
      structured("c", "s3|u3", "route|driver|route_choice", { slot: "route_choice", value: "最快" }),
      structured("d", "s3|u3", "route|driver|avoid_tolls", { slot: "avoid_tolls", value: true }),
    ], plan());
    expect(atomicSlots).toHaveLength(2);
  });

  it("suppresses a raw duplicate only when complete structured lineage covers it", () => {
    const complete = structured("state", "s1|u1", "climate|driver|temperature");
    complete.sourceMessageIds = ["u1"];
    const raw: CockpitEvidence = { id: "u1", source: "l0", content: complete.content };
    expect(mergeCockpitEvidence([complete, raw], plan())).toHaveLength(1);

    const partial = structured("partial", "s1|u1", "climate|driver|temperature", {
      construction_quality: { ...completeQuality, status: "partial", issues: ["missing_slot"] },
    });
    partial.sourceMessageIds = ["u1"];
    expect(mergeCockpitEvidence([partial, raw], plan())).toHaveLength(2);
  });

  it("resolves two dates from typed validity intervals and returns typed values", () => {
    const september10 = structured("old", "s1|u1", "navigation|driver|alias", {
      domain: "navigation",
      slot: "会合点",
      target: "虹桥站P6",
      value: "虹桥站P6",
      valid_from: "2026-09-01",
      valid_to: "2026-09-14",
    });
    const september16 = structured("new", "s2|u2", "navigation|driver|alias", {
      domain: "navigation",
      slot: "会合点",
      target: "浦东机场P2",
      value: "浦东机场P2",
      relation: "updated",
      supersedes: ["old"],
      valid_from: "2026-09-15",
    });
    const comparisonPlan = plan({
      originalQuery: "‘会合点’在9月10日和9月16日分别指哪里？",
      highRisk: true,
      risks: ["multi-time-comparison"],
      requiredDates: ["9月10日", "9月16日"],
    });
    const resolved = resolveCockpitDatePoints(comparisonPlan, [september10, september16]);
    expect(resolved.map((item) => item.basis)).toEqual(["validity-interval", "validity-interval"]);
    expect(resolved.map((item) => getCockpitStructuredValue(item.evidence))).toEqual(["虹桥站P6", "浦东机场P2"]);
  });

  it("assembles a multi-slot final state and rejects incomplete construction", () => {
    const route = structured("route-v2", "s2|u2", "commute|driver|route_choice", {
      domain: "commute",
      slot: "路线选择",
      value: "最快路线",
      relation: "updated",
      supersedes: ["route-v1"],
      mentioned_at: "2026-08-29T11:00:00+08:00",
    });
    const toll = structured("toll-v2", "s2|u2", "commute|driver|avoid_tolls", {
      domain: "commute",
      slot: "收费道路",
      value: "避开",
      relation: "updated",
      supersedes: ["toll-v1"],
      mentioned_at: "2026-08-29T11:00:00+08:00",
    });
    const finalPlan = plan({
      originalQuery: "通勤定稿后，路线选择和收费道路现在分别是什么？",
      highRisk: true,
      risks: ["latest-final-update"],
    });
    const finalState = resolveCockpitFinalState(finalPlan, [route, toll]);
    expect(finalState?.facts).toEqual([
      { label: "路线选择", value: "最快路线" },
      { label: "收费道路", value: "避开" },
    ]);
    expect(finalState?.evidenceChain).toHaveLength(2);

    const incomplete = structured("broken", "s3|u3", "commute|driver|route_choice", {
      relation: "updated",
      supersedes: ["route-v2"],
      construction_quality: { ...completeQuality, status: "partial", issues: ["missing_value"] },
    });
    const assessment = assessCockpitEvidence(finalPlan, [incomplete]);
    expect(assessment.sufficient).toBe(false);
    expect(assessment.reasons).toContain("structured_construction_contract_incomplete");
  });

  it("assembles an authoritative cross-session current snapshot without changing Core dedup scope", () => {
    const route = structured("route-current", "session-a|u1", "commute|driver|route_choice", {
      domain: "commute",
      slot: "路线选择",
      value: "最少拥堵",
      action_status: "confirmed",
      relation: "asserted",
      mentioned_at: "2026-08-28T09:00:00+08:00",
      source_session_ids: ["session-a"],
    });
    const toll = structured("toll-current", "session-b|u2", "commute|driver|avoid_tolls", {
      domain: "commute",
      slot: "收费道路",
      value: "允许",
      action_status: "confirmed",
      relation: "asserted",
      mentioned_at: "2026-08-29T09:00:00+08:00",
      source_session_ids: ["session-b"],
    });
    const currentPlan = plan({
      originalQuery: "现在路线选择和收费道路分别是什么？",
      highRisk: true,
      risks: ["latest-final-update", "cross-session-synthesis"],
    });
    const assessment = assessCockpitEvidence(currentPlan, [route, toll]);
    expect(assessment.sufficient).toBe(true);
    expect(assessment.provenanceGroups).toBe(2);
    expect(resolveCockpitFinalState(currentPlan, [route, toll])).toMatchObject({
      relation: "asserted",
      facts: [
        { label: "路线选择", value: "最少拥堵" },
        { label: "收费道路", value: "允许" },
      ],
    });

    // The compact prose intentionally shares no route/toll wording with the
    // query. High-risk recall must use the validated typed fields rather than
    // discard these records at the content-only topic gate.
    const merged = mergeCockpitEvidence([route, toll], currentPlan);
    expect(merged.map((item) => item.id)).toEqual(["route-current", "toll-current"]);
    expect(assessCockpitEvidence(currentPlan, merged).sufficient).toBe(true);
  });

  it("requires a real transition edge when the question explicitly asks about changes", () => {
    const route = structured("route-current", "session-a|u1", "commute|driver|route_choice", {
      domain: "commute",
      slot: "路线选择",
      value: "最少拥堵",
      action_status: "confirmed",
      relation: "asserted",
    });
    const toll = structured("toll-current", "session-b|u2", "commute|driver|avoid_tolls", {
      domain: "commute",
      slot: "收费道路",
      value: "允许",
      action_status: "confirmed",
      relation: "asserted",
    });
    const changedPlan = plan({
      originalQuery: "通勤多次变更后，路线选择和收费道路最终分别是什么？",
      highRisk: true,
      risks: ["latest-final-update"],
    });

    const assessment = assessCockpitEvidence(changedPlan, [route, toll]);
    expect(assessment.sufficient).toBe(false);
    expect(assessment.reasons).toContain("final_update_relation_missing");
  });

  it("never projects partial structured rows into date or frequency answers", () => {
    const valid = structured("valid", "session-a|u1", "charging|driver|station", {
      domain: "charging-event",
      slot: "station",
      value: "青岚补能中心",
      target: "青岚补能中心",
      valid_from: "2026-09-01",
      valid_to: "2026-09-30",
    });
    const partial = structured("partial", "session-b|u2", "charging|driver|station", {
      domain: "charging-event",
      slot: "station",
      value: "错误候选站",
      target: "错误候选站",
      valid_from: "2026-09-01",
      valid_to: "2026-09-30",
      construction_quality: { ...completeQuality, status: "partial", issues: ["missing_scope"] },
    });
    const datePlan = plan({
      originalQuery: "补能站在9月10日和9月16日分别是哪一个？",
      highRisk: true,
      risks: ["multi-time-comparison"],
      requiredDates: ["9月10日", "9月16日"],
    });

    expect(resolveCockpitDatePoints(datePlan, [partial, valid]).map((item) => item.evidence.id))
      .toEqual(["valid", "valid"]);
    expect(summarizeCockpitEventFrequency([partial, valid])).toEqual([
      { label: "青岚补能中心", count: 1 },
    ]);
  });
});
