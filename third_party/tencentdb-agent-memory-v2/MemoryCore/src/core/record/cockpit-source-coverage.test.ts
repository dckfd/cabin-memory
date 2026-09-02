import { describe, expect, it } from "vitest";

import {
  detectCockpitSourceCoverageObligations,
  sourceCoverageObligationToMemory,
} from "./cockpit-source-coverage.js";

function user(content: string, id = "u1") {
  return { id, role: "user" as const, content, timestamp: 1 };
}

describe("cockpit deterministic source coverage", () => {
  it("requires every independently queryable POI criterion without extracting its value", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "【冯遥】请在酒店周边推荐评分4.5分以上、门票20元以下、游玩时间2小时-3小时的景点。",
    )]);

    expect(obligations.map((entry) => [entry.domain, entry.slot, entry.constraintTarget]))
      .toEqual([
        ["selection", "rating_constraint", undefined],
        ["selection", "price_constraint", "ticket"],
        ["selection", "duration_constraint", undefined],
        ["selection", "category_constraint", undefined],
        ["selection", "location_constraint", undefined],
      ]);
    expect(obligations.every((entry) => !("value" in entry) && !("operator" in entry))).toBe(true);
  });

  it("distinguishes price targets and media release periods", () => {
    const obligations = detectCockpitSourceCoverageObligations([
      user("帮我找个人均消费50-100元、评分4分以上的餐馆。", "u1"),
      user("我想找个2000年代的印度电影看看。", "u2"),
      user("酒店每晚预算500元以下。", "u3"),
    ]);

    expect(obligations).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceMessageId: "u1", slot: "price_constraint", constraintTarget: "per_capita" }),
      expect.objectContaining({ sourceMessageId: "u1", slot: "rating_constraint" }),
      expect.objectContaining({ sourceMessageId: "u2", slot: "category_constraint" }),
      expect.objectContaining({ sourceMessageId: "u2", slot: "release_period_constraint" }),
      expect.objectContaining({ sourceMessageId: "u3", slot: "price_constraint", constraintTarget: "room" }),
    ]));
  });

  it("covers appointment content/time and ranking policies", () => {
    const obligations = detectCockpitSourceCoverageObligations([
      user("先约4月4日上午10点去服务中心做车辆检查。", "u1"),
      user("选补能点按休息设施、评分、距离排序。", "u2"),
    ]);

    expect(obligations).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceMessageId: "u1", domain: "schedule", slot: "appointment_content" }),
      expect.objectContaining({ sourceMessageId: "u1", domain: "schedule", slot: "appointment_time" }),
      expect.objectContaining({ sourceMessageId: "u1", domain: "navigation", slot: "destination" }),
      expect.objectContaining({ sourceMessageId: "u2", domain: "selection", slot: "ranking_policy" }),
    ]));
    expect(obligations).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceMessageId: "u2", slot: "rating_constraint" }),
    ]));

    const cancellation = detectCockpitSourceCoverageObligations([
      user("4月7日下午去服务中心的检查最终取消。", "u3"),
    ]);
    expect(cancellation).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceMessageId: "u3", slot: "appointment_content" }),
      expect.objectContaining({ sourceMessageId: "u3", slot: "appointment_time" }),
      expect.objectContaining({ sourceMessageId: "u3", domain: "navigation", slot: "destination" }),
    ]));

    const replacement = detectCockpitSourceCoverageObligations([
      user("取消原安排，改约4月8日下午3点去5号牛街清真超市内。", "u4"),
    ]);
    expect(replacement).toEqual(expect.arrayContaining([
      expect.objectContaining({ sourceMessageId: "u4", domain: "navigation", slot: "destination" }),
    ]));
  });

  it("does not mistake a destination-less appointment action for a place", () => {
    const obligations = detectCockpitSourceCoverageObligations([
      user("约在下午3点去做车辆检查。"),
    ]);

    expect(obligations).toEqual(expect.arrayContaining([
      expect.objectContaining({ domain: "schedule", slot: "appointment_content" }),
      expect.objectContaining({ domain: "schedule", slot: "appointment_time" }),
    ]));
    expect(obligations).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ domain: "navigation", slot: "destination" }),
    ]));
  });

  it("requires a distinct source-named identity for every member of a destination state map", () => {
    const source = "【程野】路线口令：早餐地点是松林文化馆，返程地点是云岭客运站，会客地点是滨河创意园。";
    const [destination] = detectCockpitSourceCoverageObligations([user(source)])
      .filter((entry) => entry.reason === "explicit_named_destination_state");

    expect(destination).toMatchObject({
      domain: "navigation",
      slot: "destination",
      requiredFactCount: 3,
      requiredStateQualifierCount: 3,
      requiresStateQualifier: true,
      requiresDistinctEvidenceBindings: true,
      requiresSetAudit: true,
    });
    expect(destination.evidenceGroups).toHaveLength(3);
    expect(destination.evidenceGroups.map((group) =>
      source.normalize("NFKC").slice(group.start, group.end)
    )).toEqual([
      "早餐地点是",
      "返程地点是",
      "会客地点是",
    ]);
    expect(destination.evidenceGroups.map((group) => group.stateQualifier)).toEqual([
      "早餐地点",
      "返程地点",
      "会客地点",
    ]);
    expect(JSON.stringify(destination)).not.toContain("松林文化馆");
    expect(JSON.stringify(destination)).not.toContain("云岭客运站");
    expect(JSON.stringify(destination)).not.toContain("滨河创意园");

    const scaffold = sourceCoverageObligationToMemory(destination);
    expect(scaffold.metadata).toMatchObject({
      coverage_requires_state_qualifier: true,
      coverage_required_state_qualifier_count: 3,
      coverage_required_state_qualifiers: ["早餐地点", "返程地点", "会客地点"],
    });
  });

  it("keeps named-map cardinality through corrections and rejects generic or queried destinations", () => {
    const [updated] = detectCockpitSourceCoverageObligations([user(
      "更新路线口令：早餐地点改成北岸书店，返程地点仍是云岭客运站，会客地点改为南湖剧场。",
    )]).filter((entry) => entry.reason === "explicit_named_destination_state");
    expect(updated).toMatchObject({
      requiredFactCount: 3,
      requiredStateQualifierCount: 3,
      requiresSetAudit: true,
    });

    const generic = detectCockpitSourceCoverageObligations([
      user("这个目的地是北岸书店。"),
      user("把刚才的目的地改成南湖剧场。", "u2"),
    ]);
    const queried = detectCockpitSourceCoverageObligations([user("请问早餐地点是什么？")]);
    expect(generic).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: "explicit_named_destination_state" }),
    ]));
    expect(queried).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: "explicit_named_destination_state" }),
    ]));
  });

  it("activates one named member only from an exact live prior and ignores ordinary location properties", () => {
    const noPrior = detectCockpitSourceCoverageObligations([
      user("早餐地点改成北岸书店。"),
    ]);
    const exactPrior = detectCockpitSourceCoverageObligations([
      user("早餐地点改成北岸书店。"),
    ], [{
      record_id: "prior-breakfast",
      metadata: {
        domain: "navigation",
        slot: "destination",
        state_qualifier: "早餐地点",
        subject: "user",
        relation: "asserted",
      },
    }]);
    const ordinaryProperties = detectCockpitSourceCoverageObligations([
      user("会议地点是北楼，工作地点是南楼。"),
      user("默认目的地设为机场。", "u2"),
    ]);

    expect(noPrior).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: "explicit_named_destination_state" }),
    ]));
    expect(exactPrior).toEqual(expect.arrayContaining([
      expect.objectContaining({
        reason: "explicit_named_destination_state",
        requiredStateQualifierCount: 1,
        evidenceGroups: [expect.objectContaining({ stateQualifier: "早餐地点" })],
      }),
    ]));
    expect(ordinaryProperties).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ reason: "explicit_named_destination_state" }),
    ]));
  });

  it("keeps named-map context within one command segment and scopes prior activation", () => {
    const segmented = detectCockpitSourceCoverageObligations([user(
      "【乙】车机口令：午饭地点是东楼，返程地点是西门。会议地点是北楼。",
    )]).filter((entry) => entry.reason === "explicit_named_destination_state");
    const leakedContext = detectCockpitSourceCoverageObligations([user(
      "【乙】请保存为路线口令。会议地点是北楼，工作地点是南楼。",
    )]).filter((entry) => entry.reason === "explicit_named_destination_state");
    const mixedSpeakers = detectCockpitSourceCoverageObligations([user(
      "【甲】路线口令：早餐地点是东楼，【乙】返程地点是西门。",
    )]).filter((entry) => entry.reason === "explicit_named_destination_state");
    const priorForOtherPerson = [{
      record_id: "prior-a",
      metadata: {
        domain: "navigation",
        slot: "destination",
        state_qualifier: "早餐地点",
        subject: "甲",
        relation: "asserted",
      },
    }];
    const crossPerson = detectCockpitSourceCoverageObligations([
      user("【乙】早餐地点改成北岸书店。"),
    ], priorForOtherPerson).filter((entry) => entry.reason === "explicit_named_destination_state");
    const missingPriorSubject = detectCockpitSourceCoverageObligations([
      user("早餐地点改成北岸书店。"),
    ], [{
      record_id: "prior-without-subject",
      metadata: {
        domain: "navigation",
        slot: "destination",
        state_qualifier: "早餐地点",
        relation: "asserted",
      },
    }]).filter((entry) => entry.reason === "explicit_named_destination_state");
    const samePerson = detectCockpitSourceCoverageObligations([
      user("【甲】早餐地点改成北岸书店。"),
    ], priorForOtherPerson).filter((entry) => entry.reason === "explicit_named_destination_state");
    const ambiguousVehicle = detectCockpitSourceCoverageObligations([
      user("【甲】早餐地点改成北岸书店。"),
    ], [
      { ...priorForOtherPerson[0], record_id: "prior-car-a", metadata: {
        ...priorForOtherPerson[0].metadata,
        vehicle_scope: "车辆A",
      } },
      { ...priorForOtherPerson[0], record_id: "prior-car-b", metadata: {
        ...priorForOtherPerson[0].metadata,
        vehicle_scope: "车辆B",
      } },
    ]).filter((entry) => entry.reason === "explicit_named_destination_state");

    expect(segmented).toHaveLength(1);
    expect(segmented[0].evidenceGroups.map((group) => group.stateQualifier))
      .toEqual(["午饭地点", "返程地点"]);
    expect(leakedContext).toEqual([]);
    expect(mixedSpeakers).toEqual([]);
    expect(crossPerson).toEqual([]);
    expect(missingPriorSubject).toEqual([]);
    expect(samePerson).toEqual([
      expect.objectContaining({
        requiredStateQualifierCount: 1,
        evidenceGroups: [expect.objectContaining({ stateQualifier: "早餐地点" })],
      }),
    ]);
    expect(ambiguousVehicle).toEqual([]);
  });

  it("does not turn informational questions, reminder offsets or assistant text into numeric obligations", () => {
    const obligations = detectCockpitSourceCoverageObligations([
      user("这家店评分是多少？"),
      user("七点出发，提前10分钟提醒。", "u2"),
      { id: "a1", role: "assistant" as const, content: "门票20元，评分4.8分。", timestamp: 2 },
    ]);

    expect(obligations).toEqual([]);
  });

  it("builds a non-persistable structural scaffold only", () => {
    const [obligation] = detectCockpitSourceCoverageObligations([
      user("推荐一个门票免费的景点。"),
    ]).filter((entry) => entry.slot === "price_constraint");
    const scaffold = sourceCoverageObligationToMemory(obligation);

    expect(scaffold).toMatchObject({
      priority: 0,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "price_constraint",
        constraint_target: "ticket",
        construction_quality: {
          status: "partial",
          issues: ["source_coverage_obligation"],
        },
      },
    });
    expect(scaffold.metadata.value).toBeUndefined();
  });

  it("counts independent same-slot facts inside one source event", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。",
    )]);
    const rating = obligations.find((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({
      sourceMessageId: "u1",
      domain: "selection",
      slot: "rating_constraint",
      requiredFactCount: 2,
    });
  });

  it("counts both rating and selected-category facts in person-scoped recommendation clauses", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "给冯遥找评分4.5分以上的景点，给林静找评分4.0分以上的景点。",
    )]);
    const rating = obligations.find((entry) => entry.slot === "rating_constraint");
    const category = obligations.find((entry) => entry.slot === "category_constraint");

    expect(rating).toMatchObject({ requiredFactCount: 2, requiresSetAudit: true });
    expect(rating?.evidenceGroups).toHaveLength(2);
    expect(category).toMatchObject({ requiredFactCount: 2, requiresSetAudit: true });
    expect(category?.evidenceGroups).toHaveLength(2);
  });

  it("counts conditional same-slot branches without extracting their values", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "默认要求景点评分4.5分以上，带孩子时要求景点评分4.0分以上。",
    )]);
    const rating = obligations.find((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({ requiredFactCount: 2 });
    expect(JSON.stringify(rating)).not.toContain("4.5");
    expect(JSON.stringify(rating)).not.toContain("4.0");
  });

  it("raises the lower bound for a shared predicate with coordinated people", () => {
    const [rating] = detectCockpitSourceCoverageObligations([user(
      "冯遥和林静都要求景点评分4.5分以上。",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({
      requiredFactCount: 2,
      requiresSetAudit: true,
    });
    expect(rating.evidenceGroups).toHaveLength(1);
  });

  it("scopes coordinated subjects to the clause that owns each obligation", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "冯遥和林静都要找景点，我自己的门票预算100元以内。",
    )]);
    const category = obligations.find((entry) => entry.slot === "category_constraint");
    const ticket = obligations.find((entry) =>
      entry.slot === "price_constraint" && entry.constraintTarget === "ticket"
    );

    expect(category).toMatchObject({ requiredFactCount: 2, requiredSubjectCount: 2 });
    expect(ticket).toMatchObject({ requiredFactCount: 1 });
    expect(ticket?.requiredSubjectCount).toBeUndefined();
  });

  it("does not mistake coordinated criteria for coordinated people", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "景点评分4.5分以上和门票100元以下都要满足。",
    )]);
    const rating = obligations.find((entry) => entry.slot === "rating_constraint");
    const ticket = obligations.find((entry) =>
      entry.slot === "price_constraint" && entry.constraintTarget === "ticket"
    );

    expect(rating).toMatchObject({ requiredFactCount: 1 });
    expect(ticket).toMatchObject({ requiredFactCount: 1 });
    expect(rating?.requiredSubjectCount).toBeUndefined();
    expect(ticket?.requiredSubjectCount).toBeUndefined();
  });

  it("uses the exact high-confidence participant cardinality for a shared predicate", () => {
    const [rating] = detectCockpitSourceCoverageObligations([user(
      "冯遥、林静和王强都要求景点评分4.5分以上。",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({
      requiredFactCount: 3,
      requiredSubjectCount: 3,
      requiresSetAudit: true,
    });
    expect(rating.evidenceGroups).toHaveLength(1);
  });

  it("keeps conditional ranking clauses as two independently evidenced policies", () => {
    const [ranking] = detectCockpitSourceCoverageObligations([user(
      "默认按距离排序，电量低时按充电速度优先。",
    )]).filter((entry) => entry.slot === "ranking_policy");

    expect(ranking).toMatchObject({ requiredFactCount: 2, requiresSetAudit: true });
    expect(ranking.evidenceGroups).toHaveLength(2);
  });

  it("does not bind a price value across competing constraint targets", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "门票没要求但人均预算100元以下。",
    )]);

    expect(obligations).toEqual(expect.arrayContaining([
      expect.objectContaining({ slot: "price_constraint", constraintTarget: "per_capita" }),
    ]));
    expect(obligations).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ slot: "price_constraint", constraintTarget: "ticket" }),
    ]));
  });

  it("does not create hard memory obligations from informational questions", () => {
    const obligations = detectCockpitSourceCoverageObligations([
      user("这家店评分4.5分以上吗？", "rating-question"),
      user("这里门票100元以下吗？", "ticket-question"),
      user("现在是按距离排序吗？", "ranking-question"),
      user("年检怎么办理？", "appointment-question"),
    ]);

    expect(obligations).toEqual([]);
  });

  it("lets explicit information questions override polite request words", () => {
    const obligations = detectCockpitSourceCoverageObligations([
      user("请问这家店评分4.5分以上吗？", "polite-rating-question"),
      user("帮我看看这里门票100元以下吗？", "polite-ticket-question"),
    ]);

    expect(obligations).toEqual([]);
  });

  it("does not create category, location or release-period obligations from selection questions", () => {
    const obligations = detectCockpitSourceCoverageObligations([
      user("能推荐景点吗？", "category-question"),
      user("请问附近有没有景点可推荐？", "location-question"),
      user("想找近10年的电影吗？", "release-question"),
    ]);

    expect(obligations).toEqual([]);
  });

  it("treats a punctuation-only multi-criterion confirmation as informational", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "景点评分4.5分以上，门票100元以内？",
      "bare-confirmation-question",
    )]);

    expect(obligations).toEqual([]);
  });

  it("keeps a direct state-change command with a trailing politeness question", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "请把景点评分设置为4.5分以上，门票限制100元以内，可以吗？",
      "direct-command-with-politeness",
    )]);

    expect(obligations).toEqual(expect.arrayContaining([
      expect.objectContaining({ slot: "rating_constraint" }),
      expect.objectContaining({ slot: "price_constraint", constraintTarget: "ticket" }),
    ]));
  });

  it("keeps a polite direct selection request with a trailing confirmation particle", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "帮我找评分4.5分以上的景点好吗？",
      "polite-selection-command",
    )]);

    expect(obligations).toEqual(expect.arrayContaining([
      expect.objectContaining({ slot: "rating_constraint" }),
      expect.objectContaining({ slot: "category_constraint" }),
    ]));
  });

  it("keeps an asserted selection clause without importing a neighboring question", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "请帮我找餐馆，附近有什么景点？",
      "mixed-selection-speech-acts",
    )]);

    expect(obligations.map((entry) => entry.slot)).toEqual(["category_constraint"]);
  });

  it("separates enumerated appointment events but keeps a date interval together", () => {
    const enumerated = detectCockpitSourceCoverageObligations([user(
      "4月1日10点检查、4月2日11点保养。",
      "enumerated",
    )]);
    const conjoined = detectCockpitSourceCoverageObligations([user(
      "4月1日10点检查以及4月2日11点保养。",
      "conjoined",
    )]);
    const interval = detectCockpitSourceCoverageObligations([user(
      "安排4月1日至4月2日做车辆检查。",
      "interval",
    )]);

    for (const obligations of [enumerated, conjoined]) {
      expect(obligations.find((entry) => entry.slot === "appointment_content"))
        .toMatchObject({
          requiredFactCount: 2,
          requiresSetAudit: true,
          requiresDistinctEvidenceBindings: true,
        });
      const time = obligations.find((entry) => entry.slot === "appointment_time");
      expect(time).toMatchObject({
        requiredFactCount: 2,
        requiresSetAudit: true,
        requiresDistinctEvidenceBindings: true,
      });
      expect(time?.evidenceGroups).toHaveLength(2);
    }
    const intervalTime = interval.find((entry) => entry.slot === "appointment_time");
    expect(intervalTime).toMatchObject({ requiredFactCount: 1 });
    expect(intervalTime?.evidenceGroups).toHaveLength(1);
  });

  it("splits high-confidence elliptical same-slot alternatives into evidence groups", () => {
    const ticket = detectCockpitSourceCoverageObligations([user(
      "景点门票成人100元、儿童免费。",
      "ticket",
    )]).find((entry) => entry.slot === "price_constraint" && entry.constraintTarget === "ticket");
    const conditionalRating = detectCockpitSourceCoverageObligations([user(
      "带孩子时评分4.0分以上，否则4.5分以上。",
      "conditional-rating",
    )]).find((entry) => entry.slot === "rating_constraint");
    const personRating = detectCockpitSourceCoverageObligations([user(
      "冯遥评分4.5分以上、林静4.0分以上。",
      "person-rating",
    )]).find((entry) => entry.slot === "rating_constraint");

    for (const obligation of [ticket, conditionalRating, personRating]) {
      expect(obligation).toMatchObject({
        requiredFactCount: 2,
        requiresSetAudit: true,
        requiresDistinctEvidenceBindings: true,
      });
      expect(obligation?.evidenceGroups).toHaveLength(2);
    }
  });

  it("recognizes comma-separated elliptical rating facts", () => {
    const [rating] = detectCockpitSourceCoverageObligations([user(
      "冯遥评分4.5分以上，林静4.0分以上。",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({
      requiredFactCount: 2,
      requiresSetAudit: true,
      requiresDistinctEvidenceBindings: true,
    });
    expect(rating.evidenceGroups).toHaveLength(2);
  });

  it("keeps a separate generic price obligation beside a specific target", () => {
    const prices = detectCockpitSourceCoverageObligations([user(
      "门票100元以内，停车预算50元以内。",
    )]).filter((entry) => entry.slot === "price_constraint");

    expect(prices).toEqual(expect.arrayContaining([
      expect.objectContaining({ constraintTarget: "ticket" }),
      expect.objectContaining({ constraintTarget: "generic" }),
    ]));
    expect(prices).toHaveLength(2);
  });

  it("raises and types a shared two-date binding axis", () => {
    const [rating] = detectCockpitSourceCoverageObligations([user(
      "4月1日和4月2日都要求景点评分4.5分以上。",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({
      requiredFactCount: 2,
      requiredTemporalCount: 2,
      requiresSetAudit: true,
      requiresDistinctEvidenceBindings: true,
    });
    expect(rating.evidenceGroups).toHaveLength(2);
  });

  it("types condition and seat axes instead of calling them people", () => {
    const [weather] = detectCockpitSourceCoverageObligations([user(
      "晴天和雨天都要求景点评分4.5分以上。",
      "weather",
    )]).filter((entry) => entry.slot === "rating_constraint");
    const [seat] = detectCockpitSourceCoverageObligations([user(
      "副驾和后排都要求景点评分4.5分以上。",
      "seat",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(weather).toMatchObject({
      requiredFactCount: 2,
      requiredConditionCount: 2,
    });
    expect(weather.requiredSubjectCount).toBeUndefined();
    expect(seat).toMatchObject({
      requiredFactCount: 2,
      requiredSeatZoneCount: 2,
    });
    expect(seat.requiredSubjectCount).toBeUndefined();
  });

  it("requires both people when separate clauses enumerate person-scoped facts", () => {
    const [rating] = detectCockpitSourceCoverageObligations([user(
      "给冯遥找评分4.5分以上的景点，给林静找评分4.5分以上的景点。",
      "separate-person-clauses",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({
      requiredFactCount: 2,
      requiredSubjectCount: 2,
      requiresSetAudit: true,
      requiresDistinctEvidenceBindings: true,
    });
  });

  it("limits appointment time evidence to its owning event", () => {
    const source = "明天10点安排检查，后天8点提醒带伞。";
    const appointmentTime = detectCockpitSourceCoverageObligations([user(source)])
      .find((entry) => entry.slot === "appointment_time");

    expect(appointmentTime).toMatchObject({ requiredFactCount: 1 });
    expect(appointmentTime?.evidenceGroups).toHaveLength(1);
    const [evidence] = appointmentTime?.evidenceGroups ?? [];
    expect(source.slice(evidence.start, evidence.end)).toContain("明天10点");
    expect(source.slice(evidence.start, evidence.end)).not.toContain("后天8点");
  });

  it("assigns source-level event anchors consistently across appointment slots", () => {
    const obligations = detectCockpitSourceCoverageObligations([user(
      "4月1日10点做检查、4月2日11点去维修厂做保养。",
    )]);
    const anchors = (slot: string) => obligations
      .find((entry) => entry.slot === slot)
      ?.evidenceGroups.map((group) => group.eventAnchor);

    expect(anchors("appointment_content")).toHaveLength(2);
    expect(anchors("appointment_time")).toEqual(anchors("appointment_content"));
    expect(anchors("destination")).toEqual([anchors("appointment_content")?.[1]]);
    expect(anchors("appointment_content")?.every(Boolean)).toBe(true);
  });

  it("treats repetition and correction occurrence counts as conservative lower bounds", () => {
    const [repeated] = detectCockpitSourceCoverageObligations([user(
      "景点评分4.5分以上，记住，评分4.5分以上。",
      "repeat",
    )]).filter((entry) => entry.slot === "rating_constraint");
    const [corrected] = detectCockpitSourceCoverageObligations([user(
      "先按评分4.5分以上，不对，改成评分4.0分以上。",
      "correct",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(repeated).toMatchObject({
      requiredFactCount: 1,
      requiresSetAudit: true,
      requiresDistinctEvidenceBindings: false,
    });
    expect(repeated.evidenceGroups).toHaveLength(2);
    expect(corrected).toMatchObject({
      requiredFactCount: 1,
      requiresSetAudit: true,
      requiresDistinctEvidenceBindings: false,
    });
    expect(corrected.evidenceGroups).toHaveLength(2);
  });

  it("keeps same-value facts on two dates as a multi-event audit", () => {
    const [rating] = detectCockpitSourceCoverageObligations([user(
      "4月1日景点评分4.5分以上，4月2日景点评分4.5分以上。",
    )]).filter((entry) => entry.slot === "rating_constraint");

    expect(rating).toMatchObject({ requiredFactCount: 2, requiresSetAudit: true });
    expect(rating.evidenceGroups).toHaveLength(2);
  });
});
