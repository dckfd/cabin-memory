import { describe, expect, it, vi } from "vitest";

import { formatBatchConflictPrompt } from "../prompts/l1-dedup.js";
import { searchResultToMemoryRecord } from "./l1-dedup.js";
import type { StorageAdapter } from "../storage/adapter.js";
import type { IMemoryStore } from "../store/types.js";
import {
  mergeMemoryMetadata,
  parseMemoryMetadata,
  writeMemory,
  type ExtractedMemory,
  type MemoryRecord,
} from "./l1-writer.js";
import {
  finalizeCockpitMetadataAfterDedup,
  hasCanonicalCockpitQualifiedStateEpisode,
  normalizeCockpitExtractedMemory,
} from "./cockpit-memory-contract.js";

describe("cockpit L1 metadata lifecycle", () => {
  it("retains metadata and event times from dense candidate recall", () => {
    const candidate = searchResultToMemoryRecord({
      record_id: "route-v1",
      content: "The driver requested an airport route.",
      type: "episodic",
      priority: 70,
      scene_name: "navigation",
      score: 0.92,
      timestamp_str: "2026-08-25T09:05:00+08:00",
      timestamp_start: "2026-08-25T09:00:00+08:00",
      timestamp_end: "2026-08-25T09:05:00+08:00",
      version: 2,
      session_key: "session-key",
      session_id: "session-1",
      team_id: "team-a",
      task_id: "task-a",
      user_id: "driver-a",
      agent_id: "vehicle-agent",
      metadata_json: JSON.stringify({
        occupant_scope: "driver",
        action_status: "requested",
      }),
    });

    expect(candidate.metadata).toEqual({
      occupant_scope: "driver",
      action_status: "requested",
    });
    expect(candidate.timestamps).toEqual([
      "2026-08-25T09:00:00+08:00",
      "2026-08-25T09:05:00+08:00",
    ]);
  });

  it("includes old and new scope/time metadata in the dedup prompt", () => {
    const existing: MemoryRecord = {
      id: "old-route",
      content: "The driver requested navigation to the airport.",
      type: "episodic",
      priority: 70,
      scene_name: "navigation",
      source_message_ids: ["u1"],
      metadata: {
        occupant_scope: "driver",
        vehicle_scope: "vehicle-a",
        action_status: "requested",
        mentioned_at: "2026-08-25T09:00:00+08:00",
      },
      timestamps: ["2026-08-25T01:00:00.000Z"],
      createdAt: "2026-08-25T01:00:00.000Z",
      updatedAt: "2026-08-25T01:00:00.000Z",
      version: 1,
      sessionKey: "session-key",
      sessionId: "session-1",
    };
    const next: ExtractedMemory & { record_id: string } = {
      record_id: "new-route",
      content: "The driver changed the destination to Hongqiao Airport.",
      type: "episodic",
      priority: 78,
      scene_name: "navigation",
      source_message_ids: ["u2"],
      metadata: {
        action_status: "selected",
        target: "Hongqiao Airport",
        episode_key: "navigation:session-1",
      },
    };

    const prompt = formatBatchConflictPrompt([{
      newMemory: next,
      candidates: [existing],
    }]);

    expect(prompt).toContain('"occupant_scope": "driver"');
    expect(prompt).toContain('"vehicle_scope": "vehicle-a"');
    expect(prompt).toContain('"action_status": "selected"');
    expect(prompt).toContain('"episode_key": "navigation:session-1"');
  });

  it("merges metadata safely with newer and explicit evidence winning", () => {
    const polluted = JSON.parse(
      '{"vehicle_scope":"vehicle-a","__proto__":{"polluted":true}}',
    );
    expect(parseMemoryMetadata("not-json")).toEqual({});
    expect(mergeMemoryMetadata(
      polluted,
      { action_status: "requested", seat_zone: "driver" },
      { action_status: "verified", target: "22°C" },
    )).toEqual({
      vehicle_scope: "vehicle-a",
      seat_zone: "driver",
      action_status: "verified",
      target: "22°C",
    });
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it("normalizes typed state against actual user evidence without inventing an update", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "驾驶员把温度设为22度并已完成。",
        type: "episodic",
        priority: 80,
        scene_name: "cabin-comfort",
        source_message_ids: ["u1"],
        metadata: {
          domain: "climate",
          slot: "temperature",
          value: 22,
          unit: "℃",
          subject: "driver",
          occupant_scope: "driver",
          seat_zone: "driver",
          action_status: "completed",
        },
      },
      sourceMessages: [
        { id: "u1", role: "user", content: "温度调到22度", timestamp: Date.parse("2026-08-29T10:00:00+08:00") },
        { id: "a1", role: "assistant", content: "已完成", timestamp: Date.parse("2026-08-29T10:00:01+08:00") },
      ],
      sessionId: "session-1",
      constructionModel: "deepseek-v4-flash",
    });

    expect(normalized.scene_name).toBe("climate");
    expect(normalized.source_message_ids).toEqual(["u1"]);
    expect(normalized.metadata).toMatchObject({
      schema_version: "cockpit-state-v1",
      state_key: "climate|driver|unspecified-vehicle|driver|temperature",
      episode_key: "session-1|u1",
      relation: "asserted",
      source_message_ids: ["u1"],
      source_session_ids: ["session-1"],
      construction_model: "deepseek-v4-flash",
      construction_stage: "l1",
      construction_quality: { status: "complete", issues: [] },
    });
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .toContain("canonicalized_scene_name_to_controlled_domain");
  });

  it("marks a Flash-invented preference partial when its only source is a pure information query", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户要求景点评分4.5分以上。",
        type: "instruction",
        priority: 70,
        scene_name: "selection",
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "user",
          relation: "asserted",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "请问这家店评分4.5分以上吗？",
        timestamp: 1,
      }],
      sessionId: "session-1",
      constructionModel: "deepseek-v4-flash",
    });

    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["informational_query_source"]),
    });
  });

  it("does not confuse a direct state-change command with a trailing politeness question", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户要求景点评分4.5分以上。",
        type: "instruction",
        priority: 70,
        scene_name: "selection",
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "user",
          relation: "asserted",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "请把景点评分设置为4.5分以上，可以吗？",
        timestamp: 1,
      }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "complete",
      issues: [],
    });
  });

  it("does not mark a polite direct selection command as an informational query", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户要求景点评分4.5分以上。",
        type: "instruction",
        priority: 70,
        scene_name: "selection",
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "user",
          relation: "asserted",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "帮我找评分4.5分以上的景点好吗？",
        timestamp: 1,
      }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "complete",
      issues: [],
    });
  });

  it("keeps an asserted clause when another clause in the same turn is informational", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户要求景点评分4.5分以上。",
        type: "instruction",
        priority: 70,
        scene_name: "selection",
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "user",
          relation: "asserted",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "景点评分要求4.5分以上，门票多少钱？",
        timestamp: 1,
      }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "complete",
      issues: [],
    });
  });

  it("removes an appointment end invented from a point time", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "周宁预约4月5日上午10点做车辆检查。",
        type: "episodic",
        priority: 70,
        scene_name: "schedule",
        source_message_ids: ["u1"],
        metadata: {
          domain: "schedule",
          slot: "appointment_time",
          value: "2026-04-05T10:00:00+08:00",
          valid_from: "2026-04-05T10:00:00+08:00",
          valid_to: "2026-04-05T11:00:00+08:00",
          subject: "周宁",
          action_status: "requested",
          episode_key: "vehicle-check",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "【周宁】先约4月5日上午10点去服务中心做车辆检查。",
        timestamp: 1,
      }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.valid_from).toBe("2026-04-05T10:00:00+08:00");
    expect(normalized.metadata.valid_to).toBeUndefined();
    expect(normalized.metadata.activity_start_time).toBe("2026-04-05T10:00:00+08:00");
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .toContain("removed_ungrounded_appointment_valid_to");
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .toContain("projected_iso_time_value_to_activity_start_time");
  });

  it.each([
    { label: "natural Chinese time", value: "4月5日上午10点" },
    { label: "timezone-free ISO", value: "2026-04-05T10:00:00" },
    { label: "invalid calendar ISO", value: "2026-02-30T10:00:00+08:00" },
  ])("does not project $label into a structured event time", ({ value }) => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: `车辆检查时间是${value}。`,
        type: "episodic",
        priority: 70,
        scene_name: "schedule-draft",
        source_message_ids: ["u1"],
        metadata: {
          domain: "schedule",
          slot: "appointment_time",
          value,
          subject: "driver",
          action_status: "requested",
        },
      },
      sourceMessages: [{ id: "u1", role: "user", content: `车辆检查时间是${value}`, timestamp: 1 }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.activity_start_time).toBeUndefined();
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs ?? [])
      .not.toContain("projected_iso_time_value_to_activity_start_time");
  });

  it("keeps an explicit structured event time instead of replacing it from value", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "车辆检查时间是4月5日上午10点。",
        type: "episodic",
        priority: 70,
        scene_name: "schedule",
        source_message_ids: ["u1"],
        metadata: {
          domain: "schedule",
          slot: "appointment_time",
          value: "2026-04-05T10:00:00+08:00",
          activity_start_time: "2026-04-05T09:30:00+08:00",
          subject: "driver",
          action_status: "requested",
        },
      },
      sourceMessages: [{ id: "u1", role: "user", content: "车辆检查时间是4月5日上午10点", timestamp: 1 }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.activity_start_time).toBe("2026-04-05T09:30:00+08:00");
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs ?? [])
      .not.toContain("projected_iso_time_value_to_activity_start_time");
  });

  it("retains an appointment end grounded by an explicit range", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "周宁预约4月5日上午10点到11点做车辆检查。",
        type: "episodic",
        priority: 70,
        scene_name: "schedule",
        source_message_ids: ["u1"],
        metadata: {
          domain: "schedule",
          slot: "appointment_time",
          value: "2026-04-05T10:00:00+08:00",
          valid_from: "2026-04-05T10:00:00+08:00",
          valid_to: "2026-04-05T11:00:00+08:00",
          subject: "周宁",
          action_status: "requested",
          episode_key: "vehicle-check",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "【周宁】预约4月5日上午10点到11点做车辆检查。",
        timestamp: 1,
      }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.valid_to).toBe("2026-04-05T11:00:00+08:00");
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .not.toContain("removed_ungrounded_appointment_valid_to");
  });

  it("does not borrow appointment-end evidence from another event in the same message", () => {
    const source = "4月1日10点开会，4月2日10点到11点保养。";
    const normalizeEvent = (value: string, end: string) => {
      const start = source.indexOf(value);
      return normalizeCockpitExtractedMemory({
        memory: {
          content: `用户安排${value}。`,
          type: "episodic",
          priority: 70,
          scene_name: "schedule",
          source_message_ids: ["u1"],
          metadata: {
            domain: "schedule",
            slot: "appointment_content",
            value,
            valid_from: "2026-04-01T10:00:00+08:00",
            valid_to: end,
            activity_start_time: "2026-04-01T10:00:00+08:00",
            activity_end_time: end,
            subject: "user",
            action_status: "requested",
            episode_key: value,
          },
        },
        sourceMessages: [{ id: "u1", role: "user", content: source, timestamp: 1 }],
        sourceEvidenceSpans: [{ start, end: start + value.length }],
        sessionId: "session-1",
      });
    };

    const meeting = normalizeEvent("开会", "2026-04-01T10:30:00+08:00");
    const maintenance = normalizeEvent("保养", "2026-04-02T11:00:00+08:00");

    expect(meeting.metadata.valid_to).toBeUndefined();
    expect(meeting.metadata.activity_end_time).toBeUndefined();
    expect(maintenance.metadata.valid_to).toBe("2026-04-02T11:00:00+08:00");
    expect(maintenance.metadata.activity_end_time).toBe("2026-04-02T11:00:00+08:00");
  });

  it("binds a generic first-person subject to an explicit transcript speaker", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户请求把常用目的地设为奥林匹克公园。",
        type: "episodic",
        priority: 70,
        scene_name: "legacy-free-route-label",
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "奥林匹克公园",
          subject: "user",
          state_key: "navigation|user|unspecified-vehicle|unspecified-zone|destination",
          episode_key: "route-1",
          action_status: "requested",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "[session:001] [source_role=user] 嗯，那个，【冯遥】我的常用目的地是奥林匹克公园。",
        timestamp: 1,
      }],
      sessionId: "session-1",
    });

    expect(normalized.scene_name).toBe("navigation");
    expect(normalized.metadata).toMatchObject({
      subject: "冯遥",
      state_key: "navigation|冯遥|unspecified-vehicle|unspecified-zone|destination",
      construction_quality: {
        status: "complete",
        repairs: expect.arrayContaining([
          "bound_first_person_subject_to_explicit_speaker",
          "canonicalized_state_key",
        ]),
      },
    });
  });

  it("does not replace an explicitly different target person with the speaker", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "赵琪的副驾温度偏好是24度。",
        type: "persona",
        priority: 70,
        scene_name: "climate",
        source_message_ids: ["u1"],
        metadata: {
          domain: "climate",
          slot: "temperature",
          value: 24,
          subject: "赵琪",
          seat_zone: "front-passenger",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "【冯遥】赵琪坐副驾时温度设为24度。",
        timestamp: 1,
      }],
      sessionId: "session-1",
    });

    expect(normalized.metadata).toMatchObject({
      subject: "赵琪",
      state_key: "climate|赵琪|unspecified-vehicle|front-passenger|temperature",
      construction_quality: { status: "complete" },
    });
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .not.toContain("bound_first_person_subject_to_explicit_speaker");
  });

  it("marks a generic subject partial when its sources contain multiple explicit speakers", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户将温度设为24度。",
        type: "persona",
        priority: 70,
        scene_name: "climate",
        source_message_ids: ["u1", "u2"],
        metadata: {
          domain: "climate",
          slot: "temperature",
          value: 24,
          subject: "user",
        },
      },
      sourceMessages: [
        { id: "u1", role: "user", content: "【冯遥】温度设为24度。", timestamp: 1 },
        { id: "u2", role: "user", content: "【尤遥】我也设为24度。", timestamp: 2 },
      ],
      sessionId: "session-1",
    });

    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["ambiguous_source_speaker"]),
    });
  });

  it("marks unknown evidence lineage invalid instead of silently accepting it", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "去公司",
        type: "semantic",
        priority: 60,
        scene_name: "navigation",
        source_message_ids: ["invented-id"],
        metadata: { domain: "navigation", slot: "destination", value: "公司" },
      },
      sourceMessages: [{ id: "u1", role: "user", content: "去公司", timestamp: 1 }],
      sessionId: "session-1",
      constructionModel: "deepseek-v4-flash",
    });

    expect(normalized.source_message_ids).toEqual([]);
    expect(normalized.metadata?.construction_quality).toMatchObject({
      status: "invalid",
      issues: expect.arrayContaining(["unknown_source_message_id", "missing_user_source"]),
    });
  });

  it("does not accept unknown sentinels as a real cockpit scope", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户明早七点半接人。",
        type: "episodic",
        priority: 70,
        scene_name: "navigation",
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "pickup_time",
          value: "07:30",
          subject: "unknown",
          action_status: "requested",
        },
      },
      sourceMessages: [{ id: "u1", role: "user", content: "明早七点半接人", timestamp: 1 }],
      sessionId: "session-1",
    });

    expect(normalized.metadata).toMatchObject({
      state_key: "navigation|unspecified-subject|unspecified-vehicle|unspecified-zone|pickup_time",
      construction_quality: {
        status: "partial",
        issues: expect.arrayContaining(["missing_scope"]),
      },
    });
  });

  it("qualifies independent price constraints and canonicalizes their target", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "冯遥要求门票不超过20元。",
        type: "episodic",
        priority: 70,
        scene_name: "selection",
        source_message_ids: ["u1"],
        metadata: {
          domain: "search",
          slot: "budget_constraint",
          value: "门票20元以下",
          constraint_target: "门票",
          subject: "user",
          action_status: "requested",
        },
      },
      sourceMessages: [{ id: "u1", role: "user", content: "【冯遥】门票价格20元以下。", timestamp: 1 }],
      sessionId: "session-1",
    });

    expect(normalized.metadata).toMatchObject({
      domain: "selection",
      slot: "price_constraint",
      constraint_target: "ticket",
      subject: "冯遥",
      state_key: "selection|冯遥|unspecified-vehicle|unspecified-zone|price_constraint@ticket",
      construction_quality: {
        status: "complete",
        repairs: expect.arrayContaining([
          "canonicalized_slot_token",
          "canonicalized_controlled_slot_domain",
          "canonicalized_constraint_target",
        ]),
      },
    });
  });

  it("derives stable cross-session identities for source-verified named state members", () => {
    const normalizeNamedState = (
      sourceId: string,
      source: string,
      value: string,
      sessionId: string,
      episodeKey: string,
      timestamp: number,
    ) => normalizeCockpitExtractedMemory({
      memory: {
        content: `程野的早餐地点是${value}。`,
        type: "instruction",
        priority: 75,
        scene_name: "navigation",
        source_message_ids: [sourceId],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value,
          subject: "程野",
          state_qualifier: "早餐地点",
          relation: "asserted",
          episode_key: episodeKey,
        },
      },
      sourceMessages: [{ id: sourceId, role: "user", content: source, timestamp }],
      sessionId,
    });

    const first = normalizeNamedState(
      "u1",
      "【程野】路线口令：早餐地点是松林文化馆，返程地点是云岭客运站。",
      "松林文化馆",
      "session-a",
      "model-episode-a",
      1,
    );
    const second = normalizeNamedState(
      "u2",
      "【程野】更新路线口令：早餐地点改成北岸书店，返程地点仍是云岭客运站。",
      "北岸书店",
      "session-b",
      "model-episode-b",
      2,
    );

    expect(first.metadata).toMatchObject({
      state_qualifier: "早餐地点",
      state_key: expect.stringMatching(/^navigation\|程野\|unspecified-vehicle\|unspecified-zone\|destination@q2-[a-f0-9]{64}$/u),
      episode_key: expect.stringMatching(/^qualified-state:[a-f0-9]{64}$/u),
      construction_quality: { status: "complete" },
    });
    expect(second.metadata.state_key).toBe(first.metadata.state_key);
    expect(second.metadata.episode_key).toBe(first.metadata.episode_key);
    expect(hasCanonicalCockpitQualifiedStateEpisode(first)).toBe(true);
    expect(hasCanonicalCockpitQualifiedStateEpisode(second)).toBe(true);
    expect((first.metadata.construction_quality as Record<string, unknown>).repairs)
      .toContain("canonicalized_qualified_state_episode_key");
  });

  it("uses a collision-safe qualifier digest while preserving NFKC identity", () => {
    const normalizeNamedState = (qualifier: string, sourceId: string, sessionId: string) =>
      normalizeCockpitExtractedMemory({
        memory: {
          content: `程野的${qualifier}是松林文化馆。`,
          type: "instruction",
          priority: 75,
          scene_name: "navigation",
          source_message_ids: [sourceId],
          metadata: {
            domain: "navigation",
            slot: "destination",
            value: "松林文化馆",
            subject: "程野",
            state_qualifier: qualifier,
            relation: "asserted",
          },
        },
        sourceMessages: [{
          id: sourceId,
          role: "user",
          content: `【程野】路线口令：${qualifier}是松林文化馆，返程地点是云岭客运站。`,
          timestamp: 1,
        }],
        sessionId,
      });

    const slash = normalizeNamedState("A/B地点", "u-slash", "session-slash");
    const space = normalizeNamedState("A B地点", "u-space", "session-space");
    const fullWidth = normalizeNamedState("Ａ地点", "u-full", "session-full");
    const ascii = normalizeNamedState("A地点", "u-ascii", "session-ascii");
    const lower = normalizeNamedState("a地点", "u-lower", "session-lower");

    expect(slash.metadata.state_key).toMatch(/destination@q2-[a-f0-9]{64}$/u);
    expect(space.metadata.state_key).toMatch(/destination@q2-[a-f0-9]{64}$/u);
    expect(slash.metadata.state_key).not.toBe(space.metadata.state_key);
    expect(fullWidth.metadata.state_key).toBe(ascii.metadata.state_key);
    expect(fullWidth.metadata.episode_key).toBe(ascii.metadata.episode_key);
    expect(ascii.metadata.state_key).not.toBe(lower.metadata.state_key);
    expect(ascii.metadata.episode_key).not.toBe(lower.metadata.episode_key);
    expect([slash, space, fullWidth, ascii, lower].every((entry) =>
      (entry.metadata.construction_quality as Record<string, unknown>).status === "complete"
    )).toBe(true);
  });

  it("does not authorize an ordinary location property as a named navigation map member", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "乙的会议地点是北楼。",
        type: "instruction",
        priority: 70,
        scene_name: "navigation",
        source_message_ids: ["u-ordinary"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "北楼",
          subject: "乙",
          state_qualifier: "会议地点",
          relation: "asserted",
        },
      },
      sourceMessages: [{
        id: "u-ordinary",
        role: "user",
        content: "【乙】会议地点是北楼。",
        timestamp: 1,
      }],
      sessionId: "session-ordinary",
    });

    expect(normalized.metadata.state_qualifier).toBeUndefined();
    expect(normalized.metadata.state_key)
      .toBe("navigation|乙|unspecified-vehicle|unspecified-zone|destination");
    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["unverified_state_qualifier"]),
    });
  });

  it("does not authorize a named map member under a different model-selected person", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "甲的早餐地点是北楼。",
        type: "instruction",
        priority: 70,
        scene_name: "navigation",
        source_message_ids: ["u-wrong-owner"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "北楼",
          subject: "甲",
          state_qualifier: "早餐地点",
          relation: "asserted",
        },
      },
      sourceMessages: [{
        id: "u-wrong-owner",
        role: "user",
        content: "【乙】路线口令：早餐地点是北楼，返程地点是南楼。",
        timestamp: 1,
      }],
      sessionId: "session-wrong-owner",
    });

    expect(normalized.metadata.state_qualifier).toBeUndefined();
    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["unverified_state_qualifier"]),
    });
  });

  it("keeps dated, valid-time and conditional named records in event episodes", () => {
    const normalizePartitioned = (params: {
      sourceId: string;
      sessionId: string;
      type: "episodic" | "instruction";
      metadata: Record<string, unknown>;
    }) => normalizeCockpitExtractedMemory({
      memory: {
        content: "程野的早餐地点是松林文化馆。",
        type: params.type,
        priority: 75,
        scene_name: "navigation",
        source_message_ids: [params.sourceId],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "松林文化馆",
          subject: "程野",
          state_qualifier: "早餐地点",
          relation: "asserted",
          episode_key: "qualified-state:forged-by-model",
          ...params.metadata,
        },
      },
      sourceMessages: [{
        id: params.sourceId,
        role: "user",
        content: "【程野】路线口令：早餐地点是松林文化馆，返程地点是云岭客运站。",
        timestamp: 1,
      }],
      sessionId: params.sessionId,
    });

    const datedA = normalizePartitioned({
      sourceId: "u-date-a",
      sessionId: "session-date-a",
      type: "episodic",
      metadata: {
        action_status: "requested",
        activity_start_time: "2026-09-03T08:00:00+08:00",
      },
    });
    const datedB = normalizePartitioned({
      sourceId: "u-date-b",
      sessionId: "session-date-b",
      type: "episodic",
      metadata: {
        action_status: "requested",
        activity_start_time: "2026-09-04T08:00:00+08:00",
      },
    });
    const valid = normalizePartitioned({
      sourceId: "u-valid",
      sessionId: "session-valid",
      type: "instruction",
      metadata: { valid_from: "2026-09-05T00:00:00+08:00" },
    });
    const conditional = normalizePartitioned({
      sourceId: "u-condition",
      sessionId: "session-condition",
      type: "instruction",
      metadata: { condition: "仅在工作日" },
    });

    expect(datedA.metadata.episode_key).not.toBe(datedB.metadata.episode_key);
    expect([datedA, datedB, valid, conditional].every((entry) =>
      typeof entry.metadata.episode_key === "string"
      && !entry.metadata.episode_key.startsWith("qualified-state:")
    )).toBe(true);
    expect([datedA, datedB, valid, conditional].every((entry) =>
      !hasCanonicalCockpitQualifiedStateEpisode(entry)
      && ((entry.metadata.construction_quality as Record<string, unknown>).repairs as string[])
        .includes("replaced_ineligible_qualified_state_episode_key")
    )).toBe(true);
  });

  it("rejects a value masquerading as a state qualifier", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "程野的早餐地点是松林文化馆。",
        type: "instruction",
        priority: 75,
        scene_name: "navigation",
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "松林文化馆",
          subject: "程野",
          state_qualifier: "松林文化馆",
          relation: "asserted",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "【程野】早餐地点是松林文化馆。",
        timestamp: 1,
      }],
      sessionId: "session-a",
    });

    expect(normalized.metadata.state_qualifier).toBeUndefined();
    expect(normalized.metadata.state_key)
      .toBe("navigation|程野|unspecified-vehicle|unspecified-zone|destination");
    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["invalid_state_qualifier"]),
    });
  });

  it("completes a missing qualifier only from one exact superseded record identity", () => {
    const priorMentionedAt = new Date(1).toISOString();
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "程野把它改成北岸书店。",
        type: "instruction",
        priority: 75,
        scene_name: "navigation",
        source_message_ids: ["u2"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "北岸书店",
          subject: "程野",
          relation: "updated",
          supersedes: ["prior-breakfast"],
        },
      },
      sourceMessages: [{ id: "u2", role: "user", content: "【程野】把它改成北岸书店。", timestamp: 2 }],
      sessionId: "session-b",
      knownLineage: [{
        recordId: "prior-breakfast",
        type: "instruction",
        scene_name: "navigation",
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "松林文化馆",
          subject: "程野",
          state_qualifier: "早餐地点",
          state_key: "navigation|程野|unspecified-vehicle|unspecified-zone|destination@早餐地点",
          episode_key: "qualified-state:0123456789abcdef01234567",
          relation: "asserted",
          mentioned_at: priorMentionedAt,
        },
      }],
    });

    expect(normalized.metadata).toMatchObject({
      state_qualifier: "早餐地点",
      state_key: "navigation|程野|unspecified-vehicle|unspecified-zone|destination@早餐地点",
      episode_key: "qualified-state:0123456789abcdef01234567",
      supersedes: ["prior-breakfast"],
      construction_quality: {
        status: "complete",
        repairs: expect.arrayContaining([
          "reused_superseded_state_identity",
          "reused_superseded_episode_key",
        ]),
      },
    });
  });

  it("rejects a pronoun update when more than one live named member is compatible", () => {
    const priorMentionedAt = new Date(1).toISOString();
    const lineage = [
      ["prior-breakfast", "早餐地点", "松林文化馆"],
      ["prior-return", "返程地点", "云岭客运站"],
    ].map(([recordId, qualifier, value]) => ({
      recordId,
      type: "instruction" as const,
      scene_name: "navigation",
      metadata: {
        domain: "navigation",
        slot: "destination",
        value,
        subject: "程野",
        state_qualifier: qualifier,
        state_key: `navigation|程野|unspecified-vehicle|unspecified-zone|destination@${qualifier}`,
        episode_key: `qualified-state:${recordId}`,
        relation: "asserted",
        mentioned_at: priorMentionedAt,
      },
    }));
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "程野把它改成北岸书店。",
        type: "instruction",
        priority: 75,
        scene_name: "navigation",
        source_message_ids: ["u2"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "北岸书店",
          subject: "程野",
          relation: "updated",
          supersedes: ["prior-breakfast"],
        },
      },
      sourceMessages: [{
        id: "u2",
        role: "user",
        content: "【程野】把它改成北岸书店。",
        timestamp: 2,
      }],
      sessionId: "session-b",
      knownLineage: lineage,
    });

    expect(normalized.metadata.state_qualifier).toBeUndefined();
    expect(normalized.metadata.state_key)
      .toBe("navigation|程野|unspecified-vehicle|unspecified-zone|destination");
    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["unverified_supersedes"]),
    });
  });

  it("canonicalizes a model-emitted default destination to the controlled destination slot", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "蒋澄的默认导航目的地是北京人卫酒店。",
        type: "episodic",
        priority: 70,
        scene_name: "navigation",
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "default_destination",
          value: "北京人卫酒店",
          subject: "蒋澄",
          action_status: "requested",
          state_key: "navigation|蒋澄|unspecified-vehicle|unspecified-zone|default_destination",
        },
      },
      sourceMessages: [{
        id: "u1",
        role: "user",
        content: "【蒋澄】默认目的地设为北京人卫酒店。",
        timestamp: 1,
      }],
      sessionId: "session-1",
      constructionModel: "deepseek-v4-flash",
    });

    expect(normalized.metadata).toMatchObject({
      domain: "navigation",
      slot: "destination",
      value: "北京人卫酒店",
      subject: "蒋澄",
      state_key: "navigation|蒋澄|unspecified-vehicle|unspecified-zone|destination",
      construction_quality: {
        status: "complete",
        repairs: expect.arrayContaining([
          "canonicalized_slot_token",
          "canonicalized_state_key",
        ]),
      },
    });
  });

  it("fails closed on placeholder fact values", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户请求推荐一个景点。",
        type: "episodic",
        priority: 70,
        scene_name: "selection",
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "unresolved",
          subject: "user",
          action_status: "requested",
        },
      },
      sourceMessages: [{ id: "u1", role: "user", content: "推荐一个景点。", timestamp: 1 }],
      sessionId: "session-1",
    });

    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["placeholder_value"]),
    });
  });

  it("records lossless provenance and identity normalization as repairs, not semantic defects", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "驾驶员请求22度。",
        type: "episodic",
        priority: 70,
        scene_name: "climate",
        source_message_ids: ["u1", "a1"],
        metadata: {
          domain: "climate",
          slot: "temperature",
          value: 22,
          subject: "driver",
          seat_zone: "driver",
          action_status: "requested",
          state_key: "wrong|passenger|temperature",
          mentioned_at: "2020-01-01T00:00:00.000Z",
        },
      },
      sourceMessages: [
        { id: "u1", role: "user", content: "温度调到22度", timestamp: Date.parse("2026-08-29T10:00:00+08:00") },
        { id: "a1", role: "assistant", content: "好的", timestamp: Date.parse("2026-08-29T10:00:01+08:00") },
      ],
      sessionId: "session-1",
    });

    expect(normalized.metadata).toMatchObject({
      state_key: "climate|driver|unspecified-vehicle|driver|temperature",
      mentioned_at: "2026-08-29T02:00:00.000Z",
      construction_quality: {
        status: "complete",
        issues: [],
        repairs: expect.arrayContaining([
          "removed_unsupported_source_role",
          "canonicalized_state_key",
          "bound_mentioned_at_to_evidence",
        ]),
      },
    });
  });

  it("normalizes scalar supersedes and reuses an exact prior transition identity", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "驾驶员把目的地改为上海南站。",
        type: "episodic",
        priority: 78,
        scene_name: "navigation",
        source_message_ids: ["u2"],
        metadata: {
          domain: "navigation_destination",
          slot: "target",
          value: "上海南站",
          subject: "driver",
          relation: "updated",
          action_status: "requested",
          state_key: "navigation_destination|driver|car|front|target",
          episode_key: "invented-new-episode",
          supersedes: "route-v1",
        },
      },
      sourceMessages: [
        { id: "u2", role: "user", content: "目的地改成上海南站", timestamp: 2 },
      ],
      sessionId: "session-2",
      knownLineage: [{
        recordId: "route-v1",
        type: "episodic",
        scene_name: "legacy-lineage-route-label",
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "虹桥火车站",
          subject: "driver",
          vehicle_scope: "car",
          seat_zone: "front",
          state_key: "navigation|driver|car|front|destination",
          episode_key: "pickup-lihang",
          mentioned_at: "1970-01-01T00:00:00.001Z",
        },
      }],
    });

    expect(normalized.scene_name).toBe("navigation");
    expect(normalized.metadata).toMatchObject({
      domain: "navigation",
      slot: "destination",
      value: "上海南站",
      state_key: "navigation|driver|car|front|destination",
      episode_key: "pickup-lihang",
      supersedes: ["route-v1"],
      relation: "updated",
      construction_quality: {
        status: "complete",
        issues: [],
        repairs: expect.arrayContaining([
          "normalized_scalar_supersedes",
          "reused_superseded_state_identity",
          "reused_superseded_episode_key",
        ]),
      },
    });
  });

  it.each([
    {
      label: "another memory type",
      lineageType: "instruction" as const,
      lineageScene: "navigation",
      lineageMentionedAt: "1970-01-01T00:00:00.001Z",
    },
    {
      label: "another controlled class",
      lineageType: "episodic" as const,
      lineageScene: "notification",
      lineageDomain: "notification",
      lineageSlot: "broadcast_policy",
      lineageStateKey: "notification|driver|car|front|broadcast_policy",
      lineageMentionedAt: "1970-01-01T00:00:00.001Z",
    },
    {
      label: "the source-event future",
      lineageType: "episodic" as const,
      lineageScene: "navigation",
      lineageDomain: "navigation",
      lineageSlot: "destination",
      lineageStateKey: "navigation|driver|car|front|destination",
      lineageMentionedAt: "1970-01-01T00:00:00.003Z",
    },
  ])("does not reuse a referenced predecessor from $label", ({
    lineageType,
    lineageScene,
    lineageDomain = "navigation",
    lineageSlot = "destination",
    lineageStateKey = "navigation|driver|car|front|destination",
    lineageMentionedAt,
  }) => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "驾驶员把目的地改为上海南站。",
        type: "episodic",
        priority: 78,
        scene_name: "navigation",
        source_message_ids: ["u2"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "上海南站",
          subject: "driver",
          relation: "updated",
          action_status: "requested",
          state_key: "navigation|driver|car|front|destination",
          episode_key: "unverified-new-episode",
          supersedes: ["route-v1"],
        },
      },
      sourceMessages: [{ id: "u2", role: "user", content: "目的地改成上海南站", timestamp: 2 }],
      sessionId: "session-2",
      knownLineage: [{
        recordId: "route-v1",
        type: lineageType,
        scene_name: lineageScene,
        metadata: {
          domain: lineageDomain,
          slot: lineageSlot,
          value: "虹桥火车站",
          subject: "driver",
          vehicle_scope: "car",
          seat_zone: "front",
          state_key: lineageStateKey,
          episode_key: "pickup-lihang",
          mentioned_at: lineageMentionedAt,
        },
      }],
    });

    expect(normalized.metadata.episode_key).toBe("unverified-new-episode");
    expect(normalized.metadata.construction_quality).toMatchObject({
      status: "partial",
      issues: expect.arrayContaining(["unverified_supersedes"]),
    });
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .not.toEqual(expect.arrayContaining([
        "reused_superseded_state_identity",
        "reused_superseded_episode_key",
      ]));
  });

  it.each([
    { label: "an episode alias", supersedes: "shared-appointment" },
    { label: "a record ID", supersedes: "prior-time" },
  ])("does not rewrite a transition to the wrong state through $label", ({ supersedes }) => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "把车辆检查的内容改成轮胎检查。",
        type: "episodic",
        priority: 78,
        scene_name: "schedule",
        source_message_ids: ["u2"],
        metadata: {
          domain: "schedule",
          slot: "appointment_content",
          value: "轮胎检查",
          subject: "driver",
          vehicle_scope: "car",
          seat_zone: "driver",
          relation: "updated",
          action_status: "requested",
          state_key: "schedule|driver|car|driver|appointment_content",
          episode_key: "shared-appointment",
          supersedes: [supersedes],
        },
      },
      sourceMessages: [{
        id: "u2",
        role: "user",
        content: "把车辆检查的内容改成轮胎检查",
        timestamp: 2,
      }],
      sessionId: "session-2",
      knownLineage: [{
        recordId: "prior-time",
        type: "episodic",
        scene_name: "schedule",
        metadata: {
          domain: "schedule",
          slot: "appointment_time",
          value: "2026-04-05T10:00:00+08:00",
          subject: "driver",
          vehicle_scope: "car",
          seat_zone: "driver",
          state_key: "schedule|driver|car|driver|appointment_time",
          episode_key: "shared-appointment",
          mentioned_at: "1970-01-01T00:00:00.001Z",
        },
      }],
    });

    expect(normalized.metadata).toMatchObject({
      domain: "schedule",
      slot: "appointment_content",
      state_key: "schedule|driver|car|driver|appointment_content",
      episode_key: "shared-appointment",
      construction_quality: {
        status: "partial",
        issues: expect.arrayContaining(["unverified_supersedes"]),
      },
    });
    expect((normalized.metadata.construction_quality as Record<string, unknown>).repairs)
      .not.toEqual(expect.arrayContaining(["reused_superseded_state_identity"]));
  });

  it("does not reuse a ticket predecessor for a per-capita price constraint", () => {
    const normalized = normalizeCockpitExtractedMemory({
      memory: {
        content: "人均预算改成两百元。",
        type: "instruction",
        priority: 78,
        scene_name: "selection",
        source_message_ids: ["u2"],
        metadata: {
          domain: "selection",
          slot: "price_constraint",
          constraint_target: "per_capita",
          value: 200,
          subject: "driver",
          relation: "updated",
          state_key: "selection|driver|car|front|price_constraint@ticket",
          episode_key: "new-budget-episode",
          supersedes: ["ticket-budget"],
        },
      },
      sourceMessages: [{ id: "u2", role: "user", content: "人均预算改成两百元", timestamp: 2 }],
      sessionId: "session-2",
      knownLineage: [{
        recordId: "ticket-budget",
        type: "instruction",
        scene_name: "selection",
        metadata: {
          domain: "selection",
          slot: "price_constraint",
          constraint_target: "ticket",
          value: 300,
          subject: "driver",
          vehicle_scope: "car",
          seat_zone: "front",
          state_key: "selection|driver|car|front|price_constraint@ticket",
          episode_key: "ticket-budget-episode",
          mentioned_at: "1970-01-01T00:00:00.001Z",
        },
      }],
    });

    expect(normalized.metadata).toMatchObject({
      constraint_target: "per_capita",
      state_key: "selection|driver|unspecified-vehicle|unspecified-zone|price_constraint@per_capita",
      episode_key: "new-budget-episode",
      construction_quality: {
        status: "partial",
        issues: expect.arrayContaining(["unverified_supersedes"]),
      },
    });
  });

  it("downgrades terminal action claims without tool evidence but accepts an explicit tool result", () => {
    const withoutTool = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户请求打开车窗，助手称已完成。",
        type: "episodic",
        priority: 70,
        scene_name: "vehicle-control",
        source_message_ids: ["u1", "a1"],
        metadata: {
          domain: "vehicle_control",
          slot: "window",
          value: "open",
          subject: "user",
          action_status: "completed",
        },
      },
      sourceMessages: [
        { id: "u1", role: "user", content: "打开车窗", timestamp: 1 },
        { id: "a1", role: "assistant", content: "已经打开", timestamp: 2 },
      ],
      sessionId: "session-1",
    });
    expect(withoutTool.metadata).toMatchObject({
      action_status: "requested",
      evidence_roles: ["user"],
      construction_quality: {
        status: "complete",
        repairs: expect.arrayContaining([
          "removed_unsupported_source_role",
          "downgraded_action_without_tool_evidence",
        ]),
      },
    });

    const withTool = normalizeCockpitExtractedMemory({
      memory: {
        content: "用户打开车窗的请求已执行。",
        type: "episodic",
        priority: 70,
        scene_name: "vehicle-control",
        source_message_ids: ["u1", "t1"],
        metadata: {
          domain: "vehicle_control",
          slot: "window",
          value: "open",
          subject: "user",
          action_status: "completed",
        },
      },
      sourceMessages: [
        { id: "u1", role: "user", content: "打开车窗", timestamp: 1 },
        { id: "t1", role: "assistant", content: "[source_role=tool] {\"status\":\"success\"}", timestamp: 2 },
      ],
      sessionId: "session-1",
    });
    expect(withTool.metadata).toMatchObject({
      action_status: "completed",
      evidence_roles: ["user", "tool"],
      construction_quality: { status: "complete" },
    });
    expect((withTool.metadata.construction_quality as Record<string, unknown>).repairs)
      .not.toContain("downgraded_action_without_tool_evidence");
  });

  it("keeps TencentDB merge lineage distinct from semantic update lineage", () => {
    const base = {
      schema_version: "cockpit-state-v1",
      relation: "asserted",
      construction_quality: {
        status: "partial",
        score: 90,
        issues: ["missing_supersedes"],
        source_count: 1,
        user_source_count: 1,
      },
    };
    expect(finalizeCockpitMetadataAfterDedup(base, "update", ["old-state"])).toMatchObject({
      relation: "updated",
      supersedes: ["old-state"],
      construction_quality: { status: "complete", score: 100, issues: [] },
    });
    expect(finalizeCockpitMetadataAfterDedup(base, "merge", ["duplicate-record"])).toMatchObject({
      relation: "asserted",
      merged_from_record_ids: ["duplicate-record"],
      construction_quality: { status: "partial", issues: ["missing_supersedes"] },
    });
  });

  it("preserves old scope metadata and increments a merged record version", async () => {
    const appendFile = vi.fn(async () => undefined);
    const upsertL1 = vi.fn(async () => true);
    const vectorStore = {
      queryL1Records: vi.fn(async () => [{
        record_id: "route-v1",
        session_id: "session-0",
        team_id: "team-a",
        user_id: "driver-a",
        agent_id: "vehicle-agent",
        version: 4,
        metadata_json: JSON.stringify({
          occupant_scope: "driver",
          vehicle_scope: "vehicle-a",
          action_status: "requested",
          mentioned_at: "2026-08-25T09:00:00+08:00",
        }),
      }]),
      deleteL1Batch: vi.fn(async () => true),
      upsertL1,
    } as unknown as IMemoryStore;
    const storage = { appendFile } as unknown as StorageAdapter;

    const record = await writeMemory({
      memory: {
        content: "The route was changed to Hongqiao Airport.",
        type: "episodic",
        priority: 78,
        scene_name: "navigation",
        source_message_ids: ["u2"],
        metadata: {
          action_status: "selected",
          target: "Hongqiao Airport",
        },
      },
      decision: {
        record_id: "route-v2",
        action: "update",
        target_ids: ["route-v1"],
        merged_content: "The driver replaced the old route with Hongqiao Airport.",
        merged_type: "episodic",
        merged_priority: 82,
        merged_timestamps: ["2026-08-25T01:00:00.000Z"],
        merged_metadata: {
          action_status: "confirmed",
          episode_key: "navigation:session-1",
          supersedes: ["route-v1"],
        },
      },
      baseDir: "/unused",
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      userId: "driver-a",
      agentId: "vehicle-agent",
      vectorStore,
      storage,
    });

    expect(record?.version).toBe(5);
    expect(record?.metadata).toEqual({
      occupant_scope: "driver",
      vehicle_scope: "vehicle-a",
      action_status: "confirmed",
      mentioned_at: "2026-08-25T09:00:00+08:00",
      target: "Hongqiao Airport",
      episode_key: "navigation:session-1",
      supersedes: ["route-v1"],
      source_message_ids: ["u2"],
      source_session_id: "session-1",
      source_session_ids: ["session-0", "session-1"],
    });
    expect(vectorStore.deleteL1Batch).toHaveBeenCalledWith(
      ["route-v1"],
      expect.objectContaining({ sessionKey: "session-key", sessionId: "session-1" }),
    );
    expect(appendFile).toHaveBeenCalledTimes(1);
    expect(upsertL1).toHaveBeenCalledWith(
      expect.objectContaining({ version: 5 }),
      undefined,
    );
  });
});
