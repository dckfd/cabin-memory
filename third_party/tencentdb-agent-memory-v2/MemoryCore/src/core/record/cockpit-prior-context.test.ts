import { describe, expect, it, vi } from "vitest";

import type { IMemoryStore, L1RecordRow } from "../store/types.js";
import { loadCockpitPriorMemoryContext } from "./cockpit-prior-context.js";

const scope = {
  teamId: "team-a",
  userId: "user-a",
  agentId: "agent-a",
  taskId: "task-a",
};

function row(overrides: Partial<L1RecordRow> = {}): L1RecordRow {
  return {
    record_id: "record-1",
    content: "用户计划前往虹桥火车站。",
    type: "episodic",
    priority: 70,
    scene_name: "navigation",
    session_key: "session-key",
    session_id: "old-session",
    team_id: scope.teamId,
    task_id: scope.taskId,
    user_id: scope.userId,
    agent_id: scope.agentId,
    version: 1,
    timestamp_str: "2026-08-30T09:00:00.000Z",
    timestamp_start: "2026-08-30T09:00:00.000Z",
    timestamp_end: "2026-08-30T09:00:00.000Z",
    created_time: "2026-08-30T09:00:00.000Z",
    updated_time: "2026-08-30T09:00:00.000Z",
    metadata_json: JSON.stringify({
      schema_version: "cockpit-state-v1",
      domain: "navigation",
      slot: "destination",
      value: "虹桥火车站",
      target: "虹桥火车站",
      state_qualifier: "早餐地点",
      state_key: "navigation|user|car|driver|destination@早餐地点",
      episode_key: "route-1",
      activity_start_time: "2026-08-30T09:30:00.000Z",
      condition: "下班后",
      trigger: "离开公司",
      record_kind: "event",
      source_message_ids: ["message-1"],
      source_session_id: "old-session",
      source_session_ids: ["old-session"],
      construction_quality: { status: "complete" },
      private_unrelated_field: "must-not-enter-prompt-context",
    }),
    ...overrides,
  };
}

describe("bounded cockpit prior construction context", () => {
  it("returns nothing unless the exact four-dimension scope is available", async () => {
    const queryL1Paginated = vi.fn(async () => ({ rows: [row()], total: 1 }));
    const result = await loadCockpitPriorMemoryContext({
      vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      teamId: scope.teamId,
      userId: scope.userId,
      agentId: scope.agentId,
    });

    expect(result).toEqual([]);
    expect(queryL1Paginated).not.toHaveBeenCalled();
  });

  it("keeps only prior complete cockpit records in the exact scope", async () => {
    const queryL1Paginated = vi.fn(async () => ({
      rows: [
        row(),
        row({ record_id: "current", session_id: "current-session" }),
        row({ record_id: "other-user", user_id: "other-user" }),
        row({
          record_id: "partial",
          metadata_json: JSON.stringify({
            schema_version: "cockpit-state-v1",
            construction_quality: { status: "partial" },
          }),
        }),
        row({
          record_id: "generic",
          metadata_json: JSON.stringify({ construction_quality: { status: "complete" } }),
        }),
      ],
      total: 5,
    }));
    const result = await loadCockpitPriorMemoryContext({
      vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      ...scope,
      currentSessionId: "current-session",
    });

    expect(queryL1Paginated).toHaveBeenCalledWith({
      ...scope,
      limit: 72,
      offset: 0,
    });
    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      record_id: "record-1",
      session_id: "old-session",
      type: "episodic",
      scene_name: "navigation",
    });
    expect(result[0].metadata).toMatchObject({
      schema_version: "cockpit-state-v1",
      state_qualifier: "早餐地点",
      state_key: "navigation|user|car|driver|destination@早餐地点",
      episode_key: "route-1",
      target: "虹桥火车站",
      activity_start_time: "2026-08-30T09:30:00.000Z",
      condition: "下班后",
      trigger: "离开公司",
      record_kind: "event",
      source_message_ids: ["message-1"],
      source_session_id: "old-session",
      source_session_ids: ["old-session"],
    });
    expect(result[0].metadata).not.toHaveProperty("private_unrelated_field");
    expect(result[0].metadata).not.toHaveProperty("construction_quality");
  });

  it("excludes future event-time records during out-of-order recovery", async () => {
    const queryL1Paginated = vi.fn(async () => ({
      rows: [
        row({
          record_id: "future",
          session_id: "future-session",
          metadata_json: JSON.stringify({
            schema_version: "cockpit-state-v1",
            mentioned_at: "2026-08-31T09:00:00.000Z",
            construction_quality: { status: "complete" },
          }),
        }),
        row({
          record_id: "prior",
          session_id: "prior-session",
          metadata_json: JSON.stringify({
            schema_version: "cockpit-state-v1",
            mentioned_at: "2026-08-29T09:00:00.000Z",
            construction_quality: { status: "complete" },
          }),
        }),
        row({
          record_id: "unproven",
          session_id: "unknown-time-session",
          timestamp_str: "not-a-time",
          metadata_json: JSON.stringify({
            schema_version: "cockpit-state-v1",
            construction_quality: { status: "complete" },
          }),
        }),
      ],
      total: 3,
    }));

    const result = await loadCockpitPriorMemoryContext({
      vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      ...scope,
      currentSessionId: "current-session",
      currentEventTimeMs: Date.parse("2026-08-30T09:00:00.000Z"),
    });

    expect(result.map((entry) => entry.record_id)).toEqual(["prior"]);
  });

  it("pages past newer persisted rows to find bounded prior evidence", async () => {
    const futureRows = Array.from({ length: 72 }, (_, index) => row({
      record_id: `future-${index}`,
      session_id: `future-session-${index}`,
      metadata_json: JSON.stringify({
        schema_version: "cockpit-state-v1",
        mentioned_at: "2026-08-31T09:00:00.000Z",
        construction_quality: { status: "complete" },
      }),
    }));
    const prior = row({
      record_id: "prior-after-page",
      session_id: "prior-after-page-session",
      metadata_json: JSON.stringify({
        schema_version: "cockpit-state-v1",
        mentioned_at: "2026-08-29T09:00:00.000Z",
        construction_quality: { status: "complete" },
      }),
    });
    const allRows = [...futureRows, prior];
    const queryL1Paginated = vi.fn(async ({ limit, offset }: { limit: number; offset: number }) => ({
      rows: allRows.slice(offset, offset + limit),
      total: allRows.length,
    }));

    const result = await loadCockpitPriorMemoryContext({
      vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      ...scope,
      currentEventTimeMs: Date.parse("2026-08-30T09:00:00.000Z"),
    });

    expect(queryL1Paginated).toHaveBeenCalledTimes(2);
    expect(queryL1Paginated).toHaveBeenNthCalledWith(2, {
      ...scope,
      limit: 72,
      offset: 72,
    });
    expect(result.map((entry) => entry.record_id)).toEqual(["prior-after-page"]);
  });
});
