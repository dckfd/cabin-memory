import { describe, expect, it, vi } from "vitest";

import { extractL1Memories } from "./l1-extractor.js";
import { detectCockpitSourceCoverageObligations } from "./cockpit-source-coverage.js";
import type { LLMRunner } from "../types.js";
import type { IMemoryStore } from "../store/types.js";
import type { StorageAdapter } from "../storage/adapter.js";

const messages = [{
  id: "D1:1",
  role: "user" as const,
  content: "Caroline completed a meaningful charity race last weekend.",
  timestamp: Date.UTC(2023, 4, 8),
}];

function params(runner: LLMRunner, timeoutMs: number) {
  return {
    messages,
    sessionKey: "test-session",
    baseDir: "/tmp/tdai-l1-extractor-test",
    config: {},
    options: {
      enableDedup: false,
      llmRunner: runner,
      llmTimeoutMs: timeoutMs,
    },
  };
}

function coverageEvidenceSpans(
  content: string,
  slot: string,
  constraintTarget?: string,
  sourceId = "u1",
) {
  const obligation = detectCockpitSourceCoverageObligations([{
    id: sourceId,
    role: "user" as const,
    content,
    timestamp: 1,
  }]).find((entry) =>
    entry.slot === slot && entry.constraintTarget === constraintTarget
  );
  if (!obligation) throw new Error(`missing test coverage obligation for ${slot}`);
  const normalized = content.normalize("NFKC");
  return obligation.evidenceGroups.map((group) => ({
    start: group.start,
    end: group.end,
    quote: normalized.slice(group.start, group.end),
  }));
}

function coverageEvidenceGroupIds(
  content: string,
  slot: string,
  constraintTarget?: string,
  sourceId = "u1",
) {
  const obligation = detectCockpitSourceCoverageObligations([{
    id: sourceId,
    role: "user" as const,
    content,
    timestamp: 1,
  }]).find((entry) =>
    entry.slot === slot && entry.constraintTarget === constraintTarget
  );
  if (!obligation) throw new Error(`missing test coverage obligation for ${slot}`);
  return obligation.evidenceGroups.map((group) => group.id);
}

function cockpitStructuredRetryFixture() {
  const proposal = {
    content: "用户计划七点出发。",
    type: "episodic",
    priority: 70,
    source_message_ids: ["u1"],
    metadata: {
      domain: "navigation",
      slot: "departure_time",
      value: "07:00",
      subject: "user",
      state_key: "navigation|user|unspecified-vehicle|unspecified-zone|departure_time",
      episode_key: "route-1",
      relation: "asserted",
      action_status: "requested",
    } as Record<string, unknown>,
  };
  const scene = (memory: typeof proposal) => JSON.stringify([{
    scene_name: "navigation",
    message_ids: ["u1"],
    memories: [memory],
  }]);
  const reconciledProposal = structuredClone(proposal);
  reconciledProposal.metadata.input_candidate_ids = ["primary:0", "atomic:0"];
  return {
    primary: scene(proposal),
    atomic: scene(proposal),
    reconciled: scene(reconciledProposal),
    malformed: "```json\n[{\"scene_name\":\"navigation\",\"message_ids\":[\"u1\"],\"memories\":[{\"content\":\"用户说\"平时\"七点出发。\",\"type\":\"episodic\",\"priority\":70,\"source_message_ids\":[\"u1\"],\"metadata\":{}}]}]\n```",
  };
}

describe("extractL1Memories failure and timeout contract", () => {
  it("passes the resolved runtime timeout to the L1 runner", async () => {
    const run = vi.fn(async () => "[]");
    const result = await extractL1Memories(params({ run }, 600_000));

    expect(result.success).toBe(true);
    expect(run).toHaveBeenCalledWith(expect.objectContaining({
      taskId: "l1-extraction",
      timeoutMs: 600_000,
    }));
  });

  it("does not report malformed provider output as a successful empty extraction", async () => {
    const runner: LLMRunner = {
      run: vi.fn(async () => "upstream returned an incomplete response"),
    };

    const result = await extractL1Memories(params(runner, 600_000));

    expect(result.success).toBe(false);
    expect(result.storedCount).toBe(0);
  });

  it("injects only exact-scope authoritative prior identities in cockpit mode", async () => {
    const run = vi.fn(async () => "[]");
    const queryL1Paginated = vi.fn(async () => ({
      rows: [{
        record_id: "route-v1",
        content: "用户计划前往虹桥火车站。",
        type: "episodic",
        priority: 70,
        scene_name: "navigation",
        session_key: "old-key",
        session_id: "old-session",
        team_id: "team-a",
        task_id: "task-a",
        user_id: "user-a",
        agent_id: "agent-a",
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
          state_key: "navigation|user|car|driver|destination",
          episode_key: "route-1",
          construction_quality: { status: "complete" },
        }),
      }],
      total: 1,
    }));

    const result = await extractL1Memories({
      messages: [{
        id: "u2",
        role: "user",
        content: "把刚才的目的地改成上海南站",
        timestamp: Date.UTC(2026, 7, 30, 9, 20),
      }],
      sessionKey: "current-key",
      sessionId: "current-session",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/tmp/tdai-l1-extractor-test",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      },
    });

    expect(result.success).toBe(true);
    expect(queryL1Paginated).toHaveBeenCalledTimes(1);
    expect(run).toHaveBeenCalledWith(expect.objectContaining({
      prompt: expect.stringContaining('"record_id": "route-v1"'),
      thinkingMode: "disabled",
    }));
  });

  it("bounds shared batch prior context to the earliest source event", async () => {
    const run = vi.fn(async () => "[]");
    const queryL1Paginated = vi.fn(async () => ({
      rows: [{
        record_id: "between-batch-events",
        content: "这是较早批次尚未发生的状态。",
        type: "episodic",
        priority: 70,
        scene_name: "navigation",
        session_key: "old-key",
        session_id: "old-session",
        team_id: "team-a",
        task_id: "task-a",
        user_id: "user-a",
        agent_id: "agent-a",
        version: 1,
        timestamp_str: "2026-08-30T10:00:00.000Z",
        timestamp_start: "2026-08-30T10:00:00.000Z",
        timestamp_end: "2026-08-30T10:00:00.000Z",
        created_time: "2026-08-30T10:00:00.000Z",
        updated_time: "2026-08-30T10:00:00.000Z",
        metadata_json: JSON.stringify({
          schema_version: "cockpit-state-v1",
          domain: "navigation",
          slot: "destination",
          value: "未来地点",
          state_key: "navigation|user|car|driver|destination",
          episode_key: "future-route",
          mentioned_at: "2026-08-30T10:00:00.000Z",
          construction_quality: { status: "complete" },
        }),
      }],
      total: 1,
    }));

    const result = await extractL1Memories({
      messages: [
        {
          id: "u-early",
          role: "user",
          content: "先记一下我偏好安静的环境。",
          timestamp: Date.UTC(2026, 7, 30, 9),
        },
        {
          id: "u-late",
          role: "user",
          content: "之后我们再继续聊其他安排。",
          timestamp: Date.UTC(2026, 7, 30, 11),
        },
      ],
      sessionKey: "current-key",
      sessionId: "current-session",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/tmp/tdai-l1-extractor-test",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      },
    });

    expect(result.success).toBe(true);
    expect(queryL1Paginated).toHaveBeenCalledTimes(1);
    for (const [rawCall] of run.mock.calls as Array<[{ prompt: string }]>) {
      const call = rawCall;
      expect(call.prompt).not.toContain('"record_id": "between-batch-events"');
    }
  });

  it("retries an exact provider content-risk once with only optional prior context omitted", async () => {
    const run = vi.fn()
      .mockRejectedValueOnce(new Error("Content Exists Risk"))
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce("[]");
    const queryL1Paginated = vi.fn(async () => ({
      rows: [{
        record_id: "benign-prior-v1",
        content: "用户之前导航到虹桥火车站。",
        type: "episodic",
        priority: 70,
        scene_name: "navigation",
        session_key: "old-key",
        session_id: "old-session",
        team_id: "team-a",
        task_id: "task-a",
        user_id: "user-a",
        agent_id: "agent-a",
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
          state_key: "navigation|user|car|driver|destination",
          episode_key: "route-1",
          mentioned_at: "2026-08-30T09:00:00.000Z",
          construction_quality: { status: "complete" },
        }),
      }],
      total: 1,
    }));

    const result = await extractL1Memories({
      messages: [{
        id: "u2",
        role: "user",
        content: "平时说临时目的地就导航到上海南站。",
        timestamp: Date.UTC(2026, 7, 30, 10),
      }],
      sessionKey: "current-key",
      sessionId: "current-session",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/tmp/tdai-l1-extractor-test",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      },
    });

    expect(result.success).toBe(true);
    expect(run).toHaveBeenCalledTimes(3);
    expect(run).toHaveBeenNthCalledWith(1, expect.objectContaining({
      taskId: "l1-extraction",
      prompt: expect.stringContaining('"record_id": "benign-prior-v1"'),
    }));
    expect(run).toHaveBeenNthCalledWith(2, expect.objectContaining({
      taskId: "l1-extraction-context-minimized",
      prompt: expect.not.stringContaining('"record_id": "benign-prior-v1"'),
    }));
  });

  it("applies the same bounded prior omission to atomic compilation and reconciliation", async () => {
    const proposal = {
      content: "用户要求导航到上海南站。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u2"],
      metadata: {
        domain: "navigation",
        slot: "destination",
        value: "上海南站",
        subject: "user",
        relation: "asserted",
        state_key: "navigation|user|unspecified-vehicle|unspecified-zone|destination",
        episode_key: "route-2",
        action_status: "requested",
      } as Record<string, unknown>,
    };
    const primary = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u2"],
      memories: [proposal],
    }]);
    const reconciledProposal = structuredClone(proposal);
    reconciledProposal.metadata.input_candidate_ids = ["primary:0", "atomic:0"];
    const reconciled = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u2"],
      memories: [reconciledProposal],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(primary)
      .mockRejectedValueOnce(new Error("400 Content Exists Risk"))
      .mockResolvedValueOnce(primary)
      .mockRejectedValueOnce(new Error("Content Exists Risk"))
      .mockResolvedValueOnce(reconciled);
    const queryL1Paginated = vi.fn(async () => ({
      rows: [{
        record_id: "climate-prior-v1",
        content: "用户此前要求空调温度为二十四度。",
        type: "persona",
        priority: 70,
        scene_name: "climate",
        session_key: "old-key",
        session_id: "old-session",
        team_id: "team-a",
        task_id: "task-a",
        user_id: "user-a",
        agent_id: "agent-a",
        version: 1,
        timestamp_str: "2026-08-30T09:00:00.000Z",
        timestamp_start: "2026-08-30T09:00:00.000Z",
        timestamp_end: "2026-08-30T09:00:00.000Z",
        created_time: "2026-08-30T09:00:00.000Z",
        updated_time: "2026-08-30T09:00:00.000Z",
        metadata_json: JSON.stringify({
          schema_version: "cockpit-state-v1",
          domain: "climate",
          slot: "temperature",
          value: "24°C",
          state_key: "climate|user|car|driver|temperature",
          episode_key: "climate-1",
          mentioned_at: "2026-08-30T09:00:00.000Z",
          construction_quality: { status: "complete" },
        }),
      }],
      total: 1,
    }));

    const result = await extractL1Memories({
      messages: [{ id: "u2", role: "user", content: "导航到上海南站。", timestamp: Date.UTC(2026, 7, 30, 10) }],
      sessionKey: "current-key",
      sessionId: "current-session",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
        vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-atomic-compiler-context-minimized",
      prompt: expect.not.stringContaining('"record_id": "climate-prior-v1"'),
    }));
    expect(run).toHaveBeenNthCalledWith(5, expect.objectContaining({
      taskId: "l1-cockpit-construction-reconcile-context-minimized",
      prompt: expect.not.stringContaining('"record_id": "climate-prior-v1"'),
    }));
    expect(result.records).toHaveLength(1);
    expect(result.constructionQuality).toMatchObject({ complete: 1, partial: 0, invalid: 0 });
  });

  it("uses the independent compiler when both primary prompt variants hit provider content-risk", async () => {
    const atomicProposal = {
      content: "蒋澄要求临时目的地导航到北京人卫酒店。",
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "navigation",
        slot: "destination",
        value: "北京人卫酒店",
        subject: "蒋澄",
        relation: "asserted",
        state_key: "navigation|蒋澄|unspecified-vehicle|unspecified-zone|destination",
        episode_key: "temporary-destination-rule",
        action_status: "requested",
      } as Record<string, unknown>,
    };
    const atomic = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [atomicProposal],
    }]);
    const reconciledProposal = structuredClone(atomicProposal);
    reconciledProposal.metadata.input_candidate_ids = ["atomic:0"];
    const reconciled = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [reconciledProposal],
    }]);
    const run = vi.fn()
      .mockRejectedValueOnce(new Error("Content Exists Risk"))
      .mockRejectedValueOnce(new Error("Content Exists Risk"))
      .mockResolvedValueOnce(atomic)
      .mockResolvedValueOnce(reconciled);

    const result = await extractL1Memories({
      messages: [
        {
          id: "u1",
          role: "user",
          content: "[cockpit-p10-s08:001] [source_time=2026-03-10T10:00:00+08:00] [source_role=user] 【蒋澄】平时说临时目的的就导航到北京人卫酒店。",
          timestamp: Date.UTC(2026, 2, 10, 2),
        },
        {
          id: "a1",
          role: "assistant",
          content: "[cockpit-p10-s08:002] [source_time=2026-03-10T10:00:00+08:00] [source_role=assistant] 基础目的地口令已记录。",
          timestamp: Date.UTC(2026, 2, 10, 2),
        },
      ],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(4);
    expect(run).toHaveBeenNthCalledWith(2, expect.objectContaining({
      taskId: "l1-extraction-context-minimized",
      prompt: expect.stringContaining("【蒋澄】平时说临时目的的就导航到北京人卫酒店。"),
    }));
    const minimizedPrimaryPrompt = (run.mock.calls[1]?.[0] as { prompt: string }).prompt;
    expect(minimizedPrimaryPrompt).not.toContain("source_time=");
    expect(minimizedPrimaryPrompt).not.toContain("基础目的地口令已记录");
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-atomic-compiler",
      prompt: expect.stringContaining("北京人卫酒店"),
    }));
    expect(result.records).toHaveLength(1);
    expect(result.constructionQuality).toMatchObject({ complete: 1, partial: 0, invalid: 0 });
    expect(result.records[0].metadata).toMatchObject({
      construction_primary_status: "content_risk_unavailable",
      construction_compiler_status: "passed",
      construction_reconciliation_status: "passed",
    });
  });

  it("fails closed when primary and independent compiler are both content-risk rejected", async () => {
    const run = vi.fn().mockRejectedValue(new Error("Content Exists Risk"));
    const result = await extractL1Memories({
      messages: [
        {
          id: "u1",
          role: "user",
          content: "[cockpit-p10-s08:001] [source_time=2026-03-10T10:00:00+08:00] [source_role=user] 【蒋澄】导航到北京人卫酒店。",
          timestamp: Date.UTC(2026, 2, 10, 2),
        },
        { id: "a1", role: "assistant", content: "已记录。", timestamp: Date.UTC(2026, 2, 10, 2) },
      ],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
      },
    });

    expect(result.success).toBe(false);
    expect(result.storedCount).toBe(0);
    expect(run).toHaveBeenCalledTimes(3);
  });

  it("does not retry unrelated provider failures", async () => {
    const run = vi.fn().mockRejectedValue(new Error("401 invalid api key"));
    const result = await extractL1Memories({
      ...params({ run }, 600_000),
      options: {
        ...params({ run }, 600_000).options,
        promptMode: "cockpit",
      },
    });

    expect(result.success).toBe(false);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("uses an independent source-only Flash pass to add a missing complete atomic slot", async () => {
    const primary = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [{
        content: "用户计划七点出发。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "departure_time",
          value: "07:00",
          subject: "user",
          state_key: "navigation|user|unspecified-vehicle|unspecified-zone|departure_time",
          episode_key: "route-1",
          relation: "asserted",
          action_status: "requested",
        },
      }],
    }]);
    const atomic = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [
        JSON.parse(primary)[0].memories[0],
        {
          content: "用户请求在出发前十分钟提醒。",
          type: "episodic",
          priority: 70,
          source_message_ids: ["u1"],
          metadata: {
            domain: "reminder",
            slot: "reminder_time",
            value: "出发前10分钟",
            subject: "user",
            state_key: "reminder|user|unspecified-vehicle|unspecified-zone|reminder_time",
            episode_key: "route-1",
            relation: "asserted",
            action_status: "requested",
          },
        },
      ],
    }]);
    const reconciledPayload = JSON.parse(atomic) as Array<{
      memories: Array<{ metadata: Record<string, unknown> }>;
    }>;
    reconciledPayload[0].memories[0].metadata.input_candidate_ids = ["primary:0", "atomic:0"];
    reconciledPayload[0].memories[1].metadata.input_candidate_ids = ["atomic:1"];
    const reconciled = JSON.stringify(reconciledPayload);
    const run = vi.fn()
      .mockResolvedValueOnce(primary)
      .mockResolvedValueOnce(atomic)
      .mockResolvedValueOnce(reconciled);
    const appendFile = vi.fn(async () => undefined);
    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发，提前十分钟提醒", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(3);
    expect(run).toHaveBeenNthCalledWith(2, expect.objectContaining({
      taskId: "l1-cockpit-atomic-compiler",
      thinkingMode: "disabled",
      retryOnLength: true,
    }));
    const compilerPrompt = (run.mock.calls[1]?.[0] as { prompt: string }).prompt;
    expect(compilerPrompt).not.toContain("用户计划七点出发。");
    expect(compilerPrompt).not.toContain("construction_quality");
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-construction-reconcile",
      thinkingMode: "disabled",
      retryOnLength: true,
    }));
    expect(result.records).toHaveLength(2);
    expect(result.constructionQuality).toMatchObject({ complete: 2, partial: 0, invalid: 0 });
    expect(result.records.map((record) => (record.metadata as Record<string, unknown>).construction_compiler_status))
      .toEqual(["passed", "passed"]);
    expect(result.records.map((record) => (record.metadata as Record<string, unknown>).construction_reconciliation_status))
      .toEqual(["passed", "passed"]);
  });

  it("restarts the whole independent construction transaction once after malformed structured output", async () => {
    const fixture = cockpitStructuredRetryFixture();
    const run = vi.fn()
      .mockResolvedValueOnce(fixture.primary)
      .mockResolvedValueOnce(fixture.atomic)
      .mockResolvedValueOnce(fixture.malformed)
      .mockResolvedValueOnce(fixture.atomic)
      .mockResolvedValueOnce(fixture.reconciled);
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-atomic-compiler-transaction-retry",
    }));
    expect(run).toHaveBeenNthCalledWith(5, expect.objectContaining({
      taskId: "l1-cockpit-construction-reconcile-transaction-retry",
    }));
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0].metadata.construction_transaction_attempt).toBe(2);
    expect(appendFile).toHaveBeenCalledTimes(1);
  });

  it("absorbs two rejected reconciliation gates inside one bounded construction transaction", async () => {
    const fixture = cockpitStructuredRetryFixture();
    type FixtureMemory = {
      content: string;
      type: string;
      priority: number;
      source_message_ids: string[];
      metadata: Record<string, unknown>;
    };
    type FixtureScene = {
      scene_name: string;
      message_ids: string[];
      memories: FixtureMemory[];
    };
    const primaryScenes = JSON.parse(fixture.primary) as FixtureScene[];
    const acceptedAtomicScenes = JSON.parse(fixture.atomic) as FixtureScene[];
    const firstAtomic = structuredClone(acceptedAtomicScenes[0].memories[0]);
    firstAtomic.metadata.relation = "updated";
    firstAtomic.metadata.supersedes = ["prior-1"];
    const secondAtomic = structuredClone(firstAtomic);
    secondAtomic.metadata.supersedes = ["prior-2"];
    const conflictingAtomicScenes = structuredClone(acceptedAtomicScenes);
    conflictingAtomicScenes[0].memories = [firstAtomic, secondAtomic];
    const firstReconciled = structuredClone(firstAtomic);
    firstReconciled.metadata.input_candidate_ids = ["atomic:0"];
    const secondReconciled = structuredClone(secondAtomic);
    secondReconciled.metadata.input_candidate_ids = ["atomic:1"];
    const acceptedAtomic = structuredClone(acceptedAtomicScenes[0].memories[0]);
    acceptedAtomic.metadata.input_candidate_ids = ["atomic:0"];
    const reconciliation = (memories: FixtureMemory[]) => JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories,
    }]);
    const conflicting = reconciliation([firstReconciled, secondReconciled]);
    const accepted = reconciliation([acceptedAtomic]);
    const primaryOutput = JSON.stringify(primaryScenes);
    const conflictingAtomicOutput = JSON.stringify(conflictingAtomicScenes);
    const acceptedAtomicOutput = JSON.stringify(acceptedAtomicScenes);
    const run = vi.fn()
      .mockResolvedValueOnce(primaryOutput)
      .mockResolvedValueOnce(conflictingAtomicOutput)
      .mockResolvedValueOnce(conflicting)
      .mockResolvedValueOnce(conflicting)
      .mockResolvedValueOnce(conflictingAtomicOutput)
      .mockResolvedValueOnce(conflicting)
      .mockResolvedValueOnce(conflicting)
      .mockResolvedValueOnce(acceptedAtomicOutput)
      .mockResolvedValueOnce(accepted);
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(9);
    expect(run).toHaveBeenNthCalledWith(5, expect.objectContaining({
      taskId: "l1-cockpit-atomic-compiler-transaction-retry",
    }));
    expect(run).toHaveBeenNthCalledWith(8, expect.objectContaining({
      taskId: "l1-cockpit-atomic-compiler-transaction-retry",
    }));
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0].metadata.construction_transaction_attempt).toBe(3);
    expect(appendFile).toHaveBeenCalledTimes(1);
  });

  it("fails closed after exhausting the bounded reconciliation-gate retry budget", async () => {
    const fixture = cockpitStructuredRetryFixture();
    const atomicScenes = JSON.parse(fixture.atomic) as Array<{
      scene_name: string;
      message_ids: string[];
      memories: Array<{
        content: string;
        type: string;
        priority: number;
        source_message_ids: string[];
        metadata: Record<string, unknown>;
      }>;
    }>;
    atomicScenes[0].memories[0].metadata.relation = "updated";
    atomicScenes[0].memories[0].metadata.supersedes = ["non-live-prior"];
    const rejectedScenes = structuredClone(atomicScenes);
    rejectedScenes[0].memories[0].metadata.input_candidate_ids = ["atomic:0"];
    const atomicOutput = JSON.stringify(atomicScenes);
    const rejectedOutput = JSON.stringify(rejectedScenes);
    const run = vi.fn().mockResolvedValueOnce(fixture.primary);
    for (let attempt = 0; attempt < 3; attempt += 1) {
      run.mockResolvedValueOnce(atomicOutput)
        .mockResolvedValueOnce(rejectedOutput)
        .mockResolvedValueOnce(rejectedOutput);
    }
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(10);
    expect(result).toMatchObject({ success: false, extractedCount: 0, storedCount: 0, records: [] });
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("bounds malformed structured-output recovery to one transaction restart and writes nothing on exhaustion", async () => {
    const fixture = cockpitStructuredRetryFixture();
    const run = vi.fn()
      .mockResolvedValueOnce(fixture.primary)
      .mockResolvedValueOnce(fixture.atomic)
      .mockResolvedValueOnce(fixture.malformed)
      .mockResolvedValueOnce(fixture.atomic)
      .mockResolvedValueOnce(fixture.malformed);
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result).toMatchObject({ success: false, extractedCount: 0, storedCount: 0, records: [] });
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("uses deterministic atomic assembly before spending a bounded model repair", async () => {
    const proposal = {
      content: "用户计划七点出发。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "navigation",
        slot: "departure_time",
        value: "07:00",
        subject: "user",
        state_key: "navigation|user|unspecified-vehicle|unspecified-zone|departure_time",
        episode_key: "route-1",
        relation: "asserted",
        action_status: "requested",
      },
    };
    const scene = (memory: typeof proposal) => JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [memory],
    }]);
    const rejected = structuredClone(proposal);
    rejected.metadata.relation = "negated";
    (rejected.metadata as Record<string, unknown>).input_candidate_ids = ["primary:0", "atomic:0"];
    const run = vi.fn()
      .mockResolvedValueOnce(scene(proposal))
      .mockResolvedValueOnce(scene(proposal))
      .mockResolvedValueOnce(scene(rejected));
    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(3);
    expect(result.records).toHaveLength(1);
    expect(result.constructionQuality).toMatchObject({ complete: 1, partial: 0, invalid: 0 });
    expect((result.records[0].metadata.construction_quality as Record<string, unknown>).repairs)
      .toContain("restored_complete_atomic_obligation");
  });

  it("repairs instead of silently dropping a partial atomic obligation", async () => {
    const atomicProposal = {
      content: "用户计划七点出发。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "navigation",
        slot: "departure_time",
        value: "07:00",
        episode_key: "route-1",
        relation: "asserted",
        action_status: "requested",
      } as Record<string, unknown>,
    };
    const repairedProposal = structuredClone(atomicProposal);
    repairedProposal.metadata.subject = "user";
    repairedProposal.metadata.input_candidate_ids = ["atomic:0"];
    const scene = (memory: typeof atomicProposal) => JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [memory],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(scene(atomicProposal))
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(scene(repairedProposal));

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(4);
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-construction-reconcile",
      prompt: expect.stringContaining("reconciliation_uncovered_atomic_candidate"),
    }));
    expect(result.records).toHaveLength(1);
    expect(result.constructionQuality).toMatchObject({ complete: 1, partial: 0, invalid: 0 });
    expect(result.records[0].metadata).toMatchObject({
      construction_reconciliation_status: "passed",
      subject: "user",
    });
  });

  it("repairs a slot omitted by both model passes when the Chinese source coverage ledger requires it", async () => {
    const sourceContent = "【冯遥】请推荐一个评分4.5分以上且门票免费的景点。";
    const coverageIds = {
      rating: "coverage:u1:selection:rating_constraint",
      price: "coverage:u1:selection:price_constraint:ticket",
      category: "coverage:u1:selection:category_constraint",
    };
    const directedFact = (
      slot: string,
      value: string,
      content: string,
      extraMetadata: Record<string, unknown> = {},
    ) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [{
        content,
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot,
          value,
          subject: "冯遥",
          relation: "asserted",
          episode_key: "poi-search-1",
          action_status: "requested",
          ...extraMetadata,
          coverage_evidence_spans: coverageEvidenceSpans(
            sourceContent,
            slot,
            typeof extraMetadata.constraint_target === "string"
              ? extraMetadata.constraint_target
              : undefined,
          ),
        },
      }],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(directedFact(
        "rating_constraint",
        "评分4.5分以上",
        "冯遥要求景点评分在4.5分以上。",
      ))
      .mockResolvedValueOnce(directedFact(
        "price_constraint",
        "门票免费",
        "冯遥要求景点门票免费。",
        { constraint_target: "ticket" },
      ))
      .mockResolvedValueOnce(directedFact(
        "category_constraint",
        "景点",
        "冯遥要选择景点。",
      ))
      // The deterministic assembler restores each independently compiled
      // complete atomic candidate and binds its exact coverage obligation.
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{
        id: "u1",
        role: "user",
        content: sourceContent,
        timestamp: 1,
      }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(6);
    expect(run).toHaveBeenNthCalledWith(2, expect.objectContaining({
      prompt: expect.stringContaining(coverageIds.price),
    }));
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining(coverageIds.rating),
    }));
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining(coverageIds.price),
    }));
    expect(run).toHaveBeenNthCalledWith(5, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining(coverageIds.category),
    }));
    expect(run).toHaveBeenNthCalledWith(6, expect.objectContaining({
      taskId: "l1-cockpit-construction-reconcile",
    }));
    expect(result.records).toHaveLength(3);
    expect(result.constructionQuality).toMatchObject({ complete: 3, partial: 0, invalid: 0 });
    expect(result.records.map((record) => record.metadata.slot).sort())
      .toEqual(["category_constraint", "price_constraint", "rating_constraint"]);
    expect(result.records.find((record) => record.metadata.slot === "price_constraint")?.metadata.state_key)
      .toBe("selection|冯遥|unspecified-vehicle|unspecified-zone|price_constraint@ticket");
  });

  it("accepts server-authored evidence group ids without asking the model to recalculate NFKC offsets", async () => {
    const sourceContent = "景点评分４．５分以上。";
    const evidenceGroupIds = coverageEvidenceGroupIds(sourceContent, "rating_constraint");
    const directedFact = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [{
        content: "用户要求景点评分4.5分以上。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "user",
          relation: "asserted",
          episode_key: "poi-search-1",
          action_status: "requested",
          coverage_evidence_group_ids: evidenceGroupIds,
        },
      }],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("coverage-fact") ? directedFact : "[]"
    );

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(4);
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0].metadata.coverage_evidence_group_ids).toBeUndefined();
    expect(result.records[0].metadata.coverage_evidence_spans).toBeUndefined();
  });

  it("replaces duplicate model paths with one verified fact per named destination state", async () => {
    const sourceContent = "【程野】路线口令：早餐地点是松林文化馆，返程地点是云岭客运站，会客地点是滨河创意园。";
    const evidenceGroupIds = coverageEvidenceGroupIds(sourceContent, "destination");
    const facts = [
      ["早餐地点", "松林文化馆"],
      ["返程地点", "云岭客运站"],
      ["会客地点", "滨河创意园"],
    ] as const;
    const scene = (
      type: "persona" | "episodic" | "instruction",
      directed: boolean,
    ) => JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: facts.map(([stateQualifier, value], index) => ({
        content: `程野的${stateQualifier}是${value}。`,
        type,
        priority: 75,
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value,
          subject: "程野",
          state_qualifier: stateQualifier,
          relation: "asserted",
          episode_key: `${type}-episode-${index}`,
          ...(type === "episodic" ? { action_status: "requested" } : {}),
          ...(directed
            ? { coverage_evidence_group_ids: [evidenceGroupIds[index]] }
            : {}),
        },
      })),
    }]);
    const run = vi.fn(async (request: { taskId?: string }) => {
      if (request.taskId === "l1-extraction") return scene("persona", false);
      if (request.taskId?.includes("atomic-compiler")) return scene("episodic", false);
      if (request.taskId?.includes("coverage-fact")) return scene("instruction", true);
      if (request.taskId?.includes("construction-reconcile")) return "[]";
      throw new Error(`unexpected task ${request.taskId}`);
    });

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining('"requiredStateQualifierCount": 3'),
    }));
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-verifier",
    }));
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(3);
    expect(result.records.every((record) => record.type === "instruction")).toBe(true);
    expect(result.records.map((record) => record.metadata.state_qualifier).sort())
      .toEqual(facts.map(([qualifier]) => qualifier).sort());
    expect(new Set(result.records.map((record) => record.metadata.state_key)).size).toBe(3);
    expect(new Set(result.records.map((record) => record.metadata.episode_key)).size).toBe(3);
    expect(result.constructionQuality).toMatchObject({ complete: 3, partial: 0, invalid: 0 });
  });

  it.each(["返程地点", "是", "改成"])(
    "fails closed when qualifier %s is not the exact label bound to its source group",
    async (invalidQualifier) => {
    const sourceContent = "路线口令：早餐地点是松林文化馆，返程地点是云岭客运站。";
    const evidenceGroupIds = coverageEvidenceGroupIds(sourceContent, "destination");
    const crossedBinding = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [{
        content: "用户的返程地点是云岭客运站。",
        type: "instruction",
        priority: 75,
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "云岭客运站",
          subject: "user",
          state_qualifier: invalidQualifier,
          relation: "asserted",
          coverage_evidence_group_ids: [evidenceGroupIds[0]],
        },
      }],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("coverage-fact") ? crossedBinding : "[]"
    );
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result).toMatchObject({ success: false, extractedCount: 0, storedCount: 0, records: [] });
    expect(appendFile).not.toHaveBeenCalled();
    },
  );

  it("binds a missing bookkeeping field only when one fact has one unambiguous evidence group", async () => {
    const sourceContent = "景点评分4.6分以上。";
    const directedFact = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [{
        content: "用户要求景点评分4.6分以上。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.6分以上",
          subject: "user",
          relation: "asserted",
          episode_key: "poi-search-1",
          action_status: "requested",
        },
      }],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("coverage-fact") ? directedFact : "[]"
    );

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0].metadata.construction_quality).toMatchObject({
      status: "complete",
      repairs: expect.arrayContaining(["deterministically_bound_sole_coverage_group"]),
    });
  });

  it("fails closed when a directed fact names an unknown evidence group", async () => {
    const sourceContent = "景点评分4.7分以上。";
    const directedFact = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [{
        content: "用户要求景点评分4.7分以上。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.7分以上",
          subject: "user",
          relation: "asserted",
          episode_key: "poi-search-1",
          action_status: "requested",
          coverage_evidence_group_ids: ["coverage:u1:selection:rating_constraint:evidence:unknown"],
        },
      }],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("coverage-fact") ? directedFact : "[]"
    );
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it.each(["persona", "work_fact", "instruction|episodic"])(
    "fails closed when a directed coverage fact uses the unsupported type %s",
    async (directedType) => {
      const sourceContent = "景点评分4.7分以上。";
      const directedFact = JSON.stringify([{
        scene_name: "selection",
        message_ids: ["u1"],
        memories: [{
          content: "用户要求景点评分4.7分以上。",
          type: directedType,
          priority: 70,
          source_message_ids: ["u1"],
          metadata: {
            domain: "selection",
            slot: "rating_constraint",
            value: "4.7分以上",
            subject: "user",
            relation: "asserted",
            episode_key: "poi-search-1",
            action_status: "requested",
            coverage_evidence_group_ids: coverageEvidenceGroupIds(
              sourceContent,
              "rating_constraint",
            ),
          },
        }],
      }]);
      const run = vi.fn(async (request: { taskId?: string }) =>
        request.taskId?.includes("coverage-fact") ? directedFact : "[]"
      );
      const appendFile = vi.fn(async () => undefined);

      const result = await extractL1Memories({
        messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
        sessionKey: "session-key",
        sessionId: "session-1",
        teamId: "team-a",
        taskId: "task-a",
        userId: "user-a",
        agentId: "agent-a",
        baseDir: "/unused",
        config: {},
        options: {
          enableDedup: false,
          llmRunner: { run },
          llmTimeoutMs: 600_000,
          promptMode: "cockpit",
          constructionModel: "deepseek-v4-flash",
        },
        storage: { appendFile } as unknown as StorageAdapter,
      });

      expect(run).toHaveBeenCalledTimes(5);
      expect(result.success).toBe(false);
      expect(result.records).toEqual([]);
      expect(appendFile).not.toHaveBeenCalled();
    },
  );

  it("fails closed when evidence group ids conflict with legacy evidence spans", async () => {
    const sourceContent = "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。";
    const evidenceGroupIds = coverageEvidenceGroupIds(sourceContent, "rating_constraint");
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const directedFact = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [{
        content: "冯遥要求景点评分4.5分以上。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "冯遥",
          relation: "asserted",
          action_status: "requested",
          coverage_evidence_group_ids: [evidenceGroupIds[0]],
          coverage_evidence_spans: [evidenceSpans[1]],
        },
      }],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("coverage-fact") ? directedFact : "[]"
    );
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("does not infer missing evidence bindings for a multi-fact multi-group obligation", async () => {
    const sourceContent = "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。";
    const ratingFact = (subject: string, value: string) => ({
      content: `${subject}要求景点评分${value}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value,
        subject,
        relation: "asserted",
        action_status: "requested",
      },
    });
    const directedFacts = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [
        ratingFact("冯遥", "4.5分以上"),
        ratingFact("林静", "4.0分以上"),
      ],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("coverage-fact") ? directedFacts : "[]"
    );
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("keeps a complete authoritative general candidate without invoking directed replacement", async () => {
    const sourceContent = "景点评分4.8分以上。";
    const fact = {
      content: "用户要求景点评分4.8分以上。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.8分以上",
        subject: "user",
        relation: "asserted",
        action_status: "requested",
      },
    };
    const general = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [fact],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) => {
      if (request.taskId?.includes("atomic-compiler")) return general;
      if (request.taskId?.includes("coverage-fact")) {
        throw new Error("authoritative general candidate must suppress directed replacement");
      }
      return "[]";
    });

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run.mock.calls.filter(([request]) =>
      request.taskId?.includes("coverage-fact")
    )).toHaveLength(0);
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0]).toMatchObject({
      scene_name: "selection",
      metadata: { domain: "selection", slot: "rating_constraint", value: "4.8分以上" },
    });
  });

  it("uses structured coverage metadata despite a legacy free-text scene label", async () => {
    const sourceContent = "景点评分4.8分以上。";
    const fact = {
      content: "用户要求景点评分4.8分以上。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.8分以上",
        subject: "user",
        relation: "asserted",
        episode_key: "poi-search-1",
        action_status: "requested",
      },
    };
    const general = JSON.stringify([{
      scene_name: "misc",
      message_ids: ["u1"],
      memories: [fact],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) => {
      if (request.taskId?.includes("atomic-compiler")) return general;
      if (request.taskId?.includes("coverage-fact")) {
        throw new Error("structured general coverage must suppress directed replacement");
      }
      return "[]";
    });

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(3);
    expect(run.mock.calls.filter(([request]) =>
      request.taskId?.includes("coverage-fact")
    )).toHaveLength(0);
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0]).toMatchObject({
      scene_name: "selection",
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.8分以上",
        construction_quality: {
          repairs: expect.arrayContaining(["canonicalized_scene_name_to_controlled_domain"]),
        },
      },
    });
  });

  it("directs same-slot coverage obligations through separate source events", async () => {
    const sourceContents: Record<string, string> = {
      u1: "门票预算100元以内。",
      u2: "门票必须免费。",
    };
    const directedFact = (
      sourceId: string,
      value: string,
      episodeKey: string,
    ) => JSON.stringify([{
      scene_name: "selection",
      message_ids: [sourceId],
      memories: [{
        content: `用户要求${value}。`,
        type: "episodic",
        priority: 70,
        source_message_ids: [sourceId],
        metadata: {
          domain: "selection",
          slot: "price_constraint",
          value,
          constraint_target: "ticket",
          subject: "user",
          relation: "asserted",
          episode_key: episodeKey,
          action_status: "requested",
          coverage_evidence_spans: coverageEvidenceSpans(
            sourceContents[sourceId],
            "price_constraint",
            "ticket",
            sourceId,
          ),
        },
      }],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(directedFact("u1", "门票100元以内", "price-u1"))
      .mockResolvedValueOnce(directedFact("u2", "门票免费", "price-u2"))
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [
        { id: "u1", role: "user", content: sourceContents.u1, timestamp: 1 },
        { id: "u2", role: "user", content: sourceContents.u2, timestamp: 2 },
      ],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining('"id": "u1"'),
    }));
    expect(run.mock.calls[2][0].prompt).not.toContain('"id": "u2"');
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining('"id": "u2"'),
    }));
    expect(run.mock.calls[3][0].prompt).not.toContain('"id": "u1"');
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(2);
    expect(result.records.map((record) => record.metadata.value).sort())
      .toEqual(["门票100元以内", "门票免费"].sort());
    expect(result.records.map((record) => record.source_message_ids))
      .toEqual(expect.arrayContaining([["u1"], ["u2"]]));
  });

  it("keeps identical facts from different exact source events", async () => {
    const sourceContents: Record<string, string> = {
      u1: "景点评分4.5分以上。",
      u2: "景点评分4.5分以上。",
    };
    const directedFact = (sourceId: string) => JSON.stringify([{
      scene_name: "selection",
      message_ids: [sourceId],
      memories: [{
        content: "用户要求景点评分4.5分以上。",
        type: "episodic",
        priority: 70,
        source_message_ids: [sourceId],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "user",
          relation: "asserted",
          action_status: "requested",
          coverage_evidence_spans: coverageEvidenceSpans(
            sourceContents[sourceId],
            "rating_constraint",
            undefined,
            sourceId,
          ),
        },
      }],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(directedFact("u1"))
      .mockResolvedValueOnce(directedFact("u2"))
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [
        { id: "u1", role: "user", content: sourceContents.u1, timestamp: 1 },
        { id: "u2", role: "user", content: sourceContents.u2, timestamp: 2 },
      ],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(2);
    expect(result.records.map((record) => record.metadata.value))
      .toEqual(["4.5分以上", "4.5分以上"]);
    expect(result.records.map((record) => record.source_message_ids))
      .toEqual(expect.arrayContaining([["u1"], ["u2"]]));
    expect(new Set(result.records.map((record) => record.metadata.episode_key)).size).toBe(2);
  });

  it("fails closed when an atomic fact merges two covered source events", async () => {
    const sourceContent = "景点评分4.5分以上。";
    const mixedSourceScene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1", "u2"],
      memories: [{
        content: "用户要求景点评分4.5分以上。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1", "u2"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "4.5分以上",
          subject: "user",
          relation: "asserted",
          action_status: "requested",
        },
      }],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(mixedSourceScene)
      .mockResolvedValueOnce(mixedSourceScene);
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [
        { id: "u1", role: "user", content: sourceContent, timestamp: 1 },
        { id: "u2", role: "user", content: sourceContent, timestamp: 2 },
      ],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(3);
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("bounds directed source-only calls across the whole transaction retry", async () => {
    const sourceContent = "冯遥评分4.5分以上、林静4.0分以上。";
    const sourceIds = Array.from({ length: 9 }, (_, index) => `u${index + 1}`);
    const factsBySource = new Map(sourceIds.map((sourceId) => {
      const evidenceSpans = coverageEvidenceSpans(
        sourceContent,
        "rating_constraint",
        undefined,
        sourceId,
      );
      const memories = [
        { subject: "冯遥", value: "4.5分以上", evidence: evidenceSpans[0] },
        { subject: "林静", value: "4.0分以上", evidence: evidenceSpans[1] },
      ].map(({ subject, value, evidence }) => ({
        content: `${subject}要求景点评分${value}。`,
        type: "episodic",
        priority: 70,
        source_message_ids: [sourceId],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value,
          subject,
          relation: "asserted",
          action_status: "requested",
          coverage_evidence_spans: [evidence],
        },
      }));
      return [sourceId, JSON.stringify([{
        scene_name: "selection",
        message_ids: [sourceId],
        memories,
      }])] as const;
    }));
    const run = vi.fn(async (request: { taskId?: string; prompt?: string }) => {
      if (request.taskId?.includes("coverage-fact")) {
        const sourceId = sourceIds.find((candidate) =>
          request.prompt?.includes(`\"id\": \"${candidate}\"`)
        );
        if (!sourceId) throw new Error("missing exact source in directed prompt");
        return factsBySource.get(sourceId) ?? "[]";
      }
      return "[]";
    });
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: sourceIds.map((id, index) => ({
        id,
        role: "user" as const,
        content: sourceContent,
        timestamp: index + 1,
      })),
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        maxMessagesPerExtraction: 10,
        maxMemoriesPerSession: 40,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
    expect(run.mock.calls.filter(([request]) =>
      request.taskId?.includes("coverage-fact")
    )).toHaveLength(16);
    expect(run).toHaveBeenCalledTimes(18);
  });

  it("audits the complete same-source same-slot fact set instead of accepting one matching row", async () => {
    const sourceContent = "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。";
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const ratingFact = (subject: string, value: string, evidenceIndex: number) => ({
      content: `${subject}要求景点评分${value}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value,
        subject,
        relation: "asserted",
        action_status: "requested",
        coverage_evidence_spans: [evidenceSpans[evidenceIndex]],
      },
    });
    const scene = (memories: ReturnType<typeof ratingFact>[]) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const feng = ratingFact("冯遥", "4.5分以上", 0);
    const lin = ratingFact("林静", "4.0分以上", 1);
    const run = vi.fn()
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{
        id: "u1",
        role: "user",
        content: sourceContent,
        timestamp: 1,
      }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining('"requiredFactCount": 2'),
    }));
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-verifier",
    }));
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(2);
    expect(result.records.map((record) => [record.metadata.subject, record.metadata.value]).sort())
      .toEqual([
        ["冯遥", "4.5分以上"],
        ["林静", "4.0分以上"],
      ].sort());
  });

  it("replaces conflicting primary and general subsets with the verified directed set", async () => {
    const sourceContent = "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。";
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const ratingFact = (subject: string, value: string, evidenceIndex: number) => ({
      content: `${subject}要求景点评分${value}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value,
        subject,
        relation: "asserted",
        action_status: "requested",
        coverage_evidence_spans: [evidenceSpans[evidenceIndex]],
      },
    });
    const scene = (memories: ReturnType<typeof ratingFact>[]) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const feng = ratingFact("冯遥", "4.5分以上", 0);
    const lin = ratingFact("林静", "4.0分以上", 1);
    const hallucinated = ratingFact("王强", "3.0分以上", 1);
    const run = vi.fn()
      .mockResolvedValueOnce(scene([feng, hallucinated]))
      .mockResolvedValueOnce(scene([feng, hallucinated]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(true);
    expect(result.records.map((record) => [record.metadata.subject, record.metadata.value]).sort())
      .toEqual([
        ["冯遥", "4.5分以上"],
        ["林静", "4.0分以上"],
      ].sort());
    expect(result.records.some((record) => record.metadata.subject === "王强")).toBe(false);
  });

  it("does not let two memory types for one subject satisfy shared-subject cardinality", async () => {
    const sourceContent = "冯遥和林静都要求景点评分4.5分以上。";
    const [evidence] = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const duplicate = (type: "episodic" | "instruction") => ({
      content: "冯遥要求景点评分4.5分以上。",
      type,
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "冯遥",
        relation: "asserted",
        ...(type === "episodic" ? { action_status: "requested" } : {}),
        coverage_evidence_spans: [evidence],
      },
    });
    const scene = (memories: unknown[]) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const run = vi.fn(async (request: { taskId?: string }) => {
      if (request.taskId === "l1-extraction" || request.taskId?.includes("atomic-compiler")) {
        return scene([duplicate("episodic")]);
      }
      if (request.taskId?.includes("coverage-fact")) {
        return scene([duplicate("episodic"), duplicate("instruction")]);
      }
      throw new Error(`unexpected task ${request.taskId}`);
    });
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("canonicalizes blind assertion episode drift instead of trusting either model label", async () => {
    const sourceContent = "默认要求景点评分4.5分以上。";
    const [evidence] = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const fact = (episodeKey: string) => ({
      content: "用户默认要求景点评分4.5分以上。",
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "user",
        condition: "默认",
        relation: "asserted",
        episode_key: episodeKey,
        coverage_evidence_spans: [evidence],
      },
    });
    const scene = (memory: ReturnType<typeof fact>) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [memory],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene(fact("primary-label")))
      .mockResolvedValueOnce(scene(fact("general-label")))
      .mockResolvedValueOnce(scene(fact("blind-pass-a")))
      .mockResolvedValueOnce(scene(fact("blind-pass-b")))
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0].metadata.episode_key).toMatch(/^coverage-event:[0-9a-f]{24}$/u);
    expect(["primary-label", "general-label", "blind-pass-a", "blind-pass-b"])
      .not.toContain(result.records[0].metadata.episode_key);
  });

  it("does not duplicate a complete cross-domain appointment under a free-text scene label", async () => {
    const sourceContent = "[source_time=2026-03-16T09:00:00+08:00] 【周宁】先约4月5日上午10点去北京市东城区故宫博物院内做车辆检查。";
    const episode = "episode-vehicle-check";
    const facts = [
      {
        content: "周宁预约在2026年4月5日上午10点做车辆检查。",
        type: "episodic",
        priority: 75,
        source_message_ids: ["u1"],
        metadata: {
          domain: "schedule",
          slot: "appointment_time",
          value: "2026-04-05T10:00:00+08:00",
          subject: "周宁",
          relation: "asserted",
          action_status: "requested",
          episode_key: episode,
        },
      },
      {
        content: "周宁预约做车辆检查。",
        type: "episodic",
        priority: 75,
        source_message_ids: ["u1"],
        metadata: {
          domain: "schedule",
          slot: "appointment_content",
          value: "车辆检查",
          subject: "周宁",
          relation: "asserted",
          action_status: "requested",
          episode_key: episode,
        },
      },
      {
        content: "周宁预约前往北京市东城区故宫博物院内。",
        type: "episodic",
        priority: 75,
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "destination",
          value: "北京市东城区故宫博物院内",
          subject: "周宁",
          relation: "asserted",
          action_status: "requested",
          episode_key: episode,
        },
      },
    ];
    const general = JSON.stringify([{
      scene_name: "session-vehicle-check",
      message_ids: ["u1"],
      memories: facts,
    }]);
    const run = vi.fn(async (request: { taskId?: string }) => {
      if (request.taskId?.includes("atomic-compiler")) return general;
      if (request.taskId?.includes("coverage-fact")) {
        throw new Error("complete structured facts must suppress redundant directed compilation");
      }
      return "[]";
    });

    const result = await extractL1Memories({
      messages: [{
        id: "u1",
        role: "user",
        content: sourceContent,
        timestamp: Date.parse("2026-03-16T09:00:00+08:00"),
      }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(3);
    expect(run.mock.calls.filter(([request]) =>
      request.taskId?.includes("coverage-fact")
    )).toHaveLength(0);
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(3);
    expect(result.records.map((record) => `${record.metadata.domain}.${record.metadata.slot}`).sort())
      .toEqual([
        "navigation.destination",
        "schedule.appointment_content",
        "schedule.appointment_time",
      ]);
    expect(result.records.find((record) => record.metadata.slot === "appointment_time")?.metadata)
      .toMatchObject({
        value: "2026-04-05T10:00:00+08:00",
        activity_start_time: "2026-04-05T10:00:00+08:00",
      });
  });

  it("keeps simultaneous appointment-time facts distinct by deterministic evidence event", async () => {
    const sourceContent = "4月1日10点做车辆检查，4月1日10点做体检。";
    const contentEvidence = coverageEvidenceSpans(sourceContent, "appointment_content");
    const timeEvidence = coverageEvidenceSpans(sourceContent, "appointment_time");
    const start = "2026-04-01T10:00:00+08:00";
    const contentFact = (value: string, evidenceIndex: number, episodeKey: string) => ({
      content: `用户安排${value}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_content",
        value,
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        episode_key: episodeKey,
        coverage_evidence_spans: [contentEvidence[evidenceIndex]],
      },
    });
    const timeFact = (evidenceIndex: number, episodeKey: string) => ({
      content: "用户在4月1日10点有安排。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_time",
        value: start,
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        episode_key: episodeKey,
        coverage_evidence_spans: [timeEvidence[evidenceIndex]],
      },
    });
    const contentFacts = [
      contentFact("车辆检查", 0, "vehicle-check"),
      contentFact("体检", 1, "medical-check"),
    ];
    const timeFacts = [timeFact(0, "vehicle-check"), timeFact(1, "medical-check")];
    const scene = (memories: unknown[]) => JSON.stringify([{
      scene_name: "schedule",
      message_ids: ["u1"],
      memories,
    }]);
    const run = vi.fn(async (request: { taskId?: string; prompt?: string }) => {
      if (request.taskId === "l1-extraction" || request.taskId?.includes("atomic-compiler")) {
        return scene([...contentFacts, ...timeFacts]);
      }
      if (request.taskId?.includes("coverage-fact")) {
        return request.prompt?.includes('"slot": "appointment_content"')
          ? scene(contentFacts)
          : scene(timeFacts);
      }
      if (request.taskId?.includes("construction-reconcile")) return "[]";
      throw new Error(`unexpected task ${request.taskId}`);
    });

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(true);
    const times = result.records.filter((record) => record.metadata.slot === "appointment_time");
    const contents = result.records.filter((record) => record.metadata.slot === "appointment_content");
    expect(times).toHaveLength(2);
    expect(new Set(times.map((record) => record.metadata.episode_key)).size).toBe(2);
    expect(new Set(times.map((record) => record.metadata.episode_key)))
      .toEqual(new Set(contents.map((record) => record.metadata.episode_key)));
  });

  it("binds destination, appointment content and appointment time to the same source event", async () => {
    const sourceContent = "4月1日10点做检查、4月2日11点去维修厂做保养。";
    const contentEvidence = coverageEvidenceSpans(sourceContent, "appointment_content");
    const timeEvidence = coverageEvidenceSpans(sourceContent, "appointment_time");
    const [destinationEvidence] = coverageEvidenceSpans(sourceContent, "destination");
    const firstStart = "2026-04-01T10:00:00+08:00";
    const secondStart = "2026-04-02T11:00:00+08:00";
    const contentFact = (value: string, start: string, evidenceIndex: number) => ({
      content: `用户在${start}安排${value}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_content",
        value,
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        episode_key: `model-content-${evidenceIndex}`,
        coverage_evidence_spans: [contentEvidence[evidenceIndex]],
      },
    });
    const timeFact = (start: string, evidenceIndex: number) => ({
      content: `用户的预约时间为${start}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_time",
        value: start,
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        episode_key: `model-time-${evidenceIndex}`,
        coverage_evidence_spans: [timeEvidence[evidenceIndex]],
      },
    });
    const contents = [
      contentFact("检查", firstStart, 0),
      contentFact("保养", secondStart, 1),
    ];
    const times = [timeFact(firstStart, 0), timeFact(secondStart, 1)];
    const destination = {
      content: "用户去维修厂做保养。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "navigation",
        slot: "destination",
        value: "维修厂",
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: secondStart,
        activity_start_time: secondStart,
        episode_key: "model-destination",
        coverage_evidence_spans: [destinationEvidence],
      },
    };
    const scene = (sceneName: string, memories: unknown[]) => JSON.stringify([{
      scene_name: sceneName,
      message_ids: ["u1"],
      memories,
    }]);
    const run = vi.fn(async (request: { taskId?: string; prompt?: string }) => {
      if (request.taskId === "l1-extraction" || request.taskId?.includes("atomic-compiler")) {
        return "[]";
      }
      if (request.taskId?.includes("coverage-fact")) {
        if (request.prompt?.includes('"slot": "appointment_content"')) {
          return scene("schedule", contents);
        }
        if (request.prompt?.includes('"slot": "appointment_time"')) {
          return scene("schedule", times);
        }
        if (request.prompt?.includes('"slot": "destination"')) {
          return scene("navigation", [destination]);
        }
      }
      if (request.taskId?.includes("construction-reconcile")) return "[]";
      throw new Error(`unexpected task ${request.taskId}`);
    });

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(5);
    const firstContent = result.records.find((record) =>
      record.metadata.slot === "appointment_content" && record.metadata.value === "检查"
    );
    const secondContent = result.records.find((record) =>
      record.metadata.slot === "appointment_content" && record.metadata.value === "保养"
    );
    const secondTime = result.records.find((record) =>
      record.metadata.slot === "appointment_time" && record.metadata.value === secondStart
    );
    const storedDestination = result.records.find((record) => record.metadata.slot === "destination");
    expect(storedDestination?.metadata.episode_key).toBe(secondContent?.metadata.episode_key);
    expect(storedDestination?.metadata.episode_key).toBe(secondTime?.metadata.episode_key);
    expect(storedDestination?.metadata.episode_key).not.toBe(firstContent?.metadata.episode_key);
  });

  it("keeps directed evidence coordinates unchanged during content-risk fallback", async () => {
    const sourceContent = "[cockpit-x:001] [source_time=2026-04-01T10:00:00+08:00] [source_role=user] 默认要求景点评分4.5分以上。";
    const [evidence] = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const fact = {
      content: "用户默认要求景点评分4.5分以上。",
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "user",
        condition: "默认",
        relation: "asserted",
        coverage_evidence_spans: [evidence],
      },
    };
    const scene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [fact],
    }]);
    const queryL1Paginated = vi.fn(async () => ({
      rows: [{
        record_id: "prior-temperature",
        content: "用户此前要求温度为24度。",
        type: "persona",
        priority: 70,
        scene_name: "climate",
        session_key: "old-key",
        session_id: "old-session",
        team_id: "team-a",
        task_id: "task-a",
        user_id: "user-a",
        agent_id: "agent-a",
        version: 1,
        timestamp_str: "2026-03-31T09:00:00.000Z",
        timestamp_start: "2026-03-31T09:00:00.000Z",
        timestamp_end: "2026-03-31T09:00:00.000Z",
        created_time: "2026-03-31T09:00:00.000Z",
        updated_time: "2026-03-31T09:00:00.000Z",
        metadata_json: JSON.stringify({
          schema_version: "cockpit-state-v1",
          domain: "climate",
          slot: "temperature",
          value: 24,
          subject: "user",
          state_key: "climate|user|unspecified-vehicle|unspecified-zone|temperature",
          episode_key: "temperature-prior",
          mentioned_at: "2026-03-31T09:00:00.000Z",
          construction_quality: { status: "complete" },
        }),
      }],
      total: 1,
    }));
    const run = vi.fn()
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce(scene)
      .mockRejectedValueOnce(new Error("Content Exists Risk"))
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce("[]");
    const errors: string[] = [];

    const result = await extractL1Memories({
      messages: [{
        id: "u1",
        role: "user",
        content: sourceContent,
        timestamp: Date.parse("2026-04-01T10:00:00+08:00"),
      }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
        vectorStore: { queryL1Paginated } as unknown as IMemoryStore,
      },
      logger: {
        debug: vi.fn(),
        info: vi.fn(),
        warn: vi.fn(),
        error: (message: string) => errors.push(message),
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(result.success, errors.join("\n")).toBe(true);
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      retryOnLength: false,
    }));
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler-context-minimized",
      prompt: expect.stringContaining("source_time=2026-04-01T10:00:00+08:00"),
      retryOnLength: false,
    }));
    expect((run.mock.calls[3][0] as { prompt: string }).prompt)
      .toContain(`\"start\": ${evidence.start}`);
  });

  it("fails closed when the two source-only fact-set passes disagree", async () => {
    const sourceContent = "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。";
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const ratingFact = (subject: string, value: string, evidenceIndex: number) => ({
      content: `${subject}要求景点评分${value}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value,
        subject,
        relation: "asserted",
        action_status: "requested",
        coverage_evidence_spans: [evidenceSpans[evidenceIndex]],
      },
    });
    const scene = (memories: ReturnType<typeof ratingFact>[]) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const feng = ratingFact("冯遥", "4.5分以上", 0);
    const lin = ratingFact("林静", "4.0分以上", 1);
    const disagreeingLin = ratingFact("林静", "4.1分以上", 1);
    const run = vi.fn()
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce(scene([feng, disagreeingLin]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce(scene([feng, disagreeingLin]));
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(7);
    expect(run).toHaveBeenNthCalledWith(7, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-verifier-transaction-retry",
    }));
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("does not let one undated fact copied across date evidence satisfy temporal cardinality", async () => {
    const sourceContent = "4月1日和4月2日都要求景点评分4.5分以上。";
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const copiedFact = (evidenceIndex: number) => ({
      content: "用户要求景点评分4.5分以上。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        episode_key: "copied-undated-fact",
        coverage_evidence_spans: [evidenceSpans[evidenceIndex] ?? evidenceSpans[0]],
      },
    });
    const scene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [copiedFact(0), copiedFact(1)],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("construction-reconcile") ? "[]" : scene
    );
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("does not let one condition copied across two condition axes satisfy binding cardinality", async () => {
    const sourceContent = "晴天和雨天都要求景点评分4.5分以上。";
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const copiedFact = (evidenceIndex: number) => ({
      content: "用户在晴天要求景点评分4.5分以上。",
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "user",
        condition: "晴天",
        relation: "asserted",
        coverage_evidence_spans: [evidenceSpans[evidenceIndex]],
      },
    });
    const scene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [copiedFact(0), copiedFact(1)],
    }]);
    const run = vi.fn(async () => scene);
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("does not let one person copied across separately enumerated people satisfy source cardinality", async () => {
    const sourceContent = "给冯遥设置评分4.5分以上，给林静设置评分4.5分以上。";
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const copiedFact = (evidenceIndex: number) => ({
      content: "冯遥要求景点评分4.5分以上。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "冯遥",
        relation: "asserted",
        action_status: "requested",
        coverage_evidence_spans: [evidenceSpans[evidenceIndex]],
      },
    });
    const scene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [copiedFact(0), copiedFact(1)],
    }]);
    const run = vi.fn(async () => scene);
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("keeps same-value same-slot facts distinct when their effective dates differ", async () => {
    const sourceContent = "2026年4月1日景点评分4.5分以上，2026年4月2日景点评分4.5分以上。";
    const evidenceSpans = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const datedFact = (validFrom: string, evidenceIndex: number) => ({
      content: `${validFrom.slice(0, 10)}景点评分要求为4.5分以上。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: validFrom,
        time_precision: "day",
        episode_key: `rating-${validFrom.slice(0, 10)}`,
        coverage_evidence_spans: [evidenceSpans[evidenceIndex]],
      },
    });
    const aprilFirst = datedFact("2026-04-01T00:00:00+08:00", 0);
    const aprilSecond = datedFact("2026-04-02T00:00:00+08:00", 1);
    const scene = (memories: ReturnType<typeof datedFact>[]) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene([aprilFirst]))
      .mockResolvedValueOnce(scene([aprilFirst]))
      .mockResolvedValueOnce(scene([aprilFirst, aprilSecond]))
      .mockResolvedValueOnce(scene([aprilFirst, aprilSecond]))
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(2);
    expect(result.records.map((record) => record.metadata.valid_from).sort()).toEqual([
      "2026-04-01T00:00:00+08:00",
      "2026-04-02T00:00:00+08:00",
    ]);
  });

  it("keeps two appointment intervals distinct when only their end times differ", async () => {
    const sourceContent = "2026年4月1日10点到11点做车辆检查，2026年4月1日10点到12点做体检。";
    const contentEvidence = coverageEvidenceSpans(sourceContent, "appointment_content");
    const timeEvidence = coverageEvidenceSpans(sourceContent, "appointment_time");
    const contentFact = (
      value: string,
      start: string,
      end: string,
      evidenceIndex: number,
      episodeKey: string,
    ) => ({
      content: `${start.slice(0, 16)}安排${value}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_content",
        value,
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        activity_end_time: end,
        time_precision: "minute",
        episode_key: episodeKey,
        coverage_evidence_spans: [contentEvidence[evidenceIndex]],
      },
    });
    const timeFact = (
      start: string,
      end: string,
      evidenceIndex: number,
      episodeKey: string,
    ) => ({
      content: `预约从${start}持续到${end}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_time",
        value: start,
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        activity_end_time: end,
        time_precision: "minute",
        episode_key: episodeKey,
        coverage_evidence_spans: [timeEvidence[evidenceIndex]],
      },
    });
    const start = "2026-04-01T10:00:00+08:00";
    const contentFacts = [
      contentFact("车辆检查", start, "2026-04-01T11:00:00+08:00", 0, "vehicle-check"),
      contentFact("体检", start, "2026-04-01T12:00:00+08:00", 1, "medical-check"),
    ];
    const timeFacts = [
      timeFact(start, "2026-04-01T11:00:00+08:00", 0, "vehicle-check"),
      timeFact(start, "2026-04-01T12:00:00+08:00", 1, "medical-check"),
    ];
    const scene = (memories: unknown[]) => JSON.stringify([{
      scene_name: "schedule",
      message_ids: ["u1"],
      memories,
    }]);
    const allFacts = [...contentFacts, ...timeFacts];
    const run = vi.fn(async (request: { taskId?: string; prompt?: string }) => {
      if (request.taskId === "l1-extraction" || request.taskId?.includes("atomic-compiler")) {
        return scene(allFacts);
      }
      if (request.taskId?.includes("coverage-fact")) {
        return request.prompt?.includes('"slot": "appointment_content"')
          ? scene(contentFacts)
          : scene(timeFacts);
      }
      if (request.taskId?.includes("construction-reconcile")) return "[]";
      throw new Error(`unexpected task ${request.taskId}`);
    });

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(true);
    expect(run).toHaveBeenCalledTimes(7);
    expect(result.records.filter((record) => record.metadata.slot === "appointment_time"))
      .toHaveLength(2);
    expect(result.records
      .filter((record) => record.metadata.slot === "appointment_time")
      .map((record) => record.metadata.activity_end_time)
      .sort()).toEqual([
        "2026-04-01T11:00:00+08:00",
        "2026-04-01T12:00:00+08:00",
      ]);
  });

  it("fails closed when blind appointment sets disagree only on activity end time", async () => {
    const sourceContent = "默认在2026年4月1日10点到11点做车辆检查。";
    const [contentEvidence] = coverageEvidenceSpans(sourceContent, "appointment_content");
    const [timeEvidence] = coverageEvidenceSpans(sourceContent, "appointment_time");
    const start = "2026-04-01T10:00:00+08:00";
    const correctEnd = "2026-04-01T11:00:00+08:00";
    const contentFact = {
      content: "用户默认在2026年4月1日10点做车辆检查。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_content",
        value: "车辆检查",
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        activity_end_time: correctEnd,
        time_precision: "minute",
        episode_key: "vehicle-check",
        coverage_evidence_spans: [contentEvidence],
      },
    };
    const timeFact = (end: string) => ({
      content: `车辆检查从${start}持续到${end}。`,
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "schedule",
        slot: "appointment_time",
        value: start,
        subject: "user",
        relation: "asserted",
        action_status: "requested",
        valid_from: start,
        activity_start_time: start,
        activity_end_time: end,
        time_precision: "minute",
        episode_key: "vehicle-check",
        coverage_evidence_spans: [timeEvidence],
      },
    });
    const correctTime = timeFact(correctEnd);
    const disagreeingTime = timeFact("2026-04-01T12:00:00+08:00");
    const scene = (memories: unknown[]) => JSON.stringify([{
      scene_name: "schedule",
      message_ids: ["u1"],
      memories,
    }]);
    const allFacts = [contentFact, correctTime];
    const run = vi.fn(async (request: { taskId?: string; prompt?: string }) => {
      if (request.taskId === "l1-extraction" || request.taskId?.includes("atomic-compiler")) {
        return scene(allFacts);
      }
      if (request.taskId?.includes("coverage-fact")) {
        if (request.prompt?.includes('"slot": "appointment_content"')) {
          return scene([contentFact]);
        }
        return request.taskId.includes("verifier")
          ? scene([disagreeingTime])
          : scene([correctTime]);
      }
      if (request.taskId?.includes("construction-reconciliation")) return "[]";
      throw new Error(`unexpected task ${request.taskId}`);
    });
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(11);
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("fails closed when a same-slot set compiler duplicates one fact", async () => {
    const sourceContent = "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。";
    const [firstEvidenceSpan] = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const feng = {
      content: "冯遥要求景点评分4.5分以上。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "冯遥",
        relation: "asserted",
        action_status: "requested",
        coverage_evidence_spans: [firstEvidenceSpan],
      },
    };
    const scene = (memories: Array<typeof feng>) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, structuredClone(feng)]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, structuredClone(feng)]));
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{
        id: "u1",
        role: "user",
        content: sourceContent,
        timestamp: 1,
      }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("expands one shared slot predicate into separately scoped people", async () => {
    const sourceContent = "冯遥和林静都要求景点评分4.5分以上。";
    const [sharedEvidence] = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const fact = (subject: string) => ({
      content: `${subject}要求景点评分4.5分以上。`,
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject,
        relation: "asserted",
        coverage_evidence_spans: [sharedEvidence],
      },
    });
    const feng = fact("冯遥");
    const lin = fact("林静");
    const scene = (memories: Array<typeof feng>) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-verifier",
    }));
    expect(result.records.map((record) => record.metadata.subject).sort())
      .toEqual(["冯遥", "林静"].sort());
    expect(result.records.every((record) => record.type === "instruction")).toBe(true);
  });

  it("collapses repeated same-slot evidence only after two blind sets agree", async () => {
    const sourceContent = "景点评分4.5分以上，记住，评分4.5分以上。";
    const repeatedEvidence = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const fact = {
      content: "用户要求景点评分4.5分以上。",
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "user",
        relation: "asserted",
        coverage_evidence_spans: repeatedEvidence,
      },
    };
    const scene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [fact],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0].metadata.value).toBe("4.5分以上");
  });

  it("does not let two distinct candidates cover only the first of two evidence groups", async () => {
    const sourceContent = "给冯遥设置景点评分4.5分以上，给林静设置景点评分4.0分以上。";
    const [firstEvidence] = coverageEvidenceSpans(sourceContent, "rating_constraint");
    const fact = (subject: string, value: string) => ({
      content: `${subject}要求景点评分${value}。`,
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value,
        subject,
        relation: "asserted",
        coverage_evidence_spans: [firstEvidence],
      },
    });
    const feng = fact("冯遥", "4.5分以上");
    const lin = fact("林静", "4.0分以上");
    const scene = (memories: Array<typeof feng>) => JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories,
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, lin]))
      .mockResolvedValueOnce(scene([feng]))
      .mockResolvedValueOnce(scene([feng, lin]));
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("treats a complete persistent coverage fact as instruction rather than forcing episodic", async () => {
    const sourceContent = "以后给我推荐时，评分至少4.5分。";
    const fact = {
      content: "以后推荐内容时，用户要求评分至少4.5分。",
      type: "instruction",
      priority: 75,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "至少4.5分",
        subject: "user",
        relation: "asserted",
        condition: "future_recommendation",
      },
    };
    const scene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [fact],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce("[]");

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: sourceContent, timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile: vi.fn(async () => undefined) } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(3);
    expect(run).not.toHaveBeenCalledWith(expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
    }));
    expect(result.success).toBe(true);
    expect(result.records).toHaveLength(1);
    expect(result.records[0].type).toBe("instruction");
  });

  it("fails closed when the directed coverage compiler crosses slots", async () => {
    const sourceContent = "我要求景点门票必须免费。";
    const coverageId = "coverage:u1:selection:price_constraint:ticket";
    const wrongDirectedFact = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [{
        content: "用户要求评分高于4.5分。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "selection",
          slot: "rating_constraint",
          value: "评分高于4.5分",
          subject: "user",
          relation: "asserted",
          episode_key: "poi-search-1",
          action_status: "requested",
          coverage_evidence_spans: coverageEvidenceSpans(
            sourceContent,
            "price_constraint",
            "ticket",
          ),
        },
      }],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(wrongDirectedFact)
      .mockResolvedValueOnce("[]")
      .mockResolvedValueOnce(wrongDirectedFact);
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{
        id: "u1",
        role: "user",
        content: sourceContent,
        timestamp: 1,
      }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(5);
    expect(run).toHaveBeenNthCalledWith(3, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler",
      prompt: expect.stringContaining(coverageId),
    }));
    expect(run).toHaveBeenNthCalledWith(4, expect.objectContaining({
      taskId: "l1-cockpit-atomic-compiler-transaction-retry",
    }));
    expect(run).toHaveBeenNthCalledWith(5, expect.objectContaining({
      taskId: "l1-cockpit-coverage-fact-compiler-transaction-retry",
    }));
    expect(result.success).toBe(false);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("fails closed without persisting primary candidates when the independent compiler is unavailable", async () => {
    const primary = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [{
        content: "用户计划七点出发。",
        type: "episodic",
        priority: 70,
        source_message_ids: ["u1"],
        metadata: {
          domain: "navigation",
          slot: "departure_time",
          value: "07:00",
          subject: "user",
          episode_key: "route-1",
          relation: "asserted",
          action_status: "requested",
        },
      }],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(primary)
      .mockRejectedValueOnce(new Error("Insufficient Balance"));
    const appendFile = vi.fn(async () => undefined);
    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(2);
    expect(result.success).toBe(false);
    expect(result.extractedCount).toBe(0);
    expect(result.storedCount).toBe(0);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("does not persist a preference hallucinated by both Flash passes from a pure information query", async () => {
    const proposal = {
      content: "用户要求景点评分4.5分以上。",
      type: "instruction",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "selection",
        slot: "rating_constraint",
        value: "4.5分以上",
        subject: "user",
        relation: "asserted",
      },
    };
    const scene = JSON.stringify([{
      scene_name: "selection",
      message_ids: ["u1"],
      memories: [proposal],
    }]);
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("construction-reconcile") ? "[]" : scene
    );
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{
        id: "u1",
        role: "user",
        content: "请问这家店评分4.5分以上吗？",
        timestamp: 1,
      }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
        constructionModel: "deepseek-v4-flash",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(result.success).toBe(true);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });

  it("fails closed without persisting candidates when reconciliation loses provider availability", async () => {
    const proposal = {
      content: "用户计划七点出发。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u1"],
      metadata: {
        domain: "navigation",
        slot: "departure_time",
        value: "07:00",
        subject: "user",
        episode_key: "route-1",
        relation: "asserted",
        action_status: "requested",
      },
    };
    const scene = JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u1"],
      memories: [proposal],
    }]);
    const run = vi.fn()
      .mockResolvedValueOnce(scene)
      .mockResolvedValueOnce(scene)
      .mockRejectedValueOnce(new Error("Insufficient Balance"));
    const appendFile = vi.fn(async () => undefined);

    const result = await extractL1Memories({
      messages: [{ id: "u1", role: "user", content: "七点出发", timestamp: 1 }],
      sessionKey: "session-key",
      sessionId: "session-1",
      teamId: "team-a",
      taskId: "task-a",
      userId: "user-a",
      agentId: "agent-a",
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        promptMode: "cockpit",
      },
      storage: { appendFile } as unknown as StorageAdapter,
    });

    expect(run).toHaveBeenCalledTimes(3);
    expect(result.success).toBe(false);
    expect(result.extractedCount).toBe(0);
    expect(result.storedCount).toBe(0);
    expect(result.records).toEqual([]);
    expect(appendFile).not.toHaveBeenCalled();
  });
});
