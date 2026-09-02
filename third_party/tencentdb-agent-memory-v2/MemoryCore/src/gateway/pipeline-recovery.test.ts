import { describe, expect, it, vi } from "vitest";
import type { IStateBackend, TaskPayload } from "../core/state/types.js";
import {
  enqueuePipelineRecovery,
  parsePipelineRecoveryRequest,
  type PipelineRecoveryRequest,
} from "./pipeline-recovery.js";

function request(recoveryId: string, sessions = ["session-01", "session-02"]): PipelineRecoveryRequest {
  return {
    recovery_id: recoveryId,
    instance_id: "instance-a",
    expected_task_count: sessions.length,
    tasks: sessions.map((sessionId) => ({
      session_id: sessionId,
      team_id: "team-a",
      agent_id: "agent-a",
    })),
  };
}

function backendWith(enqueueTask: (task: TaskPayload) => Promise<void>): IStateBackend {
  return { enqueueTask } as unknown as IStateBackend;
}

describe("pipeline recovery contract", () => {
  it("accepts an exact scoped manifest and produces a stable digest", () => {
    const body = request("recovery-valid-01");
    const first = parsePipelineRecoveryRequest(body, "instance-a");
    const second = parsePipelineRecoveryRequest(body, "instance-a");

    expect(first.ok).toBe(true);
    expect(second.ok).toBe(true);
    if (first.ok && second.ok) {
      expect(first.value).toEqual(body);
      expect(first.digest).toMatch(/^[a-f0-9]{64}$/);
      expect(second.digest).toBe(first.digest);
    }
  });

  it("fails closed on scope mismatch, truncation, duplicates, and extra fields", () => {
    const scopeMismatch = parsePipelineRecoveryRequest(request("recovery-scope-01"), "instance-b");
    expect(scopeMismatch).toMatchObject({ ok: false, status: 403 });

    const truncated = request("recovery-truncated-01");
    truncated.expected_task_count = 3;
    expect(parsePipelineRecoveryRequest(truncated, "instance-a"))
      .toMatchObject({ ok: false, status: 400 });

    const duplicate = request("recovery-duplicate-01", ["session-01", "session-01"]);
    expect(parsePipelineRecoveryRequest(duplicate, "instance-a"))
      .toMatchObject({ ok: false, status: 400 });

    const extra = { ...request("recovery-extra-01"), force: true };
    expect(parsePipelineRecoveryRequest(extra, "instance-a"))
      .toMatchObject({ ok: false, status: 400 });
  });

  it("enqueues only L1 tasks with authenticated scope and recovery provenance", async () => {
    const accepted: TaskPayload[] = [];
    const body = request("recovery-enqueue-01");
    const parsed = parsePipelineRecoveryRequest(body, "instance-a");
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    const result = await enqueuePipelineRecovery(
      backendWith(async (task) => { accepted.push(task); }),
      parsed.value,
      parsed.digest,
      1_000,
    );

    expect(result).toMatchObject({
      ok: true,
      enqueued_count: 2,
      already_accepted_count: 0,
      idempotent_replay: false,
    });
    expect(accepted).toHaveLength(2);
    expect(accepted.map((task) => ({
      type: task.type,
      instanceId: task.instanceId,
      teamId: task.teamId,
      agentId: task.agentId,
      sessionId: task.sessionId,
      priority: task.priority,
      data: task.data,
    }))).toEqual([
      {
        type: "L1",
        instanceId: "instance-a",
        teamId: "team-a",
        agentId: "agent-a",
        sessionId: "session-01",
        priority: 0,
        data: {
          instanceId: "instance-a",
          teamId: "team-a",
          agentId: "agent-a",
          triggeredBy: "pipeline_recovery",
          recoveryId: "recovery-enqueue-01",
        },
      },
      {
        type: "L1",
        instanceId: "instance-a",
        teamId: "team-a",
        agentId: "agent-a",
        sessionId: "session-02",
        priority: 0,
        data: {
          instanceId: "instance-a",
          teamId: "team-a",
          agentId: "agent-a",
          triggeredBy: "pipeline_recovery",
          recoveryId: "recovery-enqueue-01",
        },
      },
    ]);
    expect(accepted[0].id).toMatch(/^recovery-l1-[a-f0-9]{32}$/);
    expect(accepted[1].createdAt).toBe(1_001);
  });

  it("is idempotent for an identical retry and rejects recovery_id content drift", async () => {
    const enqueueTask = vi.fn(async (_task: TaskPayload) => {});
    const firstBody = request("recovery-idempotent-01");
    const first = parsePipelineRecoveryRequest(firstBody, "instance-a");
    expect(first.ok).toBe(true);
    if (!first.ok) return;

    await enqueuePipelineRecovery(backendWith(enqueueTask), first.value, first.digest, 2_000);
    const replay = await enqueuePipelineRecovery(backendWith(enqueueTask), first.value, first.digest, 3_000);
    expect(replay).toMatchObject({
      ok: true,
      enqueued_count: 0,
      already_accepted_count: 2,
      idempotent_replay: true,
    });
    expect(enqueueTask).toHaveBeenCalledTimes(2);

    const changed = parsePipelineRecoveryRequest(
      request("recovery-idempotent-01", ["session-03"]),
      "instance-a",
    );
    expect(changed.ok).toBe(true);
    if (!changed.ok) return;
    await expect(enqueuePipelineRecovery(backendWith(enqueueTask), changed.value, changed.digest, 4_000))
      .resolves.toMatchObject({ ok: false, status: 409 });
    expect(enqueueTask).toHaveBeenCalledTimes(2);
  });

  it("resumes after a partial enqueue failure without duplicating accepted tasks", async () => {
    const body = request("recovery-partial-01");
    const parsed = parsePipelineRecoveryRequest(body, "instance-a");
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;

    let calls = 0;
    await expect(enqueuePipelineRecovery(
      backendWith(async () => {
        calls++;
        if (calls === 2) throw new Error("synthetic queue failure");
      }),
      parsed.value,
      parsed.digest,
      5_000,
    )).rejects.toThrow("synthetic queue failure");

    const recovered: TaskPayload[] = [];
    const retry = await enqueuePipelineRecovery(
      backendWith(async (task) => { recovered.push(task); }),
      parsed.value,
      parsed.digest,
      6_000,
    );
    expect(retry).toMatchObject({ ok: true, enqueued_count: 1, already_accepted_count: 1 });
    expect(recovered.map((task) => task.sessionId)).toEqual(["session-02"]);
  });
});
