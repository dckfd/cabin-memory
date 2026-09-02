import { describe, expect, it, vi } from "vitest";

import type { IStateBackend, TaskPayload } from "../core/state/types.js";
import {
  clearL1CascadeOutcome,
  clearL2CascadeOutcome,
  PipelineWorker,
  recordL1CascadeOutcome,
  suppressL1Cascade,
  suppressL2Cascade,
  type TaskExecutor,
} from "./pipeline-worker.js";

function l1Task(): TaskPayload {
  return {
    id: "l1-cascade-contract",
    type: "L1",
    instanceId: "instance-a",
    sessionId: "session-a",
    teamId: "team-a",
    agentId: "agent-a",
    priority: 0,
    createdAt: 1,
    data: {},
    _msgId: "message-a",
  } as TaskPayload;
}

function harness(executeL1: TaskExecutor["executeL1"], maxRetries = 0) {
  let owner: string | undefined;
  const backend = {
    acquireLock: vi.fn(async (_key: string, ownerId: string) => {
      owner = ownerId;
      return true;
    }),
    renewLock: vi.fn(async (_key: string, ownerId: string) => owner === ownerId),
    releaseLock: vi.fn(async (_key: string, ownerId: string) => {
      if (owner === ownerId) owner = undefined;
    }),
    ackTask: vi.fn(async () => {}),
    enqueueTask: vi.fn(async () => {}),
    updateSessionState: vi.fn(async () => {}),
  } as unknown as IStateBackend;
  const onL1Complete = vi.fn(async () => {});
  const executor: TaskExecutor = {
    executeL1,
    executeL2: vi.fn(async () => {}),
    executeL3: vi.fn(async () => {}),
  };
  const worker = new PipelineWorker(backend, executor, {
    workerId: "worker-a",
    lockRenewIntervalMs: 1_000,
    lockTtlMs: 10_000,
    maxRetries,
    retryBaseDelayMs: 0,
    onL1Complete,
  }, {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  });
  return { backend, onL1Complete, worker };
}

async function process(worker: PipelineWorker, task: TaskPayload): Promise<void> {
  await (worker as unknown as { processTask(task: TaskPayload): Promise<void> }).processTask(task);
}

describe("PipelineWorker L1 no-op cascade contract", () => {
  it("ACKs and resets an explicitly empty L1 outcome without advancing L2", async () => {
    const task = l1Task();
    const { backend, onL1Complete, worker } = harness(async (current) => {
      recordL1CascadeOutcome(current, { storedCount: 0, profileScopes: [] });
    });

    await process(worker, task);

    expect(backend.ackTask).toHaveBeenCalledWith("message-a");
    expect(backend.updateSessionState).toHaveBeenCalledTimes(1);
    expect(backend.updateSessionState).toHaveBeenCalledWith(
      "instance-a",
      "session-a",
      { conversation_count: 0 },
      "team-a",
      "agent-a",
    );
    expect(onL1Complete).not.toHaveBeenCalled();
    expect(backend.enqueueTask).not.toHaveBeenCalled();
    expect(worker.getMetrics()).toMatchObject({ tasksCompleted: 1, tasksFailed: 0 });
  });

  it("advances every exact non-empty profile scope", async () => {
    const task = l1Task();
    const { backend, onL1Complete, worker } = harness(async (current) => {
      recordL1CascadeOutcome(current, {
        storedCount: 2,
        profileScopes: ["profile-scope-a", "profile-scope-b"],
      });
    });

    await process(worker, task);

    expect(onL1Complete).toHaveBeenCalledTimes(2);
    expect(onL1Complete).toHaveBeenNthCalledWith(
      1, "profile-scope-a", "instance-a", "team-a", "agent-a",
    );
    expect(onL1Complete).toHaveBeenNthCalledWith(
      2, "profile-scope-b", "instance-a", "team-a", "agent-a",
    );
    expect(backend.updateSessionState).toHaveBeenCalledTimes(3);
    expect(worker.getMetrics()).toMatchObject({ tasksCompleted: 1, tasksFailed: 0 });
  });

  it.each([
    ["missing", undefined],
    ["null member", [null]],
    ["empty-string member", [""]],
  ])("keeps the legacy session fallback for a %s marker", async (_label, marker) => {
    const task = l1Task();
    const { onL1Complete, worker } = harness(async (current) => {
      if (marker !== undefined) {
        (current as TaskPayload & { _l2ProfileScopes?: unknown })._l2ProfileScopes = marker;
      }
    });

    await process(worker, task);

    expect(onL1Complete).toHaveBeenCalledOnce();
    expect(onL1Complete).toHaveBeenCalledWith(
      "session-a", "instance-a", "team-a", "agent-a",
    );
  });

  it("rejects contradictory or malformed runner outcomes before publication", () => {
    const task = l1Task();
    expect(() => recordL1CascadeOutcome(task, { storedCount: 1, profileScopes: [] }))
      .toThrow("Inconsistent L1 cascade outcome");
    expect(() => recordL1CascadeOutcome(task, { storedCount: 0, profileScopes: ["scope-a"] }))
      .toThrow("Inconsistent L1 cascade outcome");
    expect(() => recordL1CascadeOutcome(task, { storedCount: 1, profileScopes: [""] }))
      .toThrow("Invalid L1 cascade profileScopes");
    expect((task as TaskPayload & { _l2ProfileScopes?: unknown })._l2ProfileScopes).toBeUndefined();
  });

  it("clears attempt-local evidence and can publish an intentional successful skip", () => {
    const task = l1Task();
    suppressL1Cascade(task);
    expect((task as TaskPayload & { _l2ProfileScopes?: unknown })._l2ProfileScopes).toEqual([]);

    clearL1CascadeOutcome(task);
    expect((task as TaskPayload & { _l2ProfileScopes?: unknown })._l2ProfileScopes).toBeUndefined();
  });

  it("fails the worker path closed for a contradictory runner result", async () => {
    const task = l1Task();
    const onDeadLetter = vi.fn(async () => {});
    const { backend, onL1Complete, worker } = harness(async (current) => {
      recordL1CascadeOutcome(current, { storedCount: 1, profileScopes: [] });
    });
    (worker as unknown as { config: { onDeadLetter?: typeof onDeadLetter } }).config.onDeadLetter = onDeadLetter;

    await process(worker, task);

    expect(onL1Complete).not.toHaveBeenCalled();
    expect(backend.updateSessionState).not.toHaveBeenCalled();
    expect(onDeadLetter).toHaveBeenCalledOnce();
    expect(worker.getMetrics()).toMatchObject({
      tasksCompleted: 0,
      tasksFailed: 1,
      tasksDeadLettered: 1,
    });
  });

  it("strips delivery and cascade markers from a retry task", async () => {
    const task = l1Task() as TaskPayload & {
      _l2ProfileScopes?: unknown;
      _l2Skipped?: unknown;
      _deferredEnqueue?: unknown;
    };
    task._l2ProfileScopes = [];
    task._l2Skipped = true;
    task._deferredEnqueue = [{
      ...l1Task(),
      id: "must-not-be-enqueued-from-failed-attempt",
      _msgId: undefined,
    } as TaskPayload];
    const { backend, worker } = harness(async () => {
      throw new Error("retry me");
    }, 1);

    await process(worker, task);

    const retry = (backend.enqueueTask as ReturnType<typeof vi.fn>).mock.calls
      .map(([queued]) => queued as TaskPayload & Record<string, unknown>)
      .find((queued) => queued.id.includes("-retry1-"));
    expect(retry).toBeDefined();
    expect(retry).not.toHaveProperty("_msgId");
    expect(retry).not.toHaveProperty("_l2ProfileScopes");
    expect(retry).not.toHaveProperty("_l2Skipped");
    expect(retry).not.toHaveProperty("_deferredEnqueue");
    expect(retry?.data).toMatchObject({ retryCount: 1 });
    expect(backend.enqueueTask).not.toHaveBeenCalledWith(
      expect.objectContaining({ id: "must-not-be-enqueued-from-failed-attempt" }),
    );
    expect(worker.getMetrics()).toMatchObject({ tasksCompleted: 0, tasksFailed: 1, tasksRetried: 1 });
  });

  it("suppresses L2→L3 only for an explicit current-attempt decision", async () => {
    const task = { ...l1Task(), id: "l2-skip", type: "L2" as const };
    const { backend, worker } = harness(async () => {});
    const executor = (worker as unknown as { executor: TaskExecutor }).executor;
    executor.executeL2 = async (current) => {
      clearL2CascadeOutcome(current);
      suppressL2Cascade(current);
    };

    await process(worker, task);

    expect(backend.enqueueTask).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "L3" }),
    );
    expect(worker.getMetrics()).toMatchObject({ tasksCompleted: 1, tasksFailed: 0 });
  });
});
