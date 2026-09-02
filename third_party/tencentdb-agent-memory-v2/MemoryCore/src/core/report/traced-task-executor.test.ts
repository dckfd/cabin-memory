import { describe, expect, it, vi } from "vitest";

import type { TaskPayload } from "../state/types.js";
import type { TaskExecutionLease, TaskExecutor } from "../../services/pipeline-worker.js";
import { TracedTaskExecutor } from "./traced-task-executor.js";

function task(type: TaskPayload["type"]): TaskPayload {
  return {
    id: `task-${type}`,
    type,
    instanceId: "instance-1",
    sessionId: "session-1",
    priority: 1,
    createdAt: 1,
  };
}

describe("TracedTaskExecutor cancellation", () => {
  it("forwards the exact AbortSignal through every decorated task method", async () => {
    const executeL1 = vi.fn(async () => {});
    const executeL2 = vi.fn(async () => {});
    const executeL3 = vi.fn(async () => {});
    const executeFlush = vi.fn(async () => {});
    const executeOffloadL1 = vi.fn(async () => {});
    const executeOffloadL15 = vi.fn(async () => {});
    const executeOffloadL2 = vi.fn(async () => {});
    const inner: TaskExecutor = {
      executeL1,
      executeL2,
      executeL3,
      executeFlush,
      executeOffloadL1,
      executeOffloadL15,
      executeOffloadL2,
    };
    const traced = new TracedTaskExecutor(inner);
    const controller = new AbortController();
    const lease: TaskExecutionLease = { ownerId: "attempt-1", assertHeld: vi.fn(async () => {}) };

    await traced.executeL1(task("L1"), controller.signal, lease);
    await traced.executeL2(task("L2"), controller.signal, lease);
    await traced.executeL3(task("L3"), controller.signal, lease);
    await traced.executeFlush!(task("flush"), controller.signal, lease);
    await traced.executeOffloadL1!(task("offload-l1"), controller.signal, lease);
    await traced.executeOffloadL15!(task("offload-l15"), controller.signal, lease);
    await traced.executeOffloadL2!(task("offload-l2"), controller.signal, lease);

    for (const call of [
      executeL1,
      executeL2,
      executeL3,
      executeFlush,
      executeOffloadL1,
      executeOffloadL15,
      executeOffloadL2,
    ]) {
      expect(call).toHaveBeenCalledOnce();
      expect(call.mock.calls[0]?.[1]).toBe(controller.signal);
      expect(call.mock.calls[0]?.[2]).toBe(lease);
    }
  });

  it("forwards the signal through the flush-to-L1 fallback", async () => {
    const executeL1 = vi.fn(async () => {});
    const traced = new TracedTaskExecutor({
      executeL1,
      executeL2: vi.fn(async () => {}),
      executeL3: vi.fn(async () => {}),
    });
    const controller = new AbortController();
    const flushTask = task("flush");
    const lease: TaskExecutionLease = { ownerId: "attempt-2", assertHeld: vi.fn(async () => {}) };

    await traced.executeFlush!(flushTask, controller.signal, lease);

    expect(executeL1).toHaveBeenCalledWith(flushTask, controller.signal, lease);
  });
});
