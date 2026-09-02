import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { IStateBackend, TaskPayload } from "../core/state/types.js";
import { StorageAdapter, createStagedStorageTransaction } from "../core/storage/adapter.js";
import { LocalStorageBackend } from "../core/storage/local-backend.js";
import { PipelineWorker, type TaskExecutor } from "./pipeline-worker.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  while (temporaryDirectories.length > 0) {
    rmSync(temporaryDirectories.pop()!, { recursive: true, force: true });
  }
});

function l2Task(): TaskPayload {
  return {
    id: "l2-lock-loss",
    type: "L2",
    instanceId: "instance-a",
    sessionId: "session-a",
    teamId: "team-a",
    agentId: "agent-a",
    priority: 1,
    createdAt: 1,
    data: {},
  };
}

describe("PipelineWorker lost-lease profile fencing", () => {
  it("aborts an in-flight staged L2 write and never publishes or ACKs it", async () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-worker-stage-"));
    temporaryDirectories.push(root);
    const canonical = new StorageAdapter(new LocalStorageBackend(root));
    await canonical.writeFile("persona.md", "stable");

    let currentOwner: string | undefined;
    const ackTask = vi.fn(async () => {});
    const releaseLock = vi.fn(async (_key: string, ownerId: string) => {
      if (currentOwner === ownerId) currentOwner = undefined;
    });
    const backend = {
      acquireLock: vi.fn(async (_key: string, ownerId: string) => {
        currentOwner = ownerId;
        return true;
      }),
      renewLock: vi.fn(async () => false),
      releaseLock,
      ackTask,
      enqueueTask: vi.fn(async () => {}),
    } as unknown as IStateBackend;

    const executeL2 = vi.fn<TaskExecutor["executeL2"]>(async (_task, signal, lease) => {
      const transaction = createStagedStorageTransaction(canonical, signal);
      await transaction.storage.writeFile("persona.md", "partial-model-output");
      await new Promise<void>((resolve) => {
        if (signal?.aborted) return resolve();
        signal?.addEventListener("abort", () => resolve(), { once: true });
      });
      await lease?.assertHeld();
      await transaction.commit(signal);
    });
    const executor: TaskExecutor = {
      executeL1: vi.fn(async () => {}),
      executeL2,
      executeL3: vi.fn(async () => {}),
    };
    const worker = new PipelineWorker(backend, executor, {
      workerId: "worker-a",
      lockRenewIntervalMs: 5,
      lockTtlMs: 100,
      maxRetries: 0,
    }, {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    });

    await (worker as unknown as { processTask(task: TaskPayload): Promise<void> }).processTask(l2Task());

    await expect(canonical.readFile("persona.md")).resolves.toBe("stable");
    expect(ackTask).not.toHaveBeenCalled();
    expect(executeL2).toHaveBeenCalledOnce();
    expect(worker.getMetrics()).toMatchObject({
      lockRenewFailed: 1,
      executionAborted: 1,
      tasksFailed: 1,
    });
  });

  it("uses a unique attempt owner so a stale task cannot release a replacement lease", async () => {
    let currentOwner: string | undefined;
    let staleOwner = "";
    let worker: PipelineWorker;
    const releaseLock = vi.fn(async (_key: string, ownerId: string) => {
      if (currentOwner === ownerId) currentOwner = undefined;
    });
    const backend = {
      acquireLock: vi.fn(async (_key: string, ownerId: string) => {
        staleOwner = ownerId;
        currentOwner = ownerId;
        return true;
      }),
      renewLock: vi.fn(async (_key: string, ownerId: string) => {
        if (ownerId === staleOwner) {
          currentOwner = "worker-a:replacement-attempt";
          (worker as unknown as { activeLocks: Map<string, string> }).activeLocks.set(
            "pipeline:{instance-a:team-a:agent-a}",
            currentOwner,
          );
          return false;
        }
        return currentOwner === ownerId;
      }),
      releaseLock,
      ackTask: vi.fn(async () => {}),
      enqueueTask: vi.fn(async () => {}),
    } as unknown as IStateBackend;
    const executor: TaskExecutor = {
      executeL1: vi.fn(async () => {}),
      executeL2: vi.fn(async (_task, signal) => {
        await new Promise<void>((resolve) => {
          if (signal?.aborted) return resolve();
          signal?.addEventListener("abort", () => resolve(), { once: true });
        });
      }),
      executeL3: vi.fn(async () => {}),
    };
    worker = new PipelineWorker(backend, executor, {
      workerId: "worker-a",
      lockRenewIntervalMs: 5,
      lockTtlMs: 100,
      maxRetries: 0,
    }, {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    });

    await (worker as unknown as { processTask(task: TaskPayload): Promise<void> }).processTask(l2Task());

    expect(staleOwner).toMatch(/^worker-a:l2-lock-loss:/);
    expect(staleOwner).not.toBe("worker-a");
    expect(releaseLock).toHaveBeenCalledWith(expect.any(String), staleOwner);
    expect(currentOwner).toBe("worker-a:replacement-attempt");
    expect((worker as unknown as { activeLocks: Map<string, string> }).activeLocks)
      .toEqual(new Map([["pipeline:{instance-a:team-a:agent-a}", "worker-a:replacement-attempt"]]));
  });
});
