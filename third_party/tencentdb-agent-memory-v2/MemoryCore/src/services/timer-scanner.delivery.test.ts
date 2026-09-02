import { describe, expect, it, vi } from "vitest";

import type { IStateBackend, TaskPayload, TimerEntry } from "../core/state/types.js";
import { TimerScanner } from "./timer-scanner.js";

const remoteMember = "instance-a\x00scope:team:team-a|agent:agent-a|session:profile-a:L3_quota";
const expired: TimerEntry = { member: remoteMember, fireAtMs: 1 };

function makeScanner(options?: { rearmFailsOnce?: boolean }) {
  const claimExpiredFromShard = vi.fn()
    .mockResolvedValueOnce([expired])
    .mockResolvedValueOnce([expired])
    .mockResolvedValue([]);
  const enqueueTask = vi.fn()
    .mockRejectedValueOnce(new Error("queue unavailable"))
    .mockResolvedValue(undefined);
  const setTimerIfEarlier = options?.rearmFailsOnce
    ? vi.fn().mockRejectedValueOnce(new Error("timer unavailable")).mockResolvedValue(undefined)
    : vi.fn().mockResolvedValue(undefined);
  const backend = {
    timerShardCount: 1,
    getTimerShardKeyByIndex: vi.fn(() => "timer-shard-0"),
    claimExpiredFromShard,
    enqueueTask,
    setTimerIfEarlier,
  } as unknown as IStateBackend & {
    timerShardCount: number;
    getTimerShardKeyByIndex(index: number): string;
    claimExpiredFromShard(key: string, now: number, count: number): Promise<TimerEntry[]>;
  };
  const logger = {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
  const scanner = new TimerScanner(backend, {
    nodeId: "scanner-test",
    deliveryRetryBaseMs: 1,
    deliveryRetryMaxMs: 1,
  }, logger);
  return { backend, enqueueTask, setTimerIfEarlier, scanner };
}

async function scan(scanner: TimerScanner): Promise<void> {
  await (scanner as unknown as { scan(): Promise<void> }).scan();
}

describe("TimerScanner durable queue handoff", () => {
  it("re-arms an exact claimed timer after enqueue failure, then delivers it with full isolation", async () => {
    const { enqueueTask, setTimerIfEarlier, scanner } = makeScanner();

    await scan(scanner);
    expect(setTimerIfEarlier).toHaveBeenCalledWith(
      "instance-a",
      "scope:team:team-a|agent:agent-a|session:profile-a:L3_quota",
      expect.any(Number),
    );
    expect(scanner.getMetrics()).toMatchObject({
      tasksEnqueued: 0,
      timerDeliveryRetries: 1,
      pendingTimerRearms: 0,
    });

    await scan(scanner);
    const delivered = enqueueTask.mock.calls[1][0] as TaskPayload;
    expect(delivered).toMatchObject({
      type: "L3",
      instanceId: "instance-a",
      sessionId: "profile-a",
      teamId: "team-a",
      agentId: "agent-a",
      priority: 2,
      data: {
        timerMember: "scope:team:team-a|agent:agent-a|session:profile-a:L3_quota",
      },
    });
    expect(scanner.getMetrics()).toMatchObject({ tasksEnqueued: 1, pendingTimerRearms: 0 });
  });

  it("retains a re-arm in process until timer persistence recovers", async () => {
    const { setTimerIfEarlier, scanner } = makeScanner({ rearmFailsOnce: true });

    await scan(scanner);
    expect(scanner.getMetrics()).toMatchObject({ pendingTimerRearms: 1, timerRearmFailures: 1 });

    await scan(scanner);
    expect(setTimerIfEarlier).toHaveBeenCalledTimes(2);
    expect(scanner.getMetrics()).toMatchObject({ pendingTimerRearms: 0 });
  });
});
