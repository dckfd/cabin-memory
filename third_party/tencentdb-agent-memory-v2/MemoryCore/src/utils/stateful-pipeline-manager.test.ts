import { describe, expect, it, vi } from "vitest";

import type { MemoryTdaiConfig } from "../config.js";
import type { IStateBackend } from "../core/state/types.js";
import { parsePipelineTimerMember } from "../core/state/timer-member.js";
import { createStatefulPipelineManager } from "./pipeline-factory.js";


describe("StatefulPipelineManager extraction gate", () => {
  it("does not enqueue or arm memory work when extraction is disabled", async () => {
    const backend = {
      captureAtomic: vi.fn(),
      getSessionState: vi.fn(),
      updateSessionState: vi.fn(),
      enqueueTask: vi.fn(),
      removeTimer: vi.fn(),
      setTimer: vi.fn(),
      setTimerIfEarlier: vi.fn(),
    } as unknown as IStateBackend;
    const logger = {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    };
    const config = {
      extraction: { enabled: false },
      pipeline: {
        everyNConversations: 1,
        enableWarmup: false,
        l1IdleTimeoutSeconds: 1,
        l2DelayAfterL1Seconds: 1,
        l2MinIntervalSeconds: 0,
        l2MaxIntervalSeconds: 60,
        sessionActiveWindowHours: 24,
      },
    } as MemoryTdaiConfig;
    const manager = createStatefulPipelineManager(
      config,
      backend,
      "instance-test",
      logger,
    );

    await manager.start();
    await manager.notifyConversation("session-1", [], "instance-test", 1);
    await manager.flushSession("session-1", "instance-test");
    await manager.advanceL2TimerAfterL1("session-1", "instance-test");
    await manager.armL2MaxInterval("session-1", "instance-test");
    await manager.enqueueL1Drain("session-1", "instance-test");
    const l1Deferred = await manager.armL1IdleAfterDrain("session-1", "instance-test");
    const l2Deferred = await manager.deferL2ForQuota("session-1", "instance-test");
    const l3Deferred = await manager.deferL3ForQuota("session-1", "instance-test");

    expect(backend.captureAtomic).not.toHaveBeenCalled();
    expect(backend.getSessionState).not.toHaveBeenCalled();
    expect(backend.enqueueTask).not.toHaveBeenCalled();
    expect(backend.setTimer).not.toHaveBeenCalled();
    expect(backend.setTimerIfEarlier).not.toHaveBeenCalled();
    expect(l1Deferred).toBe(false);
    expect(l2Deferred).toBe(false);
    expect(l3Deferred).toBe(false);
    expect(logger.info).toHaveBeenCalledWith(
      expect.stringContaining("memory.extraction.enabled=false"),
    );
  });

  it("treats an existing earlier timer as a durable quota deferral", async () => {
    const backend = {
      getSessionState: vi.fn(async () => undefined),
      setTimerIfEarlier: vi.fn(async () => false),
    } as unknown as IStateBackend;
    const config = {
      extraction: { enabled: true },
      pipeline: {
        everyNConversations: 1,
        enableWarmup: false,
        l1IdleTimeoutSeconds: 1,
        l2DelayAfterL1Seconds: 2,
        l2MinIntervalSeconds: 0,
        l2MaxIntervalSeconds: 60,
        sessionActiveWindowHours: 24,
      },
    } as MemoryTdaiConfig;
    const manager = createStatefulPipelineManager(config, backend, "instance-test");

    await expect(manager.armL1IdleAfterDrain("session-1", "instance-test", "team-a", "agent-a"))
      .resolves.toBe(true);
    await expect(manager.deferL2ForQuota("profile-1", "instance-test", "team-a", "agent-a"))
      .resolves.toBe(true);
    await expect(manager.deferL3ForQuota("profile-1", "instance-test", "team-a", "agent-a"))
      .resolves.toBe(true);

    expect(backend.setTimerIfEarlier).toHaveBeenCalledTimes(3);
    expect(backend.setTimerIfEarlier).toHaveBeenCalledWith(
      "instance-test",
      "scope:team:team-a|agent:agent-a|session:session-1:L1_idle",
      expect.any(Number),
    );
    expect(backend.setTimerIfEarlier).toHaveBeenCalledWith(
      "instance-test",
      "scope:team:team-a|agent:agent-a|session:profile-1:L2_schedule",
      expect.any(Number),
    );
    expect(backend.setTimerIfEarlier).toHaveBeenCalledWith(
      "instance-test",
      "scope:team:team-a|agent:agent-a|session:profile-1:L3_quota",
      expect.any(Number),
    );
    expect(parsePipelineTimerMember(
      "scope:team:team-a|agent:agent-a|session:profile-1:L3_quota",
    )).toMatchObject({
      taskType: "L3",
      sessionId: "profile-1",
      teamId: "team-a",
      agentId: "agent-a",
    });
  });

  it("propagates timer persistence failures instead of acknowledging quota-blocked work", async () => {
    const backend = {
      getSessionState: vi.fn(async () => undefined),
      setTimerIfEarlier: vi.fn(async () => { throw new Error("timer unavailable"); }),
    } as unknown as IStateBackend;
    const config = {
      extraction: { enabled: true },
      pipeline: {
        everyNConversations: 1,
        enableWarmup: false,
        l1IdleTimeoutSeconds: 1,
        l2DelayAfterL1Seconds: 2,
        l2MinIntervalSeconds: 0,
        l2MaxIntervalSeconds: 60,
        sessionActiveWindowHours: 24,
      },
    } as MemoryTdaiConfig;
    const manager = createStatefulPipelineManager(config, backend, "instance-test");

    await expect(manager.armL1IdleAfterDrain("session-1", "instance-test"))
      .rejects.toThrow("timer unavailable");
    await expect(manager.deferL2ForQuota("profile-1", "instance-test"))
      .rejects.toThrow("timer unavailable");
    await expect(manager.deferL3ForQuota("profile-1", "instance-test"))
      .rejects.toThrow("timer unavailable");
  });

  it("fails quota deferral closed when service mode omits the required instance", async () => {
    const backend = {
      getSessionState: vi.fn(),
      setTimerIfEarlier: vi.fn(),
    } as unknown as IStateBackend;
    const config = {
      extraction: { enabled: true },
      pipeline: {
        everyNConversations: 1,
        enableWarmup: false,
        l1IdleTimeoutSeconds: 1,
        l2DelayAfterL1Seconds: 2,
        l2MinIntervalSeconds: 0,
        l2MaxIntervalSeconds: 60,
        sessionActiveWindowHours: 24,
      },
    } as MemoryTdaiConfig;
    const manager = createStatefulPipelineManager(config, backend, "__unset__");

    await expect(manager.armL1IdleAfterDrain("session-1")).resolves.toBe(false);
    await expect(manager.deferL2ForQuota("profile-1")).resolves.toBe(false);
    await expect(manager.deferL3ForQuota("profile-1")).resolves.toBe(false);
    expect(backend.getSessionState).not.toHaveBeenCalled();
    expect(backend.setTimerIfEarlier).not.toHaveBeenCalled();
  });
});
