import { describe, expect, it, vi } from "vitest";

import {
  CleanContextRunner,
  type EmbeddedAgentRuntimeLike,
} from "./clean-context-runner.js";

function runtimeFrom(
  run: (params: { abortSignal?: AbortSignal }) => Promise<unknown>,
): EmbeddedAgentRuntimeLike {
  return {
    runEmbeddedPiAgent:
      run as unknown as NonNullable<EmbeddedAgentRuntimeLike["runEmbeddedPiAgent"]>,
  };
}

describe("CleanContextRunner cancellation", () => {
  it("forwards the exact caller AbortSignal to the embedded provider", async () => {
    const controller = new AbortController();
    const embeddedRun = vi.fn(async (params: { abortSignal?: AbortSignal }) => {
      expect(params.abortSignal).toBe(controller.signal);
      return { payloads: [{ text: "ok" }] };
    });
    const runner = new CleanContextRunner({
      config: {},
      agentRuntime: runtimeFrom(embeddedRun),
    });

    await expect(
      runner.run({
        prompt: "test",
        taskId: "abort-forwarding",
        abortSignal: controller.signal,
      }),
    ).resolves.toBe("ok");
    expect(embeddedRun).toHaveBeenCalledOnce();
  });

  it("rejects an already-aborted run before invoking the provider", async () => {
    const reason = new Error("pipeline lock lost");
    const controller = new AbortController();
    controller.abort(reason);
    const embeddedRun = vi.fn(async () => ({ payloads: [{ text: "late" }] }));
    const runner = new CleanContextRunner({
      config: {},
      agentRuntime: runtimeFrom(embeddedRun),
    });

    await expect(
      runner.run({
        prompt: "test",
        taskId: "abort-before-admission",
        abortSignal: controller.signal,
      }),
    ).rejects.toBe(reason);
    expect(embeddedRun).not.toHaveBeenCalled();
  });

  it("discards a provider result returned after lock loss", async () => {
    const reason = new Error("pipeline lock lost");
    const controller = new AbortController();
    const embeddedRun = vi.fn(async () => {
      controller.abort(reason);
      return { payloads: [{ text: "late output must not persist" }] };
    });
    const runner = new CleanContextRunner({
      config: {},
      agentRuntime: runtimeFrom(embeddedRun),
    });

    await expect(
      runner.run({
        prompt: "test",
        taskId: "abort-after-provider",
        abortSignal: controller.signal,
      }),
    ).rejects.toBe(reason);
  });
});
