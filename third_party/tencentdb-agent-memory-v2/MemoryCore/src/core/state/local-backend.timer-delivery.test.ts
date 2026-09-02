import { afterEach, describe, expect, it, vi } from "vitest";

import { LocalStateBackend } from "./local-backend.js";

afterEach(() => {
  vi.useRealTimers();
});

describe("LocalStateBackend timer delivery handoff", () => {
  it("re-arms the exact timer when its first async queue handoff fails", async () => {
    vi.useFakeTimers();
    const onTimerExpired = vi.fn()
      .mockRejectedValueOnce(new Error("queue unavailable"))
      .mockResolvedValueOnce(undefined);
    const backend = new LocalStateBackend({
      onTimerExpired,
      timerDeliveryRetryBaseMs: 10,
      timerDeliveryRetryMaxMs: 10,
    });

    await backend.setTimer("instance-a", "scope:team:t|agent:a|session:s:L3_quota", Date.now() + 1);
    await vi.advanceTimersByTimeAsync(1);

    expect(onTimerExpired).toHaveBeenCalledTimes(1);
    expect(backend.getSnapshot()).toMatchObject({ timers: 1, timerDeliveryRetries: 1 });

    await vi.advanceTimersByTimeAsync(10);

    expect(onTimerExpired).toHaveBeenCalledTimes(2);
    expect(onTimerExpired).toHaveBeenLastCalledWith(expect.objectContaining({
      member: "scope:team:t|agent:a|session:s:L3_quota",
    }));
    expect(backend.getSnapshot()).toMatchObject({ timers: 0, timerDeliveryRetries: 1 });
  });
});

