import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { initOffloadBridge } from "./bridge.js";

async function tempRoot(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "tdai-short-term-test-"));
}

describe("StandaloneOffloadBridge", () => {
  it("keeps small results inline and restores counters after restart", async () => {
    const dataDir = await tempRoot();
    const first = await initOffloadBridge({ dataDir });
    const result = await first.toolCall("agent:test:inline", "pwd", "/workspace", {
      eventId: "evt-inline-1",
    });

    expect(result.status).toBe("inline");
    expect(result.inlineContent).toBe("/workspace");
    expect(result.nodeId).toBeNull();

    const restarted = await initOffloadBridge({ dataDir });
    const status = await restarted.getStatus("agent:test:inline");
    expect(status.inlineCount).toBe(1);
    expect(status.offloadedCount).toBe(0);
    expect(status.checkpointExists).toBe(true);

    const duplicate = await restarted.toolCall("agent:test:inline", "pwd", "/workspace", {
      eventId: "evt-inline-1",
    });
    expect(duplicate.idempotent).toBe(true);
    expect((await restarted.getStatus("agent:test:inline")).inlineCount).toBe(1);
  });

  it("offloads large output losslessly and supports verified node drill-down", async () => {
    const dataDir = await tempRoot();
    let summaryCalls = 0;
    const bridge = await initOffloadBridge({
      dataDir,
      summarizer: async () => {
        summaryCalls += 1;
        return "Index validation failed at /workspace/index.db; rebuild the vector index.";
      },
    });
    const raw = [
      "Validating vector index",
      "ERROR: dimension mismatch at /workspace/index.db",
      ...Array.from({ length: 100 }, (_, index) => `row ${index}: diagnostic payload`),
    ].join("\n");

    const result = await bridge.toolCall("agent:test:offload", "validate_index", raw, {
      eventId: "evt-large-1",
      isError: true,
    });
    expect(result.status).toBe("offloaded");
    expect(result.compression).toBe("llm");
    expect(result.nodeId).toMatch(/^node-/);
    expect(result.tokensSaved).toBeGreaterThan(0);
    expect(result.summary).toContain("dimension mismatch");

    const drill = await bridge.drillDown("agent:test:offload", result.nodeId!);
    expect(drill.content).toBe(raw);
    expect(drill.verified).toBe(true);

    const context = await bridge.beforePrompt("agent:test:offload");
    expect(context.workingContext).toContain(result.nodeId!);
    expect(context.workingContext).toContain("Summaries are lossy");
    expect(context.canvas).toContain("flowchart TD");
    expect(context.offloadedCount).toBe(1);

    const duplicate = await bridge.toolCall("agent:test:offload", "validate_index", raw, {
      eventId: "evt-large-1",
      isError: true,
    });
    expect(duplicate.idempotent).toBe(true);
    expect(summaryCalls).toBe(1);

    const status = await bridge.getStatus("agent:test:offload");
    const checkpoint = JSON.parse(await readFile(path.join(status.dataDir, "checkpoint.json"), "utf8"));
    expect(Object.keys(checkpoint.nodes)).toEqual([result.nodeId]);
    expect((await readFile(path.join(status.dataDir, "steps.jsonl"), "utf8")).trim().split("\n")).toHaveLength(1);
  });

  it("uses an explicit fallback without losing critical evidence", async () => {
    const dataDir = await tempRoot();
    const bridge = await initOffloadBridge({
      dataDir,
      thresholds: { chars: 20 },
      summarizer: async () => { throw new Error("provider unavailable"); },
    });
    const raw = "start\nERROR: permission denied for /srv/data/index.db\nNext action: repair permissions";
    const result = await bridge.toolCall("agent:test:fallback", "repair", raw, {
      eventId: "evt-fallback",
    });

    expect(result.status).toBe("offloaded");
    expect(result.compression).toBe("extractive_fallback");
    expect(result.fallbackReason).toContain("provider unavailable");
    expect(result.summary).toContain("permission denied");
    expect((await bridge.drillDown("agent:test:fallback", result.nodeId!)).content).toBe(raw);
  });

  it("serializes concurrent duplicate events and rejects event payload changes", async () => {
    const dataDir = await tempRoot();
    let calls = 0;
    const bridge = await initOffloadBridge({
      dataDir,
      thresholds: { chars: 10 },
      summarizer: async () => {
        calls += 1;
        await new Promise((resolve) => setTimeout(resolve, 10));
        return "summary";
      },
    });
    const options = { eventId: "evt-concurrent" };
    const [a, b] = await Promise.all([
      bridge.toolCall("agent:test:concurrent", "exec", "a sufficiently long result", options),
      bridge.toolCall("agent:test:concurrent", "exec", "a sufficiently long result", options),
    ]);

    expect([a.idempotent, b.idempotent].sort()).toEqual([false, true]);
    expect(calls).toBe(1);
    await expect(
      bridge.toolCall("agent:test:concurrent", "exec", "different payload", options),
    ).rejects.toThrow("event_id payload mismatch");
  });
});
