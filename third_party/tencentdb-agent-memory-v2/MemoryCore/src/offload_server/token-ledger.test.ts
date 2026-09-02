import { describe, expect, it } from "vitest";
import type { StorageAdapter } from "../core/storage/adapter.js";
import {
  normalizeTokenUsage,
  summarizeTokenLedger,
  writeBuildTokenRecord,
  writeMainTokenRecord,
} from "./token-ledger.js";

function memoryStorage(): StorageAdapter {
  const files = new Map<string, string>();
  return {
    async writeFile(key: string, content: string | Buffer) {
      files.set(key, content.toString());
    },
    async readFile(key: string) {
      return files.get(key) ?? null;
    },
    async readdirNames(prefix: string, suffix?: string) {
      return [...files.keys()]
        .filter((key) => key.startsWith(prefix) && (!suffix || key.endsWith(suffix)))
        .map((key) => key.slice(prefix.length));
    },
  } as unknown as StorageAdapter;
}

describe("offload token ledger", () => {
  it("normalizes OpenAI and Anthropic-compatible usage", () => {
    expect(normalizeTokenUsage({
      prompt_tokens: 100,
      completion_tokens: 20,
      total_tokens: 120,
      prompt_tokens_details: { cached_tokens: 40 },
    }, 1, 1)).toMatchObject({
      input_tokens: 100,
      output_tokens: 20,
      total_tokens: 120,
      cache_read_tokens: 40,
      source: "provider",
    });

    expect(normalizeTokenUsage({
      input_tokens: 50,
      output_tokens: 10,
      cache_read_input_tokens: 30,
      cache_creation_input_tokens: 20,
    }, 1, 1)).toMatchObject({
      input_tokens: 100,
      output_tokens: 10,
      total_tokens: 110,
      source: "provider",
    });
  });

  it("aggregates build cost, gross saving, net saving and break-even", async () => {
    const storage = memoryStorage();
    const basePath = "offload/proxy-test";
    await writeBuildTokenRecord(storage, basePath, {
      kind: "build",
      record_id: "l1-task",
      timestamp: "2026-08-18T00:00:00.000Z",
      session_id: "proxy-test",
      stage: "l1",
      model: "test",
      duration_ms: 100,
      usage: normalizeTokenUsage({ prompt_tokens: 800, completion_tokens: 200 }, 0, 0),
    });
    await writeBuildTokenRecord(storage, basePath, {
      kind: "build",
      record_id: "l2-task",
      timestamp: "2026-08-18T00:00:01.000Z",
      session_id: "proxy-test",
      stage: "l2",
      model: "test",
      duration_ms: 200,
      usage: normalizeTokenUsage(undefined, 400, 100),
    });
    await writeMainTokenRecord(storage, basePath, {
      kind: "main",
      record_id: "main-1",
      timestamp: "2026-08-18T00:00:02.000Z",
      session_id: "proxy-test",
      protocol: "openai",
      applied: true,
      before_estimated_tokens: 6_000,
      forwarded_estimated_tokens: 2_000,
      estimated_gross_saved_tokens: 4_000,
      actual_forwarded_input_tokens: 2_100,
      actual_output_tokens: 300,
      forwarded_input_source: "provider",
      mmd_injected: 1,
    });

    const summary = await summarizeTokenLedger(storage, basePath, "proxy-test");
    expect(summary.build).toMatchObject({
      calls: 2,
      input_tokens: 1_200,
      output_tokens: 300,
      total_tokens: 1_500,
      provider_measured_calls: 1,
      estimated_calls: 1,
    });
    expect(summary.main).toMatchObject({
      estimated_gross_saved_tokens: 4_000,
      provider_measured_calls: 1,
      estimated_calls: 0,
    });
    expect(summary.net.estimated_net_saved_tokens).toBe(2_500);
    expect(summary.net.estimated_net_savings_ratio).toBeCloseTo(2_500 / 6_100);
    expect(summary.net.break_even_reuse_calls).toBe(1);
  });
});
