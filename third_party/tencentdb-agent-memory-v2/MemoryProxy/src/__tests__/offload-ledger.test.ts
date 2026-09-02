import { afterEach, describe, expect, it, vi } from "vitest";
import type { ProxyConfig } from "../types.js";
import type { OffloadProcessResult } from "../offload/bridge.js";
import { reportOffloadMainUsage } from "../offload/ledger.js";

function config(): ProxyConfig {
  return {
    tdai: {
      endpoint: "http://memory-core:8420",
      apiKey: "core-key",
      offload: { timeoutMs: 1_000 },
    },
  } as ProxyConfig;
}

function result(): OffloadProcessResult {
  return {
    body: { messages: [] },
    applied: true,
    beforeTokens: 4_000,
    afterTokens: 1_500,
    savedTokens: 2_500,
    mmdInjected: 1,
    ingestedToolPairs: 1,
    ingestedPrompt: true,
    offloadSessionId: "proxy-test",
    offloadServiceId: "default",
    requestId: "request-1",
  };
}

function summary() {
  return {
    session_id: "proxy-test",
    build: {
      calls: 0,
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      provider_measured_calls: 0,
      estimated_calls: 0,
    },
    main: {
      calls: 1,
      applied_calls: 1,
      actual_forwarded_input_tokens: 1_500,
      actual_output_tokens: 0,
      estimated_counterfactual_input_tokens: 4_000,
      estimated_gross_saved_tokens: 2_500,
      provider_measured_calls: 0,
      estimated_calls: 1,
    },
    net: {
      estimated_net_saved_tokens: 2_500,
      estimated_net_savings_ratio: 0.625,
      break_even_reuse_calls: 0,
    },
  };
}

describe("offload main usage ledger", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("marks request-side fallback input as estimated when streaming usage is absent", async () => {
    let posted: Record<string, unknown> | undefined;
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    vi.stubGlobal("fetch", vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      posted = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response(JSON.stringify({ code: 0, data: summary() }), { status: 200 });
    }));

    await reportOffloadMainUsage({
      config: config(),
      result: result(),
      usage: undefined,
      protocol: "openai",
    });

    expect(posted).toMatchObject({
      actual_forwarded_input_tokens: 1_500,
      forwarded_input_source: "estimated",
    });
  });

  it("preserves provider input usage and provenance when present", async () => {
    let posted: Record<string, unknown> | undefined;
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    vi.stubGlobal("fetch", vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      posted = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response(JSON.stringify({ code: 0, data: summary() }), { status: 200 });
    }));

    await reportOffloadMainUsage({
      config: config(),
      result: result(),
      usage: { prompt_tokens: 1_620, completion_tokens: 80 },
      protocol: "openai",
    });

    expect(posted).toMatchObject({
      actual_forwarded_input_tokens: 1_620,
      actual_output_tokens: 80,
      forwarded_input_source: "provider",
    });
  });
});
