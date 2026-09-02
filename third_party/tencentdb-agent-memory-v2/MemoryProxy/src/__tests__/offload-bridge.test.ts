import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ProxyConfig } from "../types.js";
import {
  buildOffloadSessionId,
  evaluateOffloadCostGate,
  extractOffloadToolPairs,
  foldMmdIntoSystem,
  processOffloadRequest,
  resetOffloadDedupeForTests,
} from "../offload/bridge.js";

function makeConfig(overrides: Partial<ProxyConfig["tdai"]["offload"]> = {}): ProxyConfig {
  return {
    tdai: {
      enabled: true,
      endpoint: "http://memory-core:8420",
      apiKey: "test-core-key",
      serviceId: "default",
      memory: {
        enabled: true,
        inject: true,
        domainProfile: "smart-cockpit",
        timezone: "Asia/Shanghai",
        writeL0: true,
        recallL1: true,
        injectL2L3: true,
        l1Limit: 5,
        l2Limit: 3,
        timeoutMs: 3_000,
      },
      offload: {
        enabled: true,
        ingest: true,
        compact: true,
        contextWindowTokens: 8_192,
        minSavingsTokens: 1,
        recentMessagesLimit: 8,
        maxRecentMessageChars: 2_000,
        timeoutMs: 1_000,
        dedupeTtlSeconds: 21_600,
        costAware: {
          enabled: true,
          expectedReuseCalls: 4,
          minToolResultTokens: 512,
          minProjectedNetTokens: 256,
          minProjectedNetRatio: 0.1,
          promptOnlyEnabled: false,
          minPromptTokens: 24,
          activeTaskPromptBudget: 1,
          activeTaskTtlSeconds: 900,
          pressureRatio: 0.5,
        },
        ...overrides,
      },
    },
  } as ProxyConfig;
}

function ok(data: unknown): Response {
  return new Response(JSON.stringify({ code: 0, message: "ok", request_id: "test", data }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

describe("offload bridge", () => {
  beforeEach(() => {
    resetOffloadDedupeForTests();
    vi.spyOn(console, "info").mockImplementation(() => undefined);
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("extracts completed OpenAI tool pairs", () => {
    const pairs = extractOffloadToolPairs([
      { role: "user", content: "inspect the file" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call-1",
          type: "function",
          function: { name: "read_file", arguments: "{\"path\":\"/tmp/a\"}" },
        }],
      },
      { role: "tool", tool_call_id: "call-1", content: "file body", duration_ms: 12 },
    ], "openai", "2026-08-18T00:00:00.000Z");

    expect(pairs).toEqual([{
      tool_name: "read_file",
      tool_call_id: "call-1",
      params: { path: "/tmp/a" },
      result: "file body",
      timestamp: "2026-08-18T00:00:00.000Z",
      duration_ms: 12,
    }]);
  });

  it("extracts completed Anthropic tool pairs", () => {
    const pairs = extractOffloadToolPairs([
      { role: "user", content: [{ type: "text", text: "run diagnostics" }] },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "toolu-1", name: "exec", input: { cmd: "pwd" } }],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "toolu-1", content: "workspace" }],
      },
    ], "anthropic", "2026-08-18T00:00:00.000Z");

    expect(pairs).toEqual([{
      tool_name: "exec",
      tool_call_id: "toolu-1",
      params: { cmd: "pwd" },
      result: "workspace",
      timestamp: "2026-08-18T00:00:00.000Z",
    }]);
  });

  it("isolates the same conversation id by user namespace", () => {
    const common = { serviceId: "space-a", agentSource: "claude-code", sessionKey: "conv-1" };
    const a = buildOffloadSessionId({ ...common, userNamespace: "user-a" });
    const b = buildOffloadSessionId({ ...common, userNamespace: "user-b" });
    expect(a).not.toBe(b);
    expect(a).toMatch(/^proxy-[a-f0-9]{40}$/);
  });

  it("skips short low-value builds but admits reusable long tool output", () => {
    const config = makeConfig().tdai.offload;
    const small = evaluateOffloadCostGate({
      config,
      pairs: [{
        tool_name: "set_volume",
        tool_call_id: "small",
        params: { level: 5 },
        result: "ok",
        timestamp: "2026-08-18T00:00:00.000Z",
      }],
      recentMessages: [{ role: "user", content: "音量调到五" }],
      prompt: "继续",
      beforeTokens: 300,
      sessionQualified: false,
    });
    expect(small.allow_tools).toBe(false);
    expect(small.allow_prompt).toBe(false);
    expect(small.decision).toBe("accumulate");
    expect(small.estimated_build_tokens).toBeGreaterThan(0);
    expect(small.scheduled_build_tokens).toBe(0);
    expect(small.projected_net_saved_tokens).toBeLessThan(0);

    const large = evaluateOffloadCostGate({
      config,
      pairs: Array.from({ length: 4 }, (_, index) => ({
        tool_name: "exec",
        tool_call_id: `large-${index}`,
        params: { cmd: "diagnose" },
        result: "x".repeat(8_000),
        timestamp: "2026-08-18T00:00:00.000Z",
      })),
      recentMessages: [{ role: "user", content: "Diagnose and repair the failing build across all packages." }],
      prompt: "Continue the repair and preserve all important findings.",
      beforeTokens: 10_000,
      sessionQualified: false,
    });
    expect(large.allow_tools).toBe(true);
    expect(large.allow_prompt).toBe(true);
    expect(large.projected_net_saved_tokens).toBeGreaterThan(0);
    expect(large.scheduled_build_tokens).toBe(large.estimated_build_tokens);
    expect(large.reserved_lifecycle_tokens).toBeGreaterThan(0);
    expect(large.projected_total_cost_tokens).toBe(
      large.estimated_build_tokens + large.reserved_lifecycle_tokens,
    );
    expect(large.projected_net_saved_tokens).toBe(
      large.projected_gross_saved_tokens - large.projected_total_cost_tokens,
    );
  });

  it("defers prompt-only L1.5 until a task qualifies or context pressure requires it", () => {
    const config = makeConfig().tdai.offload;
    const longPrompt = "Investigate this multi-step request carefully and preserve the important findings. ".repeat(8);
    const deferred = evaluateOffloadCostGate({
      config,
      pairs: [],
      recentMessages: [],
      prompt: longPrompt,
      beforeTokens: 1_000,
      sessionQualified: false,
    });
    expect(deferred.allow_prompt).toBe(false);
    expect(deferred.decision).toBe("defer-prompt");

    const active = evaluateOffloadCostGate({
      config,
      pairs: [],
      recentMessages: [],
      prompt: "done",
      beforeTokens: 1_000,
      sessionQualified: true,
    });
    expect(active.allow_prompt).toBe(true);
    expect(active.decision).toBe("active-task");

    const pressure = evaluateOffloadCostGate({
      config,
      pairs: [],
      recentMessages: [],
      prompt: "continue",
      beforeTokens: 4_200,
      sessionQualified: false,
    });
    expect(pressure.allow_prompt).toBe(true);
    expect(pressure.decision).toBe("context-pressure");
  });

  it("keeps rejected tool results eligible and admits them after a later request makes the batch economical", async () => {
    const firstMessages = [
      { role: "user", content: "检查第一项" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "acc-1",
          type: "function",
          function: { name: "diagnose", arguments: "{}" },
        }],
      },
      { role: "tool", tool_call_id: "acc-1", content: "x".repeat(3_500) },
      { role: "user", content: "继续" },
    ];
    const secondMessages = [
      ...firstMessages.slice(0, -1),
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "acc-2",
          type: "function",
          function: { name: "diagnose", arguments: "{}" },
        }],
      },
      { role: "tool", tool_call_id: "acc-2", content: "y".repeat(12_000) },
      { role: "user", content: "继续" },
    ];
    const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      requests.push({ url: String(url), body });
      if (String(url).endsWith("/compact")) {
        return ok({ messages: body.messages ?? [], report: {} });
      }
      return ok({});
    }));

    const first = await processOffloadRequest({
      config: makeConfig(),
      protocol: "openai",
      body: { model: "test", messages: firstMessages },
      sourceMessages: firstMessages,
      sessionKey: "conv-accumulate",
      userNamespace: "user-a",
      agentSource: "codebuddy",
      spaceId: "space-a",
    });
    expect(first.gate?.decision).toBe("accumulate");
    expect(first.ingestedToolPairs).toBe(0);
    expect(requests.filter((request) => request.url.endsWith("/ingest"))).toHaveLength(0);

    const second = await processOffloadRequest({
      config: makeConfig(),
      protocol: "openai",
      body: { model: "test", messages: secondMessages },
      sourceMessages: secondMessages,
      sessionKey: "conv-accumulate",
      userNamespace: "user-a",
      agentSource: "codebuddy",
      spaceId: "space-a",
    });
    expect(second.gate?.decision).toBe("build");
    expect(second.ingestedToolPairs).toBe(2);
    const toolIngest = requests.find((request) => (
      request.url.endsWith("/ingest") && Array.isArray(request.body.tool_pairs)
      && request.body.tool_pairs.length > 0
    ));
    expect(toolIngest?.body.tool_pairs).toHaveLength(2);
  });

  it("bounds active-task L1.5 to one follow-up prompt per qualified tool batch", async () => {
    const initialMessages = [
      { role: "user", content: "diagnose the complete build log" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "budget-tool",
          type: "function",
          function: { name: "exec", arguments: "{}" },
        }],
      },
      { role: "tool", tool_call_id: "budget-tool", content: "x".repeat(20_000) },
      { role: "user", content: "continue" },
    ];
    const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      requests.push({ url: String(url), body });
      return ok(String(url).endsWith("/compact")
        ? { messages: body.messages ?? [], report: {} }
        : {});
    }));

    const run = (messages: unknown[]) => processOffloadRequest({
      config: makeConfig({ contextWindowTokens: 32_768 }),
      protocol: "openai",
      body: { model: "test", messages },
      sourceMessages: messages,
      sessionKey: "conv-budget",
      userNamespace: "user-a",
      agentSource: "codebuddy",
      spaceId: "space-a",
    });

    const initial = await run(initialMessages);
    expect(initial.gate?.decision).toBe("build");
    expect(initial.ingestedToolPairs).toBe(1);
    expect(initial.ingestedPrompt).toBe(true);

    const firstFollowUpMessages = [
      ...initialMessages,
      { role: "assistant", content: "diagnosis complete" },
      { role: "user", content: "what should I do next?" },
    ];
    const firstFollowUp = await run(firstFollowUpMessages);
    expect(firstFollowUp.gate?.decision).toBe("active-task");
    expect(firstFollowUp.ingestedPrompt).toBe(true);

    const secondFollowUp = await run([
      ...firstFollowUpMessages,
      { role: "assistant", content: "apply the fix" },
      { role: "user", content: "and after that?" },
    ]);
    expect(secondFollowUp.gate?.decision).toBe("defer-prompt");
    expect(secondFollowUp.ingestedPrompt).toBe(false);
    expect(requests.filter((request) => request.url.endsWith("/ingest"))).toHaveLength(3);
  });

  it("folds MMD into protocol-native system context", () => {
    const mmd = {
      role: "user",
      content: "<current_task_context>graph TD; A-->B</current_task_context>",
      _mmdContextMessage: "active",
      _mmdVersion: "v1",
    };

    const openai = foldMmdIntoSystem({ messages: [] }, [
      { role: "system", content: "base instructions" },
      mmd,
      { role: "user", content: "continue" },
    ], "openai");
    expect(openai.messages).toEqual([
      {
        role: "system",
        content: "base instructions\n\n<current_task_context>graph TD; A-->B</current_task_context>",
      },
      { role: "user", content: "continue" },
    ]);

    const cachedSystem = [{
      type: "text",
      text: "base instructions",
      cache_control: { type: "ephemeral" },
    }];
    const anthropic = foldMmdIntoSystem({ system: cachedSystem, messages: [] }, [
      mmd,
      { role: "user", content: [{ type: "text", text: "continue" }] },
    ], "anthropic");
    expect(anthropic.system).toEqual([
      cachedSystem[0],
      { type: "text", text: "<current_task_context>graph TD; A-->B</current_task_context>" },
    ]);
    expect(JSON.stringify(anthropic.messages)).not.toContain("current_task_context");
    expect(JSON.stringify(anthropic)).not.toContain("_mmdContextMessage");
  });

  it("ingests once, injects MMD, strips internal metadata, and applies only net savings", async () => {
    const hugeResult = "x".repeat(12_000);
    const sourceMessages = [
      { role: "user", content: "fix the failing build" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call-build",
          type: "function",
          function: { name: "exec", arguments: "{\"cmd\":\"npm test\"}" },
        }],
      },
      { role: "tool", tool_call_id: "call-build", content: hugeResult },
      { role: "user", content: "continue" },
    ];
    const body = { model: "test-model", messages: sourceMessages };
    const requests: Array<{ url: string; body: Record<string, unknown>; headers: Headers }> = [];
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      const parsed = JSON.parse(String(init?.body)) as Record<string, unknown>;
      requests.push({ url: String(url), body: parsed, headers });
      if (String(url).endsWith("/compact")) {
        return ok({
          messages: [
            sourceMessages[0],
            sourceMessages[1],
            {
              role: "tool",
              tool_call_id: "call-build",
              content: "[Offloaded Tool Result]\nSummary: tests failed in one module",
              _offloaded: true,
            },
            {
              role: "user",
              content: "<current_task_context>\n```mermaid\ngraph TD; A-->B\n```\n</current_task_context>",
              _mmdContextMessage: "active",
              _mmdVersion: "abc123",
              _mmdFilename: "001-build.mmd",
            },
            sourceMessages[3],
          ],
          report: { mmdInjected: 1, mildReplacements: 1 },
        });
      }
      return ok({});
    });
    vi.stubGlobal("fetch", fetchMock);

    const first = await processOffloadRequest({
      config: makeConfig(),
      protocol: "openai",
      body,
      sourceMessages,
      sessionKey: "conv-1",
      userNamespace: "user-a",
      agentSource: "claude-code",
      spaceId: "space-a",
    });

    expect(first.applied).toBe(true);
    expect(first.savedTokens).toBeGreaterThan(2_000);
    expect(first.mmdInjected).toBe(1);
    expect(first.ingestedToolPairs).toBe(1);
    expect(first.ingestedPrompt).toBe(true);
    expect(JSON.stringify(first.body)).toContain("<current_task_context>");
    expect((first.body.messages as Array<Record<string, unknown>>)[0]).toMatchObject({
      role: "system",
    });
    expect((first.body.messages as Array<Record<string, unknown>>).filter(
      (message) => message.role === "user" && JSON.stringify(message).includes("<current_task_context>"),
    )).toHaveLength(0);
    expect(JSON.stringify(first.body)).not.toContain("_mmdContextMessage");
    expect(JSON.stringify(first.body)).not.toContain("_offloaded");
    expect(requests.filter((request) => request.url.endsWith("/ingest"))).toHaveLength(2);
    expect(requests.filter((request) => request.url.endsWith("/compact"))).toHaveLength(1);
    expect(requests[0].headers.get("authorization")).toBe("Bearer test-core-key");
    expect(requests[0].headers.get("x-tdai-service-id")).toBe("space-a");

    const callsAfterFirst = requests.length;
    const second = await processOffloadRequest({
      config: makeConfig(),
      protocol: "openai",
      body,
      sourceMessages,
      sessionKey: "conv-1",
      userNamespace: "user-a",
      agentSource: "claude-code",
      spaceId: "space-a",
    });
    expect(second.applied).toBe(true);
    expect(second.ingestedToolPairs).toBe(0);
    expect(second.ingestedPrompt).toBe(false);
    expect(requests).toHaveLength(callsAfterFirst + 1);
    expect(requests.at(-1)?.url).toMatch(/\/compact$/);

    // A tool loop can append assistant text after the same human prompt. That
    // must not schedule L1.5 again for the already-seen user turn.
    const third = await processOffloadRequest({
      config: makeConfig(),
      protocol: "openai",
      body: { ...body, messages: [...sourceMessages, { role: "assistant", content: "working" }] },
      sourceMessages: [...sourceMessages, { role: "assistant", content: "working" }],
      sessionKey: "conv-1",
      userNamespace: "user-a",
      agentSource: "claude-code",
      spaceId: "space-a",
    });
    expect(third.ingestedToolPairs).toBe(0);
    expect(third.ingestedPrompt).toBe(false);
    expect(requests.filter((request) => request.url.endsWith("/ingest"))).toHaveLength(2);
  });

  it("keeps the original request when MMD would increase tokens", async () => {
    const body = { model: "test-model", messages: [{ role: "user", content: "hi" }] };
    vi.stubGlobal("fetch", vi.fn(async (url: string | URL | Request) => {
      if (String(url).endsWith("/compact")) {
        return ok({
          messages: [
            { role: "user", content: "a very long MMD that is larger than the original request", _mmdContextMessage: "active" },
            { role: "user", content: "hi" },
          ],
          report: { mmdInjected: 1 },
        });
      }
      return ok({});
    }));

    const result = await processOffloadRequest({
      config: makeConfig(),
      protocol: "anthropic",
      body,
      sourceMessages: body.messages,
      sessionKey: "conv-small",
      userNamespace: "user-a",
      agentSource: "claude-code",
    });

    expect(result.applied).toBe(false);
    expect(result.body).toBe(body);
    expect(result.savedTokens).toBeLessThan(0);
  });

  it("adopts Core truncation even when the report has no truncation counter", async () => {
    const body = {
      model: "test-model",
      messages: [
        { role: "user", content: "inspect" },
        { role: "tool", tool_call_id: "raw-tool", content: "x".repeat(8_000) },
        { role: "user", content: "continue" },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(async (url: string | URL | Request) => {
      if (String(url).endsWith("/compact")) {
        return ok({
          messages: [
            body.messages[0],
            { role: "tool", tool_call_id: "raw-tool", content: "x".repeat(200) + "[truncated]" },
            body.messages[2],
          ],
          report: {
            mmdInjected: 0,
            mildReplacements: 0,
            aggressiveDeleted: 0,
            emergencyDeleted: 0,
          },
        });
      }
      return ok({});
    }));

    const result = await processOffloadRequest({
      config: makeConfig({ ingest: false }),
      protocol: "openai",
      body,
      sourceMessages: body.messages,
      sessionKey: "conv-truncate",
      userNamespace: "user-a",
      agentSource: "claude-code",
    });

    expect(result.applied).toBe(true);
    expect(result.savedTokens).toBeGreaterThan(1_000);
  });

  it("fails open when MemoryCore is unavailable", async () => {
    const body = { model: "test-model", messages: [{ role: "user", content: "continue" }] };
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new Error("connection refused");
    }));

    const result = await processOffloadRequest({
      config: makeConfig(),
      protocol: "openai",
      body,
      sourceMessages: body.messages,
      sessionKey: "conv-fail-open",
      userNamespace: "user-a",
      agentSource: "claude-code",
    });

    expect(result.applied).toBe(false);
    expect(result.body).toBe(body);
  });
});
