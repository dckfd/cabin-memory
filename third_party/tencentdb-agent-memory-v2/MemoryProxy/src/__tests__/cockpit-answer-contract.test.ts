import { describe, expect, it } from "vitest";

import {
  rewriteOpenAiJsonAnswer,
  rewriteOpenAiSseAnswer,
  safeCockpitFallback,
  validateCockpitAnswer,
  type CockpitAnswerContract,
} from "../injection/cockpit-answer-contract.js";

function contract(overrides: Partial<CockpitAnswerContract> = {}): CockpitAnswerContract {
  return {
    version: 1,
    enforce: true,
    language: "zh",
    sufficient: true,
    risks: ["aggregation-frequency"],
    requiredFacts: [{ value: "星港充电站" }, { value: "2" }],
    requiredDateLabels: [],
    fallbackAnswer: "星港充电站，共2次。",
    ...overrides,
  };
}

describe("cockpit response-side answer contract", () => {
  it("requires every deterministic fact and rejects false abstention", () => {
    expect(validateCockpitAnswer(contract(), "星港充电站，共2次。").valid).toBe(true);
    expect(validateCockpitAnswer(contract(), "星港充电站。").failures).toContain("missing-value:2");
    expect(validateCockpitAnswer(contract(), "无法从历史记录确定。").failures)
      .toContain("sufficient-evidence-but-abstained");
  });

  it("requires a scoped abstention when the evidence gate is closed", () => {
    const insufficient = contract({ sufficient: false, requiredFacts: [], fallbackAnswer: undefined });
    expect(validateCockpitAnswer(insufficient, "现有历史没有说明该字段，无法确定。").valid).toBe(true);
    expect(validateCockpitAnswer(insufficient, "她就读海湾中学。").failures)
      .toContain("insufficient-evidence-without-abstention");
    expect(safeCockpitFallback(insufficient)).toContain("证据不足");
  });

  it("preserves ownership, date coverage, cancellation, and blocks internals/tools", () => {
    const scoped = contract({
      risks: ["multi-time-comparison", "cross-session-synthesis", "latest-final-update"],
      requiredFacts: [
        { label: "驾驶员林舟", value: "22度" },
        { label: "副驾小满", value: "24℃" },
      ],
      requiredDateLabels: ["8月12日", "8月18日"],
      requiredRelation: "cancelled",
    });
    const complete = "8月12日驾驶员林舟为22℃；8月18日副驾小满为24°C；旧设置已取消。";
    expect(validateCockpitAnswer(scoped, complete).valid).toBe(true);
    expect(validateCockpitAnswer(scoped, "根据 evidence #2，旧设置已取消。", true).failures)
      .toEqual(expect.arrayContaining(["tool-call-not-final-answer", "internal-label-leak"]));
  });

  it("rewrites non-streaming and streaming OpenAI responses without tool calls", () => {
    const json = JSON.stringify({
      id: "chat-1",
      choices: [{ message: { role: "assistant", content: "wrong", tool_calls: [{ id: "x" }] } }],
      usage: { total_tokens: 9 },
    });
    const rewrittenJson = JSON.parse(rewriteOpenAiJsonAnswer(json, "安全结论")!);
    expect(rewrittenJson.choices[0].message.content).toBe("安全结论");
    expect(rewrittenJson.choices[0].message.tool_calls).toBeUndefined();
    expect(rewrittenJson.usage.total_tokens).toBe(9);

    const sse = [
      'data: {"id":"chat-1","model":"m","choices":[{"delta":{"content":"wrong"},"finish_reason":null}]}',
      'data: {"id":"chat-1","model":"m","choices":[],"usage":{"total_tokens":9}}',
      "data: [DONE]",
      "",
    ].join("\n\n");
    const rewrittenSse = rewriteOpenAiSseAnswer(sse, "安全结论");
    expect(rewrittenSse).toContain('"content":"安全结论"');
    expect(rewrittenSse).not.toContain("wrong");
    expect(rewrittenSse).toContain('"total_tokens":9');
    expect(rewrittenSse).toContain("data: [DONE]");
  });
});
