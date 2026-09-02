import { describe, expect, it } from "vitest";
import {
  applyDeepSeekThinkingContract,
  isOfficialDeepSeekRequest,
} from "./deepseek-request-contract.js";

describe("DeepSeek request contract", () => {
  it("only identifies official DeepSeek model requests", () => {
    expect(isOfficialDeepSeekRequest("https://api.deepseek.com", "deepseek-v4-flash")).toBe(true);
    expect(isOfficialDeepSeekRequest("https://api.deepseek.com/v1", "deepseek-v4-pro")).toBe(true);
    expect(isOfficialDeepSeekRequest("https://proxy.example.com/v1", "deepseek-v4-flash")).toBe(false);
    expect(isOfficialDeepSeekRequest("https://api.deepseek.com", "qwen3.5-35b-a3b")).toBe(false);
    expect(isOfficialDeepSeekRequest("not-a-url", "deepseek-v4-flash")).toBe(false);
  });

  it("preserves the provider-default request byte-for-byte", () => {
    const body = '{"model":"deepseek-v4-flash","messages":[]}';
    expect(applyDeepSeekThinkingContract(body, "provider-default")).toBe(body);
  });

  it("disables thinking without changing the semantic request", () => {
    const body = JSON.stringify({
      model: "deepseek-v4-flash",
      messages: [{ role: "user", content: "提取记忆" }],
      max_tokens: 2048,
      reasoning_effort: "high",
    });
    const rewritten = JSON.parse(applyDeepSeekThinkingContract(body, "disabled"));

    expect(rewritten).toEqual({
      model: "deepseek-v4-flash",
      messages: [{ role: "user", content: "提取记忆" }],
      max_tokens: 2048,
      thinking: { type: "disabled" },
    });
  });

  it("enables thinking while preserving an explicit effort", () => {
    const body = JSON.stringify({
      model: "deepseek-v4-pro",
      messages: [],
      reasoning_effort: "medium",
    });
    const rewritten = JSON.parse(applyDeepSeekThinkingContract(body, "enabled"));

    expect(rewritten.thinking).toEqual({ type: "enabled" });
    expect(rewritten.reasoning_effort).toBe("medium");
  });

  it("rejects malformed or non-object request bodies", () => {
    expect(() => applyDeepSeekThinkingContract("not-json", "disabled"))
      .toThrow("requires a JSON request body");
    expect(() => applyDeepSeekThinkingContract("[]", "disabled"))
      .toThrow("requires a JSON object request body");
  });
});
