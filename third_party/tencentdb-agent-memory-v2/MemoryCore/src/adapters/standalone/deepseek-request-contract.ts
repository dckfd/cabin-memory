/**
 * Provider-bound request contract for official DeepSeek chat completions.
 *
 * DeepSeek V4 enables thinking by default. Memory construction is a
 * deterministic transformation task, so callers may explicitly disable
 * thinking without changing the default behavior of ordinary chat/answer
 * requests. The helper is intentionally narrow: it only rewrites requests to
 * the official DeepSeek endpoint for a DeepSeek model.
 */

export type LLMThinkingMode = "provider-default" | "disabled" | "enabled";

export function isOfficialDeepSeekRequest(baseUrl: string, model: string): boolean {
  if (!model.toLowerCase().startsWith("deepseek-")) return false;
  try {
    return new URL(baseUrl).hostname.toLowerCase() === "api.deepseek.com";
  } catch {
    return false;
  }
}

export function applyDeepSeekThinkingContract(
  body: string,
  mode: LLMThinkingMode,
): string {
  if (mode === "provider-default") return body;

  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    throw new Error("DeepSeek request contract requires a JSON request body");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("DeepSeek request contract requires a JSON object request body");
  }

  const request = { ...(parsed as Record<string, unknown>) };
  request.thinking = { type: mode };
  if (mode === "disabled") {
    // Avoid sending a contradictory effort control beside an explicit
    // non-thinking request. The official API treats `thinking` as the mode
    // switch and `reasoning_effort` as an enabled-mode control.
    delete request.reasoning_effort;
  }
  return JSON.stringify(request);
}
