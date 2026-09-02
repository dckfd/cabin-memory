export interface CockpitAnswerFact {
  /** User-facing field/target. When set, both label and value must be present. */
  label?: string;
  value: string;
  /** Accepted surface variants, for example `twice` for numeric value `2`. */
  aliases?: string[];
}

export interface CockpitAnswerContract {
  version: 1;
  enforce: true;
  language: "zh" | "en";
  sufficient: boolean;
  risks: string[];
  requiredFacts: CockpitAnswerFact[];
  requiredDateLabels: string[];
  /** Cancellation/negation must remain explicit when this is set. */
  requiredRelation?: "cancelled" | "negated" | "updated";
  /** Deterministic, evidence-grounded output used when validation fails. */
  fallbackAnswer?: string;
}

export interface CockpitAnswerValidation {
  valid: boolean;
  failures: string[];
}

const INTERNAL_LEAK_PATTERNS = [
  /<\/?tdai_[^>]*>/iu,
  /\bsource\s*=/iu,
  /\bevidence\s*(?:#|no\.?|number)?\s*\d+/iu,
  /\bcitation\s*(?:#|\[)?\s*\d*/iu,
  /(?:证据|召回|来源)\s*(?:编号|序号|#)?\s*\d+/u,
  /\b(?:tdai_grounded|tdai_recalled|tdai_evidence_status)\b/iu,
];

const REFUSAL_PATTERNS = [
  /(?:无法|不能|不足以|无从)(?:从.{0,16})?(?:确定|判断|得知|回答)/u,
  /(?:没有|未)(?:提供|说明|记录|提及).{0,18}(?:无法|不能|不确定)/u,
  /\b(?:cannot|can't|unable to|not enough|insufficient)\b.{0,24}\b(?:determine|tell|know|answer|evidence|information)\b/iu,
  /\b(?:not specified|not stated|not recorded|unknown from (?:the )?(?:history|records))\b/iu,
];

const CANCELLED_PATTERNS = [
  /(?:已|被)?(?:取消|撤销|作废|不再有效|不生效)/u,
  /\b(?:cancelled|canceled|revoked|void|no longer (?:active|valid|applies?))\b/iu,
];

const NEGATED_PATTERNS = [
  /(?:不再|不是|无需|不要|禁止|未启用|未生效)/u,
  /\b(?:not|no longer|never|disabled|does not apply)\b/iu,
];

function normalized(value: string): string {
  return value
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/摄氏(?:度)?/gu, "c")
    .replace(/(?<=\d)度/gu, "c")
    .replace(/°c|℃/gu, "c")
    .replace(/[\s\p{P}\p{S}]+/gu, "");
}

function containsSurface(answer: string, surface: string): boolean {
  const haystack = normalized(answer);
  const needle = normalized(surface);
  if (!needle) return true;
  if (haystack.includes(needle)) return true;

  // Numeric counts should accept natural English count words without making
  // arbitrary semantic equivalences elsewhere in the answer.
  const countAliases: Record<string, string[]> = {
    "1": ["once", "one time", "一次"],
    "2": ["twice", "two times", "两次", "二次"],
    "3": ["thrice", "three times", "三次"],
  };
  return (countAliases[surface.trim()] ?? []).some((item) => haystack.includes(normalized(item)));
}

function hasAny(answer: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(answer));
}

export function validateCockpitAnswer(
  contract: CockpitAnswerContract,
  answer: string,
  hasToolCalls = false,
): CockpitAnswerValidation {
  const failures: string[] = [];
  const trimmed = answer.trim();

  if (!trimmed) failures.push("empty-answer");
  if (hasToolCalls) failures.push("tool-call-not-final-answer");
  if (hasAny(trimmed, INTERNAL_LEAK_PATTERNS)) failures.push("internal-label-leak");

  const refuses = hasAny(trimmed, REFUSAL_PATTERNS);
  if (!contract.sufficient) {
    if (!refuses) failures.push("insufficient-evidence-without-abstention");
    return { valid: failures.length === 0, failures };
  }
  if (refuses) failures.push("sufficient-evidence-but-abstained");

  for (const fact of contract.requiredFacts) {
    if (fact.label && !containsSurface(trimmed, fact.label)) {
      failures.push(`missing-label:${fact.label}`);
    }
    const variants = [fact.value, ...(fact.aliases ?? [])];
    if (!variants.some((surface) => containsSurface(trimmed, surface))) {
      failures.push(`missing-value:${fact.label ?? fact.value}`);
    }
  }

  for (const date of contract.requiredDateLabels) {
    if (!containsSurface(trimmed, date)) failures.push(`missing-date:${date}`);
  }

  if (contract.requiredRelation === "cancelled" && !hasAny(trimmed, CANCELLED_PATTERNS)) {
    failures.push("cancelled-relation-omitted");
  }
  if (contract.requiredRelation === "negated" && !hasAny(trimmed, NEGATED_PATTERNS)) {
    failures.push("negated-relation-omitted");
  }

  return { valid: failures.length === 0, failures };
}

export function safeCockpitFallback(contract: CockpitAnswerContract): string {
  if (contract.fallbackAnswer?.trim()) return contract.fallbackAnswer.trim();
  return contract.language === "zh"
    ? "现有历史对所问字段的证据不足，无法确定。"
    : "The available history is insufficient to determine the requested field.";
}

export function rewriteOpenAiJsonAnswer(respText: string, replacement: string): string | null {
  try {
    const payload = JSON.parse(respText) as Record<string, unknown>;
    const choices = payload.choices;
    if (!Array.isArray(choices) || choices.length === 0) return null;
    const first = choices[0] as Record<string, unknown>;
    const message = first.message;
    if (!message || typeof message !== "object") return null;
    const rewritten: Record<string, unknown> = {
      ...(message as Record<string, unknown>),
      content: replacement,
    };
    delete rewritten.tool_calls;
    first.message = rewritten;
    first.finish_reason = "stop";
    return JSON.stringify(payload);
  } catch {
    return null;
  }
}

/**
 * Replace assistant content while retaining the upstream's request/model ids
 * and usage event. The result remains valid OpenAI Chat Completions SSE.
 */
export function rewriteOpenAiSseAnswer(sseText: string, replacement: string): string {
  let envelope: Record<string, unknown> = {};
  let usageEvent: Record<string, unknown> | null = null;
  for (const line of sseText.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const raw = trimmed.slice(5).trim();
    if (!raw || raw === "[DONE]") continue;
    try {
      const event = JSON.parse(raw) as Record<string, unknown>;
      if (Object.keys(envelope).length === 0) {
        envelope = Object.fromEntries(
          ["id", "object", "created", "model", "system_fingerprint"]
            .filter((key) => event[key] !== undefined)
            .map((key) => [key, event[key]]),
        );
      }
      if (event.usage && typeof event.usage === "object") usageEvent = event;
    } catch {
      // Ignore malformed upstream events; the rewritten response is canonical.
    }
  }

  const contentEvent = {
    ...envelope,
    choices: [{ index: 0, delta: { role: "assistant", content: replacement }, finish_reason: null }],
  };
  const stopEvent = {
    ...envelope,
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  };
  const events: Record<string, unknown>[] = [contentEvent, stopEvent];
  if (usageEvent) events.push(usageEvent);
  return `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`;
}
