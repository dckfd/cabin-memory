/**
 * Durable per-session token ledger for the offload pipeline.
 *
 * Build calls (L1/L1.5/L2) use provider-reported usage when available and a
 * CJK-aware estimate otherwise. Main-model records keep the provider's actual
 * forwarded input separate from the estimated counterfactual saving: the
 * uncompressed request is intentionally never sent to the provider.
 */
import type { StorageAdapter } from "../core/storage/adapter.js";

export type OffloadBuildStage = "l1" | "l15" | "l2";

export interface NormalizedTokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  source: "provider" | "estimated";
}

export interface BuildTokenRecord {
  kind: "build";
  record_id: string;
  timestamp: string;
  session_id: string;
  stage: OffloadBuildStage;
  model: string;
  duration_ms: number;
  usage: NormalizedTokenUsage;
}

export interface MainTokenRecord {
  kind: "main";
  record_id: string;
  timestamp: string;
  session_id: string;
  protocol: "openai" | "anthropic";
  applied: boolean;
  before_estimated_tokens: number;
  forwarded_estimated_tokens: number;
  estimated_gross_saved_tokens: number;
  actual_forwarded_input_tokens: number;
  actual_output_tokens: number;
  /** Missing only on records written before measurement provenance was added. */
  forwarded_input_source?: "provider" | "estimated";
  mmd_injected: number;
  gate?: Record<string, unknown>;
}

export interface TokenLedgerSummary {
  session_id: string;
  build: {
    calls: number;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    provider_measured_calls: number;
    estimated_calls: number;
    by_stage: Record<OffloadBuildStage, {
      calls: number;
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
    }>;
  };
  main: {
    calls: number;
    applied_calls: number;
    actual_forwarded_input_tokens: number;
    actual_output_tokens: number;
    estimated_counterfactual_input_tokens: number;
    estimated_gross_saved_tokens: number;
    provider_measured_calls: number;
    estimated_calls: number;
  };
  net: {
    estimated_net_saved_tokens: number;
    estimated_net_savings_ratio: number;
    break_even_reuse_calls: number | null;
  };
}

const LEDGER_DIR = "token-ledger/";

function finiteInt(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.round(value));
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.max(0, Math.round(parsed));
  }
  return 0;
}

function getRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

/** Normalize OpenAI, Anthropic, DeepSeek and common gateway usage shapes. */
export function normalizeTokenUsage(
  rawUsage: unknown,
  estimatedInput: number,
  estimatedOutput: number,
): NormalizedTokenUsage {
  const raw = getRecord(rawUsage);
  if (!raw) {
    const input = finiteInt(estimatedInput);
    const output = finiteInt(estimatedOutput);
    return {
      input_tokens: input,
      output_tokens: output,
      total_tokens: input + output,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      source: "estimated",
    };
  }

  const details = getRecord(raw.prompt_tokens_details);
  const cacheCreation = getRecord(raw.cache_creation);
  const cacheRead = finiteInt(raw.cache_read_input_tokens)
    || finiteInt(raw.prompt_cache_hit_tokens)
    || finiteInt(details?.cached_tokens);
  const cacheWrite = finiteInt(raw.cache_creation_input_tokens)
    || finiteInt(raw.prompt_cache_write_tokens)
    || finiteInt(cacheCreation?.ephemeral_5m_input_tokens)
    + finiteInt(cacheCreation?.ephemeral_1h_input_tokens);
  const explicitPrompt = finiteInt(raw.prompt_tokens);
  const nonCachedInput = finiteInt(raw.input_tokens);
  const input = explicitPrompt || (nonCachedInput + cacheRead + cacheWrite);
  const output = finiteInt(raw.completion_tokens) || finiteInt(raw.output_tokens);
  const explicitTotal = finiteInt(raw.total_tokens);

  if (input <= 0 && output <= 0 && explicitTotal <= 0) {
    return normalizeTokenUsage(undefined, estimatedInput, estimatedOutput);
  }

  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: explicitTotal || input + output,
    cache_read_tokens: cacheRead,
    cache_write_tokens: cacheWrite,
    source: "provider",
  };
}

export function estimateLedgerTextTokens(text: string): number {
  let cjk = 0;
  for (const char of text) {
    const code = char.codePointAt(0) ?? 0;
    if (
      (code >= 0x4e00 && code <= 0x9fff) ||
      (code >= 0x3400 && code <= 0x4dbf) ||
      (code >= 0xf900 && code <= 0xfaff)
    ) cjk += 1;
  }
  return Math.max(1, Math.ceil(cjk / 1.7 + Math.max(0, text.length - cjk) / 4));
}

export function estimateLedgerMessagesTokens(
  messages: Array<{ role: string; content: string }>,
): number {
  return messages.reduce(
    (total, message) => total + estimateLedgerTextTokens(message.role) + estimateLedgerTextTokens(message.content) + 4,
    0,
  );
}

function safeRecordId(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 240);
}

export async function writeBuildTokenRecord(
  storage: StorageAdapter,
  basePath: string,
  record: BuildTokenRecord,
): Promise<void> {
  await storage.writeFile(
    `${basePath}/${LEDGER_DIR}build-${safeRecordId(record.record_id)}.json`,
    JSON.stringify(record),
  );
}

export async function writeMainTokenRecord(
  storage: StorageAdapter,
  basePath: string,
  record: MainTokenRecord,
): Promise<void> {
  await storage.writeFile(
    `${basePath}/${LEDGER_DIR}main-${safeRecordId(record.record_id)}.json`,
    JSON.stringify(record),
  );
}

function emptyStage() {
  return { calls: 0, input_tokens: 0, output_tokens: 0, total_tokens: 0 };
}

export async function summarizeTokenLedger(
  storage: StorageAdapter,
  basePath: string,
  sessionId: string,
): Promise<TokenLedgerSummary> {
  const prefix = `${basePath}/${LEDGER_DIR}`;
  const names = await storage.readdirNames(prefix, ".json");
  const records = await Promise.all(names.map(async (name) => {
    const raw = await storage.readFile(`${prefix}${name}`);
    if (!raw) return undefined;
    try {
      return JSON.parse(raw) as BuildTokenRecord | MainTokenRecord;
    } catch {
      return undefined;
    }
  }));

  const byStage: TokenLedgerSummary["build"]["by_stage"] = {
    l1: emptyStage(),
    l15: emptyStage(),
    l2: emptyStage(),
  };
  const summary: TokenLedgerSummary = {
    session_id: sessionId,
    build: {
      calls: 0,
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      provider_measured_calls: 0,
      estimated_calls: 0,
      by_stage: byStage,
    },
    main: {
      calls: 0,
      applied_calls: 0,
      actual_forwarded_input_tokens: 0,
      actual_output_tokens: 0,
      estimated_counterfactual_input_tokens: 0,
      estimated_gross_saved_tokens: 0,
      provider_measured_calls: 0,
      estimated_calls: 0,
    },
    net: {
      estimated_net_saved_tokens: 0,
      estimated_net_savings_ratio: 0,
      break_even_reuse_calls: null,
    },
  };

  for (const record of records) {
    if (!record) continue;
    if (record.kind === "build") {
      const usage = record.usage;
      summary.build.calls += 1;
      summary.build.input_tokens += usage.input_tokens;
      summary.build.output_tokens += usage.output_tokens;
      summary.build.total_tokens += usage.total_tokens;
      if (usage.source === "provider") summary.build.provider_measured_calls += 1;
      else summary.build.estimated_calls += 1;
      const stage = byStage[record.stage];
      stage.calls += 1;
      stage.input_tokens += usage.input_tokens;
      stage.output_tokens += usage.output_tokens;
      stage.total_tokens += usage.total_tokens;
      continue;
    }

    summary.main.calls += 1;
    if (record.applied) summary.main.applied_calls += 1;
    summary.main.actual_forwarded_input_tokens += record.actual_forwarded_input_tokens;
    summary.main.actual_output_tokens += record.actual_output_tokens;
    summary.main.estimated_gross_saved_tokens += record.estimated_gross_saved_tokens;
    // Legacy records predate provenance. Their non-zero `actual_*` fields
    // were populated only from provider usage, so classify them as measured.
    if (record.forwarded_input_source === "estimated") summary.main.estimated_calls += 1;
    else summary.main.provider_measured_calls += 1;
    summary.main.estimated_counterfactual_input_tokens +=
      record.actual_forwarded_input_tokens > 0
        ? record.actual_forwarded_input_tokens + record.estimated_gross_saved_tokens
        : record.before_estimated_tokens;
  }

  const net = summary.main.estimated_gross_saved_tokens - summary.build.total_tokens;
  summary.net.estimated_net_saved_tokens = net;
  // Baseline is the uncompressed main-model input.  Build tokens belong only
  // to the optimized path, so adding them to the denominator would make the
  // reported saving ratio look artificially smaller.
  const counterfactualTotal = summary.main.estimated_counterfactual_input_tokens;
  summary.net.estimated_net_savings_ratio = counterfactualTotal > 0 ? net / counterfactualTotal : 0;
  const avgGrossPerApplied = summary.main.applied_calls > 0
    ? summary.main.estimated_gross_saved_tokens / summary.main.applied_calls
    : 0;
  summary.net.break_even_reuse_calls = avgGrossPerApplied > 0
    ? Math.ceil(summary.build.total_tokens / avgGrossPerApplied)
    : null;

  return summary;
}
