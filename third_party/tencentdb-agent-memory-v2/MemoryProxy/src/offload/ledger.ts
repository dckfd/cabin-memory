/** Report main-model token economics to MemoryCore's durable offload ledger. */
import type { ProxyConfig } from "../types.js";
import { getActualInputTokens } from "../rate-limit/usage.js";
import type { OffloadProcessResult, OffloadProtocol } from "./bridge.js";

interface LedgerEnvelope<T> {
  code?: number;
  data?: T;
}

export interface OffloadLedgerSummary {
  session_id: string;
  build: {
    calls: number;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    provider_measured_calls: number;
    estimated_calls: number;
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

function numberField(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, Math.round(value));
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : 0;
  }
  return 0;
}

function actualOutputTokens(
  usage: Record<string, unknown> | null | undefined,
  protocol: OffloadProtocol,
): number {
  if (!usage) return 0;
  return protocol === "openai"
    ? numberField(usage.completion_tokens)
    : numberField(usage.output_tokens);
}

/**
 * Fail-open accounting call. A ledger outage must never affect the answer
 * already returned by the upstream model.
 */
export async function reportOffloadMainUsage(input: {
  config: ProxyConfig;
  result: OffloadProcessResult | null | undefined;
  usage: Record<string, unknown> | null | undefined;
  protocol: OffloadProtocol;
}): Promise<OffloadLedgerSummary | undefined> {
  const { config, result, usage, protocol } = input;
  if (!result?.offloadSessionId || !result.offloadServiceId || !result.requestId) return undefined;

  const providerInputTokens = getActualInputTokens(usage, protocol);
  // Streaming providers are not required to emit usage. Preserve a useful
  // ledger value while making its provenance explicit instead of writing a
  // misleading zero into an `actual_*` field.
  const forwardedInputTokens = providerInputTokens > 0
    ? providerInputTokens
    : result.afterTokens;
  const forwardedInputSource = providerInputTokens > 0 ? "provider" : "estimated";

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.tdai.offload.timeoutMs);
  try {
    const response = await fetch(
      `${config.tdai.endpoint.replace(/\/+$/, "")}/v2/offload/usage/report`,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${config.tdai.apiKey}`,
          "x-tdai-service-id": result.offloadServiceId,
        },
        body: JSON.stringify({
          session_id: result.offloadSessionId,
          request_id: result.requestId,
          timestamp: new Date().toISOString(),
          protocol,
          applied: result.applied,
          before_estimated_tokens: result.beforeTokens,
          forwarded_estimated_tokens: result.afterTokens,
          estimated_gross_saved_tokens: result.applied ? Math.max(0, result.savedTokens) : 0,
          actual_forwarded_input_tokens: forwardedInputTokens,
          actual_output_tokens: actualOutputTokens(usage, protocol),
          forwarded_input_source: forwardedInputSource,
          mmd_injected: result.mmdInjected,
          ...(result.gate ? { gate: result.gate } : {}),
        }),
        signal: controller.signal,
      },
    );
    const parsed = JSON.parse(await response.text()) as LedgerEnvelope<OffloadLedgerSummary>;
    if (!response.ok || parsed.code !== 0 || !parsed.data) {
      throw new Error(`HTTP ${response.status}, code=${parsed.code ?? "?"}`);
    }
    const summary = parsed.data;
    console.info(
      `[offload-ledger] session=${summary.session_id} gross=${summary.main.estimated_gross_saved_tokens} ` +
      `build=${summary.build.total_tokens} net=${summary.net.estimated_net_saved_tokens} ` +
      `ratio=${(summary.net.estimated_net_savings_ratio * 100).toFixed(2)}% ` +
      `break_even=${summary.net.break_even_reuse_calls ?? "n/a"}`,
    );
    return summary;
  } catch (error) {
    console.warn(
      `[offload-ledger] main usage report failed session=${result.offloadSessionId}: ` +
      `${error instanceof Error ? error.message : String(error)}`,
    );
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}
