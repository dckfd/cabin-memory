/**
 * MemoryCore offload bridge.
 *
 * The proxy keeps the client-facing conversation untouched, but sends completed
 * tool calls and new human prompts to MemoryCore's asynchronous offload pipeline.
 * On every following main-model request it asks MemoryCore to compact the request.
 * MemoryCore's marked MMD messages are then folded into the protocol-native
 * system context (OpenAI system message / Anthropic top-level system field), and
 * the returned body is adopted only when it is strictly smaller. This makes MMD
 * injection transparent to OpenAI/Anthropic clients and fail-open by design.
 */

import { createHash } from "node:crypto";
import type { ProxyConfig, TdaiConfig } from "../types.js";

export type OffloadProtocol = "openai" | "anthropic";

type JsonRecord = Record<string, unknown>;

export interface OffloadToolPair {
  tool_name: string;
  tool_call_id: string;
  params: unknown;
  result: unknown;
  error?: string;
  timestamp: string;
  duration_ms?: number;
}

export interface OffloadProcessInput {
  config: ProxyConfig;
  protocol: OffloadProtocol;
  /** Body after the normal injection pipeline; this is what will be forwarded. */
  body: Record<string, unknown>;
  /** Snapshot taken before normal injection, used only for offload extraction. */
  sourceMessages: unknown[];
  sessionKey: string;
  userNamespace: string;
  agentSource: string;
  spaceId?: string;
  /** Proxy request/trace id used as the durable ledger id. */
  requestId?: string;
}

export interface OffloadGateTelemetry {
  decision: string;
  reason: string;
  raw_tool_tokens: number;
  /** Candidate L1/L1.5/L2 cycle cost used by the admission decision. */
  estimated_build_tokens: number;
  /** Cost of stages actually scheduled by this request after gating. */
  scheduled_build_tokens: number;
  projected_gross_saved_tokens: number;
  projected_net_saved_tokens: number;
  projected_net_savings_ratio: number;
  expected_reuse_calls: number;
  allow_tools: boolean;
  allow_prompt: boolean;
  prompt_tokens: number;
  prompt_only_enabled: boolean;
  reserved_lifecycle_tokens: number;
  projected_total_cost_tokens: number;
  active_task_prompt_budget: number;
  context_ratio: number;
}

export interface OffloadProcessResult {
  body: Record<string, unknown>;
  applied: boolean;
  beforeTokens: number;
  afterTokens: number;
  savedTokens: number;
  mmdInjected: number;
  ingestedToolPairs: number;
  ingestedPrompt: boolean;
  offloadSessionId?: string;
  offloadServiceId?: string;
  requestId?: string;
  gate?: OffloadGateTelemetry;
}

interface ToolCallDefinition {
  name: string;
  params: unknown;
  timestamp?: string;
}

interface RecentMessage {
  role: "user" | "assistant";
  content: string;
}

interface OffloadEnvelope<T> {
  code?: number;
  message?: string;
  data?: T;
}

interface CompactData {
  messages?: unknown[];
  report?: {
    mmdInjected?: number;
    mildReplacements?: number;
    fastPathReplaced?: number;
    fastPathDeleted?: number;
    aggressiveDeleted?: number;
    emergencyDeleted?: number;
    [key: string]: unknown;
  };
}

const INTERNAL_MESSAGE_KEYS = new Set([
  "_offloaded",
  "_mmdContextMessage",
  "_mmdInjection",
  "_mmdVersion",
  "_mmdFilename",
  "_contextOffloadProcessed",
  "_cachedTokens",
  "_tokenCount",
]);

// Process-local hot-path dedupe. MemoryCore performs durable dedupe after L1,
// while this cache prevents repeated tool-loop requests from duplicating items
// in pending.jsonl before L1 gets a chance to run.
const dedupeReservations = new Map<string, number>();
interface QualifiedSessionState {
  expiresAt: number;
  remainingPromptChecks: number;
}
const qualifiedSessions = new Map<string, QualifiedSessionState>();
const MAX_DEDUPE_ENTRIES = 50_000;
let dedupeOps = 0;

function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hash(value: string, length = 32): string {
  return createHash("sha256").update(value).digest("hex").slice(0, length);
}

/**
 * Build a path-safe, user-isolated MemoryCore session id.
 * Raw user/session values never enter storage paths or logs.
 */
export function buildOffloadSessionId(input: {
  serviceId: string;
  userNamespace: string;
  agentSource: string;
  sessionKey: string;
}): string {
  const digest = hash(
    [input.serviceId, input.userNamespace, input.agentSource, input.sessionKey].join("\u0000"),
    40,
  );
  return `proxy-${digest}`;
}

/** Make a stable JSON-only snapshot before another injector mutates messages. */
export function snapshotMessagesForOffload(messages: unknown[]): unknown[] {
  try {
    return structuredClone(messages);
  } catch {
    // Request bodies came from JSON, so this is a safe compatibility fallback
    // for runtimes without structuredClone or unusual extension objects.
    return JSON.parse(JSON.stringify(messages)) as unknown[];
  }
}

function parseArguments(value: unknown): unknown {
  if (typeof value !== "string") return value ?? {};
  const trimmed = value.trim();
  if (!trimmed) return {};
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return value;
  }
}

function normalizeTimestamp(value: unknown, fallback: string): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    const millis = value < 10_000_000_000 ? value * 1000 : value;
    const date = new Date(millis);
    if (!Number.isNaN(date.getTime())) return date.toISOString();
  }
  if (typeof value === "string" && value.trim()) {
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) return date.toISOString();
  }
  return fallback;
}

function extractToolDefinitions(messages: unknown[]): Map<string, ToolCallDefinition> {
  const calls = new Map<string, ToolCallDefinition>();

  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const inner = raw.type === "message" && isRecord(raw.message) ? raw.message : raw;
    if (inner.role !== "assistant") continue;
    const timestamp = normalizeTimestamp(inner.timestamp ?? raw.timestamp, "");

    if (Array.isArray(inner.tool_calls)) {
      for (const rawCall of inner.tool_calls) {
        if (!isRecord(rawCall) || typeof rawCall.id !== "string" || !rawCall.id) continue;
        const fn = isRecord(rawCall.function) ? rawCall.function : undefined;
        calls.set(rawCall.id, {
          name: typeof fn?.name === "string" ? fn.name : "unknown",
          params: parseArguments(fn?.arguments),
          ...(timestamp ? { timestamp } : {}),
        });
      }
    }

    if (!Array.isArray(inner.content)) continue;
    for (const rawBlock of inner.content) {
      if (!isRecord(rawBlock)) continue;
      if (rawBlock.type !== "tool_use" && rawBlock.type !== "toolCall") continue;
      if (typeof rawBlock.id !== "string" || !rawBlock.id) continue;
      calls.set(rawBlock.id, {
        name: typeof rawBlock.name === "string" ? rawBlock.name : "unknown",
        params: rawBlock.type === "toolCall"
          ? parseArguments(rawBlock.arguments)
          : (rawBlock.input ?? {}),
        ...(timestamp ? { timestamp } : {}),
      });
    }
  }

  return calls;
}

function contentToResult(content: unknown): unknown {
  if (!Array.isArray(content)) return content ?? "";
  if (content.length === 1 && isRecord(content[0]) && content[0].type === "text") {
    return typeof content[0].text === "string" ? content[0].text : content;
  }
  return content;
}

function errorFromResult(result: unknown, isError: unknown): string | undefined {
  if (isError !== true) return undefined;
  if (typeof result === "string") return result.slice(0, 2_000);
  try {
    return JSON.stringify(result).slice(0, 2_000);
  } catch {
    return "tool execution failed";
  }
}

/** Extract completed pairs from OpenAI, Anthropic, or OpenClaw-compatible history. */
export function extractOffloadToolPairs(
  messages: unknown[],
  _protocol: OffloadProtocol,
  now = new Date().toISOString(),
): OffloadToolPair[] {
  const calls = extractToolDefinitions(messages);
  const pairs = new Map<string, OffloadToolPair>();

  const addResult = (
    id: unknown,
    result: unknown,
    meta: JsonRecord,
    fallbackName?: unknown,
    isError?: unknown,
  ): void => {
    if (typeof id !== "string" || !id || pairs.has(id)) return;
    const call = calls.get(id);
    const normalizedResult = contentToResult(result);
    const duration = meta.duration_ms ?? meta.durationMs;
    pairs.set(id, {
      tool_name: call?.name ?? (typeof fallbackName === "string" ? fallbackName : "unknown"),
      tool_call_id: id,
      params: call?.params ?? {},
      result: normalizedResult,
      ...(errorFromResult(normalizedResult, isError ?? meta.is_error ?? meta.isError)
        ? { error: errorFromResult(normalizedResult, isError ?? meta.is_error ?? meta.isError) }
        : {}),
      timestamp: normalizeTimestamp(meta.timestamp, call?.timestamp || now),
      ...(typeof duration === "number" && Number.isFinite(duration)
        ? { duration_ms: duration }
        : {}),
    });
  };

  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const inner = raw.type === "message" && isRecord(raw.message) ? raw.message : raw;
    const role = inner.role ?? raw.role ?? raw.type;

    if (role === "tool" || role === "toolResult" || role === "tool_result") {
      addResult(
        inner.tool_call_id ?? inner.toolCallId ?? inner.tool_use_id ?? inner.id,
        inner.content ?? inner.result ?? inner.output,
        inner,
        inner.name ?? inner.toolName,
      );
    }

    if (role !== "user" || !Array.isArray(inner.content)) continue;
    for (const rawBlock of inner.content) {
      if (!isRecord(rawBlock)) continue;
      if (rawBlock.type !== "tool_result" && rawBlock.type !== "toolResult") continue;
      addResult(
        rawBlock.tool_use_id ?? rawBlock.toolCallId ?? rawBlock.tool_call_id,
        rawBlock.content ?? rawBlock.result,
        rawBlock,
        rawBlock.name,
        rawBlock.is_error ?? rawBlock.isError,
      );
    }
  }

  return [...pairs.values()];
}

function messageText(content: unknown): string {
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const rawBlock of content) {
    if (!isRecord(rawBlock)) continue;
    if (rawBlock.type === "text" || rawBlock.type === "input_text" || rawBlock.type === "output_text") {
      if (typeof rawBlock.text === "string" && rawBlock.text.trim()) parts.push(rawBlock.text.trim());
    }
  }
  return parts.join("\n").trim();
}

function isInjectedContext(text: string): boolean {
  return text.includes("<current_task_context>") || text.includes("<history_task_context>");
}

export function extractRecentOffloadMessages(
  messages: unknown[],
  limit: number,
  maxChars: number,
): RecentMessage[] {
  const recent: RecentMessage[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const inner = raw.type === "message" && isRecord(raw.message) ? raw.message : raw;
    if (inner.role !== "user" && inner.role !== "assistant") continue;
    const text = messageText(inner.content);
    if (!text || isInjectedContext(text)) continue;
    recent.push({
      role: inner.role,
      content: text.length > maxChars ? `${text.slice(0, maxChars)}…` : text,
    });
  }
  return recent.slice(-Math.max(1, limit));
}

function latestPrompt(messages: unknown[]): {
  prompt?: string;
  humanTurns: number;
  latestUserIndex: number;
} {
  let prompt: string | undefined;
  let humanTurns = 0;
  let latestUserIndex = -1;
  for (let index = 0; index < messages.length; index++) {
    const raw = messages[index];
    if (!isRecord(raw)) continue;
    const inner = raw.type === "message" && isRecord(raw.message) ? raw.message : raw;
    if (inner.role !== "user") continue;
    const text = messageText(inner.content);
    if (!text || isInjectedContext(text)) continue;
    humanTurns += 1;
    prompt = text;
    latestUserIndex = index;
  }
  return { prompt, humanTurns, latestUserIndex };
}

function pruneDedupe(now: number): void {
  dedupeOps += 1;
  if (dedupeOps % 256 !== 0 && dedupeReservations.size <= MAX_DEDUPE_ENTRIES) return;
  for (const [key, expiresAt] of dedupeReservations) {
    if (expiresAt <= now) dedupeReservations.delete(key);
  }
  for (const [key, state] of qualifiedSessions) {
    if (state.expiresAt <= now) qualifiedSessions.delete(key);
  }
  if (dedupeReservations.size <= MAX_DEDUPE_ENTRIES) return;
  const overflow = dedupeReservations.size - MAX_DEDUPE_ENTRIES;
  let removed = 0;
  for (const key of dedupeReservations.keys()) {
    dedupeReservations.delete(key);
    removed += 1;
    if (removed >= overflow) break;
  }
}

function reserveDedupe(key: string, ttlSeconds: number): boolean {
  const now = Date.now();
  pruneDedupe(now);
  const existing = dedupeReservations.get(key);
  if (existing && existing > now) return false;
  dedupeReservations.set(key, now + ttlSeconds * 1000);
  return true;
}

function dedupeAvailable(key: string): boolean {
  const now = Date.now();
  pruneDedupe(now);
  const existing = dedupeReservations.get(key);
  return !existing || existing <= now;
}

function reserveQualifiedPromptCheck(sessionId: string): boolean {
  const now = Date.now();
  const state = qualifiedSessions.get(sessionId);
  if (!state || state.expiresAt <= now) {
    qualifiedSessions.delete(sessionId);
    return false;
  }
  if (state.remainingPromptChecks <= 0) return false;
  state.remainingPromptChecks -= 1;
  return true;
}

function restoreQualifiedPromptCheck(sessionId: string, maxPromptChecks: number): void {
  const state = qualifiedSessions.get(sessionId);
  if (!state || state.expiresAt <= Date.now()) return;
  state.remainingPromptChecks = Math.min(
    Math.max(0, Math.floor(maxPromptChecks)),
    state.remainingPromptChecks + 1,
  );
}

function markQualifiedSession(
  sessionId: string,
  ttlSeconds: number,
  promptBudget: number,
): void {
  qualifiedSessions.set(sessionId, {
    expiresAt: Date.now() + Math.max(1, ttlSeconds) * 1000,
    remainingPromptChecks: Math.max(0, Math.floor(promptBudget)),
  });
}

function ensureQualifiedSession(
  sessionId: string,
  ttlSeconds: number,
  promptBudget: number,
): void {
  const existing = qualifiedSessions.get(sessionId);
  if (existing && existing.expiresAt > Date.now()) return;
  markQualifiedSession(sessionId, ttlSeconds, promptBudget);
}

function releaseDedupe(key: string): void {
  dedupeReservations.delete(key);
}

/** Test-only reset; deliberately exported so retries/dedupe are deterministic. */
export function resetOffloadDedupeForTests(): void {
  dedupeReservations.clear();
  qualifiedSessions.clear();
  dedupeOps = 0;
}

function estimateTextTokens(text: string): number {
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

export function estimateOffloadTokens(value: unknown): number {
  try {
    return estimateTextTokens(JSON.stringify(value));
  } catch {
    return 1;
  }
}

/**
 * Estimate whether a new build cycle can repay itself through future reuse.
 * Constants mirror the current Core prompt caps (2k result chars, 500 param
 * chars) but remain deliberately conservative. Context pressure always wins:
 * preventing a provider context overflow is more important than token ROI.
 */
export function evaluateOffloadCostGate(input: {
  config: TdaiConfig["offload"];
  pairs: OffloadToolPair[];
  recentMessages: RecentMessage[];
  prompt?: string;
  beforeTokens: number;
  sessionQualified: boolean;
}): OffloadGateTelemetry {
  const policy = input.config.costAware;
  const expectedReuse = Math.max(1, Math.round(policy.expectedReuseCalls));
  const rawToolTokens = input.pairs.reduce(
    (total, pair) => total + estimateOffloadTokens(pair.result),
    0,
  );
  const recentTokens = input.recentMessages.reduce(
    (total, message) => total + estimateTextTokens(message.content) + 4,
    0,
  );
  const promptTokens = input.prompt ? estimateTextTokens(input.prompt) : 0;
  const contextRatio = input.config.contextWindowTokens > 0
    ? input.beforeTokens / input.config.contextWindowTokens
    : 0;
  const pressure = contextRatio >= policy.pressureRatio;

  const l1Tokens = input.pairs.length > 0
    ? 1_000 + recentTokens + input.pairs.reduce(
        (total, pair) => total
          + Math.min(1_200, estimateOffloadTokens(pair.result))
          + Math.min(300, estimateOffloadTokens(pair.params))
          + 120,
        0,
      )
    : 0;
  const l15EstimateTokens = 900 + recentTokens + Math.min(500, promptTokens) + 150;
  const l15Tokens = input.prompt ? l15EstimateTokens : 0;
  // L2 runs after enough unmapped entries accumulate. Charge the current
  // batch its proportional expected share instead of pretending every pair
  // immediately causes a full graph rewrite.
  const l2Share = Math.min(1, input.pairs.length / 6);
  const l2Tokens = input.pairs.length > 0
    ? Math.round(l2Share * (2_200 + Math.min(4_000, recentTokens) + input.pairs.length * 120 + 800))
    : 0;
  const buildTokens = l1Tokens + l15Tokens + l2Tokens;
  const activeTaskPromptBudget = Number.isFinite(policy.activeTaskPromptBudget)
    ? Math.max(0, Math.floor(policy.activeTaskPromptBudget))
    : 1;
  // A tool build can activate one or more future L1.5 lifecycle checks. Reserve
  // that expected cost at admission time so a marginal build cannot look
  // profitable merely by moving part of its cost into later requests.
  const lifecycleReserveTokens = input.pairs.length > 0
    ? activeTaskPromptBudget * l15EstimateTokens
    : 0;
  const projectedTotalCostTokens = buildTokens + lifecycleReserveTokens;
  const summaryTokensPerPair = 96;
  const perCallSavings = Math.max(0, rawToolTokens - input.pairs.length * summaryTokensPerPair);
  const projectedGross = perCallSavings * expectedReuse;
  const projectedNet = projectedGross - projectedTotalCostTokens;
  const projectedNetRatio = projectedGross > 0 ? projectedNet / projectedGross : 0;

  let allowTools = input.pairs.length > 0;
  let allowPrompt = Boolean(input.prompt);
  let decision = "disabled";
  let reason = "cost-aware gate disabled";

  if (policy.enabled) {
    allowTools = input.pairs.length > 0 && (
      pressure || (
        rawToolTokens >= policy.minToolResultTokens &&
        projectedNet >= policy.minProjectedNetTokens &&
        projectedNetRatio >= policy.minProjectedNetRatio
      )
    );
    allowPrompt = Boolean(input.prompt) && (
      pressure ||
      allowTools ||
      input.sessionQualified ||
      (policy.promptOnlyEnabled && promptTokens >= policy.minPromptTokens)
    );

    if (pressure) {
      decision = "context-pressure";
      reason = `context ratio ${contextRatio.toFixed(3)} >= ${policy.pressureRatio}`;
    } else if (allowTools) {
      decision = "build";
      reason = `projected net ${projectedNet}, ratio ${(projectedNetRatio * 100).toFixed(1)}%`;
    } else if (input.pairs.length > 0) {
      decision = "accumulate";
      reason = rawToolTokens < policy.minToolResultTokens
        ? `raw tool tokens ${rawToolTokens} < ${policy.minToolResultTokens}`
        : projectedNet < policy.minProjectedNetTokens
          ? `projected net ${projectedNet} < ${policy.minProjectedNetTokens}`
          : `projected net ratio ${(projectedNetRatio * 100).toFixed(1)}% < ${(policy.minProjectedNetRatio * 100).toFixed(1)}%`;
    } else if (allowPrompt) {
      decision = input.sessionQualified ? "active-task" : "prompt-judge";
      reason = input.sessionQualified
        ? "existing qualified task requires lifecycle judgment"
        : `speculative prompt-only enabled and prompt tokens ${promptTokens} >= ${policy.minPromptTokens}`;
    } else {
      decision = policy.promptOnlyEnabled ? "skip-short-prompt" : "defer-prompt";
      reason = policy.promptOnlyEnabled
        ? `prompt tokens ${promptTokens} < ${policy.minPromptTokens}`
        : "prompt-only L1.5 deferred until tools qualify, task is active, or context pressure rises";
    }
  }

  const candidateBuildTokens = input.pairs.length > 0
    ? buildTokens
    : (input.prompt ? l15Tokens : 0);
  const scheduledBuildTokens = allowTools ? buildTokens : (allowPrompt ? l15Tokens : 0);
  return {
    decision,
    reason,
    raw_tool_tokens: rawToolTokens,
    estimated_build_tokens: candidateBuildTokens,
    scheduled_build_tokens: scheduledBuildTokens,
    projected_gross_saved_tokens: projectedGross,
    projected_net_saved_tokens: projectedNet,
    projected_net_savings_ratio: Number(projectedNetRatio.toFixed(4)),
    expected_reuse_calls: expectedReuse,
    allow_tools: allowTools,
    allow_prompt: allowPrompt,
    prompt_tokens: promptTokens,
    prompt_only_enabled: policy.promptOnlyEnabled,
    reserved_lifecycle_tokens: lifecycleReserveTokens,
    projected_total_cost_tokens: projectedTotalCostTokens,
    active_task_prompt_budget: activeTaskPromptBudget,
    context_ratio: Number(contextRatio.toFixed(4)),
  };
}

function boundaryTimestampBefore(pairs: OffloadToolPair[]): string | undefined {
  let earliest = Number.POSITIVE_INFINITY;
  for (const pair of pairs) {
    const parsed = Date.parse(pair.timestamp);
    if (Number.isFinite(parsed)) earliest = Math.min(earliest, parsed);
  }
  return Number.isFinite(earliest) ? new Date(Math.max(0, earliest - 1)).toISOString() : undefined;
}

function stripInternalMetadata(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripInternalMetadata);
  if (!isRecord(value)) return value;
  const out: JsonRecord = {};
  for (const [key, child] of Object.entries(value)) {
    if (INTERNAL_MESSAGE_KEYS.has(key)) continue;
    out[key] = stripInternalMetadata(child);
  }
  return out;
}

function isMmdMessage(value: unknown): value is JsonRecord {
  return isRecord(value) && (
    typeof value._mmdContextMessage === "string" ||
    value._mmdInjection === true
  );
}

function appendSystemText(content: unknown, block: string): unknown {
  if (typeof content === "string") {
    return content.length > 0 ? `${content}\n\n${block}` : block;
  }
  if (Array.isArray(content)) {
    return [...content, { type: "text", text: block }];
  }
  if (content === undefined || content === null) return block;
  return `${JSON.stringify(content)}\n\n${block}`;
}

/**
 * Move MemoryCore's synthetic MMD user messages into the real system context.
 *
 * Core deliberately stays protocol-neutral and marks injected messages with
 * internal fields. The proxy is the correct layer to normalize them because it
 * knows whether the upstream expects OpenAI's message-level system role or
 * Anthropic's top-level `system` field. Unreadable marked messages are retained
 * so a malformed extension can never silently discard context.
 */
export function foldMmdIntoSystem(
  body: Record<string, unknown>,
  messages: unknown[],
  protocol: OffloadProtocol,
): Record<string, unknown> {
  const mmdTexts: string[] = [];
  const retainedMessages: unknown[] = [];

  for (const raw of messages) {
    if (!isMmdMessage(raw)) {
      retainedMessages.push(raw);
      continue;
    }
    const text = messageText(raw.content);
    if (!text) {
      retainedMessages.push(raw);
      continue;
    }
    mmdTexts.push(text);
  }

  const cleanMessages = stripInternalMetadata(retainedMessages) as unknown[];
  if (mmdTexts.length === 0) return { ...body, messages: cleanMessages };
  const mmdBlock = mmdTexts.join("\n\n");

  if (protocol === "anthropic") {
    return {
      ...body,
      system: appendSystemText(body.system, mmdBlock),
      messages: cleanMessages,
    };
  }

  const systemIndex = cleanMessages.findIndex(
    (raw) => isRecord(raw) && raw.role === "system",
  );
  if (systemIndex < 0) {
    return {
      ...body,
      messages: [{ role: "system", content: mmdBlock }, ...cleanMessages],
    };
  }

  const systemMessage = cleanMessages[systemIndex] as JsonRecord;
  const nextMessages = [...cleanMessages];
  nextMessages[systemIndex] = {
    ...systemMessage,
    content: appendSystemText(systemMessage.content, mmdBlock),
  };
  return { ...body, messages: nextMessages };
}

async function postOffload<T>(
  config: TdaiConfig,
  serviceId: string,
  path: string,
  payload: unknown,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.offload.timeoutMs);
  try {
    const response = await fetch(`${config.endpoint.replace(/\/+$/, "")}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${config.apiKey}`,
        "x-tdai-service-id": serviceId,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const text = await response.text();
    let envelope: OffloadEnvelope<T>;
    try {
      envelope = JSON.parse(text) as OffloadEnvelope<T>;
    } catch {
      throw new Error(`MemoryCore returned non-JSON (HTTP ${response.status})`);
    }
    if (!response.ok || envelope.code !== 0) {
      throw new Error(`MemoryCore offload failed (HTTP ${response.status}, code=${envelope.code ?? "?"})`);
    }
    return envelope.data as T;
  } finally {
    clearTimeout(timer);
  }
}

function compactReportCount(report: CompactData["report"], field: string): number {
  const value = report?.[field];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function enabled(config: ProxyConfig): boolean {
  return Boolean(
    config.tdai.enabled &&
    config.tdai.endpoint &&
    config.tdai.offload.enabled,
  );
}

/**
 * Ingest the new turn and transparently compact the outgoing request.
 * Any MemoryCore/network/validation failure returns the original body.
 */
export async function processOffloadRequest(input: OffloadProcessInput): Promise<OffloadProcessResult> {
  const beforeTokens = estimateOffloadTokens(input.body);
  const baseResult: OffloadProcessResult = {
    body: input.body,
    applied: false,
    beforeTokens,
    afterTokens: beforeTokens,
    savedTokens: 0,
    mmdInjected: 0,
    ingestedToolPairs: 0,
    ingestedPrompt: false,
  };
  if (!enabled(input.config) || !Array.isArray(input.body.messages)) return baseResult;

  const offloadConfig = input.config.tdai.offload;
  const serviceId = input.spaceId || input.config.tdai.serviceId;
  const offloadSessionId = buildOffloadSessionId({
    serviceId,
    userNamespace: input.userNamespace || "anonymous",
    agentSource: input.agentSource,
    sessionKey: input.sessionKey,
  });
  baseResult.offloadSessionId = offloadSessionId;
  baseResult.offloadServiceId = serviceId;
  baseResult.requestId = input.requestId
    ?? `offload-${hash(`${offloadSessionId}\u0000${Date.now()}\u0000${Math.random()}`, 24)}`;

  const recentMessages = extractRecentOffloadMessages(
    input.sourceMessages,
    offloadConfig.recentMessagesLimit,
    offloadConfig.maxRecentMessageChars,
  );
  const { prompt, humanTurns, latestUserIndex } = latestPrompt(input.sourceMessages);
  const allPairs = extractOffloadToolPairs(input.sourceMessages, input.protocol);

  // Inspect dedupe without reserving first. A batch rejected by the economics
  // gate must remain eligible on the next request so several small results can
  // accumulate into one worthwhile L1/L2 build.
  const candidatePairs = offloadConfig.ingest
    ? allPairs.filter((pair) => {
        const key = `tool:${offloadSessionId}:${hash(pair.tool_call_id)}`;
        return dedupeAvailable(key);
      })
    : [];

  // Reserve at most one configured lifecycle check before evaluating the
  // request. The reservation is restored unless this request actually queues
  // a prompt-only L1.5 call, which also makes concurrent requests respect the
  // same bounded budget.
  const qualificationReserved = offloadConfig.costAware.enabled
    ? reserveQualifiedPromptCheck(offloadSessionId)
    : false;
  const sessionQualified = qualificationReserved;
  const gate = evaluateOffloadCostGate({
    config: offloadConfig,
    pairs: candidatePairs,
    recentMessages,
    prompt,
    beforeTokens,
    sessionQualified,
  });
  baseResult.gate = gate;

  const pairReservations: string[] = [];
  const newPairs = gate.allow_tools
    ? candidatePairs.filter((pair) => {
        const key = `tool:${offloadSessionId}:${hash(pair.tool_call_id)}`;
        if (!reserveDedupe(key, offloadConfig.dedupeTtlSeconds)) return false;
        pairReservations.push(key);
        return true;
      })
    : [];

  let promptReservation: string | undefined;
  if (offloadConfig.ingest && prompt && gate.allow_prompt) {
    // Stable across assistant/tool-loop growth after the latest human turn,
    // but changes when the user genuinely sends another (even identical) turn.
    const key = `prompt:${offloadSessionId}:${hash(`${humanTurns}\u0000${latestUserIndex}\u0000${prompt}`)}`;
    if (reserveDedupe(key, offloadConfig.dedupeTtlSeconds)) promptReservation = key;
  }

  const consumesQualification = qualificationReserved
    && gate.allow_prompt
    && !gate.allow_tools
    && Boolean(promptReservation);
  if (qualificationReserved && !consumesQualification) {
    restoreQualifiedPromptCheck(
      offloadSessionId,
      offloadConfig.costAware.activeTaskPromptBudget,
    );
  }

  const ingestJobs: Promise<void>[] = [];
  if (newPairs.length > 0) {
    ingestJobs.push(
      postOffload<unknown>(input.config.tdai, serviceId, "/v2/offload/ingest", {
        session_id: offloadSessionId,
        tool_pairs: newPairs,
        recent_messages: recentMessages,
      }).then(() => {
        baseResult.ingestedToolPairs = newPairs.length;
        markQualifiedSession(
          offloadSessionId,
          offloadConfig.costAware.activeTaskTtlSeconds,
          offloadConfig.costAware.activeTaskPromptBudget,
        );
      }).catch((error: unknown) => {
        for (const key of pairReservations) releaseDedupe(key);
        console.warn(
          `[offload-proxy] ingest tools failed session=${offloadSessionId}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }),
    );
  }
  if (prompt && promptReservation) {
    ingestJobs.push(
      postOffload<unknown>(input.config.tdai, serviceId, "/v2/offload/ingest", {
        session_id: offloadSessionId,
        tool_pairs: [],
        prompt,
        recent_messages: recentMessages,
        ...(!sessionQualified && newPairs.length > 0
          ? { boundary_timestamp: boundaryTimestampBefore(newPairs) }
          : {}),
      }).then(() => {
        baseResult.ingestedPrompt = true;
      }).catch((error: unknown) => {
        releaseDedupe(promptReservation!);
        if (consumesQualification) {
          restoreQualifiedPromptCheck(
            offloadSessionId,
            offloadConfig.costAware.activeTaskPromptBudget,
          );
        }
        console.warn(
          `[offload-proxy] ingest prompt failed session=${offloadSessionId}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }),
    );
  }

  if (offloadConfig.costAware.enabled && (
    candidatePairs.length > newPairs.length || (prompt && !gate.allow_prompt)
  )) {
    console.info(
      `[offload-gate] session=${offloadSessionId} decision=${gate.decision} ` +
      `tools=${candidatePairs.length}->${newPairs.length} raw=${gate.raw_tool_tokens} ` +
      `candidate_build=${gate.estimated_build_tokens} scheduled_build=${gate.scheduled_build_tokens} ` +
      `gross=${gate.projected_gross_saved_tokens} ` +
      `net=${gate.projected_net_saved_tokens} reason=${gate.reason}`,
    );
  }

  let compactJob: Promise<CompactData | undefined> = Promise.resolve(undefined);
  if (offloadConfig.compact) {
    const messages = input.body.messages as unknown[];
    const contextWindow = offloadConfig.contextWindowTokens;
    compactJob = postOffload<CompactData>(input.config.tdai, serviceId, "/v2/offload/compact", {
      session_id: offloadSessionId,
      messages,
      ratio: Math.min(2, beforeTokens / contextWindow),
      context_window: contextWindow,
      total_tokens: beforeTokens,
      message_tokens: messages.map(estimateOffloadTokens),
    }).catch((error: unknown) => {
      console.warn(
        `[offload-proxy] compact failed session=${offloadSessionId}: ${error instanceof Error ? error.message : String(error)}`,
      );
      return undefined;
    });
  }

  const [compactData] = await Promise.all([compactJob, Promise.all(ingestJobs)]);
  if (!compactData || !Array.isArray(compactData.messages)) return baseResult;

  const mmdInjected = compactData.messages.filter(
    (raw) => isRecord(raw) && (raw._mmdContextMessage || raw._mmdInjection),
  ).length;
  baseResult.mmdInjected = Math.max(
    mmdInjected,
    compactReportCount(compactData.report, "mmdInjected"),
  );
  if (baseResult.mmdInjected > 0) {
    // Recover a bounded lifecycle check after process restart or task-state
    // expiry, but never refill a budget consumed earlier in the same TTL.
    ensureQualifiedSession(
      offloadSessionId,
      offloadConfig.costAware.activeTaskTtlSeconds,
      offloadConfig.costAware.activeTaskPromptBudget,
    );
  }

  const candidateBody = foldMmdIntoSystem(input.body, compactData.messages, input.protocol);
  const cleanMessages = candidateBody.messages as unknown[];
  const afterTokens = estimateOffloadTokens(candidateBody);
  const savedTokens = beforeTokens - afterTokens;
  const candidateChanged = JSON.stringify(candidateBody) !== JSON.stringify(input.body);
  baseResult.afterTokens = afterTokens;
  baseResult.savedTokens = savedTokens;

  const changedCount =
    compactReportCount(compactData.report, "mildReplacements") +
    compactReportCount(compactData.report, "fastPathReplaced") +
    compactReportCount(compactData.report, "fastPathDeleted") +
    compactReportCount(compactData.report, "aggressiveDeleted") +
    compactReportCount(compactData.report, "emergencyDeleted") +
    baseResult.mmdInjected;

  // Do not rely only on report counters: MemoryCore's aggressive/emergency
  // truncation path can reduce a tool result without incrementing a dedicated
  // counter. A changed body plus measured net savings is the authoritative gate.
  if (savedTokens >= offloadConfig.minSavingsTokens && candidateChanged) {
    baseResult.body = candidateBody;
    baseResult.applied = true;
    console.info(
      `[offload-proxy] applied session=${offloadSessionId} protocol=${input.protocol} ` +
      `tokens=${beforeTokens}->${afterTokens} saved=${savedTokens} mmd=${baseResult.mmdInjected} ` +
      `pairs=${baseResult.ingestedToolPairs} prompt=${baseResult.ingestedPrompt ? 1 : 0}`,
    );
  } else if (candidateChanged || changedCount > 0) {
    console.info(
      `[offload-proxy] kept original session=${offloadSessionId} protocol=${input.protocol} ` +
      `tokens=${beforeTokens}->${afterTokens} savings=${savedTokens} required=${offloadConfig.minSavingsTokens}`,
    );
  }

  return baseResult;
}
