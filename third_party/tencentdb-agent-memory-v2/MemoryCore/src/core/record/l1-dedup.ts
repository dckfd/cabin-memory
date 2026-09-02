/**
 * L1 Memory Conflict Detection (Batch Mode): decides how to handle multiple new
 * memories against existing records in a single LLM call.
 *
 * v4: Removed JSONL-based Jaccard fallback. Candidate recall now relies exclusively
 *     on vector search (primary) and FTS5 BM25 (degraded). If neither is available,
 *     conflict detection is skipped entirely — all memories go straight to store.
 *
 * Two-phase approach:
 * 1. Candidate search per new memory — vector recall or FTS5 keyword recall (fast, no LLM)
 * 2. Batch LLM judgment on all new memories + their candidate pools (single call)
 */

import type { MemoryPromptMode } from "../../config.js";
import type { ExtractedMemory, MemoryRecord, DedupDecision, MemoryType } from "./l1-writer.js";
import { parseMemoryMetadata } from "./l1-writer.js";
import { formatBatchConflictPrompt, getConflictDetectionSystemPrompt } from "../prompts/l1-dedup.js";
import type { CandidateMatch } from "../prompts/l1-dedup.js";
import { CleanContextRunner } from "../../utils/clean-context-runner.js";
import { sanitizeJsonForParse } from "../../utils/sanitize.js";
import type {
  IMemoryStore,
  IsolationFilter,
  L1FtsResult,
  L1SearchResult,
} from "../store/types.js";
import { buildFtsQuery } from "../store/sqlite.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { LLMRunner, Logger, TraceContext } from "../types.js";
import { buildTraceParams } from "../types.js";

const TAG = "[memory-tdai][l1-dedup]";
const L1_DEDUP_TIMEOUT_MS = Number(process.env.TDAI_L1_TIMEOUT_MS) || 180_000;

function throwIfAborted(signal: AbortSignal | undefined, context: string): void {
  if (!signal?.aborted) return;
  if (signal.reason instanceof Error) throw signal.reason;
  throw new Error(`${context}: aborted`);
}

// ============================
// Core function (batch mode)
// ============================

/**
 * Batch conflict detection: compare all new memories against existing records
 * in a single LLM call.
 *
 * Candidate recall strategy (3-tier degradation):
 * 1. Vector recall (vectorStore + embeddingService) — cosine similarity (best)
 * 2. FTS5 keyword recall (vectorStore with FTS available) — BM25 ranking (degraded)
 * 3. Skip conflict detection entirely — all memories go straight to "store"
 *
 * The old JSONL-based Jaccard fallback has been removed. If neither vector search
 * nor FTS is available, we skip dedup rather than paying the O(N) full-file-scan cost.
 *
 * @param memories - Newly extracted memories (with record_id)
 * @param config - OpenClaw config (for LLM access)
 * @param logger - Optional logger
 * @param model - Optional model override
 * @param vectorStore - Optional vector store for cosine similarity search
 * @param embeddingService - Optional embedding service for computing query vectors
 * @param conflictRecallTopK - Top-K candidates to recall per new memory (default: 5)
 * @returns Array of dedup decisions, one per new memory
 */
export async function batchDedup(params: {
  memories: Array<ExtractedMemory & { record_id: string }>;
  config: unknown;
  logger?: Logger;
  model?: string;
  /** Prompt family for conflict detection (default: chat). */
  promptMode?: MemoryPromptMode;
  /** Vector store for cosine similarity candidate recall */
  vectorStore?: IMemoryStore;
  /** Embedding service for computing query vectors */
  embeddingService?: EmbeddingService;
  /** Top-K candidates per new memory (default: 5) */
  conflictRecallTopK?: number;
  /** Override embedding timeout for capture-path calls (milliseconds) */
  embeddingTimeoutMs?: number;
  /** Host-neutral LLM runner — when provided, used instead of CleanContextRunner. */
  llmRunner?: LLMRunner;
  /** Isolation filter applied to candidate recall so dedup never crosses tenants. */
  filter?: IsolationFilter;
  /** langfuse 上报身份四元组（team/user/agent/session），透传给 llmRunner。 */
  traceContext?: TraceContext;
  /** Distributed-lock cancellation for recall and provider judgment. */
  abortSignal?: AbortSignal;
}): Promise<DedupDecision[]> {
  const { memories, config, logger, model, promptMode = "chat", vectorStore, embeddingService, llmRunner, filter, traceContext, abortSignal } = params;
  const topK = params.conflictRecallTopK ?? 5;
  throwIfAborted(abortSignal, "L1 dedup before recall");

  if (memories.length === 0) {
    return [];
  }

  const storeAll = () =>
    memories.map((m) => ({
      record_id: m.record_id,
      action: "store" as const,
      target_ids: [],
    }));

  // Determine what recall capabilities are available
  const hasVectorData = vectorStore && (await vectorStore.countL1()) > 0;
  throwIfAborted(abortSignal, "L1 dedup after count");
  const hasFts = vectorStore?.isFtsAvailable() ?? false;

  // Fast path: no recall capability at all → skip dedup
  if (!hasVectorData && !hasFts) {
    logger?.debug?.(`${TAG} No vector data and no FTS available, skipping conflict detection for ${memories.length} memories`);
    return storeAll();
  }

  // Phase 1: Find candidates
  //
  // Decision tree (after the fast-path guard above, vectorStore is guaranteed non-null):
  //   hasVectorData + embeddingService → Tier 1 vector recall (FTS fallback on error)
  //   otherwise hasFts                → Tier 2 FTS keyword recall
  //   otherwise                       → skip dedup (defensive; shouldn't reach here)
  let matches: CandidateMatch[];

  if (hasVectorData && embeddingService) {
    // === Tier 1: Vector recall mode ===
    logger?.debug?.(`${TAG} Using vector recall mode (topK=${topK})`);
    matches = await findCandidatesByVector(memories, vectorStore!, embeddingService, topK, logger, params.embeddingTimeoutMs, filter);
  } else if (hasFts) {
    // === Tier 2: FTS keyword recall ===
    logger?.debug?.(`${TAG} Using FTS keyword recall mode (no embedding service or no vector data)`);
    matches = await findCandidatesByFts(memories, vectorStore!, logger, filter);
  } else {
    // Shouldn't reach here given the fast-path check above, but be defensive
    logger?.debug?.(`${TAG} No usable recall path, skipping conflict detection`);
    return storeAll();
  }
  throwIfAborted(abortSignal, "L1 dedup after candidate recall");

  // Check if any memory has candidates
  const hasAnyCandidates = matches.some((m) => m.candidates.length > 0);

  if (!hasAnyCandidates) {
    logger?.debug?.(`${TAG} No similar records found for any memory, all will be stored`);
    return storeAll();
  }

  // Phase 2: Batch LLM judgment
  return runLlmJudgment(matches, memories, config, logger, model, promptMode, llmRunner, traceContext, abortSignal);
}

/**
 * Phase 2: Run batch LLM judgment on candidate matches.
 */
async function runLlmJudgment(
  matches: CandidateMatch[],
  memories: Array<ExtractedMemory & { record_id: string }>,
  config: unknown,
  logger: Logger | undefined,
  model: string | undefined,
  promptMode: MemoryPromptMode,
  llmRunner?: LLMRunner,
  traceContext?: TraceContext,
  abortSignal?: AbortSignal,
): Promise<DedupDecision[]> {
  logger?.debug?.(`${TAG} Running batch conflict detection for ${memories.length} memories (promptMode=${promptMode})`);

  try {
    const userPrompt = formatBatchConflictPrompt(matches);
    const systemPrompt = getConflictDetectionSystemPrompt(promptMode);
    let result: string;

    // langfuse trace 语义：见 l1-extractor.ts 里的说明。dedup 是 L1 的子步骤，
    // 用独立 name 便于在 UI 上区分 "抽取阶段" vs "去重判定阶段"。
    const traceParams = buildTraceParams("memory.l1-dedup", traceContext);

    if (llmRunner) {
      // Use the host-neutral LLMRunner interface
      result = await llmRunner.run({
        prompt: userPrompt,
        systemPrompt,
        taskId: "l1-conflict-detection",
        timeoutMs: L1_DEDUP_TIMEOUT_MS,
        thinkingMode: "disabled",
        retryOnLength: true,
        abortSignal,
        ...traceParams,
      });
    } else {
      // Fallback: create CleanContextRunner (OpenClaw path)
      const runner = new CleanContextRunner({
        config,
        modelRef: model,
        enableTools: false,
        logger,
      });

      result = await runner.run({
        prompt: userPrompt,
        systemPrompt,
        taskId: "l1-conflict-detection",
        timeoutMs: L1_DEDUP_TIMEOUT_MS,
        thinkingMode: "disabled",
        retryOnLength: true,
        abortSignal,
        ...traceParams,
      });
    }

    const decisions = parseBatchResult(result, memories, matches, logger);
    return decisions;
  } catch (err) {
    throwIfAborted(abortSignal, "L1 dedup provider judgment");
    logger?.warn?.(
      `${TAG} Batch conflict detection failed, defaulting all to store: ${err instanceof Error ? err.message : String(err)}`,
    );
    return memories.map((m) => ({
      record_id: m.record_id,
      action: "store" as const,
      target_ids: [],
    }));
  }
}

// ============================
// Candidate recall strategies
// ============================

/** Preserve lifecycle/scope metadata identically across dense and FTS recall. */
export function searchResultToMemoryRecord(
  row: L1SearchResult | L1FtsResult,
): MemoryRecord {
  const metadata = parseMemoryMetadata(row.metadata_json);
  return {
    id: row.record_id,
    content: row.content,
    type: row.type as MemoryRecord["type"],
    priority: row.priority,
    scene_name: row.scene_name,
    source_message_ids: Array.isArray(metadata.source_message_ids)
      ? metadata.source_message_ids.filter((entry): entry is string => typeof entry === "string")
      : [],
    metadata,
    timestamps: Array.from(new Set([
      row.timestamp_start, row.timestamp_end, row.timestamp_str,
    ].filter(Boolean))),
    createdAt: "",
    updatedAt: "",
    sessionKey: row.session_key,
    sessionId: row.session_id,
  };
}

/**
 * Vector-based candidate recall (aligned with prototype):
 * batch-embed new memories → cosine search in VectorStore → exclude self-batch → return candidates.
 */
async function findCandidatesByVector(
  memories: Array<ExtractedMemory & { record_id: string }>,
  vectorStore: IMemoryStore,
  embeddingService: EmbeddingService,
  topK: number,
  logger?: Logger,
  embeddingTimeoutMs?: number,
  filter?: IsolationFilter,
): Promise<CandidateMatch[]> {
  const newRecordIds = new Set(memories.map((m) => m.record_id));

  // Batch-compute embeddings for all new memories
  const texts = memories.map((m) => m.content);
  const embeddings = await embeddingService.embedBatch(texts, embeddingTimeoutMs ? { timeoutMs: embeddingTimeoutMs } : undefined);

  const matches: CandidateMatch[] = [];

  for (let i = 0; i < memories.length; i++) {
    const mem = memories[i];
    const queryVec = embeddings[i];

    // Vector search top-K (request extra to account for self-batch filtering)
    const searchResults = filter
      ? await vectorStore.searchL1Vector(queryVec, topK + memories.length, mem.content, filter)
      : await vectorStore.searchL1Vector(queryVec, topK + memories.length, mem.content);

    // Exclude records from current batch, convert to MemoryRecord format
    const candidates: MemoryRecord[] = searchResults
      .filter((r) => !newRecordIds.has(r.record_id))
      .slice(0, topK)
      .map(searchResultToMemoryRecord);

    matches.push({ newMemory: mem, candidates });
  }

  logger?.debug?.(
    `${TAG} Vector recall: ${matches.map((m) => `${m.newMemory.record_id}→${m.candidates.length}`).join(", ")}`,
  );

  return matches;
}

/**
 * FTS5-based candidate recall:
 * Uses the FTS index for efficient BM25-ranked keyword matching.
 * This replaces the old Jaccard word-overlap fallback entirely.
 */
async function findCandidatesByFts(
  memories: Array<ExtractedMemory & { record_id: string }>,
  vectorStore: IMemoryStore,
  _logger?: Logger,
  filter?: IsolationFilter,
): Promise<CandidateMatch[]> {
  const newRecordIds = new Set(memories.map((m) => m.record_id));
  const matches: CandidateMatch[] = [];

  for (const mem of memories) {
    const ftsQuery = buildFtsQuery(mem.content);
    if (ftsQuery) {
      const ftsResults = filter
        ? await vectorStore.searchL1Fts(ftsQuery, 10, filter)
        : await vectorStore.searchL1Fts(ftsQuery, 10);
      // Filter out records from the current batch
      const candidates: MemoryRecord[] = ftsResults
        .filter((r) => !newRecordIds.has(r.record_id))
        .slice(0, 5)
        .map(searchResultToMemoryRecord);
      matches.push({ newMemory: mem, candidates });
    } else {
      matches.push({ newMemory: mem, candidates: [] });
    }
  }

  _logger?.debug?.(`${TAG} FTS keyword recall: ${matches.map((m) => `${m.newMemory.record_id}→${m.candidates.length}`).join(", ")}`);
  return matches;
}

// ============================
// Result parsing
// ============================

const VALID_TYPES: MemoryType[] = ["persona", "episodic", "instruction", "work_fact", "work_task", "work_method", "work_artifact"];

/**
 * Parse the LLM's batch conflict detection JSON response.
 *
 * Expected format: [{record_id, action, target_ids, merged_content,
 * merged_type, merged_priority, merged_timestamps, merged_metadata}]
 */
export function parseBatchResult(
  raw: string,
  memories: Array<ExtractedMemory & { record_id: string }>,
  matches: CandidateMatch[],
  logger?: Logger,
): DedupDecision[] {
  try {
    // Strip markdown code block wrappers
    let cleaned = raw.trim();
    if (cleaned.startsWith("```")) {
      cleaned = cleaned.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
    }

    // Extract JSON array
    const arrayMatch = cleaned.match(/\[[\s\S]*\]/);
    if (!arrayMatch) {
      logger?.warn?.(`${TAG} No JSON array found in conflict detection response`);
      return fallbackStoreAll(memories);
    }

    // Sanitize control characters inside JSON string literals that LLM may produce
    const sanitized = sanitizeJsonForParse(arrayMatch[0]);
    const parsed = JSON.parse(sanitized) as unknown[];

    if (!Array.isArray(parsed)) {
      logger?.warn?.(`${TAG} Conflict detection response is not an array`);
      return fallbackStoreAll(memories);
    }

    // Build decisions from LLM output
    const decisions: DedupDecision[] = [];
    const validActions = ["store", "update", "merge", "skip"];
    const memoryIds = new Set(memories.map((memory) => memory.record_id));
    const allowedTargets = new Map(matches.map((match) => [
      match.newMemory.record_id,
      new Set(match.candidates.map((candidate) => candidate.id)),
    ]));
    const decidedIds = new Set<string>();
    const claimedTargets = new Set<string>();

    for (const item of parsed) {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        throw new Error("conflict decision must be an object");
      }
      const d = item as Record<string, unknown>;

      if (typeof d.record_id !== "string" || !d.record_id) {
        throw new Error("conflict decision has an empty record_id");
      }
      const recordId = d.record_id;
      if (!memoryIds.has(recordId)) {
        throw new Error(`conflict decision references unknown record ${recordId}`);
      }
      if (decidedIds.has(recordId)) {
        throw new Error(`duplicate conflict decision for ${recordId}`);
      }
      decidedIds.add(recordId);

      if (typeof d.action !== "string" || !validActions.includes(d.action)) {
        throw new Error(`invalid conflict action for ${recordId}`);
      }
      const action = d.action as DedupDecision["action"];
      if (d.target_ids !== undefined && !Array.isArray(d.target_ids)) {
        throw new Error(`target_ids must be an array for ${recordId}`);
      }
      const targetIds = (d.target_ids ?? []) as unknown[];
      if (targetIds.some((target) => typeof target !== "string" || target.length === 0)) {
        throw new Error(`target_ids contains a non-string/empty ID for ${recordId}`);
      }
      const normalizedTargets = targetIds as string[];
      if (new Set(normalizedTargets).size !== normalizedTargets.length) {
        throw new Error(`duplicate target_ids for ${recordId}`);
      }

      if (action === "store" || action === "skip") {
        if (normalizedTargets.length > 0) {
          throw new Error(`${action} decision must not delete targets for ${recordId}`);
        }
      } else {
        if (normalizedTargets.length === 0) {
          throw new Error(`${action} decision requires a target for ${recordId}`);
        }
        const allowed = allowedTargets.get(recordId) ?? new Set<string>();
        for (const targetId of normalizedTargets) {
          if (!allowed.has(targetId)) {
            throw new Error(`unauthorized target ${targetId} for ${recordId}`);
          }
          if (claimedTargets.has(targetId)) {
            throw new Error(`target ${targetId} is claimed by multiple new memories`);
          }
          claimedTargets.add(targetId);
        }
      }

      decisions.push({
        record_id: recordId,
        action,
        target_ids: normalizedTargets,
        merged_content: typeof d.merged_content === "string" ? d.merged_content : undefined,
        merged_type: VALID_TYPES.includes(d.merged_type as MemoryType) ? (d.merged_type as MemoryType) : undefined,
        merged_priority: typeof d.merged_priority === "number" ? d.merged_priority : undefined,
        merged_timestamps: Array.isArray(d.merged_timestamps) ? d.merged_timestamps.map(String) : undefined,
        merged_metadata: (
          d.merged_metadata
          && typeof d.merged_metadata === "object"
          && !Array.isArray(d.merged_metadata)
        ) ? parseMemoryMetadata(d.merged_metadata) : undefined,
      });
    }

    // Ensure all memories have a decision (fill missing with "store")
    for (const mem of memories) {
      if (!decidedIds.has(mem.record_id)) {
        logger?.debug?.(`${TAG} No decision for record ${mem.record_id}, defaulting to store`);
        decisions.push({
          record_id: mem.record_id,
          action: "store",
          target_ids: [],
        });
      }
    }

    return decisions;
  } catch (err) {
    logger?.warn?.(`${TAG} Failed to parse conflict detection result: ${err instanceof Error ? err.message : String(err)}`);
    return fallbackStoreAll(memories);
  }
}

/**
 * Fallback: store all memories when parsing fails.
 */
function fallbackStoreAll(memories: Array<ExtractedMemory & { record_id: string }>): DedupDecision[] {
  return memories.map((m) => ({
    record_id: m.record_id,
    action: "store" as const,
    target_ids: [],
  }));
}
