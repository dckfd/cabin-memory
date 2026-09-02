/**
 * conversation_search tool: Agent-callable tool for searching L0 conversation records.
 *
 * Supports three search strategies with automatic degradation:
 *   1. **hybrid** (default) — FTS5 keyword + vector embedding in parallel,
 *      merged via Reciprocal Rank Fusion (RRF).
 *   2. **embedding** — pure vector similarity (when FTS5 is unavailable).
 *   3. **fts** — pure FTS5 keyword search (when embedding is unavailable).
 *
 * The tool is registered via `api.registerTool()` in index.ts.
 */

import type { IMemoryStore, L0SearchResult } from "../store/types.js";
import { buildFtsQuery } from "../store/sqlite.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { Logger } from "../types.js";
import { planRetrievalQueries } from "../retrieval/query-planner.js";
import { coverageFirstRrfMerge } from "../retrieval/coverage-fusion.js";
import { shouldExpandConversationContext, shouldPreferConsensusFusion } from "../retrieval/context-policy.js";

// ============================
// Types
// ============================

export interface ConversationSearchResultItem {
  id: string;
  session_key: string;
  session_id: string;
  /** Role of the message sender: "user" or "assistant" */
  role: string;
  /** Text content of this single message */
  content: string;
  score: number;
  recorded_at: string;
  timestamp: number;
  /** True when included as context around a direct search hit. */
  is_context?: boolean;
}

export interface ConversationSearchResult {
  results: ConversationSearchResultItem[];
  total: number;
  /** Actual search strategy used: "hybrid", "embedding", "fts", or "none". */
  strategy: string;
  /** Optional message, e.g. when embedding is not configured. */
  message?: string;
}

const TAG = "[memory-tdai][tdai_conversation_search]";

// ============================
// RRF (Reciprocal Rank Fusion)
// ============================

/** Standard RRF constant from the original RRF paper. */
const RRF_K = 60;

/**
 * Merge multiple ranked lists of `ConversationSearchResultItem` via Reciprocal
 * Rank Fusion. Items appearing in multiple lists get their RRF scores summed.
 *
 * Returns items sorted by descending RRF score. The `score` field of each
 * returned item is replaced by the RRF score for consistent ranking semantics.
 */
function rrfMergeL0(...lists: ConversationSearchResultItem[][]): ConversationSearchResultItem[] {
  const map = new Map<string, { item: ConversationSearchResultItem; rrfScore: number }>();

  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      const score = 1 / (RRF_K + rank + 1);
      const existing = map.get(item.id);
      if (existing) {
        existing.rrfScore += score;
      } else {
        map.set(item.id, { item, rrfScore: score });
      }
    }
  }

  return [...map.values()]
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(({ item, rrfScore }) => ({ ...item, score: rrfScore }));
}

async function expandAdjacentContext(
  vectorStore: IMemoryStore,
  hits: ConversationSearchResultItem[],
  contextWindow: number,
  directHitLimit: number,
): Promise<ConversationSearchResultItem[]> {
  const window = Math.max(0, Math.min(3, Math.floor(contextWindow)));
  if (window === 0 || !vectorStore.queryL0BySessionIds || !hits.some((item) => item.session_id)) return hits;
  const rows = await vectorStore.queryL0BySessionIds(
    [...new Set(hits.map((item) => item.session_id).filter(Boolean))],
    Math.max(directHitLimit * (window * 2 + 1) * 2, 50),
  );
  const bySession = new Map<string, typeof rows>();
  for (const row of rows) {
    const group = bySession.get(row.session_id) ?? [];
    group.push(row);
    bySession.set(row.session_id, group);
  }
  const expanded: ConversationSearchResultItem[] = [];
  const seen = new Set<string>();
  for (const hit of hits) {
    const group = bySession.get(hit.session_id) ?? [];
    const index = group.findIndex((row) => row.record_id === hit.id);
    const neighborhood = index < 0 ? [] : group.slice(Math.max(0, index - window), index + window + 1);
    for (const row of neighborhood) {
      if (seen.has(row.record_id)) continue;
      seen.add(row.record_id);
      expanded.push({
        id: row.record_id,
        session_key: row.session_key,
        session_id: row.session_id,
        role: row.role,
        content: row.message_text,
        score: row.record_id === hit.id ? hit.score : hit.score * 0.5,
        recorded_at: row.recorded_at,
        timestamp: row.timestamp,
        is_context: row.record_id !== hit.id,
      });
    }
    if (!seen.has(hit.id)) {
      seen.add(hit.id);
      expanded.push(hit);
    }
  }
  return expanded;
}

// ============================
// Search implementation
// ============================

export async function executeConversationSearch(params: {
  query: string;
  limit: number;
  sessionKey?: string;
  vectorStore?: IMemoryStore;
  embeddingService?: EmbeddingService;
  logger?: Logger;
  multiHop?: boolean;
  maxQueries?: number;
  /** Minimum distinct candidates reserved for each hybrid source. */
  hybridSourceFloor?: number;
  requestedStrategy?: "hybrid" | "keyword" | "embedding";
  hybridFloorPolicy?: "fixed" | "constrained_consensus";
  /** Number of adjacent messages to include on each side of every direct hit. */
  contextWindow?: number;
  contextPolicy?: "auto" | "always";
}): Promise<ConversationSearchResult> {
  const {
    query,
    limit,
    sessionKey: sessionFilter,
    vectorStore,
    embeddingService,
    logger,
    multiHop = false,
    maxQueries = 4,
    hybridSourceFloor = 1,
    requestedStrategy = "hybrid",
    hybridFloorPolicy = "fixed",
    contextWindow = 0,
    contextPolicy = "always",
  } = params;
  const effectiveContextWindow = contextPolicy === "auto" && !shouldExpandConversationContext(query)
    ? 0
    : contextWindow;

  if (multiHop) {
    const plan = planRetrievalQueries(query, maxQueries);
    if (plan.decomposed) {
      const perQueryLimit = Math.max(limit, Math.ceil(limit * 1.5));
      const branches: ConversationSearchResult[] = [];
      // Avoid bursting several remote embedding requests at once. Sequential
      // facets preserve hybrid semantics under provider rate limits.
      for (const facet of plan.queries) {
        branches.push(await executeConversationSearch({
          ...params,
          query: facet,
          limit: perQueryLimit,
          multiHop: false,
          contextWindow: 0,
        }));
      }
      const merged = coverageFirstRrfMerge(branches.map((branch) => branch.results), limit);
      const expanded = vectorStore
        ? await expandAdjacentContext(vectorStore, merged, effectiveContextWindow, limit)
        : merged;
      return {
        results: expanded,
        total: expanded.length,
        strategy: `multi_query:${branches.map((branch) => branch.strategy).join("+")}`,
      };
    }
  }

  logger?.debug?.(
    `${TAG} CALLED: query="${query.slice(0, 100)}", limit=${limit}, ` +
    `sessionFilter=${sessionFilter ?? "(none)"}, ` +
    `vectorStore=${vectorStore ? "available" : "UNAVAILABLE"}, ` +
    `embeddingService=${embeddingService ? "available" : "UNAVAILABLE"}`,
  );

  if (!query || query.trim().length === 0) {
    logger?.debug?.(`${TAG} Empty query, returning empty`);
    return { results: [], total: 0, strategy: "none" };
  }

  if (!vectorStore) {
    logger?.warn?.(`${TAG} VectorStore not available`);
    return { results: [], total: 0, strategy: "none" };
  }

  // ── Determine available capabilities ──
  const hasEmbedding = !!embeddingService && requestedStrategy !== "keyword";
  const hasFts = vectorStore.isFtsAvailable() && requestedStrategy !== "embedding";

  if (!hasEmbedding && !hasFts) {
    logger?.warn?.(`${TAG} Neither EmbeddingService nor FTS5 available — cannot search`);
    return {
      results: [],
      total: 0,
      strategy: "none",
      message:
        "Embedding service is not configured and FTS is not available. " +
        "Conversation search requires an embedding provider or FTS5 support. " +
        "Please configure an embedding provider in the embedding.provider setting (e.g. openai_compatible).",
    };
  }

  // ── Over-retrieve for later filtering and RRF merging ──
  const candidateK = sessionFilter ? limit * 4 : limit * 3;

  // ── Run available search strategies in parallel ──
  const [ftsItems, vecItems] = await Promise.all([
    // FTS5 keyword search on L0
    (async (): Promise<ConversationSearchResultItem[]> => {
      if (!hasFts) return [];
      try {
        const ftsQuery = buildFtsQuery(query);
        if (!ftsQuery) {
          logger?.debug?.(`${TAG} [hybrid-fts] No usable FTS tokens from query`);
          return [];
        }
        logger?.debug?.(`${TAG} [hybrid-fts] FTS5 query: "${ftsQuery}"`);
        const ftsResults = await vectorStore.searchL0Fts(ftsQuery, candidateK);
        logger?.debug?.(`${TAG} [hybrid-fts] FTS5 returned ${ftsResults.length} candidates`);
        return ftsResults.map((r) => ({
          id: r.record_id,
        session_key: r.session_key,
          session_id: r.session_id,
          role: r.role,
          content: r.message_text,
          score: r.score,
          recorded_at: r.recorded_at,
          timestamp: r.timestamp,
        }));
      } catch (err) {
        logger?.warn?.(
          `${TAG} [hybrid-fts] FTS5 search failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
        );
        return [];
      }
    })(),

    // Vector embedding search on L0
    (async (): Promise<ConversationSearchResultItem[]> => {
      if (!hasEmbedding) return [];
      try {
        logger?.debug?.(`${TAG} [hybrid-vec] Generating query embedding...`);
        const queryEmbedding = await embeddingService!.embed(query);
        logger?.debug?.(
          `${TAG} [hybrid-vec] Embedding OK, dims=${queryEmbedding.length}, searching top-${candidateK}...`,
        );
        const vecResults: L0SearchResult[] = await vectorStore.searchL0Vector(queryEmbedding, candidateK, query);
        logger?.debug?.(`${TAG} [hybrid-vec] Vector search returned ${vecResults.length} candidates`);
        return vecResults.map((r) => ({
          id: r.record_id,
          session_key: r.session_key,
          session_id: r.session_id,
          role: r.role,
          content: r.message_text,
          score: r.score,
          recorded_at: r.recorded_at,
          timestamp: r.timestamp,
        }));
      } catch (err) {
        logger?.warn?.(
          `${TAG} [hybrid-vec] Embedding search failed (non-fatal): ${err instanceof Error ? err.message : String(err)}`,
        );
        return [];
      }
    })(),
  ]);

  // ── Determine effective strategy ──
  const ftsOk = ftsItems.length > 0;
  const vecOk = vecItems.length > 0;
  let strategy: string;

  if (ftsOk && vecOk) {
    strategy = "hybrid";
  } else if (vecOk) {
    strategy = "embedding";
  } else if (ftsOk) {
    strategy = "fts";
  } else {
    logger?.debug?.(`${TAG} Both search paths returned 0 results`);
    return { results: [], total: 0, strategy: hasEmbedding ? "embedding" : "fts" };
  }

  // ── Merge results ──
  let results: ConversationSearchResultItem[];
  if (strategy === "hybrid") {
    const sourceFloor = hybridFloorPolicy === "constrained_consensus" && shouldPreferConsensusFusion(query)
      ? 0
      : Math.max(0, Math.min(Math.floor(limit / 2), Math.floor(hybridSourceFloor)));
    results = coverageFirstRrfMerge([ftsItems, vecItems], Number.MAX_SAFE_INTEGER, RRF_K, sourceFloor);
    logger?.debug?.(
      `${TAG} [hybrid] RRF merged: fts=${ftsItems.length}, vec=${vecItems.length} → ${results.length} unique`,
    );
  } else {
    // Single-source: use whichever list has results (already sorted by score)
    results = ftsOk ? ftsItems : vecItems;
  }

  // ── Apply session key filter ──
  if (sessionFilter) {
    const preFilterCount = results.length;
    results = results.filter((r) => r.session_key === sessionFilter);
    logger?.debug?.(`${TAG} After session filter "${sessionFilter}": ${results.length}/${preFilterCount}`);
  }

  // ── Trim to requested limit ──
  let trimmed = results.slice(0, limit);

  trimmed = await expandAdjacentContext(vectorStore, trimmed, effectiveContextWindow, limit);

  logger?.debug?.(
    `${TAG} RESULT (strategy=${strategy}): returning ${trimmed.length} messages ` +
    `(scores: [${trimmed.map((r) => r.score.toFixed(3)).join(", ")}])`,
  );

  return {
    results: trimmed,
    total: trimmed.length,
    strategy,
  };
}

// ============================
// Tool response formatter
// ============================

export function formatConversationSearchResponse(result: ConversationSearchResult): string {
  if (result.message) {
    return result.message;
  }
  if (result.results.length === 0) {
    return "No matching conversation messages found.";
  }

  const lines: string[] = [
    `Found ${result.total} matching message(s):`,
    "",
  ];

  for (const item of result.results) {
    const scoreStr = typeof item.score === "number" ? ` (score: ${item.score.toFixed(3)})` : "";
    const dateStr = item.recorded_at ? ` [${item.recorded_at}]` : "";
    lines.push(`---`);
    const contextStr = item.is_context ? " [adjacent context]" : " [direct hit]";
    lines.push(`**[${item.role}]** Session: ${item.session_key}${dateStr}${scoreStr}${contextStr}`);
    lines.push("");
    lines.push(item.content);
    lines.push("");
  }

  return lines.join("\n");
}
