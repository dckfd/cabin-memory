/**
 * L1 Memory Writer: writes extracted memories to JSONL files.
 *
 * File naming: records/YYYY-MM-DD.jsonl (daily shards, all sessions merged).
 * Each record includes sessionKey for traceability.
 *
 * Write strategy:
 * - JSONL is the append-only persistent store (source of truth for backup/recovery).
 * - VectorStore (SQLite) is the primary retrieval engine.
 * - On update/merge, old records are deleted from VectorStore in real-time;
 *   JSONL is append-only and cleaned up periodically by memory-cleaner.
 *
 * Supports store (append), update, merge, and skip operations.
 *
 * v3: Aligned with Kenty's prompt output format — 3 memory types (persona/episodic/instruction),
 * numeric priority, scene_name, source_message_ids, metadata, timestamps.
 */

import crypto from "node:crypto";
import {
  DEFAULT_ISOLATION_ID,
  rowMatchesIsolation,
  type IMemoryStore,
  type IsolationFilter,
  type L1RecordRow,
} from "../store/types.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";
import type { Logger } from "../types.js";
import { finalizeCockpitMetadataAfterDedup } from "./cockpit-memory-contract.js";

// ============================
// Types
// ============================

/** L1 memory types: chat-mode legacy types + code/work-mode team memory types. */
export type MemoryType =
  | "persona"
  | "episodic"
  | "instruction"
  | "work_fact"
  | "work_task"
  | "work_method"
  | "work_artifact";

/** Metadata for episodic memories (time plus optional smart-cockpit scope). */
export interface EpisodicMetadata {
  activity_start_time?: string; // ISO 8601
  activity_end_time?: string; // ISO 8601
  /** When the source utterance/tool evidence was observed, not event time. */
  mentioned_at?: string;
  timezone?: string;
  time_precision?: "instant" | "minute" | "hour" | "day" | "range" | "unknown";
  temporal_status?: "resolved" | "relative_unresolved" | "unknown";
  domain?: string;
  action_status?:
    | "requested"
    | "selected"
    | "clarified"
    | "confirmed"
    | "executed"
    | "verified"
    | "completed"
    | "failed"
    | "cancelled"
    | "unresolved";
  occupant_scope?: string;
  vehicle_scope?: string;
  seat_zone?: string;
  target?: string;
  unit?: string;
  confidence?: number | string;
  /** Stable operation/entity key used for updates within one exact scope. */
  episode_key?: string;
  /** Earlier episode or record identifiers explicitly replaced by this state. */
  supersedes?: string[];
  /** Records physically consolidated by dedup merge; not a semantic update edge. */
  merged_from_record_ids?: string[];
  source_session_id?: string;
  source_session_ids?: string[];
  /** Evidence IDs are mirrored into metadata because all active stores retain metadata_json. */
  source_message_ids?: string[];
  evidence_roles?: string[];
  /** Versioned, machine-readable cockpit state contract. */
  schema_version?: "cockpit-state-v1" | string;
  record_kind?: "event" | "state_assertion" | string;
  state_key?: string;
  slot?: string;
  value?: unknown;
  subject?: string;
  relation?: "asserted" | "updated" | "cancelled" | "negated" | string;
  valid_from?: string;
  valid_to?: string;
  construction_model?: string;
  construction_stage?: string;
  construction_compiler_model?: string;
  construction_compiler_status?: "passed" | "failed" | string;
  construction_reconciliation_model?: string;
  construction_reconciliation_status?: "passed" | "failed" | "skipped" | string;
  construction_assembler_status?: "passed" | "failed" | string;
  construction_assembler_version?: string;
  input_candidate_ids?: string[];
  canonicalized_input_candidate_ids?: string[];
  construction_review_model?: string;
  construction_review_status?: "passed" | "failed" | string;
  construction_review_mode?: "independent_source_compiler" | string;
  construction_quality?: Record<string, unknown>;
}

/**
 * A persisted memory record in L1 JSONL files.
 *
 * v3 changes from v2:
 * - `importance: "high"|"medium"|"low"` → `priority: number` (0-100, -1 for strict global instructions)
 * - Added `scene_name`, `source_message_ids`, `metadata`, `timestamps`
 * - Removed `keywords` (will be rebuilt from content for search)
 * - MemoryType reduced from 4 to 3 (removed "preference", folded into "persona")
 */
export interface MemoryRecord {
  /** Unique ID for dedup updates */
  id: string;
  /** Memory content */
  content: string;
  /** Memory type: persona / episodic / instruction */
  type: MemoryType;
  /** Priority score: 0-100 (higher = more important), -1 = strict global instruction */
  priority: number;
  /** Scene name this memory belongs to */
  scene_name: string;
  /** Source message IDs that contributed to this memory */
  source_message_ids: string[];
  /** Type-specific metadata (e.g., activity_start_time for episodic) */
  metadata: EpisodicMetadata | Record<string, unknown>;
  /** Timestamp trail: all timestamps related to this memory (for merge history tracking) */
  timestamps: string[];
  /** Creation timestamp (ISO) */
  createdAt: string;
  /** Last update timestamp (ISO) */
  updatedAt: string;
  /** Monotonic version. New memories start at 1; update/merge increments by 1. */
  version?: number;
  /** Source session key (conversation channel identifier) */
  sessionKey: string;
  /** Source session ID (single conversation instance identifier) */
  sessionId: string;
  /** Optional task dimension for L0/L1 filtering. */
  taskId?: string;
  /**
   * Three-dim tenancy isolation (new in this branch).
   *
   * `userId` / `agentId` are mandatory for new writes once gateway-level
   * isolation enforcement is on, but kept optional on the type to avoid
   * breaking pre-isolation call sites and tests during rollout. The SQLite
   * upsert defaults them to '' if missing; the migration script backfills
   * existing rows with `__legacy__`.
   *
   * See `docs/l0l3-tenant-isolation-design.md`.
   */
  teamId?: string;
  userId?: string;
  agentId?: string;
}

/**
 * A memory as extracted by LLM (before dedup / persistence).
 * Matches the output format of Kenty's extraction prompt.
 */
export interface ExtractedMemory {
  content: string;
  type: MemoryType;
  priority: number;
  source_message_ids: string[];
  metadata: EpisodicMetadata | Record<string, unknown>;
  /** Scene name this memory was extracted in */
  scene_name: string;
}

export type DedupAction = "store" | "update" | "merge" | "skip";

/**
 * v3 batch dedup decision — one per new memory, aligned with Kenty's conflict detection prompt.
 *
 * Key changes:
 * - `targetId` → `target_ids` (array, supports multi-target merge/update)
 * - Added `merged_type`, `merged_priority`, `merged_timestamps` for cross-type merge
 */
export interface DedupDecision {
  /** Which new memory this decision is about */
  record_id: string;
  action: DedupAction;
  /** IDs of existing records to replace/remove (for update/merge) */
  target_ids: string[];
  /** Merged/updated content text (for update/merge) */
  merged_content?: string;
  /** Best type after merge (for update/merge, may differ from original) */
  merged_type?: MemoryType;
  /** Priority after merge (for update/merge) */
  merged_priority?: number;
  /** Union of all related timestamps (for update/merge) */
  merged_timestamps?: string[];
  /** Scope/time/lifecycle metadata retained across merge/update. */
  merged_metadata?: EpisodicMetadata | Record<string, unknown>;
}

const TAG = "[memory-tdai][l1-writer]";

/** A durable L1 sink rejected a write required by the strict cockpit path. */
export class L1PersistenceError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "L1PersistenceError";
  }
}

// ============================
// Core functions
// ============================

/**
 * Generate a unique memory ID.
 */
export function generateMemoryId(): string {
  return `m_${Date.now()}_${crypto.randomBytes(4).toString("hex")}`;
}

/** Parse persisted metadata without allowing malformed JSON to break a write. */
export function parseMemoryMetadata(value: unknown): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      return parseMemoryMetadata(JSON.parse(value));
    } catch {
      return {};
    }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).filter(
      ([key, entry]) =>
        !["__proto__", "prototype", "constructor"].includes(key)
        && entry !== undefined,
    ),
  );
}

/**
 * Merge old → new → explicit LLM metadata. Later evidence wins while missing
 * fields retain tenant/seat/time provenance from the previous record.
 */
export function mergeMemoryMetadata(
  ...values: unknown[]
): Record<string, unknown> {
  const parsed = values.map(parseMemoryMetadata);
  const merged = Object.assign({}, ...parsed);
  // These fields are append-only provenance. Plain Object.assign used to
  // discard old evidence during every update/merge even though metadata_json
  // is the only provenance-bearing column shared by SQLite and TCVDB.
  for (const key of ["source_message_ids", "source_session_ids", "evidence_roles", "supersedes", "merged_from_record_ids"]) {
    const union = [...new Set(parsed.flatMap((value) =>
      Array.isArray(value[key])
        ? (value[key] as unknown[]).filter((entry): entry is string => typeof entry === "string" && entry.length > 0)
        : []
    ))];
    if (union.length > 0) merged[key] = union;
  }
  return merged;
}

/**
 * Write a memory record according to the dedup decision.
 *
 * - store: append new record
 * - update: remove target records + append updated record
 * - merge: remove target records + append merged record
 * - skip: do nothing
 *
 * v3: supports multi-target removal for update/merge.
 * v3.1: optional VectorStore + EmbeddingService for dual-write (JSONL + vector).
 */
export async function writeMemory(params: {
  memory: ExtractedMemory;
  decision: DedupDecision;
  baseDir: string;
  sessionKey: string;
  sessionId?: string;
  taskId?: string;
  /** Tenancy isolation propagated into MemoryRecord and downstream store. */
  teamId?: string;
  userId?: string;
  agentId?: string;
  logger?: Logger;
  /** Optional vector store for dual-write (JSONL + vector DB) */
  vectorStore?: IMemoryStore;
  /** Optional embedding service (required when vectorStore is provided) */
  embeddingService?: EmbeddingService;
  /** StorageAdapter for file operations (COS/local). Falls back to fs when absent. */
  storage?: StorageAdapter;
  /** Fail instead of degrading when an enabled durable sink rejects a write. */
  strictPersistence?: boolean;
  /** Build the exact record without mutating JSONL or the vector store. */
  prepareOnly?: boolean;
  /** Shared transaction timestamp used to make a prepared batch reproducible. */
  nowIso?: string;
}): Promise<MemoryRecord | null> {
  const {
    memory,
    decision,
    baseDir,
    sessionKey,
    sessionId,
    taskId,
    teamId,
    userId,
    agentId,
    logger,
    vectorStore,
    embeddingService,
    storage,
    strictPersistence = false,
    prepareOnly = false,
    nowIso,
  } = params;

  if (decision.action === "skip") {
    logger?.debug?.(`${TAG} Skipping memory: ${memory.content.slice(0, 50)}...`);
    return null;
  }

  const now = nowIso ?? new Date().toISOString();

  let nextVersion = 1;
  let existingMetadata: Record<string, unknown>[] = [];
  if ((decision.action === "update" || decision.action === "merge") && decision.target_ids.length > 0 && vectorStore) {
    try {
      const requestedIds = [...new Set(decision.target_ids)];
      if (strictPersistence && requestedIds.length !== decision.target_ids.length) {
        throw new L1PersistenceError(
          `duplicate update/merge predecessor IDs for ${decision.record_id}`,
        );
      }

      // Session is provenance rather than an authorization boundary: a later
      // cockpit session may legitimately update a state created in an earlier
      // session. Tenant, agent and task dimensions remain mandatory whenever
      // the caller supplied them.
      const predecessorScope: IsolationFilter = {
        teamId,
        userId,
        agentId,
        taskId,
      };
      const fetched: L1RecordRow[] = [];
      // TCVDB documentIds accepts at most 20 primary keys per query.
      for (let offset = 0; offset < requestedIds.length; offset += 20) {
        const recordIds = requestedIds.slice(offset, offset + 20);
        fetched.push(...await vectorStore.queryL1Records({
          recordIds,
          ...predecessorScope,
        }));
      }

      const requestedSet = new Set(requestedIds);
      const safeRows = fetched.filter((row) =>
        requestedSet.has(row.record_id) && rowMatchesIsolation(row, predecessorScope)
      );
      if (strictPersistence) {
        const returnedIds = safeRows.map((row) => row.record_id);
        const returnedSet = new Set(returnedIds);
        const hasUnsafeOrExtraRows = safeRows.length !== fetched.length;
        const hasDuplicates = returnedSet.size !== returnedIds.length;
        const hasMissingRows = requestedIds.some((id) => !returnedSet.has(id));
        if (hasUnsafeOrExtraRows || hasDuplicates || hasMissingRows) {
          throw new L1PersistenceError(
            `predecessor authorization/coverage mismatch for ${decision.record_id}`,
          );
        }
      }

      const existing = safeRows;
      const maxVersion = existing.reduce((max, row) => Math.max(max, row.version ?? 0), 0);
      nextVersion = maxVersion + 1;
      existingMetadata = existing.map((row) => mergeMemoryMetadata(
        parseMemoryMetadata(row.metadata_json),
        row.session_id ? { source_session_ids: [row.session_id] } : undefined,
      ));
    } catch (err) {
      if (strictPersistence) {
        throw new L1PersistenceError(
          `failed to read update/merge predecessors for ${decision.record_id}`,
          { cause: err },
        );
      }
      logger?.warn?.(`${TAG} Failed to read existing memory version/metadata, defaulting to v1: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  // Determine final content, type, priority based on action
  let finalContent: string;
  let finalType: MemoryType;
  let finalPriority: number;
  let finalTimestamps: string[];
  let finalMetadata: Record<string, unknown>;

  if (decision.action === "merge" || decision.action === "update") {
    finalContent = decision.merged_content ?? memory.content;
    finalType = decision.merged_type ?? memory.type;
    finalPriority = decision.merged_priority ?? memory.priority;
    finalTimestamps = decision.merged_timestamps ?? [now];
    finalMetadata = mergeMemoryMetadata(
      ...existingMetadata,
      memory.metadata,
      decision.merged_metadata,
      { source_message_ids: memory.source_message_ids },
      sessionId ? { source_session_id: sessionId, source_session_ids: [sessionId] } : undefined,
    );
    finalMetadata = finalizeCockpitMetadataAfterDedup(finalMetadata, decision.action, decision.target_ids);
  } else {
    // store
    finalContent = memory.content;
    finalType = memory.type;
    finalPriority = memory.priority;
    finalTimestamps = [now];
    finalMetadata = mergeMemoryMetadata(
      memory.metadata,
      { source_message_ids: memory.source_message_ids },
      sessionId ? { source_session_id: sessionId, source_session_ids: [sessionId] } : undefined,
    );
  }

  const finalSourceMessageIds = Array.isArray(finalMetadata.source_message_ids)
    ? finalMetadata.source_message_ids.filter((entry): entry is string => typeof entry === "string")
    : memory.source_message_ids;

  const record: MemoryRecord = {
    id: decision.record_id || generateMemoryId(),
    content: finalContent,
    type: finalType,
    priority: finalPriority,
    scene_name: memory.scene_name,
    source_message_ids: finalSourceMessageIds,
    metadata: finalMetadata,
    timestamps: finalTimestamps,
    createdAt: now,
    updatedAt: now,
    version: nextVersion,
    sessionKey,
    sessionId: sessionId || DEFAULT_ISOLATION_ID,
    taskId,
    teamId,
    // Tenancy isolation — propagated end-to-end so SQLite / TCVDB upsert
    // can persist the row's owner. Empty strings preserve pre-isolation
    // behaviour for callers that haven't been updated yet.
    userId: userId || DEFAULT_ISOLATION_ID,
    agentId: agentId || DEFAULT_ISOLATION_ID,
  };

  if (prepareOnly) return record;

  // The strict cockpit path must use the receipt-backed batch transaction.
  // Keeping a second strict mutation path here would reintroduce delete-first
  // partial writes for any future caller that bypasses the transaction.
  if (strictPersistence) {
    throw new L1PersistenceError(
      `strict persistence for ${record.id} requires prepareOnly plus commitL1PersistencePlan`,
    );
  }

  const recordKey = memoryRecordStorageKey(record);
  const shardDate = recordKey.slice(StoragePaths.recordsDir.length, -".jsonl".length);

  // Helper: append a JSONL line
  // - standalone (no storage): write to local fs
  // - service (storage provided): write via StorageAdapter, no fs fallback
  //
  // Guard log (CR-2 fix, 2026-05-19): if storage is absent, emit a warn so any
  // missed wiring (e.g. caller forgot to pass storage in service mode) is
  // immediately visible instead of silently writing to ephemeral pod fs.
  // In standalone mode this warn is benign — the gateway auto-wires a
  // LocalStorageBackend at startup (server.ts:199-203), so storage should
  // normally be defined. Seeing this warn = caller forgot to pass it.
  const appendRecord = async (line: string) => {
    if (storage) {
      await storage.appendFile(recordKey, line);
    } else {
      logger?.warn?.(
        `${TAG} [CR-2 guard] writeMemory called without storage adapter; ` +
        `falling back to local fs at ${baseDir}/records/${shardDate}.jsonl. ` +
        `In service mode this means JSONL is written to ephemeral pod fs and ` +
        `will be lost on restart. Caller must pass 'storage' to writeMemory.`,
      );
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const recordsDir = path.default.join(baseDir, "records");
      await fs.default.mkdir(recordsDir, { recursive: true });
      await fs.default.appendFile(path.default.join(recordsDir, `${shardDate}.jsonl`), line, "utf-8");
    }
  };

  if ((decision.action === "update" || decision.action === "merge") && decision.target_ids.length > 0) {
    // Remove target records from VectorStore (real-time deletion for retrieval accuracy).
    // JSONL is append-only — old records remain in files and are cleaned up periodically
    // by memory-cleaner (which reconciles against VectorStore as source of truth).
    if (vectorStore) {
      try {
        const deleteFilter = teamId || userId || agentId || sessionId
          ? { teamId, userId, agentId, sessionId: sessionId || undefined, sessionKey }
          : undefined;
        let deleted: boolean;
        if (deleteFilter) {
          deleted = await vectorStore.deleteL1Batch(decision.target_ids, deleteFilter);
        } else {
          deleted = await vectorStore.deleteL1Batch(decision.target_ids);
        }
        if (strictPersistence && deleted !== true) {
          throw new L1PersistenceError(
            `vector target deletion returned false for ${decision.record_id}`,
          );
        }
        logger?.debug?.(`${TAG} VectorStore: deleted ${decision.target_ids.length} target record(s) for ${decision.action}`);
      } catch (err) {
        if (strictPersistence) {
          throw new L1PersistenceError(
            `vector target deletion failed for ${decision.record_id}`,
            { cause: err },
          );
        }
        logger?.warn?.(
          `${TAG} VectorStore delete failed for ${decision.action}: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
    try {
      await appendRecord(JSON.stringify(record) + "\n");
    } catch (err) {
      if (strictPersistence) {
        throw new L1PersistenceError(`JSONL append failed for ${record.id}`, { cause: err });
      }
      logger?.warn?.(`${TAG} JSONL append failed (non-fatal, VDB write continues): ${err instanceof Error ? err.message : String(err)}`);
    }
    logger?.debug?.(`${TAG} ${decision.action} memory: removed [${decision.target_ids.join(",")}] from VectorStore → ${record.id}: ${finalContent.slice(0, 80)}...`);
  } else {
    // store: append a new line
    try {
      await appendRecord(JSON.stringify(record) + "\n");
    } catch (err) {
      if (strictPersistence) {
        throw new L1PersistenceError(`JSONL append failed for ${record.id}`, { cause: err });
      }
      logger?.warn?.(`${TAG} JSONL append failed (non-fatal, VDB write continues): ${err instanceof Error ? err.message : String(err)}`);
    }
    logger?.debug?.(`${TAG} Stored memory ${record.id}: ${finalContent.slice(0, 80)}...`);
  }

  // === Vector Store dual-write ===
  if (vectorStore) {
    try {
      logger?.debug?.(
        `${TAG} [vec-dual-write] START id=${record.id}, contentLen=${record.content.length}, ` +
        `content="${record.content.slice(0, 80)}..."`,
      );

      let embedding: Float32Array | undefined;

      if (embeddingService) {
        try {
          embedding = await embeddingService.embed(record.content);
          logger?.debug?.(
            `${TAG} [vec-dual-write] Embedding OK: dims=${embedding.length}, ` +
            `norm=${Math.sqrt(Array.from(embedding).reduce((s, v) => s + v * v, 0)).toFixed(4)}`,
          );
        } catch (embedErr) {
          // Embedding failed — pass undefined to upsert() which writes
          // metadata + FTS only, skipping the vec0 table.
          logger?.warn(
            `${TAG} [vec-dual-write] Embedding FAILED for id=${record.id}, ` +
            `will write metadata only: ${embedErr instanceof Error ? embedErr.message : String(embedErr)}`,
          );
        }
      }

      const upsertOk = await vectorStore.upsertL1(record, embedding);
      if (strictPersistence && upsertOk !== true) {
        throw new L1PersistenceError(`vector upsert returned false for ${record.id}`);
      }
      logger?.debug?.(`${TAG} [vec-dual-write] upsert result=${upsertOk} id=${record.id}`);
    } catch (err) {
      if (strictPersistence) {
        throw err instanceof L1PersistenceError
          ? err
          : new L1PersistenceError(`vector upsert failed for ${record.id}`, { cause: err });
      }
      // Vector write failure should NOT block the main JSONL write
      logger?.warn?.(
        `${TAG} [vec-dual-write] FAILED (JSONL already written) id=${record.id}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  } else {
    logger?.debug?.(
      `${TAG} [vec-dual-write] SKIPPED id=${record.id}: vectorStore=${!!vectorStore}`,
    );
  }

  return record;
}

// ============================
// Helpers
// ============================

function formatLocalDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Resolve the append-only shard for an already prepared record. */
export function memoryRecordStorageKey(record: Pick<MemoryRecord, "createdAt">): string {
  return StoragePaths.record(formatLocalDate(new Date(record.createdAt)));
}
