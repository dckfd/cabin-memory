/**
 * L1 Memory Reader: reads persisted L1 memory records.
 *
 * Provides two data paths:
 *
 * 1. **SQLite** (preferred): `queryMemoryRecords()` — uses VectorStore's `queryL1Records()`
 *    with composite indexes on (session_key, updated_time) and (session_id, updated_time)
 *    for efficient session-scoped and time-range queries.
 *
 * 2. **JSONL** (fallback): `readMemoryRecords()` / `readAllMemoryRecords()` — reads from
 *    `records/YYYY-MM-DD.jsonl` files. Used when VectorStore is unavailable or degraded.
 */

import type { MemoryRecord, MemoryType, EpisodicMetadata } from "./l1-writer.js";
import type { IMemoryStore, L1RecordRow, L1QueryFilter } from "../store/types.js";
import { StorageAdapter } from "../storage/adapter.js";
import { LocalStorageBackend } from "../storage/local-backend.js";
import { StoragePaths } from "../storage/types.js";
import { readL1PersistenceReceipts } from "./l1-persistence-transaction.js";

// Re-export types that readers need
export type { MemoryRecord, MemoryType, EpisodicMetadata } from "./l1-writer.js";
export type { L1QueryFilter } from "../store/types.js";
import type { Logger } from "../types.js";

const TAG = "[memory-tdai] [l1-reader]";

function preferLatestRecord(left: MemoryRecord, right: MemoryRecord): MemoryRecord {
  const leftVersion = typeof left.version === "number" ? left.version : 0;
  const rightVersion = typeof right.version === "number" ? right.version : 0;
  if (leftVersion !== rightVersion) return leftVersion > rightVersion ? left : right;
  const leftTime = left.updatedAt || left.createdAt || "";
  const rightTime = right.updatedAt || right.createdAt || "";
  return leftTime.localeCompare(rightTime) >= 0 ? left : right;
}

/**
 * JSONL is append-only, so receipt-backed retries may leave duplicate lines
 * and committed update batches leave predecessor rows. Collapse deterministic
 * IDs and apply committed target tombstones; hide rows from any uncommitted
 * transaction so fallback reads never authorize a partial batch.
 */
async function canonicalizeJsonlRecords(
  records: MemoryRecord[],
  storage: StorageAdapter,
): Promise<MemoryRecord[]> {
  const pendingRecordIds = new Set<string>();
  const committedTargetIds = new Set<string>();
  const receipts = await readL1PersistenceReceipts(storage);
  for (const receipt of receipts) {
    if (receipt.stage === "visibility_committed" || receipt.stage === "committed") {
      receipt.target_ids.forEach((id) => committedTargetIds.add(id));
    } else {
      receipt.records.forEach((record) => pendingRecordIds.add(record.id));
    }
  }
  const byId = new Map<string, MemoryRecord>();
  for (const record of records) {
    if (!record?.id || pendingRecordIds.has(record.id) || committedTargetIds.has(record.id)) continue;
    const previous = byId.get(record.id);
    byId.set(record.id, previous ? preferLatestRecord(previous, record) : record);
  }
  return [...byId.values()];
}

// ============================
// SQLite-based queries (preferred)
// ============================

/**
 * Query L1 memory records from SQLite via VectorStore.
 *
 * This is the **preferred** read path — it uses the composite index
 * `idx_l1_session_updated(session_id, updated_time)` for efficient
 * session-scoped and time-range queries.
 *
 * All timestamps are UTC ISO 8601 (as stored by l1-writer's dual-write).
 *
 * Falls back to empty array if VectorStore is null or degraded.
 */
export async function queryMemoryRecords(
  vectorStore: IMemoryStore | null | undefined,
  filter?: L1QueryFilter,
  logger?: Logger,
): Promise<MemoryRecord[]> {
  if (!vectorStore) {
    logger?.warn(`${TAG} queryMemoryRecords: no VectorStore available, returning empty`);
    return [];
  }

  const rows = await vectorStore.queryL1Records(filter);
  return rows.map(rowToMemoryRecord);
}

/**
 * Convert a raw SQLite L1RecordRow to a MemoryRecord (same shape as JSONL records).
 */
function rowToMemoryRecord(row: L1RecordRow): MemoryRecord {
  let metadata: EpisodicMetadata | Record<string, never> = {};
  try {
    metadata = JSON.parse(row.metadata_json) as EpisodicMetadata | Record<string, never>;
  } catch {
    // malformed JSON — use empty object
  }

  // Reconstruct timestamps array from timestamp_start / timestamp_end
  const timestamps: string[] = [];
  if (row.timestamp_str) timestamps.push(row.timestamp_str);
  if (row.timestamp_start && row.timestamp_start !== row.timestamp_str) timestamps.push(row.timestamp_start);
  if (row.timestamp_end && row.timestamp_end !== row.timestamp_str && row.timestamp_end !== row.timestamp_start) {
    timestamps.push(row.timestamp_end);
  }

  return {
    id: row.record_id,
    content: row.content,
    type: row.type as MemoryType,
    priority: row.priority,
    scene_name: row.scene_name,
    // Provenance is mirrored into metadata_json by l1-writer so both SQLite
    // and TCVDB can reconstruct it without a storage-schema migration.
    source_message_ids: Array.isArray(metadata.source_message_ids)
      ? metadata.source_message_ids.filter((entry): entry is string => typeof entry === "string")
      : [],
    metadata,
    timestamps,
    createdAt: row.created_time,
    updatedAt: row.updated_time,
    version: row.version ?? 0,
    sessionKey: row.session_key,
    sessionId: row.session_id,
    taskId: row.task_id,
    teamId: row.team_id,
    userId: row.user_id,
    agentId: row.agent_id,
  };
}

// ============================
// JSONL-based reads (fallback)
// ============================

/**
 * Read all memory records for a session from JSONL files.
 *
 * Current naming mode:
 * - Daily merged file: records/YYYY-MM-DD.jsonl (all sessions in one file)
 */
export async function readMemoryRecords(
  sessionKey: string,
  baseDir: string,
  logger?: Logger,
  storage?: StorageAdapter,
): Promise<MemoryRecord[]> {
  const dateFilePattern = /^\d{4}-\d{2}-\d{2}\.jsonl$/;

  let entries: string[];
  try {
    if (storage) {
      entries = await storage.readdirNames(StoragePaths.recordsDir, ".jsonl");
    } else {
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const recordsDir = path.default.join(baseDir, "records");
      const dirEntries = await fs.default.readdir(recordsDir, { withFileTypes: true });
      entries = dirEntries
        .filter((entry: import("node:fs").Dirent) => entry.isFile() && dateFilePattern.test(entry.name))
        .map((entry: import("node:fs").Dirent) => entry.name);
    }
  } catch {
    // Directory doesn't exist yet
    return [];
  }

  const targetFiles = entries
    .filter((name) => dateFilePattern.test(name))
    .sort();

  if (targetFiles.length === 0) {
    return [];
  }

  const records: MemoryRecord[] = [];

  for (const fileName of targetFiles) {
    let raw: string | null;
    try {
      if (storage) {
        raw = await storage.readFile(`${StoragePaths.recordsDir}${fileName}`);
      } else {
        const fs = await import("node:fs/promises");
        const path = await import("node:path");
        raw = await fs.default.readFile(path.default.join(baseDir, "records", fileName), "utf-8");
      }
    } catch {
      logger?.warn?.(`${TAG} Failed to read L1 file: ${fileName}`);
      continue;
    }

    if (!raw) continue;

    const lines = raw.split("\n").filter((line) => line.trim());
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      try {
        const parsed = JSON.parse(line) as Partial<MemoryRecord>;
        if (parsed.sessionKey !== sessionKey) {
          continue;
        }
        records.push(parsed as MemoryRecord);
      } catch {
        logger?.warn?.(`${TAG} Skipping malformed JSONL line in ${fileName}:${i + 1}`);
      }
    }
  }

  let canonicalRecords: MemoryRecord[];
  try {
    const receiptStorage = storage ?? new StorageAdapter(new LocalStorageBackend(baseDir));
    canonicalRecords = await canonicalizeJsonlRecords(records, receiptStorage);
  } catch (error) {
    logger?.error?.(
      `${TAG} Failed to canonicalize session JSONL records: `
      + `${error instanceof Error ? error.message : String(error)}`,
    );
    return [];
  }
  canonicalRecords.sort((a, b) => {
    const ta = a.updatedAt || a.createdAt || "";
    const tb = b.updatedAt || b.createdAt || "";
    return ta.localeCompare(tb);
  });

  return canonicalRecords;
}

/**
 * Read ALL memory records across all session JSONL files.
 */
export async function readAllMemoryRecords(
  baseDir: string,
  logger?: Logger,
  storage?: StorageAdapter,
): Promise<MemoryRecord[]> {
  try {
    let files: string[];
    if (storage) {
      files = await storage.readdirNames(StoragePaths.recordsDir, ".jsonl");
    } else {
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const recordsDir = path.default.join(baseDir, "records");
      files = (await fs.default.readdir(recordsDir)).filter((f: string) => f.endsWith(".jsonl"));
    }

    const allRecords: MemoryRecord[] = [];

    for (const file of files) {
      try {
        let raw: string | null;
        if (storage) {
          raw = await storage.readFile(`${StoragePaths.recordsDir}${file}`);
        } else {
          const fs = await import("node:fs/promises");
          const path = await import("node:path");
          raw = await fs.default.readFile(path.default.join(baseDir, "records", file), "utf-8");
        }
        if (!raw) continue;
        const lines = raw.split("\n").filter((line: string) => line.trim());
        for (const line of lines) {
          try {
            allRecords.push(JSON.parse(line) as MemoryRecord);
          } catch {
            logger?.warn?.(`${TAG} Skipping malformed JSONL line in ${file}`);
          }
        }
      } catch {
        logger?.warn?.(`${TAG} Failed to read ${file}`);
      }
    }

    const receiptStorage = storage ?? new StorageAdapter(new LocalStorageBackend(baseDir));
    const canonicalRecords = await canonicalizeJsonlRecords(allRecords, receiptStorage);
    canonicalRecords.sort((a, b) => {
      const ta = a.updatedAt || a.createdAt || "";
      const tb = b.updatedAt || b.createdAt || "";
      return ta.localeCompare(tb);
    });

    return canonicalRecords;

  } catch (error) {
    // Missing records are an ordinary empty state. Receipt corruption or an
    // unavailable receipt listing also lands here deliberately: return no
    // evidence (fail closed) and keep the cause observable.
    logger?.error?.(
      `${TAG} Failed to canonicalize JSONL records: `
      + `${error instanceof Error ? error.message : String(error)}`,
    );
    return [];
  }
}

// ============================
// Helpers
// ============================

function sanitizeFilename(name: string): string {
  return name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_");
}
