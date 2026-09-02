import { createHash } from "node:crypto";

import type { MemoryRecord } from "./l1-writer.js";

/** Reserved metadata is written only to the database copy, never JSONL. */
export const L1_TRANSACTION_METADATA_KEY = "__tdai_l1_transaction";
export const L1_PENDING_MEMORY_TYPE = "l1_transaction_pending";
export const L1_COMMIT_MEMORY_TYPE = "l1_transaction_commit";

export function markL1RecordPending(record: MemoryRecord, batchId: string): MemoryRecord {
  return {
    ...record,
    metadata: {
      ...(record.metadata as Record<string, unknown>),
      [L1_TRANSACTION_METADATA_KEY]: batchId,
    },
  };
}

export function l1TransactionIdFromMetadata(metadata: unknown): string | undefined {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) return undefined;
  const value = (metadata as Record<string, unknown>)[L1_TRANSACTION_METADATA_KEY];
  return typeof value === "string" && value.startsWith("l1tx_") ? value : undefined;
}

export function inspectL1MetadataJson(raw: unknown): {
  transactionId?: string;
  sanitizedJson: string;
  validJson: boolean;
  hasReservedMarker: boolean;
} {
  if (typeof raw !== "string") {
    return { sanitizedJson: "{}", validJson: false, hasReservedMarker: false };
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { sanitizedJson: raw, validJson: false, hasReservedMarker: false };
    }
    const metadata = parsed as Record<string, unknown>;
    const hasReservedMarker = Object.prototype.hasOwnProperty.call(metadata, L1_TRANSACTION_METADATA_KEY);
    const transactionId = l1TransactionIdFromMetadata(metadata);
    if (!hasReservedMarker) {
      return { sanitizedJson: raw, validJson: true, hasReservedMarker: false };
    }
    const sanitized = { ...metadata };
    delete sanitized[L1_TRANSACTION_METADATA_KEY];
    return {
      transactionId,
      sanitizedJson: JSON.stringify(sanitized),
      validJson: true,
      hasReservedMarker: true,
    };
  } catch {
    return { sanitizedJson: raw, validJson: false, hasReservedMarker: false };
  }
}

export function createL1TransactionCommitRecordId(batchId: string): string {
  return `l1txc_${createHash("sha256").update(batchId, "utf8").digest("hex")}`;
}
