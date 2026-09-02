import { createHash } from "node:crypto";

import type { ConversationMessage } from "../conversation/l0-recorder.js";
import type { EmbeddingService } from "../store/embedding.js";
import type { IMemoryStore, IsolationFilter } from "../store/types.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StoragePaths } from "../storage/types.js";
import type { Logger } from "../types.js";
import {
  L1PersistenceError,
  memoryRecordStorageKey,
  writeMemory,
  type DedupDecision,
  type ExtractedMemory,
  type MemoryRecord,
} from "./l1-writer.js";
import { markL1RecordPending } from "./l1-persistence-visibility.js";
import { canonicalCockpitSceneClass } from "./cockpit-ontology.js";

const TAG = "[memory-tdai][l1-persistence-transaction]";
const RECEIPT_SCHEMA = "l1-persistence-transaction-v1" as const;

function throwIfAborted(signal: AbortSignal | undefined, context: string): void {
  if (!signal?.aborted) return;
  if (signal.reason instanceof Error) throw signal.reason;
  throw new Error(`${context}: aborted`);
}

type L1PersistenceStage =
  | "prepared"
  | "jsonl_written"
  | "vectors_written"
  | "targets_deleted"
  | "visibility_committed"
  | "committed";

const L1_PERSISTENCE_STAGES: readonly L1PersistenceStage[] = [
  "prepared",
  "jsonl_written",
  "vectors_written",
  "targets_deleted",
  "visibility_committed",
  "committed",
];

interface L1PersistenceStageMarker {
  schema_version: typeof RECEIPT_SCHEMA;
  batch_id: string;
  plan_digest: string;
  stage: Exclude<L1PersistenceStage, "prepared">;
  created_at: string;
}

export interface L1PersistenceOutcome {
  extractedCount: number;
  sceneNames: string[];
  lastSceneName?: string;
  constructionQuality?: {
    model?: string;
    complete: number;
    partial: number;
    invalid: number;
    averageScore: number;
  };
}

export interface L1PersistenceScope {
  sessionKey: string;
  sessionId?: string;
  taskId?: string;
  teamId?: string;
  userId?: string;
  agentId?: string;
}

interface L1PersistenceReceipt {
  schema_version: typeof RECEIPT_SCHEMA;
  batch_id: string;
  plan_digest: string;
  stage: L1PersistenceStage;
  scope: L1PersistenceScope;
  records: MemoryRecord[];
  target_ids: string[];
  outcome: L1PersistenceOutcome;
  created_at: string;
  updated_at: string;
}

export interface L1PreparedPersistencePlan {
  batchId: string;
  scope: L1PersistenceScope;
  records: MemoryRecord[];
  targetIds: string[];
  outcome: L1PersistenceOutcome;
  preparedAt: string;
}

export interface L1CommittedPersistenceResult extends L1PersistenceOutcome {
  records: MemoryRecord[];
}

function canonicalValue(value: unknown): unknown {
  // Match JSON persistence semantics exactly: object properties whose value is
  // undefined disappear, while undefined array entries become null. Otherwise
  // a receipt that was valid before serialization can fail its own digest on
  // the first replay after process restart.
  if (value === undefined) return null;
  if (Array.isArray(value)) {
    return value.map((entry) => entry === undefined ? null : canonicalValue(entry));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .filter(([, entry]) => entry !== undefined)
      .sort(([left], [right]) => compareCanonicalStrings(left, right))
      .map(([key, entry]) => [key, canonicalValue(entry)]));
  }
  return value;
}

/** Locale-independent UTF-16 code-unit ordering for persisted identities. */
function compareCanonicalStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function digest(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(canonicalValue(value)), "utf8")
    .digest("hex");
}

/**
 * Batch identity is derived only from the exact L0 evidence set and isolation
 * scope, so a retry can find its receipt before consulting mutable L1 state or
 * invoking the model again.
 */
export function createL1PersistenceBatchId(
  messages: ConversationMessage[],
  scope: L1PersistenceScope,
): string {
  const sourceReceipt = messages
    .map((message) => ({
      id: message.id,
      role: message.role,
      timestamp: message.timestamp,
      content: message.content,
    }))
    .sort((left, right) =>
      left.timestamp - right.timestamp
      || compareCanonicalStrings(left.id, right.id)
      || compareCanonicalStrings(left.role, right.role)
      || compareCanonicalStrings(left.content, right.content)
    );
  return `l1tx_${digest({ schema: RECEIPT_SCHEMA, scope, sources: sourceReceipt })}`;
}

/** Stable per-fact ID used by replay to upsert the same logical row. */
export function createL1PersistenceRecordId(
  batchId: string,
  memory: ExtractedMemory,
): string {
  const metadata = memory.metadata as Record<string, unknown>;
  const identity = {
    batch_id: batchId,
    scene_name: canonicalCockpitSceneClass(
      memory.scene_name,
      metadata.domain,
      metadata.slot,
    ) ?? memory.scene_name,
    type: memory.type,
    source_message_ids: [...new Set(memory.source_message_ids)].sort(),
    fact: {
      schema_version: metadata.schema_version,
      record_kind: metadata.record_kind,
      domain: metadata.domain,
      slot: metadata.slot,
      state_key: metadata.state_key,
      episode_key: metadata.episode_key,
      subject: metadata.subject,
      occupant_scope: metadata.occupant_scope,
      vehicle_scope: metadata.vehicle_scope,
      seat_zone: metadata.seat_zone,
      constraint_target: metadata.constraint_target,
      value: metadata.value,
      target: metadata.target,
      unit: metadata.unit,
      condition: metadata.condition,
      trigger: metadata.trigger,
      valid_from: metadata.valid_from,
      valid_to: metadata.valid_to,
      activity_start_time: metadata.activity_start_time,
      activity_end_time: metadata.activity_end_time,
      timezone: metadata.timezone,
      relation: metadata.relation,
      action_status: metadata.action_status,
      supersedes: Array.isArray(metadata.supersedes)
        ? [...new Set(metadata.supersedes.filter((entry): entry is string => typeof entry === "string"))].sort()
        : [],
    },
  };
  return `m_l1tx_${digest(identity).slice(0, 40)}`;
}

function receiptPlanDigest(receipt: Pick<
  L1PersistenceReceipt,
  "batch_id" | "scope" | "records" | "target_ids" | "outcome" | "created_at"
>): string {
  return digest({
    batch_id: receipt.batch_id,
    scope: receipt.scope,
    records: receipt.records,
    target_ids: receipt.target_ids,
    outcome: receipt.outcome,
    created_at: receipt.created_at,
  });
}

function assertReceipt(value: unknown, expectedBatchId: string): L1PersistenceReceipt {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new L1PersistenceError(`invalid L1 transaction receipt for ${expectedBatchId}`);
  }
  const receipt = value as L1PersistenceReceipt;
  if (receipt.schema_version !== RECEIPT_SCHEMA
    || receipt.batch_id !== expectedBatchId
    || !L1_PERSISTENCE_STAGES.includes(receipt.stage)
    || !receipt.scope || typeof receipt.scope !== "object"
    || !Array.isArray(receipt.records)
    || !Array.isArray(receipt.target_ids)
    || receipt.target_ids.some((id) => typeof id !== "string" || id.length === 0)
    || !receipt.outcome || typeof receipt.outcome !== "object"
    || !Array.isArray(receipt.outcome.sceneNames)
    || typeof receipt.outcome.extractedCount !== "number"
    || typeof receipt.created_at !== "string"
    || typeof receipt.updated_at !== "string"
    || typeof receipt.plan_digest !== "string") {
    throw new L1PersistenceError(`malformed L1 transaction receipt for ${expectedBatchId}`);
  }
  const recordIds = receipt.records.map((record) => record?.id);
  if (recordIds.some((id) => typeof id !== "string" || id.length === 0)
    || new Set(recordIds).size !== recordIds.length
    || new Set(receipt.target_ids).size !== receipt.target_ids.length
    || receipt.target_ids.some((id) => recordIds.includes(id))) {
    throw new L1PersistenceError(`unsafe record/target identity set for ${expectedBatchId}`);
  }
  if (receipt.plan_digest !== receiptPlanDigest(receipt)) {
    throw new L1PersistenceError(`L1 transaction receipt digest mismatch for ${expectedBatchId}`);
  }
  return receipt;
}

function assertStageMarker(
  value: unknown,
  receipt: L1PersistenceReceipt,
  expectedStage: Exclude<L1PersistenceStage, "prepared">,
): L1PersistenceStageMarker {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new L1PersistenceError(
      `invalid L1 transaction stage marker ${expectedStage} for ${receipt.batch_id}`,
    );
  }
  const marker = value as L1PersistenceStageMarker;
  if (marker.schema_version !== RECEIPT_SCHEMA
    || marker.batch_id !== receipt.batch_id
    || marker.plan_digest !== receipt.plan_digest
    || marker.stage !== expectedStage
    || typeof marker.created_at !== "string"
    || marker.created_at.length === 0) {
    throw new L1PersistenceError(
      `malformed L1 transaction stage marker ${expectedStage} for ${receipt.batch_id}`,
    );
  }
  return marker;
}

async function readReceipt(
  storage: StorageAdapter,
  batchId: string,
  abortSignal?: AbortSignal,
): Promise<L1PersistenceReceipt | null> {
  throwIfAborted(abortSignal, `receipt read ${batchId}`);
  const raw = await storage.readFile(StoragePaths.l1Transaction(batchId));
  throwIfAborted(abortSignal, `receipt read ${batchId}`);
  if (raw === null) return null;
  try {
    const durablePlan = assertReceipt(JSON.parse(raw), batchId);
    let effective = durablePlan;
    let gapFound = false;
    const durableStageIndex = L1_PERSISTENCE_STAGES.indexOf(durablePlan.stage);
    for (let index = durableStageIndex + 1; index < L1_PERSISTENCE_STAGES.length; index += 1) {
      const stage = L1_PERSISTENCE_STAGES[index] as Exclude<L1PersistenceStage, "prepared">;
      const markerRaw = await storage.readFile(StoragePaths.l1TransactionStage(batchId, stage));
      throwIfAborted(abortSignal, `receipt marker read ${batchId}:${stage}`);
      if (markerRaw === null) {
        gapFound = true;
        continue;
      }
      if (gapFound) {
        throw new L1PersistenceError(
          `non-contiguous L1 transaction stage markers for ${batchId}`,
        );
      }
      const marker = assertStageMarker(JSON.parse(markerRaw), durablePlan, stage);
      effective = { ...effective, stage, updated_at: marker.created_at };
    }
    return effective;
  } catch (error) {
    throw error instanceof L1PersistenceError
      ? error
      : new L1PersistenceError(`failed to parse L1 transaction receipt for ${batchId}`, { cause: error });
  }
}

async function writeReceipt(
  storage: StorageAdapter,
  receipt: L1PersistenceReceipt,
  stage: L1PersistenceStage,
  abortSignal?: AbortSignal,
): Promise<L1PersistenceReceipt> {
  throwIfAborted(abortSignal, `receipt marker write ${receipt.batch_id}:${stage}`);
  if (stage === "prepared") {
    throw new L1PersistenceError(`prepared is an immutable plan, not a stage marker`);
  }
  const currentIndex = L1_PERSISTENCE_STAGES.indexOf(receipt.stage);
  const requestedIndex = L1_PERSISTENCE_STAGES.indexOf(stage);
  if (requestedIndex <= currentIndex) return receipt;
  if (requestedIndex !== currentIndex + 1) {
    throw new L1PersistenceError(
      `refusing non-monotonic L1 transaction stage ${receipt.stage} -> ${stage}`,
    );
  }
  const marker: L1PersistenceStageMarker = {
    schema_version: RECEIPT_SCHEMA,
    batch_id: receipt.batch_id,
    plan_digest: receipt.plan_digest,
    stage,
    created_at: new Date().toISOString(),
  };
  const markerPath = StoragePaths.l1TransactionStage(receipt.batch_id, stage);
  await storage.writeFileIfAbsent(markerPath, `${JSON.stringify(marker)}\n`);
  throwIfAborted(abortSignal, `receipt marker write ${receipt.batch_id}:${stage}`);

  // Read back the winner's immutable marker. This handles concurrent workers
  // and a lost acknowledgement without trusting this process's local value.
  const durableRaw = await storage.readFile(markerPath);
  throwIfAborted(abortSignal, `receipt marker verification ${receipt.batch_id}:${stage}`);
  if (durableRaw === null) {
    throw new L1PersistenceError(
      `L1 transaction stage marker disappeared for ${receipt.batch_id}:${stage}`,
    );
  }
  const durableMarker = assertStageMarker(JSON.parse(durableRaw), receipt, stage);
  return { ...receipt, stage, updated_at: durableMarker.created_at };
}

function receiptResult(receipt: L1PersistenceReceipt): L1CommittedPersistenceResult {
  return { ...receipt.outcome, records: receipt.records };
}

/**
 * Authorization boundary for predecessor deletion. Sessions are provenance:
 * a later session can replace an earlier state owned by the same tenant,
 * agent and task. Never widen across those authority dimensions.
 */
function predecessorAuthorityFilter(scope: L1PersistenceScope): IsolationFilter | undefined {
  return scope.teamId || scope.userId || scope.agentId || scope.taskId
    ? {
      teamId: scope.teamId,
      userId: scope.userId,
      agentId: scope.agentId,
      taskId: scope.taskId,
    }
    : undefined;
}

async function replayReceipt(params: {
  receipt: L1PersistenceReceipt;
  storage: StorageAdapter;
  vectorStore: IMemoryStore;
  embeddingService?: EmbeddingService;
  logger?: Logger;
  abortSignal?: AbortSignal;
}): Promise<L1CommittedPersistenceResult> {
  const { storage, vectorStore, embeddingService, logger, abortSignal } = params;
  let { receipt } = params;
  throwIfAborted(abortSignal, `transaction replay ${receipt.batch_id}`);

  if (receipt.stage === "committed") return receiptResult(receipt);

  if (!vectorStore.commitL1Transaction || !vectorStore.isL1TransactionCommitted) {
    throw new L1PersistenceError(
      `vector store lacks strict L1 transaction visibility for ${receipt.batch_id}`,
    );
  }

  if (receipt.stage === "prepared") {
    const byShard = new Map<string, MemoryRecord[]>();
    for (const record of receipt.records) {
      const key = memoryRecordStorageKey(record);
      byShard.set(key, [...(byShard.get(key) ?? []), record]);
    }
    for (const [key, records] of byShard) {
      throwIfAborted(abortSignal, `JSONL append ${receipt.batch_id}`);
      await storage.appendFile(key, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
      throwIfAborted(abortSignal, `JSONL append ${receipt.batch_id}`);
    }
    receipt = await writeReceipt(storage, receipt, "jsonl_written", abortSignal);
  }

  if (receipt.stage === "jsonl_written") {
    try {
      for (const record of receipt.records) {
        throwIfAborted(abortSignal, `vector preparation ${receipt.batch_id}`);
        let embedding: Float32Array | undefined;
        if (embeddingService) {
          try {
            embedding = await embeddingService.embed(record.content);
            throwIfAborted(abortSignal, `embedding ${receipt.batch_id}`);
          } catch (error) {
            logger?.warn?.(
              `${TAG} embedding failed for ${record.id}; committing metadata/FTS only: `
              + `${error instanceof Error ? error.message : String(error)}`,
            );
          }
        }
        throwIfAborted(abortSignal, `vector upsert ${receipt.batch_id}`);
        const upserted = await vectorStore.upsertL1(
          markL1RecordPending(record, receipt.batch_id),
          embedding,
        );
        if (upserted !== true) {
          throw new L1PersistenceError(`vector upsert returned false for ${record.id}`);
        }
        throwIfAborted(abortSignal, `vector upsert ${receipt.batch_id}`);
      }
    } catch (error) {
      // Never delete deterministic pending IDs here. Another worker may be
      // replaying the same immutable receipt and may already have completed
      // those exact upserts. Deleting by record ID cannot distinguish this
      // failed attempt from the successful one and can therefore erase a row
      // after the other worker commits visibility. Partial rows are tagged
      // with this transaction ID, remain invisible without the commit
      // sentinel, and are safely overwritten by the next idempotent replay.
      logger?.warn?.(
        `${TAG} leaving deterministic pending rows for replay batch=${receipt.batch_id}: `
        + `${error instanceof Error ? error.message : String(error)}`,
      );
      throwIfAborted(abortSignal, `vector replay ${receipt.batch_id}`);
      throw error instanceof L1PersistenceError
        ? error
        : new L1PersistenceError(`vector batch replay failed for ${receipt.batch_id}`, { cause: error });
    }
    receipt = await writeReceipt(storage, receipt, "vectors_written", abortSignal);
  }

  if (receipt.stage === "vectors_written") {
    if (receipt.target_ids.length > 0) {
      throwIfAborted(abortSignal, `target deletion ${receipt.batch_id}`);
      let deleted: boolean;
      try {
        deleted = await vectorStore.deleteL1Batch(
          receipt.target_ids,
          predecessorAuthorityFilter(receipt.scope),
        );
      } catch (error) {
        throwIfAborted(abortSignal, `target deletion ${receipt.batch_id}`);
        throw new L1PersistenceError(
          `target batch deletion failed for ${receipt.batch_id}`,
          { cause: error },
        );
      }
      if (deleted !== true) {
        throw new L1PersistenceError(`target batch deletion returned false for ${receipt.batch_id}`);
      }
      throwIfAborted(abortSignal, `target deletion ${receipt.batch_id}`);
    }
    receipt = await writeReceipt(storage, receipt, "targets_deleted", abortSignal);
  }

  if (receipt.stage === "targets_deleted") {
    let visible: boolean;
    try {
      throwIfAborted(abortSignal, `visibility commit ${receipt.batch_id}`);
      visible = await vectorStore.isL1TransactionCommitted(receipt.batch_id);
      if (!visible) {
        const committed = await vectorStore.commitL1Transaction(receipt.batch_id);
        throwIfAborted(abortSignal, `visibility commit ${receipt.batch_id}`);
        if (committed !== true) {
          visible = await vectorStore.isL1TransactionCommitted(receipt.batch_id);
          throwIfAborted(abortSignal, `visibility verification ${receipt.batch_id}`);
          if (!visible) {
            throw new L1PersistenceError(
              `vector visibility commit returned false for ${receipt.batch_id}`,
            );
          }
        } else {
          visible = await vectorStore.isL1TransactionCommitted(receipt.batch_id);
          throwIfAborted(abortSignal, `visibility verification ${receipt.batch_id}`);
        }
      }
    } catch (error) {
      throwIfAborted(abortSignal, `visibility commit ${receipt.batch_id}`);
      throw error instanceof L1PersistenceError
        ? error
        : new L1PersistenceError(
          `vector visibility commit failed for ${receipt.batch_id}`,
          { cause: error },
        );
    }
    if (!visible) {
      throw new L1PersistenceError(
        `vector visibility commit could not be verified for ${receipt.batch_id}`,
      );
    }
    receipt = await writeReceipt(storage, receipt, "visibility_committed", abortSignal);
  }

  if (receipt.stage === "visibility_committed") {
    receipt = await writeReceipt(storage, receipt, "committed", abortSignal);
  }

  logger?.info?.(
    `${TAG} committed batch=${receipt.batch_id} records=${receipt.records.length} targets=${receipt.target_ids.length}`,
  );
  return receiptResult(receipt);
}

/** Resume before model construction; null means no durable transaction exists. */
export async function resumeL1PersistenceTransaction(params: {
  batchId: string;
  storage: StorageAdapter;
  vectorStore: IMemoryStore;
  embeddingService?: EmbeddingService;
  logger?: Logger;
  abortSignal?: AbortSignal;
}): Promise<L1CommittedPersistenceResult | null> {
  const receipt = await readReceipt(params.storage, params.batchId, params.abortSignal);
  if (!receipt) return null;
  return replayReceipt({ ...params, receipt });
}

export async function prepareL1PersistencePlan(params: {
  batchId: string;
  scope: L1PersistenceScope;
  memoriesWithIds: Array<ExtractedMemory & { record_id: string }>;
  decisions: DedupDecision[];
  baseDir: string;
  vectorStore: IMemoryStore;
  outcome: L1PersistenceOutcome;
  logger?: Logger;
  abortSignal?: AbortSignal;
}): Promise<L1PreparedPersistencePlan> {
  const {
    batchId,
    scope,
    memoriesWithIds,
    decisions,
    baseDir,
    vectorStore,
    outcome,
    logger,
    abortSignal,
  } = params;
  throwIfAborted(abortSignal, `transaction planning ${batchId}`);
  const decisionMap = new Map(decisions.map((decision) => [decision.record_id, decision]));
  const preparedAt = new Date().toISOString();
  const records: MemoryRecord[] = [];
  const targetIds = new Set<string>();

  for (const memory of memoriesWithIds) {
    throwIfAborted(abortSignal, `transaction planning ${batchId}`);
    const decision = decisionMap.get(memory.record_id) ?? {
      record_id: memory.record_id,
      action: "store" as const,
      target_ids: [],
    };
    const record = await writeMemory({
      memory,
      decision,
      baseDir,
      sessionKey: scope.sessionKey,
      sessionId: scope.sessionId,
      taskId: scope.taskId,
      teamId: scope.teamId,
      userId: scope.userId,
      agentId: scope.agentId,
      logger,
      vectorStore,
      strictPersistence: true,
      prepareOnly: true,
      nowIso: preparedAt,
    });
    if (!record) continue;
    throwIfAborted(abortSignal, `transaction planning ${batchId}`);
    records.push(record);
    if (decision.action === "update" || decision.action === "merge") {
      decision.target_ids.forEach((id) => targetIds.add(id));
    }
  }

  const recordIds = records.map((record) => record.id);
  const newIds = new Set(recordIds);
  if (newIds.size !== recordIds.length) {
    throw new L1PersistenceError(`transaction ${batchId} contains duplicate deterministic new IDs`);
  }
  if (targetIds.size > 0 && [...targetIds].some((id) => newIds.has(id))) {
    throw new L1PersistenceError(`transaction ${batchId} would delete one of its deterministic new rows`);
  }

  return {
    batchId,
    scope,
    records,
    targetIds: [...targetIds].sort(),
    outcome,
    preparedAt,
  };
}

export async function commitL1PersistencePlan(params: {
  plan: L1PreparedPersistencePlan;
  storage: StorageAdapter;
  vectorStore: IMemoryStore;
  embeddingService?: EmbeddingService;
  logger?: Logger;
  abortSignal?: AbortSignal;
}): Promise<L1CommittedPersistenceResult> {
  throwIfAborted(params.abortSignal, `transaction plan commit ${params.plan.batchId}`);
  const receipt: L1PersistenceReceipt = {
    schema_version: RECEIPT_SCHEMA,
    batch_id: params.plan.batchId,
    plan_digest: "",
    stage: "prepared",
    scope: params.plan.scope,
    records: params.plan.records,
    target_ids: params.plan.targetIds,
    outcome: params.plan.outcome,
    created_at: params.plan.preparedAt,
    updated_at: params.plan.preparedAt,
  };
  receipt.plan_digest = receiptPlanDigest(receipt);
  await params.storage.writeFileIfAbsent(
    StoragePaths.l1Transaction(receipt.batch_id),
    `${JSON.stringify(receipt)}\n`,
  );
  throwIfAborted(params.abortSignal, `transaction plan commit ${params.plan.batchId}`);
  const durable = await readReceipt(params.storage, params.plan.batchId, params.abortSignal);
  if (!durable) {
    throw new L1PersistenceError(`transaction plan disappeared for ${params.plan.batchId}`);
  }
  if (durable.plan_digest !== receipt.plan_digest) {
    throw new L1PersistenceError(`conflicting transaction plan for ${params.plan.batchId}`);
  }
  return replayReceipt({ ...params, receipt: durable });
}

/** Load receipts for canonical JSONL fallback filtering. */
export async function readL1PersistenceReceipts(
  storage: StorageAdapter,
): Promise<Array<Pick<L1PersistenceReceipt, "stage" | "records" | "target_ids">>> {
  // Receipt availability is part of the JSONL visibility contract. A backend
  // listing error must not be interpreted as "there are no transactions",
  // because that would expose pending rows and resurrect committed targets.
  const names = await storage.readdirNames(StoragePaths.l1TransactionsDir, ".json");
  const receipts: Array<Pick<L1PersistenceReceipt, "stage" | "records" | "target_ids">> = [];
  for (const name of names) {
    const match = /^(l1tx_[^.]+)\.json$/u.exec(name);
    if (!match) continue;
    const batchId = match[1];
    const receipt = await readReceipt(storage, batchId);
    if (receipt) receipts.push(receipt);
  }
  return receipts;
}
