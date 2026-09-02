import { describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { parseConfig } from "../../config.js";
import { createL1Runner } from "../../utils/pipeline-factory.js";
import type { ConversationMessage } from "../conversation/l0-recorder.js";
import type { IMemoryStore, IsolationFilter } from "../store/types.js";
import type { StorageAdapter } from "../storage/adapter.js";
import { StorageAdapter as ConcreteStorageAdapter } from "../storage/adapter.js";
import { LocalStorageBackend } from "../storage/local-backend.js";
import { StoragePaths } from "../storage/types.js";
import { extractL1Memories } from "./l1-extractor.js";
import { readAllMemoryRecords, readMemoryRecords } from "./l1-reader.js";
import {
  commitL1PersistencePlan,
  createL1PersistenceBatchId,
  createL1PersistenceRecordId,
  prepareL1PersistencePlan,
  resumeL1PersistenceTransaction,
  type L1PreparedPersistencePlan,
  type L1PersistenceScope,
} from "./l1-persistence-transaction.js";
import {
  writeMemory,
  type ExtractedMemory,
  type MemoryRecord,
} from "./l1-writer.js";

type ReceiptStage =
  | "prepared"
  | "jsonl_written"
  | "vectors_written"
  | "targets_deleted"
  | "visibility_committed"
  | "committed";

class FaultStorage {
  readonly files = new Map<string, string>();
  readonly appendCalls: Array<{ key: string; content: string }> = [];
  failAppendMode: "before" | "after" | null = null;
  failReceiptStageOnce: ReceiptStage | null = null;
  failReceiptList = false;

  async readFile(key: string): Promise<string | null> {
    return this.files.get(key) ?? null;
  }

  async writeFile(key: string, content: string | Buffer): Promise<void> {
    const text = Buffer.isBuffer(content) ? content.toString("utf8") : content;
    if (key.startsWith(StoragePaths.l1TransactionsDir) && this.failReceiptStageOnce) {
      const stage = (JSON.parse(text) as { stage?: ReceiptStage }).stage;
      if (stage === this.failReceiptStageOnce) {
        this.failReceiptStageOnce = null;
        throw new Error(`synthetic receipt ${stage} failure`);
      }
    }
    this.files.set(key, text);
  }

  async writeFileIfAbsent(key: string, content: string | Buffer): Promise<boolean> {
    const text = Buffer.isBuffer(content) ? content.toString("utf8") : content;
    if (this.files.has(key)) return false;
    if (key.startsWith(StoragePaths.l1TransactionsDir) && this.failReceiptStageOnce) {
      const stage = (JSON.parse(text) as { stage?: ReceiptStage }).stage;
      if (stage === this.failReceiptStageOnce) {
        this.failReceiptStageOnce = null;
        throw new Error(`synthetic receipt ${stage} failure`);
      }
    }
    this.files.set(key, text);
    return true;
  }

  async appendFile(key: string, content: string): Promise<void> {
    this.appendCalls.push({ key, content });
    if (this.failAppendMode === "before") {
      this.failAppendMode = null;
      throw new Error("synthetic append failure before commit");
    }
    this.files.set(key, `${this.files.get(key) ?? ""}${content}`);
    if (this.failAppendMode === "after") {
      this.failAppendMode = null;
      throw new Error("synthetic lost append acknowledgement");
    }
  }

  async readdirNames(prefix: string, suffix?: string): Promise<string[]> {
    if (this.failReceiptList && prefix === StoragePaths.l1TransactionsDir) {
      throw new Error("synthetic receipt listing failure");
    }
    return [...this.files.keys()]
      .filter((key) => key.startsWith(prefix) && (!suffix || key.endsWith(suffix)))
      .map((key) => key.slice(prefix.length));
  }
}

type FailureMode = "false" | "throw";

class FaultVectorStore {
  readonly rows = new Map<string, MemoryRecord>();
  readonly upsertCalls: string[] = [];
  readonly deleteCalls: Array<{ ids: string[]; filter?: IsolationFilter }> = [];
  failUpsertAt: number | null = null;
  failUpsertMode: FailureMode = "false";
  commitFailedUpsert = true;
  deleteFailures: FailureMode[] = [];
  readonly committedTransactions = new Set<string>();
  readonly visibilityCommitCalls: string[] = [];
  failVisibilityCommit: FailureMode | null = null;
  afterUpsert?: (record: MemoryRecord) => void | Promise<void>;

  async upsertL1(record: MemoryRecord): Promise<boolean> {
    this.upsertCalls.push(record.id);
    const shouldFail = this.failUpsertAt === this.upsertCalls.length;
    if (!shouldFail || this.commitFailedUpsert) this.rows.set(record.id, record);
    await this.afterUpsert?.(record);
    if (!shouldFail) return true;
    if (this.failUpsertMode === "throw") throw new Error("synthetic vector upsert failure");
    return false;
  }

  async deleteL1Batch(ids: string[], filter?: IsolationFilter): Promise<boolean> {
    this.deleteCalls.push({ ids: [...ids], filter });
    const failure = this.deleteFailures.shift();
    if (failure === "throw") throw new Error("synthetic vector delete failure");
    if (failure === "false") return false;
    ids.forEach((id) => this.rows.delete(id));
    return true;
  }

  async queryL1Records(): Promise<[]> {
    return [];
  }

  async commitL1Transaction(batchId: string): Promise<boolean> {
    this.visibilityCommitCalls.push(batchId);
    const failure = this.failVisibilityCommit;
    this.failVisibilityCommit = null;
    if (failure === "throw") throw new Error("synthetic visibility commit failure");
    if (failure === "false") return false;
    this.committedTransactions.add(batchId);
    return true;
  }

  async isL1TransactionCommitted(batchId: string): Promise<boolean> {
    return this.committedTransactions.has(batchId);
  }
}

const scope: L1PersistenceScope = {
  sessionKey: "cockpit-session-key",
  sessionId: "cockpit-session-id",
  taskId: "trip-task",
  teamId: "fleet-team",
  userId: "driver-user",
  agentId: "cockpit-agent",
};

function memoryRecord(id: string, content = id): MemoryRecord {
  const timestamp = "2026-09-01T08:00:00.000Z";
  return {
    id,
    content,
    type: "episodic",
    priority: 80,
    scene_name: "navigation",
    source_message_ids: [`source-${id}`],
    metadata: {
      schema_version: "cockpit-state-v1",
      record_kind: "event",
      domain: "navigation",
      slot: "destination",
      value: content,
      source_message_ids: [`source-${id}`],
    },
    timestamps: [timestamp],
    createdAt: timestamp,
    updatedAt: timestamp,
    version: 1,
    sessionKey: scope.sessionKey,
    sessionId: scope.sessionId!,
    taskId: scope.taskId,
    teamId: scope.teamId,
    userId: scope.userId,
    agentId: scope.agentId,
  };
}

function persistencePlan(
  batchId: string,
  records: MemoryRecord[],
  targetIds: string[] = [],
): L1PreparedPersistencePlan {
  return {
    batchId,
    scope,
    records,
    targetIds,
    outcome: {
      extractedCount: records.length,
      sceneNames: ["navigation"],
      lastSceneName: "navigation",
    },
    preparedAt: "2026-09-01T08:00:00.000Z",
  };
}

function storageAdapter(storage: FaultStorage): StorageAdapter {
  return storage as unknown as StorageAdapter;
}

function memoryStore(store: FaultVectorStore): IMemoryStore {
  return store as unknown as IMemoryStore;
}

function receiptStage(storage: FaultStorage, batchId: string): ReceiptStage {
  const raw = storage.files.get(StoragePaths.l1Transaction(batchId));
  if (!raw) throw new Error(`missing receipt ${batchId}`);
  let stage = (JSON.parse(raw) as { stage: ReceiptStage }).stage;
  const stages: ReceiptStage[] = [
    "jsonl_written",
    "vectors_written",
    "targets_deleted",
    "visibility_committed",
    "committed",
  ];
  for (const candidate of stages) {
    if (storage.files.has(StoragePaths.l1TransactionStage(batchId, candidate))) stage = candidate;
    else break;
  }
  return stage;
}

describe("L1 persistence transaction identity", () => {
  it("is order-insensitive but binds every isolation and exact evidence axis", () => {
    const messages: ConversationMessage[] = [
      { id: "u-1", role: "user", content: "去虹桥机场", timestamp: 100 },
      { id: "a-1", role: "assistant", content: "好的", timestamp: 101 },
    ];
    const baseline = createL1PersistenceBatchId(messages, scope);

    expect(createL1PersistenceBatchId([...messages].reverse(), scope)).toBe(baseline);
    expect(createL1PersistenceBatchId(messages, { ...scope, taskId: "other-task" })).not.toBe(baseline);
    expect(createL1PersistenceBatchId(messages, { ...scope, userId: "other-user" })).not.toBe(baseline);
    expect(createL1PersistenceBatchId(
      [{ ...messages[0], content: "去虹橋機場" }, messages[1]],
      scope,
    )).not.toBe(baseline);
    expect(createL1PersistenceBatchId(
      [{ ...messages[0], id: "u-2" }, messages[1]],
      scope,
    )).not.toBe(baseline);
  });

  it("keeps record IDs stable across metadata/source ordering and changes them for evidence or Unicode", () => {
    const batchId = "l1tx_identity";
    const first: ExtractedMemory = {
      content: "导航到 Café",
      type: "episodic",
      priority: 80,
      scene_name: "navigation",
      source_message_ids: ["u-2", "u-1"],
      metadata: {
        domain: "navigation",
        slot: "destination",
        value: "Café",
        relation: "asserted",
        supersedes: ["old-2", "old-1"],
      },
    };
    const reordered: ExtractedMemory = {
      ...first,
      source_message_ids: ["u-1", "u-2"],
      metadata: {
        supersedes: ["old-1", "old-2"],
        relation: "asserted",
        value: "Café",
        slot: "destination",
        domain: "navigation",
      },
    };

    const baseline = createL1PersistenceRecordId(batchId, first);
    expect(createL1PersistenceRecordId(batchId, reordered)).toBe(baseline);
    expect(createL1PersistenceRecordId(batchId, {
      ...first,
      scene_name: "model-authored-route-session-label",
    })).toBe(baseline);
    expect(createL1PersistenceRecordId(batchId, {
      ...first,
      source_message_ids: ["u-1"],
    })).not.toBe(baseline);
    expect(createL1PersistenceRecordId(batchId, {
      ...first,
      metadata: { ...first.metadata, value: "Cafe\u0301" },
    })).not.toBe(baseline);
    expect(createL1PersistenceRecordId(batchId, {
      ...first,
      scene_name: "model-authored-route-session-label",
      metadata: { ...first.metadata, domain: "uncontrolled-domain", slot: "uncontrolled-slot" },
    })).not.toBe(createL1PersistenceRecordId(batchId, {
      ...first,
      scene_name: "another-uncontrolled-label",
      metadata: { ...first.metadata, domain: "uncontrolled-domain", slot: "uncontrolled-slot" },
    }));
    expect(createL1PersistenceRecordId(batchId, {
      ...first,
      scene_name: "uncontrolled-scene ",
      metadata: { ...first.metadata, domain: "uncontrolled-domain", slot: "uncontrolled-slot" },
    })).not.toBe(createL1PersistenceRecordId(batchId, {
      ...first,
      scene_name: "uncontrolled-scene",
      metadata: { ...first.metadata, domain: "uncontrolled-domain", slot: "uncontrolled-slot" },
    }));
  });

  it("rejects duplicate deterministic rows before creating a receipt", async () => {
    const extracted: ExtractedMemory & { record_id: string } = {
      content: "导航到虹桥机场",
      type: "episodic",
      priority: 80,
      scene_name: "navigation",
      source_message_ids: ["u-1"],
      metadata: { domain: "navigation", slot: "destination", value: "虹桥机场" },
      record_id: "m_l1tx_duplicate",
    };

    await expect(prepareL1PersistencePlan({
      batchId: "l1tx_duplicate_plan",
      scope,
      memoriesWithIds: [extracted, structuredClone(extracted)],
      decisions: [],
      baseDir: "/unused",
      vectorStore: memoryStore(new FaultVectorStore()),
      outcome: { extractedCount: 2, sceneNames: ["navigation"] },
    })).rejects.toThrow("duplicate deterministic new IDs");
  });
});

describe("L1 persistence transaction recovery", () => {
  it("returns no JSONL evidence when transaction visibility receipts cannot be listed", async () => {
    const storage = new FaultStorage();
    const record = memoryRecord("must-not-leak");
    storage.files.set(
      "records/2026-09-01.jsonl",
      `${JSON.stringify(record)}\n`,
    );
    storage.failReceiptList = true;
    const error = vi.fn();
    const logger = { error } as never;

    await expect(readAllMemoryRecords("/unused", logger, storageAdapter(storage)))
      .resolves.toEqual([]);
    await expect(readMemoryRecords(scope.sessionKey, "/unused", logger, storageAdapter(storage)))
      .resolves.toEqual([]);
    expect(error).toHaveBeenCalledTimes(2);
  });

  it("loads receipts in the storage-free local JSONL fallback and hides a pending batch", async () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-local-jsonl-receipt-"));
    try {
      const storage = new ConcreteStorageAdapter(new LocalStorageBackend(root));
      const vector = new FaultVectorStore();
      vector.failUpsertAt = 1;
      vector.failUpsertMode = "false";
      const batchId = "l1tx_local_fallback_pending";
      const pending = memoryRecord("local-pending");

      await expect(commitL1PersistencePlan({
        plan: persistencePlan(batchId, [pending]),
        storage,
        vectorStore: memoryStore(vector),
      })).rejects.toThrow(/vector/u);
      await expect(readAllMemoryRecords(root)).resolves.toEqual([]);
      await expect(readMemoryRecords(scope.sessionKey, root)).resolves.toEqual([]);

      vector.failUpsertAt = null;
      await resumeL1PersistenceTransaction({
        batchId,
        storage,
        vectorStore: memoryStore(vector),
      });
      await expect(readAllMemoryRecords(root)).resolves.toMatchObject([{ id: pending.id }]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it.each(["false", "throw"] as const)(
    "keeps database rows pending when the logical visibility commit returns %s",
    async (failureMode) => {
      const storage = new FaultStorage();
      const vector = new FaultVectorStore();
      const batchId = `l1tx_visibility_${failureMode}`;
      const pending = memoryRecord(`pending-${failureMode}`);
      vector.failVisibilityCommit = failureMode;

      await expect(commitL1PersistencePlan({
        plan: persistencePlan(batchId, [pending]),
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      })).rejects.toThrow(/visibility/u);
      expect(receiptStage(storage, batchId)).toBe("targets_deleted");
      expect(vector.rows.get(pending.id)?.metadata).toHaveProperty(
        "__tdai_l1_transaction",
        batchId,
      );
      await expect(readAllMemoryRecords("/unused", undefined, storageAdapter(storage)))
        .resolves.toEqual([]);

      await resumeL1PersistenceTransaction({
        batchId,
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      });
      expect(receiptStage(storage, batchId)).toBe("committed");
      expect(vector.committedTransactions.has(batchId)).toBe(true);
      await expect(readAllMemoryRecords("/unused", undefined, storageAdapter(storage)))
        .resolves.toMatchObject([{ id: pending.id }]);
    },
  );

  it("converges concurrent workers on one immutable plan and monotonic marker chain", async () => {
    const storage = new FaultStorage();
    const vector = new FaultVectorStore();
    const batchId = "l1tx_concurrent_workers";
    const pending = memoryRecord("concurrent-record");
    const plan = persistencePlan(batchId, [pending]);

    const results = await Promise.all([
      commitL1PersistencePlan({
        plan,
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      }),
      commitL1PersistencePlan({
        plan: structuredClone(plan),
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      }),
    ]);
    expect(results).toHaveLength(2);
    expect(receiptStage(storage, batchId)).toBe("committed");
    const basePlan = JSON.parse(storage.files.get(StoragePaths.l1Transaction(batchId)) ?? "null") as {
      stage?: ReceiptStage;
    };
    expect(basePlan.stage).toBe("prepared");
    expect([...storage.files.keys()].filter((key) => key.includes(`${batchId}.`) && key.endsWith(".stage.json")))
      .toHaveLength(5);
    const visible = await readAllMemoryRecords("/unused", undefined, storageAdapter(storage));
    expect(visible.map((row) => row.id)).toEqual([pending.id]);
  });

  it("hides and retains a partial pending vector batch after lock loss, then replays safely", async () => {
    const storage = new FaultStorage();
    const vector = new FaultVectorStore();
    const controller = new AbortController();
    const abortReason = new Error("synthetic distributed lock lost");
    const batchId = "l1tx_abort_during_vectors";
    const first = memoryRecord("abort-first");
    const second = memoryRecord("abort-second");
    vector.afterUpsert = () => {
      vector.afterUpsert = undefined;
      controller.abort(abortReason);
    };

    await expect(commitL1PersistencePlan({
      plan: persistencePlan(batchId, [first, second]),
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
      abortSignal: controller.signal,
    })).rejects.toBe(abortReason);

    expect(receiptStage(storage, batchId)).toBe("jsonl_written");
    expect(vector.deleteCalls).toHaveLength(0);
    expect(vector.rows.has(first.id)).toBe(true);
    expect(vector.rows.has(second.id)).toBe(false);
    expect(vector.rows.get(first.id)?.metadata).toHaveProperty(
      "__tdai_l1_transaction",
      batchId,
    );
    expect(vector.committedTransactions.has(batchId)).toBe(false);
    await expect(readAllMemoryRecords("/unused", undefined, storageAdapter(storage)))
      .resolves.toEqual([]);

    await expect(resumeL1PersistenceTransaction({
      batchId,
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
    })).resolves.toMatchObject({ records: [{ id: first.id }, { id: second.id }] });
    expect(receiptStage(storage, batchId)).toBe("committed");
  });

  it("does not let an aborted concurrent replay delete a successful worker's committed row", async () => {
    const storage = new FaultStorage();
    const vector = new FaultVectorStore();
    const controller = new AbortController();
    const abortReason = new Error("synthetic losing worker lock loss");
    const batchId = "l1tx_concurrent_abort_after_peer_commit";
    const pending = memoryRecord("concurrent-abort-record");
    const plan = persistencePlan(batchId, [pending]);
    let winningReplay: Promise<unknown> | undefined;

    vector.afterUpsert = async () => {
      vector.afterUpsert = undefined;
      winningReplay = commitL1PersistencePlan({
        plan: structuredClone(plan),
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      });
      await winningReplay;
      controller.abort(abortReason);
    };

    await expect(commitL1PersistencePlan({
      plan,
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
      abortSignal: controller.signal,
    })).rejects.toBe(abortReason);
    await expect(winningReplay).resolves.toBeDefined();

    expect(receiptStage(storage, batchId)).toBe("committed");
    expect(vector.deleteCalls).toHaveLength(0);
    expect(vector.rows.has(pending.id)).toBe(true);
    expect(vector.committedTransactions.has(batchId)).toBe(true);
    await expect(readAllMemoryRecords("/unused", undefined, storageAdapter(storage)))
      .resolves.toMatchObject([{ id: pending.id }]);
  });

  it("resumes before a second model call and never enters compatibility store-all after a sink failure", async () => {
    const storage = new FaultStorage();
    const vector = new FaultVectorStore();
    storage.failAppendMode = "before";
    const run = vi.fn(async () => JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u-strict"],
      memories: [{
        content: "用户要导航到虹桥机场。",
        type: "episodic",
        priority: 80,
        source_message_ids: ["u-strict"],
        metadata: {
          activity_start_time: "2026-09-01T09:00:00+08:00",
          action_status: "requested",
        },
      }],
    }]));
    const params = {
      messages: [{
        id: "u-strict",
        role: "user" as const,
        content: "九点导航到虹桥机场",
        timestamp: Date.UTC(2026, 8, 1, 1, 0),
      }],
      sessionKey: scope.sessionKey,
      sessionId: scope.sessionId,
      taskId: scope.taskId,
      teamId: scope.teamId,
      userId: scope.userId,
      agentId: scope.agentId,
      baseDir: "/unused",
      config: {},
      options: {
        enableDedup: false,
        llmRunner: { run },
        llmTimeoutMs: 600_000,
        vectorStore: memoryStore(vector),
        strictPersistence: true,
      },
      storage: storageAdapter(storage),
    };

    const failed = await extractL1Memories(params);
    expect(failed).toMatchObject({ success: false, storedCount: 0, records: [] });
    expect(run).toHaveBeenCalledTimes(1);
    expect(storage.appendCalls).toHaveLength(1);
    expect(vector.upsertCalls).toHaveLength(0);

    const resumed = await extractL1Memories(params);
    expect(resumed).toMatchObject({ success: true, extractedCount: 1, storedCount: 1 });
    expect(run).toHaveBeenCalledTimes(1);
    expect(storage.appendCalls).toHaveLength(2);
    expect(vector.upsertCalls).toHaveLength(1);
  });

  it.each(["before", "after"] as const)(
    "fails closed on JSONL append %s mutation and retry converges without visible duplicates",
    async (mode) => {
      const batchId = `l1tx_append_${mode}`;
      const storage = new FaultStorage();
      const vector = new FaultVectorStore();
      const record = memoryRecord(`new-${mode}`);
      storage.files.set(
        "records/2026-08-31.jsonl",
        `${JSON.stringify(memoryRecord("unrelated-old"))}\n`,
      );
      storage.failAppendMode = mode;

      await expect(commitL1PersistencePlan({
        plan: persistencePlan(batchId, [record]),
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      })).rejects.toThrow(/append/u);

      expect(receiptStage(storage, batchId)).toBe("prepared");
      expect(vector.upsertCalls).toHaveLength(0);
      expect(vector.deleteCalls).toHaveLength(0);
      const pendingRead = await readAllMemoryRecords("/unused", undefined, storageAdapter(storage));
      expect(pendingRead.map((entry) => entry.id)).toEqual(["unrelated-old"]);

      const recovered = await resumeL1PersistenceTransaction({
        batchId,
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      });
      expect(recovered?.records.map((entry) => entry.id)).toEqual([record.id]);
      expect(receiptStage(storage, batchId)).toBe("committed");
      expect(vector.rows.has(record.id)).toBe(true);
      const committedRead = await readAllMemoryRecords("/unused", undefined, storageAdapter(storage));
      expect(committedRead.map((entry) => entry.id).sort()).toEqual([record.id, "unrelated-old"].sort());
      expect(committedRead.filter((entry) => entry.id === record.id)).toHaveLength(1);
    },
  );

  it.each(["false", "throw"] as const)(
    "retains hidden deterministic rows on a partial upsert %s and retry converges before target deletion",
    async (failureMode) => {
      const batchId = `l1tx_upsert_${failureMode}`;
      const storage = new FaultStorage();
      const vector = new FaultVectorStore();
      const first = memoryRecord(`new-1-${failureMode}`);
      const second = memoryRecord(`new-2-${failureMode}`);
      const old = memoryRecord(`old-${failureMode}`);
      vector.rows.set(old.id, old);
      vector.failUpsertAt = 2;
      vector.failUpsertMode = failureMode;

      await expect(commitL1PersistencePlan({
        plan: persistencePlan(batchId, [first, second], [old.id]),
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      })).rejects.toThrow(/vector/u);

      expect(receiptStage(storage, batchId)).toBe("jsonl_written");
      expect(vector.deleteCalls).toHaveLength(0);
      expect(vector.rows.has(first.id)).toBe(true);
      expect(vector.rows.has(second.id)).toBe(true);
      expect(vector.rows.has(old.id)).toBe(true);
      await expect(readAllMemoryRecords("/unused", undefined, storageAdapter(storage)))
        .resolves.toEqual([]);

      vector.failUpsertAt = null;
      await resumeL1PersistenceTransaction({
        batchId,
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      });
      expect(receiptStage(storage, batchId)).toBe("committed");
      expect(vector.rows.has(first.id)).toBe(true);
      expect(vector.rows.has(second.id)).toBe(true);
      expect(vector.rows.has(old.id)).toBe(false);
      expect(storage.appendCalls).toHaveLength(1);
      expect(vector.deleteCalls.at(-1)?.ids).toEqual([old.id]);
    },
  );

  it.each(["false", "throw"] as const)(
    "keeps the receipt replayable when target deletion returns %s",
    async (failureMode) => {
      const batchId = `l1tx_delete_${failureMode}`;
      const storage = new FaultStorage();
      const vector = new FaultVectorStore();
      const old = memoryRecord(`old-delete-${failureMode}`);
      const next = memoryRecord(`new-delete-${failureMode}`);
      vector.rows.set(old.id, old);
      storage.files.set(
        "records/2026-08-31.jsonl",
        `${JSON.stringify(old)}\n`,
      );
      vector.deleteFailures = [failureMode];

      await expect(commitL1PersistencePlan({
        plan: persistencePlan(batchId, [next], [old.id]),
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      })).rejects.toThrow(/delet/u);

      expect(receiptStage(storage, batchId)).toBe("vectors_written");
      expect(storage.appendCalls).toHaveLength(1);
      expect(vector.upsertCalls).toHaveLength(1);
      const pendingRead = await readAllMemoryRecords("/unused", undefined, storageAdapter(storage));
      expect(pendingRead.map((entry) => entry.id)).toEqual([old.id]);

      await resumeL1PersistenceTransaction({
        batchId,
        storage: storageAdapter(storage),
        vectorStore: memoryStore(vector),
      });
      expect(receiptStage(storage, batchId)).toBe("committed");
      expect(storage.appendCalls).toHaveLength(1);
      expect(vector.upsertCalls).toHaveLength(1);
      expect(vector.rows.has(old.id)).toBe(false);
      expect(vector.rows.has(next.id)).toBe(true);
      const committedRead = await readAllMemoryRecords("/unused", undefined, storageAdapter(storage));
      expect(committedRead.map((entry) => entry.id)).toEqual([next.id]);
    },
  );

  it("repeats idempotent target deletion after a crash before its receipt stage is durable", async () => {
    const batchId = "l1tx_crash_after_delete";
    const storage = new FaultStorage();
    const vector = new FaultVectorStore();
    const old = memoryRecord("old-crash-target");
    const next = memoryRecord("new-crash-target");
    vector.rows.set(old.id, old);
    storage.failReceiptStageOnce = "targets_deleted";

    await expect(commitL1PersistencePlan({
      plan: persistencePlan(batchId, [next], [old.id]),
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
    })).rejects.toThrow("synthetic receipt targets_deleted failure");

    expect(receiptStage(storage, batchId)).toBe("vectors_written");
    expect(vector.rows.has(old.id)).toBe(false);
    expect(vector.deleteCalls.map((call) => call.ids)).toEqual([[old.id]]);

    await resumeL1PersistenceTransaction({
      batchId,
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
    });
    expect(receiptStage(storage, batchId)).toBe("committed");
    expect(vector.deleteCalls.map((call) => call.ids)).toEqual([[old.id], [old.id]]);
    expect(storage.appendCalls).toHaveLength(1);
    expect(vector.upsertCalls).toHaveLength(1);

    const callsBeforeCommittedReplay = {
      append: storage.appendCalls.length,
      upsert: vector.upsertCalls.length,
      delete: vector.deleteCalls.length,
    };
    await resumeL1PersistenceTransaction({
      batchId,
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
    });
    expect({
      append: storage.appendCalls.length,
      upsert: vector.upsertCalls.length,
      delete: vector.deleteCalls.length,
    }).toEqual(callsBeforeCommittedReplay);
  });

  it("keeps the L0 checkpoint behind a failed transaction and advances it only after receipt replay", async () => {
    const storage = new FaultStorage();
    const vector = Object.assign(new FaultVectorStore(), {
      isDegraded: () => false,
      queryL0GroupedBySessionId: vi.fn(async () => [{
        sessionId: scope.sessionId,
        teamId: scope.teamId,
        taskId: scope.taskId,
        userId: scope.userId,
        agentId: scope.agentId,
        messages: [{
          id: "u-checkpoint",
          role: "user",
          content: "七点出发",
          timestamp: Date.UTC(2026, 8, 1, 1, 0),
          recordedAtMs: 12_345,
        }],
      }]),
    });
    const proposal = {
      content: "用户计划七点出发。",
      type: "episodic",
      priority: 70,
      source_message_ids: ["u-checkpoint"],
      metadata: {
        domain: "navigation",
        slot: "departure_time",
        value: "07:00",
        subject: "user",
        state_key: "navigation|user|unspecified-vehicle|unspecified-zone|departure_time",
        episode_key: "route-1",
        relation: "asserted",
        action_status: "requested",
      } as Record<string, unknown>,
    };
    const scene = (memory: typeof proposal) => JSON.stringify([{
      scene_name: "navigation",
      message_ids: ["u-checkpoint"],
      memories: [memory],
    }]);
    const reconciled = structuredClone(proposal);
    reconciled.metadata.input_candidate_ids = ["primary:0", "atomic:0"];
    const run = vi.fn(async (request: { taskId?: string }) =>
      request.taskId?.includes("reconcile") ? scene(reconciled) : scene(proposal)
    );
    const logger = {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    };
    const runner = createL1Runner({
      pluginDataDir: "/unused",
      cfg: parseConfig({
        promptMode: "cockpit",
        extraction: {
          promptMode: "cockpit",
          enableDedup: false,
          maxMessagesPerExtraction: 10,
          maxMemoriesPerSession: 20,
          model: "deepseek-v4-flash",
        },
      }),
      openclawConfig: {},
      vectorStore: memoryStore(vector),
      embeddingService: undefined,
      logger,
      llmRunner: { run },
      storage: storageAdapter(storage),
    });
    storage.failAppendMode = "before";

    await expect(runner({ sessionKey: scope.sessionKey }))
      .rejects.toThrow("checkpoint cursor was not advanced");
    expect(storage.files.has(StoragePaths.checkpoint)).toBe(false);
    expect(run).toHaveBeenCalledTimes(3);

    await expect(runner({ sessionKey: scope.sessionKey }))
      .resolves.toMatchObject({ processedCount: 1, storedCount: 1 });
    expect(run).toHaveBeenCalledTimes(3);
    const checkpoint = JSON.parse(storage.files.get(StoragePaths.checkpoint) ?? "null") as {
      runner_states?: Record<string, { last_l1_cursor?: number }>;
      total_memories_extracted?: number;
    };
    expect(checkpoint.runner_states?.[scope.sessionKey].last_l1_cursor).toBe(12_345);
    expect(checkpoint.total_memories_extracted).toBe(1);
  });

  it("propagates lock loss to the construction provider and creates no receipt or checkpoint", async () => {
    const storage = new FaultStorage();
    const vector = Object.assign(new FaultVectorStore(), {
      isDegraded: () => false,
      queryL0GroupedBySessionId: vi.fn(async () => [{
        sessionId: scope.sessionId,
        teamId: scope.teamId,
        taskId: scope.taskId,
        userId: scope.userId,
        agentId: scope.agentId,
        messages: [{
          id: "u-abort-provider",
          role: "user",
          content: "八点导航到机场",
          timestamp: Date.UTC(2026, 8, 1, 0, 0),
          recordedAtMs: 22_222,
        }],
      }]),
    });
    const controller = new AbortController();
    const abortReason = new Error("synthetic lock lost during provider call");
    const run = vi.fn(async (request: { abortSignal?: AbortSignal }) => {
      expect(request.abortSignal).toBe(controller.signal);
      controller.abort(abortReason);
      return "[]";
    });
    const runner = createL1Runner({
      pluginDataDir: "/unused",
      cfg: parseConfig({
        promptMode: "cockpit",
        extraction: {
          promptMode: "cockpit",
          enableDedup: false,
          maxMessagesPerExtraction: 10,
          maxMemoriesPerSession: 20,
          model: "deepseek-v4-flash",
        },
      }),
      openclawConfig: {},
      vectorStore: memoryStore(vector),
      embeddingService: undefined,
      logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
      llmRunner: { run },
      storage: storageAdapter(storage),
    });

    await expect(runner({
      sessionKey: scope.sessionKey,
      abortSignal: controller.signal,
    })).rejects.toBe(abortReason);

    expect(run).toHaveBeenCalledOnce();
    expect(vector.upsertCalls).toHaveLength(0);
    expect(storage.files.has(StoragePaths.checkpoint)).toBe(false);
    expect([...storage.files.keys()].some((key) => key.startsWith(StoragePaths.l1TransactionsDir)))
      .toBe(false);
  });
});

describe("L1 writer strict/compatibility boundary", () => {
  const extracted: ExtractedMemory = {
    content: "导航到虹桥机场",
    type: "episodic",
    priority: 80,
    scene_name: "navigation",
    source_message_ids: ["u-1"],
    metadata: { domain: "navigation", slot: "destination", value: "虹桥机场" },
  };

  it("rejects strict direct mutations before touching either sink", async () => {
    const storage = new FaultStorage();
    const vector = new FaultVectorStore();

    await expect(writeMemory({
      memory: extracted,
      decision: { record_id: "strict-direct", action: "store", target_ids: [] },
      baseDir: "/unused",
      sessionKey: scope.sessionKey,
      sessionId: scope.sessionId,
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
      strictPersistence: true,
    })).rejects.toThrow("requires prepareOnly plus commitL1PersistencePlan");
    expect(storage.appendCalls).toHaveLength(0);
    expect(vector.upsertCalls).toHaveLength(0);
    expect(vector.deleteCalls).toHaveLength(0);
  });

  it("retains legacy non-strict best-effort writes", async () => {
    const storage = new FaultStorage();
    const vector = new FaultVectorStore();
    storage.failAppendMode = "before";
    vector.failUpsertAt = 1;
    vector.failUpsertMode = "false";
    vector.commitFailedUpsert = false;
    const warn = vi.fn();

    await expect(writeMemory({
      memory: extracted,
      decision: { record_id: "compat-direct", action: "store", target_ids: [] },
      baseDir: "/unused",
      sessionKey: scope.sessionKey,
      sessionId: scope.sessionId,
      storage: storageAdapter(storage),
      vectorStore: memoryStore(vector),
      logger: { warn } as never,
    })).resolves.toMatchObject({ id: "compat-direct" });
    expect(storage.appendCalls).toHaveLength(1);
    expect(vector.upsertCalls).toEqual(["compat-direct"]);
    expect(warn).toHaveBeenCalled();
  });
});
