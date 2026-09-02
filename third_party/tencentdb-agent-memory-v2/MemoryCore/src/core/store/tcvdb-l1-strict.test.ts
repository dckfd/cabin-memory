import { describe, expect, it, vi } from "vitest";

import type { MemoryRecord } from "../record/l1-writer.js";
import {
  L1_COMMIT_MEMORY_TYPE,
  L1_PENDING_MEMORY_TYPE,
  L1_TRANSACTION_METADATA_KEY,
  createL1TransactionCommitRecordId,
  markL1RecordPending,
} from "../record/l1-persistence-visibility.js";
import { TcvdbMemoryStore } from "./tcvdb.js";

function createStore(embeddingEnabled = false): TcvdbMemoryStore {
  return new TcvdbMemoryStore({
    url: "http://127.0.0.1:1",
    username: "test",
    apiKey: "test-placeholder",
    database: "testdb",
    embeddingModel: "test-model",
    timeout: 100,
    embeddingEnabled,
  });
}

function record(): MemoryRecord {
  return {
    id: "record-1",
    content: "destination",
    type: "episodic",
    priority: 80,
    scene_name: "navigation",
    source_message_ids: ["source-1"],
    metadata: {},
    timestamps: ["2026-09-01T00:00:00.000Z"],
    createdAt: "2026-09-01T00:00:00.000Z",
    updatedAt: "2026-09-01T00:00:00.000Z",
    sessionKey: "session-key",
    sessionId: "session-id",
    taskId: "task-id",
    teamId: "team-id",
    userId: "user-id",
    agentId: "agent-id",
  };
}

describe("TCVDB strict L1 sink semantics", () => {
  it("does not report a degraded upsert as successful", async () => {
    const store = createStore();
    Object.assign(store as unknown as Record<string, unknown>, { degraded: true });
    await expect(store.upsertL1(record())).resolves.toBe(false);
  });

  it("pushes the complete isolation filter into batch deletion", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async () => 2);
    const query = vi.fn(async () => ({ documents: [] }));
    Object.assign(store as unknown as Record<string, unknown>, {
      client: { deleteDoc, query },
    });

    await expect(store.deleteL1Batch(["old-1", "old-2"], {
      teamId: "team-id",
      userId: "user-id",
      agentId: "agent-id",
      sessionId: "session-id",
      sessionKey: "session-key",
      taskId: "task-id",
    })).resolves.toBe(true);

    expect(deleteDoc).toHaveBeenCalledOnce();
    const request = deleteDoc.mock.calls[0]?.[1] as { query: { filter?: string } };
    expect(request.query.filter).toContain('team_id = "team-id"');
    expect(request.query.filter).toContain('user_id = "user-id"');
    expect(request.query.filter).toContain('agent_id = "agent-id"');
    expect(request.query.filter).toContain('session_id = "session-id"');
    expect(request.query.filter).toContain('session_key = "session-key"');
    expect(request.query.filter).toContain('task_id = "task-id"');
    expect(query).toHaveBeenCalledWith("testdb_l1_memories", {
      retrieveVector: false,
      documentIds: ["old-1", "old-2"],
      outputFields: ["id"],
    });
  });

  it("fails closed when zero affected rows still exist after deletion", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async () => 0);
    const query = vi.fn(async () => ({ documents: [{ id: "old-1" }, { id: "old-2" }] }));
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });

    await expect(store.deleteL1Batch(["old-1", "old-2"])).resolves.toBe(false);
  });

  it("fails closed when a partial deletion leaves any requested row", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async () => 1);
    const query = vi.fn(async () => ({ documents: [{ id: "old-2" }] }));
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });

    await expect(store.deleteL1Batch(["old-1", "old-2"])).resolves.toBe(false);
  });

  it("accepts a lost acknowledgement only after strong readback proves absence", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async () => { throw new Error("connection reset after commit"); });
    const query = vi.fn(async () => ({ documents: [] }));
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });

    await expect(store.deleteL1Batch(["old-1", "old-2"])).resolves.toBe(true);
  });

  it("fails closed when delete and verification both fail", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async () => { throw new Error("delete unavailable"); });
    const query = vi.fn(async () => { throw new Error("read unavailable"); });
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });

    await expect(store.deleteL1Batch(["old-1"])).resolves.toBe(false);
  });

  it("chunks deletion and verification at the documentIds limit", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async (_collection: string, params: { query: { documentIds: string[] } }) => (
      params.query.documentIds.length
    ));
    const query = vi.fn(async () => ({ documents: [] }));
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });
    const ids = Array.from({ length: 45 }, (_, index) => `old-${index}`);

    await expect(store.deleteL1Batch(ids)).resolves.toBe(true);
    expect(deleteDoc).toHaveBeenCalledTimes(3);
    expect(query).toHaveBeenCalledTimes(3);
    expect(deleteDoc.mock.calls.map((call) => (
      (call[1] as { query: { documentIds: string[] } }).query.documentIds.length
    ))).toEqual([20, 20, 5]);
  });

  it("uses a strong commit sentinel to hide pending rows across every TCVDB read path", async () => {
    const store = createStore(true);
    const batchId = "l1tx_tcvdb_visibility";
    const docs = new Map<string, Record<string, unknown>>();
    const upsert = vi.fn(async (_collection: string, incoming: Record<string, unknown>[]) => {
      incoming.forEach((doc) => docs.set(String(doc.id), structuredClone(doc)));
    });
    const query = vi.fn(async (_collection: string, params: Record<string, unknown>) => {
      const ids = params.documentIds as string[] | undefined;
      const selected = ids
        ? ids.map((id) => docs.get(id)).filter((doc): doc is Record<string, unknown> => doc !== undefined)
        : [...docs.values()];
      const offset = Number(params.offset ?? 0);
      const limit = Number(params.limit ?? selected.length);
      return { documents: selected.slice(offset, offset + limit) };
    });
    const search = vi.fn(async () => ({ documents: [[...docs.values()].map((doc) => ({ ...doc, score: 0.9 }))] }));
    Object.assign(store as unknown as Record<string, unknown>, {
      client: { upsert, query, search },
    });

    const pending = markL1RecordPending(record(), batchId);
    await expect(store.upsertL1(pending)).resolves.toBe(true);
    expect([...docs.values()][0].memory_type).toBe(L1_PENDING_MEMORY_TYPE);
    await expect(store.queryL1Records({ recordIds: [pending.id] })).resolves.toEqual([]);
    await expect(store.countL1()).resolves.toBe(0);
    await expect(store.getAllL1Texts()).resolves.toEqual([]);
    await expect(store.queryL1Paginated({ limit: 10, offset: 0 })).resolves.toEqual({ rows: [], total: 0 });
    await expect(store.searchL1Hybrid({ query: "destination", topK: 10 })).resolves.toEqual([]);

    await expect(store.commitL1Transaction(batchId)).resolves.toBe(true);
    const sentinel = docs.get(createL1TransactionCommitRecordId(batchId));
    expect(sentinel?.memory_type).toBe(L1_COMMIT_MEMORY_TYPE);
    const rows = await store.queryL1Records({ recordIds: [pending.id] });
    expect(rows.map((row) => row.record_id)).toEqual([pending.id]);
    expect(JSON.parse(rows[0].metadata_json)).not.toHaveProperty(L1_TRANSACTION_METADATA_KEY);
    await expect(store.countL1()).resolves.toBe(1);
    await expect(store.getAllL1Texts()).resolves.toHaveLength(1);
    await expect(store.queryL1Paginated({ limit: 10, offset: 0 })).resolves.toMatchObject({ total: 1 });
    await expect(store.searchL1Hybrid({ query: "destination", topK: 10 }))
      .resolves.toMatchObject([{ record_id: pending.id }]);
  });
});
