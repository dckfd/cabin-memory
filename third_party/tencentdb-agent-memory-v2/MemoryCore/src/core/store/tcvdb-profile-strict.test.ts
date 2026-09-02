import { describe, expect, it, vi } from "vitest";

import type { ProfileSyncRecord } from "./types.js";
import { TcvdbMemoryStore } from "./tcvdb.js";

function createStore(): TcvdbMemoryStore {
  return new TcvdbMemoryStore({
    url: "http://127.0.0.1:1",
    username: "test",
    apiKey: "test-placeholder",
    database: "testdb",
    embeddingModel: "test-model",
    timeout: 100,
    embeddingEnabled: false,
  });
}

function profile(index: number, overrides: Partial<ProfileSyncRecord> = {}): ProfileSyncRecord {
  return {
    id: `profile-${index}`,
    type: "l2",
    filename: `scene-${index}.md`,
    content: `content-${index}`,
    contentMd5: `md5-${index}`,
    teamId: "team-a",
    agentId: "agent-a",
    version: 0,
    baselineVersion: 0,
    createdAtMs: 1,
    updatedAtMs: 1,
    ...overrides,
  };
}

describe("TCVDB strict profile publication", () => {
  it("chunks baseline reads, writes, and readback verification at 20 IDs", async () => {
    const store = createStore();
    const documents = new Map<string, Record<string, unknown>>();
    const query = vi.fn(async (_collection: string, params: { documentIds?: string[] }) => ({
      documents: (params.documentIds ?? [])
        .map((id) => documents.get(id))
        .filter((document): document is Record<string, unknown> => document !== undefined),
    }));
    const upsert = vi.fn(async (_collection: string, incoming: Array<Record<string, unknown>>) => {
      for (const document of incoming) {
        documents.set(String(document.id), structuredClone(document));
      }
    });
    const mutationGuard = vi.fn(async () => {});
    Object.assign(store as unknown as Record<string, unknown>, { client: { query, upsert } });

    await expect(store.syncProfiles(
      Array.from({ length: 45 }, (_, index) => profile(index)),
      { strict: true, mutationGuard },
    )).resolves.toBeUndefined();

    expect(upsert).toHaveBeenCalledTimes(3);
    expect(mutationGuard).toHaveBeenCalledTimes(3);
    expect(upsert.mock.calls.map((call) => call[1].length)).toEqual([20, 20, 5]);
    expect(query).toHaveBeenCalledTimes(6);
    expect(query.mock.calls.map((call) => call[1].documentIds?.length)).toEqual([20, 20, 5, 20, 20, 5]);
  });

  it("fails closed on an optimistic-version conflict", async () => {
    const store = createStore();
    const query = vi.fn(async () => ({
      documents: [{
        id: "profile-1",
        filename: "scene-1.md",
        content_md5: "remote-md5",
        version: 2,
        created_at_ms: 1,
      }],
    }));
    const upsert = vi.fn(async () => {});
    Object.assign(store as unknown as Record<string, unknown>, { client: { query, upsert } });

    await expect(store.syncProfiles(
      [profile(1, { baselineVersion: 1 })],
      { strict: true },
    )).rejects.toThrow(/Conflict/);
    expect(upsert).not.toHaveBeenCalled();
  });

  it("recovers a lost upsert acknowledgement only when readback proves durability", async () => {
    const store = createStore();
    const documents = new Map<string, Record<string, unknown>>();
    const query = vi.fn(async (_collection: string, params: { documentIds?: string[] }) => ({
      documents: (params.documentIds ?? [])
        .map((id) => documents.get(id))
        .filter((document): document is Record<string, unknown> => document !== undefined),
    }));
    const upsert = vi.fn(async (_collection: string, incoming: Array<Record<string, unknown>>) => {
      for (const document of incoming) documents.set(String(document.id), structuredClone(document));
      throw new Error("connection reset after commit");
    });
    Object.assign(store as unknown as Record<string, unknown>, { client: { query, upsert } });

    await expect(store.syncProfiles([profile(1)], { strict: true })).resolves.toBeUndefined();
    expect(query).toHaveBeenCalledTimes(3);
  });

  it("rejects an acknowledged upsert whose readback is missing", async () => {
    const store = createStore();
    const query = vi.fn(async () => ({ documents: [] }));
    const upsert = vi.fn(async () => {});
    Object.assign(store as unknown as Record<string, unknown>, { client: { query, upsert } });

    await expect(store.syncProfiles([profile(1)], { strict: true }))
      .rejects.toThrow(/verification failed/);
  });

  it("accepts a lost delete acknowledgement only after absence is proven", async () => {
    const store = createStore();
    const documents = new Map<string, Record<string, unknown>>([
      ["profile-1", { id: "profile-1" }],
    ]);
    const deleteDoc = vi.fn(async (_collection: string, params: { query: { documentIds: string[] } }) => {
      for (const id of params.query.documentIds) documents.delete(id);
      throw new Error("connection reset after commit");
    });
    const query = vi.fn(async (_collection: string, params: { documentIds?: string[] }) => ({
      documents: (params.documentIds ?? [])
        .map((id) => documents.get(id))
        .filter((document): document is Record<string, unknown> => document !== undefined),
    }));
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });

    await expect(store.deleteProfiles(["profile-1"], { strict: true })).resolves.toBeUndefined();
  });

  it("rejects a partial delete and refuses strict writes while degraded", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async () => 1);
    const query = vi.fn(async () => ({ documents: [{ id: "profile-2" }] }));
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });

    await expect(store.deleteProfiles(["profile-1", "profile-2"], { strict: true }))
      .rejects.toThrow(/verification failed/);

    Object.assign(store as unknown as Record<string, unknown>, { degraded: true });
    await expect(store.syncProfiles([profile(1)], { strict: true })).rejects.toThrow(/degraded/);
    await expect(store.deleteProfiles(["profile-1"], { strict: true })).rejects.toThrow(/degraded/);
  });

  it("honours cancellation before any remote mutation", async () => {
    const store = createStore();
    const query = vi.fn(async () => ({ documents: [] }));
    const upsert = vi.fn(async () => {});
    Object.assign(store as unknown as Record<string, unknown>, { client: { query, upsert } });
    const controller = new AbortController();
    controller.abort(new Error("lease lost"));

    await expect(store.syncProfiles([profile(1)], {
      strict: true,
      abortSignal: controller.signal,
    })).rejects.toThrow("lease lost");
    expect(query).not.toHaveBeenCalled();
    expect(upsert).not.toHaveBeenCalled();
  });

  it("revalidates the exact lease before a VDB mutation and fails before upsert on loss", async () => {
    const store = createStore();
    const query = vi.fn(async () => ({ documents: [] }));
    const upsert = vi.fn(async () => {});
    const mutationGuard = vi.fn(async () => { throw new Error("replacement owns lease"); });
    Object.assign(store as unknown as Record<string, unknown>, { client: { query, upsert } });

    await expect(store.syncProfiles([profile(1)], {
      strict: true,
      mutationGuard,
    })).rejects.toThrow("replacement owns lease");
    expect(query).toHaveBeenCalledOnce();
    expect(mutationGuard).toHaveBeenCalledOnce();
    expect(upsert).not.toHaveBeenCalled();
  });

  it("guards every chunked delete mutation", async () => {
    const store = createStore();
    const deleteDoc = vi.fn(async () => 0);
    const query = vi.fn(async () => ({ documents: [] }));
    const mutationGuard = vi.fn(async () => {});
    Object.assign(store as unknown as Record<string, unknown>, { client: { deleteDoc, query } });
    const ids = Array.from({ length: 45 }, (_, index) => `profile-${index}`);

    await expect(store.deleteProfiles(ids, { strict: true, mutationGuard })).resolves.toBeUndefined();
    expect(deleteDoc).toHaveBeenCalledTimes(3);
    expect(query).toHaveBeenCalledTimes(3);
    expect(mutationGuard).toHaveBeenCalledTimes(3);
  });
});
