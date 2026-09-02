import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { writeMemory, type MemoryRecord } from "../record/l1-writer.js";
import {
  L1_TRANSACTION_METADATA_KEY,
  markL1RecordPending,
} from "../record/l1-persistence-visibility.js";
import { VectorStore } from "./sqlite.js";

const temporaryDirectories: string[] = [];

function record(
  id: string,
  version: number,
  sessionId = "old-session",
  sessionKey = "old-key",
  metadata: Record<string, unknown> = {},
): MemoryRecord {
  const now = "2026-09-01T00:00:00.000Z";
  return {
    id,
    content: id,
    type: "episodic",
    priority: 80,
    scene_name: "navigation",
    source_message_ids: [`source-${id}`],
    metadata,
    timestamps: [now],
    createdAt: now,
    updatedAt: now,
    version,
    sessionKey,
    sessionId,
    taskId: "trip-task",
    teamId: "fleet-team",
    userId: "driver-user",
    agentId: "cockpit-agent",
  };
}

afterEach(() => {
  while (temporaryDirectories.length > 0) {
    rmSync(temporaryDirectories.pop()!, { recursive: true, force: true });
  }
});

describe("SQLite L1 conjunctive queries", () => {
  it("honours recordIds and every scope predicate without polluting predecessor state", async () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-sqlite-l1-query-"));
    temporaryDirectories.push(root);
    const store = new VectorStore(join(root, "memory.db"), 0);
    store.init();

    const target = record("target", 3, "old-session", "old-key", { target_only: true });
    const unrelated = record("unrelated", 99, "old-session", "old-key", { pollution: true });
    const wrongKey = record("wrong-key", 50, "old-session", "different-key", { wrong_key: true });
    expect(store.upsertL1(target, undefined)).toBe(true);
    expect(store.upsertL1(unrelated, undefined)).toBe(true);
    expect(store.upsertL1(wrongKey, undefined)).toBe(true);

    expect(store.queryL1Records({
      recordIds: [target.id],
      teamId: target.teamId,
      userId: target.userId,
      agentId: target.agentId,
      taskId: target.taskId,
    }).map((row) => row.record_id)).toEqual([target.id]);
    expect(store.queryL1Records({
      sessionId: "old-session",
      sessionKey: "old-key",
    }).map((row) => row.record_id).sort()).toEqual([target.id, unrelated.id].sort());

    const prepared = await writeMemory({
      memory: {
        content: "updated destination",
        type: "episodic",
        priority: 90,
        scene_name: "navigation",
        source_message_ids: ["source-new"],
        metadata: { current_only: true },
      },
      decision: {
        record_id: "new-record",
        action: "update",
        target_ids: [target.id],
        merged_content: "updated destination",
      },
      baseDir: root,
      sessionKey: "new-key",
      sessionId: "new-session",
      taskId: target.taskId,
      teamId: target.teamId,
      userId: target.userId,
      agentId: target.agentId,
      vectorStore: store,
      strictPersistence: true,
      prepareOnly: true,
      nowIso: "2026-09-01T01:00:00.000Z",
    });

    expect(prepared?.version).toBe(4);
    expect(prepared?.metadata).toMatchObject({ target_only: true, current_only: true });
    expect(prepared?.metadata).not.toHaveProperty("pollution");
    expect(prepared?.metadata).not.toHaveProperty("wrong_key");
    store.close();
  });

  it("hides every pending read path until one logical visibility commit", () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-sqlite-l1-visibility-"));
    temporaryDirectories.push(root);
    const store = new VectorStore(join(root, "memory.db"), 0);
    store.init();

    const batchId = "l1tx_sqlite_visibility";
    const pending = {
      ...record("pendingdestination", 1),
      content: "pendingdestination",
    };
    expect(store.upsertL1(markL1RecordPending(pending, batchId), undefined)).toBe(true);

    expect(store.queryL1Records({ recordIds: [pending.id] })).toEqual([]);
    expect(store.countL1()).toBe(0);
    expect(store.getAllL1Texts()).toEqual([]);
    expect(store.queryL1Paginated({ limit: 10, offset: 0 })).toEqual({ rows: [], total: 0 });
    expect(store.searchL1Fts("pendingdestination", 10)).toEqual([]);

    expect(store.commitL1Transaction(batchId)).toBe(true);
    expect(store.isL1TransactionCommitted(batchId)).toBe(true);
    const queried = store.queryL1Records({ recordIds: [pending.id] });
    expect(queried.map((row) => row.record_id)).toEqual([pending.id]);
    expect(JSON.parse(queried[0].metadata_json)).not.toHaveProperty(L1_TRANSACTION_METADATA_KEY);
    expect(store.countL1()).toBe(1);
    expect(store.getAllL1Texts().map((row) => row.record_id)).toEqual([pending.id]);
    expect(store.queryL1Paginated({ limit: 10, offset: 0 }).total).toBe(1);
    expect(store.searchL1Fts("pendingdestination", 10).map((row) => row.record_id)).toEqual([pending.id]);
    store.close();
  });
});
