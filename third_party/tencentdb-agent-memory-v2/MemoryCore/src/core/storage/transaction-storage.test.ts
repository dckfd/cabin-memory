import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  createScopedStorageAdapter,
  createStagedStorageTransaction,
  StorageAdapter,
} from "./adapter.js";
import { LocalStorageBackend } from "./local-backend.js";
import type {
  IStorageBackend,
  ListObjectsOptions,
  ListResult,
  StorageObject,
} from "./types.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  while (temporaryDirectories.length > 0) {
    rmSync(temporaryDirectories.pop()!, { recursive: true, force: true });
  }
});

class PagedBackend implements IStorageBackend {
  readonly type = "local" as const;
  readonly keys: string[];
  readonly markers: Array<string | undefined> = [];

  constructor(prefix = "") {
    this.keys = Array.from({ length: 12_345 }, (_, index) =>
      `${prefix}.metadata/l1-transactions/l1tx_${String(index).padStart(5, "0")}.json`
    );
  }

  async putObject(): Promise<void> { throw new Error("unused"); }
  async appendObject(): Promise<void> { throw new Error("unused"); }
  async getObject(): Promise<StorageObject | null> { return null; }
  async exists(): Promise<boolean> { return false; }
  async deleteObject(): Promise<void> { /* unused */ }
  async deleteByPrefix(): Promise<number> { return 0; }
  async listObjects(_prefix: string, opts?: ListObjectsOptions): Promise<ListResult> {
    this.markers.push(opts?.marker);
    const start = opts?.marker ? this.keys.findIndex((key) => key > opts.marker!) : 0;
    const safeStart = start < 0 ? this.keys.length : start;
    const page = this.keys.slice(safeStart, safeStart + (opts?.maxKeys ?? 100));
    return {
      entries: page.map((key) => ({
        key,
        size: 1,
        lastModified: new Date(0),
        isDirectory: false,
      })),
      nextMarker: safeStart + page.length < this.keys.length ? page.at(-1) : undefined,
      total: this.keys.length,
    };
  }
}

class OrderedBackend implements IStorageBackend {
  readonly type = "local" as const;
  readonly objects = new Map<string, Buffer>();
  readonly mutations: string[] = [];
  failKey?: string;

  async putObject(key: string, content: string | Buffer): Promise<void> {
    this.mutations.push(`put:${key}`);
    if (key === this.failKey) throw new Error(`injected failure for ${key}`);
    this.objects.set(key, Buffer.from(content));
  }
  async appendObject(): Promise<void> { throw new Error("unused"); }
  async getObject(key: string): Promise<StorageObject | null> {
    const content = this.objects.get(key);
    return content
      ? { key, content: Buffer.from(content), size: content.length, lastModified: new Date(0) }
      : null;
  }
  async exists(key: string): Promise<boolean> { return this.objects.has(key); }
  async deleteObject(key: string): Promise<void> {
    this.mutations.push(`delete:${key}`);
    if (key === this.failKey) throw new Error(`injected failure for ${key}`);
    this.objects.delete(key);
  }
  async deleteByPrefix(prefix: string): Promise<number> {
    const keys = [...this.objects.keys()].filter((key) => key.startsWith(prefix));
    for (const key of keys) await this.deleteObject(key);
    return keys.length;
  }
  async listObjects(prefix: string, opts?: ListObjectsOptions): Promise<ListResult> {
    let keys = [...this.objects.keys()].filter((key) => key.startsWith(prefix)).sort();
    if (opts?.marker) keys = keys.filter((key) => key > opts.marker!);
    const page = keys.slice(0, opts?.maxKeys ?? 100);
    return {
      entries: page.map((key) => ({
        key,
        size: this.objects.get(key)!.length,
        lastModified: new Date(0),
        isDirectory: false,
      })),
      nextMarker: page.length < keys.length ? page.at(-1) : undefined,
      total: keys.length,
    };
  }
}

describe("transaction storage fencing", () => {
  it("paginates beyond ten thousand immutable receipts without truncation", async () => {
    const adapter = new StorageAdapter(new PagedBackend());
    const names = await adapter.readdirNames(".metadata/l1-transactions/", ".json");
    expect(names).toHaveLength(12_345);
    expect(names[0]).toBe("l1tx_00000.json");
    expect(names.at(-1)).toBe("l1tx_12344.json");
  });

  it("re-scopes every pagination marker across 12,345 per-instance receipts", async () => {
    const scope = "tenant-a/instance-b/";
    const backend = new PagedBackend(scope);
    const adapter = createScopedStorageAdapter(new StorageAdapter(backend), scope);

    const names = await adapter.readdirNames(".metadata/l1-transactions/", ".json");

    expect(names).toHaveLength(12_345);
    expect(names[0]).toBe("l1tx_00000.json");
    expect(names.at(-1)).toBe("l1tx_12344.json");
    expect(backend.markers[0]).toBeUndefined();
    expect(backend.markers.slice(1).every((marker) => marker?.startsWith(scope)))
      .toBe(true);
  });

  it("creates a local transaction owner exactly once and atomically replaces ordinary files", async () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-storage-fence-"));
    temporaryDirectories.push(root);
    const adapter = new StorageAdapter(new LocalStorageBackend(root));
    await expect(adapter.writeFileIfAbsent("tx/plan.json", "first")).resolves.toBe(true);
    await expect(adapter.writeFileIfAbsent("tx/plan.json", "second")).resolves.toBe(false);
    await expect(adapter.readFile("tx/plan.json")).resolves.toBe("first");
    await adapter.writeFile("state/current.json", "v1");
    await adapter.writeFile("state/current.json", "v2");
    await expect(adapter.readFile("state/current.json")).resolves.toBe("v2");
  });

  it("keeps profile writes and deletes invisible until the stage is committed", async () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-profile-stage-"));
    temporaryDirectories.push(root);
    const canonical = new StorageAdapter(new LocalStorageBackend(root));
    await canonical.writeFile("scene_blocks/a.md", "old-a");
    await canonical.writeFile("scene_blocks/b.md", "old-b");

    const transaction = createStagedStorageTransaction(canonical);
    await transaction.storage.writeFile("scene_blocks/a.md", "new-a");
    await transaction.storage.unlink("scene_blocks/b.md");
    await transaction.storage.writeFile("scene_blocks/c.md", "new-c");

    await expect(transaction.storage.readFile("scene_blocks/a.md")).resolves.toBe("new-a");
    await expect(transaction.storage.readdirNames("scene_blocks/", ".md"))
      .resolves.toEqual(["a.md", "c.md"]);
    await expect(canonical.readFile("scene_blocks/a.md")).resolves.toBe("old-a");
    await expect(canonical.readFile("scene_blocks/b.md")).resolves.toBe("old-b");
    await expect(canonical.readFile("scene_blocks/c.md")).resolves.toBeNull();

    await transaction.commit();

    await expect(canonical.readFile("scene_blocks/a.md")).resolves.toBe("new-a");
    await expect(canonical.readFile("scene_blocks/b.md")).resolves.toBeNull();
    await expect(canonical.readFile("scene_blocks/c.md")).resolves.toBe("new-c");
  });

  it("discards model tool writes when the queue lease aborts before publish", async () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-profile-abort-"));
    temporaryDirectories.push(root);
    const canonical = new StorageAdapter(new LocalStorageBackend(root));
    await canonical.writeFile("persona.md", "stable");
    const controller = new AbortController();
    const transaction = createStagedStorageTransaction(canonical, controller.signal);

    // This is the same StorageAdapter handed to the model's write/edit tools.
    await transaction.storage.writeFile("persona.md", "partial-model-output");
    controller.abort(new Error("lock lost"));

    await expect(transaction.commit(controller.signal)).rejects.toThrow("lock lost");
    transaction.discard();
    await expect(canonical.readFile("persona.md")).resolves.toBe("stable");
  });

  it("fails closed if profile code tries to stage an append-only journal", async () => {
    const root = mkdtempSync(join(tmpdir(), "tdai-profile-append-"));
    temporaryDirectories.push(root);
    const transaction = createStagedStorageTransaction(
      new StorageAdapter(new LocalStorageBackend(root)),
    );

    await expect(transaction.storage.appendFile("records/2026-09-01.jsonl", "row\n"))
      .rejects.toThrow("does not support append-only objects");
  });

  it("publishes ordinary profile data before the checkpoint and guards every mutation", async () => {
    const backend = new OrderedBackend();
    backend.objects.set("scene_blocks/obsolete.md", Buffer.from("old"));
    const transaction = createStagedStorageTransaction(new StorageAdapter(backend));
    await transaction.storage.writeFile(".metadata/checkpoint.json", "checkpoint-next");
    await transaction.storage.writeFile("persona.md", "persona-next");
    await transaction.storage.writeFile("scene_blocks/a.md", "scene-next");
    await transaction.storage.unlink("scene_blocks/obsolete.md");
    const guard = async (): Promise<void> => { backend.mutations.push("guard"); };

    await transaction.commitData(undefined, guard);

    expect(backend.objects.get("persona.md")?.toString()).toBe("persona-next");
    expect(backend.objects.get("scene_blocks/a.md")?.toString()).toBe("scene-next");
    expect(backend.objects.has("scene_blocks/obsolete.md")).toBe(false);
    expect(backend.objects.has(".metadata/checkpoint.json")).toBe(false);
    expect(backend.mutations).toEqual([
      "guard", "put:persona.md",
      "guard", "put:scene_blocks/a.md",
      "guard", "delete:scene_blocks/obsolete.md",
    ]);

    await transaction.commit(undefined, guard);
    expect(backend.mutations.slice(-2)).toEqual(["guard", "put:.metadata/checkpoint.json"]);
    expect(backend.objects.get(".metadata/checkpoint.json")?.toString()).toBe("checkpoint-next");
  });

  it("never advances the checkpoint when ordinary profile publication fails", async () => {
    const backend = new OrderedBackend();
    backend.failKey = "scene_blocks/b.md";
    const transaction = createStagedStorageTransaction(new StorageAdapter(backend));
    await transaction.storage.writeFile(".metadata/checkpoint.json", "checkpoint-next");
    await transaction.storage.writeFile("scene_blocks/a.md", "scene-a");
    await transaction.storage.writeFile("scene_blocks/b.md", "scene-b");

    await expect(transaction.commit()).rejects.toThrow("injected failure");

    expect(backend.objects.get("scene_blocks/a.md")?.toString()).toBe("scene-a");
    expect(backend.objects.has("scene_blocks/b.md")).toBe(false);
    expect(backend.objects.has(".metadata/checkpoint.json")).toBe(false);
  });
});
