/**
 * Copy-on-write storage used by one L2/L3 execution.
 *
 * Long-running model tools must never receive the canonical profile backend:
 * they can issue several writes before the provider call returns, while the
 * queue worker may lose its distributed lease at any point.  This backend
 * keeps those writes and tombstones in memory, provides read-your-writes, and
 * exposes an explicit commit that the pipeline invokes only after its final
 * lease check.
 *
 * This is deliberately scoped to ordinary profile objects.  Conditional
 * create and append-only objects (L0/L1 journals and transaction receipts)
 * retain their dedicated durable protocols and are rejected here.
 */

import type {
  IStorageBackend,
  ListEntry,
  ListObjectsOptions,
  ListResult,
  PutObjectOptions,
  StorageObject,
} from "./types.js";

interface StagedWrite {
  content: Buffer;
  contentType?: string;
  metadata?: Record<string, string>;
  modifiedAt: Date;
}

type StageState =
  | "open"
  | "publishing-data"
  | "data-published"
  | "committing-checkpoint"
  | "committed"
  | "discarded";

type MutationGuard = () => Promise<void>;

function isCheckpointKey(key: string): boolean {
  // Storage-backed CheckpointManager uses the first path.  Keep the legacy
  // spelling fenced as well so a model tool or migrated deployment cannot
  // accidentally publish an old-format checkpoint ahead of profile data.
  return key === ".metadata/checkpoint.json"
    || key === ".metadata/recall_checkpoint.json";
}

function abortError(signal: AbortSignal | undefined, context: string): Error | undefined {
  if (!signal?.aborted) return undefined;
  if (signal.reason instanceof Error) return signal.reason;
  return new Error(`${context}: aborted`);
}

function validateObjectKey(key: string): void {
  if (!key || typeof key !== "string" || key.includes("\0") || key.startsWith("/") || key.startsWith("\\")) {
    throw new Error(`Invalid staged storage key: ${JSON.stringify(key)}`);
  }
  const normalized = key.replace(/\\+/g, "/");
  if (normalized.split("/").some((part) => part === "..")) {
    throw new Error(`Path traversal rejected in staged storage key: ${key}`);
  }
}

function cloneMetadata(metadata: Record<string, string> | undefined): Record<string, string> | undefined {
  return metadata ? { ...metadata } : undefined;
}

/**
 * Read-through, read-your-writes overlay for one profile transaction.
 *
 * `commit()` is intentionally the only operation that can touch `base`.
 * Callers must run their distributed-lease guard immediately before it.
 */
export class StagedStorageBackend implements IStorageBackend {
  readonly type: "local" | "cos";
  private readonly writes = new Map<string, StagedWrite>();
  private readonly tombstones = new Set<string>();
  private state: StageState = "open";

  constructor(
    private readonly base: IStorageBackend,
    private readonly abortSignal?: AbortSignal,
  ) {
    this.type = base.type;
  }

  private assertOpen(context: string): void {
    const aborted = abortError(this.abortSignal, context);
    if (aborted) throw aborted;
    if (this.state !== "open") {
      throw new Error(`Staged storage is ${this.state}; ${context} is not allowed`);
    }
  }

  private assertReadable(context: string): void {
    const aborted = abortError(this.abortSignal, context);
    if (aborted) throw aborted;
    if (this.state === "discarded") {
      throw new Error(`Staged storage is discarded; ${context} is not allowed`);
    }
  }

  async putObject(key: string, content: string | Buffer, opts?: PutObjectOptions): Promise<void> {
    this.assertOpen("putObject");
    validateObjectKey(key);
    this.tombstones.delete(key);
    this.writes.set(key, {
      content: Buffer.from(content),
      contentType: opts?.contentType,
      metadata: cloneMetadata(opts?.metadata),
      modifiedAt: new Date(),
    });
  }

  async appendObject(_key: string, _content: string | Buffer): Promise<void> {
    this.assertOpen("appendObject");
    throw new Error(
      "Staged profile storage does not support append-only objects; " +
      "use the dedicated L0/L1 transaction protocol",
    );
  }

  async getObject(key: string): Promise<StorageObject | null> {
    this.assertReadable("getObject");
    validateObjectKey(key);
    if (this.tombstones.has(key)) return null;
    const staged = this.writes.get(key);
    if (staged) {
      return {
        key,
        content: Buffer.from(staged.content),
        contentType: staged.contentType,
        metadata: cloneMetadata(staged.metadata),
        lastModified: new Date(staged.modifiedAt),
        size: staged.content.length,
      };
    }
    return this.base.getObject(key);
  }

  async exists(key: string): Promise<boolean> {
    return (await this.getObject(key)) !== null;
  }

  async listObjects(prefix: string, opts?: ListObjectsOptions): Promise<ListResult> {
    this.assertReadable("listObjects");
    if (typeof prefix !== "string" || prefix.includes("\0") || prefix.startsWith("/") || prefix.startsWith("\\")) {
      throw new Error(`Invalid staged storage prefix: ${JSON.stringify(prefix)}`);
    }
    if (prefix.replace(/\\+/g, "/").split("/").some((part) => part === "..")) {
      throw new Error(`Path traversal rejected in staged storage prefix: ${prefix}`);
    }

    // Fetch the complete base view. Profile transactions are bounded by the
    // configured scene cap, and full collection is required to merge staged
    // creations/deletions without page-boundary gaps.
    const baseFiles = new Map<string, ListEntry>();
    const seenMarkers = new Set<string>();
    let marker: string | undefined;
    do {
      const page = await this.base.listObjects(prefix, {
        recursive: true,
        maxKeys: 1000,
        ...(marker ? { marker } : {}),
      });
      for (const entry of page.entries) {
        if (!entry.isDirectory) baseFiles.set(entry.key, entry);
      }
      const next = page.nextMarker;
      if (!next) break;
      if (next === marker || seenMarkers.has(next)) {
        throw new Error(`Staged storage pagination marker did not advance for prefix: ${prefix}`);
      }
      seenMarkers.add(next);
      marker = next;
    } while (true);

    for (const key of this.tombstones) {
      if (key.startsWith(prefix)) baseFiles.delete(key);
    }
    for (const [key, staged] of this.writes) {
      if (!key.startsWith(prefix)) continue;
      baseFiles.set(key, {
        key,
        size: staged.content.length,
        lastModified: new Date(staged.modifiedAt),
        isDirectory: false,
      });
    }

    let entries: ListEntry[];
    if (opts?.recursive) {
      entries = [...baseFiles.values()];
    } else {
      const shallow = new Map<string, ListEntry>();
      for (const entry of baseFiles.values()) {
        const rest = entry.key.slice(prefix.length);
        const slash = rest.indexOf("/");
        if (slash < 0) {
          shallow.set(entry.key, entry);
        } else {
          const directoryKey = `${prefix}${rest.slice(0, slash + 1)}`;
          if (!shallow.has(directoryKey)) {
            shallow.set(directoryKey, {
              key: directoryKey,
              size: 0,
              lastModified: entry.lastModified,
              isDirectory: true,
            });
          }
        }
      }
      entries = [...shallow.values()];
    }

    entries.sort((left, right) => left.key < right.key ? -1 : left.key > right.key ? 1 : 0);
    if (opts?.marker) entries = entries.filter((entry) => entry.key > opts.marker!);
    const total = entries.length;
    const maxKeys = Math.max(1, opts?.maxKeys ?? 100);
    const page = entries.slice(0, maxKeys);
    return {
      entries: page,
      nextMarker: page.length < total ? page.at(-1)?.key : undefined,
      total,
    };
  }

  async deleteObject(key: string): Promise<void> {
    this.assertOpen("deleteObject");
    validateObjectKey(key);
    this.writes.delete(key);
    this.tombstones.add(key);
  }

  async deleteByPrefix(prefix: string): Promise<number> {
    this.assertOpen("deleteByPrefix");
    let deleted = 0;
    const seen = new Set<string>();
    let marker: string | undefined;
    do {
      const page = await this.listObjects(prefix, {
        recursive: true,
        maxKeys: 1000,
        ...(marker ? { marker } : {}),
      });
      for (const entry of page.entries) {
        if (entry.isDirectory || seen.has(entry.key)) continue;
        seen.add(entry.key);
        await this.deleteObject(entry.key);
        deleted++;
      }
      marker = page.nextMarker;
    } while (marker);
    return deleted;
  }

  hasChanges(): boolean {
    return this.writes.size > 0 || this.tombstones.size > 0;
  }

  private async guardMutation(
    signal: AbortSignal | undefined,
    mutationGuard: MutationGuard | undefined,
    context: string,
  ): Promise<void> {
    const beforeGuardAbort = abortError(signal ?? this.abortSignal, context);
    if (beforeGuardAbort) throw beforeGuardAbort;
    await mutationGuard?.();
    const afterGuardAbort = abortError(signal ?? this.abortSignal, context);
    if (afterGuardAbort) throw afterGuardAbort;
  }

  private async publishSubset(
    predicate: (key: string) => boolean,
    signal?: AbortSignal,
    mutationGuard?: MutationGuard,
  ): Promise<void> {
    // Destinations are made durable before sources are removed. This keeps
    // rename/copy-style edits recoverable if a backend call fails midway.
    const writes = [...this.writes.entries()]
      .filter(([key]) => predicate(key))
      .sort(([left], [right]) => left.localeCompare(right));
    for (const [key, staged] of writes) {
      await this.guardMutation(signal, mutationGuard, `staged storage put ${key}`);
      await this.base.putObject(key, staged.content, {
        contentType: staged.contentType,
        metadata: cloneMetadata(staged.metadata),
      });
      this.writes.delete(key);
    }

    const tombstones = [...this.tombstones]
      .filter(predicate)
      .sort((left, right) => left.localeCompare(right));
    for (const key of tombstones) {
      await this.guardMutation(signal, mutationGuard, `staged storage delete ${key}`);
      await this.base.deleteObject(key);
      this.tombstones.delete(key);
    }
  }

  /**
   * Publish every ordinary profile/index/persona object but retain checkpoint
   * writes in the overlay.  Pipelines call this only after strict VDB sync;
   * the checkpoint is their final commit marker.
   */
  async commitData(signal?: AbortSignal, mutationGuard?: MutationGuard): Promise<void> {
    this.assertOpen("commitData");
    this.state = "publishing-data";
    try {
      await this.publishSubset((key) => !isCheckpointKey(key), signal, mutationGuard);
      this.state = "data-published";
    } catch (error) {
      // Some ordinary objects may already be durable.  The checkpoint remains
      // unpublished and the pending queue task will reconcile from VDB.
      this.state = "discarded";
      this.writes.clear();
      this.tombstones.clear();
      throw error;
    }
  }

  /**
   * Finish publication with checkpoint objects last.  Direct callers retain
   * the old one-call API; it now internally enforces data-before-checkpoint.
   */
  async commit(signal?: AbortSignal, mutationGuard?: MutationGuard): Promise<void> {
    if (this.state === "open") {
      await this.commitData(signal, mutationGuard);
    }
    if (this.state !== "data-published") {
      throw new Error(`Staged storage is ${this.state}; commit is not allowed`);
    }

    this.state = "committing-checkpoint";
    try {
      await this.publishSubset(isCheckpointKey, signal, mutationGuard);
      if (this.writes.size > 0 || this.tombstones.size > 0) {
        throw new Error("Staged storage commit left non-checkpoint objects unpublished");
      }
      this.state = "committed";
    } catch (error) {
      this.state = "discarded";
      this.writes.clear();
      this.tombstones.clear();
      throw error;
    }
  }

  discard(): void {
    if (this.state === "committed") {
      throw new Error("Cannot discard a committed staged storage transaction");
    }
    this.state = "discarded";
    this.writes.clear();
    this.tombstones.clear();
  }
}
