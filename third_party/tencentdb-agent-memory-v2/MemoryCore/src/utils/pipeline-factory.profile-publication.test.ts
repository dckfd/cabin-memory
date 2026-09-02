import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const runnerHarness = vi.hoisted(() => ({
  l1Records: [] as Array<Record<string, unknown>>,
  sceneRun: undefined as undefined | ((options: Record<string, unknown>) => Promise<Record<string, unknown>>),
  personaRun: undefined as undefined | ((options: Record<string, unknown>) => Promise<boolean>),
}));

vi.mock("../core/record/l1-reader.js", () => ({
  queryMemoryRecords: vi.fn(async () => structuredClone(runnerHarness.l1Records)),
}));

vi.mock("../core/scene/scene-extractor.js", () => ({
  SceneExtractor: class {
    constructor(private readonly options: Record<string, unknown>) {}
    async extract(): Promise<Record<string, unknown>> {
      if (!runnerHarness.sceneRun) throw new Error("scene harness not configured");
      return runnerHarness.sceneRun(this.options);
    }
  },
}));

vi.mock("../core/persona/persona-trigger.js", () => ({
  PersonaTrigger: class {
    async shouldGenerate(): Promise<{ should: boolean; reason: string }> {
      return { should: true, reason: "publication fault test" };
    }
  },
}));

vi.mock("../core/persona/persona-generator.js", () => ({
  PersonaGenerator: class {
    constructor(private readonly options: Record<string, unknown>) {}
    async generateLocalPersona(): Promise<boolean> {
      if (!runnerHarness.personaRun) throw new Error("persona harness not configured");
      return runnerHarness.personaRun(this.options);
    }
  },
}));

import { parseConfig } from "../config.js";
import {
  buildProfileStableId,
} from "../core/profile/profile-sync.js";
import { StorageAdapter } from "../core/storage/adapter.js";
import type {
  IStorageBackend,
  ListObjectsOptions,
  ListResult,
  StorageObject,
} from "../core/storage/types.js";
import type {
  IMemoryStore,
  ProfileRecord,
  ProfileSyncOptions,
  ProfileSyncRecord,
} from "../core/store/types.js";
import {
  buildProfileL2Key,
  createL2Runner,
  createL3Runner,
} from "./pipeline-factory.js";

const SCOPE = "team:team-a|agent:agent-a";
const PREFIX = `profiles/${encodeURIComponent(SCOPE)}/`;
const SCENE = "驾驶偏好.md";
const SCENE_A = "导航偏好.md";
const SCENE_B = "温控偏好.md";

function md5(content: string): string {
  return createHash("md5").update(content).digest("hex");
}

class FaultBackend implements IStorageBackend {
  readonly type = "local" as const;
  readonly objects = new Map<string, Buffer>();
  readonly events: string[] = [];
  failSuffix?: string;

  async putObject(key: string, content: string | Buffer): Promise<void> {
    this.events.push(`put:${key}`);
    if (this.failSuffix && key.endsWith(this.failSuffix)) {
      throw new Error(`injected storage failure: ${this.failSuffix}`);
    }
    this.objects.set(key, Buffer.from(content));
  }
  async appendObject(): Promise<void> { throw new Error("append is not used"); }
  async getObject(key: string): Promise<StorageObject | null> {
    const content = this.objects.get(key);
    return content
      ? { key, content: Buffer.from(content), size: content.length, lastModified: new Date(1) }
      : null;
  }
  async exists(key: string): Promise<boolean> { return this.objects.has(key); }
  async deleteObject(key: string): Promise<void> {
    this.events.push(`delete:${key}`);
    if (this.failSuffix && key.endsWith(this.failSuffix)) {
      throw new Error(`injected storage failure: ${this.failSuffix}`);
    }
    this.objects.delete(key);
  }
  async deleteByPrefix(prefix: string): Promise<number> {
    const keys = [...this.objects.keys()].filter((key) => key.startsWith(prefix));
    for (const key of keys) await this.deleteObject(key);
    return keys.length;
  }
  async listObjects(prefix: string, options?: ListObjectsOptions): Promise<ListResult> {
    let keys = [...this.objects.keys()].filter((key) => key.startsWith(prefix)).sort();
    if (options?.marker) keys = keys.filter((key) => key > options.marker!);
    const page = keys.slice(0, options?.maxKeys ?? 100);
    return {
      entries: page.map((key) => ({
        key,
        size: this.objects.get(key)!.length,
        lastModified: new Date(1),
        isDirectory: false,
      })),
      nextMarker: page.length < keys.length ? page.at(-1) : undefined,
      total: keys.length,
    };
  }
}

interface RemoteHarness {
  profiles: ProfileRecord[];
  events: string[];
  writes: number;
}

function createRemoteStore(remote: RemoteHarness): IMemoryStore {
  const store = {
    isDegraded: () => false,
    pullProfiles: async () => structuredClone(remote.profiles),
    syncProfiles: async (records: ProfileSyncRecord[], options?: ProfileSyncOptions) => {
      await options?.mutationGuard?.();
      if (options?.abortSignal?.aborted) throw options.abortSignal.reason;
      remote.events.push("vdb-upsert");
      remote.writes++;
      const next = new Map(remote.profiles.map((profile) => [profile.id, profile] as const));
      for (const record of records) {
        next.set(record.id, {
          ...record,
          version: (record.baselineVersion ?? -1) + 1,
          updatedAtMs: Date.now(),
        });
      }
      remote.profiles = [...next.values()];
    },
    deleteProfiles: async (ids: string[], options?: ProfileSyncOptions) => {
      await options?.mutationGuard?.();
      if (options?.abortSignal?.aborted) throw options.abortSignal.reason;
      remote.events.push("vdb-delete");
      remote.writes++;
      const removed = new Set(ids);
      remote.profiles = remote.profiles.filter((profile) => !removed.has(profile.id));
    },
  };
  return store as unknown as IMemoryStore;
}

const logger = {
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
};

function config() {
  return parseConfig({
    persona: {
      triggerEveryN: 1,
      promptMode: "cockpit",
    },
  });
}

function l1Record(): Record<string, unknown> {
  return {
    id: "l1-a",
    content: "把导航设为避开高速，并把温度设为二十二度。",
    createdAt: "2026-09-01T00:00:00.000Z",
    updatedAt: "2026-09-01T00:00:01.000Z",
    teamId: "team-a",
    agentId: "agent-a",
    sessionId: "session-a",
  };
}

function remoteScene(filename = SCENE, content = "稳定场景"): ProfileRecord {
  return {
    id: buildProfileStableId(SCOPE, "l2", filename),
    type: "l2",
    filename,
    content,
    contentMd5: md5(content),
    teamId: "team-a",
    agentId: "agent-a",
    version: 1,
    createdAtMs: 1,
    updatedAtMs: 1,
  };
}

beforeEach(() => {
  runnerHarness.l1Records = [l1Record()];
  runnerHarness.sceneRun = undefined;
  runnerHarness.personaRun = undefined;
  vi.clearAllMocks();
});

describe("L2/L3 profile publication ordering", () => {
  it("does not let a lost L2 attempt mutate VDB before its exact lease is revalidated", async () => {
    const backend = new FaultBackend();
    const remote: RemoteHarness = { profiles: [], events: backend.events, writes: 0 };
    const storage = new StorageAdapter(backend);
    const commitGuard = vi.fn(async () => { throw new Error("replacement owns lease"); });
    runnerHarness.sceneRun = async (options) => {
      const staged = options.storage as StorageAdapter;
      await staged.writeFile(`scene_blocks/${SCENE_A}`, "避开高速");
      await staged.writeFile(".metadata/scene_index.json", JSON.stringify([{ filename: SCENE_A }]));
      return { success: true, memoriesProcessed: 1, emptyExtraction: false };
    };
    const runner = createL2Runner({
      pluginDataDir: "/virtual",
      cfg: config(),
      openclawConfig: {},
      vectorStore: createRemoteStore(remote),
      logger,
      storage,
      commitGuard,
    });

    await expect(runner(buildProfileL2Key({
      teamId: "team-a",
      agentId: "agent-a",
      sessionId: "session-a",
    }))).rejects.toThrow("replacement owns lease");

    expect(commitGuard).toHaveBeenCalledOnce();
    expect(remote.writes).toBe(0);
    expect([...backend.objects.keys()].filter((key) => key.startsWith(PREFIX))).toEqual([]);
  });

  it("keeps checkpoint old when COS fails after VDB, then reconciles data from VDB on retry", async () => {
    const backend = new FaultBackend();
    backend.failSuffix = `scene_blocks/${SCENE_B}`;
    const remote: RemoteHarness = { profiles: [], events: backend.events, writes: 0 };
    const storage = new StorageAdapter(backend);
    const commitGuard = vi.fn(async () => { backend.events.push("lease"); });
    let firstRun = true;
    runnerHarness.sceneRun = async (options) => {
      if (!firstRun) return { success: true, memoriesProcessed: 0 };
      const staged = options.storage as StorageAdapter;
      await staged.writeFile(`scene_blocks/${SCENE_A}`, "避开高速");
      await staged.writeFile(`scene_blocks/${SCENE_B}`, "二十二度");
      await staged.writeFile(".metadata/scene_index.json", JSON.stringify([
        { filename: SCENE_A },
        { filename: SCENE_B },
      ]));
      return { success: true, memoriesProcessed: 1, emptyExtraction: false };
    };
    const runner = createL2Runner({
      pluginDataDir: "/virtual",
      cfg: config(),
      openclawConfig: {},
      vectorStore: createRemoteStore(remote),
      logger,
      storage,
      commitGuard,
    });
    const l2Key = buildProfileL2Key({
      teamId: "team-a",
      agentId: "agent-a",
      sessionId: "session-a",
    });

    await expect(runner(l2Key)).rejects.toThrow("injected storage failure");
    expect(remote.writes).toBe(1);
    expect(backend.objects.has(`${PREFIX}.metadata/checkpoint.json`)).toBe(false);
    expect(backend.events.indexOf("vdb-upsert")).toBeLessThan(
      backend.events.findIndex((event) => event.startsWith("put:")),
    );

    backend.failSuffix = undefined;
    firstRun = false;
    await expect(runner(l2Key)).resolves.toBeUndefined();
    expect(backend.objects.get(`${PREFIX}scene_blocks/${SCENE_A}`)?.toString()).toBe("避开高速");
    expect(backend.objects.get(`${PREFIX}scene_blocks/${SCENE_B}`)?.toString()).toBe("二十二度");
    expect(backend.objects.has(`${PREFIX}.metadata/checkpoint.json`)).toBe(false);
  });

  it("fences an L3 VDB write when the attempt lease was replaced", async () => {
    const backend = new FaultBackend();
    const scene = remoteScene();
    backend.objects.set(`${PREFIX}scene_blocks/${SCENE}`, Buffer.from(scene.content));
    backend.objects.set(`${PREFIX}.metadata/scene_index.json`, Buffer.from(JSON.stringify([{ filename: SCENE }])));
    const remote: RemoteHarness = { profiles: [scene], events: backend.events, writes: 0 };
    const commitGuard = vi.fn(async () => { throw new Error("L3 lease lost"); });
    runnerHarness.personaRun = async (options) => {
      await (options.storage as StorageAdapter).writeFile("persona.md", "稳定画像");
      return true;
    };
    const runner = createL3Runner({
      pluginDataDir: "/virtual",
      cfg: config(),
      openclawConfig: {},
      vectorStore: createRemoteStore(remote),
      logger,
      storage: new StorageAdapter(backend),
      profileScope: SCOPE,
      commitGuard,
    });

    await expect(runner()).rejects.toThrow("L3 lease lost");
    expect(remote.writes).toBe(0);
    expect(backend.objects.has(`${PREFIX}persona.md`)).toBe(false);
    expect(backend.objects.has(`${PREFIX}.metadata/checkpoint.json`)).toBe(false);
  });

  it("publishes L3 persona and VDB but leaves checkpoint old when its final write fails", async () => {
    const backend = new FaultBackend();
    const scene = remoteScene();
    backend.objects.set(`${PREFIX}scene_blocks/${SCENE}`, Buffer.from(scene.content));
    backend.objects.set(`${PREFIX}.metadata/scene_index.json`, Buffer.from(JSON.stringify([{ filename: SCENE }])));
    backend.failSuffix = ".metadata/checkpoint.json";
    const remote: RemoteHarness = { profiles: [scene], events: backend.events, writes: 0 };
    const commitGuard = vi.fn(async () => { backend.events.push("lease"); });
    runnerHarness.personaRun = async (options) => {
      await (options.storage as StorageAdapter).writeFile("persona.md", "稳定画像");
      return true;
    };
    const runner = createL3Runner({
      pluginDataDir: "/virtual",
      cfg: config(),
      openclawConfig: {},
      vectorStore: createRemoteStore(remote),
      logger,
      storage: new StorageAdapter(backend),
      profileScope: SCOPE,
      commitGuard,
    });

    await expect(runner()).rejects.toThrow("injected storage failure");
    expect(remote.writes).toBe(1);
    expect(backend.objects.get(`${PREFIX}persona.md`)?.toString()).toContain("稳定画像");
    expect(backend.objects.has(`${PREFIX}.metadata/checkpoint.json`)).toBe(false);
    expect(backend.events.at(-1)).toBe(`put:${PREFIX}.metadata/checkpoint.json`);
  });
});
