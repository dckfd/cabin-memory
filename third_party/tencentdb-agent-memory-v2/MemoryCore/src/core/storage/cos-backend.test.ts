import { describe, expect, it, vi } from "vitest";

import type { CosCredential, ICredentialProvider } from "./types.js";
import {
  CosStorageBackend,
  SharedCosClient,
  type CosClientLike,
} from "./cos-backend.js";

const credential: CosCredential = {
  secretId: "test-id",
  secretKey: "test-key",
  bucket: "test-bucket-1000000000",
  region: "ap-test",
  prefix: "credential-prefix/",
};

function error(code: string, statusCode: number): Error & { code: string; statusCode: number } {
  return Object.assign(new Error(code), { code, statusCode });
}

function createBackend(client: CosClientLike, prefix = "tenant/instance/"): {
  backend: CosStorageBackend;
  provider: ICredentialProvider;
} {
  const provider: ICredentialProvider = {
    getCosCredential: vi.fn(async () => credential),
    invalidate: vi.fn(),
  };
  const sharedClient = new SharedCosClient({
    credentialProvider: provider,
    clientFactory: async () => client,
  });
  return {
    backend: new CosStorageBackend({ sharedClient, prefix }),
    provider,
  };
}

function unusedClient(overrides: Partial<CosClientLike>): CosClientLike {
  const unused = async (): Promise<never> => { throw new Error("unexpected COS call"); };
  return {
    putObject: unused,
    appendObject: unused,
    headObject: unused,
    getObject: unused,
    getBucket: unused,
    deleteObject: unused,
    deleteMultipleObject: unused,
    ...overrides,
  };
}

describe("COS storage transaction fencing", () => {
  it("creates exactly one immutable winner with APPEND position zero", async () => {
    const objects = new Set<string>();
    const appendObject = vi.fn(async (params: Record<string, unknown>) => {
      const key = String(params.Key);
      if (objects.has(key)) throw error("AppendPositionError", 409);
      expect(params.Position).toBe(0);
      objects.add(key);
      return {};
    });
    const headObject = vi.fn(async (params: Record<string, unknown>) => {
      if (!objects.has(String(params.Key))) throw error("NoSuchKey", 404);
      return { headers: { "content-length": "7" } };
    });
    const { backend } = createBackend(unusedClient({ appendObject, headObject }));

    const outcomes = await Promise.all(Array.from({ length: 12 }, () => (
      backend.putObjectIfAbsent(".metadata/l1-transactions/plan.json", "payload")
    )));

    expect(outcomes.filter(Boolean)).toHaveLength(1);
    expect(outcomes.filter((outcome) => !outcome)).toHaveLength(11);
    expect(appendObject).toHaveBeenCalledTimes(12);
    expect(appendObject.mock.calls[0]?.[0]).toMatchObject({
      Bucket: credential.bucket,
      Region: credential.region,
      Key: "tenant/instance/.metadata/l1-transactions/plan.json",
      Position: 0,
    });
  });

  it("treats a lost create acknowledgement as an uncertain non-winner", async () => {
    const appendObject = vi.fn(async () => { throw new Error("socket closed after commit"); });
    const headObject = vi.fn(async () => ({ headers: { "content-length": "7" } }));
    const { backend } = createBackend(unusedClient({ appendObject, headObject }));

    await expect(backend.putObjectIfAbsent("receipt.json", "payload")).resolves.toBe(false);
    expect(headObject).toHaveBeenCalledOnce();
  });

  it("rethrows an uncertain create failure when no durable object can be verified", async () => {
    const appendObject = vi.fn(async () => { throw new Error("socket closed before commit"); });
    const headObject = vi.fn(async () => { throw error("NoSuchKey", 404); });
    const { backend } = createBackend(unusedClient({ appendObject, headObject }));

    await expect(backend.putObjectIfAbsent("receipt.json", "payload"))
      .rejects.toThrow("socket closed before commit");
  });

  it("re-reads append position after a concurrent writer wins", async () => {
    const headObject = vi.fn()
      .mockRejectedValueOnce(error("NoSuchKey", 404))
      .mockResolvedValueOnce({ headers: { "content-length": "4" } });
    const appendObject = vi.fn()
      .mockRejectedValueOnce(error("AppendPositionError", 409))
      .mockResolvedValueOnce({});
    const { backend } = createBackend(unusedClient({ appendObject, headObject }));

    await expect(backend.appendObject("records/2026-09-01.jsonl", "next")).resolves.toBeUndefined();
    expect(appendObject.mock.calls.map((call) => call[0].Position)).toEqual([0, 4]);
  });

  it("scopes both list prefix and continuation marker and returns relative keys", async () => {
    const getBucket = vi.fn(async () => ({
      Contents: [{
        Key: "tenant/instance/.metadata/l1-transactions/r-2.json",
        Size: "5",
        LastModified: "2026-09-01T00:00:00.000Z",
      }],
      CommonPrefixes: [],
      IsTruncated: "true",
      NextMarker: "tenant/instance/.metadata/l1-transactions/r-2.json",
    }));
    const { backend } = createBackend(unusedClient({ getBucket }));

    await expect(backend.listObjects(".metadata/l1-transactions/", {
      recursive: true,
      marker: ".metadata/l1-transactions/r-1.json",
      maxKeys: 100,
    })).resolves.toMatchObject({
      entries: [{ key: ".metadata/l1-transactions/r-2.json", size: 5 }],
      nextMarker: ".metadata/l1-transactions/r-2.json",
    });
    expect(getBucket).toHaveBeenCalledWith(expect.objectContaining({
      Prefix: "tenant/instance/.metadata/l1-transactions/",
      Marker: "tenant/instance/.metadata/l1-transactions/r-1.json",
    }));
  });

  it("fails closed if COS returns a key outside the instance namespace", async () => {
    const getBucket = vi.fn(async () => ({
      Contents: [{ Key: "another-instance/private.json", Size: "1" }],
      IsTruncated: "false",
    }));
    const { backend } = createBackend(unusedClient({ getBucket }));

    await expect(backend.listObjects("", { recursive: true }))
      .rejects.toThrow("out-of-scope key");
  });
});
