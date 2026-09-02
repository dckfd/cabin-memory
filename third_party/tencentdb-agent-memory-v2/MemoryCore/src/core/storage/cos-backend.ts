/**
 * Tencent Cloud COS implementation of IStorageBackend.
 *
 * The backend deliberately lives in public core storage rather than an
 * optional private integration: service-mode transaction fencing must have
 * the same auditable semantics as standalone local storage.
 */

import type {
  CosCredential,
  ICredentialProvider,
  IStorageBackend,
  ListEntry,
  ListObjectsOptions,
  ListResult,
  PutObjectOptions,
  StorageLogger,
  StorageObject,
} from "./types.js";

const TAG = "[storage][cos]";
const MAX_LIST_KEYS = 1_000;
const MAX_APPEND_ATTEMPTS = 6;

interface CosErrorLike extends Error {
  code?: string;
  statusCode?: number;
  headers?: Record<string, unknown>;
}

interface CosObjectSummary {
  Key?: string;
  Size?: string | number;
  LastModified?: string;
}

export interface CosClientLike {
  putObject(params: Record<string, unknown>): Promise<Record<string, unknown>>;
  appendObject(params: Record<string, unknown>): Promise<Record<string, unknown>>;
  headObject(params: Record<string, unknown>): Promise<Record<string, unknown>>;
  getObject(params: Record<string, unknown>): Promise<Record<string, unknown>>;
  getBucket(params: Record<string, unknown>): Promise<{
    Contents?: CosObjectSummary[];
    CommonPrefixes?: Array<{ Prefix?: string }>;
    IsTruncated?: string | boolean;
    NextMarker?: string;
  }>;
  deleteObject(params: Record<string, unknown>): Promise<Record<string, unknown>>;
  deleteMultipleObject(params: Record<string, unknown>): Promise<{
    Deleted?: Array<{ Key?: string }>;
    Error?: Array<{ Key?: string; Code?: string; Message?: string }>;
  }>;
}

type CosConstructor = new (options: Record<string, unknown>) => CosClientLike;

export interface SharedCosClientOptions {
  credentialProvider: ICredentialProvider;
  logger?: StorageLogger;
  /** Domain suffix without the bucket, for example cos-internal.ap-guangzhou.tencentcos.cn. */
  cosEndpointDomain?: string;
  /** Test seam; production loads cos-nodejs-sdk-v5 lazily. */
  clientFactory?: (credential: CosCredential, endpointDomain?: string) => CosClientLike | Promise<CosClientLike>;
}

export interface SharedCosContext {
  client: CosClientLike;
  credential: CosCredential;
}

/**
 * One credential-aware COS SDK client shared by all per-instance prefixes.
 * A refreshed credential object causes the SDK client to be rebuilt.
 */
export class SharedCosClient {
  private readonly credentialProvider: ICredentialProvider;
  private readonly logger?: StorageLogger;
  private readonly cosEndpointDomain?: string;
  private readonly clientFactory?: SharedCosClientOptions["clientFactory"];
  private cachedCredential: CosCredential | null = null;
  private cachedClient: CosClientLike | null = null;

  constructor(options: SharedCosClientOptions) {
    this.credentialProvider = options.credentialProvider;
    this.logger = options.logger;
    this.cosEndpointDomain = options.cosEndpointDomain;
    this.clientFactory = options.clientFactory;
  }

  async getContext(): Promise<SharedCosContext> {
    const credential = await this.credentialProvider.getCosCredential();
    if (!this.cachedClient || this.cachedCredential !== credential) {
      this.cachedClient = this.clientFactory
        ? await this.clientFactory(credential, this.cosEndpointDomain)
        : await this.createSdkClient(credential);
      this.cachedCredential = credential;
      this.logger?.debug?.(`${TAG} SDK client refreshed for bucket=${credential.bucket}`);
    }
    return { client: this.cachedClient, credential };
  }

  async getClient(): Promise<CosClientLike> {
    return (await this.getContext()).client;
  }

  invalidate(): void {
    this.credentialProvider.invalidate();
    this.cachedCredential = null;
    this.cachedClient = null;
  }

  private async createSdkClient(credential: CosCredential): Promise<CosClientLike> {
    // Keep the optional dependency genuinely optional for standalone builds.
    const packageName = "cos-nodejs-sdk-v5";
    const module = await import(packageName) as { default?: CosConstructor } & Record<string, unknown>;
    const Constructor = (module.default ?? module) as unknown as CosConstructor;
    return new Constructor({
      SecretId: credential.secretId,
      SecretKey: credential.secretKey,
      SecurityToken: credential.token,
      Domain: this.cosEndpointDomain ? `{Bucket}.${this.cosEndpointDomain}` : undefined,
      Protocol: "https:",
      KeepAlive: true,
    });
  }
}

export interface CosStorageBackendOptions {
  sharedClient?: SharedCosClient;
  credentialProvider?: ICredentialProvider;
  prefix?: string;
  logger?: StorageLogger;
  cosEndpointDomain?: string;
}

export class CosStorageBackend implements IStorageBackend {
  readonly type = "cos" as const;
  private readonly sharedClient: SharedCosClient;
  private readonly prefixOverride?: string;
  private readonly logger?: StorageLogger;

  constructor(options: CosStorageBackendOptions) {
    if (!options.sharedClient && !options.credentialProvider) {
      throw new Error(`${TAG} sharedClient or credentialProvider is required`);
    }
    this.sharedClient = options.sharedClient ?? new SharedCosClient({
      credentialProvider: options.credentialProvider!,
      logger: options.logger,
      cosEndpointDomain: options.cosEndpointDomain,
    });
    this.prefixOverride = options.prefix === undefined
      ? undefined
      : normalizePrefix(options.prefix);
    this.logger = options.logger;
  }

  async putObject(key: string, content: string | Buffer, options?: PutObjectOptions): Promise<void> {
    const buffer = toBuffer(content);
    await this.withContext(async ({ client, credential }) => {
      await client.putObject({
        ...this.objectIdentity(credential, key),
        Body: buffer,
        ContentLength: buffer.length,
        ContentType: options?.contentType,
        Headers: metadataHeaders(options),
      });
    });
  }

  /**
   * Atomic create-if-absent via APPEND Object at position zero.
   *
   * COS appendable objects do not participate in versioning, and the service
   * atomically rejects position zero when the key already exists.  This avoids
   * the versioning caveat of x-cos-forbid-overwrite and is a natural fit for
   * immutable transaction plans and stage markers.
   */
  async putObjectIfAbsent(
    key: string,
    content: string | Buffer,
    options?: PutObjectOptions,
  ): Promise<boolean> {
    const buffer = toBuffer(content);
    try {
      await this.withContext(async ({ client, credential }) => {
        await client.appendObject({
          ...this.objectIdentity(credential, key),
          Body: buffer,
          ContentLength: buffer.length,
          ContentType: options?.contentType,
          Position: 0,
          Headers: metadataHeaders(options),
        });
      });
      return true;
    } catch (error) {
      // A timeout can be a lost acknowledgement after COS created the object.
      // In both that case and a genuine competing-writer conflict, returning
      // false makes the caller read and validate the immutable winner.
      if (isConflict(error) || await this.existsAfterUncertainCreate(key)) {
        return false;
      }
      throw error;
    }
  }

  async appendObject(key: string, content: string | Buffer): Promise<void> {
    const buffer = toBuffer(content);
    for (let attempt = 1; attempt <= MAX_APPEND_ATTEMPTS; attempt++) {
      const position = await this.objectLength(key);
      try {
        await this.withContext(async ({ client, credential }) => {
          await client.appendObject({
            ...this.objectIdentity(credential, key),
            Body: buffer,
            ContentLength: buffer.length,
            Position: position,
            Headers: {},
          });
        });
        return;
      } catch (error) {
        if (errorCode(error) === "ObjectNotAppendable") {
          throw new Error(`${TAG} cannot append to normal COS object: ${key}`, { cause: error });
        }
        if (!isConflict(error) || attempt === MAX_APPEND_ATTEMPTS) throw error;
      }
    }
  }

  async getObject(key: string): Promise<StorageObject | null> {
    try {
      return await this.withContext(async ({ client, credential }) => {
        const result = await client.getObject({
          ...this.objectIdentity(credential, key),
          Headers: {},
        });
        const headers = asHeaders(result.headers);
        const body = result.Body;
        const content = Buffer.isBuffer(body) ? body : Buffer.from(String(body ?? ""));
        return {
          key,
          content,
          contentType: headerValue(headers, "content-type"),
          metadata: extractMetadata(headers),
          lastModified: parseDate(headerValue(headers, "last-modified")),
          size: numberValue(headerValue(headers, "content-length")) ?? content.length,
        };
      });
    } catch (error) {
      if (isNotFound(error)) return null;
      throw error;
    }
  }

  async exists(key: string): Promise<boolean> {
    try {
      await this.withContext(async ({ client, credential }) => {
        await client.headObject({ ...this.objectIdentity(credential, key), Headers: {} });
      });
      return true;
    } catch (error) {
      if (isNotFound(error)) return false;
      throw error;
    }
  }

  async listObjects(prefix: string, options?: ListObjectsOptions): Promise<ListResult> {
    return this.withContext(async ({ client, credential }) => {
      const namespacePrefix = this.namespacePrefix(credential);
      const fullPrefix = this.fullKey(credential, prefix, true);
      const marker = options?.marker
        ? this.fullKey(credential, options.marker, false)
        : undefined;
      const maxKeys = Math.max(1, Math.min(MAX_LIST_KEYS, options?.maxKeys ?? 100));
      const result = await client.getBucket({
        Bucket: credential.bucket,
        Region: credential.region,
        Prefix: fullPrefix,
        Marker: marker,
        MaxKeys: maxKeys,
        Delimiter: options?.recursive ? undefined : "/",
        Headers: {},
      });

      const entries: ListEntry[] = [];
      for (const object of result.Contents ?? []) {
        const relativeKey = stripNamespace(String(object.Key ?? ""), namespacePrefix);
        entries.push({
          key: relativeKey,
          size: numberValue(object.Size) ?? 0,
          lastModified: parseDate(object.LastModified) ?? new Date(0),
          isDirectory: false,
        });
      }
      for (const directory of result.CommonPrefixes ?? []) {
        const relativeKey = stripNamespace(String(directory.Prefix ?? ""), namespacePrefix);
        entries.push({
          key: relativeKey,
          size: 0,
          lastModified: new Date(0),
          isDirectory: true,
        });
      }
      entries.sort((left, right) => left.key.localeCompare(right.key));

      const truncated = result.IsTruncated === true || result.IsTruncated === "true";
      const rawNextMarker = result.NextMarker
        ?? (truncated ? result.Contents?.at(-1)?.Key : undefined)
        ?? (truncated ? result.CommonPrefixes?.at(-1)?.Prefix : undefined);
      return {
        entries,
        nextMarker: rawNextMarker ? stripNamespace(rawNextMarker, namespacePrefix) : undefined,
      };
    });
  }

  async deleteObject(key: string): Promise<void> {
    try {
      await this.withContext(async ({ client, credential }) => {
        await client.deleteObject({ ...this.objectIdentity(credential, key), Headers: {} });
      });
    } catch (error) {
      if (!isNotFound(error)) throw error;
    }
  }

  async deleteByPrefix(prefix: string): Promise<number> {
    let deleted = 0;
    // Always request the first remaining page. This cannot skip keys when the
    // previous page has just been deleted underneath a continuation marker.
    while (true) {
      const page = await this.withContext(async ({ client, credential }) => {
        const result = await client.getBucket({
          Bucket: credential.bucket,
          Region: credential.region,
          Prefix: this.fullKey(credential, prefix, true),
          MaxKeys: MAX_LIST_KEYS,
          Headers: {},
        });
        const keys = (result.Contents ?? [])
          .map((object) => String(object.Key ?? ""))
          .filter(Boolean);
        if (keys.length === 0) return 0;
        const outcome = await client.deleteMultipleObject({
          Bucket: credential.bucket,
          Region: credential.region,
          Objects: keys.map((Key) => ({ Key })),
          Quiet: false,
          Headers: {},
        });
        if ((outcome.Error?.length ?? 0) > 0) {
          throw new Error(`${TAG} partial deleteByPrefix failure for ${outcome.Error!.length} objects`);
        }
        return outcome.Deleted?.length ?? keys.length;
      });
      if (page === 0) return deleted;
      deleted += page;
    }
  }

  private async withContext<T>(operation: (context: SharedCosContext) => Promise<T>): Promise<T> {
    let context = await this.sharedClient.getContext();
    try {
      return await operation(context);
    } catch (error) {
      if (!isAuthError(error)) throw error;
      this.sharedClient.invalidate();
      context = await this.sharedClient.getContext();
      return operation(context);
    }
  }

  private async existsAfterUncertainCreate(key: string): Promise<boolean> {
    try {
      return await this.exists(key);
    } catch {
      return false;
    }
  }

  private async objectLength(key: string): Promise<number> {
    try {
      return await this.withContext(async ({ client, credential }) => {
        const result = await client.headObject({
          ...this.objectIdentity(credential, key),
          Headers: {},
        });
        return numberValue(headerValue(asHeaders(result.headers), "content-length")) ?? 0;
      });
    } catch (error) {
      if (isNotFound(error)) return 0;
      throw error;
    }
  }

  private objectIdentity(credential: CosCredential, key: string): Record<string, unknown> {
    return {
      Bucket: credential.bucket,
      Region: credential.region,
      Key: this.fullKey(credential, key, false),
    };
  }

  private namespacePrefix(credential: CosCredential): string {
    return this.prefixOverride ?? normalizePrefix(credential.prefix);
  }

  private fullKey(credential: CosCredential, key: string, allowEmpty: boolean): string {
    validateRelativeKey(key, allowEmpty);
    return `${this.namespacePrefix(credential)}${key}`;
  }
}

function normalizePrefix(prefix: string): string {
  if (prefix.includes("\0") || prefix.includes("\\")) {
    throw new Error(`${TAG} invalid namespace prefix`);
  }
  const normalized = prefix.replace(/^\/+|\/+$/g, "");
  if (normalized.split("/").some((segment) => segment === "..")) {
    throw new Error(`${TAG} namespace prefix traversal rejected`);
  }
  return normalized ? `${normalized}/` : "";
}

function validateRelativeKey(key: string, allowEmpty: boolean): void {
  if ((!allowEmpty && key.length === 0) || key.includes("\0") || key.includes("\\") || key.startsWith("/")) {
    throw new Error(`${TAG} invalid object key`);
  }
  if (key.split("/").some((segment) => segment === "..")) {
    throw new Error(`${TAG} object key traversal rejected`);
  }
}

function stripNamespace(key: string, prefix: string): string {
  if (!key.startsWith(prefix)) {
    throw new Error(`${TAG} COS returned an out-of-scope key`);
  }
  return key.slice(prefix.length);
}

function toBuffer(content: string | Buffer): Buffer {
  return typeof content === "string" ? Buffer.from(content, "utf8") : content;
}

function metadataHeaders(options?: PutObjectOptions): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const [rawKey, value] of Object.entries(options?.metadata ?? {})) {
    const key = rawKey.toLowerCase().replace(/[^a-z0-9-]/g, "-");
    if (key) headers[`x-cos-meta-${key}`] = value;
  }
  return headers;
}

function asHeaders(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function headerValue(headers: Record<string, unknown>, name: string): string | undefined {
  const matchingKey = Object.keys(headers).find((key) => key.toLowerCase() === name.toLowerCase());
  const value = matchingKey ? headers[matchingKey] : undefined;
  return value === undefined || value === null ? undefined : String(value);
}

function extractMetadata(headers: Record<string, unknown>): Record<string, string> | undefined {
  const metadata: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase().startsWith("x-cos-meta-") && value !== undefined) {
      metadata[key.slice("x-cos-meta-".length)] = String(value);
    }
  }
  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

function numberValue(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

function parseDate(value: unknown): Date | undefined {
  if (value === undefined) return undefined;
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? undefined : date;
}

function errorCode(error: unknown): string | undefined {
  return error && typeof error === "object" ? String((error as CosErrorLike).code ?? "") || undefined : undefined;
}

function errorStatus(error: unknown): number | undefined {
  return error && typeof error === "object" ? (error as CosErrorLike).statusCode : undefined;
}

function isNotFound(error: unknown): boolean {
  return errorStatus(error) === 404 || ["NoSuchKey", "NotFound"].includes(errorCode(error) ?? "");
}

function isAuthError(error: unknown): boolean {
  return errorStatus(error) === 401 || errorStatus(error) === 403;
}

function isConflict(error: unknown): boolean {
  return errorStatus(error) === 409 || [
    "AppendPositionError",
    "FileAlreadyExists",
    "ObjectNotAppendable",
    "PathConflict",
    "UploadConflict",
  ].includes(errorCode(error) ?? "");
}
