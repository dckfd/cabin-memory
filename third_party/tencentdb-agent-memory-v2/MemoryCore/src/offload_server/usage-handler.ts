/** HTTP handlers for durable offload token-ledger reporting/query. */
import type http from "node:http";
import type { StorageAdapter } from "../core/storage/adapter.js";
import { buildOffloadBasePath } from "./session-utils.js";
import { MainUsageReportSchema, UsageQuerySchema } from "./schemas.js";
import { summarizeTokenLedger, writeMainTokenRecord } from "./token-ledger.js";

type SendJson = (res: http.ServerResponse, status: number, body: unknown) => void;

export async function handleMainUsageReport(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  storage: StorageAdapter,
  requestId: string,
  parseJsonBody: <T>(req: http.IncomingMessage) => Promise<T>,
  sendJson: SendJson,
  successEnvelope: <T>(data: T, requestId: string) => unknown,
  errorEnvelope: (code: number, message: string, requestId: string) => unknown,
): Promise<void> {
  const parsed = MainUsageReportSchema.safeParse(await parseJsonBody(req));
  if (!parsed.success) {
    sendJson(res, 400, errorEnvelope(400, parsed.error.message, requestId));
    return;
  }
  const data = parsed.data;
  const basePath = buildOffloadBasePath(data.session_id);
  await writeMainTokenRecord(storage, basePath, {
    kind: "main",
    record_id: data.request_id,
    timestamp: data.timestamp ?? new Date().toISOString(),
    session_id: data.session_id,
    protocol: data.protocol,
    applied: data.applied,
    before_estimated_tokens: data.before_estimated_tokens,
    forwarded_estimated_tokens: data.forwarded_estimated_tokens,
    estimated_gross_saved_tokens: data.estimated_gross_saved_tokens,
    actual_forwarded_input_tokens: data.actual_forwarded_input_tokens,
    actual_output_tokens: data.actual_output_tokens,
    ...(data.forwarded_input_source
      ? { forwarded_input_source: data.forwarded_input_source }
      : {}),
    mmd_injected: data.mmd_injected,
    ...(data.gate ? { gate: data.gate } : {}),
  });
  const summary = await summarizeTokenLedger(storage, basePath, data.session_id);
  sendJson(res, 200, successEnvelope(summary, requestId));
}

export async function handleUsageQuery(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  storage: StorageAdapter,
  requestId: string,
  parseJsonBody: <T>(req: http.IncomingMessage) => Promise<T>,
  sendJson: SendJson,
  successEnvelope: <T>(data: T, requestId: string) => unknown,
  errorEnvelope: (code: number, message: string, requestId: string) => unknown,
): Promise<void> {
  const parsed = UsageQuerySchema.safeParse(await parseJsonBody(req));
  if (!parsed.success) {
    sendJson(res, 400, errorEnvelope(400, parsed.error.message, requestId));
    return;
  }
  const basePath = buildOffloadBasePath(parsed.data.session_id);
  const summary = await summarizeTokenLedger(storage, basePath, parsed.data.session_id);
  sendJson(res, 200, successEnvelope(summary, requestId));
}
