/**
 * Explicit, operator-only recovery for LocalStateBackend's volatile task queue.
 *
 * This module deliberately does not touch L0 or pipeline checkpoints.  It only
 * recreates L1 work items after an audited process restart.  The normal L1
 * cursor makes completed sessions cheap no-ops and lets unfinished sessions
 * continue from their durable checkpoint.
 */

import { createHash } from "node:crypto";
import type { IStateBackend, TaskPayload } from "../core/state/types.js";

const MAX_RECOVERY_TASKS = 5_000;
const MAX_LEDGER_ENTRIES = 64;
const RECOVERY_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const CONTROL_CHAR_RE = /[\u0000-\u001f\u007f-\u009f]/;

export interface PipelineRecoveryTaskInput {
  session_id: string;
  team_id: string;
  agent_id: string;
}

export interface PipelineRecoveryRequest {
  recovery_id: string;
  instance_id: string;
  expected_task_count: number;
  tasks: PipelineRecoveryTaskInput[];
}

export type PipelineRecoveryParseResult =
  | { ok: true; value: PipelineRecoveryRequest; digest: string }
  | { ok: false; status: 400 | 403; message: string };

export type PipelineRecoveryEnqueueResult =
  | {
      ok: true;
      recovery_id: string;
      expected_task_count: number;
      enqueued_count: number;
      already_accepted_count: number;
      idempotent_replay: boolean;
      digest: string;
    }
  | { ok: false; status: 409; message: string };

interface RecoveryLedgerEntry {
  digest: string;
  acceptedTaskIds: Set<string>;
  total: number;
}

/**
 * Process-local idempotency is intentional: if the process restarts, the local
 * queue is lost too, so the same sealed request must be accepted again.
 */
const recoveryLedger = new Map<string, RecoveryLedgerEntry>();

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeIdentifier(value: unknown, maxLength = 256): value is string {
  return typeof value === "string"
    && value.length > 0
    && value.length <= maxLength
    && value.trim() === value
    && !CONTROL_CHAR_RE.test(value);
}

function hasOnlyKeys(value: Record<string, unknown>, allowed: readonly string[]): boolean {
  const allowedSet = new Set(allowed);
  return Object.keys(value).every((key) => allowedSet.has(key));
}

/** Validate and canonically seal an operator recovery request. */
export function parsePipelineRecoveryRequest(
  body: unknown,
  authenticatedInstanceId: string,
): PipelineRecoveryParseResult {
  if (!isObject(body)) return { ok: false, status: 400, message: "Request body must be an object" };
  if (!hasOnlyKeys(body, ["recovery_id", "instance_id", "expected_task_count", "tasks"])) {
    return { ok: false, status: 400, message: "Request body contains unsupported fields" };
  }

  if (typeof body.recovery_id !== "string" || !RECOVERY_ID_RE.test(body.recovery_id)) {
    return { ok: false, status: 400, message: "Invalid recovery_id" };
  }
  if (!isSafeIdentifier(body.instance_id)) {
    return { ok: false, status: 400, message: "Invalid instance_id" };
  }
  if (body.instance_id !== authenticatedInstanceId) {
    return { ok: false, status: 403, message: "instance_id must match x-tdai-service-id" };
  }
  if (!Number.isInteger(body.expected_task_count)
      || (body.expected_task_count as number) < 1
      || (body.expected_task_count as number) > MAX_RECOVERY_TASKS) {
    return { ok: false, status: 400, message: `expected_task_count must be an integer from 1 to ${MAX_RECOVERY_TASKS}` };
  }
  if (!Array.isArray(body.tasks)
      || body.tasks.length !== body.expected_task_count
      || body.tasks.length > MAX_RECOVERY_TASKS) {
    return { ok: false, status: 400, message: "tasks length must exactly match expected_task_count" };
  }

  const tasks: PipelineRecoveryTaskInput[] = [];
  const identities = new Set<string>();
  for (const rawTask of body.tasks) {
    if (!isObject(rawTask) || !hasOnlyKeys(rawTask, ["session_id", "team_id", "agent_id"])) {
      return { ok: false, status: 400, message: "Each task must contain only session_id, team_id, and agent_id" };
    }
    if (!isSafeIdentifier(rawTask.session_id)
        || !isSafeIdentifier(rawTask.team_id)
        || !isSafeIdentifier(rawTask.agent_id)) {
      return { ok: false, status: 400, message: "Task identifiers must be non-empty, bounded, and contain no control characters" };
    }
    const task = {
      session_id: rawTask.session_id,
      team_id: rawTask.team_id,
      agent_id: rawTask.agent_id,
    };
    const identity = `${task.team_id}\u0000${task.agent_id}\u0000${task.session_id}`;
    if (identities.has(identity)) {
      return { ok: false, status: 400, message: "Recovery request contains duplicate scoped sessions" };
    }
    identities.add(identity);
    tasks.push(task);
  }

  const value: PipelineRecoveryRequest = {
    recovery_id: body.recovery_id,
    instance_id: body.instance_id,
    expected_task_count: body.expected_task_count as number,
    tasks,
  };
  const digest = createHash("sha256").update(JSON.stringify(value)).digest("hex");
  return { ok: true, value, digest };
}

function recoveryTaskId(request: PipelineRecoveryRequest, task: PipelineRecoveryTaskInput): string {
  const digest = createHash("sha256")
    .update(request.instance_id)
    .update("\u0000")
    .update(request.recovery_id)
    .update("\u0000")
    .update(task.team_id)
    .update("\u0000")
    .update(task.agent_id)
    .update("\u0000")
    .update(task.session_id)
    .digest("hex")
    .slice(0, 32);
  return `recovery-l1-${digest}`;
}

function rememberLedgerEntry(key: string, entry: RecoveryLedgerEntry): void {
  recoveryLedger.set(key, entry);
  while (recoveryLedger.size > MAX_LEDGER_ENTRIES) {
    const oldest = recoveryLedger.keys().next().value as string | undefined;
    if (!oldest) break;
    recoveryLedger.delete(oldest);
  }
}

/**
 * Enqueue the sealed L1 task set without writing L0. Reusing a recovery_id with
 * different content fails closed; an identical retry is idempotent while this
 * process (and therefore its volatile queue) remains alive.
 */
export async function enqueuePipelineRecovery(
  backend: IStateBackend,
  request: PipelineRecoveryRequest,
  digest: string,
  nowMs = Date.now(),
): Promise<PipelineRecoveryEnqueueResult> {
  const ledgerKey = `${request.instance_id}\u0000${request.recovery_id}`;
  let ledger = recoveryLedger.get(ledgerKey);
  if (ledger && ledger.digest !== digest) {
    return { ok: false, status: 409, message: "recovery_id was already used with different content" };
  }
  if (!ledger) {
    ledger = { digest, acceptedTaskIds: new Set(), total: request.expected_task_count };
    rememberLedgerEntry(ledgerKey, ledger);
  }

  const alreadyAcceptedCount = ledger.acceptedTaskIds.size;
  let enqueuedCount = 0;
  for (const [index, input] of request.tasks.entries()) {
    const id = recoveryTaskId(request, input);
    if (ledger.acceptedTaskIds.has(id)) continue;

    const task: TaskPayload = {
      id,
      type: "L1",
      instanceId: request.instance_id,
      sessionId: input.session_id,
      teamId: input.team_id,
      agentId: input.agent_id,
      priority: 0,
      data: {
        instanceId: request.instance_id,
        teamId: input.team_id,
        agentId: input.agent_id,
        triggeredBy: "pipeline_recovery",
        recoveryId: request.recovery_id,
      },
      createdAt: nowMs + index,
    };
    await backend.enqueueTask(task);
    ledger.acceptedTaskIds.add(id);
    enqueuedCount++;
  }

  return {
    ok: true,
    recovery_id: request.recovery_id,
    expected_task_count: request.expected_task_count,
    enqueued_count: enqueuedCount,
    already_accepted_count: alreadyAcceptedCount,
    idempotent_replay: enqueuedCount === 0 && ledger.acceptedTaskIds.size === ledger.total,
    digest,
  };
}
