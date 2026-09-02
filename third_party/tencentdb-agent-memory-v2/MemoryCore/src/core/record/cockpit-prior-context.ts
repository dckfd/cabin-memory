/**
 * Bounded, isolation-safe structured context for cockpit L1 construction.
 *
 * This is construction context, not recall: it only exposes already-complete
 * cockpit records so the extractor can reuse exact state/episode identities
 * and name explicit transition edges. It does not alter TencentDB's original
 * storage, session-scoped deduplication, vector recall, or ranking semantics.
 */

import type { IMemoryStore, L1RecordRow } from "../store/types.js";
import type { Logger } from "../types.js";
import { COCKPIT_STATE_SCHEMA_VERSION } from "./cockpit-memory-contract.js";

const TAG = "[memory-tdai][cockpit-prior-context]";
export const DEFAULT_COCKPIT_PRIOR_CONTEXT_LIMIT = 24;
const MAX_CONTEXT_CONTENT_CHARS = 320;

const IDENTITY_METADATA_KEYS = [
  "schema_version",
  "domain",
  "slot",
  "value",
  "target",
  "unit",
  "relation",
  "state_key",
  "episode_key",
  "action_status",
  "subject",
  "occupant_scope",
  "vehicle_scope",
  "seat_zone",
  "valid_from",
  "valid_to",
  "activity_start_time",
  "activity_end_time",
  "condition",
  "trigger",
  "constraint_target",
  "state_qualifier",
  "record_kind",
  "mentioned_at",
  "timezone",
  "time_precision",
  "temporal_status",
  "supersedes",
  "source_message_ids",
  "source_session_id",
  "source_session_ids",
] as const;

export interface CockpitPriorMemoryContext {
  record_id: string;
  session_id: string;
  type: string;
  scene_name: string;
  content: string;
  updated_time: string;
  metadata: Record<string, unknown>;
}

function parseMetadata(value: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined;
  } catch {
    return undefined;
  }
}

function qualityIsComplete(metadata: Record<string, unknown>): boolean {
  const quality = metadata.construction_quality;
  return Boolean(
    quality
      && typeof quality === "object"
      && !Array.isArray(quality)
      && (quality as Record<string, unknown>).status === "complete",
  );
}

function compactMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  const compact: Record<string, unknown> = {};
  for (const key of IDENTITY_METADATA_KEYS) {
    if (metadata[key] !== undefined) compact[key] = metadata[key];
  }
  return compact;
}

function isExactIsolationMatch(
  row: L1RecordRow,
  scope: { teamId: string; userId: string; agentId: string; taskId: string },
): boolean {
  return row.team_id === scope.teamId
    && row.user_id === scope.userId
    && row.agent_id === scope.agentId
    && row.task_id === scope.taskId;
}

/**
 * Read recent authoritative structured records under an exact four-dimension
 * scope. If the caller cannot prove that scope, no context is returned.
 */
export async function loadCockpitPriorMemoryContext(params: {
  vectorStore?: IMemoryStore;
  teamId?: string;
  userId?: string;
  agentId?: string;
  taskId?: string;
  currentSessionId?: string;
  /**
   * Source-event upper bound for out-of-order recovery. Records mentioned
   * after the current batch are future evidence and must never be exposed as
   * "prior" construction context merely because they were persisted first.
   */
  currentEventTimeMs?: number;
  limit?: number;
  logger?: Logger;
}): Promise<CockpitPriorMemoryContext[]> {
  const {
    vectorStore,
    teamId,
    userId,
    agentId,
    taskId,
    currentSessionId,
    currentEventTimeMs,
    logger,
  } = params;
  const query = vectorStore?.queryL1Paginated;
  if (!query || !teamId || !userId || !agentId || !taskId) {
    logger?.debug?.(`${TAG} skipped: exact team/user/agent/task scope unavailable`);
    return [];
  }

  const limit = Math.max(1, Math.min(params.limit ?? DEFAULT_COCKPIT_PRIOR_CONTEXT_LIMIT, 48));
  try {
    // JSON metadata cannot be filtered portably in both SQLite and TCVDB, so
    // page through a bounded window and enforce schema/quality/event-time
    // locally. Paging matters during repair: the newest persisted rows may all
    // belong to sessions that are later in event time than the missing batch.
    const pageSize = Math.min(limit * 3, 144);
    const maxRowsToScan = Math.min(Math.max(limit * 10, pageSize), 480);
    const scope = { teamId, userId, agentId, taskId };
    const context: CockpitPriorMemoryContext[] = [];
    let offset = 0;
    let filteredFutureRecords = 0;
    while (context.length < limit && offset < maxRowsToScan) {
      const result = await query.call(vectorStore, {
        teamId,
        userId,
        agentId,
        taskId,
        limit: Math.min(pageSize, maxRowsToScan - offset),
        offset,
      });
      if (result.rows.length === 0) break;

      for (const row of result.rows) {
        if (row.session_id === currentSessionId || !isExactIsolationMatch(row, scope)) continue;
        const metadata = parseMetadata(row.metadata_json);
        if (!metadata
          || metadata.schema_version !== COCKPIT_STATE_SCHEMA_VERSION
          || !qualityIsComplete(metadata)) continue;

        if (Number.isFinite(currentEventTimeMs)) {
          const rawMentionedAt = metadata.mentioned_at;
          const mentionedAtMs = typeof rawMentionedAt === "string" || typeof rawMentionedAt === "number"
            ? new Date(rawMentionedAt).getTime()
            : new Date(row.timestamp_str).getTime();
          // Unparseable event time cannot prove that this is prior evidence.
          // Exclude it during a bounded recovery instead of permitting a
          // future-state leak through persistence order.
          if (!Number.isFinite(mentionedAtMs) || mentionedAtMs > currentEventTimeMs!) {
            filteredFutureRecords++;
            continue;
          }
        }

        context.push({
          record_id: row.record_id,
          session_id: row.session_id,
          type: row.type,
          scene_name: row.scene_name,
          content: row.content.slice(0, MAX_CONTEXT_CONTENT_CHARS),
          updated_time: row.updated_time,
          metadata: compactMetadata(metadata),
        });
        if (context.length >= limit) break;
      }

      offset += result.rows.length;
      if (offset >= result.total || result.rows.length < pageSize) break;
    }
    logger?.debug?.(
      `${TAG} loaded ${context.length} authoritative record(s)`
      + ` (scanned=${offset}, future_or_unproven_filtered=${filteredFutureRecords})`,
    );
    return context;
  } catch (error) {
    logger?.warn?.(`${TAG} unavailable; continuing without prior context: ${error instanceof Error ? error.message : String(error)}`);
    return [];
  }
}
