/**
 * TDAI Gateway — Request/Response types for the HTTP API.
 */

// ============================
// Common
// ============================

export interface GatewayErrorResponse {
  error: string;
  code?: string;
}

// ============================
// /health
// ============================

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  uptime: number;
  stores: {
    vectorStore: boolean;
    embeddingService: boolean;
  };
}

// ============================
// /recall
// ============================

export interface RecallRequest {
  query: string;
  session_key: string;
  user_id?: string;
  /** Enable deterministic query decomposition and coverage-first fusion. */
  multi_hop?: boolean;
  max_queries?: number;
  l1_limit?: number;
  /** Include raw L0 evidence in addition to L1/persona/scene context. */
  include_l0?: boolean;
  l0_limit?: number;
  context_window?: number;
  context_policy?: "auto" | "always";
}

export interface RecallResponse {
  context: string;
  strategy?: string;
  memory_count?: number;
  l0_count?: number;
}

// ============================
// /capture
// ============================

export interface CaptureRequest {
  user_content: string;
  assistant_content: string;
  session_key: string;
  session_id?: string;
  user_id?: string;
  messages?: unknown[];
}

export interface CaptureResponse {
  l0_recorded: number;
  scheduler_notified: boolean;
}

// ============================
// /search/memories
// ============================

export interface MemorySearchRequest {
  query: string;
  limit?: number;
  type?: string;
  scene?: string;
  multi_hop?: boolean;
  max_queries?: number;
  hybrid_source_floor?: number;
  strategy?: "hybrid" | "keyword" | "embedding";
  hybrid_floor_policy?: "fixed" | "constrained_consensus";
}

export interface MemorySearchResponse {
  results: string;
  total: number;
  strategy: string;
}

// ============================
// /search/conversations
// ============================

export interface ConversationSearchRequest {
  query: string;
  limit?: number;
  session_key?: string;
  multi_hop?: boolean;
  max_queries?: number;
  hybrid_source_floor?: number;
  strategy?: "hybrid" | "keyword" | "embedding";
  hybrid_floor_policy?: "fixed" | "constrained_consensus";
  context_window?: number;
  context_policy?: "auto" | "always";
}

export interface ConversationSearchResponse {
  results: string;
  total: number;
  strategy?: string;
}

// ============================
// /session/end
// ============================

export interface SessionEndRequest {
  session_key: string;
  user_id?: string;
}

export interface SessionEndResponse {
  flushed: boolean;
}

// ============================
// /seed
// ============================

/**
 * Request body for `POST /seed`.
 *
 * Accepts the same input formats as the CLI `seed` command:
 * - Format A: `{ sessions: [{ sessionKey, conversations: [[...msgs]] }] }`
 * - Format B: `[{ sessionKey, conversations: [[...msgs]] }]`
 *
 * Wrapped in an envelope with optional control fields.
 */
export interface SeedRequest {
  /**
   * Seed input data — either Format A object or Format B array.
   * This is the same structure accepted by `openclaw memory-tdai seed --input`.
   */
  data: unknown;
  /** Fallback session key when input sessions lack one. */
  session_key?: string;
  /** Require each round to have both user and assistant messages. */
  strict_round_role?: boolean;
  /** Auto-fill missing timestamps (default: true). */
  auto_fill_timestamps?: boolean;
  /** Plugin config overrides (deep-merged on top of gateway memory config). */
  config_override?: Record<string, unknown>;
}

export interface SeedResponse {
  sessions_processed: number;
  rounds_processed: number;
  messages_processed: number;
  l0_recorded: number;
  duration_ms: number;
  output_dir: string;
}
