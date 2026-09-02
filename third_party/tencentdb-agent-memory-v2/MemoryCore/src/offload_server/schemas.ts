/**
 * Offload Server — Request validation schemas (Zod).
 */
import { z } from "zod";

/** Safe session ID: alphanumeric, underscore, hyphen, dot, colon allowed. No slashes or path traversal. Max 500 chars. */
const safeSessionId = z.string().min(1).max(500, {
  message: "sessionId must not exceed 500 characters",
}).regex(/^[a-zA-Z0-9_.\-:]+$/, {
  message: "Must only contain alphanumeric, underscore, hyphen, dot, or colon characters",
});

const ToolPairSchema = z.object({
  tool_name: z.string(),
  tool_call_id: z.string(),
  params: z.unknown(),
  result: z.unknown(),
  error: z.string().optional(),
  timestamp: z.string(),
  duration_ms: z.number().optional(),
});

/** Recent message item: user/assistant text only (no tool_call/tool_result). */
const RecentMessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string(),
});

export const IngestRequestSchema = z
  .object({
    session_id: safeSessionId,
    tool_pairs: z.array(ToolPairSchema).default([]),
    /** Current user prompt that triggers L1.5 task judgment. Must be non-empty (whitespace-only is rejected). */
    prompt: z.string().trim().min(1, { message: "prompt must not be empty or whitespace-only" }).optional(),
    /** Recent history messages (user/assistant only, no tool calls). */
    recent_messages: z.array(RecentMessageSchema).optional(),
    /**
     * Optional delayed-gate boundary. When prompt judgment was intentionally
     * deferred until tool results accumulated, Proxy places the boundary just
     * before the earliest accumulated tool call so L2 can still map them.
     */
    boundary_timestamp: z.string().datetime().optional(),
  })
  .refine(
    (data) => data.tool_pairs.length > 0 || (data.prompt && data.prompt.length > 0),
    { message: "Either tool_pairs must be non-empty or prompt must be provided" },
  );

export type IngestRequest = z.infer<typeof IngestRequestSchema>;

/**
 * Each message must have non-empty `role` and `content` fields.
 * Intentionally lenient: any non-empty string role is accepted to support
 * OpenAI / Anthropic / OpenClaw-wrapped formats without over-specifying.
 */
const CompactionMessageSchema = z
  .record(z.string(), z.unknown())
  .refine(
    (msg) => typeof msg["role"] === "string" && (msg["role"] as string).length > 0,
    { message: "Each message must have a non-empty 'role' field" },
  );

export const CompactionRequestSchema = z.object({
  session_id: safeSessionId,
  messages: z.array(CompactionMessageSchema),
  ratio: z.number().min(0).max(2),
});

export type CompactionRequest = z.infer<typeof CompactionRequestSchema>;

/** Extended compaction schema with token metadata for L3 compression. */
export const CompactionRequestSchemaV2 = z.object({
  session_id: safeSessionId,
  messages: z.array(CompactionMessageSchema),
  ratio: z.number().min(0).max(2),
  context_window: z.number().int().min(1),
  total_tokens: z.number().int().min(0),
  message_tokens: z.array(z.number()).optional(),
});

export type CompactionRequestV2 = z.infer<typeof CompactionRequestSchemaV2>;

export const MmdQuerySchema = z.object({
  session_id: safeSessionId,
  limit: z.number().int().min(1).optional(),
});

const GateTelemetrySchema = z.object({
  decision: z.string(),
  reason: z.string(),
  raw_tool_tokens: z.number().int().min(0),
  estimated_build_tokens: z.number().int().min(0),
  scheduled_build_tokens: z.number().int().min(0).optional(),
  projected_gross_saved_tokens: z.number().int().min(0),
  projected_net_saved_tokens: z.number().int(),
  projected_net_savings_ratio: z.number().optional(),
  expected_reuse_calls: z.number().int().min(1),
  prompt_tokens: z.number().int().min(0).optional(),
  prompt_only_enabled: z.boolean().optional(),
  reserved_lifecycle_tokens: z.number().int().min(0).optional(),
  projected_total_cost_tokens: z.number().int().min(0).optional(),
  active_task_prompt_budget: z.number().int().min(0).optional(),
}).passthrough();

export const MainUsageReportSchema = z.object({
  session_id: safeSessionId,
  request_id: z.string().min(1).max(500),
  timestamp: z.string().optional(),
  protocol: z.enum(["openai", "anthropic"]),
  applied: z.boolean(),
  before_estimated_tokens: z.number().int().min(0),
  forwarded_estimated_tokens: z.number().int().min(0),
  estimated_gross_saved_tokens: z.number().int().min(0),
  actual_forwarded_input_tokens: z.number().int().min(0),
  actual_output_tokens: z.number().int().min(0),
  /** Whether forwarded input usage came from the provider or request-side estimation. */
  forwarded_input_source: z.enum(["provider", "estimated"]).optional(),
  mmd_injected: z.number().int().min(0),
  gate: GateTelemetrySchema.optional(),
});

export const UsageQuerySchema = z.object({
  session_id: safeSessionId,
});
