/**
 * Durable standalone short-term-memory bridge.
 *
 * The OpenClaw plugin already owns a rich event-driven Offload pipeline.  The
 * Gateway cannot rely on those lifecycle hooks, so this bridge exposes the
 * same essential invariants through explicit calls:
 *
 * - deterministic Inline / Offload decisions;
 * - lossless, content-addressed raw evidence;
 * - optional LLM summaries with a safe extractive fallback;
 * - idempotent event processing;
 * - working context, Mermaid task canvas, and atomic checkpoints;
 * - restart recovery and node_id drill-down with SHA-256 verification.
 */
import { createHash } from "node:crypto";
import { constants as fsConstants } from "node:fs";
import { access, appendFile, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { fastEstimateTokens } from "./fast-token-estimate.js";
import { callLlm, type LlmCallerConfig } from "./local-llm/llm-caller.js";
import type { PluginLogger } from "./types.js";

export interface StandaloneOffloadThresholds {
  chars: number;
  lines: number;
  errorChars: number;
}

export interface OffloadToolCallOptions {
  eventId?: string;
  isError?: boolean;
  forceOffload?: boolean;
  recentContext?: string;
}

export interface OffloadDecision {
  status: "inline" | "offloaded";
  eventId: string;
  idempotent: boolean;
  nodeId: string | null;
  summary: string;
  inlineContent?: string;
  rawChars: number;
  summaryChars: number;
  tokensSaved: number;
  compression: "none" | "llm" | "extractive_fallback";
  fallbackReason?: string;
}

export interface OffloadPromptContext {
  mmdContent: string;
  workingContext: string;
  canvas: string;
  tokensSaved: number;
  offloadedCount: number;
}

export interface OffloadStatus {
  sessionKey: string;
  inlineCount: number;
  offloadedCount: number;
  totalTokensSaved: number;
  workingContextChars: number;
  canvasChars: number;
  dataDir: string;
  checkpointExists: boolean;
}

export interface DrillDownResult {
  nodeId: string;
  eventId: string;
  toolName: string;
  content: string;
  sha256: string;
  verified: boolean;
}

interface StepRecord {
  version: 1;
  event_id: string;
  node_id: string | null;
  tool_name: string;
  status: "inline" | "offloaded";
  summary: string;
  raw_chars: number;
  summary_chars: number;
  tokens_saved: number;
  compression: OffloadDecision["compression"];
  ref_file?: string;
  sha256?: string;
  created_at: string;
}

interface NodeCheckpoint {
  eventId: string;
  toolName: string;
  refFile: string;
  sha256: string;
  summary: string;
}

interface SessionCheckpoint {
  version: 1;
  sessionKey: string;
  updatedAt: string;
  inlineCount: number;
  offloadedCount: number;
  totalTokensSaved: number;
  events: Record<string, { status: "inline" | "offloaded"; nodeId: string | null; inputSha?: string }>;
  nodes: Record<string, NodeCheckpoint>;
}

interface SessionPaths {
  dir: string;
  refs: string;
  task: string;
  steps: string;
  workingContext: string;
  canvas: string;
  checkpoint: string;
}

export interface OffloadBridgeOptions {
  dataDir?: string;
  llmConfig?: LlmCallerConfig;
  logger?: PluginLogger;
  thresholds?: Partial<StandaloneOffloadThresholds>;
  /** Test seam; production uses llmConfig through callLlm. */
  summarizer?: (input: {
    toolName: string;
    toolResult: string;
    recentContext: string;
  }) => Promise<string>;
}

const DEFAULT_THRESHOLDS: StandaloneOffloadThresholds = {
  chars: 4_000,
  lines: 80,
  errorChars: 800,
};

const EMPTY_CHECKPOINT = (sessionKey: string): SessionCheckpoint => ({
  version: 1,
  sessionKey,
  updatedAt: new Date(0).toISOString(),
  inlineCount: 0,
  offloadedCount: 0,
  totalTokensSaved: 0,
  events: {},
  nodes: {},
});

export class StandaloneOffloadBridge {
  private readonly root: string;
  private readonly logger?: PluginLogger;
  private readonly thresholds: StandaloneOffloadThresholds;
  private readonly summarizer?: OffloadBridgeOptions["summarizer"];
  private readonly llmConfig?: LlmCallerConfig;
  private readonly locks = new Map<string, Promise<unknown>>();

  constructor(options: OffloadBridgeOptions = {}) {
    this.root = path.resolve(options.dataDir ?? path.join(process.cwd(), "data", "short-term"));
    this.logger = options.logger;
    this.thresholds = { ...DEFAULT_THRESHOLDS, ...(options.thresholds ?? {}) };
    this.summarizer = options.summarizer;
    this.llmConfig = options.llmConfig;
  }

  async initialize(): Promise<void> {
    await mkdir(this.root, { recursive: true });
  }

  async toolCall(
    sessionKey: string,
    toolName: string,
    toolResult: string,
    options: OffloadToolCallOptions = {},
  ): Promise<OffloadDecision> {
    validateInput(sessionKey, "sessionKey");
    validateInput(toolName, "toolName");
    if (typeof toolResult !== "string") throw new TypeError("toolResult must be a string");

    return this.withSessionLock(sessionKey, async () => {
      const paths = this.paths(sessionKey);
      await this.ensureSession(paths, sessionKey);
      const checkpoint = await this.readCheckpoint(paths, sessionKey);
      const eventId = options.eventId?.trim() || stableId(`${toolName}\0${toolResult}`, "evt");
      const existing = checkpoint.events[eventId];
      if (existing) {
        const inputSha = sha256(`${toolName}\0${toolResult}`);
        if (existing.inputSha && existing.inputSha !== inputSha) {
          throw new Error(`event_id payload mismatch: ${eventId}`);
        }
        const step = await this.findStep(paths, eventId);
        return {
          status: existing.status,
          eventId,
          idempotent: true,
          nodeId: existing.nodeId,
          summary: step?.summary ?? checkpoint.nodes[existing.nodeId ?? ""]?.summary ?? "",
          ...(existing.status === "inline" ? { inlineContent: toolResult } : {}),
          rawChars: step?.raw_chars ?? toolResult.length,
          summaryChars: step?.summary_chars ?? 0,
          tokensSaved: step?.tokens_saved ?? 0,
          compression: step?.compression ?? (existing.status === "inline" ? "none" : "extractive_fallback"),
        };
      }

      const lineCount = toolResult.length === 0 ? 0 : toolResult.split(/\r?\n/).length;
      const shouldOffload = options.forceOffload === true
        || toolResult.length >= this.thresholds.chars
        || lineCount >= this.thresholds.lines
        || (options.isError === true && toolResult.length >= this.thresholds.errorChars);

      if (!shouldOffload) {
        const summary = compactInline(toolResult);
        const step: StepRecord = {
          version: 1,
          event_id: eventId,
          node_id: null,
          tool_name: toolName,
          status: "inline",
          summary,
          raw_chars: toolResult.length,
          summary_chars: summary.length,
          tokens_saved: 0,
          compression: "none",
          created_at: new Date().toISOString(),
        };
        checkpoint.inlineCount += 1;
        checkpoint.events[eventId] = {
          status: "inline",
          nodeId: null,
          inputSha: sha256(`${toolName}\0${toolResult}`),
        };
        await this.commitStep(paths, checkpoint, step);
        return {
          status: "inline",
          eventId,
          idempotent: false,
          nodeId: null,
          summary,
          inlineContent: toolResult,
          rawChars: toolResult.length,
          summaryChars: summary.length,
          tokensSaved: 0,
          compression: "none",
        };
      }

      const digest = sha256(toolResult);
      const nodeId = `node-${sha256(`${eventId}\0${digest}`).slice(0, 16)}`;
      const refFile = `ref-${digest}.txt`;
      await writeOnce(path.join(paths.refs, refFile), toolResult);

      let compression: OffloadDecision["compression"] = "llm";
      let fallbackReason: string | undefined;
      let summary: string;
      try {
        summary = sanitizeSummary(await this.summarize(toolName, toolResult, options.recentContext ?? ""));
        if (!summary) throw new Error("summarizer returned empty output");
      } catch (error) {
        compression = "extractive_fallback";
        fallbackReason = error instanceof Error ? error.message : String(error);
        summary = extractiveSummary(toolResult);
        this.logger?.warn?.(`[offload-bridge] summary fallback for ${eventId}: ${fallbackReason}`);
      }

      summary = addEvidenceGuard(summary, toolResult);
      const tokensSaved = Math.max(0, fastEstimateTokens(toolResult) - fastEstimateTokens(summary));
      const step: StepRecord = {
        version: 1,
        event_id: eventId,
        node_id: nodeId,
        tool_name: toolName,
        status: "offloaded",
        summary,
        raw_chars: toolResult.length,
        summary_chars: summary.length,
        tokens_saved: tokensSaved,
        compression,
        ref_file: refFile,
        sha256: digest,
        created_at: new Date().toISOString(),
      };
      checkpoint.offloadedCount += 1;
      checkpoint.totalTokensSaved += tokensSaved;
      checkpoint.events[eventId] = {
        status: "offloaded",
        nodeId,
        inputSha: sha256(`${toolName}\0${toolResult}`),
      };
      checkpoint.nodes[nodeId] = { eventId, toolName, refFile, sha256: digest, summary };
      await this.commitStep(paths, checkpoint, step);

      return {
        status: "offloaded",
        eventId,
        idempotent: false,
        nodeId,
        summary,
        rawChars: toolResult.length,
        summaryChars: summary.length,
        tokensSaved,
        compression,
        ...(fallbackReason ? { fallbackReason } : {}),
      };
    });
  }

  async beforePrompt(sessionKey: string): Promise<OffloadPromptContext> {
    validateInput(sessionKey, "sessionKey");
    return this.withSessionLock(sessionKey, async () => {
      const paths = this.paths(sessionKey);
      await this.ensureSession(paths, sessionKey);
      const checkpoint = await this.readCheckpoint(paths, sessionKey);
      const workingContext = await readOptional(paths.workingContext);
      const canvas = await readOptional(paths.canvas);
      return {
        // Backward-compatible field used by the original three-route bridge.
        mmdContent: canvas,
        workingContext,
        canvas,
        tokensSaved: checkpoint.totalTokensSaved,
        offloadedCount: checkpoint.offloadedCount,
      };
    });
  }

  async getStatus(sessionKey: string): Promise<OffloadStatus> {
    validateInput(sessionKey, "sessionKey");
    return this.withSessionLock(sessionKey, async () => {
      const paths = this.paths(sessionKey);
      await this.ensureSession(paths, sessionKey);
      const checkpoint = await this.readCheckpoint(paths, sessionKey);
      const workingContext = await readOptional(paths.workingContext);
      const canvas = await readOptional(paths.canvas);
      return {
        sessionKey,
        inlineCount: checkpoint.inlineCount,
        offloadedCount: checkpoint.offloadedCount,
        totalTokensSaved: checkpoint.totalTokensSaved,
        workingContextChars: workingContext.length,
        canvasChars: canvas.length,
        dataDir: paths.dir,
        checkpointExists: await exists(paths.checkpoint),
      };
    });
  }

  async drillDown(sessionKey: string, nodeId: string): Promise<DrillDownResult> {
    validateInput(sessionKey, "sessionKey");
    validateInput(nodeId, "nodeId");
    return this.withSessionLock(sessionKey, async () => {
      const paths = this.paths(sessionKey);
      const checkpoint = await this.readCheckpoint(paths, sessionKey);
      const node = checkpoint.nodes[nodeId];
      if (!node) throw new Error(`Unknown node_id: ${nodeId}`);
      const content = await readFile(path.join(paths.refs, node.refFile), "utf8");
      const actual = sha256(content);
      return {
        nodeId,
        eventId: node.eventId,
        toolName: node.toolName,
        content,
        sha256: actual,
        verified: actual === node.sha256,
      };
    });
  }

  private async summarize(toolName: string, toolResult: string, recentContext: string): Promise<string> {
    if (this.summarizer) return this.summarizer({ toolName, toolResult, recentContext });
    if (!this.llmConfig?.apiKey) throw new Error("LLM summarizer is not configured");
    return callLlm(this.llmConfig, {
      label: "standalone-offload",
      systemPrompt:
        "You compress tool output for an autonomous agent. Return only a concise factual summary. " +
        "Preserve exact errors, paths, identifiers, counts, commands, decisions, and next actions. " +
        "Do not invent facts. Keep the summary under 1200 characters; raw evidence remains available by node_id.",
      userPrompt:
        `Tool: ${toolName}\nRecent task context:\n${recentContext.slice(-2_000)}\n\n` +
        `Tool output:\n${toolResult.slice(0, 12_000)}`,
      temperature: 0.1,
    }, this.logger);
  }

  private paths(sessionKey: string): SessionPaths {
    const readable = sessionKey.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 48) || "session";
    const dir = path.join(this.root, `${readable}-${sha256(sessionKey).slice(0, 12)}`);
    return {
      dir,
      refs: path.join(dir, "refs"),
      task: path.join(dir, "task.json"),
      steps: path.join(dir, "steps.jsonl"),
      workingContext: path.join(dir, "working_context.md"),
      canvas: path.join(dir, "canvas.mmd"),
      checkpoint: path.join(dir, "checkpoint.json"),
    };
  }

  private async ensureSession(paths: SessionPaths, sessionKey: string): Promise<void> {
    await mkdir(paths.refs, { recursive: true });
    if (!(await exists(paths.task))) {
      await atomicWrite(paths.task, JSON.stringify({ version: 1, sessionKey, createdAt: new Date().toISOString() }, null, 2));
    }
    if (!(await exists(paths.checkpoint))) {
      await atomicWrite(paths.checkpoint, JSON.stringify(EMPTY_CHECKPOINT(sessionKey), null, 2));
    }
    if (!(await exists(paths.workingContext))) await atomicWrite(paths.workingContext, "# Working Context\n\nNo offloaded evidence yet.\n");
    if (!(await exists(paths.canvas))) await atomicWrite(paths.canvas, "flowchart TD\n  start([Task started])\n");
  }

  private async readCheckpoint(paths: SessionPaths, sessionKey: string): Promise<SessionCheckpoint> {
    try {
      const parsed = JSON.parse(await readFile(paths.checkpoint, "utf8")) as SessionCheckpoint;
      if (parsed.version !== 1 || parsed.sessionKey !== sessionKey) throw new Error("checkpoint identity mismatch");
      parsed.events ??= {};
      parsed.nodes ??= {};
      return parsed;
    } catch (error) {
      if (!(await exists(paths.checkpoint))) return EMPTY_CHECKPOINT(sessionKey);
      throw new Error(`Invalid offload checkpoint: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private async commitStep(paths: SessionPaths, checkpoint: SessionCheckpoint, step: StepRecord): Promise<void> {
    await appendFile(paths.steps, `${JSON.stringify(step)}\n`, "utf8");
    checkpoint.updatedAt = new Date().toISOString();
    await this.rebuildDerivedFiles(paths, checkpoint);
    await atomicWrite(paths.checkpoint, JSON.stringify(checkpoint, null, 2));
  }

  private async rebuildDerivedFiles(paths: SessionPaths, checkpoint: SessionCheckpoint): Promise<void> {
    const nodes = Object.entries(checkpoint.nodes).slice(-20);
    const context = nodes.length === 0
      ? "# Working Context\n\nNo offloaded evidence yet.\n"
      : [
          "# Working Context",
          "",
          "Summaries are lossy. Use `node_id` drill-down whenever exact evidence is required.",
          "",
          ...nodes.map(([nodeId, node]) => `## ${nodeId} — ${node.toolName}\n${node.summary}`),
          "",
        ].join("\n");
    await atomicWrite(paths.workingContext, context);

    const lines = ["flowchart TD", "  start([Task started])"];
    let previous = "start";
    nodes.forEach(([nodeId, node], index) => {
      const graphId = `n${index + 1}_${nodeId.replace(/[^a-zA-Z0-9_]/g, "_")}`;
      const label = `${node.toolName}: ${node.summary}`.replace(/[\[\]{}()"\n\r]/g, " ").slice(0, 100);
      lines.push(`  ${graphId}["${label}"]`);
      lines.push(`  ${previous} --> ${graphId}`);
      previous = graphId;
    });
    await atomicWrite(paths.canvas, `${lines.join("\n")}\n`);
  }

  private async findStep(paths: SessionPaths, eventId: string): Promise<StepRecord | null> {
    const content = await readOptional(paths.steps);
    for (const line of content.split("\n")) {
      if (!line.trim()) continue;
      try {
        const step = JSON.parse(line) as StepRecord;
        if (step.event_id === eventId) return step;
      } catch {
        // A corrupt historical line must not prevent recovery from checkpoint.
      }
    }
    return null;
  }

  private withSessionLock<T>(sessionKey: string, fn: () => Promise<T>): Promise<T> {
    const previous = this.locks.get(sessionKey) ?? Promise.resolve();
    const next = previous.then(fn, fn);
    const gate = next.then(() => undefined, () => undefined);
    this.locks.set(sessionKey, gate);
    return next.finally(() => {
      if (this.locks.get(sessionKey) === gate) this.locks.delete(sessionKey);
    });
  }
}

export async function initOffloadBridge(options: OffloadBridgeOptions = {}): Promise<StandaloneOffloadBridge> {
  const bridge = new StandaloneOffloadBridge(options);
  await bridge.initialize();
  return bridge;
}

function validateInput(value: string, field: string): void {
  if (typeof value !== "string" || value.trim().length === 0) throw new Error(`${field} is required`);
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function stableId(value: string, prefix: string): string {
  return `${prefix}-${sha256(value).slice(0, 24)}`;
}

function compactInline(value: string): string {
  return value.replace(/\s+/g, " ").trim().slice(0, 500);
}

function sanitizeSummary(value: string): string {
  return value.trim().replace(/^```(?:markdown|text)?\s*/i, "").replace(/```$/, "").trim().slice(0, 1_500);
}

function extractiveSummary(value: string): string {
  const lines = value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const critical = criticalEvidence(lines);
  const head = lines.slice(0, 5);
  const tail = lines.slice(-5);
  return [...new Set([...head, ...critical, ...tail])].join("\n").slice(0, 1_500) || "(empty tool output)";
}

function criticalEvidence(lines: string[]): string[] {
  const important = /\b(error|exception|failed|failure|fatal|warning|next action|sha-?256|exit code|status|path|command)\b|(?:^|\s)(?:\/[^\s]+|[A-Za-z]:\\[^\s]+)/i;
  return lines.filter((line) => important.test(line)).slice(0, 8).map((line) => line.slice(0, 320));
}

function addEvidenceGuard(summary: string, raw: string): string {
  const normalized = summary.toLowerCase().replace(/\s+/g, " ");
  const missing = criticalEvidence(raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean))
    .filter((line) => !normalized.includes(line.toLowerCase().replace(/\s+/g, " ")))
    .slice(0, 5);
  if (missing.length === 0) return summary;
  return `${summary}\n\nCritical evidence:\n${missing.map((line) => `- ${line}`).join("\n")}`.slice(0, 2_000);
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await access(filePath, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function readOptional(filePath: string): Promise<string> {
  try {
    return await readFile(filePath, "utf8");
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return "";
    throw error;
  }
}

async function writeOnce(filePath: string, content: string): Promise<void> {
  try {
    await writeFile(filePath, content, { encoding: "utf8", flag: "wx" });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    const existing = await readFile(filePath, "utf8");
    if (sha256(existing) !== sha256(content)) throw new Error(`Content-addressed ref collision: ${path.basename(filePath)}`);
  }
}

async function atomicWrite(filePath: string, content: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temp = `${filePath}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  await writeFile(temp, content, "utf8");
  await rename(temp, filePath);
}
