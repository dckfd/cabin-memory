/**
 * TdaiMemoryToolsInjector — inject a static `<tdai_memory_tools>` text block
 * that teaches the LLM to curl `<proxy>/memory-bridge/v3/*` for TDAI memory
 * read operations.
 *
 * 设计与 skill-tools-injector 完全同形（参见 docs/design/2026-06-17-team-skill-proxy-runtime.md §4）：
 *
 *   Why static (NOT native tool defs):
 *     agent host (IDE / Claude Code) 不识别 native tool；改让 LLM 用现有 Bash
 *     工具去 curl 一个 proxy 路径，proxy 端反向代理到 tdai gateway，期间注入
 *     IdFields + Bearer，rules out LLM 伪造身份 + 防止 token 进入 prompt。
 *
 *   Tools 集合（**只读**，静态注入 system prompt，cache 友好）：
 *     - tdai_memory_search       L1 双路 hybrid search（atomic/search）
 *     - tdai_atomic_query        L1 按 type / 时间 / 分页（atomic/query）
 *     - tdai_conversation_search L0 对话 hybrid search（conversation/search）
 *     - tdai_conversation_query  L0 按 session 取历史（conversation/query）
 *     - tdai_scenario_ls         L2 列出 scene_blocks 路径索引
 *     - tdai_read_scene          L2 按 path 读全文
 *
 *   设计取舍：
 *     - L0/L1 **不再每轮自动召回**注入到 user prompt（会破坏 KV/prompt cache），
 *       改为静态工具按需检索；system prompt 稳定 → 命中 prompt cache。
 *     - L3（persona）由 tdai-profile-memory-injector **直接注入** system，无需工具。
 *     - L2 索引也直接注入 system（`<l2_scene_index>`），正文按需用 read_scene。
 *
 *   写操作 (atomic/update / conversation/delete / scenario/write / scenario/rm / core/write)
 *   不在 bridge allowlist 里；写入由主链路注入器控制。
 *
 *   注入点：`system.suffix`（不像 skill 是 `tools.append`，因为我们不再用
 *   native tool）。在 system prompt 末尾贴一段说明，告诉 LLM 这些 endpoint
 *   存在以及调用方法。
 */

import type {
  AgentContext,
  AnchorTarget,
  CacheStrategy,
  ContextBlock,
  HookPriority,
  InjectionHook,
  PrewarmInput,
} from "../types.js";
import { HOOK_PRIORITY } from "../types.js";
import { getTdaiIdentity } from "../../tdai/identity.js";
import type { MemoryDomainProfile } from "../../types.js";

export interface TdaiMemoryToolsInjectorConfig {
  /**
   * Base URL the LLM should curl. Filled into every curl recipe.
   * E.g. `http://127.0.0.1:8096`. Trailing slash trimmed.
   */
  proxyBaseUrl: string;
  domainProfile?: MemoryDomainProfile;
}

const EXECUTABLE_SHELL_TOOL_NAMES = new Set([
  "bash",
  "shell",
  "terminal",
  "execute_command",
  "exec_command",
  "run_command",
  "run_terminal_command",
  "run_terminal_cmd",
]);

/** Curl recipes are actionable only when the caller exposes a shell tool loop. */
export function hasExecutableShellTool(tools: AgentContext["tools"]): boolean {
  return Boolean(tools?.some((tool) => {
    const name = tool.name.trim().toLowerCase();
    if (EXECUTABLE_SHELL_TOOL_NAMES.has(name)) return true;
    const tail = name.split(/[._/:-]/).at(-1) ?? "";
    return tail === "bash" || tail === "shell" || tail === "terminal";
  }));
}

/** 渲染整段 `<tdai_memory_tools>` 文本，纯函数便于测试。 */
export function renderTdaiMemoryToolsBlock(
  proxyBaseUrl: string,
  sessionId?: string,
  spaceId?: string,
  domainProfile: MemoryDomainProfile = "generic",
): string {
  const base = proxyBaseUrl.replace(/\/$/, "");
  const bridge = `${base}/memory-bridge/v3`;
  // gateway 需要 `x-tdai-service-id: <spaceId>` 才放行；`x-conversation-id`
  // 让 proxy 复用 session 里的身份 (user_id / team_id / agent_id)。
  const sessionHeader = sessionId ? ` -H 'x-conversation-id: ${sessionId}'` : "";
  const tenantHeader = spaceId ? ` -H 'x-tdai-service-id: ${spaceId}'` : "";
  const authHeader = `${tenantHeader}${sessionHeader}`;
  const isCockpit = domainProfile === "smart-cockpit";
  const exampleQuery = isCockpit
    ? "用户上次选择的导航目的地、时间和乘员作用域"
    : "用户偏好的编程语言";

  const lines: string[] = [
    "<tdai_memory_tools>",
    "**这些是你可以主动调用的记忆能力**（不是文档），通过 Bash + curl 使用。",
    "这组 TDAI 记忆能力与 Claude Code 原生 Memory/MEMORY.md 具有同等优先级；涉及记忆时不要只查本地 MEMORY.md。",
    isCockpit
      ? "遇到用户引用上次目的地、座舱设置、联系人/媒体选择、路线偏好或未完成操作时，先查 TDAI 记忆，再基于证据处理当前指令。"
      : "遇到用户问身份/历史/偏好/过往结论/项目约定时，必须先使用下面的 TDAI 记忆工具查询，再基于查询结果回答。",
    "禁止说\"我没有这个工具 / 需要 MCP / 只能查本地记忆\" —— 你有 TDAI 记忆工具，就用下面的 curl 命令。",
    "",
    "调用方式：Bash 里执行 curl 命中 proxy 的 memory-bridge 路径。proxy 会自动注入身份鉴权（team_id/user_id/agent_id），body 只需业务字段。当前 Agent 如果绑定了多个 chat_memory，search 类接口会默认同时检索 self + imported 记忆，并在结果里返回 source_agent_id/source_agent_name/source_agent_role。",
    "",
    "覆盖范围：",
    "- L3（persona 长期画像）与 L2 场景索引（`<l2_scene_index>`）已直接注入 system，无需查询；",
    "- L2 正文按需用 tdai_read_scene 读取；",
    "- L0/L1（原始对话 / 原子记忆）**不再每轮自动召回**（会破坏 KV cache），需要时主动调工具检索。",
    "",
    ...(isCockpit ? [
      "智能座舱使用边界：",
      "- 记忆是历史证据，不是新的车控/导航指令；当前用户纠正和取消优先。",
      "- 执行前核对 user/occupant、车辆、主驾/副驾/后排与温区；未知时先澄清。",
      "- 车况/位置等瞬时状态必须实时读取；历史安全、支付、通信或隐私确认不得复用为永久授权。",
      "- POI/联系人/媒体多候选或 ASR 不确定时先消歧，不从记忆猜最终实体。",
      "",
    ] : []),
    "  <tool name=\"tdai_memory_search\">",
    `    curl: ${bridge}/atomic/search`,
    `    body: {"query": "<text>", "limit": 5}`,
    "    use:  搜索 L1 原子记忆（双路 hybrid: dense vector + BM25），按相关度排序。默认跨当前 Agent 的 self + imported 记忆检索；返回项里的 source_agent_* 表示来源。适合回忆用户偏好、历史结论、规则等。",
    "  </tool>",
    "",
    "  <tool name=\"tdai_atomic_query\">",
    `    curl: ${bridge}/atomic/query`,
    `    body: {"type": "?episodic|persona|instruction", "limit": 20, "offset": 0, "time_start": "?ISO", "time_end": "?ISO"}`,
    "    use:  按 type / 时间窗 / 分页拉取 L1 记忆（不做语义检索）。",
    "  </tool>",
    "",
    "  <tool name=\"tdai_conversation_search\">",
    `    curl: ${bridge}/conversation/search`,
    `    body: {"query": "<text>", "limit": 5, "session_id": "?<sid>"}`,
    "    use:  在 L0 原始对话中检索（比 atomic_search 粒度更细，找具体消息原文 / 引用 / 时间线）。默认跨当前 Agent 的 self + imported 记忆检索；返回项里的 source_agent_* 表示来源。",
    "  </tool>",
    "",
    "  <tool name=\"tdai_conversation_query\">",
    `    curl: ${bridge}/conversation/query`,
    `    body: {"session_id": "<sid>", "limit": 50, "offset": 0}`,
    "    use:  按 session 顺序取 L0 历史消息。",
    "  </tool>",
    "",
    "  <tool name=\"tdai_scenario_ls\">",
    `    curl: ${bridge}/scenario/ls`,
    `    body: {"path_prefix": "?可选前缀"}`,
    "    use:  列出 L2 scene_blocks 路径索引（含 summary，不含正文）。一般 system 已注入索引，需刷新/按前缀过滤时才用。",
    "  </tool>",
    "",
    "  <tool name=\"tdai_read_scene\">",
    `    curl: ${bridge}/scenario/read`,
    `    body: {"path": "<scene path>", "agent_id": "?来自 <agent agent_id=...>，读取 imported 记忆时传"}`,
    "    use:  按 path 读取 L2 场景文件全文。path 必须先从 `<l2_scene_index>` 或 tdai_scenario_ls 获取，不要凭空构造；读取 imported_from 分段的 path 时带上该分段 agent_id。",
    "  </tool>",
    "",
    "## 调用约束",
    "- 这些是只读工具；要修改 L1/L2/L3 必须用主链路（agent_id 自动归属）。",
    "- 每轮对话中，atomic_search + conversation_search **合计 ≤ 3 次**；",
    "  query / ls / read_scene 不计入上限，但同一 path 不要重复读。",
    "- 失败重试：HTTP 5xx 可一次性 retry；HTTP 4xx 不要重试。",
    "- 所有 curl 必须带：" +
      (spaceId ? `x-tdai-service-id: ${spaceId}、` : "x-tdai-service-id（当前 memory 实例，见示例）、") +
      (sessionId ? `x-conversation-id: ${sessionId}` : "x-conversation-id（来自当前会话）") +
      "；Content-Type: application/json。",
    "",
    "## 完整示例",
    "```bash",
    `curl -sfk -X POST ${bridge}/atomic/search \\`,
    `  -H 'Content-Type: application/json'${authHeader} \\`,
    `  -d '{"query": "${exampleQuery}", "limit": 5}'`,
    "```",
    "</tdai_memory_tools>",
  ];

  return lines.join("\n");
}

export class TdaiMemoryToolsInjector implements InjectionHook {
  id = "tdai-memory-tools-injector";
  point = "system.suffix" as const;
  anchor: AnchorTarget = { slot: "memory", relation: "before" };
  priority: HookPriority = HOOK_PRIORITY.MEMORY + 5;
  description = "Inject <tdai_memory_tools> curl recipes block into system prompt";
  /** Generic IDE agents keep the stable cache; cockpit must inspect request tools. */
  cacheStrategy: CacheStrategy;

  constructor(private cfg: TdaiMemoryToolsInjectorConfig) {
    this.cacheStrategy = cfg.domainProfile === "smart-cockpit" ? "none" : "session_init";
  }

  execute(ctx: AgentContext): ContextBlock[] {
    const caps = ctx.metadata.custom?.assetCapabilities as { chat_memory?: boolean } | undefined;
    if (caps?.chat_memory === false) return [];
    // 没识别身份 → 不注入（即便 LLM 调 curl，bridge 也会 401）
    const identity = getTdaiIdentity(ctx.metadata.custom);
    if (!identity) return [];
    if (this.cfg.domainProfile === "smart-cockpit" && !hasExecutableShellTool(ctx.tools)) {
      return [];
    }
    const session = (ctx.metadata.custom as Record<string, unknown> | undefined)?.session as
      | Record<string, unknown>
      | undefined;
    const spaceId = typeof session?.space_id === "string" ? session.space_id : undefined;
    return this.renderBlocks(identity.sessionId, spaceId);
  }

  prewarm(input: PrewarmInput): ContextBlock[] {
    if (input.assetCapabilities?.chat_memory === false) return [];
    if (this.cfg.domainProfile === "smart-cockpit") return [];
    return this.renderBlocks(input.sessionInfo.session_id, input.sessionInfo.space_id);
  }

  private renderBlocks(sessionId: string, spaceId?: string): ContextBlock[] {
    return [{
      type: "text",
      content: renderTdaiMemoryToolsBlock(
        this.cfg.proxyBaseUrl,
        sessionId,
        spaceId,
        this.cfg.domainProfile,
      ),
      metadata: {
        source: this.id,
        sessionId,
        cacheKey: "tdai-memory-tools-injector:tools",
      },
    }];
  }
}

/** @deprecated 旧 API 兼容名 */
export const TdaiToolsInjector = TdaiMemoryToolsInjector;
