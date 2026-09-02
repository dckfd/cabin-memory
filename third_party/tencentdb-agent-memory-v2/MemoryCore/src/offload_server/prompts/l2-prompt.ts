/**
 * L2 MMD Generation Prompt — generates/updates Mermaid flowcharts.
 */
import type { OffloadEntry } from "../types.js";

export {
  L2_SYSTEM_PROMPT,
  buildL2SystemPrompt,
} from "../../offload/prompts/system-prompts.js";

/**
 * Build the L2 user prompt for MMD generation.
 */
export function buildL2UserPrompt(opts: {
  existingMmd: string | null;
  entries: OffloadEntry[];
  recentHistory?: string | null;
  currentTurn?: string | null;
  taskLabel: string;
  mmdPrefix: string;
  charCount: number;
}): string {
  const { existingMmd, entries, recentHistory, currentTurn, taskLabel, mmdPrefix, charCount } = opts;
  const parts: string[] = [];

  // History section
  if (recentHistory) {
    parts.push(`## 近期对话历史：\n${recentHistory}`);
  } else {
    parts.push("## 近期对话历史：\n(无可用历史)");
  }

  if (currentTurn) {
    parts.push(`\n## 当前最新一轮：\n${currentTurn}`);
  }

  parts.push(`\n## MMD prefix: ${mmdPrefix}`);
  parts.push(`（所有节点 ID 必须以此前缀开头，如 ${mmdPrefix}-N1, ${mmdPrefix}-N2...）`);
  parts.push(`\n## Current task label: ${taskLabel}`);

  // Char count warning
  if (charCount > 2500) {
    parts.push(`\n## Current MMD size: ${charCount} chars (budget: 4000 chars)`);
    parts.push("⚠ 接近上限，请积极合并节点、精简 summary，优先使用 replace 模式微调而非 write 全量重写。");
  } else if (charCount > 2000) {
    parts.push(`\n## Current MMD size: ${charCount} chars (budget: 4000 chars)`);
    parts.push("注意控制增长，合并同类节点。");
  }

  // Existing MMD with line numbers
  parts.push("\n## Existing Mermaid content:");
  if (existingMmd) {
    const lines = existingMmd.split("\n");
    for (let i = 0; i < lines.length; i++) {
      parts.push(`L${i + 1}: ${lines[i]}`);
    }
  } else {
    parts.push("(empty — create new)");
  }

  // New entries
  parts.push("\n## New offload entries to incorporate:");
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    parts.push(`${i + 1}. [${e.tool_call_id}] ${e.tool_call} → ${e.summary} (${e.timestamp})`);
  }

  parts.push("\n请根据系统指令生成/更新 Mermaid 流程图，并输出合法的 JSON 对象（含 node_mapping）。");
  return parts.join("\n");
}
