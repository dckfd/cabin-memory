/**
 * L1.5 Task Judgment Prompt — determines task lifecycle.
 */
import type { MmdMeta } from "../types.js";

export {
  L15_SYSTEM_PROMPT,
  buildL15SystemPrompt,
} from "../../offload/prompts/system-prompts.js";

export interface L15CurrentMmd {
  filename: string;
  content: string;
}

/**
 * Build the L1.5 user prompt for task judgment.
 */
export function buildL15UserPrompt(
  recentMessages: string,
  currentMmd: L15CurrentMmd | null,
  metas: MmdMeta[],
): string {
  const parts: string[] = [];

  parts.push("## 1. 最近的对话上下文 (Recent messages):");
  parts.push(recentMessages);
  parts.push("\n## 2. 当前挂载的任务图 (Active Mermaid — 完整内容):");

  if (currentMmd && currentMmd.filename) {
    parts.push(`**File:** ${currentMmd.filename}`);
    parts.push(`\n\`\`\`mermaid\n${currentMmd.content}\n\`\`\``);
  } else {
    parts.push("(none - 当前处于闲置状态，无活跃任务)");
  }

  parts.push("\n## 3. 历史可用的任务图 (Available Mermaid task files):");

  if (metas.length === 0) {
    parts.push("(none - 暂无历史长任务)");
  } else {
    for (const m of metas) {
      const total = m.doneCount + m.doingCount + m.todoCount;
      parts.push(`- **${m.filename}**`);
      parts.push(`  taskGoal: ${m.taskGoal}`);
      parts.push(
        `  progress: ${m.doneCount}/${total} done, ${m.doingCount} doing, ${m.todoCount} todo`,
      );
      if (m.updatedTime) {
        parts.push(`  lastUpdated: ${m.updatedTime}`);
      }
      if (m.nodeSummaries && m.nodeSummaries.length > 0) {
        parts.push("  recentNodes:");
        for (const n of m.nodeSummaries) {
          parts.push(`    - [${n.nodeId}] (${n.status}) ${n.summary}`);
        }
      }
      parts.push("");
    }
  }

  parts.push(
    "请严格根据系统指令的【三步思考链路】进行研判，并输出合法的 JSON 对象。",
  );
  return parts.join("\n");
}
