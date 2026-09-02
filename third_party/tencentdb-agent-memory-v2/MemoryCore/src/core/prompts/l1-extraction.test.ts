import { describe, expect, it } from "vitest";

import { EXTRACT_MEMORIES_SYSTEM_PROMPT } from "./l1-extraction.js";

describe("L1 extraction role ownership contract", () => {
  it("treats assistant-authored autobiographical claims as context only", () => {
    expect(EXTRACT_MEMORIES_SYSTEM_PROMPT).toContain(
      "只有 role=user 的发言可作为被记忆的事实来源",
    );
    expect(EXTRACT_MEMORIES_SYSTEM_PROMPT).toContain(
      "也绝不能把这些内容提取为 \"AI（姓名）\"",
    );
    expect(EXTRACT_MEMORIES_SYSTEM_PROMPT).not.toContain(
      "提取主体必须以\"用户（姓名）\"或\"AI\"为核心",
    );
  });
});
