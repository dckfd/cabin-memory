import { describe, expect, it } from "vitest";

import { renderTdaiMemoryToolsBlock } from "../injection/injectors/tdai-tools-injector.js";
import {
  MEMORY_TOOLS_GUIDE,
  buildMemoryToolsGuide,
} from "../injection/injectors/tdai-profile-memory-injector.js";

describe("smart-cockpit memory injection guidance", () => {
  it("keeps the generic tools prompt unchanged by default", () => {
    const block = renderTdaiMemoryToolsBlock(
      "http://localhost:8096",
      "session-1",
      "default",
    );
    expect(block).toContain("用户偏好的编程语言");
    expect(block).not.toContain("智能座舱使用边界");
    expect(buildMemoryToolsGuide()).toBe(MEMORY_TOOLS_GUIDE);
  });

  it("uses cockpit retrieval examples and evidence boundaries when enabled", () => {
    const block = renderTdaiMemoryToolsBlock(
      "http://localhost:8096",
      "session-1",
      "default",
      "smart-cockpit",
    );
    expect(block).toContain("上次目的地、座舱设置");
    expect(block).toContain("智能座舱使用边界");
    expect(block).toContain("主驾/副驾/后排");
    expect(block).toContain("不得复用为永久授权");
    expect(block).toContain("用户上次选择的导航目的地");
    expect(block).not.toContain("用户偏好的编程语言");
  });

  it("adapts active-recall policy away from coding examples", () => {
    const guide = buildMemoryToolsGuide("smart-cockpit");
    expect(guide).toContain("上次选的目的地是哪里");
    expect(guide).toContain("智能座舱记忆规则");
    expect(guide).toContain("瞬时状态必须通过当前工具读取");
    expect(guide).toContain("不猜最终实体");
    expect(guide).not.toContain("那个 bug 我们怎么修的");
  });
});
