import { describe, expect, it } from "vitest";
import { shouldExpandConversationContext, shouldPreferConsensusFusion } from "./context-policy.js";

describe("shouldExpandConversationContext", () => {
  it.each([
    "When did Alice start work?",
    "Why did John avoid the cafe?",
    "Which movie theme did Tim play?",
    "小王为什么离开公司？",
  ])("expands evidence-seeking detail queries: %s", (query) => {
    expect(shouldExpandConversationContext(query)).toBe(true);
  });

  it.each([
    "Did Alice move for her dog?",
    "How many scripts did Joanna write?",
    "What were the two concerts?",
    "What music events has Alice attended?",
    "Which places did the user visit?",
    "是否搬过家？",
    "有几个项目？",
  ])("avoids distractors for constrained queries: %s", (query) => {
    expect(shouldExpandConversationContext(query)).toBe(false);
  });
});

describe("shouldPreferConsensusFusion", () => {
  it.each([
    "How many days a week do I exercise?",
    "Who finished first, second and third?",
    "有几个项目？",
  ])("recognizes bounded aggregation: %s", (query) => {
    expect(shouldPreferConsensusFusion(query)).toBe(true);
  });

  it("keeps open multi-hop questions coverage-first", () => {
    expect(shouldPreferConsensusFusion("Why did Alice move after changing jobs?")).toBe(false);
  });
});
