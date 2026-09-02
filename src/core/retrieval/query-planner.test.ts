import { describe, expect, it } from "vitest";
import { planRetrievalQueries } from "./query-planner.js";

describe("planRetrievalQueries", () => {
  it("preserves a simple query without inventing variants", () => {
    expect(planRetrievalQueries("Where did Alice travel?").queries).toEqual(["Where did Alice travel?"]);
  });

  it("decomposes generic multi-event questions", () => {
    const plan = planRetrievalQueries("How long was it between Alice moved to Rome and Alice started work?");
    expect(plan.queries[0]).toContain("between");
    expect(plan.queries).toContain("Alice moved to Rome");
    expect(plan.queries).toContain("Alice started work");
    expect(plan.decomposed).toBe(true);
  });

  it("supports Chinese clauses with a bounded query count", () => {
    const plan = planRetrievalQueries("小王毕业之后去了哪里，然后什么时候开始工作？", 3);
    expect(plan.queries.length).toBeGreaterThan(1);
    expect(plan.queries.length).toBeLessThanOrEqual(3);
  });

});
