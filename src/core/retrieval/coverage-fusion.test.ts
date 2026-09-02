import { describe, expect, it } from "vitest";
import { coverageFirstRrfMerge } from "./coverage-fusion.js";

describe("coverageFirstRrfMerge", () => {
  it("retains a distinct result from every query facet", () => {
    const common = { id: "common", score: 1 };
    const result = coverageFirstRrfMerge([
      [common, { id: "event-a", score: 0.8 }],
      [common, { id: "event-b", score: 0.7 }],
    ], 3);
    expect(result.map((item) => item.id)).toEqual(["common", "event-b", "event-a"]);
  });

  it("deduplicates results and respects the budget", () => {
    const result = coverageFirstRrfMerge([
      [{ id: "a", score: 1 }],
      [{ id: "a", score: 1 }, { id: "b", score: 0.5 }],
    ], 2);
    expect(result.map((item) => item.id)).toEqual(["a", "b"]);
  });

  it("can reserve multiple distinct results for each retrieval source", () => {
    const result = coverageFirstRrfMerge([
      [{ id: "shared", score: 1 }, { id: "keyword-1", score: 0.9 }, { id: "keyword-2", score: 0.8 }],
      [{ id: "shared", score: 1 }, { id: "semantic-1", score: 0.9 }, { id: "semantic-2", score: 0.8 }],
    ], 5, 60, 2);
    expect(result.map((item) => item.id)).toEqual([
      "shared", "semantic-1", "keyword-1", "semantic-2", "keyword-2",
    ]);
  });
});
