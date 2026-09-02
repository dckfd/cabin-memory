import { describe, expect, it } from "vitest";

import { resolveL3ExecutionScopes } from "./pipeline-factory.js";

describe("L3 execution scope", () => {
  it("runs only the profile protected by the queue task's agent lock", () => {
    expect(resolveL3ExecutionScopes(
      ["team:t|agent:caroline", "team:t|agent:melanie"],
      "team:t|agent:melanie",
    )).toEqual(["team:t|agent:melanie"]);
  });

  it("preserves legacy full-sweep and default-scope behavior", () => {
    expect(resolveL3ExecutionScopes([
      "team:t|agent:caroline",
      "team:t|agent:melanie",
    ])).toEqual([
      "team:t|agent:caroline",
      "team:t|agent:melanie",
    ]);
    expect(resolveL3ExecutionScopes([])).toEqual(["global"]);
  });
});
