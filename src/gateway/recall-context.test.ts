import { describe, expect, it } from "vitest";
import { composeRecallContext } from "./recall-context.js";

describe("composeRecallContext", () => {
  it("keeps dynamic, raw, and stable evidence", () => {
    const context = composeRecallContext({ dynamicL1: "fact", l0: "quote", stable: "persona" });
    expect(context).toContain("fact");
    expect(context).toContain("quote");
    expect(context).toContain("persona");
  });

  it("does not emit empty sections", () => {
    expect(composeRecallContext({ dynamicL1: "fact" })).not.toContain("Stable memory");
  });
});
