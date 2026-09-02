import { describe, expect, it } from "vitest";

import type { ExtractedMemory } from "./l1-writer.js";
import { mergeCockpitContractReview } from "./cockpit-contract-review.js";

function memory(params: {
  state: string;
  episode?: string;
  source?: string;
  status?: "complete" | "partial" | "invalid";
  supersedes?: string[];
  issues?: string[];
}): ExtractedMemory {
  return {
    content: `${params.state} memory`,
    type: "episodic",
    priority: 70,
    scene_name: "cockpit",
    source_message_ids: [params.source ?? "u1"],
    metadata: {
      domain: "navigation",
      slot: params.state,
      value: "value",
      subject: "user",
      state_key: `navigation|user|car|driver|${params.state}`,
      episode_key: params.episode ?? "route-1",
      relation: params.supersedes ? "cancelled" : "asserted",
      supersedes: params.supersedes,
      construction_quality: {
        status: params.status ?? "complete",
        issues: params.issues ?? [],
      },
    },
  };
}

describe("cockpit contract review merge", () => {
  it("retains complete drafts and accepts only a new complete atomic identity", () => {
    const destination = memory({ state: "destination" });
    const reminder = memory({ state: "reminder_time" });
    const result = mergeCockpitContractReview(
      [destination],
      [destination, reminder, memory({ state: "hallucinated", status: "partial" })],
    );

    expect(result.memories).toEqual([destination, reminder]);
    expect(result.added).toBe(1);
    expect(result.replacedDefective).toBe(0);
  });

  it("replaces one ambiguous transition only when atomic rows cover every old reference", () => {
    const ambiguous = memory({
      state: "status",
      status: "partial",
      supersedes: ["destination-v2", "pickup-v1"],
      issues: ["ambiguous_transition_state"],
    });
    const destinationCancel = memory({
      state: "destination",
      supersedes: ["destination-v2"],
    });
    const pickupCancel = memory({
      state: "pickup_time",
      supersedes: ["pickup-v1"],
    });
    const result = mergeCockpitContractReview(
      [ambiguous],
      [destinationCancel, pickupCancel],
    );

    expect(result.memories).toEqual([destinationCancel, pickupCancel]);
    expect(result.replacedDefective).toBe(1);
  });

  it("keeps a defective draft when reviewed rows leave lineage uncovered", () => {
    const ambiguous = memory({
      state: "status",
      status: "partial",
      supersedes: ["destination-v2", "pickup-v1"],
    });
    const destinationCancel = memory({
      state: "destination",
      supersedes: ["destination-v2"],
    });
    const result = mergeCockpitContractReview([ambiguous], [destinationCancel]);

    expect(result.memories).toEqual([ambiguous, destinationCancel]);
    expect(result.replacedDefective).toBe(0);
  });

  it("never replaces a complete draft or accepts a reviewed partial duplicate", () => {
    const destination = memory({ state: "destination" });
    const reviewedPartial = memory({ state: "destination", status: "partial" });
    const result = mergeCockpitContractReview([destination], [reviewedPartial]);

    expect(result.memories).toEqual([destination]);
    expect(result.reviewedComplete).toBe(0);
  });
});
