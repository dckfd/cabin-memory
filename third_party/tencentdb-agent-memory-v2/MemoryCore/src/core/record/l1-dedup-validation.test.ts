import { describe, expect, it } from "vitest";

import type { CandidateMatch } from "../prompts/l1-dedup.js";
import { parseBatchResult } from "./l1-dedup.js";
import type { ExtractedMemory, MemoryRecord } from "./l1-writer.js";

function memory(recordId: string): ExtractedMemory & { record_id: string } {
  return {
    record_id: recordId,
    content: recordId,
    type: "episodic",
    priority: 80,
    scene_name: "navigation",
    source_message_ids: [`source-${recordId}`],
    metadata: { domain: "navigation", slot: "destination", value: recordId },
  };
}

function candidate(id: string): MemoryRecord {
  return {
    id,
    content: id,
    type: "episodic",
    priority: 70,
    scene_name: "navigation",
    source_message_ids: [`source-${id}`],
    metadata: {},
    timestamps: ["2026-09-01T00:00:00.000Z"],
    createdAt: "2026-09-01T00:00:00.000Z",
    updatedAt: "2026-09-01T00:00:00.000Z",
    sessionKey: "old-session-key",
    sessionId: "old-session-id",
  };
}

const memories = [memory("new-a"), memory("new-b")];
const matches: CandidateMatch[] = [
  { newMemory: memories[0], candidates: [candidate("old-a"), candidate("old-shared")] },
  { newMemory: memories[1], candidates: [candidate("old-b"), candidate("old-shared")] },
];

const fallback = memories.map((entry) => ({
  record_id: entry.record_id,
  action: "store" as const,
  target_ids: [],
}));

describe("L1 dedup target authorization", () => {
  it("accepts only targets recalled for that exact new memory", () => {
    const parsed = parseBatchResult(JSON.stringify([
      {
        record_id: "new-a",
        action: "update",
        target_ids: ["old-a"],
        merged_content: "updated",
        merged_type: "episodic",
        merged_priority: 80,
        merged_timestamps: ["2026-09-01T00:00:00.000Z"],
      },
      { record_id: "new-b", action: "store", target_ids: [] },
    ]), memories, matches);

    expect(parsed.map((decision) => ({
      record_id: decision.record_id,
      action: decision.action,
      target_ids: decision.target_ids,
    }))).toEqual([
      { record_id: "new-a", action: "update", target_ids: ["old-a"] },
      { record_id: "new-b", action: "store", target_ids: [] },
    ]);
  });

  it.each([
    ["forged target", [
      { record_id: "new-a", action: "update", target_ids: ["not-recalled"] },
    ]],
    ["another memory's candidate", [
      { record_id: "new-a", action: "merge", target_ids: ["old-b"] },
    ]],
    ["duplicate decision", [
      { record_id: "new-a", action: "store", target_ids: [] },
      { record_id: "new-a", action: "store", target_ids: [] },
    ]],
    ["duplicate target", [
      { record_id: "new-a", action: "merge", target_ids: ["old-a", "old-a"] },
    ]],
    ["empty destructive target", [
      { record_id: "new-a", action: "update", target_ids: [] },
    ]],
    ["target on store", [
      { record_id: "new-a", action: "store", target_ids: ["old-a"] },
    ]],
    ["unknown new record", [
      { record_id: "unknown-new", action: "update", target_ids: ["old-a"] },
    ]],
    ["shared target consumed twice", [
      { record_id: "new-a", action: "update", target_ids: ["old-shared"] },
      { record_id: "new-b", action: "merge", target_ids: ["old-shared"] },
    ]],
  ] as const)("fails closed to store-all for %s", (_label, decisions) => {
    expect(parseBatchResult(JSON.stringify(decisions), memories, matches)).toEqual(fallback);
  });
});
