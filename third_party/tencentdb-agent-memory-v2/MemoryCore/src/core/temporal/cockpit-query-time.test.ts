import { describe, expect, it } from "vitest";

import { parseConfig } from "../../config.js";
import { performAutoRecall } from "../hooks/auto-recall.js";
import { normalizeCockpitTemporalQuery } from "./cockpit-query-time.js";

const REQUEST_TIME = new Date("2026-08-26T02:30:00.000Z");

describe("cockpit query-time normalization", () => {
  it("anchors Chinese relative days and dayparts to request metadata", () => {
    const result = normalizeCockpitTemporalQuery(
      "昨天上午导航去了哪里，前天晚上听了什么？",
      { requestTime: REQUEST_TIME, timezone: "Asia/Shanghai" },
    );

    expect(result.active).toBe(true);
    expect(result.requestLocalDate).toBe("2026-08-26");
    expect(result.resolutions).toEqual([
      {
        expression: "昨天上午",
        localDate: "2026-08-25",
        dayPart: "morning",
        localRange: "2026-08-25 06:00–12:00",
      },
      {
        expression: "前天晚上",
        localDate: "2026-08-24",
        dayPart: "evening",
        localRange: "2026-08-24 18:00–24:00",
      },
    ]);
    expect(result.retrievalText).toContain("2026-08-25 06:00–12:00");
    expect(result.evidenceEnvelope).toContain("not memory ingestion time");
  });

  it("does not double-resolve tomorrow inside day after tomorrow", () => {
    const result = normalizeCockpitTemporalQuery(
      "Navigate there day after tomorrow morning, not tomorrow.",
      { requestTime: REQUEST_TIME, timezone: "UTC" },
    );

    expect(result.resolutions.map((item) => [item.expression.toLowerCase(), item.localDate])).toEqual([
      ["day after tomorrow morning", "2026-08-28"],
      ["tomorrow", "2026-08-27"],
    ]);
  });

  it("resolves plain yesterday as a full local day", () => {
    const result = normalizeCockpitTemporalQuery("昨天导航去了哪里？", {
      requestTime: REQUEST_TIME,
      timezone: "Asia/Shanghai",
    });
    expect(result.resolutions[0]?.localRange).toBe("2026-08-25 00:00–24:00");
  });

  it("falls back safely for invalid timezone and leaves retrieval-dependent phrases unresolved", () => {
    const unresolved = normalizeCockpitTemporalQuery(
      "还是播放上次那个",
      { requestTime: REQUEST_TIME, timezone: "invalid/zone" },
    );
    expect(unresolved.active).toBe(false);
    expect(unresolved.timezone).toBe("UTC");
    expect(unresolved.retrievalText).toBe("还是播放上次那个");
  });

  it("injects request-time evidence even when a new user has no memories yet", async () => {
    const cfg = parseConfig({
      promptMode: "cockpit",
      recall: { strategy: "keyword", timezone: "Asia/Shanghai" },
    });
    const result = await performAutoRecall({
      userText: "昨天上午我导航去哪里了？",
      actorId: "driver-1",
      sessionKey: "trip-1",
      cfg,
      pluginDataDir: "/tmp/tdai-temporal-test-does-not-exist",
      requestTime: REQUEST_TIME,
    });

    expect(result?.prependContext).toContain("<memory-query-time");
    expect(result?.prependContext).toContain("2026-08-25 06:00–12:00");
    expect(result?.appendSystemContext).toContain("按其中的本地绝对时间窗");
    expect(result?.recalledL1Memories).toEqual([]);
  });
});
