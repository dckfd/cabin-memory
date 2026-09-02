import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PersonaGenerator } from "./persona-generator.js";
import type { LLMRunner } from "../types.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) =>
    fs.rm(directory, { recursive: true, force: true })
  ));
});

describe("PersonaGenerator pipeline failure contract", () => {
  it("uses the configured timeout and surfaces provider failure for worker retry", async () => {
    const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "tdai-l3-test-"));
    temporaryDirectories.push(dataDir);
    const failure = new Error("provider timeout");
    const run = vi.fn(async () => { throw failure; });
    const runner: LLMRunner = { run };
    const generator = new PersonaGenerator({
      dataDir,
      config: {},
      llmRunner: runner,
      timeoutMs: 600_000,
      throwOnFailure: true,
    });

    await expect(generator.generateLocalPersona("test trigger")).rejects.toThrow(
      "provider timeout",
    );
    expect(run).toHaveBeenCalledWith(expect.objectContaining({
      taskId: "persona-generation",
      timeoutMs: 600_000,
    }));
  });
});
