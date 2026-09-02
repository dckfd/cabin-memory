import path from "node:path";
import { describe, expect, it } from "vitest";
import { generateSceneNavigation, stripSceneNavigation } from "./scene-navigation.js";

describe("scene navigation", () => {
  it("renders absolute scene paths when a data directory is supplied", () => {
    const dataDir = path.resolve("/tmp/tdai-scene-navigation-test");
    const output = generateSceneNavigation([
      {
        filename: "Project-State.md",
        summary: "Current project state",
        heat: 3,
        created: "2026-08-02T00:00:00Z",
        updated: "2026-08-02T00:00:00Z",
      },
    ], dataDir);

    expect(output).toContain(path.join(dataDir, "scene_blocks", "Project-State.md"));
    expect(output).not.toContain("### Path: scene_blocks/Project-State.md");
  });

  it("strips only the generated navigation suffix", () => {
    const persona = "# Persona\nStable facts";
    const navigation = generateSceneNavigation([
      {
        filename: "Facts.md",
        summary: "Facts",
        heat: 1,
        created: "",
        updated: "",
      },
    ]);
    expect(stripSceneNavigation(`${persona}\n\n${navigation}`)).toBe(persona);
  });
});
