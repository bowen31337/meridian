import { describe, expect, it } from "vitest";
import { validateProps } from "../manifest.js";
import type { WidgetManifest } from "../manifest.js";

const textManifest: WidgetManifest = {
  kind: "test.text",
  version: "1.0.0",
  displayName: "Text",
  propsSchema: {
    type: "object",
    required: ["text"],
    properties: {
      text: { type: "string" },
      bold: { type: "boolean" },
    },
    additionalProperties: false,
  },
};

describe("validateProps", () => {
  it("returns valid:true for correct props", () => {
    const result = validateProps(textManifest, { text: "hello" });
    expect(result.valid).toBe(true);
    expect(result.errors).toBeUndefined();
  });

  it("accepts optional props when present and correct type", () => {
    const result = validateProps(textManifest, { text: "hi", bold: true });
    expect(result.valid).toBe(true);
  });

  it("returns valid:false when required prop is missing", () => {
    const result = validateProps(textManifest, {});
    expect(result.valid).toBe(false);
    expect(result.errors).toBeDefined();
    expect(result.errors?.length).toBeGreaterThan(0);
  });

  it("returns valid:false when prop has wrong type", () => {
    const result = validateProps(textManifest, { text: 42 });
    expect(result.valid).toBe(false);
    expect(result.errors?.some((e) => e.includes("text"))).toBe(true);
  });

  it("returns valid:false with multiple errors when allErrors is set", () => {
    const result = validateProps(textManifest, { text: 42, bold: "yes" });
    expect(result.valid).toBe(false);
    expect(result.errors?.length).toBeGreaterThanOrEqual(2);
  });

  it("returns valid:false on additional properties", () => {
    const result = validateProps(textManifest, { text: "hi", unknown: true });
    expect(result.valid).toBe(false);
  });

  it("reuses compiled validator on repeated calls (same kind@version)", () => {
    // Calling twice should not throw; same validator is reused internally.
    expect(validateProps(textManifest, { text: "a" }).valid).toBe(true);
    expect(validateProps(textManifest, { text: "b" }).valid).toBe(true);
  });

  it("handles an empty propsSchema (accepts any object)", () => {
    const anyManifest: WidgetManifest = {
      kind: "test.any",
      version: "1.0.0",
      displayName: "Any",
      propsSchema: { type: "object" },
    };
    expect(validateProps(anyManifest, { whatever: 99 }).valid).toBe(true);
  });
});
