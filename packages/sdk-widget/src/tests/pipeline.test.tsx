import React from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { WidgetProps } from "../contract.js";
import { defineWidget } from "../contract.js";
import type { AuditLog } from "../audit.js";
import { WidgetRegistry } from "../pipeline.js";
import type { ContentBlockCanvasOp, WidgetError } from "../types.js";

// ---------------------------------------------------------------------------
// OTel mock — the SDK itself never configures a tracer; tests supply a noop.
// ---------------------------------------------------------------------------
const mockSpan = {
  addEvent: vi.fn(),
  setStatus: vi.fn(),
  recordException: vi.fn(),
  end: vi.fn(),
};

vi.mock("@opentelemetry/api", () => ({
  trace: {
    getTracer: () => ({
      startActiveSpan: (
        _name: string,
        _opts: unknown,
        fn: (span: typeof mockSpan) => unknown,
      ) => fn(mockSpan),
    }),
  },
  SpanStatusCode: { UNSET: 0, OK: 1, ERROR: 2 },
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAuditLog(): AuditLog & { entries: unknown[] } {
  const entries: unknown[] = [];
  return { write: (e) => entries.push(e), entries };
}

const TEXT_MANIFEST = {
  kind: "test.text",
  version: "1.0.0",
  displayName: "Text",
  propsSchema: {
    type: "object",
    required: ["text"],
    properties: { text: { type: "string" } },
    additionalProperties: false,
  },
};

const TextWidget = defineWidget(
  ({ props }: WidgetProps<{ text: string }>) => (
    <p data-testid="text-widget">{props.text as string}</p>
  ),
  TEXT_MANIFEST,
);

function makeBlock(
  overrides: Partial<ContentBlockCanvasOp["canvas_op"]> = {},
): ContentBlockCanvasOp {
  return {
    type: "canvas_op",
    canvas_op: {
      op: "set",
      widget_id: "w1",
      widget_kind: "test.text",
      props: { text: "hello" },
      sequence: 1,
      session_id: "sess1",
      timestamp: "2026-05-16T00:00:00Z",
      ...overrides,
    },
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("WidgetRegistry", () => {
  let registry: WidgetRegistry;
  let auditLog: ReturnType<typeof makeAuditLog>;
  let errors: WidgetError[];

  beforeEach(() => {
    registry = new WidgetRegistry();
    registry.register(TextWidget);
    auditLog = makeAuditLog();
    errors = [];
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders a registered widget with valid props", () => {
    const el = registry.renderCanvasOp(makeBlock(), {
      auditLog,
      onError: (e) => errors.push(e),
    });
    render(el);
    expect(screen.getByTestId("text-widget").textContent).toBe("hello");
    expect(errors).toHaveLength(0);
    expect(auditLog.entries).toHaveLength(0);
  });

  it("starts an OTel span with widget attributes on every invocation", () => {
    registry.renderCanvasOp(makeBlock(), { auditLog });
    expect(mockSpan.addEvent).toHaveBeenCalledWith(
      "widget.invocation",
      expect.objectContaining({ "widget_id": "w1", "widget_kind": "test.text" }),
    );
  });

  it("ends the span on the success path", () => {
    registry.renderCanvasOp(makeBlock(), { auditLog });
    expect(mockSpan.end).toHaveBeenCalledTimes(1);
  });

  describe("WIDGET_NOT_FOUND", () => {
    const unknownBlock = makeBlock({ widget_kind: "acme.unknown" });

    it("renders WidgetErrorDisplay", () => {
      const el = registry.renderCanvasOp(unknownBlock, { auditLog, onError: (e) => errors.push(e) });
      render(el);
      expect(screen.getByRole("alert")).toBeTruthy();
      expect(screen.getByRole("alert").textContent).toMatch(/acme\.unknown/);
    });

    it("calls onError with code WIDGET_NOT_FOUND", () => {
      registry.renderCanvasOp(unknownBlock, { auditLog, onError: (e) => errors.push(e) });
      expect(errors).toHaveLength(1);
      expect(errors[0]!.code).toBe("WIDGET_NOT_FOUND");
    });

    it("writes an error entry to the audit log", () => {
      registry.renderCanvasOp(unknownBlock, { auditLog });
      expect(auditLog.entries).toHaveLength(1);
      const entry = auditLog.entries[0] as { level: string; event: string };
      expect(entry.level).toBe("error");
      expect(entry.event).toBe("widget.render.failed");
    });

    it("marks the OTel span as ERROR", () => {
      registry.renderCanvasOp(unknownBlock, { auditLog });
      expect(mockSpan.setStatus).toHaveBeenCalledWith(
        expect.objectContaining({ code: 2 }),
      );
    });
  });

  describe("WIDGET_PROPS_INVALID", () => {
    const badBlock = makeBlock({ props: { text: 99 } });

    it("renders WidgetErrorDisplay", () => {
      const el = registry.renderCanvasOp(badBlock, { auditLog, onError: (e) => errors.push(e) });
      render(el);
      expect(screen.getByRole("alert")).toBeTruthy();
    });

    it("calls onError with code WIDGET_PROPS_INVALID", () => {
      registry.renderCanvasOp(badBlock, { auditLog, onError: (e) => errors.push(e) });
      expect(errors[0]!.code).toBe("WIDGET_PROPS_INVALID");
    });

    it("writes an error entry to the audit log", () => {
      registry.renderCanvasOp(badBlock, { auditLog });
      expect(auditLog.entries).toHaveLength(1);
    });
  });

  it("throws on duplicate registration", () => {
    expect(() => registry.register(TextWidget)).toThrowError(/already registered/);
  });
});
