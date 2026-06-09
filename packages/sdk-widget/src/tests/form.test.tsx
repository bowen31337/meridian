import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AuditLog } from "../audit.js";
import type { OnInteraction } from "../interaction.js";
import type { ContentBlockCanvasOp } from "../types.js";
import { WidgetRegistry } from "../pipeline.js";
import { FormWidget } from "../widgets/form.js";

afterEach(() => cleanup());

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

function makeAuditLog(): AuditLog & { entries: unknown[] } {
  const entries: unknown[] = [];
  return { write: (e) => entries.push(e), entries };
}

function makeBlock(
  overrides: Partial<ContentBlockCanvasOp["canvas_op"]> = {},
): ContentBlockCanvasOp {
  return {
    type: "canvas_op",
    canvas_op: {
      op: "set",
      widget_id: "wForm",
      widget_kind: "meridian.form",
      props: {
        fields: [{ name: "name", label: "Name", type: "text" }],
      },
      sequence: 1,
      session_id: "sess1",
      timestamp: "2026-06-09T00:00:00Z",
      ...overrides,
    },
  };
}

function renderForm(
  block: ContentBlockCanvasOp,
  opts: {
    onInteraction?: OnInteraction | null;
    auditLog?: AuditLog;
  } = {},
): { auditLog: AuditLog & { entries: unknown[] } } {
  const registry = new WidgetRegistry();
  registry.register(FormWidget);
  const auditLog = (opts.auditLog as AuditLog & { entries: unknown[] }) ?? makeAuditLog();
  const renderOpts = {
    auditLog,
    ...(opts.onInteraction ? { onInteraction: opts.onInteraction } : {}),
  };
  const el = registry.renderCanvasOp(block, renderOpts);
  render(el);
  return { auditLog };
}

describe("FormWidget interactions", () => {
  it("renders submit button only when onInteraction is provided", () => {
    renderForm(makeBlock(), { onInteraction: vi.fn() });
    expect(screen.getByTestId("form-submit-button")).toBeTruthy();
  });

  it("renders no submit button when onInteraction is null", () => {
    renderForm(makeBlock(), { onInteraction: null });
    expect(screen.queryByTestId("form-submit-button")).toBeNull();
  });

  it("submits the form with collected values (text/number/checkbox)", async () => {
    const onInteraction = vi.fn().mockResolvedValue(undefined);
    renderForm(
      makeBlock({
        props: {
          fields: [
            { name: "name", label: "Name", type: "text", value: "Alice" },
            { name: "age", label: "Age", type: "number", value: 30 },
            { name: "agree", label: "Agree?", type: "checkbox", value: true },
          ],
        },
      }),
      { onInteraction },
    );

    fireEvent.click(screen.getByTestId("form-submit-button"));

    await waitFor(() => expect(onInteraction).toHaveBeenCalledTimes(1));
    const event = onInteraction.mock.calls[0]?.[0] as {
      kind: string;
      payload: { values: Record<string, unknown> };
    };
    expect(event.kind).toBe("form.submit");
    expect(event.payload.values.name).toBe("Alice");
    expect(event.payload.values.age).toBe(30);
    expect(event.payload.values.agree).toBe(true);
  });

  it("submits empty number field as null", async () => {
    const onInteraction = vi.fn().mockResolvedValue(undefined);
    renderForm(
      makeBlock({
        props: {
          fields: [{ name: "qty", label: "Quantity", type: "number" }],
        },
      }),
      { onInteraction },
    );

    fireEvent.click(screen.getByTestId("form-submit-button"));

    await waitFor(() => expect(onInteraction).toHaveBeenCalled());
    const event = onInteraction.mock.calls[0]?.[0] as {
      payload: { values: Record<string, unknown> };
    };
    expect(event.payload.values.qty).toBeNull();
  });

  it("on submit failure, displays error message and writes audit log", async () => {
    const onInteraction = vi.fn().mockRejectedValue(new Error("submit boom"));
    const { auditLog } = renderForm(makeBlock(), { onInteraction });

    fireEvent.click(screen.getByTestId("form-submit-button"));

    await waitFor(() => {
      expect(screen.queryByTestId("form-widget-error")).toBeTruthy();
    });
    expect(screen.getByTestId("form-widget-error").textContent).toContain("submit boom");
    expect(auditLog.entries).toHaveLength(1);
    const entry = auditLog.entries[0] as { event: string; level: string };
    expect(entry.event).toBe("form.submit.failed");
    expect(entry.level).toBe("error");
  });

  it("renders action buttons when actions prop supplied", () => {
    renderForm(
      makeBlock({
        props: {
          fields: [{ name: "x", label: "X", type: "text" }],
          actions: [
            { name: "cancel", label: "Cancel" },
            { name: "reset", label: "Reset" },
          ],
        },
      }),
      { onInteraction: vi.fn() },
    );
    expect(screen.getByTestId("form-action-cancel").textContent).toBe("Cancel");
    expect(screen.getByTestId("form-action-reset").textContent).toBe("Reset");
  });

  it("clicking an action button dispatches a button.click interaction", async () => {
    const onInteraction = vi.fn().mockResolvedValue(undefined);
    renderForm(
      makeBlock({
        props: {
          fields: [{ name: "x", label: "X", type: "text" }],
          actions: [{ name: "cancel", label: "Cancel" }],
        },
      }),
      { onInteraction },
    );

    fireEvent.click(screen.getByTestId("form-action-cancel"));

    await waitFor(() => expect(onInteraction).toHaveBeenCalledTimes(1));
    const event = onInteraction.mock.calls[0]?.[0] as {
      kind: string;
      payload: { action: string };
    };
    expect(event.kind).toBe("button.click");
    expect(event.payload.action).toBe("cancel");
  });

  it("on action-click failure, displays error and writes audit log", async () => {
    const onInteraction = vi.fn().mockRejectedValue(new Error("click boom"));
    const { auditLog } = renderForm(
      makeBlock({
        props: {
          fields: [{ name: "x", label: "X", type: "text" }],
          actions: [{ name: "cancel", label: "Cancel" }],
        },
      }),
      { onInteraction },
    );

    fireEvent.click(screen.getByTestId("form-action-cancel"));

    await waitFor(() => {
      expect(screen.queryByTestId("form-widget-error")).toBeTruthy();
    });
    expect(screen.getByTestId("form-widget-error").textContent).toContain("click boom");
    expect(auditLog.entries).toHaveLength(1);
    const entry = auditLog.entries[0] as { event: string };
    expect(entry.event).toBe("button.click.failed");
  });
});
