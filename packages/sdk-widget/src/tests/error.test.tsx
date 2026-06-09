import { cleanup, render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => cleanup());

import { AuditLogContext, type AuditLog } from "../audit.js";
import { CanvasWidgetErrorBoundary, WidgetErrorDisplay } from "../error.js";
import type { WidgetContext } from "../contract.js";
import type { WidgetError } from "../types.js";

function makeAuditLog(): AuditLog & { entries: unknown[] } {
  const entries: unknown[] = [];
  return { write: (e) => entries.push(e), entries };
}

const ctx: WidgetContext = {
  widgetId: "w1",
  widgetKind: "test.kind",
  sessionId: "sess1",
  sequence: 1,
};

function Boom(): React.ReactElement {
  throw new Error("boom from widget");
}

describe("WidgetErrorDisplay", () => {
  it("renders the error message with data attributes", () => {
    const error: WidgetError = {
      code: "WIDGET_RENDER_ERROR",
      message: "the widget exploded",
      widget_id: "wA",
      widget_kind: "test.kindA",
      session_id: "sessA",
      timestamp: "2026-06-09T00:00:00Z",
    };
    const { container } = render(<WidgetErrorDisplay error={error} />);
    const alert = container.querySelector("[role='alert']") as HTMLElement;
    expect(alert).toBeTruthy();
    expect(alert.getAttribute("data-widget-id")).toBe("wA");
    expect(alert.getAttribute("data-widget-kind")).toBe("test.kindA");
    expect(alert.getAttribute("data-error-code")).toBe("WIDGET_RENDER_ERROR");
    expect(alert.textContent).toContain("the widget exploded");
  });
});

describe("CanvasWidgetErrorBoundary", () => {
  it("renders children when no error", () => {
    const auditLog = makeAuditLog();
    render(
      <AuditLogContext.Provider value={auditLog}>
        <CanvasWidgetErrorBoundary ctx={ctx}>
          <span data-testid="ok">healthy</span>
        </CanvasWidgetErrorBoundary>
      </AuditLogContext.Provider>,
    );
    expect(screen.getByTestId("ok")).toBeTruthy();
  });

  it("catches a render-time throw, writes audit log, calls onError, and shows fallback", () => {
    const auditLog = makeAuditLog();
    const errors: WidgetError[] = [];
    // Silence React's expected console.error during ErrorBoundary tests.
    const origError = console.error;
    console.error = vi.fn();
    try {
      render(
        <AuditLogContext.Provider value={auditLog}>
          <CanvasWidgetErrorBoundary ctx={ctx} onError={(e) => errors.push(e)}>
            <Boom />
          </CanvasWidgetErrorBoundary>
        </AuditLogContext.Provider>,
      );
    } finally {
      console.error = origError;
    }

    // Fallback rendered
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toContain("boom from widget");
    expect(alert.getAttribute("data-error-code")).toBe("WIDGET_RENDER_ERROR");

    // onError was called with the WidgetError
    expect(errors).toHaveLength(1);
    expect(errors[0].code).toBe("WIDGET_RENDER_ERROR");
    expect(errors[0].widget_id).toBe("w1");
    expect(errors[0].widget_kind).toBe("test.kind");
    expect(errors[0].session_id).toBe("sess1");

    // Audit log entry was written
    expect(auditLog.entries).toHaveLength(1);
    const entry = auditLog.entries[0] as { level: string; event: string };
    expect(entry.level).toBe("error");
    expect(entry.event).toBe("widget.render.failed");
  });

  it("catches a non-Error throw (string) and converts it to a message", () => {
    const auditLog = makeAuditLog();
    function ThrowString(): React.ReactElement {
      throw "stringy error";
    }
    const origError = console.error;
    console.error = vi.fn();
    try {
      render(
        <AuditLogContext.Provider value={auditLog}>
          <CanvasWidgetErrorBoundary ctx={ctx}>
            <ThrowString />
          </CanvasWidgetErrorBoundary>
        </AuditLogContext.Provider>,
      );
    } finally {
      console.error = origError;
    }
    expect(screen.getByRole("alert").textContent).toContain("stringy error");
  });
});
