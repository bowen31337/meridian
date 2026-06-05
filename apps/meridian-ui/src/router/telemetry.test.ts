import type { Span } from "@opentelemetry/api";
import { describe, expect, it, vi } from "vitest";
import type { AuditLogEntry } from "../workspace/types.js";
import { recordRouteFailure, recordRouteNavigationEvent } from "./telemetry.js";

vi.mock("@opentelemetry/api", () => ({
  trace: { getTracer: vi.fn() },
  SpanStatusCode: { UNSET: 0, OK: 1, ERROR: 2 },
}));

function makeSpan() {
  return {
    addEvent: vi.fn(),
    setStatus: vi.fn(),
    recordException: vi.fn(),
    end: vi.fn(),
  };
}

type MockSpan = ReturnType<typeof makeSpan>;

function asSpan(s: MockSpan): Span {
  return s as unknown as Span;
}

function makeAuditLog() {
  const entries: AuditLogEntry[] = [];
  return { write: (e: AuditLogEntry) => entries.push(e), entries };
}

describe("recordRouteNavigationEvent", () => {
  it("adds ui.navigation event with correct attributes", () => {
    const span = makeSpan();
    const ts = "2026-01-01T00:00:00.000Z";
    recordRouteNavigationEvent(asSpan(span), {
      name: "ui.navigation",
      route: "/sessions",
      timestamp: ts,
    });
    expect(span.addEvent).toHaveBeenCalledWith(
      "ui.navigation",
      expect.objectContaining({
        "event.name": "ui.navigation",
        "route.path": "/sessions",
        "event.timestamp": ts,
      }),
    );
  });

  it("records the route path for nested routes", () => {
    const span = makeSpan();
    recordRouteNavigationEvent(asSpan(span), {
      name: "ui.navigation",
      route: "/sessions/sess_123",
      timestamp: "2026-01-01T00:00:00.000Z",
    });
    expect(span.addEvent).toHaveBeenCalledWith(
      "ui.navigation",
      expect.objectContaining({ "route.path": "/sessions/sess_123" }),
    );
  });
});

describe("recordRouteFailure", () => {
  it("sets span status to ERROR", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordRouteFailure(asSpan(span), new Error("crash"), auditLog, { route: "/agents" });
    expect(span.setStatus).toHaveBeenCalledWith(expect.objectContaining({ code: 2 }));
  });

  it("records exception on span for Error instances", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    const err = new Error("crash");
    recordRouteFailure(asSpan(span), err, auditLog, { route: "/agents" });
    expect(span.recordException).toHaveBeenCalledWith(err);
  });

  it("writes error entry to audit log with route and message", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordRouteFailure(asSpan(span), new Error("page failed"), auditLog, { route: "/settings" });
    expect(auditLog.entries).toHaveLength(1);
    const [entry] = auditLog.entries;
    expect(entry?.level).toBe("error");
    expect(entry?.event).toBe("ui.route.failed");
    expect(entry?.detail?.route).toBe("/settings");
    expect(entry?.detail?.message).toBe("page failed");
  });

  it("handles non-Error failures gracefully", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordRouteFailure(asSpan(span), "string error", auditLog, { route: "/" });
    expect(auditLog.entries[0]?.detail?.message).toBe("string error");
    expect(span.recordException).not.toHaveBeenCalled();
  });

  it("includes an empty sessionId in the audit entry", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordRouteFailure(asSpan(span), new Error("oops"), auditLog, { route: "/vaults" });
    expect(auditLog.entries[0]?.sessionId).toBe("");
  });
});
