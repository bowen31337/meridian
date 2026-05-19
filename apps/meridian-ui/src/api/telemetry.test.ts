import type { Span } from "@opentelemetry/api";
import { describe, expect, it, vi } from "vitest";
import type { AuditLogEntry } from "../workspace/types.js";
import { ApiError } from "./client.js";
import { recordApiFailure, recordApiInvocationEvent } from "./telemetry.js";

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

describe("recordApiInvocationEvent", () => {
  it("adds api.invocation event with correct attributes", () => {
    const span = makeSpan();
    const ts = "2026-01-01T00:00:00.000Z";
    recordApiInvocationEvent(asSpan(span), {
      name: "api.sessions.list",
      operation: "sessions.list",
      timestamp: ts,
    });
    expect(span.addEvent).toHaveBeenCalledWith(
      "api.invocation",
      expect.objectContaining({
        "event.name": "api.sessions.list",
        "api.operation": "sessions.list",
        "event.timestamp": ts,
      }),
    );
  });

  it("includes session.id when provided", () => {
    const span = makeSpan();
    recordApiInvocationEvent(asSpan(span), {
      name: "api.sessions.get",
      operation: "sessions.get",
      timestamp: "2026-01-01T00:00:00.000Z",
      sessionId: "sess_123",
    });
    expect(span.addEvent).toHaveBeenCalledWith(
      "api.invocation",
      expect.objectContaining({ "session.id": "sess_123" }),
    );
  });

  it("omits session.id when not provided", () => {
    const span = makeSpan();
    recordApiInvocationEvent(asSpan(span), {
      name: "api.providers.list",
      operation: "providers.list",
      timestamp: "2026-01-01T00:00:00.000Z",
    });
    const attrs = span.addEvent.mock.calls[0]![1] as Record<string, unknown>;
    expect(attrs).not.toHaveProperty("session.id");
  });
});

describe("recordApiFailure", () => {
  it("sets span status to ERROR", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordApiFailure(asSpan(span), new Error("boom"), auditLog, { operation: "sessions.list" });
    expect(span.setStatus).toHaveBeenCalledWith(expect.objectContaining({ code: 2 }));
  });

  it("records the exception on the span", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    const err = new Error("boom");
    recordApiFailure(asSpan(span), err, auditLog, { operation: "sessions.list" });
    expect(span.recordException).toHaveBeenCalledWith(err);
  });

  it("writes an error entry to the audit log", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordApiFailure(asSpan(span), new Error("fetch failed"), auditLog, {
      operation: "sessions.list",
      sessionId: "sess_x",
    });
    expect(auditLog.entries).toHaveLength(1);
    const [entry] = auditLog.entries;
    expect(entry?.level).toBe("error");
    expect(entry?.event).toBe("api.sessions.list.failed");
    expect(entry?.sessionId).toBe("sess_x");
    expect(entry?.detail?.message).toBe("fetch failed");
  });

  it("uses HTTP status code for ApiError in audit detail", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordApiFailure(asSpan(span), new ApiError(404, { code: "NOT_FOUND", message: "not found" }), auditLog, {
      operation: "sessions.get",
      sessionId: "sess_y",
    });
    expect(auditLog.entries[0]?.detail?.code).toBe("404");
  });

  it("handles non-Error failures gracefully", () => {
    const span = makeSpan();
    const auditLog = makeAuditLog();
    recordApiFailure(asSpan(span), "string error", auditLog, { operation: "providers.list" });
    expect(auditLog.entries[0]?.detail?.message).toBe("string error");
    expect(span.recordException).not.toHaveBeenCalled();
  });
});
