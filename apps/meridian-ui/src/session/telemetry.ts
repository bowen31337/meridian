import { SpanStatusCode, trace } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/api";
import type { AuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";

const TRACER_NAME = "meridian.session_viewer";
const TRACER_VERSION = "0.1.0";

export function getTracer() {
  return trace.getTracer(TRACER_NAME, TRACER_VERSION);
}

export interface SessionViewerInvocationEvent {
  readonly sessionId: string;
  readonly timestamp: string;
  readonly eventCount: number;
}

export function recordInvocationEvent(span: Span, event: SessionViewerInvocationEvent): void {
  span.addEvent("session_viewer.invocation", {
    "session.id": event.sessionId,
    "event.count": event.eventCount,
    "event.timestamp": event.timestamp,
  });
}

export function recordStreamError(
  span: Span,
  err: unknown,
  auditLog: AuditLog,
  context: { sessionId: string },
): void {
  const message = err instanceof Error ? err.message : String(err);
  span.setStatus({ code: SpanStatusCode.ERROR, message });
  if (err instanceof Error) span.recordException(err);

  const entry: AuditLogEntry = {
    level: "error",
    event: "session_viewer.stream.failed",
    sessionId: context.sessionId,
    timestamp: new Date().toISOString(),
    detail: { message },
  };
  auditLog.write(entry);
}
