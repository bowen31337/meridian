import { SpanStatusCode, trace } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/api";
import type { StructuredEvent, WorkspaceFailure } from "./types.js";

const TRACER_NAME = "meridian.workspace";
const TRACER_VERSION = "0.1.0";

export function getTracer() {
  return trace.getTracer(TRACER_NAME, TRACER_VERSION);
}

export function recordInvocationEvent(span: Span, event: StructuredEvent): void {
  span.addEvent("workspace.invocation", {
    "event.name": event.name,
    "session.id": event.sessionId,
    "workspace.operation": event.operation,
    "event.timestamp": event.timestamp,
  });
}

export function recordWorkspaceFailure(span: Span, failure: WorkspaceFailure): void {
  span.setStatus({ code: SpanStatusCode.ERROR, message: failure.message });
  span.addEvent("workspace.error", {
    "error.code": failure.code,
    "error.message": failure.message,
    "session.id": failure.sessionId,
  });
  if (failure.cause instanceof Error) {
    span.recordException(failure.cause);
  }
}
