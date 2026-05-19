import { SpanStatusCode, trace } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/api";
import type { AuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";
import { ApiError } from "./client.js";

const TRACER_NAME = "meridian.api";
const TRACER_VERSION = "0.1.0";

export function getTracer() {
  return trace.getTracer(TRACER_NAME, TRACER_VERSION);
}

export interface ApiInvocationEvent {
  readonly name: string;
  readonly operation: string;
  readonly timestamp: string;
  readonly sessionId?: string;
}

export function recordApiInvocationEvent(span: Span, event: ApiInvocationEvent): void {
  span.addEvent("api.invocation", {
    "event.name": event.name,
    "api.operation": event.operation,
    "event.timestamp": event.timestamp,
    ...(event.sessionId ? { "session.id": event.sessionId } : {}),
  });
}

export function recordApiFailure(
  span: Span,
  err: unknown,
  auditLog: AuditLog,
  context: { operation: string; sessionId?: string },
): void {
  const message = err instanceof Error ? err.message : String(err);
  const code = err instanceof ApiError ? String(err.status) : "API_ERROR";
  span.setStatus({ code: SpanStatusCode.ERROR, message });
  if (err instanceof Error) span.recordException(err);

  const entry: AuditLogEntry = {
    level: "error",
    event: `api.${context.operation}.failed`,
    sessionId: context.sessionId ?? "",
    timestamp: new Date().toISOString(),
    detail: { code, message },
  };
  auditLog.write(entry);
}
