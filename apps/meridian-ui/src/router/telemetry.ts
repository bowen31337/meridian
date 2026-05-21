import { SpanStatusCode, trace } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/api";
import type { AuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";

const TRACER_NAME = "meridian.ui.router";
const TRACER_VERSION = "0.1.0";

export function getRouteTracer() {
  return trace.getTracer(TRACER_NAME, TRACER_VERSION);
}

export interface RouteNavigationEvent {
  readonly name: string;
  readonly route: string;
  readonly timestamp: string;
}

export function recordRouteNavigationEvent(span: Span, event: RouteNavigationEvent): void {
  span.addEvent("ui.navigation", {
    "event.name": event.name,
    "route.path": event.route,
    "event.timestamp": event.timestamp,
  });
}

export function recordRouteFailure(
  span: Span,
  err: unknown,
  auditLog: AuditLog,
  context: { route: string },
): void {
  const message = err instanceof Error ? err.message : String(err);
  span.setStatus({ code: SpanStatusCode.ERROR, message });
  if (err instanceof Error) span.recordException(err);

  const entry: AuditLogEntry = {
    level: "error",
    event: "ui.route.failed",
    sessionId: "",
    timestamp: new Date().toISOString(),
    detail: { route: context.route, message },
  };
  auditLog.write(entry);
}
