import { SpanStatusCode, trace } from "@opentelemetry/api";
import type { Attributes, Span } from "@opentelemetry/api";
import type { StructuredEvent, WidgetError } from "./types.js";
import { WIDGET_SDK_VERSION } from "./version.js";

const TRACER_NAME = "meridian.sdk-widget";

export function getTracer() {
  return trace.getTracer(TRACER_NAME, WIDGET_SDK_VERSION);
}

/**
 * Attaches a structured "widget.invocation" event to the active span.
 * Called once per pipeline invocation regardless of success or failure.
 */
export function recordInvocationEvent(span: Span, event: StructuredEvent): void {
  const attrs: Attributes = {};
  for (const [k, v] of Object.entries(event)) {
    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
      attrs[k] = v;
    }
  }
  span.addEvent("widget.invocation", attrs);
}

/**
 * Records a widget failure on the span: sets status to ERROR,
 * adds a "widget.error" event, and records the underlying exception if present.
 */
export function recordWidgetFailure(span: Span, error: WidgetError): void {
  span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
  span.addEvent("widget.error", {
    "widget.id": error.widget_id,
    "widget.kind": error.widget_kind,
    "session.id": error.session_id,
    "error.code": error.code,
    "error.message": error.message,
  });
  if (error.cause instanceof Error) {
    span.recordException(error.cause);
  }
}
