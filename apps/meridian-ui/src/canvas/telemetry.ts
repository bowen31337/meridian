import { trace } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/api";

const TRACER_NAME = "meridian.canvas_panel";
const TRACER_VERSION = "0.1.0";

export function getTracer() {
  return trace.getTracer(TRACER_NAME, TRACER_VERSION);
}

export interface CanvasPanelInvocationEvent {
  readonly sessionId: string;
  readonly timestamp: string;
  readonly canvasOpCount: number;
}

export function recordInvocationEvent(span: Span, event: CanvasPanelInvocationEvent): void {
  span.addEvent("canvas_panel.invocation", {
    "session.id": event.sessionId,
    "canvas_op.count": event.canvasOpCount,
    "event.timestamp": event.timestamp,
  });
}
