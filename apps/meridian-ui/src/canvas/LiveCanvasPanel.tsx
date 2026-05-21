import React, { useEffect, useMemo, useRef } from "react";
import type { AuditLog as WidgetAuditLog, AuditLogEntry as WidgetAuditLogEntry, CanvasOp } from "@meridian/sdk-widget";
import { defaultRegistry } from "@meridian/sdk-widget";
import { useMeridianApi } from "../api/context.js";
import { useListSessionEvents } from "../api/hooks/useSessionEvents.js";
import type { AuditLog } from "../workspace/audit.js";
import { getTracer, recordInvocationEvent } from "./telemetry.js";
import { applyCanvasOp, toContentBlock } from "./canvas-state.js";

function makeWidgetAuditLog(workspaceLog: AuditLog, fallbackSessionId: string): WidgetAuditLog {
  return {
    write(entry: WidgetAuditLogEntry): void {
      workspaceLog.write({
        level: entry.level,
        event: entry.event,
        sessionId: entry.session_id || fallbackSessionId,
        timestamp: entry.timestamp,
        detail: { ...entry.detail, widget_id: entry.widget_id, widget_kind: entry.widget_kind },
      });
    },
  };
}

export interface LiveCanvasPanelProps {
  readonly sessionId: string;
}

export function LiveCanvasPanel({ sessionId }: LiveCanvasPanelProps): React.ReactElement {
  const { auditLog } = useMeridianApi();
  const { data, isLoading, isError, error } = useListSessionEvents(sessionId);

  const widgetMap = useMemo(() => {
    if (!data) return new Map();
    return data.events
      .filter((e) => e.kind === "canvas_op")
      .reduce((state, event) => applyCanvasOp(state, event.payload as CanvasOp), new Map());
  }, [data]);

  // Emit OTel span + structured event each time the set of canvas_op events grows.
  const lastProcessedCountRef = useRef(0);
  useEffect(() => {
    if (!data) return;
    const canvasEvents = data.events.filter((e) => e.kind === "canvas_op");
    if (canvasEvents.length === lastProcessedCountRef.current) return;

    const tracer = getTracer();
    const timestamp = new Date().toISOString();
    tracer.startActiveSpan(
      "canvas_panel.process",
      { attributes: { "session.id": sessionId, "canvas_op.count": canvasEvents.length } },
      (span) => {
        recordInvocationEvent(span, { sessionId, timestamp, canvasOpCount: canvasEvents.length });
        span.end();
      },
    );
    lastProcessedCountRef.current = canvasEvents.length;
  }, [data, sessionId]);

  // Write audit log entry when event load fails.
  useEffect(() => {
    if (!isError || !error) return;
    const message = error instanceof Error ? error.message : String(error);
    auditLog.write({
      level: "error",
      event: "canvas_panel.load.failed",
      sessionId,
      timestamp: new Date().toISOString(),
      detail: { message },
    });
  }, [isError, error, auditLog, sessionId]);

  if (isLoading) {
    return <div data-testid="canvas-panel-loading">Loading...</div>;
  }

  if (isError) {
    const message = error instanceof Error ? error.message : "Failed to load canvas";
    return (
      <div role="alert" data-testid="canvas-panel-error">
        {message}
      </div>
    );
  }

  if (widgetMap.size === 0) {
    return <div data-testid="canvas-panel-empty">No canvas content yet.</div>;
  }

  const widgetAuditLog = makeWidgetAuditLog(auditLog, sessionId);

  return (
    <div data-testid="canvas-panel">
      {Array.from(widgetMap.values()).map((entry) => (
        <div key={entry.widget_id} data-widget-id={entry.widget_id}>
          {defaultRegistry.renderCanvasOp(toContentBlock(entry), { auditLog: widgetAuditLog })}
        </div>
      ))}
    </div>
  );
}
