import React from "react";
import { AuditLogContext } from "./audit.js";
import type { AuditLog } from "./audit.js";
import type { WidgetContext } from "./contract.js";
import type { WidgetError } from "./types.js";

interface WidgetErrorDisplayProps {
  readonly error: WidgetError;
}

/**
 * Minimal error surface shown in place of a widget that failed to render.
 * The host application is responsible for styling; this component only conveys
 * the error message and widget identity via data attributes for test/debug access.
 */
export function WidgetErrorDisplay({ error }: WidgetErrorDisplayProps): React.ReactElement {
  return (
    <div
      role="alert"
      data-widget-id={error.widget_id}
      data-widget-kind={error.widget_kind}
      data-error-code={error.code}
    >
      <span>{error.message}</span>
    </div>
  );
}

interface ErrorBoundaryProps {
  readonly children: React.ReactNode;
  readonly ctx: WidgetContext;
  readonly onError?: (error: WidgetError) => void;
}

interface ErrorBoundaryState {
  caught: WidgetError | null;
}

/**
 * Class component error boundary (required by React's error boundary API)
 * wrapped around each widget in the render pipeline.
 *
 * On a render-time throw:
 *  1. Calls `onError` with a WidgetError so the caller can react.
 *  2. Writes an "error" entry to the AuditLog from context.
 *  3. Replaces the widget tree with WidgetErrorDisplay.
 */
export class CanvasWidgetErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  // contextType lets componentDidCatch access the audit log without prop-drilling.
  static override contextType = AuditLogContext;
  declare context: React.ContextType<typeof AuditLogContext>;

  override state: ErrorBoundaryState = { caught: null };

  // getDerivedStateFromError triggers the fallback render synchronously.
  static getDerivedStateFromError(cause: unknown): ErrorBoundaryState {
    return {
      caught: {
        code: "WIDGET_RENDER_ERROR",
        message: cause instanceof Error ? cause.message : String(cause),
        // ctx fields filled in during componentDidCatch via setState
        widget_id: "",
        widget_kind: "",
        session_id: "",
        timestamp: new Date().toISOString(),
        cause,
      },
    };
  }

  override componentDidCatch(cause: unknown, info: React.ErrorInfo): void {
    const { ctx, onError } = this.props;
    const auditLog = this.context as AuditLog;
    const now = new Date().toISOString();

    const widgetError: WidgetError = {
      code: "WIDGET_RENDER_ERROR",
      message: cause instanceof Error ? cause.message : String(cause),
      widget_id: ctx.widgetId,
      widget_kind: ctx.widgetKind,
      session_id: ctx.sessionId,
      timestamp: now,
      cause,
    };

    // Update state with the full error (replaces placeholder from getDerivedStateFromError).
    this.setState({ caught: widgetError });

    auditLog.write({
      level: "error",
      event: "widget.render.failed",
      widget_id: ctx.widgetId,
      widget_kind: ctx.widgetKind,
      session_id: ctx.sessionId,
      timestamp: now,
      detail: {
        code: widgetError.code,
        message: widgetError.message,
        componentStack: info.componentStack ?? undefined,
      },
    });

    onError?.(widgetError);
  }

  override render(): React.ReactNode {
    const { caught } = this.state;
    if (caught !== null) {
      return <WidgetErrorDisplay error={caught} />;
    }
    return this.props.children;
  }
}
