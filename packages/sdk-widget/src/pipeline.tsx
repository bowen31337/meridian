import type { Span } from "@opentelemetry/api";
import type React from "react";
import { AuditLogContext } from "./audit.js";
import type { AuditLog } from "./audit.js";
import type { WidgetComponent, WidgetContext } from "./contract.js";
import { CanvasWidgetErrorBoundary, WidgetErrorDisplay } from "./error.js";
import { InteractionContext } from "./interaction.js";
import type { OnInteraction } from "./interaction.js";
import { validateProps } from "./manifest.js";
import { getTracer, recordInvocationEvent, recordWidgetFailure } from "./telemetry.js";
import type { ContentBlockCanvasOp, WidgetError } from "./types.js";

export interface RenderOptions {
  /**
   * AuditLog implementation supplied by the host. Pre-render failures are
   * written here synchronously; render-time failures go via the error boundary.
   */
  readonly auditLog: AuditLog;
  /** Called on pre-render failures (sync) and render-time failures (async via boundary). */
  readonly onError?: (error: WidgetError) => void;
  /**
   * Callback invoked when a widget emits a user interaction (form submit,
   * button click).  When omitted, interactive controls are not rendered.
   */
  readonly onInteraction?: OnInteraction;
}

function _failWithError(
  span: Span,
  options: RenderOptions,
  err: WidgetError,
  detail?: Record<string, unknown>,
): React.ReactElement {
  recordWidgetFailure(span, err);
  span.end();
  options.auditLog.write({
    level: "error",
    event: "widget.render.failed",
    widget_id: err.widget_id,
    widget_kind: err.widget_kind,
    session_id: err.session_id,
    timestamp: err.timestamp,
    detail: { code: err.code, message: err.message, ...detail },
  });
  options.onError?.(err);
  return <WidgetErrorDisplay error={err} />;
}

/**
 * Registry of widget kinds.
 * Register widgets once at app startup, then call renderCanvasOp for each
 * content_block.canvas_op received from the daemon SSE stream.
 */
export class WidgetRegistry {
  private readonly _widgets = new Map<string, WidgetComponent>();

  /**
   * Register a widget component by its manifest.kind.
   * Throws synchronously on duplicate registration — this is a programming error.
   */
  register(component: WidgetComponent): void {
    const { kind } = component.manifest;
    if (this._widgets.has(kind)) {
      throw new Error(`Widget kind "${kind}" is already registered`);
    }
    this._widgets.set(kind, component);
  }

  /** Returns the registered component for a kind, or undefined. */
  get(kind: string): WidgetComponent | undefined {
    return this._widgets.get(kind);
  }

  /**
   * Render pipeline for a single content_block.canvas_op.
   *
   * Per-invocation:
   *  1. Opens OTel span "widget.render" with widget/session attributes.
   *  2. Attaches a "widget.invocation" structured event to the span.
   *  3. Pre-render validation (kind lookup + props schema check).
   *     On failure: records span error, writes audit log, calls onError,
   *                 returns WidgetErrorDisplay.
   *  4. On success: ends the span, wraps the widget in a CanvasWidgetErrorBoundary
   *     that catches render-time failures via the same audit log + onError path.
   */
  renderCanvasOp(block: ContentBlockCanvasOp, options: RenderOptions): React.ReactElement {
    const { canvas_op: op } = block;
    const tracer = getTracer();

    return tracer.startActiveSpan(
      "widget.render",
      {
        attributes: {
          "widget.id": op.widget_id,
          "widget.kind": op.widget_kind,
          "session.id": op.session_id,
          "widget.sequence": op.sequence,
          "widget.op": op.op,
        },
      },
      (span): React.ReactElement => {
        const now = new Date().toISOString();

        // Emit structured event on every invocation regardless of outcome.
        recordInvocationEvent(span, {
          name: "widget.invocation",
          widget_id: op.widget_id,
          widget_kind: op.widget_kind,
          session_id: op.session_id,
          sequence: op.sequence,
          timestamp: now,
        });

        // --- pre-render check 1: kind must be registered ---
        const component = this._widgets.get(op.widget_kind);
        if (component === undefined) {
          return _failWithError(span, options, {
            code: "WIDGET_NOT_FOUND",
            message: `No widget registered for kind "${op.widget_kind}"`,
            widget_id: op.widget_id,
            widget_kind: op.widget_kind,
            session_id: op.session_id,
            timestamp: now,
          });
        }

        // --- pre-render check 2: props must satisfy the manifest schema ---
        const validation = validateProps(component.manifest, op.props);
        if (!validation.valid) {
          const errors = validation.errors ?? [];
          return _failWithError(
            span,
            options,
            {
              code: "WIDGET_PROPS_INVALID",
              message: `Props validation failed for "${op.widget_kind}": ${errors.join("; ")}`,
              widget_id: op.widget_id,
              widget_kind: op.widget_kind,
              session_id: op.session_id,
              timestamp: now,
            },
            { validationErrors: errors },
          );
        }

        span.end();

        const ctx: WidgetContext = {
          widgetId: op.widget_id,
          widgetKind: op.widget_kind,
          sessionId: op.session_id,
          sequence: op.sequence,
        };

        const Component = component;
        return (
          <AuditLogContext.Provider value={options.auditLog}>
            <InteractionContext.Provider value={options.onInteraction ?? null}>
              <CanvasWidgetErrorBoundary ctx={ctx} onError={options.onError}>
                <Component ctx={ctx} props={op.props} />
              </CanvasWidgetErrorBoundary>
            </InteractionContext.Provider>
          </AuditLogContext.Provider>
        );
      },
    );
  }
}

/** Module-level default registry for apps that don't need multiple registries. */
export const defaultRegistry = new WidgetRegistry();
