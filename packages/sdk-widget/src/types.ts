/** The four mutating operations a daemon can apply to a widget instance. */
export type CanvasOpKind = "set" | "patch" | "append" | "clear";

/** A single canvas operation emitted by the daemon in a content_block of type "canvas_op". */
export interface CanvasOp {
  readonly op: CanvasOpKind;
  /** Stable ID for the widget instance within the session. */
  readonly widget_id: string;
  /** Registered widget kind identifier, e.g. "meridian.text" or "acme.chart". */
  readonly widget_kind: string;
  /** Opaque props bag validated against the widget manifest's propsSchema. */
  readonly props: Record<string, unknown>;
  /** Monotonically-increasing sequence number within the session. */
  readonly sequence: number;
  readonly session_id: string;
  /** ISO 8601 timestamp from the daemon. */
  readonly timestamp: string;
}

/** A content block whose type discriminant is "canvas_op". */
export interface ContentBlockCanvasOp {
  readonly type: "canvas_op";
  readonly canvas_op: CanvasOp;
}

/** Union of all supported content block variants (extensible). */
export type ContentBlock = ContentBlockCanvasOp;

/** Structured description of a widget render failure, surfaced to callers and written to the audit log. */
export interface WidgetError {
  readonly code: string;
  readonly message: string;
  readonly widget_id: string;
  readonly widget_kind: string;
  readonly session_id: string;
  readonly timestamp: string;
  readonly cause?: unknown;
}

/** An append-only audit log entry written on every widget failure. */
export interface AuditLogEntry {
  readonly level: "info" | "warn" | "error";
  readonly event: string;
  readonly widget_id: string;
  readonly widget_kind: string;
  readonly session_id: string;
  readonly timestamp: string;
  readonly detail?: Record<string, unknown>;
}

/** The structured event attached to every OTel span, one per pipeline invocation. */
export interface StructuredEvent {
  readonly name: string;
  readonly widget_id: string;
  readonly widget_kind: string;
  readonly session_id: string;
  readonly sequence: number;
  readonly timestamp: string;
  readonly [key: string]: unknown;
}

/** Kind of user interaction originating from a canvas widget. */
export type CanvasInteractionKind = "form.submit" | "button.click";

/**
 * A user interaction event emitted by a canvas widget (form submission or
 * button click).  The harness surfaces these as new user-role messages so the
 * running session receives structured input from the UI.
 */
export interface CanvasInteraction {
  readonly kind: CanvasInteractionKind;
  readonly widget_id: string;
  readonly widget_kind: string;
  readonly session_id: string;
  readonly sequence: number;
  readonly timestamp: string;
  /** Interaction-specific data: form field values or button name. */
  readonly payload: Record<string, unknown>;
}
