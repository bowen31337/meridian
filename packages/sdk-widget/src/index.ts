// Types
export type {
  CanvasOp,
  CanvasOpKind,
  ContentBlock,
  ContentBlockCanvasOp,
  AuditLogEntry,
  StructuredEvent,
  WidgetError,
} from "./types.js";

// Manifest
export type { WidgetManifest, PropsValidationResult } from "./manifest.js";
export { validateProps } from "./manifest.js";

// Component contract
export type { WidgetComponent, WidgetContext, WidgetProps } from "./contract.js";
export { defineWidget } from "./contract.js";

// Audit log
export type { AuditLog } from "./audit.js";
export { AuditLogContext, useAuditLog } from "./audit.js";

// Telemetry
export { getTracer, recordInvocationEvent, recordWidgetFailure } from "./telemetry.js";

// Error surface
export { CanvasWidgetErrorBoundary, WidgetErrorDisplay } from "./error.js";

// Render pipeline
export type { RenderOptions } from "./pipeline.js";
export { WidgetRegistry, defaultRegistry } from "./pipeline.js";

// Layout helpers
export { Col, Divider, Grid, Row, Spacer, Stack } from "./layout.js";

// Built-in widget components
export {
  ALL_WIDGETS,
  CodeWidget,
  FormWidget,
  ImageWidget,
  MarkdownWidget,
  ProgressWidget,
  TableWidget,
  TextWidget,
} from "./widgets/index.js";
export type {
  CodeProps,
  FormField,
  FormFieldType,
  FormProps,
  ImageProps,
  MarkdownProps,
  ProgressProps,
  TableProps,
  TextProps,
} from "./widgets/index.js";
