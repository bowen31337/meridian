import type React from "react";
import type { WidgetManifest } from "./manifest.js";

/** Runtime context injected into every widget by the render pipeline. */
export interface WidgetContext {
  readonly widgetId: string;
  readonly widgetKind: string;
  readonly sessionId: string;
  /** Monotonically-increasing sequence number for this widget instance. */
  readonly sequence: number;
}

/**
 * Props contract every widget component must accept.
 * `TProps` is the widget-specific props type validated against the manifest's propsSchema.
 */
export interface WidgetProps<TProps = Record<string, unknown>> {
  readonly ctx: WidgetContext;
  readonly props: TProps;
}

/**
 * The full contract a widget component must satisfy:
 * a React function component accepting WidgetProps<TProps> with a `manifest`
 * property attached for registration.
 */
export type WidgetComponent<TProps = Record<string, unknown>> = React.FC<WidgetProps<TProps>> & {
  readonly manifest: WidgetManifest;
};

/**
 * Helper that attaches a manifest to a React function component,
 * producing a WidgetComponent ready for registry registration.
 *
 * @example
 * const TextWidget = defineWidget(
 *   ({ props }: WidgetProps<{ text: string }>) => <p>{props.text}</p>,
 *   { kind: "meridian.text", version: "1.0.0", displayName: "Text", propsSchema: { ... } },
 * );
 */
export function defineWidget<TProps = Record<string, unknown>>(
  render: React.FC<WidgetProps<TProps>>,
  manifest: WidgetManifest,
): WidgetComponent<TProps> {
  return Object.assign(render, { manifest }) as WidgetComponent<TProps>;
}
