import type React from "react";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";

export interface ProgressProps {
  /** Current progress value (must be ≥ 0). */
  readonly value: number;
  /** Maximum value; defaults to 100. */
  readonly max?: number;
  /** Optional descriptive label shown above the bar. */
  readonly label?: string;
}

const MANIFEST = {
  kind: "meridian.progress",
  version: "1.0.0",
  displayName: "Progress Bar",
  description: "Labelled progress bar with configurable value and maximum.",
  propsSchema: {
    type: "object",
    required: ["value"],
    properties: {
      value: { type: "number", minimum: 0 },
      max: { type: "number", minimum: 0 },
      label: { type: "string" },
    },
    additionalProperties: false,
  },
} as const;

function ProgressWidgetImpl({ props }: WidgetProps<ProgressProps>): React.ReactElement {
  const value = props.value as number;
  const max = (props.max as number | undefined) ?? 100;
  const label = props.label as string | undefined;
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;

  return (
    <div
      data-widget-kind="meridian.progress"
      style={{ display: "flex", flexDirection: "column", gap: 4 }}
    >
      {label != null && (
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.875rem" }}>
          <span>{label}</span>
          <span aria-hidden="true">{pct}%</span>
        </div>
      )}
      <progress
        value={value}
        max={max}
        aria-label={label}
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        style={{ width: "100%", display: "block" }}
      />
    </div>
  );
}

export const ProgressWidget = defineWidget<ProgressProps>(ProgressWidgetImpl, MANIFEST);
