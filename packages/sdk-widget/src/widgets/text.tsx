import type React from "react";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";

export interface TextProps {
  /** The text content to display. */
  readonly text: string;
}

const MANIFEST = {
  kind: "meridian.text",
  version: "1.0.0",
  displayName: "Text",
  description: "Plain-text paragraph.",
  propsSchema: {
    type: "object",
    required: ["text"],
    properties: {
      text: { type: "string" },
    },
    additionalProperties: false,
  },
} as const;

function TextWidgetImpl({ props }: WidgetProps<TextProps>): React.ReactElement {
  return (
    <p
      data-widget-kind="meridian.text"
      style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word" }}
    >
      {props.text as string}
    </p>
  );
}

export const TextWidget = defineWidget<TextProps>(TextWidgetImpl, MANIFEST);
