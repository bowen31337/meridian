import type React from "react";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";

export interface MarkdownProps {
  /** Raw markdown source to display. */
  readonly content: string;
}

const MANIFEST = {
  kind: "meridian.markdown",
  version: "1.0.0",
  displayName: "Markdown",
  description:
    "Markdown-formatted content block. The host application may apply a markdown renderer.",
  propsSchema: {
    type: "object",
    required: ["content"],
    properties: {
      content: { type: "string" },
    },
    additionalProperties: false,
  },
} as const;

function MarkdownWidgetImpl({ props }: WidgetProps<MarkdownProps>): React.ReactElement {
  return (
    <div
      data-widget-kind="meridian.markdown"
      data-format="markdown"
      style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontFamily: "inherit" }}
    >
      {props.content as string}
    </div>
  );
}

export const MarkdownWidget = defineWidget<MarkdownProps>(MarkdownWidgetImpl, MANIFEST);
