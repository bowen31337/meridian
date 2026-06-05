import type React from "react";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";

export interface CodeProps {
  /** Source code to display. */
  readonly code: string;
  /** Programming language identifier for syntax annotation (e.g. "python", "typescript"). */
  readonly language?: string;
  /** Optional filename displayed above the code block. */
  readonly filename?: string;
}

const MANIFEST = {
  kind: "meridian.code",
  version: "1.0.0",
  displayName: "Code Block",
  description: "Syntax-annotated code block. The host may apply a syntax highlighter.",
  propsSchema: {
    type: "object",
    required: ["code"],
    properties: {
      code: { type: "string" },
      language: { type: "string" },
      filename: { type: "string" },
    },
    additionalProperties: false,
  },
} as const;

function CodeWidgetImpl({ props }: WidgetProps<CodeProps>): React.ReactElement {
  const language = props.language as string | undefined;
  const filename = props.filename as string | undefined;

  return (
    <div data-widget-kind="meridian.code" style={{ fontFamily: "monospace" }}>
      {filename != null && (
        <div
          data-testid="code-filename"
          style={{
            fontSize: "0.75rem",
            padding: "4px 8px",
            borderBottom: "1px solid currentColor",
            opacity: 0.7,
          }}
        >
          {filename}
        </div>
      )}
      <pre style={{ margin: 0, padding: "12px", overflow: "auto" }} data-language={language}>
        <code>{props.code as string}</code>
      </pre>
    </div>
  );
}

export const CodeWidget = defineWidget<CodeProps>(CodeWidgetImpl, MANIFEST);
