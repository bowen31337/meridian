import React from "react";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";

export type FormFieldType = "text" | "number" | "email" | "url" | "checkbox" | "textarea";

export interface FormField {
  readonly name: string;
  readonly label: string;
  /** Defaults to "text" when omitted. */
  readonly type?: FormFieldType;
  readonly value?: unknown;
}

export interface FormProps {
  readonly fields: readonly FormField[];
  /** Optional title rendered above the fields. */
  readonly title?: string;
}

const MANIFEST = {
  kind: "meridian.form",
  version: "1.0.0",
  displayName: "Form",
  description: "Simple labelled-input form for collecting structured user input.",
  propsSchema: {
    type: "object",
    required: ["fields"],
    properties: {
      title: { type: "string" },
      fields: {
        type: "array",
        items: {
          type: "object",
          required: ["name", "label"],
          properties: {
            name: { type: "string" },
            label: { type: "string" },
            type: {
              type: "string",
              enum: ["text", "number", "email", "url", "checkbox", "textarea"],
            },
            value: {},
          },
          additionalProperties: false,
        },
      },
    },
    additionalProperties: false,
  },
} as const;

function renderField(field: FormField): React.ReactElement {
  const fieldType = field.type ?? "text";
  const id = `meridian-form-field-${field.name}`;
  const displayValue = field.value !== undefined ? String(field.value) : "";

  return (
    <div key={field.name} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label htmlFor={id} style={{ fontWeight: 500, fontSize: "0.875rem" }}>
        {field.label}
      </label>
      {fieldType === "textarea" ? (
        <textarea
          id={id}
          name={field.name}
          defaultValue={displayValue}
          rows={3}
          style={{ resize: "vertical", width: "100%", boxSizing: "border-box" }}
        />
      ) : fieldType === "checkbox" ? (
        <input
          id={id}
          type="checkbox"
          name={field.name}
          defaultChecked={Boolean(field.value)}
        />
      ) : (
        <input
          id={id}
          type={fieldType}
          name={field.name}
          defaultValue={displayValue}
          style={{ width: "100%", boxSizing: "border-box" }}
        />
      )}
    </div>
  );
}

function FormWidgetImpl({ props }: WidgetProps<FormProps>): React.ReactElement {
  const fields = (props.fields as FormField[]) ?? [];

  return (
    <form
      data-widget-kind="meridian.form"
      onSubmit={(e) => e.preventDefault()}
      style={{ display: "flex", flexDirection: "column", gap: 12 }}
    >
      {props.title != null && (
        <h3 style={{ margin: 0, fontSize: "1rem" }}>{props.title as string}</h3>
      )}
      {fields.map((field) => renderField(field))}
    </form>
  );
}

export const FormWidget = defineWidget<FormProps>(FormWidgetImpl, MANIFEST);
