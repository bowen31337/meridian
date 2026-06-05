import type React from "react";
import { useCallback, useState } from "react";
import { useAuditLog } from "../audit.js";
import { defineWidget } from "../contract.js";
import type { WidgetProps } from "../contract.js";
import { useInteraction } from "../interaction.js";
import { getTracer, recordInvocationEvent, recordWidgetFailure } from "../telemetry.js";

export type FormFieldType = "text" | "number" | "email" | "url" | "checkbox" | "textarea";

export interface FormField {
  readonly name: string;
  readonly label: string;
  /** Defaults to "text" when omitted. */
  readonly type?: FormFieldType;
  readonly value?: unknown;
}

/** An optional standalone action button rendered below the form fields. */
export interface FormAction {
  readonly name: string;
  readonly label: string;
}

export interface FormProps {
  readonly fields: readonly FormField[];
  /** Optional title rendered above the fields. */
  readonly title?: string;
  /** Optional action buttons; each click produces a button.click interaction. */
  readonly actions?: readonly FormAction[];
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
      actions: {
        type: "array",
        items: {
          type: "object",
          required: ["name", "label"],
          properties: {
            name: { type: "string" },
            label: { type: "string" },
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
        <input id={id} type="checkbox" name={field.name} defaultChecked={Boolean(field.value)} />
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

function FormWidgetImpl({ ctx, props }: WidgetProps<FormProps>): React.ReactElement {
  const fields = (props.fields as FormField[]) ?? [];
  const actions = (props.actions as FormAction[] | undefined) ?? [];

  const onInteraction = useInteraction();
  const auditLog = useAuditLog();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!onInteraction || isSubmitting) return;

      const formData = new FormData(e.currentTarget);
      const values: Record<string, unknown> = {};
      for (const field of fields) {
        const fieldType = field.type ?? "text";
        if (fieldType === "checkbox") {
          values[field.name] = formData.has(field.name);
        } else if (fieldType === "number") {
          const raw = formData.get(field.name) as string | null;
          values[field.name] = raw !== null && raw !== "" ? Number(raw) : null;
        } else {
          values[field.name] = (formData.get(field.name) as string) ?? "";
        }
      }

      const tracer = getTracer();
      const timestamp = new Date().toISOString();
      setIsSubmitting(true);
      setErrorMessage(null);

      await tracer.startActiveSpan(
        "form.submit",
        {
          attributes: {
            "widget.id": ctx.widgetId,
            "widget.kind": ctx.widgetKind,
            "session.id": ctx.sessionId,
            "widget.sequence": ctx.sequence,
          },
        },
        async (span) => {
          recordInvocationEvent(span, {
            name: "form.submit.invocation",
            widget_id: ctx.widgetId,
            widget_kind: ctx.widgetKind,
            session_id: ctx.sessionId,
            sequence: ctx.sequence,
            timestamp,
          });

          try {
            await onInteraction({
              kind: "form.submit",
              widget_id: ctx.widgetId,
              widget_kind: ctx.widgetKind,
              session_id: ctx.sessionId,
              sequence: ctx.sequence,
              timestamp,
              payload: { values },
            });
            span.end();
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            recordWidgetFailure(span, {
              code: "FORM_SUBMIT_FAILED",
              message,
              widget_id: ctx.widgetId,
              widget_kind: ctx.widgetKind,
              session_id: ctx.sessionId,
              timestamp: new Date().toISOString(),
              cause: err,
            });
            span.end();
            auditLog.write({
              level: "error",
              event: "form.submit.failed",
              widget_id: ctx.widgetId,
              widget_kind: ctx.widgetKind,
              session_id: ctx.sessionId,
              timestamp: new Date().toISOString(),
              detail: { message },
            });
            setErrorMessage(message);
          }
        },
      );

      setIsSubmitting(false);
    },
    [onInteraction, isSubmitting, fields, ctx, auditLog],
  );

  const handleActionClick = useCallback(
    async (action: FormAction) => {
      if (!onInteraction || isSubmitting) return;

      const tracer = getTracer();
      const timestamp = new Date().toISOString();
      setIsSubmitting(true);
      setErrorMessage(null);

      await tracer.startActiveSpan(
        "button.click",
        {
          attributes: {
            "widget.id": ctx.widgetId,
            "widget.kind": ctx.widgetKind,
            "session.id": ctx.sessionId,
            "button.name": action.name,
          },
        },
        async (span) => {
          recordInvocationEvent(span, {
            name: "button.click.invocation",
            widget_id: ctx.widgetId,
            widget_kind: ctx.widgetKind,
            session_id: ctx.sessionId,
            sequence: ctx.sequence,
            timestamp,
          });

          try {
            await onInteraction({
              kind: "button.click",
              widget_id: ctx.widgetId,
              widget_kind: ctx.widgetKind,
              session_id: ctx.sessionId,
              sequence: ctx.sequence,
              timestamp,
              payload: { action: action.name },
            });
            span.end();
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            recordWidgetFailure(span, {
              code: "BUTTON_CLICK_FAILED",
              message,
              widget_id: ctx.widgetId,
              widget_kind: ctx.widgetKind,
              session_id: ctx.sessionId,
              timestamp: new Date().toISOString(),
              cause: err,
            });
            span.end();
            auditLog.write({
              level: "error",
              event: "button.click.failed",
              widget_id: ctx.widgetId,
              widget_kind: ctx.widgetKind,
              session_id: ctx.sessionId,
              timestamp: new Date().toISOString(),
              detail: { message, action: action.name },
            });
            setErrorMessage(message);
          }
        },
      );

      setIsSubmitting(false);
    },
    [onInteraction, isSubmitting, ctx, auditLog],
  );

  const hasInteraction = onInteraction !== null;

  return (
    <form
      data-widget-kind="meridian.form"
      onSubmit={hasInteraction ? handleSubmit : (e) => e.preventDefault()}
      style={{ display: "flex", flexDirection: "column", gap: 12 }}
    >
      {props.title != null && (
        <h3 style={{ margin: 0, fontSize: "1rem" }}>{props.title as string}</h3>
      )}
      {fields.map((field) => renderField(field))}
      {errorMessage && (
        <div
          role="alert"
          data-testid="form-widget-error"
          style={{ color: "red", fontSize: "0.875rem" }}
        >
          {errorMessage}
        </div>
      )}
      {hasInteraction && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button type="submit" data-testid="form-submit-button" disabled={isSubmitting}>
            {isSubmitting ? "Submitting…" : "Submit"}
          </button>
          {actions.map((action) => (
            <button
              key={action.name}
              type="button"
              data-testid={`form-action-${action.name}`}
              disabled={isSubmitting}
              onClick={() => void handleActionClick(action)}
            >
              {action.label}
            </button>
          ))}
        </div>
      )}
    </form>
  );
}

export const FormWidget = defineWidget<FormProps>(FormWidgetImpl, MANIFEST);
