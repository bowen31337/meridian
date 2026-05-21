import { useCallback } from "react";
import type { CanvasInteraction } from "@meridian/sdk-widget";
import type { CanvasInteractionRequest } from "../client.js";
import { createApiClient } from "../client.js";
import { useMeridianApi } from "../context.js";
import { getTracer, recordApiFailure, recordApiInvocationEvent } from "../telemetry.js";

function now(): string {
  return new Date().toISOString();
}

/**
 * Returns a stable callback that submits a canvas interaction (form submit or
 * button click) to the harness.  Emits an OTel span and writes to the audit
 * log on failure.
 */
export function useSubmitCanvasInteraction(sessionId: string) {
  const { baseUrl, auditLog } = useMeridianApi();

  return useCallback(
    async (interaction: CanvasInteraction): Promise<void> => {
      const tracer = getTracer();
      const timestamp = now();

      await tracer.startActiveSpan("api.canvas_interaction.submit", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.canvas_interaction.submit",
          operation: "canvas_interaction.submit",
          timestamp,
          sessionId,
        });

        try {
          const body: CanvasInteractionRequest = {
            kind: interaction.kind,
            widget_id: interaction.widget_id,
            widget_kind: interaction.widget_kind,
            payload: interaction.payload,
          };
          await createApiClient(baseUrl).submitCanvasInteraction(sessionId, body);
          span.end();
        } catch (err) {
          recordApiFailure(span, err, auditLog, {
            operation: "canvas_interaction.submit",
            sessionId,
          });
          span.end();
          throw err;
        }
      });
    },
    [baseUrl, auditLog, sessionId],
  );
}
