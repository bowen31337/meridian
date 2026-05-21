import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { SendMessageRequest } from "../client.js";
import { createApiClient } from "../client.js";
import { useMeridianApi } from "../context.js";
import { queryKeys } from "../query-keys.js";
import { getTracer, recordApiFailure, recordApiInvocationEvent } from "../telemetry.js";

function now(): string {
  return new Date().toISOString();
}

export interface SendMessageArgs {
  sessionId: string;
  body: SendMessageRequest;
}

export function useSendMessage() {
  const { baseUrl, auditLog } = useMeridianApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, body }: SendMessageArgs) => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.messages.send", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.messages.send",
          operation: "messages.send",
          timestamp: now(),
          sessionId,
        });
        try {
          const result = await createApiClient(baseUrl).sendMessage(sessionId, body);
          span.end();
          return result;
        } catch (err) {
          recordApiFailure(span, err, auditLog, { operation: "messages.send", sessionId });
          span.end();
          throw err;
        }
      });
    },
    onSuccess: (_data, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.messages.all(sessionId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions.events(sessionId) });
    },
  });
}
