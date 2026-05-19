import { useQuery } from "@tanstack/react-query";
import type { ListSessionEventsParams } from "../client.js";
import { createApiClient } from "../client.js";
import { useMeridianApi } from "../context.js";
import { queryKeys } from "../query-keys.js";
import { getTracer, recordApiFailure, recordApiInvocationEvent } from "../telemetry.js";

function now(): string {
  return new Date().toISOString();
}

export function useListSessionEvents(sessionId: string, params?: ListSessionEventsParams) {
  const { baseUrl, auditLog } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.sessions.events(sessionId, params),
    queryFn: () => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.session_events.list", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.session_events.list",
          operation: "session_events.list",
          timestamp: now(),
          sessionId,
        });
        try {
          const result = await createApiClient(baseUrl).listSessionEvents(sessionId, params);
          span.end();
          return result;
        } catch (err) {
          recordApiFailure(span, err, auditLog, {
            operation: "session_events.list",
            sessionId,
          });
          span.end();
          throw err;
        }
      });
    },
    enabled: !!sessionId,
  });
}
