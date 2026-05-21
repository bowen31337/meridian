import { useQuery } from "@tanstack/react-query";
import { createApiClient } from "../client.js";
import { useMeridianApi } from "../context.js";
import { queryKeys } from "../query-keys.js";
import { getTracer, recordApiFailure, recordApiInvocationEvent } from "../telemetry.js";

function now(): string {
  return new Date().toISOString();
}

export function useListChannels() {
  const { baseUrl, auditLog } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.channels.list(),
    queryFn: () => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.channels.list", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.channels.list",
          operation: "channels.list",
          timestamp: now(),
        });
        try {
          const result = await createApiClient(baseUrl).listChannels();
          span.end();
          return result;
        } catch (err) {
          recordApiFailure(span, err, auditLog, { operation: "channels.list" });
          span.end();
          throw err;
        }
      });
    },
  });
}
