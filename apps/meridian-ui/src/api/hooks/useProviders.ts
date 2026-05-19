import { useQuery } from "@tanstack/react-query";
import { createApiClient } from "../client.js";
import { useMeridianApi } from "../context.js";
import { queryKeys } from "../query-keys.js";
import { getTracer, recordApiFailure, recordApiInvocationEvent } from "../telemetry.js";

function now(): string {
  return new Date().toISOString();
}

export function useListProviders() {
  const { baseUrl, auditLog } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.providers.list(),
    queryFn: () => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.providers.list", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.providers.list",
          operation: "providers.list",
          timestamp: now(),
        });
        try {
          const result = await createApiClient(baseUrl).listProviders();
          span.end();
          return result;
        } catch (err) {
          recordApiFailure(span, err, auditLog, { operation: "providers.list" });
          span.end();
          throw err;
        }
      });
    },
    staleTime: 5 * 60 * 1000, // provider list is stable; re-fetch every 5 minutes
  });
}
