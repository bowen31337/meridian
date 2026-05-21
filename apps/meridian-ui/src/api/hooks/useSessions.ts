import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ListSessionsParams, SessionCreateBody } from "../client.js";
import { createApiClient } from "../client.js";
import { useMeridianApi } from "../context.js";
import { queryKeys } from "../query-keys.js";
import { getTracer, recordApiFailure, recordApiInvocationEvent } from "../telemetry.js";

function now(): string {
  return new Date().toISOString();
}

export function useListSessions(params?: ListSessionsParams) {
  const { baseUrl, auditLog } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.sessions.list(params),
    queryFn: () => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.sessions.list", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.sessions.list",
          operation: "sessions.list",
          timestamp: now(),
        });
        try {
          const result = await createApiClient(baseUrl).listSessions(params);
          span.end();
          return result;
        } catch (err) {
          recordApiFailure(span, err, auditLog, { operation: "sessions.list" });
          span.end();
          throw err;
        }
      });
    },
  });
}

export function useGetSession(sessionId: string) {
  const { baseUrl, auditLog } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.sessions.detail(sessionId),
    queryFn: () => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.sessions.get", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.sessions.get",
          operation: "sessions.get",
          timestamp: now(),
          sessionId,
        });
        try {
          const result = await createApiClient(baseUrl).getSession(sessionId);
          span.end();
          return result;
        } catch (err) {
          recordApiFailure(span, err, auditLog, { operation: "sessions.get", sessionId });
          span.end();
          throw err;
        }
      });
    },
    enabled: !!sessionId,
  });
}

export function useCreateSession() {
  const { baseUrl, auditLog } = useMeridianApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: SessionCreateBody) => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.sessions.create", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.sessions.create",
          operation: "sessions.create",
          timestamp: now(),
        });
        try {
          const result = await createApiClient(baseUrl).createSession(body);
          span.end();
          return result;
        } catch (err) {
          recordApiFailure(span, err, auditLog, { operation: "sessions.create" });
          span.end();
          throw err;
        }
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions.all() });
    },
  });
}

export function useCloseSession() {
  const { baseUrl, auditLog } = useMeridianApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => {
      const tracer = getTracer();
      return tracer.startActiveSpan("api.sessions.close", async (span) => {
        recordApiInvocationEvent(span, {
          name: "api.sessions.close",
          operation: "sessions.close",
          timestamp: now(),
          sessionId,
        });
        try {
          await createApiClient(baseUrl).closeSession(sessionId);
          span.end();
        } catch (err) {
          recordApiFailure(span, err, auditLog, { operation: "sessions.close", sessionId });
          span.end();
          throw err;
        }
      });
    },
    onSuccess: (_data, sessionId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions.all() });
      queryClient.removeQueries({ queryKey: queryKeys.sessions.detail(sessionId) });
    },
  });
}
