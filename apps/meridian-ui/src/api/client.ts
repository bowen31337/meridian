import type { components, operations } from "@meridian/sdk-ts";

// Re-export SDK schema types for use throughout the app.
export type Session = components["schemas"]["Session"];
export type SessionList = components["schemas"]["SessionList"];
export type CreateSessionRequest = components["schemas"]["CreateSessionRequest"];
export type Provider = components["schemas"]["Provider"];
export type ProviderList = components["schemas"]["ProviderList"];
export type SessionEvent = components["schemas"]["SessionEvent"];
export type SessionEventKind = components["schemas"]["SessionEventKind"];
export type SessionEventList = components["schemas"]["SessionEventList"];
export type ErrorBody = components["schemas"]["ErrorResponse"];

export type ListSessionsParams = operations["listSessions"]["parameters"]["query"];
export type ListSessionEventsParams = operations["listSessionEvents"]["parameters"]["query"];

export class ApiError extends Error {
  readonly status: number;
  readonly body: ErrorBody | undefined;

  constructor(status: number, body: ErrorBody | undefined) {
    super(body?.message ?? `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${baseUrl}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    let body: ErrorBody | undefined;
    try {
      body = (await res.json()) as ErrorBody;
    } catch {
      // non-JSON error body — leave body undefined
    }
    throw new ApiError(res.status, body);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function toQuery(params?: Record<string, number | string | undefined>): string {
  if (!params) return "";
  const qs = Object.entries(params)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  return qs ? `?${qs}` : "";
}

export interface ApiClient {
  listSessions(params?: ListSessionsParams): Promise<SessionList>;
  createSession(body: CreateSessionRequest): Promise<Session>;
  getSession(sessionId: string): Promise<Session>;
  closeSession(sessionId: string): Promise<void>;
  listProviders(): Promise<ProviderList>;
  listSessionEvents(sessionId: string, params?: ListSessionEventsParams): Promise<SessionEventList>;
}

export function createApiClient(baseUrl: string): ApiClient {
  return {
    listSessions: (params) =>
      request<SessionList>(baseUrl, `/sessions${toQuery(params as Record<string, number | undefined>)}`),
    createSession: (body) =>
      request<Session>(baseUrl, "/sessions", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    getSession: (sessionId) => request<Session>(baseUrl, `/sessions/${sessionId}`),
    closeSession: (sessionId) =>
      request<void>(baseUrl, `/sessions/${sessionId}`, { method: "DELETE" }),
    listProviders: () => request<ProviderList>(baseUrl, "/providers"),
    listSessionEvents: (sessionId, params) =>
      request<SessionEventList>(
        baseUrl,
        `/sessions/${sessionId}/events${toQuery(params as Record<string, number | undefined>)}`,
      ),
  };
}
