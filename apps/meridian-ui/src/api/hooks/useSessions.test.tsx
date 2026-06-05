import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NoopAuditLog } from "../../workspace/audit.js";
import type { AuditLogEntry } from "../../workspace/types.js";
import { MeridianApiProvider } from "../context.js";
import {
  useCloseSession,
  useCreateSession,
  useGetSession,
  useListSessions,
} from "./useSessions.js";

// ---------------------------------------------------------------------------
// OTel mock — pass-through span, no SDK bootstrap needed.
// ---------------------------------------------------------------------------
const mockSpan = {
  addEvent: vi.fn(),
  setStatus: vi.fn(),
  recordException: vi.fn(),
  end: vi.fn(),
};

vi.mock("@opentelemetry/api", () => ({
  trace: {
    getTracer: () => ({
      startActiveSpan: (
        _name: string,
        optsOrFn: unknown,
        fn?: (span: typeof mockSpan) => Promise<unknown>,
      ) => {
        const callback = typeof optsOrFn === "function" ? optsOrFn : fn;
        return (callback as (span: typeof mockSpan) => Promise<unknown>)(mockSpan);
      },
    }),
  },
  SpanStatusCode: { UNSET: 0, OK: 1, ERROR: 2 },
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface TestAuditLog {
  write(entry: AuditLogEntry): void;
  entries: AuditLogEntry[];
}

function makeAuditLog(): TestAuditLog {
  const entries: AuditLogEntry[] = [];
  return { write: (e) => entries.push(e), entries };
}

function createWrapper(opts: { auditLog?: TestAuditLog; baseUrl?: string } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const baseUrl = opts.baseUrl ?? "http://api.test";
  const auditLog = opts.auditLog ?? new NoopAuditLog();
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MeridianApiProvider baseUrl={baseUrl} auditLog={auditLog}>
          {children}
        </MeridianApiProvider>
      </QueryClientProvider>
    );
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SESSION = {
  id: "s1",
  status: "active" as const,
  provider: "anthropic",
  model: "claude-3",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
  vi.clearAllMocks();
});
afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// useListSessions
// ---------------------------------------------------------------------------

describe("useListSessions", () => {
  it("returns data on success", async () => {
    const data = { sessions: [SESSION], total: 1 };
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(data));

    const { result } = renderHook(() => useListSessions(), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(data);
  });

  it("records api.invocation span event on success", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ sessions: [], total: 0 }));
    const { result } = renderHook(() => useListSessions(), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockSpan.addEvent).toHaveBeenCalledWith(
      "api.invocation",
      expect.objectContaining({ "api.operation": "sessions.list" }),
    );
    expect(mockSpan.end).toHaveBeenCalled();
  });

  it("sets isError and writes to audit log on failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "SERVER_ERROR", message: "boom" }, 500),
    );
    const auditLog = makeAuditLog();

    const { result } = renderHook(() => useListSessions(), {
      wrapper: createWrapper({ auditLog }),
    });
    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(auditLog.entries).toHaveLength(1);
    expect(auditLog.entries[0]?.event).toBe("api.sessions.list.failed");
    expect(mockSpan.setStatus).toHaveBeenCalledWith(expect.objectContaining({ code: 2 }));
  });
});

// ---------------------------------------------------------------------------
// useGetSession
// ---------------------------------------------------------------------------

describe("useGetSession", () => {
  it("fetches the correct session", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SESSION));
    const { result } = renderHook(() => useGetSession("s1"), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(SESSION);
  });

  it("writes to audit log on 404", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "NOT_FOUND", message: "not found" }, 404),
    );
    const auditLog = makeAuditLog();
    const { result } = renderHook(() => useGetSession("missing"), {
      wrapper: createWrapper({ auditLog }),
    });
    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(auditLog.entries[0]?.event).toBe("api.sessions.get.failed");
    expect(auditLog.entries[0]?.sessionId).toBe("missing");
  });

  it("is disabled when sessionId is empty", () => {
    const { result } = renderHook(() => useGetSession(""), { wrapper: createWrapper() });
    expect(result.current.fetchStatus).toBe("idle");
  });
});

// ---------------------------------------------------------------------------
// useCreateSession
// ---------------------------------------------------------------------------

describe("useCreateSession", () => {
  it("calls POST /sessions and returns the new session", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SESSION, 201));
    const { result } = renderHook(() => useCreateSession(), { wrapper: createWrapper() });

    result.current.mutate({ provider: "anthropic", model: "claude-3" });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(SESSION);
  });

  it("writes to audit log on failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "BAD_REQUEST", message: "bad input" }, 400),
    );
    const auditLog = makeAuditLog();
    const { result } = renderHook(() => useCreateSession(), {
      wrapper: createWrapper({ auditLog }),
    });

    result.current.mutate({ provider: "bad", model: "bad" });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(auditLog.entries[0]?.event).toBe("api.sessions.create.failed");
  });
});

// ---------------------------------------------------------------------------
// useCloseSession
// ---------------------------------------------------------------------------

describe("useCloseSession", () => {
  it("calls DELETE /sessions/:id", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 204 }));
    const { result } = renderHook(() => useCloseSession(), { wrapper: createWrapper() });

    result.current.mutate("s1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const url = vi.mocked(fetch).mock.calls[0]?.[0] as string;
    expect(url).toContain("/sessions/s1");
  });

  it("writes to audit log on failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "NOT_FOUND", message: "not found" }, 404),
    );
    const auditLog = makeAuditLog();
    const { result } = renderHook(() => useCloseSession(), {
      wrapper: createWrapper({ auditLog }),
    });

    result.current.mutate("s_gone");
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(auditLog.entries[0]?.event).toBe("api.sessions.close.failed");
    expect(auditLog.entries[0]?.sessionId).toBe("s_gone");
  });
});
