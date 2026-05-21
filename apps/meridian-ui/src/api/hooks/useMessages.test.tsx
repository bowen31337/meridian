import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NoopAuditLog } from "../../workspace/audit.js";
import type { AuditLogEntry } from "../../workspace/types.js";
import { MeridianApiProvider } from "../context.js";
import { useSendMessage } from "./useMessages.js";

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

const MESSAGE = {
  id: "msg1",
  session_id: "s1",
  thread_id: "t1",
  role: "user" as const,
  content: "Hello",
  sequence: 1,
  created_at: "2026-01-01T00:00:00Z",
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
  vi.clearAllMocks();
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useSendMessage", () => {
  it("calls POST /v1/sessions/:id/messages and returns the message", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(MESSAGE, 201));
    const { result } = renderHook(() => useSendMessage(), { wrapper: createWrapper() });

    result.current.mutate({ sessionId: "s1", body: { content: "Hello" } });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(MESSAGE);
    const url = vi.mocked(fetch).mock.calls[0]![0] as string;
    expect(url).toContain("/v1/sessions/s1/messages");
    const init = vi.mocked(fetch).mock.calls[0]![1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ content: "Hello" });
  });

  it("includes channel_id in the request body when provided", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(MESSAGE, 201));
    const { result } = renderHook(() => useSendMessage(), { wrapper: createWrapper() });

    result.current.mutate({ sessionId: "s1", body: { content: "Hi", channel_id: "ch1" } });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const init = vi.mocked(fetch).mock.calls[0]![1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ content: "Hi", channel_id: "ch1" });
  });

  it("records api.invocation span event with session.id", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(MESSAGE, 201));
    const { result } = renderHook(() => useSendMessage(), { wrapper: createWrapper() });

    result.current.mutate({ sessionId: "s1", body: { content: "Hello" } });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(mockSpan.addEvent).toHaveBeenCalledWith(
      "api.invocation",
      expect.objectContaining({
        "api.operation": "messages.send",
        "session.id": "s1",
      }),
    );
    expect(mockSpan.end).toHaveBeenCalled();
  });

  it("writes to audit log and sets isError on failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ code: "SERVER_ERROR", message: "boom" }, 500));
    const auditLog = makeAuditLog();
    const { result } = renderHook(() => useSendMessage(), { wrapper: createWrapper({ auditLog }) });

    result.current.mutate({ sessionId: "s1", body: { content: "Hello" } });
    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(auditLog.entries).toHaveLength(1);
    expect(auditLog.entries[0]?.event).toBe("api.messages.send.failed");
    expect(auditLog.entries[0]?.sessionId).toBe("s1");
    expect(mockSpan.setStatus).toHaveBeenCalledWith(expect.objectContaining({ code: 2 }));
  });
});
