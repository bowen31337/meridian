import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NoopAuditLog } from "../../workspace/audit.js";
import type { AuditLogEntry } from "../../workspace/types.js";
import { MeridianApiProvider } from "../context.js";
import { useListProviders } from "./useProviders.js";

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

function makeAuditLog() {
  const entries: AuditLogEntry[] = [];
  return { write: (e: AuditLogEntry) => entries.push(e), entries };
}

function createWrapper(opts: { auditLog?: ReturnType<typeof makeAuditLog> } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MeridianApiProvider
          baseUrl="http://api.test"
          auditLog={opts.auditLog ?? new NoopAuditLog()}
        >
          {children}
        </MeridianApiProvider>
      </QueryClientProvider>
    );
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
  vi.clearAllMocks();
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useListProviders", () => {
  it("returns provider list on success", async () => {
    const data = {
      providers: [{ kind: "anthropic", display_name: "Anthropic", models: ["claude-3"] }],
    };
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify(data), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const { result } = renderHook(() => useListProviders(), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(data);
  });

  it("records api.invocation span event", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify({ providers: [] }), { status: 200 }),
    );
    const { result } = renderHook(() => useListProviders(), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockSpan.addEvent).toHaveBeenCalledWith(
      "api.invocation",
      expect.objectContaining({ "api.operation": "providers.list" }),
    );
  });

  it("sets isError and writes to audit log on failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify({ code: "SERVER_ERROR", message: "fail" }), { status: 500 }),
    );
    const auditLog = makeAuditLog();
    const { result } = renderHook(() => useListProviders(), {
      wrapper: createWrapper({ auditLog }),
    });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(auditLog.entries[0]?.event).toBe("api.providers.list.failed");
  });
});
