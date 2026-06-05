import { ALL_WIDGETS, defaultRegistry } from "@meridian/sdk-widget";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import React from "react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { MeridianApiProvider } from "../api/context.js";
import { NoopAuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";
import { LiveCanvasPanel } from "./LiveCanvasPanel.js";

// ---------------------------------------------------------------------------
// OTel mock
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
        fn?: (span: typeof mockSpan) => unknown,
      ) => {
        const cb = typeof optsOrFn === "function" ? optsOrFn : fn;
        return (cb as (span: typeof mockSpan) => unknown)(mockSpan);
      },
    }),
  },
  SpanStatusCode: { UNSET: 0, OK: 1, ERROR: 2 },
}));

// ---------------------------------------------------------------------------
// Register built-in widgets once for the test module.
// ---------------------------------------------------------------------------
beforeAll(() => {
  for (const w of ALL_WIDGETS) {
    defaultRegistry.register(w);
  }
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAuditLog() {
  const entries: AuditLogEntry[] = [];
  return { write: (e: AuditLogEntry) => entries.push(e), entries };
}

function createWrapper(opts: { auditLog?: ReturnType<typeof makeAuditLog> } = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const auditLog = opts.auditLog ?? new NoopAuditLog();
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MeridianApiProvider baseUrl="http://api.test" auditLog={auditLog}>
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

function makeCanvasEvent(id: string, op: Record<string, unknown>): Record<string, unknown> {
  return {
    id,
    session_id: "sess1",
    kind: "canvas_op",
    payload: op,
    timestamp: "2026-05-21T00:00:00Z",
  };
}

function textOp(
  widgetId: string,
  text: string,
  seq: number,
  opKind = "set",
): Record<string, unknown> {
  return {
    op: opKind,
    widget_id: widgetId,
    widget_kind: "meridian.text",
    props: { text },
    sequence: seq,
    session_id: "sess1",
    timestamp: "2026-05-21T00:00:00Z",
  };
}

// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
  vi.clearAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
  cleanup();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LiveCanvasPanel — loading / error / empty", () => {
  it("shows loading state while fetch is pending", () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    expect(screen.getByTestId("canvas-panel-loading")).toBeTruthy();
  });

  it("shows empty state when there are no canvas_op events", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ events: [], total: 0 }));
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("canvas-panel-empty")).toBeTruthy());
  });

  it("shows error alert when API returns an error", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "NOT_FOUND", message: "Session not found" }, 404),
    );
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
  });

  it("writes canvas_panel.load.failed to audit log on fetch failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "SERVER_ERROR", message: "boom" }, 500),
    );
    const auditLog = makeAuditLog();
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper({ auditLog }) });
    await waitFor(() =>
      expect(auditLog.entries.some((e) => e.event === "canvas_panel.load.failed")).toBe(true),
    );
    const entry = auditLog.entries.find((e) => e.event === "canvas_panel.load.failed");
    expect(entry?.level).toBe("error");
    expect(entry?.sessionId).toBe("sess1");
  });
});

describe("LiveCanvasPanel — rendering widgets", () => {
  it("renders a text widget from a set canvas_op event", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ events: [makeCanvasEvent("e1", textOp("w1", "hello world", 1))], total: 1 }),
    );
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("canvas-panel")).toBeTruthy());
    expect(screen.getByText("hello world")).toBeTruthy();
  });

  it("renders multiple widgets in insertion order", async () => {
    const events = [
      makeCanvasEvent("e1", textOp("w1", "first", 1)),
      makeCanvasEvent("e2", textOp("w2", "second", 2)),
    ];
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ events, total: 2 }));
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByText("first")).toBeTruthy());
    expect(screen.getByText("second")).toBeTruthy();
  });

  it("sets data-widget-id on each widget wrapper", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ events: [makeCanvasEvent("e1", textOp("my-widget", "x", 1))], total: 1 }),
    );
    const { container } = render(<LiveCanvasPanel sessionId="sess1" />, {
      wrapper: createWrapper(),
    });
    await waitFor(() =>
      expect(container.querySelector("[data-widget-id='my-widget']")).toBeTruthy(),
    );
  });
});

describe("LiveCanvasPanel — state ops", () => {
  it("applies patch — merges props, shows updated text", async () => {
    const events = [
      makeCanvasEvent("e1", textOp("w1", "original", 1, "set")),
      makeCanvasEvent("e2", textOp("w1", "patched", 2, "patch")),
    ];
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ events, total: 2 }));
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByText("patched")).toBeTruthy());
    expect(screen.queryByText("original")).toBeNull();
  });

  it("applies clear — removes the widget", async () => {
    const events = [
      makeCanvasEvent("e1", textOp("w1", "visible", 1, "set")),
      makeCanvasEvent("e2", {
        op: "clear",
        widget_id: "w1",
        widget_kind: "meridian.text",
        props: {},
        sequence: 2,
        session_id: "sess1",
        timestamp: "t2",
      }),
    ];
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ events, total: 2 }));
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("canvas-panel-empty")).toBeTruthy());
    expect(screen.queryByText("visible")).toBeNull();
  });
});

describe("LiveCanvasPanel — OTel span", () => {
  it("emits canvas_panel.process span when canvas_op events arrive", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ events: [makeCanvasEvent("e1", textOp("w1", "hi", 1))], total: 1 }),
    );
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("canvas-panel")).toBeTruthy());
    expect(mockSpan.addEvent).toHaveBeenCalledWith(
      "canvas_panel.invocation",
      expect.objectContaining({ "session.id": "sess1", "canvas_op.count": 1 }),
    );
    expect(mockSpan.end).toHaveBeenCalled();
  });

  it("does not emit a span when there are no canvas_op events", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ events: [], total: 0 }));
    render(<LiveCanvasPanel sessionId="sess1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("canvas-panel-empty")).toBeTruthy());
    const panelSpanCalls = mockSpan.addEvent.mock.calls.filter(
      ([name]) => name === "canvas_panel.invocation",
    );
    expect(panelSpanCalls).toHaveLength(0);
  });
});
