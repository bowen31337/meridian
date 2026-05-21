import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MeridianApiProvider } from "../api/context.js";
import { NoopAuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";
import { AgentDetailPage } from "./AgentDetailPage.js";

// ---------------------------------------------------------------------------
// Routing mock — useParams returns agent_001 for all tests
// ---------------------------------------------------------------------------
vi.mock("react-router-dom", async () => {
  const mod = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...mod, useParams: () => ({ id: "agent_001" }) };
});

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
// Test fixtures
// ---------------------------------------------------------------------------

const CURRENT_VERSION = {
  id: "agentver_current",
  agent_id: "agent_001",
  version_number: 2,
  name: "My Agent",
  kind: "assistant",
  config: {},
  capabilities: ["read:files", "write:memory"],
  instructions: "You are a helpful assistant.",
  model_routing: { default: "claude-sonnet-4-6" },
  skills: ["skill_abc"],
  tools: [{ name: "tool_a", description: null, input_schema: null }],
  default_environment_id: null,
  hooks: ["pre_message", "post_message"],
  budgets: { max_tokens: 100000 },
  memory_store_refs: ["mem://store1"],
  metadata: null,
  created_at: "2026-02-01T00:00:00Z",
};

const AGENT_DETAIL = {
  id: "agent_001",
  name: "My Agent",
  kind: "assistant",
  default_environment_id: null,
  created_at: "2026-01-01T00:00:00Z",
  version: CURRENT_VERSION,
};

const HISTORICAL_VERSION = {
  id: "agentver_old",
  agent_id: "agent_001",
  version_number: 1,
  name: "My Agent",
  kind: "assistant",
  config: {},
  capabilities: ["read:files"],
  instructions: "You are a helpful assistant.",
  model_routing: { default: "claude-haiku-4-5" },
  skills: [],
  tools: [],
  default_environment_id: null,
  hooks: [],
  budgets: {},
  memory_store_refs: [],
  metadata: null,
  created_at: "2026-01-01T00:00:00Z",
};

const VERSION_LIST = {
  items: [CURRENT_VERSION, HISTORICAL_VERSION],
  next_cursor: null,
  limit: 20,
};

const EMPTY_VERSION_LIST = { items: [], next_cursor: null, limit: 20 };

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

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
  vi.clearAllMocks();
});

afterEach(() => {
  vi.unstubAllGlobals();
  cleanup();
});

// ---------------------------------------------------------------------------
// Loading / error states
// ---------------------------------------------------------------------------

describe("AgentDetailPage — loading / error", () => {
  it("shows loading state while agent fetch is pending", () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    expect(screen.getByTestId("agent-loading")).toBeTruthy();
  });

  it("shows error alert when agent fetch fails", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "NOT_FOUND", message: "Agent not found" }, 404),
    );
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    expect(screen.getByTestId("agent-error").textContent).toContain("Agent not found");
  });

  it("writes agent.inspector.load.failed to audit log on agent fetch failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "SERVER_ERROR", message: "boom" }, 500),
    );
    const auditLog = makeAuditLog();
    render(<AgentDetailPage />, { wrapper: createWrapper({ auditLog }) });
    await waitFor(() =>
      expect(auditLog.entries.some((e) => e.event === "agent.inspector.load.failed")).toBe(true),
    );
    const entry = auditLog.entries.find((e) => e.event === "agent.inspector.load.failed")!;
    expect(entry.level).toBe("error");
    expect(entry.detail?.agent_id).toBe("agent_001");
    expect(entry.detail?.message).toContain("boom");
  });
});

// ---------------------------------------------------------------------------
// Agent metadata rendering
// ---------------------------------------------------------------------------

describe("AgentDetailPage — agent metadata", () => {
  beforeEach(() => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
  });

  it("renders the agent name as heading", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("agent-detail-page")).toBeTruthy());
    expect(screen.getByTestId("agent-name").textContent).toBe("My Agent");
  });

  it("renders agent id, kind, and created_at", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("agent-id")).toBeTruthy());
    expect(screen.getByTestId("agent-id").textContent).toBe("agent_001");
    expect(screen.getByTestId("agent-kind").textContent).toBe("assistant");
    expect(screen.getByTestId("agent-created").textContent).toBe("2026-01-01T00:00:00Z");
  });
});

// ---------------------------------------------------------------------------
// Current version panel
// ---------------------------------------------------------------------------

describe("AgentDetailPage — current version panel", () => {
  beforeEach(() => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
  });

  it("shows current version number and id", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("current-version-panel")).toBeTruthy());
    expect(screen.getByTestId("current-version-id").textContent).toBe("agentver_current");
  });

  it("renders model_routing as JSON", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("model-routing")).toBeTruthy());
    const text = screen.getByTestId("model-routing").textContent ?? "";
    expect(text).toContain("claude-sonnet-4-6");
  });

  it("renders capability grants list", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("capabilities-list")).toBeTruthy());
    expect(screen.getByTestId("capability-read:files")).toBeTruthy();
    expect(screen.getByTestId("capability-write:memory")).toBeTruthy();
  });

  it("renders hook bindings list", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("hooks-list")).toBeTruthy());
    expect(screen.getByTestId("hook-pre_message")).toBeTruthy();
    expect(screen.getByTestId("hook-post_message")).toBeTruthy();
  });

  it("renders budget config as JSON", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("budget-config")).toBeTruthy());
    const text = screen.getByTestId("budget-config").textContent ?? "";
    expect(text).toContain("100000");
  });

  it("renders memory store refs list", async () => {
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("memory-store-refs-list")).toBeTruthy());
    expect(screen.getByTestId("memory-store-ref-mem://store1")).toBeTruthy();
  });

  it("shows empty state messages when lists are empty", async () => {
    const agentWithEmptyVersion = {
      ...AGENT_DETAIL,
      version: {
        ...CURRENT_VERSION,
        capabilities: [],
        hooks: [],
        memory_store_refs: [],
      },
    };
    vi.mocked(fetch)
      .mockReset()
      .mockResolvedValueOnce(jsonResponse(agentWithEmptyVersion))
      .mockResolvedValueOnce(jsonResponse(EMPTY_VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("current-version-panel")).toBeTruthy());
    expect(screen.getByTestId("capabilities-empty")).toBeTruthy();
    expect(screen.getByTestId("hooks-empty")).toBeTruthy();
    expect(screen.getByTestId("memory-store-refs-empty")).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// Version history panel
// ---------------------------------------------------------------------------

describe("AgentDetailPage — version history", () => {
  it("shows loading state while versions fetch is pending", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockImplementationOnce(() => new Promise(() => {}));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-loading")).toBeTruthy());
  });

  it("shows version history table when versions load", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-table")).toBeTruthy());
    expect(screen.getByTestId(`version-row-${CURRENT_VERSION.id}`)).toBeTruthy();
    expect(screen.getByTestId(`version-row-${HISTORICAL_VERSION.id}`)).toBeTruthy();
  });

  it("marks the current version with a (current) badge", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId(`current-badge-${CURRENT_VERSION.id}`)).toBeTruthy());
  });

  it("shows empty state when no versions exist", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(EMPTY_VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-empty")).toBeTruthy());
  });

  it("shows error alert when versions fetch fails", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse({ code: "SERVER_ERROR", message: "db error" }, 500));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-error")).toBeTruthy());
    expect(screen.getByTestId("versions-error").textContent).toContain("db error");
  });

  it("writes agent.inspector.versions.load.failed to audit log on versions fetch failure", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse({ code: "SERVER_ERROR", message: "db error" }, 500));
    const auditLog = makeAuditLog();
    render(<AgentDetailPage />, { wrapper: createWrapper({ auditLog }) });
    await waitFor(() =>
      expect(
        auditLog.entries.some((e) => e.event === "agent.inspector.versions.load.failed"),
      ).toBe(true),
    );
    const entry = auditLog.entries.find(
      (e) => e.event === "agent.inspector.versions.load.failed",
    )!;
    expect(entry.level).toBe("error");
    expect(entry.detail?.agent_id).toBe("agent_001");
  });
});

// ---------------------------------------------------------------------------
// Version diff
// ---------------------------------------------------------------------------

describe("AgentDetailPage — version diff", () => {
  it("shows diff panel when a historical version is selected", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-table")).toBeTruthy());
    fireEvent.click(screen.getByTestId(`version-select-${HISTORICAL_VERSION.id}`));
    await waitFor(() => expect(screen.getByTestId("version-diff-panel")).toBeTruthy());
  });

  it("diff panel shows changed model_routing field", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-table")).toBeTruthy());
    fireEvent.click(screen.getByTestId(`version-select-${HISTORICAL_VERSION.id}`));
    await waitFor(() => expect(screen.getByTestId("diff-row-model_routing")).toBeTruthy());
    const selectedCell = screen.getByTestId("diff-selected-model_routing").textContent ?? "";
    const currentCell = screen.getByTestId("diff-current-model_routing").textContent ?? "";
    expect(selectedCell).toContain("claude-haiku-4-5");
    expect(currentCell).toContain("claude-sonnet-4-6");
  });

  it("diff panel marks changed fields with aria-label", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-table")).toBeTruthy());
    fireEvent.click(screen.getByTestId(`version-select-${HISTORICAL_VERSION.id}`));
    await waitFor(() => expect(screen.getByTestId("diff-row-model_routing")).toBeTruthy());
    expect(
      screen.getByTestId("diff-row-model_routing").getAttribute("aria-label"),
    ).toBe("model_routing changed");
    expect(
      screen.getByTestId("diff-row-instructions").getAttribute("aria-label"),
    ).toBe("instructions unchanged");
  });

  it("hides diff panel when same version is clicked again to deselect", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-table")).toBeTruthy());
    const btn = screen.getByTestId(`version-select-${HISTORICAL_VERSION.id}`);
    fireEvent.click(btn);
    await waitFor(() => expect(screen.getByTestId("version-diff-panel")).toBeTruthy());
    fireEvent.click(btn);
    await waitFor(() =>
      expect(screen.queryByTestId("version-diff-panel")).toBeNull(),
    );
  });

  it("does not show diff panel when the current version is selected", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("versions-table")).toBeTruthy());
    fireEvent.click(screen.getByTestId(`version-select-${CURRENT_VERSION.id}`));
    await waitFor(() => expect(screen.queryByTestId("version-diff-panel")).toBeNull());
  });
});

// ---------------------------------------------------------------------------
// OTel span
// ---------------------------------------------------------------------------

describe("AgentDetailPage — OTel span", () => {
  it("emits agent.inspector span when agent detail loads", async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(jsonResponse(AGENT_DETAIL))
      .mockResolvedValueOnce(jsonResponse(VERSION_LIST));
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("agent-detail-page")).toBeTruthy());
    expect(mockSpan.addEvent).toHaveBeenCalledWith(
      "api.invocation",
      expect.objectContaining({ "api.operation": "agent.inspector" }),
    );
    expect(mockSpan.end).toHaveBeenCalled();
  });

  it("does not emit span on load error", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "NOT_FOUND", message: "not found" }, 404),
    );
    render(<AgentDetailPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    const inspectorSpanCalls = mockSpan.addEvent.mock.calls.filter(
      ([, attrs]) => (attrs as Record<string, unknown>)?.["api.operation"] === "agent.inspector",
    );
    expect(inspectorSpanCalls).toHaveLength(0);
  });
});
