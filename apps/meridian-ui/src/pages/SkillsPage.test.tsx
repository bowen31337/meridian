import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MeridianApiProvider } from "../api/context.js";
import { NoopAuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";
import { SkillsPage } from "./SkillsPage.js";

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
// Test helpers
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

const SKILL_1 = {
  id: "skill_abc",
  name: "My Skill",
  description: "Does things",
  created_at: "2026-01-01T00:00:00Z",
  metadata: null,
  version: {
    id: "skillver_001",
    skill_id: "skill_abc",
    version_number: 1,
    instructions: "do the thing",
    tools: [{ name: "tool_a", description: null, input_schema: null }],
    created_at: "2026-01-01T00:00:00Z",
    source_type: "api",
    source_url: null,
    source: "authored",
    derived_from_session_ids: null,
  },
};

const SKILL_FORGE = {
  id: "skill_def",
  name: "Forge Skill",
  description: "Generated",
  created_at: "2026-01-02T00:00:00Z",
  metadata: null,
  version: {
    id: "skillver_002",
    skill_id: "skill_def",
    version_number: 2,
    instructions: "auto instructions",
    tools: [],
    created_at: "2026-01-02T00:00:00Z",
    source_type: "forge",
    source_url: null,
    source: "forge",
    derived_from_session_ids: ["sess_x"],
  },
};

const AGENT_1 = {
  id: "agent_aaa",
  name: "Alpha Agent",
  kind: "assistant",
  created_at: "2026-01-01T00:00:00Z",
};

const PROPOSAL_1 = {
  id: "prop_001",
  skill_id: "skill_abc",
  instructions: "new instructions",
  tools: [{ name: "t", description: null, input_schema: null }],
  derived_from_session_ids: ["sess_1"],
  status: "PROPOSAL",
  created_at: "2026-01-10T00:00:00Z",
};

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
// Loading / error / empty states for Installed Skills tab
// ---------------------------------------------------------------------------

describe("SkillsPage — installed tab loading/error/empty", () => {
  it("shows loading state while skills fetch is pending", () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));
    render(<SkillsPage />, { wrapper: createWrapper() });
    expect(screen.getByTestId("skills-loading")).toBeTruthy();
  });

  it("shows error alert when skills fetch fails", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "SERVER_ERROR", message: "db is down" }, 500),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-error")).toBeTruthy());
    expect(screen.getByTestId("skills-error").textContent).toContain("db is down");
  });

  it("writes skill_registry.load.failed to audit log on fetch failure", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ code: "SERVER_ERROR", message: "boom" }, 500),
    );
    const auditLog = makeAuditLog();
    render(<SkillsPage />, { wrapper: createWrapper({ auditLog }) });
    await waitFor(() =>
      expect(
        auditLog.entries.some((e) => e.event === "skill_registry.load.failed"),
      ).toBe(true),
    );
    const entry = auditLog.entries.find(
      (e) => e.event === "skill_registry.load.failed",
    );
    expect(entry?.level).toBe("error");
  });

  it("shows empty state when no skills are installed", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ items: [], next_cursor: null, limit: 20 }),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-empty")).toBeTruthy());
  });
});

// ---------------------------------------------------------------------------
// Skills list rendering
// ---------------------------------------------------------------------------

describe("SkillsPage — skills list", () => {
  it("renders skill rows with name, source, and version", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ items: [SKILL_1, SKILL_FORGE], next_cursor: null, limit: 20 }),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());
    expect(screen.getByTestId(`skill-row-${SKILL_1.id}`)).toBeTruthy();
    expect(screen.getByTestId(`skill-row-${SKILL_FORGE.id}`)).toBeTruthy();
    expect(screen.getByText("My Skill")).toBeTruthy();
    expect(screen.getByText("authored")).toBeTruthy();
    expect(screen.getByText("forge")).toBeTruthy();
    expect(screen.getByText("v1")).toBeTruthy();
    expect(screen.getByText("v2")).toBeTruthy();
  });

  it("shows — for provenance when source_url is null", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 }),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());
    // source_url is null → shows "—"
    const rows = screen.getByTestId(`skill-row-${SKILL_1.id}`);
    expect(rows.textContent).toContain("—");
  });
});

// ---------------------------------------------------------------------------
// Skill expansion → version history
// ---------------------------------------------------------------------------

describe("SkillsPage — skill detail / version history", () => {
  function mockSkillsAndVersions() {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.includes("/v1/skills/skill_abc/versions")) {
        return jsonResponse({
          items: [
            {
              id: "skillver_001",
              skill_id: "skill_abc",
              version_number: 1,
              instructions: "v1 instructions",
              tools: [],
              created_at: "2026-01-01T00:00:00Z",
              source_type: "api",
              source_url: null,
              source: "authored",
              derived_from_session_ids: null,
            },
          ],
          next_cursor: null,
          limit: 20,
        });
      }
      if (url.includes("/v1/agents")) {
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      }
      return jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 });
    });
  }

  it("shows skill detail panel when a skill is selected", async () => {
    mockSkillsAndVersions();
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());

    fireEvent.click(screen.getByTestId(`skill-select-${SKILL_1.id}`));

    await waitFor(() =>
      expect(screen.getByTestId("skill-detail-panel")).toBeTruthy(),
    );
  });

  it("shows version history table after expansion", async () => {
    mockSkillsAndVersions();
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());

    fireEvent.click(screen.getByTestId(`skill-select-${SKILL_1.id}`));

    await waitFor(() =>
      expect(screen.getByTestId("versions-table")).toBeTruthy(),
    );
    expect(screen.getByTestId("version-row-skillver_001")).toBeTruthy();
  });

  it("collapses detail panel when the same skill is clicked again", async () => {
    mockSkillsAndVersions();
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());

    const btn = screen.getByTestId(`skill-select-${SKILL_1.id}`);
    fireEvent.click(btn);
    await waitFor(() =>
      expect(screen.getByTestId("skill-detail-panel")).toBeTruthy(),
    );

    fireEvent.click(btn);
    await waitFor(() =>
      expect(screen.queryByTestId("skill-detail-panel")).toBeNull(),
    );
  });

  it("shows versions-error and writes to audit log when versions load fails", async () => {
    const auditLog = makeAuditLog();
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.includes("/v1/skills/skill_abc/versions")) {
        return jsonResponse({ code: "err", message: "versions broke" }, 500);
      }
      if (url.includes("/v1/agents")) {
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      }
      return jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 });
    });
    render(<SkillsPage />, { wrapper: createWrapper({ auditLog }) });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());

    fireEvent.click(screen.getByTestId(`skill-select-${SKILL_1.id}`));

    await waitFor(() =>
      expect(screen.getByTestId("versions-error")).toBeTruthy(),
    );
    await waitFor(() =>
      expect(
        auditLog.entries.some(
          (e) => e.event === "skill_registry.versions.load.failed",
        ),
      ).toBe(true),
    );
  });
});

// ---------------------------------------------------------------------------
// Agent activation toggles
// ---------------------------------------------------------------------------

describe("SkillsPage — agent activation toggle", () => {
  function mockWithAgentAndActivations(
    activations: object[] = [],
  ) {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.includes(`/v1/agents/${AGENT_1.id}/skills`) && !url.includes("/approve")) {
        if ((fetch as ReturnType<typeof vi.fn>).mock.calls.filter(
          (c: [RequestInfo | URL, ...unknown[]]) => c[0].toString().includes(`/v1/agents/${AGENT_1.id}/skills`)
        ).length <= 1) {
          return jsonResponse({
            items: activations,
            total: activations.length,
            limit: 20,
            offset: 0,
          });
        }
      }
      if (url.includes("/v1/agents")) {
        return jsonResponse({ items: [AGENT_1], next_cursor: null, limit: 20 });
      }
      if (url.includes("/v1/skills/skill_abc/versions")) {
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      }
      return jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 });
    });
  }

  it("shows agent row with Request button when no activation exists", async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.includes(`/v1/agents/${AGENT_1.id}/skills`)) {
        return jsonResponse({ items: [], total: 0, limit: 20, offset: 0 });
      }
      if (url.includes("/v1/agents")) {
        return jsonResponse({ items: [AGENT_1], next_cursor: null, limit: 20 });
      }
      if (url.includes("/v1/skills/skill_abc/versions")) {
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      }
      return jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 });
    });

    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());
    fireEvent.click(screen.getByTestId(`skill-select-${SKILL_1.id}`));

    await waitFor(() =>
      expect(
        screen.getByTestId(`agent-activation-row-${AGENT_1.id}`),
      ).toBeTruthy(),
    );
    await waitFor(() =>
      expect(
        screen.getByTestId(`request-activation-${AGENT_1.id}`),
      ).toBeTruthy(),
    );
  });

  it("shows Approve and Revoke buttons when activation is pending", async () => {
    const pendingActivation = {
      id: "skillact_001",
      agent_id: AGENT_1.id,
      skill_id: SKILL_1.id,
      skill_version_id: "skillver_001",
      status: "pending",
      requested_at: "2026-01-01T00:00:00Z",
      approved_at: null,
      revoked_at: null,
    };

    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.includes(`/v1/agents/${AGENT_1.id}/skills`)) {
        return jsonResponse({
          items: [pendingActivation],
          total: 1,
          limit: 20,
          offset: 0,
        });
      }
      if (url.includes("/v1/agents")) {
        return jsonResponse({ items: [AGENT_1], next_cursor: null, limit: 20 });
      }
      if (url.includes("/v1/skills/skill_abc/versions")) {
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      }
      return jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 });
    });

    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());
    fireEvent.click(screen.getByTestId(`skill-select-${SKILL_1.id}`));

    await waitFor(() =>
      expect(
        screen.getByTestId(`approve-activation-${AGENT_1.id}`),
      ).toBeTruthy(),
    );
    expect(screen.getByTestId(`revoke-activation-${AGENT_1.id}`)).toBeTruthy();
  });

  it("shows only Revoke button when activation is active", async () => {
    const activeActivation = {
      id: "skillact_001",
      agent_id: AGENT_1.id,
      skill_id: SKILL_1.id,
      skill_version_id: "skillver_001",
      status: "active",
      requested_at: "2026-01-01T00:00:00Z",
      approved_at: "2026-01-02T00:00:00Z",
      revoked_at: null,
    };

    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = input.toString();
      if (url.includes(`/v1/agents/${AGENT_1.id}/skills`)) {
        return jsonResponse({
          items: [activeActivation],
          total: 1,
          limit: 20,
          offset: 0,
        });
      }
      if (url.includes("/v1/agents")) {
        return jsonResponse({ items: [AGENT_1], next_cursor: null, limit: 20 });
      }
      if (url.includes("/v1/skills/skill_abc/versions")) {
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      }
      return jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 });
    });

    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());
    fireEvent.click(screen.getByTestId(`skill-select-${SKILL_1.id}`));

    await waitFor(() =>
      expect(
        screen.getByTestId(`revoke-activation-${AGENT_1.id}`),
      ).toBeTruthy(),
    );
    expect(
      screen.queryByTestId(`approve-activation-${AGENT_1.id}`),
    ).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Forge Proposals tab
// ---------------------------------------------------------------------------

describe("SkillsPage — forge proposals tab", () => {
  function switchToProposalsTab() {
    fireEvent.click(screen.getByTestId("tab-proposals"));
  }

  // Route fetch calls by URL so the initial skills fetch doesn't consume the proposals mock.
  function mockProposalsFetch(
    proposalsResponse: () => Response,
    extraHandlers: ((url: string, init?: RequestInit) => Response | null)[] = [],
  ) {
    vi.mocked(fetch).mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = input.toString();
        for (const handler of extraHandlers) {
          const result = handler(url, init);
          if (result !== null) return result;
        }
        if (url.includes("/v1/x/skill_forge/proposals")) {
          return proposalsResponse();
        }
        // Default: empty skills list for the Installed tab
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      },
    );
  }

  it("shows loading state for proposals", () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));
    render(<SkillsPage />, { wrapper: createWrapper() });
    switchToProposalsTab();
    expect(screen.getByTestId("proposals-loading")).toBeTruthy();
  });

  it("shows error alert when proposals fetch fails", async () => {
    mockProposalsFetch(() =>
      jsonResponse({ code: "SERVER_ERROR", message: "proposals down" }, 500),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    switchToProposalsTab();
    await waitFor(() =>
      expect(screen.getByTestId("proposals-error")).toBeTruthy(),
    );
    expect(screen.getByTestId("proposals-error").textContent).toContain(
      "proposals down",
    );
  });

  it("writes forge_proposals.load.failed to audit log on fetch failure", async () => {
    mockProposalsFetch(() =>
      jsonResponse({ code: "err", message: "fail" }, 500),
    );
    const auditLog = makeAuditLog();
    render(<SkillsPage />, { wrapper: createWrapper({ auditLog }) });
    switchToProposalsTab();
    await waitFor(() =>
      expect(
        auditLog.entries.some(
          (e) => e.event === "skill_registry.forge_proposals.load.failed",
        ),
      ).toBe(true),
    );
  });

  it("shows empty state when no proposals exist", async () => {
    mockProposalsFetch(() =>
      jsonResponse({ items: [], next_cursor: null, limit: 20 }),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    switchToProposalsTab();
    await waitFor(() =>
      expect(screen.getByTestId("proposals-empty")).toBeTruthy(),
    );
  });

  it("renders proposal rows with skill_id and status", async () => {
    mockProposalsFetch(() =>
      jsonResponse({ items: [PROPOSAL_1], next_cursor: null, limit: 20 }),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    switchToProposalsTab();
    await waitFor(() =>
      expect(screen.getByTestId("proposals-table")).toBeTruthy(),
    );
    expect(screen.getByTestId(`proposal-row-${PROPOSAL_1.id}`)).toBeTruthy();
    expect(screen.getByText(PROPOSAL_1.skill_id)).toBeTruthy();
    expect(screen.getByText("PROPOSAL")).toBeTruthy();
  });

  it("approve button calls approve endpoint and invalidates query", async () => {
    let proposalsFetchCount = 0;
    mockProposalsFetch(
      () => {
        proposalsFetchCount++;
        if (proposalsFetchCount === 1) {
          return jsonResponse({ items: [PROPOSAL_1], next_cursor: null, limit: 20 });
        }
        return jsonResponse({ items: [], next_cursor: null, limit: 20 });
      },
      [
        (url, init) => {
          if (
            url.includes(`/proposals/${PROPOSAL_1.id}/approve`) &&
            (init as RequestInit)?.method === "POST"
          ) {
            return jsonResponse({ id: "skillver_new" }, 200);
          }
          return null;
        },
      ],
    );

    render(<SkillsPage />, { wrapper: createWrapper() });
    switchToProposalsTab();
    await waitFor(() =>
      expect(screen.getByTestId(`approve-proposal-${PROPOSAL_1.id}`)).toBeTruthy(),
    );

    fireEvent.click(screen.getByTestId(`approve-proposal-${PROPOSAL_1.id}`));

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls;
      expect(
        calls.some(
          (c) =>
            (c[0] as string).includes(`/proposals/${PROPOSAL_1.id}/approve`) &&
            (c[1] as RequestInit)?.method === "POST",
        ),
      ).toBe(true);
    });
  });

  it("reject button shows reason input and confirm", async () => {
    mockProposalsFetch(() =>
      jsonResponse({ items: [PROPOSAL_1], next_cursor: null, limit: 20 }),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    switchToProposalsTab();
    await waitFor(() =>
      expect(screen.getByTestId(`reject-proposal-${PROPOSAL_1.id}`)).toBeTruthy(),
    );

    fireEvent.click(screen.getByTestId(`reject-proposal-${PROPOSAL_1.id}`));

    expect(screen.getByTestId(`reject-reason-${PROPOSAL_1.id}`)).toBeTruthy();
    expect(
      screen.getByTestId(`confirm-reject-${PROPOSAL_1.id}`),
    ).toBeTruthy();
  });

  it("confirm reject calls reject endpoint with reason", async () => {
    mockProposalsFetch(
      () => jsonResponse({ items: [PROPOSAL_1], next_cursor: null, limit: 20 }),
      [
        (url, init) => {
          if (url.includes(`/proposals/${PROPOSAL_1.id}/reject`)) {
            return jsonResponse({ ...PROPOSAL_1, status: "REJECTED" }, 200);
          }
          return null;
        },
      ],
    );

    render(<SkillsPage />, { wrapper: createWrapper() });
    switchToProposalsTab();
    await waitFor(() =>
      expect(screen.getByTestId(`reject-proposal-${PROPOSAL_1.id}`)).toBeTruthy(),
    );

    fireEvent.click(screen.getByTestId(`reject-proposal-${PROPOSAL_1.id}`));
    fireEvent.change(screen.getByTestId(`reject-reason-${PROPOSAL_1.id}`), {
      target: { value: "not useful" },
    });
    fireEvent.click(screen.getByTestId(`confirm-reject-${PROPOSAL_1.id}`));

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls;
      const rejectCall = calls.find((c) =>
        (c[0] as string).includes(`/proposals/${PROPOSAL_1.id}/reject`),
      );
      expect(rejectCall).toBeTruthy();
      const body = JSON.parse((rejectCall![1] as RequestInit).body as string);
      expect(body.reason).toBe("not useful");
    });
  });

  it("shows action error alert when approve fails", async () => {
    mockProposalsFetch(
      () => jsonResponse({ items: [PROPOSAL_1], next_cursor: null, limit: 20 }),
      [
        (url, init) => {
          if (
            url.includes(`/proposals/${PROPOSAL_1.id}/approve`) &&
            (init as RequestInit)?.method === "POST"
          ) {
            return jsonResponse({ code: "err", message: "approve failed" }, 500);
          }
          return null;
        },
      ],
    );

    const auditLog = makeAuditLog();
    render(<SkillsPage />, { wrapper: createWrapper({ auditLog }) });
    switchToProposalsTab();
    await waitFor(() =>
      expect(screen.getByTestId(`approve-proposal-${PROPOSAL_1.id}`)).toBeTruthy(),
    );

    fireEvent.click(screen.getByTestId(`approve-proposal-${PROPOSAL_1.id}`));

    await waitFor(() =>
      expect(screen.getByTestId("proposals-action-error")).toBeTruthy(),
    );
    expect(
      auditLog.entries.some(
        (e) => e.event === "skill_registry.forge_proposal.approve.failed",
      ),
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

describe("SkillsPage — tab navigation", () => {
  it("starts on the Installed tab", () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));
    render(<SkillsPage />, { wrapper: createWrapper() });
    expect(screen.getByTestId("tab-installed").getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByTestId("tab-proposals").getAttribute("aria-pressed")).toBe(
      "false",
    );
  });

  it("switches to Forge Proposals tab on click", () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));
    render(<SkillsPage />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByTestId("tab-proposals"));
    expect(screen.getByTestId("tab-proposals").getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByTestId("tab-installed").getAttribute("aria-pressed")).toBe(
      "false",
    );
  });
});

// ---------------------------------------------------------------------------
// OTel span
// ---------------------------------------------------------------------------

describe("SkillsPage — OTel span", () => {
  it("emits skill_registry.load.invocation span when skills load", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse({ items: [SKILL_1], next_cursor: null, limit: 20 }),
    );
    render(<SkillsPage />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("skills-table")).toBeTruthy());
    expect(mockSpan.addEvent).toHaveBeenCalledWith(
      "api.invocation",
      expect.objectContaining({ "api.operation": "skill_registry.load" }),
    );
    expect(mockSpan.end).toHaveBeenCalled();
  });
});
