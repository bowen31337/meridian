import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MeridianApiProvider } from "../api/context.js";
import { NoopAuditLog } from "../workspace/audit.js";
import type { AuditLogEntry } from "../workspace/types.js";
import { ChatComposer } from "./ChatComposer.js";

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

function emptyList() {
  return jsonResponse({ items: [], next_cursor: null, limit: 20 });
}

const MESSAGE = {
  id: "msg1",
  session_id: "s1",
  thread_id: "t1",
  role: "user",
  content: "Hello",
  sequence: 1,
  created_at: "2026-01-01T00:00:00Z",
};

const SESSION = {
  id: "s1",
  status: "active",
  provider: "anthropic",
  model: "claude-3-5-sonnet",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const AGENT = { id: "a1", name: "Test Agent", kind: "system", created_at: "2026-01-01T00:00:00Z" };
const CHANNEL = {
  id: "ch1",
  kind: "slack",
  name: "Slack",
  status: "active",
  created_at: "2026-01-01T00:00:00Z",
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
  vi.clearAllMocks();
});
afterEach(() => {
  vi.unstubAllGlobals();
  cleanup();
});

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
describe("ChatComposer rendering", () => {
  it("renders textarea and send button", () => {
    vi.mocked(fetch).mockResolvedValue(emptyList());
    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });

    expect(screen.getByTestId("message-input")).toBeTruthy();
    expect(screen.getByTestId("send-button")).toBeTruthy();
  });

  it("hides agent picker when sessionId is provided", () => {
    vi.mocked(fetch).mockResolvedValue(emptyList());
    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });

    expect(screen.queryByTestId("agent-picker")).toBeNull();
  });

  it("shows agent picker when no sessionId is provided", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/v1/agents"))
        return Promise.resolve(jsonResponse({ items: [AGENT], next_cursor: null, limit: 20 }));
      return Promise.resolve(emptyList());
    });

    render(<ChatComposer />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("agent-picker")).toBeTruthy());
    await waitFor(() => expect(screen.getByText("Test Agent")).toBeTruthy());
  });

  it("shows channel picker when channels exist", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/v1/channels"))
        return Promise.resolve(jsonResponse({ items: [CHANNEL], next_cursor: null, limit: 20 }));
      return Promise.resolve(emptyList());
    });

    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });
    await waitFor(() => expect(screen.getByTestId("channel-picker")).toBeTruthy());
    await waitFor(() => expect(screen.getByText("Slack")).toBeTruthy());
  });

  it("hides channel picker when no channels exist", async () => {
    vi.mocked(fetch).mockResolvedValue(emptyList());
    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });

    await waitFor(() => expect(screen.queryByTestId("channel-picker")).toBeNull());
  });

  it("send button is disabled when content is empty", async () => {
    vi.mocked(fetch).mockResolvedValue(emptyList());
    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });

    const btn = screen.getByTestId("send-button") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Submit with existing session
// ---------------------------------------------------------------------------
describe("ChatComposer submit (existing session)", () => {
  it("sends message and calls onMessageSent callback", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/messages")) return Promise.resolve(jsonResponse(MESSAGE, 201));
      return Promise.resolve(emptyList());
    });

    const onMessageSent = vi.fn();
    render(<ChatComposer sessionId="s1" onMessageSent={onMessageSent} />, {
      wrapper: createWrapper(),
    });

    fireEvent.change(screen.getByTestId("message-input"), { target: { value: "Hello" } });
    fireEvent.submit(screen.getByTestId("chat-composer"));

    await waitFor(() => expect(onMessageSent).toHaveBeenCalledWith(MESSAGE));
  });

  it("clears textarea after successful send", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/messages")) return Promise.resolve(jsonResponse(MESSAGE, 201));
      return Promise.resolve(emptyList());
    });

    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });

    const textarea = screen.getByTestId("message-input") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "Hello" } });
    fireEvent.submit(screen.getByTestId("chat-composer"));

    await waitFor(() => expect(textarea.value).toBe(""));
  });

  it("emits chat_composer.invocation span event on submit", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/messages")) return Promise.resolve(jsonResponse(MESSAGE, 201));
      return Promise.resolve(emptyList());
    });

    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });
    fireEvent.change(screen.getByTestId("message-input"), { target: { value: "Hello" } });
    fireEvent.submit(screen.getByTestId("chat-composer"));

    await waitFor(() =>
      expect(mockSpan.addEvent).toHaveBeenCalledWith(
        "chat_composer.invocation",
        expect.objectContaining({
          "session.id": "s1",
          "message.content_length": 5,
        }),
      ),
    );
    expect(mockSpan.end).toHaveBeenCalled();
  });

  it("includes channel_id in the request body when a channel is selected", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/v1/channels"))
        return Promise.resolve(jsonResponse({ items: [CHANNEL], next_cursor: null, limit: 20 }));
      if (u.includes("/messages")) return Promise.resolve(jsonResponse(MESSAGE, 201));
      return Promise.resolve(emptyList());
    });

    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper() });

    await waitFor(() => expect(screen.getByTestId("channel-select")).toBeTruthy());
    fireEvent.change(screen.getByTestId("channel-select"), { target: { value: "ch1" } });
    fireEvent.change(screen.getByTestId("message-input"), { target: { value: "Hello" } });
    fireEvent.submit(screen.getByTestId("chat-composer"));

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls;
      const messageCall = calls.find((c) => (c[0] as string).includes("/messages"));
      expect(messageCall).toBeTruthy();
      const body = JSON.parse((messageCall?.[1] as RequestInit).body as string) as unknown;
      expect(body).toEqual(expect.objectContaining({ channel_id: "ch1" }));
    });
  });

  it("shows error message and writes to audit log on failure", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/messages"))
        return Promise.resolve(jsonResponse({ code: "SERVER_ERROR", message: "boom" }, 500));
      return Promise.resolve(emptyList());
    });

    const auditLog = makeAuditLog();
    render(<ChatComposer sessionId="s1" />, { wrapper: createWrapper({ auditLog }) });

    fireEvent.change(screen.getByTestId("message-input"), { target: { value: "Hello" } });
    fireEvent.submit(screen.getByTestId("chat-composer"));

    await waitFor(() => expect(screen.getByTestId("composer-error")).toBeTruthy());
    expect(screen.getByRole("alert").textContent).toContain("boom");
    expect(auditLog.entries.some((e) => e.event === "chat_composer.submit.failed")).toBe(true);
    expect(mockSpan.setStatus).toHaveBeenCalledWith(expect.objectContaining({ code: 2 }));
  });
});

// ---------------------------------------------------------------------------
// New session creation via agent picker
// ---------------------------------------------------------------------------
describe("ChatComposer new session creation", () => {
  it("creates session with selected agent then sends message", async () => {
    vi.mocked(fetch).mockImplementation((url: unknown) => {
      const u = url as string;
      if (u.includes("/v1/agents"))
        return Promise.resolve(jsonResponse({ items: [AGENT], next_cursor: null, limit: 20 }));
      if (u.endsWith("/sessions")) return Promise.resolve(jsonResponse(SESSION, 201));
      if (u.includes("/messages")) return Promise.resolve(jsonResponse(MESSAGE, 201));
      return Promise.resolve(emptyList());
    });

    const onSessionCreated = vi.fn();
    const onMessageSent = vi.fn();
    render(<ChatComposer onSessionCreated={onSessionCreated} onMessageSent={onMessageSent} />, {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(screen.getByTestId("agent-select")).toBeTruthy());
    fireEvent.change(screen.getByTestId("agent-select"), { target: { value: "a1" } });
    fireEvent.change(screen.getByTestId("message-input"), { target: { value: "Hello" } });
    fireEvent.submit(screen.getByTestId("chat-composer"));

    await waitFor(() => expect(onSessionCreated).toHaveBeenCalledWith(SESSION));
    await waitFor(() => expect(onMessageSent).toHaveBeenCalledWith(MESSAGE));
  });
});
