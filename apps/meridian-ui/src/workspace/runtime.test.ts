import { beforeEach, describe, expect, it, vi } from "vitest";
import type { WorkspaceOptions } from "./runtime.js";
import { WorkspaceRuntime } from "./runtime.js";
import type { AuditLogEntry } from "./types.js";
import { WorkspaceFailure } from "./types.js";

// ---------------------------------------------------------------------------
// OTel mock — tests supply a controllable span; no SDK bootstrap needed.
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
      startActiveSpan: async (
        _name: string,
        _opts: unknown,
        fn: (span: typeof mockSpan) => Promise<unknown>,
      ) => fn(mockSpan),
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

function makeRuntime(opts?: WorkspaceOptions): {
  runtime: WorkspaceRuntime;
  auditLog: TestAuditLog;
  errors: WorkspaceFailure[];
} {
  const auditLog = makeAuditLog();
  const errors: WorkspaceFailure[] = [];
  const runtime = new WorkspaceRuntime({
    auditLog,
    onError: (f) => errors.push(f),
    ...opts,
  });
  return { runtime, auditLog, errors };
}

const VALID_CONFIG = {
  apiBaseUrl: "https://api.meridian.example",
  sessionId: "sess_test_001",
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("WorkspaceRuntime.init", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("on success", () => {
    it("records workspace.invocation event with correct attributes", async () => {
      const { runtime } = makeRuntime();
      await runtime.init(VALID_CONFIG);
      expect(mockSpan.addEvent).toHaveBeenCalledWith(
        "workspace.invocation",
        expect.objectContaining({
          "event.name": "workspace.invocation",
          "session.id": VALID_CONFIG.sessionId,
          "workspace.operation": "init",
        }),
      );
    });

    it("ends the span", async () => {
      const { runtime } = makeRuntime();
      await runtime.init(VALID_CONFIG);
      expect(mockSpan.end).toHaveBeenCalled();
    });

    it("does not write to the audit log", async () => {
      const { runtime, auditLog } = makeRuntime();
      await runtime.init(VALID_CONFIG);
      expect(auditLog.entries).toHaveLength(0);
    });

    it("does not call onError", async () => {
      const { runtime, errors } = makeRuntime();
      await runtime.init(VALID_CONFIG);
      expect(errors).toHaveLength(0);
    });
  });

  describe("on failure (missing apiBaseUrl)", () => {
    const BAD_CONFIG = { apiBaseUrl: "", sessionId: "sess_x" };

    it("throws WorkspaceFailure", async () => {
      const { runtime } = makeRuntime();
      await expect(runtime.init(BAD_CONFIG)).rejects.toBeInstanceOf(WorkspaceFailure);
    });

    it("sets span status to ERROR", async () => {
      const { runtime } = makeRuntime();
      await expect(runtime.init(BAD_CONFIG)).rejects.toBeInstanceOf(WorkspaceFailure);
      expect(mockSpan.setStatus).toHaveBeenCalledWith(expect.objectContaining({ code: 2 }));
    });

    it("writes an error entry to the audit log", async () => {
      const { runtime, auditLog } = makeRuntime();
      await expect(runtime.init(BAD_CONFIG)).rejects.toBeInstanceOf(WorkspaceFailure);
      expect(auditLog.entries).toHaveLength(1);
      const [entry] = auditLog.entries;
      expect(entry?.level).toBe("error");
      expect(entry?.event).toBe("workspace.init.failed");
      expect(entry?.sessionId).toBe("sess_x");
    });

    it("calls onError with the WorkspaceFailure", async () => {
      const { runtime, errors } = makeRuntime();
      await expect(runtime.init(BAD_CONFIG)).rejects.toBeInstanceOf(WorkspaceFailure);
      expect(errors).toHaveLength(1);
      expect(errors[0]).toBeInstanceOf(WorkspaceFailure);
      expect(errors[0]?.code).toBe("WORKSPACE_MISSING_API_URL");
    });

    it("ends the span even when init fails", async () => {
      const { runtime } = makeRuntime();
      await expect(runtime.init(BAD_CONFIG)).rejects.toBeInstanceOf(WorkspaceFailure);
      expect(mockSpan.end).toHaveBeenCalled();
    });
  });

  describe("wraps unexpected errors as WorkspaceFailure", () => {
    it("wraps a plain Error thrown during _validate", async () => {
      const { runtime, auditLog } = makeRuntime();
      vi.spyOn(
        runtime as unknown as { _validate: () => Promise<void> },
        "_validate",
      ).mockRejectedValue(new Error("unexpected internal error"));

      const err = await runtime.init(VALID_CONFIG).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(WorkspaceFailure);
      expect((err as WorkspaceFailure).code).toBe("WORKSPACE_INIT_FAILED");
      expect((err as WorkspaceFailure).message).toContain("unexpected internal error");
      expect(auditLog.entries[0]?.level).toBe("error");
    });
  });
});
