import type { AuditLog } from "./audit.js";
import { NoopAuditLog } from "./audit.js";
import { getTracer, recordInvocationEvent, recordWorkspaceFailure } from "./telemetry.js";
import type { AuditLogEntry, StructuredEvent } from "./types.js";
import { WorkspaceFailure } from "./types.js";

export interface WorkspaceConfig {
  apiBaseUrl: string;
  sessionId: string;
}

export interface WorkspaceOptions {
  auditLog?: AuditLog;
  onError?: (failure: WorkspaceFailure) => void;
}

function now(): string {
  return new Date().toISOString();
}

export class WorkspaceRuntime {
  private readonly auditLog: AuditLog;
  private readonly onError: ((failure: WorkspaceFailure) => void) | undefined;

  constructor(options: WorkspaceOptions = {}) {
    this.auditLog = options.auditLog ?? new NoopAuditLog();
    this.onError = options.onError;
  }

  private handleFailure(
    span: Parameters<typeof recordWorkspaceFailure>[0],
    failure: WorkspaceFailure,
    auditEvent: string,
  ): void {
    recordWorkspaceFailure(span, failure);
    const entry: AuditLogEntry = {
      level: "error",
      event: auditEvent,
      sessionId: failure.sessionId,
      timestamp: failure.timestamp,
      detail: { code: failure.code, message: failure.message },
    };
    this.auditLog.write(entry);
    this.onError?.(failure);
  }

  /**
   * Initialize the workspace for a session.
   *
   * Per-invocation:
   *   1. Opens OTel span "workspace.init" with session.id attribute.
   *   2. Attaches a "workspace.invocation" structured event.
   *   3. Validates config; wraps unexpected errors as WorkspaceFailure.
   *
   * On failure: sets span to ERROR, writes to audit log, calls onError, then re-throws.
   */
  async init(config: WorkspaceConfig, options?: WorkspaceOptions): Promise<void> {
    const auditLog = options?.auditLog ?? this.auditLog;
    const onError = options?.onError ?? this.onError;
    const timestamp = now();
    const tracer = getTracer();

    await tracer.startActiveSpan(
      "workspace.init",
      { attributes: { "session.id": config.sessionId } },
      async (span) => {
        const event: StructuredEvent = {
          name: "workspace.invocation",
          sessionId: config.sessionId,
          timestamp,
          operation: "init",
        };
        recordInvocationEvent(span, event);

        try {
          await this._validate(config);
          span.end();
        } catch (err) {
          const failure =
            err instanceof WorkspaceFailure
              ? err
              : new WorkspaceFailure({
                  code: "WORKSPACE_INIT_FAILED",
                  message: err instanceof Error ? err.message : String(err),
                  sessionId: config.sessionId,
                  timestamp,
                  cause: err,
                });

          const effectiveAuditLog = auditLog;
          const effectiveOnError = onError;

          recordWorkspaceFailure(span, failure);
          effectiveAuditLog.write({
            level: "error",
            event: "workspace.init.failed",
            sessionId: failure.sessionId,
            timestamp: failure.timestamp,
            detail: { code: failure.code, message: failure.message },
          });
          effectiveOnError?.(failure);
          span.end();
          throw failure;
        }
      },
    );
  }

  private async _validate(config: WorkspaceConfig): Promise<void> {
    if (!config.apiBaseUrl) {
      throw new WorkspaceFailure({
        code: "WORKSPACE_MISSING_API_URL",
        message: "apiBaseUrl is required for workspace initialization",
        sessionId: config.sessionId,
        timestamp: now(),
      });
    }
  }
}
