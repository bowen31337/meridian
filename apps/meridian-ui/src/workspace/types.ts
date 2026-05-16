export type AuditLogLevel = "info" | "warn" | "error";

export interface AuditLogEntry {
  readonly level: AuditLogLevel;
  readonly event: string;
  readonly sessionId: string;
  readonly timestamp: string;
  readonly detail?: Record<string, unknown>;
}

export interface StructuredEvent {
  readonly name: string;
  readonly sessionId: string;
  readonly timestamp: string;
  readonly operation: string;
}

export class WorkspaceFailure extends Error {
  readonly code: string;
  readonly sessionId: string;
  readonly timestamp: string;
  readonly cause: unknown;

  constructor(opts: {
    code: string;
    message: string;
    sessionId: string;
    timestamp: string;
    cause?: unknown;
  }) {
    super(opts.message);
    this.name = "WorkspaceFailure";
    this.code = opts.code;
    this.sessionId = opts.sessionId;
    this.timestamp = opts.timestamp;
    this.cause = opts.cause;
  }
}
