import type { AuditLogEntry } from "./types.js";

export interface AuditLog {
  write(entry: AuditLogEntry): void;
}

export class NoopAuditLog implements AuditLog {
  write(_entry: AuditLogEntry): void {}
}
