import React from "react";
import type { AuditLogEntry } from "./types.js";

/** Write interface injected by the host application. All widget failures are recorded here. */
export interface AuditLog {
  write(entry: AuditLogEntry): void;
}

/** Fallback used when the host has not provided an AuditLogContext.Provider. */
const _noopAuditLog: AuditLog = {
  write: () => undefined,
};

/**
 * React context that carries the host-provided AuditLog implementation.
 * Wrap the canvas root with <AuditLogContext.Provider value={yourLog}>.
 */
export const AuditLogContext = React.createContext<AuditLog>(_noopAuditLog);

/** Convenience hook for reading the current audit log inside a function component. */
export function useAuditLog(): AuditLog {
  return React.useContext(AuditLogContext);
}
