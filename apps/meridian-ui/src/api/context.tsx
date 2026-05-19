import { type ReactNode, createContext, useContext } from "react";
import type { AuditLog } from "../workspace/audit.js";
import { NoopAuditLog } from "../workspace/audit.js";

export interface MeridianApiContextValue {
  readonly baseUrl: string;
  readonly auditLog: AuditLog;
}

const MeridianApiContext = createContext<MeridianApiContextValue>({
  baseUrl: "",
  auditLog: new NoopAuditLog(),
});

export interface MeridianApiProviderProps {
  baseUrl: string;
  auditLog?: AuditLog;
  children: ReactNode;
}

export function MeridianApiProvider({ baseUrl, auditLog, children }: MeridianApiProviderProps) {
  const value: MeridianApiContextValue = {
    baseUrl,
    auditLog: auditLog ?? new NoopAuditLog(),
  };
  return <MeridianApiContext.Provider value={value}>{children}</MeridianApiContext.Provider>;
}

export function useMeridianApi(): MeridianApiContextValue {
  return useContext(MeridianApiContext);
}
