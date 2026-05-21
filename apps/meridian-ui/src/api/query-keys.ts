import type { ListSessionEventsParams, ListSessionsParams } from "./client.js";

export const queryKeys = {
  sessions: {
    all: () => ["sessions"] as const,
    list: (params?: ListSessionsParams) => ["sessions", "list", params ?? {}] as const,
    detail: (sessionId: string) => ["sessions", sessionId] as const,
    events: (sessionId: string, params?: ListSessionEventsParams) =>
      ["sessions", sessionId, "events", params ?? {}] as const,
  },
  providers: {
    all: () => ["providers"] as const,
    list: () => ["providers", "list"] as const,
  },
  vaults: {
    all: () => ["vaults"] as const,
    list: () => ["vaults", "list"] as const,
    secrets: (vaultId: string) => ["vaults", vaultId, "secrets"] as const,
  },
} as const;
