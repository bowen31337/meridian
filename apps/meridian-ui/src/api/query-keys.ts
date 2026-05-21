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
  skills: {
    all: () => ["skills"] as const,
    list: () => ["skills", "list"] as const,
    versions: (skillId: string) => ["skills", skillId, "versions"] as const,
  },
  agents: {
    all: () => ["agents"] as const,
    list: () => ["agents", "list"] as const,
    detail: (agentId: string) => ["agents", agentId] as const,
    versions: (agentId: string) => ["agents", agentId, "versions"] as const,
    skillActivations: (agentId: string) => ["agents", agentId, "skill-activations"] as const,
  },
  forgeProposals: {
    all: () => ["forge-proposals"] as const,
    list: () => ["forge-proposals", "list"] as const,
  },
} as const;
