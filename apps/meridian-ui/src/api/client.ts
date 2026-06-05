import type { components, operations } from "@meridian/sdk-ts";

// Re-export SDK schema types for use throughout the app.
export type Session = components["schemas"]["Session"];
export type SessionList = components["schemas"]["SessionList"];
export type CreateSessionRequest = components["schemas"]["CreateSessionRequest"];
export type Provider = components["schemas"]["Provider"];
export type ProviderList = components["schemas"]["ProviderList"];
export type SessionEvent = components["schemas"]["SessionEvent"];
export type SessionEventKind = components["schemas"]["SessionEventKind"];
export type SessionEventList = components["schemas"]["SessionEventList"];
export type ErrorBody = components["schemas"]["ErrorResponse"];

export type ListSessionsParams = operations["listSessions"]["parameters"]["query"];
export type ListSessionEventsParams = operations["listSessionEvents"]["parameters"]["query"];

export interface Vault {
  id: string;
  name: string;
  backend: "os_keychain" | "encrypted_file";
  created_at: string;
}

export interface VaultList {
  items: Vault[];
}

export interface SecretMeta {
  vault_id: string;
  key: string;
  created_at: string;
  last_accessed_at: string | null;
  requester_counts: Record<string, number>;
}

export interface SecretMetaList {
  items: SecretMeta[];
}

export interface SkillTool {
  name: string;
  description: string | null;
  input_schema: Record<string, unknown> | null;
}

export interface SkillVersion {
  id: string;
  skill_id: string;
  version_number: number;
  instructions: string;
  tools: SkillTool[];
  created_at: string;
  source_type: string;
  source_url: string | null;
  source: string;
  derived_from_session_ids: string[] | null;
}

export interface Skill {
  id: string;
  name: string;
  description: string;
  created_at: string;
  metadata: Record<string, unknown> | null;
  version: SkillVersion;
}

export interface SkillList {
  items: Skill[];
  next_cursor: string | null;
  limit: number;
}

export interface SkillVersionList {
  items: SkillVersion[];
  next_cursor: string | null;
  limit: number;
}

export interface Agent {
  id: string;
  name: string;
  kind: string;
  created_at: string;
}

export interface AgentList {
  items: Agent[];
  next_cursor: string | null;
  limit: number;
}

export interface SkillActivation {
  id: string;
  agent_id: string;
  skill_id: string;
  skill_version_id: string | null;
  status: "pending" | "active" | "revoked";
  requested_at: string;
  approved_at: string | null;
  revoked_at: string | null;
}

export interface SkillActivationList {
  items: SkillActivation[];
  total: number;
  limit: number;
  offset: number;
}

export interface AgentVersion {
  id: string;
  agent_id: string;
  version_number: number;
  name: string;
  kind: string;
  config: Record<string, unknown>;
  capabilities: string[];
  instructions: string;
  model_routing: Record<string, unknown>;
  skills: string[];
  tools: Array<{
    name: string;
    description?: string | null;
    input_schema?: Record<string, unknown> | null;
  }>;
  default_environment_id: string | null;
  hooks: string[];
  budgets: Record<string, unknown>;
  memory_store_refs: string[];
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface AgentVersionList {
  items: AgentVersion[];
  next_cursor: string | null;
  limit: number;
}

export interface AgentDetail {
  id: string;
  name: string;
  kind: string;
  default_environment_id: string | null;
  created_at: string;
  version: AgentVersion;
}

export interface ForgeProposal {
  id: string;
  skill_id: string;
  instructions: string;
  tools: SkillTool[];
  derived_from_session_ids: string[] | null;
  status: "PROPOSAL" | "PROMOTED" | "REJECTED";
  created_at: string;
}

export interface ForgeProposalList {
  items: ForgeProposal[];
  next_cursor: string | null;
  limit: number;
}

export interface Channel {
  id: string;
  kind: string;
  name: string;
  status: string;
  created_at: string;
}

export interface ChannelList {
  items: Channel[];
  next_cursor: string | null;
  limit: number;
}

export interface Message {
  id: string;
  session_id: string;
  thread_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sequence: number;
  created_at: string;
}

export interface SendMessageRequest {
  content: string;
  channel_id?: string;
}

export interface CanvasInteractionRequest {
  kind: "form.submit" | "button.click";
  widget_id: string;
  widget_kind: string;
  payload: Record<string, unknown>;
}

export interface CanvasInteractionResponse {
  interaction_id: string;
  session_id: string;
  kind: string;
  widget_id: string;
  widget_kind: string;
  payload: Record<string, unknown>;
  timestamp: string;
}

export type SessionCreateBody = CreateSessionRequest & { agent_id?: string };

export class ApiError extends Error {
  readonly status: number;
  readonly body: ErrorBody | undefined;

  constructor(status: number, body: ErrorBody | undefined) {
    super(body?.message ?? `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${baseUrl}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    let body: ErrorBody | undefined;
    try {
      body = (await res.json()) as ErrorBody;
    } catch {
      // non-JSON error body — leave body undefined
    }
    throw new ApiError(res.status, body);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function toQuery(params?: Record<string, number | string | undefined>): string {
  if (!params) return "";
  const qs = Object.entries(params)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  return qs ? `?${qs}` : "";
}

export interface ApiClient {
  listSessions(params?: ListSessionsParams): Promise<SessionList>;
  createSession(body: SessionCreateBody): Promise<Session>;
  getSession(sessionId: string): Promise<Session>;
  closeSession(sessionId: string): Promise<void>;
  listProviders(): Promise<ProviderList>;
  listSessionEvents(sessionId: string, params?: ListSessionEventsParams): Promise<SessionEventList>;
  listVaults(): Promise<VaultList>;
  listVaultSecrets(vaultId: string): Promise<SecretMetaList>;
  deleteVaultSecret(vaultId: string, name: string): Promise<void>;
  listSkills(): Promise<SkillList>;
  listSkillVersions(skillId: string): Promise<SkillVersionList>;
  listAgents(): Promise<AgentList>;
  getAgent(agentId: string): Promise<AgentDetail>;
  listAgentVersions(agentId: string): Promise<AgentVersionList>;
  listAgentSkillActivations(agentId: string): Promise<SkillActivationList>;
  requestSkillActivation(agentId: string, skillId: string): Promise<SkillActivation>;
  approveSkillActivation(agentId: string, skillId: string): Promise<SkillActivation>;
  revokeSkillActivation(agentId: string, skillId: string): Promise<SkillActivation>;
  listForgeProposals(): Promise<ForgeProposalList>;
  approveForgeProposal(proposalId: string): Promise<SkillVersion>;
  rejectForgeProposal(proposalId: string, reason: string): Promise<ForgeProposal>;
  listChannels(): Promise<ChannelList>;
  sendMessage(sessionId: string, body: SendMessageRequest): Promise<Message>;
  submitCanvasInteraction(
    sessionId: string,
    body: CanvasInteractionRequest,
  ): Promise<CanvasInteractionResponse>;
}

export function createApiClient(baseUrl: string): ApiClient {
  return {
    listSessions: (params) =>
      request<SessionList>(
        baseUrl,
        `/sessions${toQuery(params as Record<string, number | undefined>)}`,
      ),
    createSession: (body) =>
      request<Session>(baseUrl, "/sessions", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    getSession: (sessionId) => request<Session>(baseUrl, `/sessions/${sessionId}`),
    closeSession: (sessionId) =>
      request<void>(baseUrl, `/sessions/${sessionId}`, { method: "DELETE" }),
    listProviders: () => request<ProviderList>(baseUrl, "/providers"),
    listSessionEvents: (sessionId, params) =>
      request<SessionEventList>(
        baseUrl,
        `/sessions/${sessionId}/events${toQuery(params as Record<string, number | undefined>)}`,
      ),
    listVaults: () => request<VaultList>(baseUrl, "/v1/vaults"),
    listVaultSecrets: (vaultId) =>
      request<SecretMetaList>(baseUrl, `/v1/vaults/${encodeURIComponent(vaultId)}/secrets`),
    deleteVaultSecret: (vaultId, name) =>
      request<void>(
        baseUrl,
        `/v1/vaults/${encodeURIComponent(vaultId)}/secrets/${encodeURIComponent(name)}?confirm=true`,
        { method: "DELETE" },
      ),
    listSkills: () => request<SkillList>(baseUrl, "/v1/skills"),
    listSkillVersions: (skillId) =>
      request<SkillVersionList>(baseUrl, `/v1/skills/${encodeURIComponent(skillId)}/versions`),
    listAgents: () => request<AgentList>(baseUrl, "/v1/agents"),
    getAgent: (agentId) =>
      request<AgentDetail>(baseUrl, `/v1/agents/${encodeURIComponent(agentId)}`),
    listAgentVersions: (agentId) =>
      request<AgentVersionList>(baseUrl, `/v1/agents/${encodeURIComponent(agentId)}/versions`),
    listAgentSkillActivations: (agentId) =>
      request<SkillActivationList>(baseUrl, `/v1/agents/${encodeURIComponent(agentId)}/skills`),
    requestSkillActivation: (agentId, skillId) =>
      request<SkillActivation>(baseUrl, `/v1/agents/${encodeURIComponent(agentId)}/skills`, {
        method: "POST",
        body: JSON.stringify({ skill_id: skillId }),
      }),
    approveSkillActivation: (agentId, skillId) =>
      request<SkillActivation>(
        baseUrl,
        `/v1/agents/${encodeURIComponent(agentId)}/skills/${encodeURIComponent(skillId)}/approve`,
        { method: "POST" },
      ),
    revokeSkillActivation: (agentId, skillId) =>
      request<SkillActivation>(
        baseUrl,
        `/v1/agents/${encodeURIComponent(agentId)}/skills/${encodeURIComponent(skillId)}`,
        { method: "DELETE" },
      ),
    listForgeProposals: () => request<ForgeProposalList>(baseUrl, "/v1/x/skill_forge/proposals"),
    approveForgeProposal: (proposalId) =>
      request<SkillVersion>(
        baseUrl,
        `/v1/x/skill_forge/proposals/${encodeURIComponent(proposalId)}/approve`,
        { method: "POST" },
      ),
    rejectForgeProposal: (proposalId, reason) =>
      request<ForgeProposal>(
        baseUrl,
        `/v1/x/skill_forge/proposals/${encodeURIComponent(proposalId)}/reject`,
        { method: "POST", body: JSON.stringify({ reason }) },
      ),
    listChannels: () => request<ChannelList>(baseUrl, "/v1/channels"),
    sendMessage: (sessionId, body) =>
      request<Message>(baseUrl, `/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    submitCanvasInteraction: (sessionId, body) =>
      request<CanvasInteractionResponse>(
        baseUrl,
        `/v1/sessions/${encodeURIComponent(sessionId)}/canvas_interactions`,
        { method: "POST", body: JSON.stringify(body) },
      ),
  };
}
