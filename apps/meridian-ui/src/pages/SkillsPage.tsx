import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  Agent,
  ForgeProposal,
  Skill,
  SkillActivation,
} from "../api/client.js";
import { createApiClient } from "../api/client.js";
import { useMeridianApi } from "../api/context.js";
import { queryKeys } from "../api/query-keys.js";
import { getTracer, recordApiInvocationEvent } from "../api/telemetry.js";

// ---------------------------------------------------------------------------
// Installed Skills panel
// ---------------------------------------------------------------------------

function useListSkills() {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.skills.list(),
    queryFn: () => createApiClient(baseUrl).listSkills(),
  });
}

function useListSkillVersions(skillId: string | null) {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.skills.versions(skillId ?? ""),
    queryFn: () => createApiClient(baseUrl).listSkillVersions(skillId!),
    enabled: skillId !== null,
  });
}

function useListAgents() {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.agents.list(),
    queryFn: () => createApiClient(baseUrl).listAgents(),
  });
}

function useListAgentSkillActivations(agentId: string) {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.agents.skillActivations(agentId),
    queryFn: () => createApiClient(baseUrl).listAgentSkillActivations(agentId),
  });
}

// ---------------------------------------------------------------------------
// Forge Proposals panel
// ---------------------------------------------------------------------------

function useListForgeProposals() {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.forgeProposals.list(),
    queryFn: () => createApiClient(baseUrl).listForgeProposals(),
  });
}

// ---------------------------------------------------------------------------
// AgentActivationRow
// ---------------------------------------------------------------------------

function findLatestActivation(
  items: SkillActivation[],
  skillId: string,
): SkillActivation | null {
  const matches = items
    .filter((a) => a.skill_id === skillId)
    .sort((a, b) => b.requested_at.localeCompare(a.requested_at));
  return matches[0] ?? null;
}

function AgentActivationRow({
  agent,
  skillId,
}: {
  agent: Agent;
  skillId: string;
}) {
  const { baseUrl, auditLog } = useMeridianApi();
  const queryClient = useQueryClient();
  const { data, isLoading } = useListAgentSkillActivations(agent.id);

  const activation = data ? findLatestActivation(data.items, skillId) : null;
  const status = activation?.status ?? null;

  function invalidate() {
    queryClient.invalidateQueries({
      queryKey: queryKeys.agents.skillActivations(agent.id),
    });
  }

  const requestMutation = useMutation({
    mutationFn: () =>
      createApiClient(baseUrl).requestSkillActivation(agent.id, skillId),
    onSuccess: invalidate,
    onError: (err) => {
      const message = err instanceof Error ? err.message : "Request failed";
      auditLog.write({
        level: "error",
        event: "skill_registry.activation.request.failed",
        sessionId: "",
        timestamp: new Date().toISOString(),
        detail: { agent_id: agent.id, skill_id: skillId, message },
      });
    },
  });

  const approveMutation = useMutation({
    mutationFn: () =>
      createApiClient(baseUrl).approveSkillActivation(agent.id, skillId),
    onSuccess: invalidate,
    onError: (err) => {
      const message = err instanceof Error ? err.message : "Approve failed";
      auditLog.write({
        level: "error",
        event: "skill_registry.activation.approve.failed",
        sessionId: "",
        timestamp: new Date().toISOString(),
        detail: { agent_id: agent.id, skill_id: skillId, message },
      });
    },
  });

  const revokeMutation = useMutation({
    mutationFn: () =>
      createApiClient(baseUrl).revokeSkillActivation(agent.id, skillId),
    onSuccess: invalidate,
    onError: (err) => {
      const message = err instanceof Error ? err.message : "Revoke failed";
      auditLog.write({
        level: "error",
        event: "skill_registry.activation.revoke.failed",
        sessionId: "",
        timestamp: new Date().toISOString(),
        detail: { agent_id: agent.id, skill_id: skillId, message },
      });
    },
  });

  const mutationError =
    requestMutation.error ?? approveMutation.error ?? revokeMutation.error;

  if (isLoading) {
    return (
      <tr data-testid={`agent-activation-row-${agent.id}`}>
        <td colSpan={3}>Loading…</td>
      </tr>
    );
  }

  return (
    <tr data-testid={`agent-activation-row-${agent.id}`}>
      <td>{agent.name}</td>
      <td data-testid={`activation-status-${agent.id}`}>{status ?? "—"}</td>
      <td>
        {mutationError && (
          <span role="alert" data-testid={`activation-error-${agent.id}`}>
            {mutationError instanceof Error
              ? mutationError.message
              : "Action failed"}
          </span>
        )}
        {status === null && (
          <button
            onClick={() => requestMutation.mutate()}
            data-testid={`request-activation-${agent.id}`}
          >
            Request
          </button>
        )}
        {status === "pending" && (
          <>
            <button
              onClick={() => approveMutation.mutate()}
              data-testid={`approve-activation-${agent.id}`}
            >
              Approve
            </button>{" "}
            <button
              onClick={() => revokeMutation.mutate()}
              data-testid={`revoke-activation-${agent.id}`}
            >
              Revoke
            </button>
          </>
        )}
        {status === "active" && (
          <button
            onClick={() => revokeMutation.mutate()}
            data-testid={`revoke-activation-${agent.id}`}
          >
            Revoke
          </button>
        )}
        {status === "revoked" && (
          <button
            onClick={() => requestMutation.mutate()}
            data-testid={`request-activation-${agent.id}`}
          >
            Re-enable
          </button>
        )}
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// SkillDetailPanel — versions + agent activations for a selected skill
// ---------------------------------------------------------------------------

function SkillDetailPanel({ skill }: { skill: Skill }) {
  const { auditLog } = useMeridianApi();

  const {
    data: versionsData,
    isLoading: versionsLoading,
    isError: versionsError,
    error: versionsErr,
  } = useListSkillVersions(skill.id);

  const {
    data: agentsData,
    isLoading: agentsLoading,
    isError: agentsError,
    error: agentsErr,
  } = useListAgents();

  useEffect(() => {
    if (!versionsError || !versionsErr) return;
    const message =
      versionsErr instanceof Error ? versionsErr.message : "Failed to load versions";
    auditLog.write({
      level: "error",
      event: "skill_registry.versions.load.failed",
      sessionId: "",
      timestamp: new Date().toISOString(),
      detail: { skill_id: skill.id, message },
    });
  }, [versionsError, versionsErr, auditLog, skill.id]);

  useEffect(() => {
    if (!agentsError || !agentsErr) return;
    const message =
      agentsErr instanceof Error ? agentsErr.message : "Failed to load agents";
    auditLog.write({
      level: "error",
      event: "skill_registry.agents.load.failed",
      sessionId: "",
      timestamp: new Date().toISOString(),
      detail: { message },
    });
  }, [agentsError, agentsErr, auditLog]);

  const versions = versionsData?.items ?? [];
  const agents = agentsData?.items ?? [];

  return (
    <section data-testid="skill-detail-panel">
      <h2>
        {skill.name} — Details
      </h2>

      <h3>Version History</h3>
      {versionsLoading ? (
        <p data-testid="versions-loading">Loading versions…</p>
      ) : versionsError ? (
        <p role="alert" data-testid="versions-error">
          {versionsErr instanceof Error
            ? versionsErr.message
            : "Failed to load versions"}
        </p>
      ) : versions.length === 0 ? (
        <p data-testid="versions-empty">No versions found.</p>
      ) : (
        <table data-testid="versions-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Source</th>
              <th>Source Type</th>
              <th>Provenance</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((ver) => (
              <tr key={ver.id} data-testid={`version-row-${ver.id}`}>
                <td>v{ver.version_number}</td>
                <td>{ver.source}</td>
                <td>{ver.source_type}</td>
                <td>{ver.source_url ?? "—"}</td>
                <td>{ver.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Agent Activations</h3>
      {agentsLoading ? (
        <p data-testid="agents-loading">Loading agents…</p>
      ) : agentsError ? (
        <p role="alert" data-testid="agents-error">
          {agentsErr instanceof Error
            ? agentsErr.message
            : "Failed to load agents"}
        </p>
      ) : agents.length === 0 ? (
        <p data-testid="agents-empty">No agents configured.</p>
      ) : (
        <table data-testid="agent-activations-table">
          <thead>
            <tr>
              <th>Agent</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((agent) => (
              <AgentActivationRow
                key={agent.id}
                agent={agent}
                skillId={skill.id}
              />
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// InstalledSkillsPanel
// ---------------------------------------------------------------------------

function InstalledSkillsPanel({
  selectedSkillId,
  onSelectSkill,
}: {
  selectedSkillId: string | null;
  onSelectSkill: (id: string) => void;
}) {
  const { auditLog } = useMeridianApi();
  const { data, isLoading, isError, error } = useListSkills();
  const lastLoadedCountRef = useRef(0);

  useEffect(() => {
    if (!isError || !error) return;
    const message =
      error instanceof Error ? error.message : "Failed to load skills";
    auditLog.write({
      level: "error",
      event: "skill_registry.load.failed",
      sessionId: "",
      timestamp: new Date().toISOString(),
      detail: { message },
    });
  }, [isError, error, auditLog]);

  useEffect(() => {
    if (!data) return;
    if (data.items.length === lastLoadedCountRef.current) return;
    const tracer = getTracer();
    const timestamp = new Date().toISOString();
    tracer.startActiveSpan(
      "skill_registry.load",
      { attributes: { "skill_registry.skill.count": data.items.length } },
      (span) => {
        recordApiInvocationEvent(span, {
          name: "skill_registry.load.invocation",
          operation: "skill_registry.load",
          timestamp,
        });
        span.end();
      },
    );
    lastLoadedCountRef.current = data.items.length;
  }, [data]);

  if (isLoading) {
    return <p data-testid="skills-loading">Loading skills…</p>;
  }

  if (isError) {
    const msg =
      error instanceof Error ? error.message : "Failed to load skills";
    return (
      <p role="alert" data-testid="skills-error">
        {msg}
      </p>
    );
  }

  const skills = data?.items ?? [];

  if (skills.length === 0) {
    return <p data-testid="skills-empty">No skills installed.</p>;
  }

  const selectedSkill =
    skills.find((s) => s.id === selectedSkillId) ?? null;

  return (
    <div data-testid="installed-skills-panel">
      <table data-testid="skills-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Source</th>
            <th>Provenance</th>
            <th>Version</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {skills.map((skill) => (
            <tr
              key={skill.id}
              data-testid={`skill-row-${skill.id}`}
              aria-selected={skill.id === selectedSkillId}
            >
              <td>
                <button
                  onClick={() => onSelectSkill(skill.id)}
                  data-testid={`skill-select-${skill.id}`}
                  aria-expanded={skill.id === selectedSkillId}
                >
                  {skill.name}
                </button>
              </td>
              <td>{skill.description}</td>
              <td>{skill.version.source}</td>
              <td>{skill.version.source_url ?? "—"}</td>
              <td>v{skill.version.version_number}</td>
              <td>{skill.created_at}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {selectedSkill && <SkillDetailPanel skill={selectedSkill} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ProposalRow
// ---------------------------------------------------------------------------

function ProposalRow({
  proposal,
  onApprove,
  onReject,
}: {
  proposal: ForgeProposal;
  onApprove: () => void;
  onReject: (reason: string) => void;
}) {
  const [rejectMode, setRejectMode] = useState(false);
  const [rejectReason, setRejectReason] = useState("");

  function handleRejectSubmit() {
    onReject(rejectReason);
    setRejectMode(false);
    setRejectReason("");
  }

  return (
    <tr data-testid={`proposal-row-${proposal.id}`}>
      <td>{proposal.skill_id}</td>
      <td>{proposal.status}</td>
      <td>{proposal.tools.length} tool(s)</td>
      <td>{proposal.created_at}</td>
      <td>
        {rejectMode ? (
          <span>
            <input
              type="text"
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              placeholder="Rejection reason"
              data-testid={`reject-reason-${proposal.id}`}
            />
            <button
              onClick={handleRejectSubmit}
              data-testid={`confirm-reject-${proposal.id}`}
            >
              Confirm
            </button>{" "}
            <button onClick={() => setRejectMode(false)}>Cancel</button>
          </span>
        ) : (
          <>
            <button
              onClick={onApprove}
              data-testid={`approve-proposal-${proposal.id}`}
            >
              Approve
            </button>{" "}
            <button
              onClick={() => setRejectMode(true)}
              data-testid={`reject-proposal-${proposal.id}`}
            >
              Reject
            </button>
          </>
        )}
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// ForgeProposalsPanel
// ---------------------------------------------------------------------------

function ForgeProposalsPanel() {
  const { baseUrl, auditLog } = useMeridianApi();
  const queryClient = useQueryClient();
  const { data, isLoading, isError, error } = useListForgeProposals();

  useEffect(() => {
    if (!isError || !error) return;
    const message =
      error instanceof Error ? error.message : "Failed to load forge proposals";
    auditLog.write({
      level: "error",
      event: "skill_registry.forge_proposals.load.failed",
      sessionId: "",
      timestamp: new Date().toISOString(),
      detail: { message },
    });
  }, [isError, error, auditLog]);

  const approveMutation = useMutation({
    mutationFn: (proposalId: string) =>
      createApiClient(baseUrl).approveForgeProposal(proposalId),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.forgeProposals.list(),
      }),
    onError: (err) => {
      const message = err instanceof Error ? err.message : "Approve failed";
      auditLog.write({
        level: "error",
        event: "skill_registry.forge_proposal.approve.failed",
        sessionId: "",
        timestamp: new Date().toISOString(),
        detail: { message },
      });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: ({ proposalId, reason }: { proposalId: string; reason: string }) =>
      createApiClient(baseUrl).rejectForgeProposal(proposalId, reason),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.forgeProposals.list(),
      }),
    onError: (err) => {
      const message = err instanceof Error ? err.message : "Reject failed";
      auditLog.write({
        level: "error",
        event: "skill_registry.forge_proposal.reject.failed",
        sessionId: "",
        timestamp: new Date().toISOString(),
        detail: { message },
      });
    },
  });

  if (isLoading) {
    return <p data-testid="proposals-loading">Loading proposals…</p>;
  }

  if (isError) {
    const msg =
      error instanceof Error ? error.message : "Failed to load proposals";
    return (
      <p role="alert" data-testid="proposals-error">
        {msg}
      </p>
    );
  }

  const proposals = data?.items ?? [];

  if (proposals.length === 0) {
    return <p data-testid="proposals-empty">No pending forge proposals.</p>;
  }

  const actionError = approveMutation.error ?? rejectMutation.error;

  return (
    <div data-testid="proposals-panel">
      {actionError && (
        <p role="alert" data-testid="proposals-action-error">
          {actionError instanceof Error
            ? actionError.message
            : "Action failed"}
        </p>
      )}
      <table data-testid="proposals-table">
        <thead>
          <tr>
            <th>Skill</th>
            <th>Status</th>
            <th>Tools</th>
            <th>Created</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {proposals.map((proposal) => (
            <ProposalRow
              key={proposal.id}
              proposal={proposal}
              onApprove={() => approveMutation.mutate(proposal.id)}
              onReject={(reason) =>
                rejectMutation.mutate({ proposalId: proposal.id, reason })
              }
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SkillsPage
// ---------------------------------------------------------------------------

export function SkillsPage() {
  const [activeTab, setActiveTab] = useState<"skills" | "proposals">("skills");
  const [selectedSkillId, setSelectedSkillId] = useState<string | null>(null);

  function handleSelectSkill(id: string) {
    setSelectedSkillId((prev) => (prev === id ? null : id));
  }

  return (
    <div data-testid="skills-page">
      <h1>Skills</h1>
      <nav data-testid="skills-tabs">
        <button
          onClick={() => setActiveTab("skills")}
          aria-pressed={activeTab === "skills"}
          data-testid="tab-installed"
        >
          Installed
        </button>{" "}
        <button
          onClick={() => setActiveTab("proposals")}
          aria-pressed={activeTab === "proposals"}
          data-testid="tab-proposals"
        >
          Forge Proposals
        </button>
      </nav>
      {activeTab === "skills" ? (
        <InstalledSkillsPanel
          selectedSkillId={selectedSkillId}
          onSelectSkill={handleSelectSkill}
        />
      ) : (
        <ForgeProposalsPanel />
      )}
    </div>
  );
}
