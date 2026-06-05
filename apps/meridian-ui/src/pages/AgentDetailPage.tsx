import { useQuery } from "@tanstack/react-query";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import type { AgentDetail, AgentVersion } from "../api/client.js";
import { createApiClient } from "../api/client.js";
import { useMeridianApi } from "../api/context.js";
import { queryKeys } from "../api/query-keys.js";
import { getTracer, recordApiInvocationEvent } from "../api/telemetry.js";

function useGetAgent(agentId: string) {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.agents.detail(agentId),
    queryFn: () => createApiClient(baseUrl).getAgent(agentId),
  });
}

function useListAgentVersions(agentId: string) {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.agents.versions(agentId),
    queryFn: () => createApiClient(baseUrl).listAgentVersions(agentId),
  });
}

// ---------------------------------------------------------------------------
// Diff helpers
// ---------------------------------------------------------------------------

const DIFF_FIELDS = [
  "instructions",
  "model_routing",
  "capabilities",
  "tools",
  "skills",
  "hooks",
  "budgets",
  "memory_store_refs",
  "config",
] as const;

type DiffField = (typeof DIFF_FIELDS)[number];

interface DiffRow {
  field: DiffField;
  selectedVal: string;
  currentVal: string;
  changed: boolean;
}

function formatValue(val: unknown): string {
  if (typeof val === "string") return val || "(empty)";
  return JSON.stringify(val, null, 2);
}

function computeDiff(selected: AgentVersion, current: AgentVersion): DiffRow[] {
  return DIFF_FIELDS.map((field) => {
    const selectedVal = formatValue(selected[field as keyof AgentVersion]);
    const currentVal = formatValue(current[field as keyof AgentVersion]);
    return { field, selectedVal, currentVal, changed: selectedVal !== currentVal };
  });
}

// ---------------------------------------------------------------------------
// CurrentVersionPanel
// ---------------------------------------------------------------------------

function CurrentVersionPanel({ version }: { version: AgentVersion }) {
  return (
    <section data-testid="current-version-panel">
      <h2>Current Version (v{version.version_number})</h2>
      <dl data-testid="version-metadata">
        <dt>Version ID</dt>
        <dd data-testid="current-version-id">{version.id}</dd>
        <dt>Created</dt>
        <dd data-testid="current-version-created">{version.created_at}</dd>
      </dl>

      <h3>Model Routing</h3>
      <pre data-testid="model-routing">{JSON.stringify(version.model_routing, null, 2)}</pre>

      <h3>Capability Grants</h3>
      {version.capabilities.length === 0 ? (
        <p data-testid="capabilities-empty">No capabilities granted.</p>
      ) : (
        <ul data-testid="capabilities-list">
          {version.capabilities.map((cap) => (
            <li key={cap} data-testid={`capability-${cap}`}>
              {cap}
            </li>
          ))}
        </ul>
      )}

      <h3>Hook Bindings</h3>
      {version.hooks.length === 0 ? (
        <p data-testid="hooks-empty">No hooks bound.</p>
      ) : (
        <ul data-testid="hooks-list">
          {version.hooks.map((hook) => (
            <li key={hook} data-testid={`hook-${hook}`}>
              {hook}
            </li>
          ))}
        </ul>
      )}

      <h3>Budget Config</h3>
      <pre data-testid="budget-config">{JSON.stringify(version.budgets, null, 2)}</pre>

      <h3>Memory Store Refs</h3>
      {version.memory_store_refs.length === 0 ? (
        <p data-testid="memory-store-refs-empty">No memory stores referenced.</p>
      ) : (
        <ul data-testid="memory-store-refs-list">
          {version.memory_store_refs.map((ref) => (
            <li key={ref} data-testid={`memory-store-ref-${ref}`}>
              {ref}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// VersionDiffPanel
// ---------------------------------------------------------------------------

function VersionDiffPanel({
  current,
  selected,
}: {
  current: AgentVersion;
  selected: AgentVersion;
}) {
  const rows = computeDiff(selected, current);
  const changedCount = rows.filter((r) => r.changed).length;

  return (
    <section data-testid="version-diff-panel">
      <h3>
        Diff: v{selected.version_number} → v{current.version_number} ({changedCount} field
        {changedCount === 1 ? "" : "s"} changed)
      </h3>
      <table data-testid="diff-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>v{selected.version_number}</th>
            <th>Current (v{current.version_number})</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ field, selectedVal, currentVal, changed }) => (
            <tr
              key={field}
              data-testid={`diff-row-${field}`}
              aria-label={changed ? `${field} changed` : `${field} unchanged`}
            >
              <td>{field}</td>
              <td>
                <pre data-testid={`diff-selected-${field}`}>{selectedVal}</pre>
              </td>
              <td>
                <pre data-testid={`diff-current-${field}`}>{currentVal}</pre>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

// ---------------------------------------------------------------------------
// VersionHistoryPanel
// ---------------------------------------------------------------------------

function VersionHistoryPanel({
  agentId,
  currentVersion,
}: {
  agentId: string;
  currentVersion: AgentVersion;
}) {
  const { auditLog } = useMeridianApi();
  const { data, isLoading, isError, error } = useListAgentVersions(agentId);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);

  useEffect(() => {
    if (!isError || !error) return;
    const message = error instanceof Error ? error.message : "Failed to load versions";
    auditLog.write({
      level: "error",
      event: "agent.inspector.versions.load.failed",
      sessionId: "",
      timestamp: new Date().toISOString(),
      detail: { agent_id: agentId, message },
    });
  }, [isError, error, auditLog, agentId]);

  if (isLoading) {
    return <p data-testid="versions-loading">Loading version history…</p>;
  }

  if (isError) {
    const msg = error instanceof Error ? error.message : "Failed to load version history";
    return (
      <p role="alert" data-testid="versions-error">
        {msg}
      </p>
    );
  }

  const versions = data?.items ?? [];
  const selectedVersion = versions.find((v) => v.id === selectedVersionId) ?? null;
  const isCurrentSelected = selectedVersionId === currentVersion.id;

  return (
    <section data-testid="version-history-panel">
      <h2>Version History</h2>
      {versions.length === 0 ? (
        <p data-testid="versions-empty">No version history available.</p>
      ) : (
        <table data-testid="versions-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Version ID</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((ver) => (
              <tr
                key={ver.id}
                data-testid={`version-row-${ver.id}`}
                aria-selected={ver.id === selectedVersionId}
              >
                <td>
                  <button
                    type="button"
                    onClick={() =>
                      setSelectedVersionId((prev) => (prev === ver.id ? null : ver.id))
                    }
                    data-testid={`version-select-${ver.id}`}
                    aria-expanded={ver.id === selectedVersionId}
                  >
                    v{ver.version_number}
                  </button>
                  {ver.id === currentVersion.id && (
                    <span data-testid={`current-badge-${ver.id}`}> (current)</span>
                  )}
                </td>
                <td>{ver.id}</td>
                <td>{ver.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {selectedVersion && !isCurrentSelected && (
        <VersionDiffPanel current={currentVersion} selected={selectedVersion} />
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// AgentDetailPage
// ---------------------------------------------------------------------------

export function AgentDetailPage(): React.ReactElement | null {
  const { id } = useParams<{ id: string }>();
  const agentId = id ?? "";
  const { auditLog } = useMeridianApi();

  const { data, isLoading, isError, error } = useGetAgent(agentId);

  const loadedRef = useRef(false);
  useEffect(() => {
    if (!data || loadedRef.current) return;
    loadedRef.current = true;
    const tracer = getTracer();
    const timestamp = new Date().toISOString();
    tracer.startActiveSpan("agent.inspector", { attributes: { "agent.id": agentId } }, (span) => {
      recordApiInvocationEvent(span, {
        name: "agent.inspector.invocation",
        operation: "agent.inspector",
        timestamp,
      });
      span.end();
    });
  }, [data, agentId]);

  useEffect(() => {
    if (!isError || !error) return;
    const message = error instanceof Error ? error.message : "Failed to load agent";
    auditLog.write({
      level: "error",
      event: "agent.inspector.load.failed",
      sessionId: "",
      timestamp: new Date().toISOString(),
      detail: { agent_id: agentId, message },
    });
  }, [isError, error, auditLog, agentId]);

  if (isLoading) {
    return <p data-testid="agent-loading">Loading agent…</p>;
  }

  if (isError) {
    const msg = error instanceof Error ? error.message : "Failed to load agent";
    return (
      <p role="alert" data-testid="agent-error">
        {msg}
      </p>
    );
  }

  if (!data) return null;

  return (
    <div data-testid="agent-detail-page">
      <h1 data-testid="agent-name">{data.name}</h1>
      <dl data-testid="agent-metadata">
        <dt>ID</dt>
        <dd data-testid="agent-id">{data.id}</dd>
        <dt>Kind</dt>
        <dd data-testid="agent-kind">{data.kind}</dd>
        <dt>Created</dt>
        <dd data-testid="agent-created">{data.created_at}</dd>
      </dl>
      <CurrentVersionPanel version={data.version} />
      <VersionHistoryPanel agentId={agentId} currentVersion={data.version} />
    </div>
  );
}
