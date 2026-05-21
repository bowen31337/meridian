import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { createApiClient } from "../api/client.js";
import { useMeridianApi } from "../api/context.js";
import { queryKeys } from "../api/query-keys.js";

function useListAgents() {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.agents.list(),
    queryFn: () => createApiClient(baseUrl).listAgents(),
  });
}

export function AgentsPage() {
  const { data, isLoading, isError, error } = useListAgents();

  if (isLoading) {
    return <p data-testid="agents-loading">Loading agents…</p>;
  }

  if (isError) {
    const msg = error instanceof Error ? error.message : "Failed to load agents";
    return (
      <p role="alert" data-testid="agents-error">
        {msg}
      </p>
    );
  }

  const agents = data?.items ?? [];

  return (
    <div data-testid="agents-page">
      <h1>Agents</h1>
      {agents.length === 0 ? (
        <p data-testid="agents-empty">No agents configured.</p>
      ) : (
        <ul data-testid="agents-list">
          {agents.map((agent) => (
            <li key={agent.id}>
              <Link to={`/agents/${agent.id}`} data-testid={`agent-link-${agent.id}`}>
                {agent.name}
              </Link>
              <span> ({agent.kind})</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
