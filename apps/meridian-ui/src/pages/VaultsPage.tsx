import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { SecretMeta, Vault } from "../api/client.js";
import { createApiClient } from "../api/client.js";
import { useMeridianApi } from "../api/context.js";
import { queryKeys } from "../api/query-keys.js";

function useListVaults() {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.vaults.list(),
    queryFn: () => createApiClient(baseUrl).listVaults(),
  });
}

function useListVaultSecrets(vaultId: string | null) {
  const { baseUrl } = useMeridianApi();
  return useQuery({
    queryKey: queryKeys.vaults.secrets(vaultId ?? ""),
    queryFn: () => createApiClient(baseUrl).listVaultSecrets(vaultId ?? ""),
    enabled: vaultId !== null,
  });
}

function useDeleteVaultSecret(vaultId: string | null) {
  const { baseUrl } = useMeridianApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => createApiClient(baseUrl).deleteVaultSecret(vaultId ?? "", name),
    onSuccess: () => {
      if (vaultId) {
        queryClient.invalidateQueries({ queryKey: queryKeys.vaults.secrets(vaultId) });
      }
    },
  });
}

function SecretRow({
  secret,
  onDelete,
}: {
  secret: SecretMeta;
  onDelete: (name: string) => void;
}) {
  const [confirming, setConfirming] = useState(false);

  function handleDeleteClick() {
    setConfirming(true);
  }

  function handleConfirm() {
    onDelete(secret.key);
    setConfirming(false);
  }

  function handleCancel() {
    setConfirming(false);
  }

  const totalRequests = Object.values(secret.requester_counts).reduce((a, b) => a + b, 0);

  return (
    <tr data-testid={`secret-row-${secret.key}`}>
      <td>{secret.key}</td>
      <td>{secret.created_at}</td>
      <td>{secret.last_accessed_at ?? "—"}</td>
      <td>{totalRequests}</td>
      <td>
        {confirming ? (
          <span>
            Delete &quot;{secret.key}&quot;?{" "}
            <button
              type="button"
              onClick={handleConfirm}
              data-testid={`confirm-delete-${secret.key}`}
            >
              Confirm
            </button>{" "}
            <button type="button" onClick={handleCancel}>
              Cancel
            </button>
          </span>
        ) : (
          <button type="button" onClick={handleDeleteClick} data-testid={`delete-${secret.key}`}>
            Delete
          </button>
        )}
      </td>
    </tr>
  );
}

function SecretsPanel({ vault }: { vault: Vault }) {
  const { data, isLoading, isError, error } = useListVaultSecrets(vault.id);
  const deleteMutation = useDeleteVaultSecret(vault.id);

  if (isLoading) {
    return <p data-testid="secrets-loading">Loading secrets…</p>;
  }

  if (isError) {
    const msg = error instanceof Error ? error.message : "Failed to load secrets";
    return (
      <p role="alert" data-testid="secrets-error">
        {msg}
      </p>
    );
  }

  const items = data?.items ?? [];

  return (
    <div data-testid="secrets-panel">
      {deleteMutation.isError && (
        <p role="alert" data-testid="delete-error">
          {deleteMutation.error instanceof Error ? deleteMutation.error.message : "Delete failed"}
        </p>
      )}
      {items.length === 0 ? (
        <p data-testid="secrets-empty">No secrets stored in this vault.</p>
      ) : (
        <table data-testid="secrets-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Created</th>
              <th>Last Accessed</th>
              <th>Requests</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {items.map((secret) => (
              <SecretRow
                key={secret.key}
                secret={secret}
                onDelete={(name) => deleteMutation.mutate(name)}
              />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function VaultsPage() {
  const [selectedVaultId, setSelectedVaultId] = useState<string | null>(null);
  const { data, isLoading, isError, error } = useListVaults();

  const vaults = data?.items ?? [];
  const selectedVault = vaults.find((v) => v.id === selectedVaultId) ?? null;

  if (isLoading) {
    return <p data-testid="vaults-loading">Loading vaults…</p>;
  }

  if (isError) {
    const msg = error instanceof Error ? error.message : "Failed to load vaults";
    return (
      <p role="alert" data-testid="vaults-error">
        {msg}
      </p>
    );
  }

  return (
    <div data-testid="vaults-page">
      <h1>Vaults</h1>
      {vaults.length === 0 ? (
        <p data-testid="vaults-empty">No vaults configured.</p>
      ) : (
        <ul data-testid="vault-list">
          {vaults.map((vault) => (
            <li key={vault.id}>
              <button
                type="button"
                onClick={() => setSelectedVaultId(vault.id === selectedVaultId ? null : vault.id)}
                aria-pressed={vault.id === selectedVaultId}
                data-testid={`vault-item-${vault.id}`}
              >
                {vault.name}
              </button>
              <span> ({vault.backend})</span>
            </li>
          ))}
        </ul>
      )}
      {selectedVault && (
        <section data-testid="vault-inspector">
          <h2>Secrets in &quot;{selectedVault.name}&quot;</h2>
          <SecretsPanel vault={selectedVault} />
        </section>
      )}
    </div>
  );
}
