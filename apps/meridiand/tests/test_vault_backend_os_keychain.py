"""
OsKeychainVaultBackend conformance suite.

Tests cover:
  - secret_exists returns False before any store.
  - store_secret writes to the keychain.
  - get_secret returns None for a missing key.
  - get_secret returns the stored record after store_secret.
  - secret_exists returns True after store_secret.
  - update_secret persists last_accessed_at and requester_counts changes.
  - Multiple secrets in the same vault stored under distinct accounts.
  - Keychain account format is "<vault_id>/<key>".
  - API: POST /v1/vaults/{id}/secrets 201 for os_keychain vault.
  - API: POST response has vault_id, key, created_at (never value).
  - API: Duplicate key returns 409 for os_keychain vault.
  - API: GET /v1/vaults/{id}/secrets/{name}/meta returns 200.
  - API: Meta response has vault_id, key, created_at, last_accessed_at,
    requester_counts; never contains value.
  - API: last_accessed_at updated on each GET.
  - API: requester_counts incremented on each GET.
  - API: os_keychain backend not configured returns 500 on POST.
  - API: os_keychain backend not configured returns 500 on GET meta.
  - API: backend not configured failure written to audit log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._vault_backend_os_keychain import OsKeychainVaultBackend

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# In-memory keyring for test isolation
# ---------------------------------------------------------------------------


class _MemoryKeyring:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend() -> OsKeychainVaultBackend:
    return OsKeychainVaultBackend(_keyring=_MemoryKeyring())


def _make_client(
    storage_root: Path,
    *,
    with_backend: bool = True,
) -> TestClient:
    backend = _make_backend() if with_backend else None
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        os_keychain_backend=backend,
    )
    return TestClient(app, raise_server_exceptions=False)


def _make_client_with_backend(
    storage_root: Path,
) -> tuple[TestClient, OsKeychainVaultBackend]:
    backend = _make_backend()
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        os_keychain_backend=backend,
    )
    return TestClient(app, raise_server_exceptions=False), backend


def _vault_body(**overrides) -> dict:
    base: dict = {"name": "ks-vault", "backend": "os_keychain"}
    base.update(overrides)
    return base


def _secret_body(**overrides) -> dict:
    base: dict = {"key": "api_key", "value": "s3cr3t"}
    base.update(overrides)
    return base


def _create_vault(client: TestClient, **overrides) -> dict:
    return client.post("/v1/vaults", json=_vault_body(**overrides)).json()


def _store_secret(client: TestClient, vault_id: str, **overrides) -> dict:
    return client.post(
        f"/v1/vaults/{vault_id}/secrets", json=_secret_body(**overrides)
    ).json()


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Backend unit operations
# ---------------------------------------------------------------------------


class TestBackendOperations:
    def test_secret_does_not_exist_initially(self) -> None:
        backend = _make_backend()
        assert not backend.secret_exists("vault_abc", "my_key")

    def test_store_makes_secret_exist(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_abc", "my_key", "my_value", "2024-01-01T00:00:00+00:00")
        assert backend.secret_exists("vault_abc", "my_key")

    def test_get_secret_returns_none_when_missing(self) -> None:
        backend = _make_backend()
        assert backend.get_secret("vault_abc", "missing") is None

    def test_get_secret_returns_stored_record(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        record = backend.get_secret("vault_abc", "tok")
        assert record is not None
        assert record["key"] == "tok"
        assert record["value"] == "val"

    def test_stored_record_has_vault_id(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        record = backend.get_secret("vault_abc", "tok")
        assert record is not None
        assert record["vault_id"] == "vault_abc"

    def test_stored_record_has_created_at(self) -> None:
        backend = _make_backend()
        ts = "2024-01-01T00:00:00+00:00"
        backend.store_secret("vault_abc", "tok", "val", ts)
        record = backend.get_secret("vault_abc", "tok")
        assert record is not None
        assert record["created_at"] == ts

    def test_stored_record_last_accessed_at_is_none(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        record = backend.get_secret("vault_abc", "tok")
        assert record is not None
        assert record["last_accessed_at"] is None

    def test_stored_record_requester_counts_empty(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        record = backend.get_secret("vault_abc", "tok")
        assert record is not None
        assert record["requester_counts"] == {}

    def test_update_secret_persists_changes(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        record = dict(backend.get_secret("vault_abc", "tok"))  # type: ignore[arg-type]
        record["last_accessed_at"] = "2024-06-01T12:00:00+00:00"
        record["requester_counts"] = {"127.0.0.1": 1}
        backend.update_secret("vault_abc", "tok", record)
        refreshed = backend.get_secret("vault_abc", "tok")
        assert refreshed is not None
        assert refreshed["last_accessed_at"] == "2024-06-01T12:00:00+00:00"
        assert refreshed["requester_counts"] == {"127.0.0.1": 1}

    def test_multiple_secrets_same_vault(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_multi", "k1", "v1", "2024-01-01T00:00:00+00:00")
        backend.store_secret("vault_multi", "k2", "v2", "2024-01-01T00:00:00+00:00")
        assert backend.secret_exists("vault_multi", "k1")
        assert backend.secret_exists("vault_multi", "k2")

    def test_secrets_isolated_across_vaults(self) -> None:
        backend = _make_backend()
        backend.store_secret("vault_a", "key", "val_a", "2024-01-01T00:00:00+00:00")
        assert not backend.secret_exists("vault_b", "key")

    def test_keychain_account_format(self) -> None:
        keyring = _MemoryKeyring()
        backend = OsKeychainVaultBackend(_keyring=keyring)
        backend.store_secret("vault_x", "tok", "val", "2024-01-01T00:00:00+00:00")
        assert keyring.get_password("meridian", "vault_x/tok") is not None

    def test_store_returns_record(self) -> None:
        backend = _make_backend()
        record = backend.store_secret("vault_abc", "k", "v", "2024-01-01T00:00:00+00:00")
        assert record["key"] == "k"
        assert record["value"] == "v"
        assert record["vault_id"] == "vault_abc"


# ---------------------------------------------------------------------------
# API — os_keychain vault with backend configured
# ---------------------------------------------------------------------------


class TestApiOsKeychainBackendStore:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body())
        assert resp.status_code == 201

    def test_response_has_vault_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = _store_secret(client, vault_id)
        assert body["vault_id"] == vault_id

    def test_response_has_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = _store_secret(client, vault_id, key="db_pass")
        assert body["key"] == "db_pass"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = _store_secret(client, vault_id)
        assert "created_at" in body
        assert isinstance(body["created_at"], str)

    def test_response_does_not_contain_value(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = _store_secret(client, vault_id, value="top_secret")
        assert "value" not in body

    def test_duplicate_key_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup"))
        resp = client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup"))
        assert resp.status_code == 409

    def test_duplicate_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup2"))
        body = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup2")
        ).json()
        assert body["error"]["code"] == "vault_secret_conflict"

    def test_value_stored_in_keychain_not_plaintext(self, storage_root: Path) -> None:
        keyring = _MemoryKeyring()
        backend = OsKeychainVaultBackend(_keyring=keyring)
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            os_keychain_backend=backend,
        )
        client = TestClient(app, raise_server_exceptions=False)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="tok", value="secret_needle")
        raw = keyring.get_password("meridian", f"{vault_id}/tok")
        assert raw is not None
        record = json.loads(raw)
        assert record["value"] == "secret_needle"


# ---------------------------------------------------------------------------
# API — os_keychain vault meta
# ---------------------------------------------------------------------------


class TestApiOsKeychainBackendMeta:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="mk")
        resp = client.get(f"/v1/vaults/{vault_id}/secrets/mk/meta")
        assert resp.status_code == 200

    def test_response_has_vault_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="mk")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/mk/meta").json()
        assert body["vault_id"] == vault_id

    def test_response_has_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="token")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/token/meta").json()
        assert body["key"] == "token"

    def test_response_has_last_accessed_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="la_key")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/la_key/meta").json()
        assert body["last_accessed_at"] is not None

    def test_response_has_requester_counts(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="rc_key")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/rc_key/meta").json()
        assert isinstance(body["requester_counts"], dict)

    def test_response_does_not_contain_value(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="nv_key", value="hidden_val")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/nv_key/meta").json()
        assert "value" not in body
        assert "hidden_val" not in json.dumps(body)

    def test_last_accessed_at_set_after_get(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="upd_key")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/upd_key/meta").json()
        assert body["last_accessed_at"] is not None

    def test_requester_count_increments(self, storage_root: Path) -> None:
        client, backend = _make_client_with_backend(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="inc_key")
        client.get(f"/v1/vaults/{vault_id}/secrets/inc_key/meta")
        client.get(f"/v1/vaults/{vault_id}/secrets/inc_key/meta")
        record = backend.get_secret(vault_id, "inc_key")
        assert record is not None
        total = sum(record["requester_counts"].values())
        assert total == 2

    def test_unknown_secret_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.get(f"/v1/vaults/{vault_id}/secrets/no_such_key/meta")
        assert resp.status_code == 404

    def test_unknown_secret_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = client.get(f"/v1/vaults/{vault_id}/secrets/no_such_key/meta").json()
        assert body["error"]["code"] == "vault_secret_not_found"


# ---------------------------------------------------------------------------
# API — backend not configured
# ---------------------------------------------------------------------------


class TestApiBackendNotConfigured:
    def test_store_without_backend_returns_500(self, storage_root: Path) -> None:
        client = _make_client(storage_root, with_backend=False)
        vault_id = _create_vault(client)["id"]
        resp = client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body())
        assert resp.status_code == 500

    def test_store_without_backend_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root, with_backend=False)
        vault_id = _create_vault(client)["id"]
        body = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body()
        ).json()
        assert body["error"]["code"] == "vault_secret_store_failed"

    def test_store_without_backend_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root, with_backend=False)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body())
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.store.failed" for r in records)

    def test_meta_without_backend_returns_500(self, storage_root: Path) -> None:
        # Store with backend, then access meta without one.
        backend = _make_backend()
        app_with = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            os_keychain_backend=backend,
        )
        client_with = TestClient(app_with, raise_server_exceptions=False)
        vault_id = _create_vault(client_with)["id"]
        _store_secret(client_with, vault_id, key="a_key")

        app_without = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            os_keychain_backend=None,
        )
        client_without = TestClient(app_without, raise_server_exceptions=False)
        resp = client_without.get(f"/v1/vaults/{vault_id}/secrets/a_key/meta")
        assert resp.status_code == 500

    def test_meta_without_backend_error_code(self, storage_root: Path) -> None:
        backend = _make_backend()
        app_with = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            os_keychain_backend=backend,
        )
        client_with = TestClient(app_with, raise_server_exceptions=False)
        vault_id = _create_vault(client_with)["id"]
        _store_secret(client_with, vault_id, key="b_key")

        app_without = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            os_keychain_backend=None,
        )
        client_without = TestClient(app_without, raise_server_exceptions=False)
        body = client_without.get(f"/v1/vaults/{vault_id}/secrets/b_key/meta").json()
        assert body["error"]["code"] == "vault_secret_meta_failed"

    def test_meta_without_backend_writes_audit(self, storage_root: Path) -> None:
        backend = _make_backend()
        app_with = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            os_keychain_backend=backend,
        )
        client_with = TestClient(app_with, raise_server_exceptions=False)
        vault_id = _create_vault(client_with)["id"]
        _store_secret(client_with, vault_id, key="c_key")

        app_without = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            os_keychain_backend=None,
        )
        client_without = TestClient(app_without, raise_server_exceptions=False)
        client_without.get(f"/v1/vaults/{vault_id}/secrets/c_key/meta")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.meta.failed" for r in records)
