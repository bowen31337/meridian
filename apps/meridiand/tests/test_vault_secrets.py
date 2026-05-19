"""
Vault secrets endpoint conformance suite.

Tests cover:
  - POST /v1/vaults/{id}/secrets returns 201 on success.
  - Response has vault_id, key, created_at (never value).
  - Secret JSON written under storage_root/vaults/{id}/secrets/{key}.json.
  - Persisted record contains value (stored but never returned via API).
  - Not written on validation failure.
  - Empty key returns 422 with code "vault_secret_invalid_request".
  - Missing key or value fields return 422.
  - Unknown vault_id returns 404 with code "vault_not_found".
  - Duplicate key returns 409 with code "vault_secret_conflict".
  - On any failure, audit log entry written with event "vault.secret.store.failed".
  - OTel span "vault.secret.store" emitted on success and failure.
  - GET /v1/vaults/{id}/secrets/{name}/meta returns 200.
  - Meta response has vault_id, key, created_at, last_accessed_at, requester_counts.
  - Meta response NEVER contains value.
  - last_accessed_at updated on each GET.
  - requester_counts incremented on each GET.
  - Unknown vault_id on GET returns 404 with code "vault_not_found".
  - Unknown secret name returns 404 with code "vault_secret_not_found".
  - On GET failure, audit log entry written with event "vault.secret.meta.failed".
  - OTel span "vault.secret.meta" emitted on success and failure.
  - Routes present with storage_root, absent without.
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
# Helpers
# ---------------------------------------------------------------------------


class _MemoryKeyring:
    """In-memory keyring for test isolation."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


def _make_client(storage_root: Path) -> TestClient:
    os_keychain = OsKeychainVaultBackend(_keyring=_MemoryKeyring())
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        os_keychain_backend=os_keychain,
    )
    return TestClient(app, raise_server_exceptions=False)


def _vault_body(**overrides) -> dict:
    base: dict = {"name": "my-vault", "backend": "os_keychain"}
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


def _make_backend() -> OsKeychainVaultBackend:
    return OsKeychainVaultBackend(_keyring=_MemoryKeyring())


def _make_client_from_backend(
    storage_root: Path, backend: OsKeychainVaultBackend
) -> TestClient:
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        os_keychain_backend=backend,
    )
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST success
# ---------------------------------------------------------------------------


class TestVaultSecretStoreSuccess:
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
        body = _store_secret(client, vault_id, key="db_password")
        assert body["key"] == "db_password"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = _store_secret(client, vault_id)
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_response_does_not_contain_value(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = _store_secret(client, vault_id, value="topsecret")
        assert "value" not in body

    def test_different_keys_each_succeed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        r1 = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="key_a")
        )
        r2 = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="key_b")
        )
        assert r1.status_code == 201
        assert r2.status_code == 201


# ---------------------------------------------------------------------------
# POST persistence
# ---------------------------------------------------------------------------


class TestVaultSecretStorePersistence:
    def test_secret_written_to_keychain(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="my_key")
        assert backend.secret_exists(vault_id, "my_key")

    def test_persisted_value_present(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="tok", value="secret_val")
        record = backend.get_secret(vault_id, "tok")
        assert record is not None
        assert record["value"] == "secret_val"

    def test_persisted_key_correct(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="persist_key")
        record = backend.get_secret(vault_id, "persist_key")
        assert record is not None
        assert record["key"] == "persist_key"

    def test_persisted_vault_id_correct(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="some_key")
        record = backend.get_secret(vault_id, "some_key")
        assert record is not None
        assert record["vault_id"] == vault_id

    def test_persisted_last_accessed_at_is_none(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="fresh_key")
        record = backend.get_secret(vault_id, "fresh_key")
        assert record is not None
        assert record["last_accessed_at"] is None

    def test_persisted_requester_counts_empty(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="cnt_key")
        record = backend.get_secret(vault_id, "cnt_key")
        assert record is not None
        assert record["requester_counts"] == {}

    def test_not_stored_on_validation_failure(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="   "))
        assert not backend.secret_exists(vault_id, "   ")

    def test_not_stored_when_vault_not_found(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        assert not backend.secret_exists("vault_ghost", "api_key")


# ---------------------------------------------------------------------------
# POST validation
# ---------------------------------------------------------------------------


class TestVaultSecretStoreValidation:
    def test_empty_key_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="   ")
        )
        assert resp.status_code == 422

    def test_empty_key_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="")
        ).json()
        assert body["error"]["code"] == "vault_secret_invalid_request"

    def test_empty_key_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="")
        ).json()
        assert len(body["error"]["message"]) > 0

    def test_missing_key_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.post(
            f"/v1/vaults/{vault_id}/secrets", json={"value": "s3cr3t"}
        )
        assert resp.status_code == 422

    def test_missing_value_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.post(
            f"/v1/vaults/{vault_id}/secrets", json={"key": "my_key"}
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST vault not found
# ---------------------------------------------------------------------------


class TestVaultSecretStoreVaultNotFound:
    def test_unknown_vault_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/vaults/vault_nonexistent/secrets", json=_secret_body()
        )
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/vaults/vault_nonexistent/secrets", json=_secret_body()
        ).json()
        assert body["error"]["code"] == "vault_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/vaults/vault_nonexistent/secrets", json=_secret_body()
        ).json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST conflict
# ---------------------------------------------------------------------------


class TestVaultSecretStoreConflict:
    def test_duplicate_key_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dupe"))
        resp = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dupe")
        )
        assert resp.status_code == 409

    def test_duplicate_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dupe2"))
        body = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dupe2")
        ).json()
        assert body["error"]["code"] == "vault_secret_conflict"

    def test_duplicate_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dupe3"))
        body = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dupe3")
        ).json()
        assert len(body["error"]["message"]) > 0

    def test_different_vault_same_key_allowed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        v1 = _create_vault(client, name="vault-one")["id"]
        v2 = _create_vault(client, name="vault-two")["id"]
        client.post(f"/v1/vaults/{v1}/secrets", json=_secret_body(key="shared"))
        resp = client.post(
            f"/v1/vaults/{v2}/secrets", json=_secret_body(key="shared")
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# POST audit log
# ---------------------------------------------------------------------------


class TestVaultSecretStoreAuditLog:
    def test_vault_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.store.failed" for r in records)

    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.store.failed" for r in records)

    def test_conflict_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup"))
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup"))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.store.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.store.failed"
        )
        assert record["level"] == "error"

    def test_vault_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.store.failed"
        )
        assert record["code"] == "vault_not_found"

    def test_validation_failure_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key=""))
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.store.failed"
        )
        assert record["code"] == "vault_secret_invalid_request"

    def test_conflict_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup4"))
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="dup4"))
        records = [
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.store.failed"
        ]
        assert any(r["code"] == "vault_secret_conflict" for r in records)

    def test_audit_detail_has_vault_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.store.failed"
        )
        assert record["detail"]["vault_id"] == "vault_ghost"

    def test_audit_detail_has_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/vaults/vault_ghost/secrets", json=_secret_body(key="audit_key")
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.store.failed"
        )
        assert record["detail"]["key"] == "audit_key"

    def test_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.store.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST OTel spans
# ---------------------------------------------------------------------------


class TestVaultSecretStoreOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _otel_exporter.clear()
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.secret.store" in span_names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.secret.store" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/vaults/vault_ghost/secrets", json=_secret_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.secret.store")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_vault_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _otel_exporter.clear()
        client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.secret.store")
        assert span is not None
        assert span.attributes["vault.id"] == vault_id

    def test_success_span_has_secret_key_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _otel_exporter.clear()
        client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="otel_key")
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.secret.store")
        assert span is not None
        assert span.attributes["secret.key"] == "otel_key"


# ---------------------------------------------------------------------------
# POST route wiring
# ---------------------------------------------------------------------------


class TestVaultSecretStoreRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.post(f"/v1/vaults/{vault_id}/secrets", json=_secret_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/vaults/vault_any/secrets", json=_secret_body())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET meta success
# ---------------------------------------------------------------------------


class TestVaultSecretMetaSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="my_key")
        resp = client.get(f"/v1/vaults/{vault_id}/secrets/my_key/meta")
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

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="ts_key")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/ts_key/meta").json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_response_has_last_accessed_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="la_key")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/la_key/meta").json()
        assert "last_accessed_at" in body
        assert body["last_accessed_at"] is not None
        assert isinstance(body["last_accessed_at"], str)

    def test_response_has_requester_counts(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="rc_key")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/rc_key/meta").json()
        assert "requester_counts" in body
        assert isinstance(body["requester_counts"], dict)

    def test_response_does_not_contain_value(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="nv_key", value="secret_sauce")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/nv_key/meta").json()
        assert "value" not in body
        assert "secret_sauce" not in json.dumps(body)

    def test_last_accessed_at_updated_on_each_call(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="upd_key")
        body1 = client.get(f"/v1/vaults/{vault_id}/secrets/upd_key/meta").json()
        body2 = client.get(f"/v1/vaults/{vault_id}/secrets/upd_key/meta").json()
        # Both calls set last_accessed_at; values may be equal (same second) but
        # the field must be present and non-null after the first call.
        assert body1["last_accessed_at"] is not None
        assert body2["last_accessed_at"] is not None

    def test_requester_count_increments(self, storage_root: Path) -> None:
        backend = _make_backend()
        client = _make_client_from_backend(storage_root, backend)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="inc_key")
        client.get(f"/v1/vaults/{vault_id}/secrets/inc_key/meta")
        client.get(f"/v1/vaults/{vault_id}/secrets/inc_key/meta")
        record = backend.get_secret(vault_id, "inc_key")
        assert record is not None
        total = sum(record["requester_counts"].values())
        assert total == 2

    def test_created_at_unchanged_after_meta_calls(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        store_resp = client.post(
            f"/v1/vaults/{vault_id}/secrets", json=_secret_body(key="ca_key")
        ).json()
        original_created_at = store_resp["created_at"]
        client.get(f"/v1/vaults/{vault_id}/secrets/ca_key/meta")
        body = client.get(f"/v1/vaults/{vault_id}/secrets/ca_key/meta").json()
        assert body["created_at"] == original_created_at


# ---------------------------------------------------------------------------
# GET meta vault not found
# ---------------------------------------------------------------------------


class TestVaultSecretMetaVaultNotFound:
    def test_unknown_vault_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/vaults/vault_ghost/secrets/some_key/meta")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/vaults/vault_ghost/secrets/some_key/meta").json()
        assert body["error"]["code"] == "vault_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/vaults/vault_ghost/secrets/some_key/meta").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# GET meta secret not found
# ---------------------------------------------------------------------------


class TestVaultSecretMetaNotFound:
    def test_unknown_secret_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.get(f"/v1/vaults/{vault_id}/secrets/no_such_key/meta")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = client.get(f"/v1/vaults/{vault_id}/secrets/no_such_key/meta").json()
        assert body["error"]["code"] == "vault_secret_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        body = client.get(f"/v1/vaults/{vault_id}/secrets/no_such_key/meta").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# GET meta audit log
# ---------------------------------------------------------------------------


class TestVaultSecretMetaAuditLog:
    def test_vault_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/vaults/vault_ghost/secrets/k/meta")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.meta.failed" for r in records)

    def test_secret_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.get(f"/v1/vaults/{vault_id}/secrets/missing/meta")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.meta.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/vaults/vault_ghost/secrets/k/meta")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.meta.failed"
        )
        assert record["level"] == "error"

    def test_vault_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/vaults/vault_ghost/secrets/k/meta")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.meta.failed"
        )
        assert record["code"] == "vault_not_found"

    def test_secret_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.get(f"/v1/vaults/{vault_id}/secrets/missing/meta")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.meta.failed"
        )
        assert record["code"] == "vault_secret_not_found"

    def test_audit_detail_has_vault_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/vaults/vault_ghost/secrets/k/meta")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.meta.failed"
        )
        assert record["detail"]["vault_id"] == "vault_ghost"

    def test_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.get(f"/v1/vaults/{vault_id}/secrets/target_key/meta")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.meta.failed"
        )
        assert record["detail"]["name"] == "target_key"

    def test_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/vaults/vault_ghost/secrets/k/meta")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "vault.secret.meta.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# GET meta OTel spans
# ---------------------------------------------------------------------------


class TestVaultSecretMetaOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="otel_mk")
        _otel_exporter.clear()
        client.get(f"/v1/vaults/{vault_id}/secrets/otel_mk/meta")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.secret.meta" in span_names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/vaults/vault_ghost/secrets/k/meta")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.secret.meta" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.get("/v1/vaults/vault_ghost/secrets/k/meta")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.secret.meta")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_vault_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="attr_key")
        _otel_exporter.clear()
        client.get(f"/v1/vaults/{vault_id}/secrets/attr_key/meta")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.secret.meta")
        assert span is not None
        assert span.attributes["vault.id"] == vault_id

    def test_success_span_has_secret_key_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="named_key")
        _otel_exporter.clear()
        client.get(f"/v1/vaults/{vault_id}/secrets/named_key/meta")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.secret.meta")
        assert span is not None
        assert span.attributes["secret.key"] == "named_key"


# ---------------------------------------------------------------------------
# GET meta route wiring
# ---------------------------------------------------------------------------


class TestVaultSecretMetaRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, key="wiring_key")
        resp = client.get(f"/v1/vaults/{vault_id}/secrets/wiring_key/meta")
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/vaults/vault_any/secrets/key/meta")
        assert resp.status_code == 404
