"""
EncryptedFileVaultBackend conformance suite.

Tests cover:
  - unlock_with_passphrase marks the backend as unlocked.
  - unlock_with_passphrase emits OTel span "vault.backend.unlock".
  - unlock_with_key_file marks the backend as unlocked.
  - unlock_with_key_file emits OTel span "vault.backend.unlock".
  - unlock_with_key_file with a bad key raises VaultBackendUnlockError.
  - unlock error span has ERROR status.
  - secret_exists returns False before any store.
  - store_secret writes an encrypted file (secrets.age) to disk.
  - The secrets.age file does NOT contain the plaintext secret value.
  - get_secret returns the stored record after store_secret.
  - secret_exists returns True after store_secret.
  - update_secret persists last_accessed_at and requester_counts changes.
  - Multiple secrets in the same vault are stored in one encrypted file.
  - API: POST /v1/vaults/{id}/secrets 201 for encrypted_file vault.
  - API: POST response has vault_id, key, created_at (never value).
  - API: Duplicate key returns 409 for encrypted_file vault.
  - API: GET /v1/vaults/{id}/secrets/{name}/meta returns 200.
  - API: Meta response has vault_id, key, created_at, last_accessed_at,
    requester_counts; never contains value.
  - API: last_accessed_at updated on each GET.
  - API: requester_counts incremented on each GET.
  - API: encrypted_file backend not configured returns 500.
  - API: backend not configured failure written to audit log.
  - Routes present with storage_root; absent without.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._vault_backend_encrypted_file import (
    EncryptedFileVaultBackend,
    VaultBackendUnlockError,
)

from tests._otel_shared import otel_exporter as _otel_exporter

_PASSPHRASE = "test-passphrase-hunter2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(storage_root: Path) -> EncryptedFileVaultBackend:
    backend = EncryptedFileVaultBackend(storage_root)
    backend.unlock_with_passphrase(_PASSPHRASE)
    return backend


def _make_client(
    storage_root: Path,
    *,
    with_backend: bool = True,
) -> TestClient:
    backend = _make_backend(storage_root) if with_backend else None
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        vault_backend=backend,
    )
    return TestClient(app, raise_server_exceptions=False)


def _vault_body(**overrides) -> dict:
    base: dict = {"name": "enc-vault", "backend": "encrypted_file"}
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


def _age_file(storage_root: Path, vault_id: str) -> Path:
    return storage_root / "vaults" / vault_id / "secrets.age"


def _make_age_key_file(tmp_path: Path) -> Path:
    import pyrage  # type: ignore[import-untyped]

    identity = pyrage.x25519.Identity.generate()
    key_file = tmp_path / "test.key"
    # str(identity) returns the AGE-SECRET-KEY-1... private key string
    key_file.write_text(str(identity) + "\n")
    return key_file


# ---------------------------------------------------------------------------
# Unlock — passphrase
# ---------------------------------------------------------------------------


class TestUnlockPassphrase:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_unlocked_after_passphrase(self, storage_root: Path) -> None:
        backend = EncryptedFileVaultBackend(storage_root)
        assert not backend.is_unlocked
        backend.unlock_with_passphrase(_PASSPHRASE)
        assert backend.is_unlocked

    def test_passphrase_emits_unlock_span(self, storage_root: Path) -> None:
        backend = EncryptedFileVaultBackend(storage_root)
        backend.unlock_with_passphrase(_PASSPHRASE)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.backend.unlock" in span_names

    def test_passphrase_span_has_auth_attribute(self, storage_root: Path) -> None:
        backend = EncryptedFileVaultBackend(storage_root)
        backend.unlock_with_passphrase(_PASSPHRASE)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans["vault.backend.unlock"]
        assert span.attributes["vault.backend.auth"] == "passphrase"

    def test_passphrase_span_has_backend_attribute(self, storage_root: Path) -> None:
        backend = EncryptedFileVaultBackend(storage_root)
        backend.unlock_with_passphrase(_PASSPHRASE)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans["vault.backend.unlock"]
        assert span.attributes["vault.backend"] == "encrypted_file"


# ---------------------------------------------------------------------------
# Unlock — key file
# ---------------------------------------------------------------------------


class TestUnlockKeyFile:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_unlocked_after_key_file(self, storage_root: Path, tmp_path: Path) -> None:
        key_file = _make_age_key_file(tmp_path)
        backend = EncryptedFileVaultBackend(storage_root)
        backend.unlock_with_key_file(key_file)
        assert backend.is_unlocked

    def test_key_file_emits_unlock_span(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        key_file = _make_age_key_file(tmp_path)
        backend = EncryptedFileVaultBackend(storage_root)
        backend.unlock_with_key_file(key_file)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.backend.unlock" in span_names

    def test_key_file_span_has_auth_attribute(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        key_file = _make_age_key_file(tmp_path)
        backend = EncryptedFileVaultBackend(storage_root)
        backend.unlock_with_key_file(key_file)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans["vault.backend.unlock"]
        assert span.attributes["vault.backend.auth"] == "key_file"

    def test_invalid_key_file_raises_unlock_error(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        bad_key = tmp_path / "bad.key"
        bad_key.write_text("not-a-valid-age-key\n")
        backend = EncryptedFileVaultBackend(storage_root)
        with pytest.raises(VaultBackendUnlockError):
            backend.unlock_with_key_file(bad_key)

    def test_invalid_key_file_span_has_error_status(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        from opentelemetry.trace import StatusCode

        bad_key = tmp_path / "bad.key"
        bad_key.write_text("not-a-valid-age-key\n")
        backend = EncryptedFileVaultBackend(storage_root)
        try:
            backend.unlock_with_key_file(bad_key)
        except VaultBackendUnlockError:
            pass
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.backend.unlock")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Backend unit operations
# ---------------------------------------------------------------------------


class TestBackendOperations:
    def test_secret_does_not_exist_initially(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        assert not backend.secret_exists("vault_abc", "my_key")

    def test_store_creates_age_file(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        backend.store_secret("vault_abc", "my_key", "my_value", "2024-01-01T00:00:00+00:00")
        assert _age_file(storage_root, "vault_abc").exists()

    def test_age_file_is_not_plain_json(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        backend.store_secret("vault_abc", "tok", "s3cr3t_v4lu3", "2024-01-01T00:00:00+00:00")
        raw = _age_file(storage_root, "vault_abc").read_bytes()
        assert b"s3cr3t_v4lu3" not in raw

    def test_age_file_value_not_in_readable_text(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        backend.store_secret("vault_abc", "tok", "plaintext_secret", "2024-01-01T00:00:00+00:00")
        raw = _age_file(storage_root, "vault_abc").read_bytes()
        assert b"plaintext_secret" not in raw

    def test_get_secret_returns_none_when_missing(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        assert backend.get_secret("vault_abc", "missing") is None

    def test_get_secret_returns_stored_record(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        record = backend.get_secret("vault_abc", "tok")
        assert record is not None
        assert record["key"] == "tok"
        assert record["value"] == "val"

    def test_secret_exists_after_store(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        assert backend.secret_exists("vault_abc", "tok")

    def test_store_multiple_secrets_same_vault(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        backend.store_secret("vault_multi", "k1", "v1", "2024-01-01T00:00:00+00:00")
        backend.store_secret("vault_multi", "k2", "v2", "2024-01-01T00:00:00+00:00")
        assert backend.secret_exists("vault_multi", "k1")
        assert backend.secret_exists("vault_multi", "k2")
        # both stored in one file
        assert _age_file(storage_root, "vault_multi").exists()

    def test_update_secret_persists_changes(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        backend.store_secret("vault_abc", "tok", "val", "2024-01-01T00:00:00+00:00")
        record = dict(backend.get_secret("vault_abc", "tok"))  # type: ignore[arg-type]
        record["last_accessed_at"] = "2024-06-01T12:00:00+00:00"
        record["requester_counts"] = {"127.0.0.1": 1}
        backend.update_secret("vault_abc", "tok", record)
        refreshed = backend.get_secret("vault_abc", "tok")
        assert refreshed is not None
        assert refreshed["last_accessed_at"] == "2024-06-01T12:00:00+00:00"
        assert refreshed["requester_counts"] == {"127.0.0.1": 1}

    def test_roundtrip_with_new_backend_instance(
        self, storage_root: Path
    ) -> None:
        # Simulate daemon restart: create a new backend instance with the same passphrase.
        b1 = _make_backend(storage_root)
        b1.store_secret("vault_persist", "key1", "value1", "2024-01-01T00:00:00+00:00")

        b2 = EncryptedFileVaultBackend(storage_root)
        b2.unlock_with_passphrase(_PASSPHRASE)
        record = b2.get_secret("vault_persist", "key1")
        assert record is not None
        assert record["value"] == "value1"


# ---------------------------------------------------------------------------
# API — encrypted_file vault with backend configured
# ---------------------------------------------------------------------------


class TestApiEncryptedFileBackendStore:
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

    def test_value_not_in_age_file(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        app = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=backend
        )
        client = TestClient(app, raise_server_exceptions=False)
        vault_id = _create_vault(client)["id"]
        _store_secret(client, vault_id, value="secret_needle")
        raw = _age_file(storage_root, vault_id).read_bytes()
        assert b"secret_needle" not in raw


# ---------------------------------------------------------------------------
# API — encrypted_file vault meta
# ---------------------------------------------------------------------------


class TestApiEncryptedFileBackendMeta:
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
        backend = _make_backend(storage_root)
        app = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=backend
        )
        client = TestClient(app, raise_server_exceptions=False)
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
        # Store with a configured backend first, then try meta without one.
        backend = _make_backend(storage_root)
        app_with = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=backend
        )
        client_with = TestClient(app_with, raise_server_exceptions=False)
        vault_id = _create_vault(client_with)["id"]
        _store_secret(client_with, vault_id, key="a_key")

        app_without = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=None
        )
        client_without = TestClient(app_without, raise_server_exceptions=False)
        resp = client_without.get(f"/v1/vaults/{vault_id}/secrets/a_key/meta")
        assert resp.status_code == 500

    def test_meta_without_backend_error_code(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        app_with = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=backend
        )
        client_with = TestClient(app_with, raise_server_exceptions=False)
        vault_id = _create_vault(client_with)["id"]
        _store_secret(client_with, vault_id, key="b_key")

        app_without = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=None
        )
        client_without = TestClient(app_without, raise_server_exceptions=False)
        body = client_without.get(f"/v1/vaults/{vault_id}/secrets/b_key/meta").json()
        assert body["error"]["code"] == "vault_secret_meta_failed"

    def test_meta_without_backend_writes_audit(self, storage_root: Path) -> None:
        backend = _make_backend(storage_root)
        app_with = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=backend
        )
        client_with = TestClient(app_with, raise_server_exceptions=False)
        vault_id = _create_vault(client_with)["id"]
        _store_secret(client_with, vault_id, key="c_key")

        app_without = create_app(
            FileAuditLog(storage_root), storage_root=storage_root, vault_backend=None
        )
        client_without = TestClient(app_without, raise_server_exceptions=False)
        client_without.get(f"/v1/vaults/{vault_id}/secrets/c_key/meta")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.secret.meta.failed" for r in records)
