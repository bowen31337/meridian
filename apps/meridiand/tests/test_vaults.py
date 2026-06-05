"""
Vaults endpoint conformance suite.

Tests cover:
  - POST /v1/vaults returns 201 on success.
  - Response has id with "vault_" prefix.
  - IDs are unique across calls.
  - Response has name, backend, created_at.
  - backend "os_keychain" accepted.
  - backend "encrypted_file" accepted.
  - Vault JSON written to storage_root/vaults/{id}.json.
  - Persisted record has correct name and backend.
  - Not written to disk on validation failure.
  - Empty name returns 422 with code "vault_invalid_request".
  - Missing required fields return 422.
  - Invalid backend returns 422.
  - On validation failure, audit log entry written with event "vault.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "vault_invalid_request" on validation failure.
  - Audit detail includes vault_id, name, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "vault.create" emitted on success.
  - OTel span "vault.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries vault.id, vault.name, vault.backend attributes.
  - create_app wires vaults router when storage_root is supplied.
  - create_app omits vaults route when storage_root is None.
  - GET /v1/vaults returns 200 with items list.
  - GET returns empty items when no vaults exist.
  - GET returns all created vaults.
  - Each item in GET response has id, name, backend, created_at.
  - OTel span "vault.list" emitted on GET.
  - DELETE /v1/vaults/{id} returns 204 on success.
  - DELETE response has no body.
  - DELETE returns 404 with code "vault_not_found" for unknown id.
  - DELETE returns 409 with code "vault_in_use" when channel references it.
  - DELETE removes vault JSON from storage.
  - Second DELETE on same id returns 404.
  - DELETE failure writes audit log entry with event "vault.delete.failed".
  - DELETE audit entry level is "error" on failure.
  - DELETE audit detail includes vault_id and message.
  - OTel span "vault.delete" emitted on success.
  - OTel span "vault.delete" emitted on failure.
  - OTel span set to ERROR status on DELETE failure.
  - Span carries vault.id attribute on DELETE.
  - DELETE route present with storage_root; absent without.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _body(**overrides) -> dict:
    base: dict = {"name": "my-vault", "backend": "os_keychain"}
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _vault_resource(storage_root: Path, vault_id: str) -> dict:
    return json.loads((storage_root / "vaults" / f"{vault_id}.json").read_text())


def _create_vault(client: TestClient, **overrides) -> dict:
    return client.post("/v1/vaults", json=_body(**overrides)).json()


# ---------------------------------------------------------------------------
# POST success
# ---------------------------------------------------------------------------


class TestVaultCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json=_body())
        assert resp.status_code == 201

    def test_os_keychain_backend_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json=_body(backend="os_keychain"))
        assert resp.status_code == 201

    def test_encrypted_file_backend_accepted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json=_body(backend="encrypted_file"))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# POST response fields
# ---------------------------------------------------------------------------


class TestVaultCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/vaults", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_vault_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/vaults", json=_body()).json()
        assert body["id"].startswith("vault_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/vaults", json=_body(name="vault-a")).json()["id"]
        id2 = client.post("/v1/vaults", json=_body(name="vault-b")).json()["id"]
        assert id1 != id2

    def test_response_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/vaults", json=_body(name="my-secrets")).json()
        assert body["name"] == "my-secrets"

    def test_response_has_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/vaults", json=_body(backend="encrypted_file")).json()
        assert body["backend"] == "encrypted_file"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/vaults", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0


# ---------------------------------------------------------------------------
# POST persistence
# ---------------------------------------------------------------------------


class TestVaultPersistence:
    def test_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = client.post("/v1/vaults", json=_body()).json()["id"]
        assert (storage_root / "vaults" / f"{vault_id}.json").exists()

    def test_persisted_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = client.post("/v1/vaults", json=_body(name="persist-vault")).json()["id"]
        resource = _vault_resource(storage_root, vault_id)
        assert resource["name"] == "persist-vault"

    def test_persisted_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = client.post("/v1/vaults", json=_body(backend="encrypted_file")).json()["id"]
        resource = _vault_resource(storage_root, vault_id)
        assert resource["backend"] == "encrypted_file"

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        vaults_dir = storage_root / "vaults"
        files = list(vaults_dir.glob("*.json")) if vaults_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# POST validation errors
# ---------------------------------------------------------------------------


class TestVaultCreateValidation:
    def test_missing_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json={"backend": "os_keychain"})
        assert resp.status_code == 422

    def test_missing_backend_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json={"name": "my-vault"})
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json=_body(name="   "))
        assert resp.status_code == 422

    def test_empty_name_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/vaults", json=_body(name="")).json()
        assert body["error"]["code"] == "vault_invalid_request"

    def test_empty_name_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/vaults", json=_body(name="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_invalid_backend_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json=_body(backend="aws_kms"))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST audit log
# ---------------------------------------------------------------------------


class TestVaultCreateAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.create.failed"
        )
        assert record["code"] == "vault_invalid_request"

    def test_failure_audit_detail_has_vault_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.create.failed"
        )
        assert record["detail"]["vault_id"].startswith("vault_")

    def test_failure_audit_detail_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.create.failed"
        )
        assert "name" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# POST OTel spans
# ---------------------------------------------------------------------------


class TestVaultCreateOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_vault_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/vaults", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.create" in span_names

    def test_failure_emits_vault_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/vaults", json=_body(name=""))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_vault_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/vaults", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.create")
        assert span is not None
        assert span.attributes["vault.id"].startswith("vault_")

    def test_success_span_has_name_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/vaults", json=_body(name="otel-vault"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.create")
        assert span is not None
        assert span.attributes["vault.name"] == "otel-vault"

    def test_success_span_has_backend_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/vaults", json=_body(backend="encrypted_file"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.create")
        assert span is not None
        assert span.attributes["vault.backend"] == "encrypted_file"


# ---------------------------------------------------------------------------
# POST route wiring
# ---------------------------------------------------------------------------


class TestVaultCreateRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/vaults", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/vaults", json=_body())
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET list success
# ---------------------------------------------------------------------------


class TestVaultListSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/vaults")
        assert resp.status_code == 200

    def test_empty_items_when_no_vaults(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/vaults").json()
        assert body["items"] == []

    def test_returns_all_created_vaults(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name="vault-a"))
        client.post("/v1/vaults", json=_body(name="vault-b"))
        body = client.get("/v1/vaults").json()
        assert len(body["items"]) == 2

    def test_single_vault_in_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = client.post("/v1/vaults", json=_body(name="solo")).json()["id"]
        body = client.get("/v1/vaults").json()
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == vault_id

    def test_item_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body())
        item = client.get("/v1/vaults").json()["items"][0]
        assert "id" in item
        assert item["id"].startswith("vault_")

    def test_item_has_name(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(name="named-vault"))
        item = client.get("/v1/vaults").json()["items"][0]
        assert item["name"] == "named-vault"

    def test_item_has_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body(backend="encrypted_file"))
        item = client.get("/v1/vaults").json()["items"][0]
        assert item["backend"] == "encrypted_file"

    def test_item_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/vaults", json=_body())
        item = client.get("/v1/vaults").json()["items"][0]
        assert "created_at" in item
        assert isinstance(item["created_at"], str)
        assert len(item["created_at"]) > 0


# ---------------------------------------------------------------------------
# GET OTel spans
# ---------------------------------------------------------------------------


class TestVaultListOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_emits_vault_list_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.get("/v1/vaults")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.list" in span_names


# ---------------------------------------------------------------------------
# GET route wiring
# ---------------------------------------------------------------------------


class TestVaultListRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/vaults")
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/vaults")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE success
# ---------------------------------------------------------------------------


class TestVaultDeleteSuccess:
    def test_delete_returns_204(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.delete(f"/v1/vaults/{vault_id}")
        assert resp.status_code == 204

    def test_delete_response_has_no_body(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.delete(f"/v1/vaults/{vault_id}")
        assert resp.content == b""

    def test_delete_removes_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.delete(f"/v1/vaults/{vault_id}")
        assert not (storage_root / "vaults" / f"{vault_id}.json").exists()

    def test_deleted_vault_absent_from_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.delete(f"/v1/vaults/{vault_id}")
        items = client.get("/v1/vaults").json()["items"]
        assert all(item["id"] != vault_id for item in items)

    def test_second_delete_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        client.delete(f"/v1/vaults/{vault_id}")
        resp = client.delete(f"/v1/vaults/{vault_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE not found
# ---------------------------------------------------------------------------


class TestVaultDeleteNotFound:
    def test_unknown_id_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.delete("/v1/vaults/vault_nonexistent")
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/vaults/vault_nonexistent").json()
        assert body["error"]["code"] == "vault_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.delete("/v1/vaults/vault_nonexistent").json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# DELETE conflict: vault in use by channel
# ---------------------------------------------------------------------------


class TestVaultDeleteInUse:
    def _create_channel_referencing(self, storage_root: Path, vault_id: str) -> None:
        channels_dir = storage_root / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": "ch_test",
            "kind": "slack",
            "config": {"token_vault_ref": vault_id},
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        (channels_dir / "ch_test.json").write_text(json.dumps(record))

    def test_channel_reference_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        self._create_channel_referencing(storage_root, vault_id)
        resp = client.delete(f"/v1/vaults/{vault_id}")
        assert resp.status_code == 409

    def test_channel_reference_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        self._create_channel_referencing(storage_root, vault_id)
        body = client.delete(f"/v1/vaults/{vault_id}").json()
        assert body["error"]["code"] == "vault_in_use"

    def test_channel_reference_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        self._create_channel_referencing(storage_root, vault_id)
        body = client.delete(f"/v1/vaults/{vault_id}").json()
        assert len(body["error"]["message"]) > 0

    def test_channel_ref_with_secret_path_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        channels_dir = storage_root / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": "ch_path",
            "kind": "slack",
            "config": {"token_vault_ref": f"{vault_id}/slack-token"},
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        (channels_dir / "ch_path.json").write_text(json.dumps(record))
        resp = client.delete(f"/v1/vaults/{vault_id}")
        assert resp.status_code == 409

    def test_unrelated_channel_allows_delete(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        channels_dir = storage_root / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": "ch_other",
            "kind": "slack",
            "config": {"token_vault_ref": "vault_other/some-token"},
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        (channels_dir / "ch_other.json").write_text(json.dumps(record))
        resp = client.delete(f"/v1/vaults/{vault_id}")
        assert resp.status_code == 204

    def test_in_use_vault_file_not_removed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        self._create_channel_referencing(storage_root, vault_id)
        client.delete(f"/v1/vaults/{vault_id}")
        assert (storage_root / "vaults" / f"{vault_id}.json").exists()


# ---------------------------------------------------------------------------
# DELETE audit log
# ---------------------------------------------------------------------------


class TestVaultDeleteAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/vaults/vault_missing")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.delete.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/vaults/vault_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.delete.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/vaults/vault_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.delete.failed"
        )
        assert record["code"] == "vault_not_found"

    def test_not_found_audit_detail_has_vault_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/vaults/vault_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.delete.failed"
        )
        assert record["detail"]["vault_id"] == "vault_missing"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.delete("/v1/vaults/vault_missing")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.delete.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_in_use_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        channels_dir = storage_root / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": "ch_audit",
            "kind": "slack",
            "config": {"token_vault_ref": vault_id},
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        (channels_dir / "ch_audit.json").write_text(json.dumps(record))
        client.delete(f"/v1/vaults/{vault_id}")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "vault.delete.failed" for r in records)

    def test_in_use_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        channels_dir = storage_root / "channels"
        channels_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "id": "ch_audit2",
            "kind": "slack",
            "config": {"token_vault_ref": vault_id},
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        (channels_dir / "ch_audit2.json").write_text(json.dumps(record))
        client.delete(f"/v1/vaults/{vault_id}")
        audit_record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "vault.delete.failed"
        )
        assert audit_record["code"] == "vault_in_use"


# ---------------------------------------------------------------------------
# DELETE OTel spans
# ---------------------------------------------------------------------------


class TestVaultDeleteOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/vaults/{vault_id}")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.delete" in span_names

    def test_not_found_emits_delete_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.delete("/v1/vaults/vault_missing")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "vault.delete" in span_names

    def test_not_found_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.delete("/v1/vaults/vault_missing")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.delete")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_vault_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        vault_id = _create_vault(client)["id"]
        _otel_exporter.clear()
        client.delete(f"/v1/vaults/{vault_id}")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("vault.delete")
        assert span is not None
        assert span.attributes["vault.id"] == vault_id


# ---------------------------------------------------------------------------
# DELETE route wiring
# ---------------------------------------------------------------------------


class TestVaultDeleteRouteWiring:
    def test_delete_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        vault_id = _create_vault(client)["id"]
        resp = client.delete(f"/v1/vaults/{vault_id}")
        assert resp.status_code != 404

    def test_delete_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/v1/vaults/vault_any")
        assert resp.status_code == 404
