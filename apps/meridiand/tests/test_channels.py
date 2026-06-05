"""
Channels endpoint conformance suite.

Tests cover:
  - POST /v1/channels returns 201 on success.
  - Response has id with "ch_" prefix.
  - IDs are unique across calls.
  - Response has kind, config, default_agent_id, default_user_profile_id,
    inbound_policy, egress_policy, created_at, updated_at.
  - default_agent_id is null when omitted; stored when provided.
  - default_user_profile_id is null when omitted; stored when provided.
  - inbound_policy defaults to "open".
  - egress_policy defaults to "enabled".
  - inbound_policy stored when provided (paired_only, quarantine).
  - egress_policy stored when provided (disabled).
  - Channel JSON written to storage_root/channels/{id}.json.
  - Persisted channel has correct kind.
  - Not written to disk on validation failure.
  - Empty kind returns 422 with code "channel_invalid_request".
  - Missing token_vault_ref in config returns 422.
  - Empty token_vault_ref in config returns 422.
  - Missing required fields return 422.
  - On validation failure, audit log entry written with event "channel.create.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "channel_invalid_request" on validation failure.
  - Audit detail includes channel_id, kind, message on failure.
  - Error response body has error.code and error.message on failure.
  - OTel span "channel.create" emitted on success.
  - OTel span "channel.create" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries channel.id and channel.kind attributes.
  - create_app wires channels router when storage_root is supplied.
  - create_app omits channels route when storage_root is None.
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
    base: dict = {
        "kind": "slack",
        "config": {"token_vault_ref": "vaults/main/slack-token"},
    }
    base.update(overrides)
    return base


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _channel_resource(storage_root: Path, channel_id: str) -> dict:
    return json.loads((storage_root / "channels" / f"{channel_id}.json").read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestChannelCreateSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body())
        assert resp.status_code == 201

    def test_with_default_agent_id_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(default_agent_id="agent_abc"))
        assert resp.status_code == 201

    def test_with_default_user_profile_id_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(default_user_profile_id="user_xyz"))
        assert resp.status_code == 201

    def test_with_inbound_policy_paired_only_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(inbound_policy="paired_only"))
        assert resp.status_code == 201

    def test_with_inbound_policy_quarantine_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(inbound_policy="quarantine"))
        assert resp.status_code == 201

    def test_with_egress_policy_disabled_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(egress_policy="disabled"))
        assert resp.status_code == 201

    def test_with_extra_config_fields_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        config = {"token_vault_ref": "vaults/main/tok", "workspace_id": "W123", "bot_name": "mybot"}
        resp = client.post("/v1/channels", json=_body(config=config))
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestChannelCreateResponse:
    def test_response_has_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert "id" in body
        assert isinstance(body["id"], str)

    def test_id_has_ch_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert body["id"].startswith("ch_")

    def test_ids_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        id1 = client.post("/v1/channels", json=_body()).json()["id"]
        id2 = client.post("/v1/channels", json=_body()).json()["id"]
        assert id1 != id2

    def test_response_has_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(kind="telegram")).json()
        assert body["kind"] == "telegram"

    def test_response_has_config(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        config = {"token_vault_ref": "vaults/main/tok", "extra": "val"}
        body = client.post("/v1/channels", json=_body(config=config)).json()
        assert body["config"] == config

    def test_default_agent_id_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert body["default_agent_id"] is None

    def test_default_agent_id_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(default_agent_id="agent_123")).json()
        assert body["default_agent_id"] == "agent_123"

    def test_default_user_profile_id_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert body["default_user_profile_id"] is None

    def test_default_user_profile_id_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(default_user_profile_id="user_456")).json()
        assert body["default_user_profile_id"] == "user_456"

    def test_inbound_policy_defaults_to_open(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert body["inbound_policy"] == "open"

    def test_inbound_policy_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(inbound_policy="quarantine")).json()
        assert body["inbound_policy"] == "quarantine"

    def test_egress_policy_defaults_to_enabled(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert body["egress_policy"] == "enabled"

    def test_egress_policy_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(egress_policy="disabled")).json()
        assert body["egress_policy"] == "disabled"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0

    def test_response_has_updated_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body()).json()
        assert "updated_at" in body
        assert isinstance(body["updated_at"], str)
        assert len(body["updated_at"]) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestChannelPersistence:
    def test_channel_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = client.post("/v1/channels", json=_body()).json()["id"]
        assert (storage_root / "channels" / f"{channel_id}.json").exists()

    def test_persisted_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = client.post("/v1/channels", json=_body(kind="discord")).json()["id"]
        resource = _channel_resource(storage_root, channel_id)
        assert resource["kind"] == "discord"

    def test_persisted_config(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        config = {"token_vault_ref": "vaults/v/tok", "region": "us-east-1"}
        channel_id = client.post("/v1/channels", json=_body(config=config)).json()["id"]
        resource = _channel_resource(storage_root, channel_id)
        assert resource["config"] == config

    def test_persisted_inbound_policy(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = client.post("/v1/channels", json=_body(inbound_policy="paired_only")).json()[
            "id"
        ]
        resource = _channel_resource(storage_root, channel_id)
        assert resource["inbound_policy"] == "paired_only"

    def test_persisted_egress_policy(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = client.post("/v1/channels", json=_body(egress_policy="disabled")).json()["id"]
        resource = _channel_resource(storage_root, channel_id)
        assert resource["egress_policy"] == "disabled"

    def test_not_written_on_validation_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        channels_dir = storage_root / "channels"
        files = list(channels_dir.glob("*.json")) if channels_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestChannelCreateValidation:
    def test_missing_kind_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json={"config": {"token_vault_ref": "v/t"}})
        assert resp.status_code == 422

    def test_empty_kind_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(kind="   "))
        assert resp.status_code == 422

    def test_empty_kind_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(kind="")).json()
        assert body["error"]["code"] == "channel_invalid_request"

    def test_missing_config_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json={"kind": "slack"})
        assert resp.status_code == 422

    def test_missing_token_vault_ref_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(config={"workspace_id": "W123"}))
        assert resp.status_code == 422

    def test_missing_token_vault_ref_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(config={"workspace_id": "W123"})).json()
        assert body["error"]["code"] == "channel_invalid_request"

    def test_empty_token_vault_ref_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(config={"token_vault_ref": "   "}))
        assert resp.status_code == 422

    def test_validation_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels", json=_body(kind="")).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_invalid_inbound_policy_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(inbound_policy="unknown"))
        assert resp.status_code == 422

    def test_invalid_egress_policy_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body(egress_policy="unknown"))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestChannelAuditLog:
    def test_validation_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.create.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.create.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_code_is_channel_invalid_request(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.create.failed"
        )
        assert record["code"] == "channel_invalid_request"

    def test_failure_audit_detail_has_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.create.failed"
        )
        assert record["detail"]["channel_id"].startswith("ch_")

    def test_failure_audit_detail_has_kind(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.create.failed"
        )
        assert "kind" in record["detail"]

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.create.failed"
        )
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestChannelOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_channel_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/channels", json=_body())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "channel.create" in span_names

    def test_failure_emits_channel_create_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "channel.create" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/channels", json=_body(kind=""))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.create")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/channels", json=_body())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.create")
        assert span is not None
        assert span.attributes["channel.id"].startswith("ch_")

    def test_success_span_has_channel_kind_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/channels", json=_body(kind="telegram"))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.create")
        assert span is not None
        assert span.attributes["channel.kind"] == "telegram"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestChannelRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels", json=_body())
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/channels", json=_body())
        assert resp.status_code == 404
