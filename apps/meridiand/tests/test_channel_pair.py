"""
Channel pair endpoint conformance suite.

Tests cover:
  - POST /v1/channels/{id}/pair returns 201 on success with empty body.
  - POST /v1/channels/{id}/pair returns 201 with user_profile_id.
  - Response has token with "pair_" prefix.
  - Tokens are unique across calls.
  - Response has channel_id, user_profile_id, created_at.
  - user_profile_id is null when omitted; stored when provided.
  - Pairing token JSON written to storage_root/pairing_tokens/{token}.json.
  - Persisted token has redeemed=false.
  - Channel not found returns 404 with code "channel_not_found".
  - Error response body has error.code and error.message on failure.
  - On failure, audit log entry written with event "channel.pair.failed".
  - Audit entry level is "error" on failure.
  - Audit entry code is "channel_not_found" on not found.
  - Audit detail includes channel_id and message on failure.
  - OTel span "channel.pair" emitted on success.
  - OTel span "channel.pair" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries channel.id attribute.
  - Route present with storage_root; absent without.
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


def _create_channel(client: TestClient) -> str:
    resp = client.post(
        "/v1/channels",
        json={"kind": "slack", "config": {"token_vault_ref": "vaults/main/tok"}},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _token_resource(storage_root: Path, token: str) -> dict:
    return json.loads((storage_root / "pairing_tokens" / f"{token}.json").read_text())


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestChannelPairSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        resp = client.post(f"/v1/channels/{channel_id}/pair", json={})
        assert resp.status_code == 201

    def test_with_user_profile_id_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        resp = client.post(f"/v1/channels/{channel_id}/pair", json={"user_profile_id": "user_abc"})
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Response fields
# ---------------------------------------------------------------------------


class TestChannelPairResponse:
    def test_response_has_token(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        body = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()
        assert "token" in body
        assert isinstance(body["token"], str)

    def test_token_has_pair_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        body = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()
        assert body["token"].startswith("pair_")

    def test_tokens_are_unique(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        t1 = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()["token"]
        t2 = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()["token"]
        assert t1 != t2

    def test_response_has_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        body = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()
        assert body["channel_id"] == channel_id

    def test_user_profile_id_null_when_omitted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        body = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()
        assert body["user_profile_id"] is None

    def test_user_profile_id_stored_when_provided(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        body = client.post(
            f"/v1/channels/{channel_id}/pair", json={"user_profile_id": "user_xyz"}
        ).json()
        assert body["user_profile_id"] == "user_xyz"

    def test_response_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        body = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()
        assert "created_at" in body
        assert isinstance(body["created_at"], str)
        assert len(body["created_at"]) > 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestChannelPairPersistence:
    def test_token_json_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()["token"]
        assert (storage_root / "pairing_tokens" / f"{token}.json").exists()

    def test_persisted_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()["token"]
        resource = _token_resource(storage_root, token)
        assert resource["channel_id"] == channel_id

    def test_persisted_redeemed_is_false(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = client.post(f"/v1/channels/{channel_id}/pair", json={}).json()["token"]
        resource = _token_resource(storage_root, token)
        assert resource["redeemed"] is False

    def test_persisted_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = client.post(
            f"/v1/channels/{channel_id}/pair", json={"user_profile_id": "user_abc"}
        ).json()["token"]
        resource = _token_resource(storage_root, token)
        assert resource["user_profile_id"] == "user_abc"

    def test_not_written_when_channel_not_found(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels/ch_nonexistent/pair", json={})
        tokens_dir = storage_root / "pairing_tokens"
        files = list(tokens_dir.glob("*.json")) if tokens_dir.exists() else []
        assert files == []


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------


class TestChannelPairNotFound:
    def test_unknown_channel_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/channels/ch_nonexistent/pair", json={})
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels/ch_nonexistent/pair", json={}).json()
        assert body["error"]["code"] == "channel_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/channels/ch_nonexistent/pair", json={}).json()
        assert len(body["error"]["message"]) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestChannelPairAuditLog:
    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels/ch_missing/pair", json={})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.pair.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels/ch_missing/pair", json={})
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.pair.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels/ch_missing/pair", json={})
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.pair.failed"
        )
        assert record["code"] == "channel_not_found"

    def test_audit_detail_has_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels/ch_missing/pair", json={})
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.pair.failed"
        )
        assert record["detail"]["channel_id"] == "ch_missing"

    def test_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/channels/ch_missing/pair", json={})
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "channel.pair.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestChannelPairOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_channel_pair_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        client.post(f"/v1/channels/{channel_id}/pair", json={})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "channel.pair" in span_names

    def test_failure_emits_channel_pair_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/channels/ch_nonexistent/pair", json={})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "channel.pair" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        client.post("/v1/channels/ch_nonexistent/pair", json={})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.pair")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        client.post(f"/v1/channels/{channel_id}/pair", json={})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.pair")
        assert span is not None
        assert span.attributes["channel.id"] == channel_id

    def test_failure_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        client.post("/v1/channels/ch_nonexistent/pair", json={})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.pair")
        assert span is not None
        assert span.attributes["channel.id"] == "ch_nonexistent"


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestChannelPairRouteWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        resp = client.post(f"/v1/channels/{channel_id}/pair", json={})
        assert resp.status_code != 404

    def test_route_absent_without_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root))
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/channels/ch_any/pair", json={})
        assert resp.status_code == 404
