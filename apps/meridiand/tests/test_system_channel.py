"""
System Channel conformance suite.

Tests cover:
  - Pairing round-trip: every registered driver completes create-channel →
    issue-token → redeem-token → inbound-resolve → outbound-deliver → confirmed.
  - Inbound resolution: sender resolves to the right UserProfile, Agent, and
    Session under the open and paired policies.
  - Outbound delivery confirmation: POST /v1/channels/{id}/outbound returns
    delivered=true via the registered ChannelDriver.
  - Untrusted-inbound quarantine: quarantine policy stores the message and
    returns quarantined=true without writing an audit entry.
  - Paired-only policy rejection: unpaired senders receive 403 with correct
    error code, and an audit entry is written.
  - OTel spans: channel.inbound, channel.outbound, and channel.pair.redeem
    spans are emitted with correct attributes and a structured invocation event.
  - Error surfacing: on failure each endpoint returns a structured error
    response and writes an audit entry.
  - Route wiring: routes are present when channel_runtime is supplied and
    absent when it is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from sdk_channel import (
    ChannelCapabilities,
    ChannelDriver,
    ChannelRuntime,
    SendRequest,
    SendResult,
    StartRequest,
    StopRequest,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Stub driver — records every call
# ---------------------------------------------------------------------------


class StubDriver(ChannelDriver):
    kind = "test.system"

    def __init__(self) -> None:
        self.starts: list[StartRequest] = []
        self.sends: list[SendRequest] = []
        self.stops: list[StopRequest] = []

    async def start(self, request: StartRequest) -> None:
        self.starts.append(request)

    async def send(self, request: SendRequest) -> SendResult:
        self.sends.append(request)
        return SendResult(
            message_id="msg-system-1",
            timestamp="2026-01-01T00:00:00+00:00",
            delivered=True,
        )

    async def stop(self, request: StopRequest) -> None:
        self.stops.append(request)

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(can_send_text=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(driver: StubDriver | None = None) -> ChannelRuntime:
    rt = ChannelRuntime()
    rt.register(driver or StubDriver())
    return rt


def _make_client(storage_root: Path, runtime: ChannelRuntime | None = None) -> TestClient:
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        channel_runtime=runtime or _make_runtime(),
    )
    return TestClient(app, raise_server_exceptions=False)


def _create_channel(
    client: TestClient,
    *,
    kind: str = "test.system",
    inbound_policy: str = "open",
    egress_policy: str = "enabled",
    default_user_profile_id: str | None = "user_default",
    default_agent_id: str | None = "agent_default",
) -> str:
    body: dict = {
        "kind": kind,
        "config": {"token_vault_ref": "vaults/main/tok"},
        "inbound_policy": inbound_policy,
        "egress_policy": egress_policy,
    }
    if default_user_profile_id is not None:
        body["default_user_profile_id"] = default_user_profile_id
    if default_agent_id is not None:
        body["default_agent_id"] = default_agent_id
    resp = client.post("/v1/channels", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _issue_pairing_token(client: TestClient, channel_id: str, user_profile_id: str | None = None) -> str:
    body: dict = {}
    if user_profile_id is not None:
        body["user_profile_id"] = user_profile_id
    resp = client.post(f"/v1/channels/{channel_id}/pair", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["token"]


def _redeem_token(client: TestClient, token: str, sender_id: str) -> dict:
    resp = client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": sender_id})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _inbound(client: TestClient, channel_id: str, sender_id: str, content: str = "hello") -> dict:
    resp = client.post(
        f"/v1/channels/{channel_id}/inbound",
        json={"sender_id": sender_id, "content": content},
    )
    return resp.json() | {"_status": resp.status_code}


def _outbound(
    client: TestClient,
    channel_id: str,
    *,
    session_id: str = "sess_test",
    recipient: str = "ext-user-1",
    content: str = "reply",
) -> dict:
    resp = client.post(
        f"/v1/channels/{channel_id}/outbound",
        json={"session_id": session_id, "recipient": recipient, "content": content},
    )
    return resp.json() | {"_status": resp.status_code}


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Pairing round-trip
# ---------------------------------------------------------------------------


class TestPairingRoundTrip:
    """
    Every driver passes the full pairing round-trip:
    create → pair → redeem → inbound resolves → outbound delivers.
    """

    def test_redeem_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        resp = client.post(
            f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"}
        )
        assert resp.status_code == 200

    def test_redeem_response_has_token(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        body = client.post(
            f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"}
        ).json()
        assert body["token"] == token

    def test_redeem_response_has_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        body = client.post(
            f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"}
        ).json()
        assert body["channel_id"] == channel_id

    def test_redeem_response_has_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        body = client.post(
            f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"}
        ).json()
        assert body["user_profile_id"] == "user_abc"

    def test_redeem_response_has_sender_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        body = client.post(
            f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"}
        ).json()
        assert body["sender_id"] == "ext-1"

    def test_redeem_creates_pairing_record(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"})
        pairing_file = storage_root / "channel_pairings" / channel_id / "ext-1.json"
        assert pairing_file.exists()

    def test_redeem_marks_token_redeemed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"})
        token_record = json.loads(
            (storage_root / "pairing_tokens" / f"{token}.json").read_text()
        )
        assert token_record["redeemed"] is True

    def test_paired_inbound_resolves_to_paired_user_profile(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, default_user_profile_id="user_default")
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_paired")
        _redeem_token(client, token, "ext-user-1")
        result = _inbound(client, channel_id, "ext-user-1")
        assert result["_status"] == 200
        assert result["user_profile_id"] == "user_paired"

    def test_paired_inbound_resolves_to_default_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, default_agent_id="agent_xyz")
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        result = _inbound(client, channel_id, "ext-user-1")
        assert result["agent_id"] == "agent_xyz"

    def test_paired_inbound_creates_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        result = _inbound(client, channel_id, "ext-user-1")
        assert "session_id" in result
        assert result["session_id"].startswith("sess_")

    def test_paired_inbound_not_quarantined(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        result = _inbound(client, channel_id, "ext-user-1")
        assert result["quarantined"] is False

    def test_outbound_delivery_confirmed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _outbound(client, channel_id)
        assert result["_status"] == 200
        assert result["delivered"] is True

    def test_outbound_has_message_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _outbound(client, channel_id)
        assert "message_id" in result
        assert isinstance(result["message_id"], str)
        assert len(result["message_id"]) > 0

    def test_outbound_has_timestamp(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _outbound(client, channel_id)
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)

    def test_driver_send_called_on_outbound(self, storage_root: Path) -> None:
        driver = StubDriver()
        rt = _make_runtime(driver)
        client = _make_client(storage_root, rt)
        channel_id = _create_channel(client)
        _outbound(client, channel_id, session_id="sess_abc", recipient="ext-user-1")
        assert len(driver.sends) == 1
        assert driver.sends[0].channel_id == channel_id
        assert driver.sends[0].recipient == "ext-user-1"


# ---------------------------------------------------------------------------
# Inbound resolution — open policy
# ---------------------------------------------------------------------------


class TestInboundResolutionOpen:
    def test_open_policy_uses_default_user_profile(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, default_user_profile_id="user_default_42")
        result = _inbound(client, channel_id, "unknown-sender")
        assert result["_status"] == 200
        assert result["user_profile_id"] == "user_default_42"

    def test_open_policy_uses_default_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, default_agent_id="agent_default_7")
        result = _inbound(client, channel_id, "unknown-sender")
        assert result["agent_id"] == "agent_default_7"

    def test_open_policy_creates_session_with_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _inbound(client, channel_id, "any-sender")
        assert result["session_id"].startswith("sess_")

    def test_open_policy_session_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _inbound(client, channel_id, "any-sender")
        session_id = result["session_id"]
        session_file = storage_root / "channel_sessions" / channel_id / f"{session_id}.json"
        assert session_file.exists()

    def test_open_policy_session_contains_correct_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(
            client,
            default_user_profile_id="user_def",
            default_agent_id="agent_def",
        )
        result = _inbound(client, channel_id, "sender-x")
        session_id = result["session_id"]
        session = json.loads(
            (storage_root / "channel_sessions" / channel_id / f"{session_id}.json").read_text()
        )
        assert session["channel_id"] == channel_id
        assert session["user_profile_id"] == "user_def"
        assert session["agent_id"] == "agent_def"
        assert session["sender_id"] == "sender-x"

    def test_open_policy_returns_not_quarantined(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _inbound(client, channel_id, "any-sender")
        assert result["quarantined"] is False

    def test_open_policy_no_audit_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _inbound(client, channel_id, "any-sender")
        records = _audit_records(storage_root)
        inbound_failures = [r for r in records if r.get("event") == "channel.inbound.failed"]
        assert inbound_failures == []


# ---------------------------------------------------------------------------
# Untrusted-inbound quarantine
# ---------------------------------------------------------------------------


class TestUntrustedInboundQuarantine:
    def _quarantine_channel(self, client: TestClient) -> str:
        return _create_channel(client, inbound_policy="quarantine")

    def test_quarantine_policy_returns_quarantined_true(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        result = _inbound(client, channel_id, "untrusted-sender")
        assert result["_status"] == 200
        assert result["quarantined"] is True

    def test_quarantine_policy_returns_quarantine_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        result = _inbound(client, channel_id, "untrusted-sender")
        assert "quarantine_id" in result
        assert result["quarantine_id"].startswith("quar_")

    def test_quarantine_record_written_to_storage(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        result = _inbound(client, channel_id, "untrusted-sender")
        quarantine_id = result["quarantine_id"]
        q_file = storage_root / "channel_quarantine" / channel_id / f"{quarantine_id}.json"
        assert q_file.exists()

    def test_quarantine_record_has_correct_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        result = _inbound(client, channel_id, "untrusted-sender", content="bad message")
        quarantine_id = result["quarantine_id"]
        q_record = json.loads(
            (
                storage_root / "channel_quarantine" / channel_id / f"{quarantine_id}.json"
            ).read_text()
        )
        assert q_record["channel_id"] == channel_id
        assert q_record["sender_id"] == "untrusted-sender"
        assert q_record["content"] == "bad message"

    def test_quarantine_no_session_created(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        _inbound(client, channel_id, "untrusted-sender")
        sessions_dir = storage_root / "channel_sessions" / channel_id
        files = list(sessions_dir.glob("*.json")) if sessions_dir.exists() else []
        assert files == []

    def test_quarantine_no_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        _inbound(client, channel_id, "untrusted-sender")
        records = _audit_records(storage_root)
        inbound_failures = [r for r in records if r.get("event") == "channel.inbound.failed"]
        assert inbound_failures == []

    def test_paired_sender_bypasses_quarantine(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_trusted")
        _redeem_token(client, token, "trusted-sender")
        result = _inbound(client, channel_id, "trusted-sender")
        assert result["quarantined"] is False
        assert result["user_profile_id"] == "user_trusted"


# ---------------------------------------------------------------------------
# Paired-only policy
# ---------------------------------------------------------------------------


class TestPairedOnlyPolicy:
    def _paired_only_channel(self, client: TestClient) -> str:
        return _create_channel(client, inbound_policy="paired_only")

    def test_unpaired_sender_returns_403(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        resp = client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        )
        assert resp.status_code == 403

    def test_unpaired_sender_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        body = client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        ).json()
        assert body["error"]["code"] == "channel_inbound_policy_rejected"

    def test_unpaired_sender_error_has_message(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        body = client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        ).json()
        assert len(body["error"]["message"]) > 0

    def test_paired_only_rejection_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.inbound.failed" for r in records)

    def test_paired_only_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "channel.inbound.failed"
        )
        assert record["level"] == "error"

    def test_paired_only_audit_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "channel.inbound.failed"
        )
        assert record["code"] == "channel_inbound_policy_rejected"

    def test_paired_only_audit_detail_has_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "channel.inbound.failed"
        )
        assert record["detail"]["channel_id"] == channel_id

    def test_paired_sender_allowed_through_paired_only(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._paired_only_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_ok")
        _redeem_token(client, token, "known-sender")
        result = _inbound(client, channel_id, "known-sender")
        assert result["_status"] == 200
        assert result["user_profile_id"] == "user_ok"


# ---------------------------------------------------------------------------
# Inbound — channel not found
# ---------------------------------------------------------------------------


class TestInboundChannelNotFound:
    def test_unknown_channel_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/channels/ch_nonexistent/inbound",
            json={"sender_id": "ext-1", "content": "hi"},
        )
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/channels/ch_nonexistent/inbound",
            json={"sender_id": "ext-1", "content": "hi"},
        ).json()
        assert body["error"]["code"] == "channel_inbound_not_found"

    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/channels/ch_nonexistent/inbound",
            json={"sender_id": "ext-1", "content": "hi"},
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.inbound.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/channels/ch_nonexistent/inbound",
            json={"sender_id": "ext-1", "content": "hi"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "channel.inbound.failed"
        )
        assert record["level"] == "error"


# ---------------------------------------------------------------------------
# Outbound — errors
# ---------------------------------------------------------------------------


class TestOutboundErrors:
    def test_unknown_channel_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/channels/ch_nonexistent/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        assert resp.status_code == 404

    def test_unknown_channel_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/channels/ch_nonexistent/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        ).json()
        assert body["error"]["code"] == "channel_outbound_not_found"

    def test_unknown_channel_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/channels/ch_nonexistent/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.outbound.failed" for r in records)

    def test_egress_disabled_returns_422(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, egress_policy="disabled")
        resp = client.post(
            f"/v1/channels/{channel_id}/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        assert resp.status_code == 422

    def test_egress_disabled_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, egress_policy="disabled")
        body = client.post(
            f"/v1/channels/{channel_id}/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        ).json()
        assert body["error"]["code"] == "channel_outbound_disabled"

    def test_egress_disabled_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, egress_policy="disabled")
        client.post(
            f"/v1/channels/{channel_id}/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.outbound.failed" for r in records)

    def test_driver_failure_wrapped_as_outbound_error(self, storage_root: Path) -> None:
        class FailingDriver(ChannelDriver):
            kind = "test.failing"

            async def start(self, r: StartRequest) -> None:
                pass

            async def send(self, r: SendRequest) -> SendResult:
                raise RuntimeError("network down")

            async def stop(self, r: StopRequest) -> None:
                pass

            def capabilities(self) -> ChannelCapabilities:
                return ChannelCapabilities()

        rt = ChannelRuntime()
        rt.register(FailingDriver())
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root, channel_runtime=rt)
        client = TestClient(app, raise_server_exceptions=False)
        channel_id = _create_channel(client, kind="test.failing")
        resp = client.post(
            f"/v1/channels/{channel_id}/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "channel_outbound_failed"

    def test_driver_failure_writes_audit(self, storage_root: Path) -> None:
        class FailingDriver2(ChannelDriver):
            kind = "test.failing2"

            async def start(self, r: StartRequest) -> None:
                pass

            async def send(self, r: SendRequest) -> SendResult:
                raise RuntimeError("timeout")

            async def stop(self, r: StopRequest) -> None:
                pass

            def capabilities(self) -> ChannelCapabilities:
                return ChannelCapabilities()

        rt = ChannelRuntime()
        rt.register(FailingDriver2())
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root, channel_runtime=rt)
        client = TestClient(app, raise_server_exceptions=False)
        channel_id = _create_channel(client, kind="test.failing2")
        client.post(
            f"/v1/channels/{channel_id}/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.outbound.failed" for r in records)


# ---------------------------------------------------------------------------
# Pairing token redeem — errors
# ---------------------------------------------------------------------------


class TestPairingTokenRedeemErrors:
    def test_unknown_token_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/pairing_tokens/pair_nonexistent/redeem",
            json={"sender_id": "ext-1"},
        )
        assert resp.status_code == 404

    def test_unknown_token_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/pairing_tokens/pair_nonexistent/redeem",
            json={"sender_id": "ext-1"},
        ).json()
        assert body["error"]["code"] == "pairing_token_not_found"

    def test_unknown_token_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/pairing_tokens/pair_nonexistent/redeem",
            json={"sender_id": "ext-1"},
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.pair.redeem.failed" for r in records)

    def test_unknown_token_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post(
            "/v1/pairing_tokens/pair_nonexistent/redeem",
            json={"sender_id": "ext-1"},
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "channel.pair.redeem.failed"
        )
        assert record["level"] == "error"

    def test_already_redeemed_returns_409(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"})
        resp = client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-2"})
        assert resp.status_code == 409

    def test_already_redeemed_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"})
        body = client.post(
            f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-2"}
        ).json()
        assert body["error"]["code"] == "pairing_token_already_redeemed"

    def test_already_redeemed_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"})
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-2"})
        records = _audit_records(storage_root)
        redeem_failures = [r for r in records if r.get("event") == "channel.pair.redeem.failed"]
        assert len(redeem_failures) >= 1


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        return _make_client(storage_root)

    def test_inbound_emits_channel_inbound_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "channel.inbound" in span_names

    def test_inbound_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.inbound")
        assert span is not None
        assert span.attributes["channel.id"] == channel_id

    def test_inbound_span_has_invocation_event(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.inbound")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_inbound_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(
            "/v1/channels/ch_nonexistent/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.inbound")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_outbound_emits_channel_outbound_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        _outbound(client, channel_id)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "channel.outbound" in span_names

    def test_outbound_span_has_channel_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        _outbound(client, channel_id)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.outbound")
        assert span is not None
        assert span.attributes["channel.id"] == channel_id

    def test_outbound_span_has_invocation_event(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        _outbound(client, channel_id, session_id="sess_x")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.outbound")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_outbound_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        _outbound(client, channel_id, session_id="sess_abc")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.outbound")
        assert span is not None
        assert span.attributes["session.id"] == "sess_abc"

    def test_redeem_emits_channel_pair_redeem_span(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _otel_exporter.clear()
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"})
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "channel.pair.redeem" in span_names

    def test_redeem_span_has_invocation_event(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _otel_exporter.clear()
        client.post(f"/v1/pairing_tokens/{token}/redeem", json={"sender_id": "ext-1"})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.pair.redeem")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_redeem_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(
            "/v1/pairing_tokens/pair_nonexistent/redeem",
            json={"sender_id": "ext-1"},
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("channel.pair.redeem")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestRouteWiring:
    def test_inbound_route_present_with_runtime(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_runtime())
        channel_id = _create_channel(client)
        resp = client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        assert resp.status_code != 404

    def test_inbound_route_absent_without_runtime(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/channels/ch_any/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        assert resp.status_code == 404

    def test_outbound_route_present_with_runtime(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_runtime())
        channel_id = _create_channel(client)
        resp = client.post(
            f"/v1/channels/{channel_id}/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        assert resp.status_code != 404

    def test_outbound_route_absent_without_runtime(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/channels/ch_any/outbound",
            json={"session_id": "s", "recipient": "r", "content": "c"},
        )
        assert resp.status_code == 404

    def test_redeem_route_present_with_runtime(self, storage_root: Path) -> None:
        # Route exists: token not found produces a structured error, not a routing 404.
        client = _make_client(storage_root, _make_runtime())
        resp = client.post(
            "/v1/pairing_tokens/pair_any/redeem",
            json={"sender_id": "ext-1"},
        )
        assert resp.json().get("error", {}).get("code") == "pairing_token_not_found"

    def test_redeem_route_absent_without_runtime(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/pairing_tokens/pair_any/redeem",
            json={"sender_id": "ext-1"},
        )
        # FastAPI routing 404 has no structured error envelope.
        assert resp.status_code == 404
        assert "error" not in resp.json()
