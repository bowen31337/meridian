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


def _session_outbound(
    client: TestClient,
    session_id: str,
    *,
    content: str = "reply",
    content_type: str = "text/plain",
) -> dict:
    resp = client.post(
        f"/v1/sessions/{session_id}/outbound",
        json={"content": content, "content_type": content_type},
    )
    return resp.json() | {"_status": resp.status_code}


def _attach_session(
    storage_root: Path,
    channel_id: str,
    session_id: str,
    *,
    sender_id: str = "ext-user-1",
    user_profile_id: str = "user_1",
    agent_id: str = "agent_1",
) -> None:
    s_dir = storage_root / "channel_sessions" / channel_id
    s_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": session_id,
        "channel_id": channel_id,
        "user_profile_id": user_profile_id,
        "agent_id": agent_id,
        "sender_id": sender_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (s_dir / f"{session_id}.json").write_text(json.dumps(record))


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
    def test_open_policy_creates_user_profile_for_sender(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client, default_user_profile_id="user_default_42")
        result = _inbound(client, channel_id, "unknown-sender")
        assert result["_status"] == 200
        # Each new sender gets an auto-created UserProfile, not the channel default.
        assert result["user_profile_id"].startswith("up_")

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
        channel_id = _create_channel(client, default_agent_id="agent_def")
        result = _inbound(client, channel_id, "sender-x")
        session_id = result["session_id"]
        session = json.loads(
            (storage_root / "channel_sessions" / channel_id / f"{session_id}.json").read_text()
        )
        assert session["channel_id"] == channel_id
        assert session["user_profile_id"] == result["user_profile_id"]
        assert session["agent_id"] == "agent_def"
        assert session["sender_id"] == "sender-x"

    def test_open_policy_returns_not_quarantined(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _inbound(client, channel_id, "any-sender")
        assert result["quarantined"] is False

    def test_open_policy_creates_pairing_for_sender(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _inbound(client, channel_id, "new-sender")
        pairing_file = storage_root / "channel_pairings" / channel_id / "new-sender.json"
        assert pairing_file.exists()
        pairing = json.loads(pairing_file.read_text())
        assert pairing["auto_created"] is True
        assert pairing["user_profile_id"].startswith("up_")

    def test_open_policy_second_inbound_reuses_profile(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        first = _inbound(client, channel_id, "returning-sender")
        second = _inbound(client, channel_id, "returning-sender")
        assert first["user_profile_id"] == second["user_profile_id"]

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

    def test_quarantine_creates_session_for_quarantine_profile(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        result = _inbound(client, channel_id, "untrusted-sender")
        session_id = result["session_id"]
        sessions_dir = storage_root / "channel_sessions" / channel_id
        assert (sessions_dir / f"{session_id}.json").exists()

    def test_quarantine_returns_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        result = _inbound(client, channel_id, "untrusted-sender")
        assert "user_profile_id" in result
        assert result["user_profile_id"].startswith("qup_")

    def test_quarantine_returns_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        result = _inbound(client, channel_id, "untrusted-sender")
        assert "session_id" in result
        assert result["session_id"].startswith("sess_")

    def test_quarantine_profile_has_minimal_caps(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        _inbound(client, channel_id, "untrusted-sender")
        qp_file = storage_root / "channel_quarantine" / channel_id / "_profile.json"
        assert qp_file.exists()
        qp = json.loads(qp_file.read_text())
        caps = json.loads(qp["metadata"])["capabilities"]
        assert caps == ["minimal"]

    def test_quarantine_reuses_quarantine_profile(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = self._quarantine_channel(client)
        first = _inbound(client, channel_id, "sender-a")
        second = _inbound(client, channel_id, "sender-b")
        assert first["user_profile_id"] == second["user_profile_id"]

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
# Inbound latency target (PRD §6.1)
# ---------------------------------------------------------------------------


class TestInboundLatencyTarget:
    """channel.inbound span records latency_ms on every code path (PRD §6.1 < 1 s p95)."""

    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _client(self, storage_root: Path) -> TestClient:
        return _make_client(storage_root)

    def test_success_span_has_latency_ms(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        span = {s.name: s for s in _otel_exporter.get_finished_spans()}.get("channel.inbound")
        assert span is not None
        assert "channel.inbound.latency_ms" in span.attributes

    def test_latency_ms_is_non_negative(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client)
        _otel_exporter.clear()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        span = {s.name: s for s in _otel_exporter.get_finished_spans()}.get("channel.inbound")
        assert span is not None
        assert span.attributes["channel.inbound.latency_ms"] >= 0

    def test_channel_not_found_span_has_latency_ms(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        _otel_exporter.clear()
        client.post(
            "/v1/channels/ch_nonexistent/inbound",
            json={"sender_id": "s", "content": "hi"},
        )
        span = {s.name: s for s in _otel_exporter.get_finished_spans()}.get("channel.inbound")
        assert span is not None
        assert "channel.inbound.latency_ms" in span.attributes

    def test_policy_rejected_span_has_latency_ms(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client, inbound_policy="paired_only")
        _otel_exporter.clear()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "stranger", "content": "hi"},
        )
        span = {s.name: s for s in _otel_exporter.get_finished_spans()}.get("channel.inbound")
        assert span is not None
        assert "channel.inbound.latency_ms" in span.attributes

    def test_quarantine_span_has_latency_ms(self, storage_root: Path) -> None:
        client = self._client(storage_root)
        channel_id = _create_channel(client, inbound_policy="quarantine")
        _otel_exporter.clear()
        client.post(
            f"/v1/channels/{channel_id}/inbound",
            json={"sender_id": "untrusted", "content": "hi"},
        )
        span = {s.name: s for s in _otel_exporter.get_finished_spans()}.get("channel.inbound")
        assert span is not None
        assert "channel.inbound.latency_ms" in span.attributes


# ---------------------------------------------------------------------------
# Session outbound fan-out
# ---------------------------------------------------------------------------


class TestSessionOutbound:
    def test_delivers_to_single_channel_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_abc")
        result = _session_outbound(client, "sess_abc")
        assert result["_status"] == 200

    def test_response_has_session_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_abc")
        result = _session_outbound(client, "sess_abc")
        assert result["session_id"] == "sess_abc"

    def test_response_has_results_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_abc")
        result = _session_outbound(client, "sess_abc")
        assert isinstance(result["results"], list)

    def test_result_contains_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_abc")
        result = _session_outbound(client, "sess_abc")
        assert result["results"][0]["channel_id"] == channel_id

    def test_result_contains_delivered_true(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_abc")
        result = _session_outbound(client, "sess_abc")
        assert result["results"][0]["delivered"] is True

    def test_result_contains_message_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_abc")
        result = _session_outbound(client, "sess_abc")
        assert "message_id" in result["results"][0]

    def test_delivers_to_multiple_channels(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        _attach_session(storage_root, ch_a, "sess_multi", sender_id="user-a")
        _attach_session(storage_root, ch_b, "sess_multi", sender_id="user-b")
        result = _session_outbound(client, "sess_multi")
        assert result["_status"] == 200
        assert len(result["results"]) == 2

    def test_driver_called_once_per_channel(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        _attach_session(storage_root, ch_a, "sess_multi2", sender_id="user-a")
        _attach_session(storage_root, ch_b, "sess_multi2", sender_id="user-b")
        _session_outbound(client, "sess_multi2")
        assert len(driver.sends) == 2

    def test_driver_called_with_sender_id_as_recipient(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_recip", sender_id="the-sender")
        _session_outbound(client, "sess_recip")
        assert driver.sends[0].recipient == "the-sender"

    def test_skips_disabled_channel(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        disabled_ch = _create_channel(client, egress_policy="disabled")
        enabled_ch = _create_channel(client)
        _attach_session(storage_root, disabled_ch, "sess_skip", sender_id="user-a")
        _attach_session(storage_root, enabled_ch, "sess_skip", sender_id="user-b")
        result = _session_outbound(client, "sess_skip")
        assert result["_status"] == 200
        assert len(result["results"]) == 1
        assert result["results"][0]["channel_id"] == enabled_ch

    def test_disabled_channel_increments_channels_skipped(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        disabled_ch = _create_channel(client, egress_policy="disabled")
        _attach_session(storage_root, disabled_ch, "sess_skip2")
        result = _session_outbound(client, "sess_skip2")
        assert result["channels_skipped"] == 1

    def test_unknown_session_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post(
            "/v1/sessions/sess_nonexistent/outbound",
            json={"content": "hello"},
        )
        assert resp.status_code == 404

    def test_unknown_session_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/sessions/sess_nonexistent/outbound",
            json={"content": "hello"},
        ).json()
        assert body["error"]["code"] == "session_outbound_not_found"

    def test_unknown_session_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/sessions/sess_nonexistent/outbound", json={"content": "hello"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.outbound.failed" for r in records)

    def test_unknown_session_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/sessions/sess_nonexistent/outbound", json={"content": "hello"})
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "session.outbound.failed"
        )
        assert record["level"] == "error"

    def test_partial_failure_delivers_to_remaining_channels(self, storage_root: Path) -> None:
        class PartiallyFailingDriver(ChannelDriver):
            kind = "test.partial"

            def __init__(self) -> None:
                self._call_count = 0

            async def start(self, r: StartRequest) -> None:
                pass

            async def send(self, r: SendRequest) -> SendResult:
                self._call_count += 1
                if self._call_count == 1:
                    raise RuntimeError("first channel down")
                return SendResult(
                    message_id="msg-ok",
                    timestamp="2026-01-01T00:00:00+00:00",
                    delivered=True,
                )

            async def stop(self, r: StopRequest) -> None:
                pass

            def capabilities(self) -> ChannelCapabilities:
                return ChannelCapabilities(can_send_text=True)

        rt = ChannelRuntime()
        rt.register(PartiallyFailingDriver())
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root, channel_runtime=rt)
        client = TestClient(app, raise_server_exceptions=False)
        ch_a = _create_channel(client, kind="test.partial")
        ch_b = _create_channel(client, kind="test.partial")
        _attach_session(storage_root, ch_a, "sess_partial", sender_id="user-a")
        _attach_session(storage_root, ch_b, "sess_partial", sender_id="user-b")
        resp = client.post("/v1/sessions/sess_partial/outbound", json={"content": "hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert any(r.get("delivered") for r in data["results"])

    def test_partial_failure_writes_audit(self, storage_root: Path) -> None:
        class AlwaysFailingDriver(ChannelDriver):
            kind = "test.fail3"

            async def start(self, r: StartRequest) -> None:
                pass

            async def send(self, r: SendRequest) -> SendResult:
                raise RuntimeError("always fails")

            async def stop(self, r: StopRequest) -> None:
                pass

            def capabilities(self) -> ChannelCapabilities:
                return ChannelCapabilities()

        rt = ChannelRuntime()
        rt.register(AlwaysFailingDriver())
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root, channel_runtime=rt)
        client = TestClient(app, raise_server_exceptions=False)
        channel_id = _create_channel(client, kind="test.fail3")
        _attach_session(storage_root, channel_id, "sess_fail")
        client.post("/v1/sessions/sess_fail/outbound", json={"content": "hi"})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "session.outbound.failed" for r in records)

    def test_emits_session_outbound_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_otel1")
        _otel_exporter.clear()
        _session_outbound(client, "sess_otel1")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.outbound" in span_names

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_otel2")
        _otel_exporter.clear()
        _session_outbound(client, "sess_otel2")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.outbound")
        assert span is not None
        assert span.attributes["session.id"] == "sess_otel2"

    def test_span_has_invocation_event(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _attach_session(storage_root, channel_id, "sess_otel3")
        _otel_exporter.clear()
        _session_outbound(client, "sess_otel3")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.outbound")
        assert span is not None
        assert "meridian.error.invocation" in [e.name for e in span.events]

    def test_span_has_channel_count_attribute(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        _attach_session(storage_root, ch_a, "sess_otel4", sender_id="user-a")
        _attach_session(storage_root, ch_b, "sess_otel4", sender_id="user-b")
        _otel_exporter.clear()
        _session_outbound(client, "sess_otel4")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.outbound")
        assert span is not None
        assert span.attributes["session.channel_count"] == 2


# ---------------------------------------------------------------------------
# Pairing rule lookup: GET /v1/channels/{channel_id}/remote/{remote_id}
# ---------------------------------------------------------------------------


def _resolve(client: TestClient, channel_id: str, remote_id: str) -> dict:
    resp = client.get(f"/v1/channels/{channel_id}/remote/{remote_id}")
    return resp.json() | {"_status": resp.status_code}


class TestPairingRuleLookup:
    """GET /v1/channels/{channel_id}/remote/{remote_id} deterministically resolves to UserProfile."""

    def test_paired_remote_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        result = _resolve(client, channel_id, "ext-user-1")
        assert result["_status"] == 200

    def test_response_has_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        result = _resolve(client, channel_id, "ext-user-1")
        assert result["user_profile_id"] == "user_abc"

    def test_response_has_channel_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        result = _resolve(client, channel_id, "ext-user-1")
        assert result["channel_id"] == channel_id

    def test_response_has_remote_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        result = _resolve(client, channel_id, "ext-user-1")
        assert result["remote_id"] == "ext-user-1"

    def test_unknown_channel_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result = _resolve(client, "ch_nonexistent", "ext-user-1")
        assert result["_status"] == 404

    def test_unknown_channel_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        result = _resolve(client, "ch_nonexistent", "ext-user-1")
        assert result["error"]["code"] == "channel_remote_not_found"

    def test_unknown_channel_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _resolve(client, "ch_nonexistent", "ext-user-1")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.pairing.resolve.failed" for r in records)

    def test_unpaired_remote_returns_404(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _resolve(client, channel_id, "no-such-remote")
        assert result["_status"] == 404

    def test_unpaired_remote_error_code(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        result = _resolve(client, channel_id, "no-such-remote")
        assert result["error"]["code"] == "channel_pairing_not_found"

    def test_unpaired_remote_writes_audit(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        _resolve(client, channel_id, "no-such-remote")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "channel.pairing.resolve.failed" for r in records)

    def test_auto_created_pairing_resolves(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        inbound_result = _inbound(client, channel_id, "auto-sender")
        result = _resolve(client, channel_id, "auto-sender")
        assert result["_status"] == 200
        assert result["user_profile_id"] == inbound_result["user_profile_id"]

    def test_different_remote_ids_resolve_independently(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token_a = _issue_pairing_token(client, channel_id, user_profile_id="user_a")
        token_b = _issue_pairing_token(client, channel_id, user_profile_id="user_b")
        _redeem_token(client, token_a, "remote-a")
        _redeem_token(client, token_b, "remote-b")
        assert _resolve(client, channel_id, "remote-a")["user_profile_id"] == "user_a"
        assert _resolve(client, channel_id, "remote-b")["user_profile_id"] == "user_b"

    def test_otel_span_emitted(self, storage_root: Path) -> None:
        from tests._otel_shared import otel_exporter

        otel_exporter.clear()
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        otel_exporter.clear()
        _resolve(client, channel_id, "ext-user-1")
        span_names = [s.name for s in otel_exporter.get_finished_spans()]
        assert "channel.pairing.resolve" in span_names

    def test_otel_span_has_channel_id(self, storage_root: Path) -> None:
        from tests._otel_shared import otel_exporter

        otel_exporter.clear()
        client = _make_client(storage_root)
        channel_id = _create_channel(client)
        token = _issue_pairing_token(client, channel_id, user_profile_id="user_abc")
        _redeem_token(client, token, "ext-user-1")
        otel_exporter.clear()
        _resolve(client, channel_id, "ext-user-1")
        spans = {s.name: s for s in otel_exporter.get_finished_spans()}
        span = spans.get("channel.pairing.resolve")
        assert span is not None
        assert span.attributes["channel.id"] == channel_id

    def test_otel_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode
        from tests._otel_shared import otel_exporter

        otel_exporter.clear()
        client = _make_client(storage_root)
        _resolve(client, "ch_nonexistent", "ext-user-1")
        spans = {s.name: s for s in otel_exporter.get_finished_spans()}
        span = spans.get("channel.pairing.resolve")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Cross-channel session reachability (PRD F-CH-3)
# ---------------------------------------------------------------------------


class TestCrossChannelSession:
    """
    Same Session is reachable from every channel the UserProfile is paired with.
    When an inbound message creates a session, that session is registered under
    all other channels where the same UserProfile has a pairing.
    """

    def test_inbound_writes_session_for_other_paired_channel(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        tok_a = _issue_pairing_token(client, ch_a, user_profile_id="user_shared")
        tok_b = _issue_pairing_token(client, ch_b, user_profile_id="user_shared")
        _redeem_token(client, tok_a, "remote-a")
        _redeem_token(client, tok_b, "remote-b")
        result = _inbound(client, ch_a, "remote-a")
        session_id = result["session_id"]
        assert (storage_root / "channel_sessions" / ch_b / f"{session_id}.json").exists()

    def test_session_on_other_channel_has_correct_sender_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        tok_a = _issue_pairing_token(client, ch_a, user_profile_id="user_shared")
        tok_b = _issue_pairing_token(client, ch_b, user_profile_id="user_shared")
        _redeem_token(client, tok_a, "remote-a")
        _redeem_token(client, tok_b, "remote-b")
        result = _inbound(client, ch_a, "remote-a")
        session_id = result["session_id"]
        sess_b = json.loads(
            (storage_root / "channel_sessions" / ch_b / f"{session_id}.json").read_text()
        )
        assert sess_b["sender_id"] == "remote-b"

    def test_session_on_other_channel_has_correct_user_profile_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        tok_a = _issue_pairing_token(client, ch_a, user_profile_id="user_shared")
        tok_b = _issue_pairing_token(client, ch_b, user_profile_id="user_shared")
        _redeem_token(client, tok_a, "remote-a")
        _redeem_token(client, tok_b, "remote-b")
        result = _inbound(client, ch_a, "remote-a")
        session_id = result["session_id"]
        sess_b = json.loads(
            (storage_root / "channel_sessions" / ch_b / f"{session_id}.json").read_text()
        )
        assert sess_b["user_profile_id"] == "user_shared"

    def test_session_outbound_delivers_to_all_paired_channels(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        tok_a = _issue_pairing_token(client, ch_a, user_profile_id="user_shared")
        tok_b = _issue_pairing_token(client, ch_b, user_profile_id="user_shared")
        _redeem_token(client, tok_a, "remote-a")
        _redeem_token(client, tok_b, "remote-b")
        inbound_result = _inbound(client, ch_a, "remote-a")
        session_id = inbound_result["session_id"]
        result = _session_outbound(client, session_id)
        assert result["_status"] == 200
        assert len(result["results"]) == 2

    def test_session_outbound_uses_correct_recipient_per_channel(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        tok_a = _issue_pairing_token(client, ch_a, user_profile_id="user_shared")
        tok_b = _issue_pairing_token(client, ch_b, user_profile_id="user_shared")
        _redeem_token(client, tok_a, "remote-a")
        _redeem_token(client, tok_b, "remote-b")
        inbound_result = _inbound(client, ch_a, "remote-a")
        session_id = inbound_result["session_id"]
        _session_outbound(client, session_id)
        recipients = {s.channel_id: s.recipient for s in driver.sends}
        assert recipients[ch_a] == "remote-a"
        assert recipients[ch_b] == "remote-b"

    def test_unpaired_channel_does_not_get_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        ch_a = _create_channel(client)
        ch_b = _create_channel(client)
        tok_a = _issue_pairing_token(client, ch_a, user_profile_id="user_only_a")
        _redeem_token(client, tok_a, "remote-a")
        result = _inbound(client, ch_a, "remote-a")
        session_id = result["session_id"]
        assert not (storage_root / "channel_sessions" / ch_b / f"{session_id}.json").exists()

    def test_three_channels_all_get_session(self, storage_root: Path) -> None:
        driver = StubDriver()
        client = _make_client(storage_root, _make_runtime(driver))
        channels = [_create_channel(client) for _ in range(3)]
        for i, ch in enumerate(channels):
            tok = _issue_pairing_token(client, ch, user_profile_id="user_triple")
            _redeem_token(client, tok, f"remote-{i}")
        result = _inbound(client, channels[0], "remote-0")
        session_id = result["session_id"]
        for i, ch in enumerate(channels[1:], 1):
            assert (storage_root / "channel_sessions" / ch / f"{session_id}.json").exists()
        outbound = _session_outbound(client, session_id)
        assert len(outbound["results"]) == 3


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

    def test_session_outbound_route_present_with_runtime(self, storage_root: Path) -> None:
        # Route exists: session not found gives a structured error, not a routing 404.
        client = _make_client(storage_root, _make_runtime())
        resp = client.post(
            "/v1/sessions/sess_any/outbound",
            json={"content": "hi"},
        )
        assert resp.json().get("error", {}).get("code") == "session_outbound_not_found"

    def test_session_outbound_route_absent_without_runtime(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/sessions/sess_any/outbound",
            json={"content": "hi"},
        )
        assert resp.status_code == 404
        assert "error" not in resp.json()

    def test_resolve_pairing_route_present_with_runtime(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_runtime())
        channel_id = _create_channel(client)
        resp = client.get(f"/v1/channels/{channel_id}/remote/no-such-remote")
        assert resp.json().get("error", {}).get("code") == "channel_pairing_not_found"

    def test_resolve_pairing_route_absent_without_runtime(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/channels/ch_any/remote/remote_any")
        assert resp.status_code == 404
        assert "error" not in resp.json()
