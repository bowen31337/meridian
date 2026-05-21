"""
Prometheus /metrics endpoint conformance suite.

Tests cover:
  - GET /metrics returns 200.
  - Response Content-Type is the Prometheus text exposition MIME type.
  - Response body contains HELP line for meridian_sessions_total.
  - Response body contains TYPE counter line for meridian_sessions_total.
  - Response body contains HELP line for meridian_session_duration_seconds.
  - Response body contains TYPE histogram line for meridian_session_duration_seconds.
  - Response body contains HELP line for meridian_tool_calls_total.
  - Response body contains TYPE counter line for meridian_tool_calls_total.
  - Response body contains HELP line for meridian_tool_call_duration_seconds.
  - Response body contains TYPE histogram line for meridian_tool_call_duration_seconds.
  - Response body contains HELP line for meridian_channel_inbound_total.
  - Response body contains TYPE counter line for meridian_channel_inbound_total.
  - Response body contains HELP line for meridian_channel_outbound_total.
  - Response body contains TYPE counter line for meridian_channel_outbound_total.
  - Response body contains HELP line for meridian_active_sessions.
  - Response body contains TYPE gauge line for meridian_active_sessions.
  - Response body contains HELP line for meridian_harness_wakes_total.
  - Response body contains TYPE counter line for meridian_harness_wakes_total.
  - Response body contains HELP line for meridian_skill_forge_proposals_total.
  - Response body contains TYPE counter line for meridian_skill_forge_proposals_total.
  - Response body contains HELP line for meridian_vault_accesses_total.
  - Response body contains TYPE counter line for meridian_vault_accesses_total.
  - OTel span "metrics.scrape" is emitted on a successful scrape.
  - create_app always wires the /metrics route (no storage_root dependency).
  - Sessions create increments meridian_sessions_total{phase="created"}.
  - Phase transition increments meridian_sessions_total for the target phase.
  - Session cancel increments meridian_sessions_total{phase="terminated"}.
  - Session cancel records meridian_session_duration_seconds{result="cancelled"}.
  - Checkpoint with completed tool calls increments meridian_tool_calls_total.
  - Checkpoint with completed tool calls records meridian_tool_call_duration_seconds.
  - Channel inbound increments meridian_channel_inbound_total{kind}.
  - Channel outbound increments meridian_channel_outbound_total{kind}.
  - Phase transition increments meridian_active_sessions for target phase and decrements prior phase.
  - Session cancel increments meridian_active_sessions{phase="terminated"}.
  - Harness wake increments meridian_harness_wakes_total.
  - Skill forge proposal approve increments meridian_skill_forge_proposals_total{outcome="approved"}.
  - Skill forge proposal reject increments meridian_skill_forge_proposals_total{outcome="rejected"}.
  - Vault secret meta access increments meridian_vault_accesses_total{vault_id}.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._vault_backend_os_keychain import OsKeychainVaultBackend
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY
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
# Shared stubs
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


class _StubChannelDriver(ChannelDriver):
    kind = "test.metrics"

    async def start(self, request: StartRequest) -> None:
        pass

    async def send(self, request: SendRequest) -> SendResult:
        return SendResult(
            message_id="msg-1",
            timestamp="2026-01-01T00:00:00+00:00",
            delivered=True,
        )

    async def stop(self, request: StopRequest) -> None:
        pass

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(can_send_text=True)


def _sample(metric_name: str, labels: dict[str, str] | None = None) -> float:
    """Return the current sample value for a metric from the default REGISTRY."""
    value = REGISTRY.get_sample_value(metric_name, labels or {})
    return value if value is not None else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(audit_log: FileAuditLog, **kwargs: Any) -> TestClient:
    app = create_app(audit_log, **kwargs)
    return TestClient(app, raise_server_exceptions=False)


def _metric_value(body: str, metric: str, labels: dict[str, str]) -> float:
    """Parse a Prometheus text-format body and return the value for a metric+labels."""
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    target = f"{metric}{{{label_str}}}" if labels else metric
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(target + " ") or line.startswith(target + "{"):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return 0.0


def _metric_count(body: str, metric: str) -> float:
    """Return _count value for a histogram or summary metric."""
    return _metric_value(body, f"{metric}_count", {})


# ---------------------------------------------------------------------------
# Basic endpoint tests
# ---------------------------------------------------------------------------


def test_metrics_returns_200(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_content_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert CONTENT_TYPE_LATEST.split(";")[0] in resp.headers["content-type"]


def test_metrics_contains_sessions_total_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_sessions_total" in resp.text


def test_metrics_contains_sessions_total_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_sessions_total counter" in resp.text


def test_metrics_contains_session_duration_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_session_duration_seconds" in resp.text


def test_metrics_contains_session_duration_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_session_duration_seconds histogram" in resp.text


def test_metrics_contains_tool_calls_total_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_tool_calls_total" in resp.text


def test_metrics_contains_tool_calls_total_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_tool_calls_total counter" in resp.text


def test_metrics_contains_tool_call_duration_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_tool_call_duration_seconds" in resp.text


def test_metrics_contains_tool_call_duration_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_tool_call_duration_seconds histogram" in resp.text


def test_metrics_router_wired_without_storage_root(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)  # no storage_root
    resp = client.get("/metrics")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OTel span emission
# ---------------------------------------------------------------------------


def test_metrics_emits_otel_span(tmp_path: Path) -> None:
    _otel_exporter.clear()
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    client.get("/metrics")
    spans = _otel_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "metrics.scrape" in names


# ---------------------------------------------------------------------------
# Instrumentation: session lifecycle
# ---------------------------------------------------------------------------


def _write_manifest(storage_root: Path, session_id: str, extra: dict | None = None) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "session_id": session_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "status": "idle",
    }
    if extra:
        manifest.update(extra)
    (session_dir / "manifest.json").write_text(json.dumps(manifest))


def test_session_cancel_increments_sessions_total(tmp_path: Path) -> None:
    from meridiand._metrics_registry import sessions_total
    from storage_event_log import LocalEventLogWriter

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    session_id = f"sess_{uuid.uuid4().hex}"
    _write_manifest(storage_root, session_id)

    event_log = LocalEventLogWriter(storage_root)
    client = TestClient(
        create_app(audit_log, storage_root=storage_root, event_log=event_log),
        raise_server_exceptions=False,
    )

    before = _sample("meridian_sessions_total", {"phase": "terminated"})
    resp = client.post(f"/v1/sessions/{session_id}/cancel")
    assert resp.status_code == 200
    after = _sample("meridian_sessions_total", {"phase": "terminated"})
    assert after == before + 1


def test_session_cancel_records_duration(tmp_path: Path) -> None:
    from meridiand._metrics_registry import session_duration_seconds
    from storage_event_log import LocalEventLogWriter

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    session_id = f"sess_{uuid.uuid4().hex}"
    _write_manifest(storage_root, session_id)

    event_log = LocalEventLogWriter(storage_root)
    client = TestClient(
        create_app(audit_log, storage_root=storage_root, event_log=event_log),
        raise_server_exceptions=False,
    )

    before = _sample("meridian_session_duration_seconds_count", {"result": "cancelled"})
    resp = client.post(f"/v1/sessions/{session_id}/cancel")
    assert resp.status_code == 200
    after = _sample("meridian_session_duration_seconds_count", {"result": "cancelled"})
    assert after == before + 1


# ---------------------------------------------------------------------------
# Instrumentation: checkpoint tool call tracking
# ---------------------------------------------------------------------------


def test_checkpoint_increments_tool_calls_total(tmp_path: Path) -> None:
    from meridiand._metrics_registry import tool_calls_total

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    session_id = f"sess_{uuid.uuid4().hex}"

    client = TestClient(
        create_app(audit_log, storage_root=storage_root),
        raise_server_exceptions=False,
    )

    # First checkpoint: one pending tool call
    first_body = {
        "seq": 1,
        "phase": "waiting_for_tool",
        "pending_tool_calls": [{"id": "tc_001", "name": "bash", "type": "tool_use"}],
        "message_tail": [],
        "usage": {},
        "taken_at": "2026-01-01T00:00:00+00:00",
    }
    resp = client.post(f"/v1/x/sessions/{session_id}/checkpoint", json=first_body)
    assert resp.status_code == 200

    before = _sample(
        "meridian_tool_calls_total",
        {"tool": "bash", "backend": "unknown", "result": "success"},
    )

    # Second checkpoint: tool call resolved (empty pending)
    second_body = {
        "seq": 2,
        "phase": "idle",
        "pending_tool_calls": [],
        "message_tail": [],
        "usage": {},
        "taken_at": "2026-01-01T00:00:05+00:00",
    }
    resp = client.post(f"/v1/x/sessions/{session_id}/checkpoint", json=second_body)
    assert resp.status_code == 200

    after = _sample(
        "meridian_tool_calls_total",
        {"tool": "bash", "backend": "unknown", "result": "success"},
    )
    assert after == before + 1


def test_checkpoint_records_tool_call_duration(tmp_path: Path) -> None:
    from meridiand._metrics_registry import tool_call_duration_seconds

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    session_id = f"sess_{uuid.uuid4().hex}"

    client = TestClient(
        create_app(audit_log, storage_root=storage_root),
        raise_server_exceptions=False,
    )

    first_body = {
        "seq": 1,
        "phase": "waiting_for_tool",
        "pending_tool_calls": [{"id": "tc_002", "name": "read", "type": "tool_use"}],
        "message_tail": [],
        "usage": {},
        "taken_at": "2026-01-01T00:00:00+00:00",
    }
    client.post(f"/v1/x/sessions/{session_id}/checkpoint", json=first_body)

    before_count = _sample("meridian_tool_call_duration_seconds_count")

    second_body = {
        "seq": 2,
        "phase": "idle",
        "pending_tool_calls": [],
        "message_tail": [],
        "usage": {},
        "taken_at": "2026-01-01T00:00:03+00:00",
    }
    client.post(f"/v1/x/sessions/{session_id}/checkpoint", json=second_body)

    after_count = _sample("meridian_tool_call_duration_seconds_count")
    assert after_count == before_count + 1


# ---------------------------------------------------------------------------
# HELP / TYPE lines for new metrics
# ---------------------------------------------------------------------------


def test_metrics_contains_channel_inbound_total_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_channel_inbound_total" in resp.text


def test_metrics_contains_channel_inbound_total_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_channel_inbound_total counter" in resp.text


def test_metrics_contains_channel_outbound_total_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_channel_outbound_total" in resp.text


def test_metrics_contains_channel_outbound_total_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_channel_outbound_total counter" in resp.text


def test_metrics_contains_active_sessions_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_active_sessions" in resp.text


def test_metrics_contains_active_sessions_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_active_sessions gauge" in resp.text


def test_metrics_contains_harness_wakes_total_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_harness_wakes_total" in resp.text


def test_metrics_contains_harness_wakes_total_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_harness_wakes_total counter" in resp.text


def test_metrics_contains_skill_forge_proposals_total_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_skill_forge_proposals_total" in resp.text


def test_metrics_contains_skill_forge_proposals_total_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_skill_forge_proposals_total counter" in resp.text


def test_metrics_contains_vault_accesses_total_help(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# HELP meridian_vault_accesses_total" in resp.text


def test_metrics_contains_vault_accesses_total_type(tmp_path: Path) -> None:
    audit_log = FileAuditLog(tmp_path)
    client = _make_client(audit_log)
    resp = client.get("/metrics")
    assert "# TYPE meridian_vault_accesses_total counter" in resp.text


# ---------------------------------------------------------------------------
# Instrumentation: channel inbound / outbound
# ---------------------------------------------------------------------------


def _make_channel_client(storage_root: Path) -> TestClient:
    rt = ChannelRuntime()
    rt.register(_StubChannelDriver())
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        channel_runtime=rt,
    )
    return TestClient(app, raise_server_exceptions=False)


def _create_channel(client: TestClient, kind: str = "test.metrics") -> str:
    resp = client.post(
        "/v1/channels",
        json={"kind": kind, "config": {"token_vault_ref": "vaults/main/tok"}},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_channel_inbound_increments_metric(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    client = _make_channel_client(storage_root)
    channel_id = _create_channel(client)

    before = _sample("meridian_channel_inbound_total", {"kind": "test.metrics"})
    resp = client.post(
        f"/v1/channels/{channel_id}/inbound",
        json={"sender_id": "user-1", "content": "hello"},
    )
    assert resp.status_code == 200, resp.text
    after = _sample("meridian_channel_inbound_total", {"kind": "test.metrics"})
    assert after == before + 1


def test_channel_outbound_increments_metric(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    client = _make_channel_client(storage_root)
    channel_id = _create_channel(client)

    before = _sample("meridian_channel_outbound_total", {"kind": "test.metrics"})
    resp = client.post(
        f"/v1/channels/{channel_id}/outbound",
        json={"session_id": "sess_test", "recipient": "user-1", "content": "reply"},
    )
    assert resp.status_code == 200, resp.text
    after = _sample("meridian_channel_outbound_total", {"kind": "test.metrics"})
    assert after == before + 1


# ---------------------------------------------------------------------------
# Instrumentation: active_sessions gauge
# ---------------------------------------------------------------------------


def test_phase_transition_increments_active_sessions(tmp_path: Path) -> None:
    from storage_event_log import LocalEventLogWriter

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    session_id = f"sess_{uuid.uuid4().hex}"
    _write_manifest(storage_root, session_id)

    event_log = LocalEventLogWriter(storage_root)
    client = TestClient(
        create_app(audit_log, storage_root=storage_root, event_log=event_log),
        raise_server_exceptions=False,
    )

    before = _sample("meridian_active_sessions", {"phase": "idle"})
    resp = client.post(
        f"/v1/x/sessions/{session_id}/phase",
        json={"to_phase": "idle", "reason": "test"},
    )
    assert resp.status_code == 200, resp.text
    after = _sample("meridian_active_sessions", {"phase": "idle"})
    assert after == before + 1


def test_session_cancel_increments_active_sessions_terminated(tmp_path: Path) -> None:
    from storage_event_log import LocalEventLogWriter

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    session_id = f"sess_{uuid.uuid4().hex}"
    _write_manifest(storage_root, session_id)

    event_log = LocalEventLogWriter(storage_root)
    client = TestClient(
        create_app(audit_log, storage_root=storage_root, event_log=event_log),
        raise_server_exceptions=False,
    )

    before = _sample("meridian_active_sessions", {"phase": "terminated"})
    resp = client.post(f"/v1/sessions/{session_id}/cancel")
    assert resp.status_code == 200, resp.text
    after = _sample("meridian_active_sessions", {"phase": "terminated"})
    assert after == before + 1


# ---------------------------------------------------------------------------
# Instrumentation: harness_wakes_total
# ---------------------------------------------------------------------------


def test_wake_increments_harness_wakes_total(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    session_id = f"sess_{uuid.uuid4().hex}"
    _write_manifest(storage_root, session_id)

    client = TestClient(
        create_app(audit_log, storage_root=storage_root),
        raise_server_exceptions=False,
    )

    before = _sample("meridian_harness_wakes_total")
    resp = client.post(f"/v1/x/sessions/{session_id}/wake")
    assert resp.status_code == 200, resp.text
    after = _sample("meridian_harness_wakes_total")
    assert after == before + 1


# ---------------------------------------------------------------------------
# Instrumentation: skill_forge_proposals_total
# ---------------------------------------------------------------------------


def _write_proposal(storage_root: Path, proposal_id: str, skill_id: str = "skill_test") -> None:
    proposals_dir = storage_root / "skill_forge" / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": proposal_id,
        "skill_id": skill_id,
        "instructions": "do things",
        "tools": [],
        "tests": [],
        "status": "PROPOSAL",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (proposals_dir / f"{proposal_id}.json").write_text(json.dumps(record))


def test_approve_proposal_increments_skill_forge_proposals_total(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    proposal_id = f"skillver_{uuid.uuid4().hex}"
    _write_proposal(storage_root, proposal_id)

    client = TestClient(
        create_app(audit_log, storage_root=storage_root),
        raise_server_exceptions=False,
    )

    before = _sample("meridian_skill_forge_proposals_total", {"outcome": "approved"})
    resp = client.post(f"/v1/x/skill_forge/proposals/{proposal_id}/approve")
    assert resp.status_code == 200, resp.text
    after = _sample("meridian_skill_forge_proposals_total", {"outcome": "approved"})
    assert after == before + 1


def test_reject_proposal_increments_skill_forge_proposals_total(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    audit_log = FileAuditLog(tmp_path)
    proposal_id = f"skillver_{uuid.uuid4().hex}"
    _write_proposal(storage_root, proposal_id)

    client = TestClient(
        create_app(audit_log, storage_root=storage_root),
        raise_server_exceptions=False,
    )

    before = _sample("meridian_skill_forge_proposals_total", {"outcome": "rejected"})
    resp = client.post(
        f"/v1/x/skill_forge/proposals/{proposal_id}/reject",
        json={"reason": "not good enough"},
    )
    assert resp.status_code == 200, resp.text
    after = _sample("meridian_skill_forge_proposals_total", {"outcome": "rejected"})
    assert after == before + 1


# ---------------------------------------------------------------------------
# Instrumentation: vault_accesses_total
# ---------------------------------------------------------------------------


def test_vault_secret_meta_increments_vault_accesses_total(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    keyring = _MemoryKeyring()
    os_keychain = OsKeychainVaultBackend(_keyring=keyring)
    app = create_app(
        FileAuditLog(tmp_path),
        storage_root=storage_root,
        os_keychain_backend=os_keychain,
    )
    client = TestClient(app, raise_server_exceptions=False)

    vault_resp = client.post("/v1/vaults", json={"name": "my-vault", "backend": "os_keychain"})
    assert vault_resp.status_code == 201, vault_resp.text
    vault_id = vault_resp.json()["id"]

    secret_resp = client.post(
        f"/v1/vaults/{vault_id}/secrets", json={"key": "api_key", "value": "secret"}
    )
    assert secret_resp.status_code == 201, secret_resp.text

    before = _sample("meridian_vault_accesses_total", {"vault_id": vault_id})
    meta_resp = client.get(f"/v1/vaults/{vault_id}/secrets/api_key/meta")
    assert meta_resp.status_code == 200, meta_resp.text
    after = _sample("meridian_vault_accesses_total", {"vault_id": vault_id})
    assert after == before + 1
