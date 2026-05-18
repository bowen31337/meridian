"""
Wake endpoint conformance suite.

Tests cover:
  - POST /v1/x/sessions/{id}/wake returns 200 with session_id, status, session,
    agent_version, active_skills, phase, thread_id, messages fields.
  - Returns 404 with code "wake_session_not_found" when session manifest is absent.
  - Session not found writes audit log entry with event "session.wake.failed".
  - Audit log detail includes session_id.
  - session field is populated from the session manifest.
  - agent_version is None when session has no agent_id.
  - agent_version is populated from $storage_root/agents/{agent_id}.json when present.
  - active_skills is [] when no skill activations exist for the agent.
  - active_skills contains only activations with status="active" for the session's agent.
  - Activations with status other than "active" are excluded from active_skills.
  - phase is "created" when no session.phase_change events exist in the event log.
  - phase reflects the 'after' field of the last session.phase_change event.
  - thread_id is None when no threads exist for the session.
  - messages is [] when no threads exist for the session.
  - messages are loaded from the most recent thread's messages.ndjson.
  - Most recent thread is determined by created_at in thread manifest.
  - messages are sorted by sequence field.
  - OTel span "session.wake" is emitted on success.
  - OTel span is set to ERROR status on failure.
  - create_app wires the wake router when storage_root is supplied.
  - create_app omits the wake route when storage_root is None.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path, audit_log: FileAuditLog) -> TestClient:
    app = create_app(audit_log, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _write_session(storage_root: Path, session_id: str, data: dict[str, Any]) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(json.dumps(data))


def _write_agent(storage_root: Path, agent_id: str, data: dict[str, Any]) -> None:
    agents_dir = storage_root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{agent_id}.json").write_text(json.dumps(data))


def _write_skill_activation(
    storage_root: Path,
    activation_id: str,
    data: dict[str, Any],
) -> None:
    activations_dir = storage_root / "skill_activations"
    activations_dir.mkdir(parents=True, exist_ok=True)
    (activations_dir / f"{activation_id}.json").write_text(json.dumps(data))


def _write_thread(
    storage_root: Path,
    session_id: str,
    thread_id: str,
    created_at: str,
    messages: list[dict[str, Any]] | None = None,
) -> None:
    thread_dir = storage_root / "threads" / session_id / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    (thread_dir / "manifest.json").write_text(
        json.dumps({"id": thread_id, "session_id": session_id, "created_at": created_at})
    )
    if messages is not None:
        lines = [json.dumps(m) for m in messages]
        (thread_dir / "messages.ndjson").write_text("\n".join(lines) + "\n")


def _write_phase_change(storage_root: Path, session_id: str, after: str) -> None:
    from storage_event_log import LocalEventLogWriter

    async def _write() -> None:
        writer = LocalEventLogWriter(storage_root)
        await writer.append(
            session_id,
            "session.phase_change",
            {"before": "created", "after": after, "reason": "test", "timestamp": "t0"},
        )

    asyncio.run(_write())


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _default_session(session_id: str, agent_id: str = "agent-1") -> dict[str, Any]:
    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "status": "active",
        "created_at": "2024-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Basic response shape
# ---------------------------------------------------------------------------


class TestWakeEndpointSuccess:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wake-sess1", _default_session("wake-sess1"))
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/wake-sess1/wake")
        assert resp.status_code == 200

    def test_response_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wake-sess2", _default_session("wake-sess2"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/wake-sess2/wake").json()
        assert body["session_id"] == "wake-sess2"

    def test_response_status_is_awake(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wake-sess3", _default_session("wake-sess3"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/wake-sess3/wake").json()
        assert body["status"] == "awake"

    def test_response_includes_session_data(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        manifest = _default_session("wake-sess4")
        _write_session(storage_root, "wake-sess4", manifest)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/wake-sess4/wake").json()
        assert body["session"] == manifest

    def test_response_has_phase_field(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wake-sess5", _default_session("wake-sess5"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/wake-sess5/wake").json()
        assert "phase" in body

    def test_response_has_thread_id_field(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wake-sess6", _default_session("wake-sess6"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/wake-sess6/wake").json()
        assert "thread_id" in body

    def test_response_has_messages_field(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wake-sess7", _default_session("wake-sess7"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/wake-sess7/wake").json()
        assert "messages" in body


# ---------------------------------------------------------------------------
# Session not found
# ---------------------------------------------------------------------------


class TestWakeSessionNotFound:
    def test_missing_session_returns_404(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/no-such-sess/wake")
        assert resp.status_code == 404

    def test_missing_session_error_code(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/no-sess2/wake").json()
        assert body["error"]["code"] == "wake_session_not_found"

    def test_missing_session_writes_audit_log(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/no-audit-sess/wake")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.wake.failed" for r in records)

    def test_missing_session_audit_detail_has_session_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.post("/v1/x/sessions/no-detail-sess/wake")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.wake.failed")
        assert rec["detail"]["session_id"] == "no-detail-sess"


# ---------------------------------------------------------------------------
# AgentVersion loading
# ---------------------------------------------------------------------------


class TestWakeAgentVersion:
    def test_agent_version_none_when_no_agent_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(
            storage_root,
            "av-sess1",
            {"session_id": "av-sess1", "status": "active"},  # no agent_id
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/av-sess1/wake").json()
        assert body["agent_version"] is None

    def test_agent_version_none_when_agent_file_missing(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(
            storage_root,
            "av-sess2",
            {"session_id": "av-sess2", "agent_id": "missing-agent", "status": "active"},
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/av-sess2/wake").json()
        assert body["agent_version"] is None

    def test_agent_version_loaded_when_file_exists(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        agent_data = {"id": "agent-abc", "name": "My Agent", "version_id": "v1"}
        _write_agent(storage_root, "agent-abc", agent_data)
        _write_session(
            storage_root,
            "av-sess3",
            {"session_id": "av-sess3", "agent_id": "agent-abc", "status": "active"},
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/av-sess3/wake").json()
        assert body["agent_version"] == agent_data


# ---------------------------------------------------------------------------
# Active skills loading
# ---------------------------------------------------------------------------


class TestWakeActiveSkills:
    def test_active_skills_empty_when_none_exist(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "sk-sess1", _default_session("sk-sess1", "agent-x"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sk-sess1/wake").json()
        assert body["active_skills"] == []

    def test_active_skills_loaded_for_agent(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        activation = {
            "id": "skillact_001",
            "agent_id": "agent-y",
            "skill_id": "bash",
            "skill_version_id": "v1",
            "status": "active",
            "requested_at": "2024-01-01T00:00:00+00:00",
            "approved_at": "2024-01-01T00:01:00+00:00",
            "revoked_at": None,
        }
        _write_skill_activation(storage_root, "skillact_001", activation)
        _write_session(storage_root, "sk-sess2", _default_session("sk-sess2", "agent-y"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sk-sess2/wake").json()
        assert len(body["active_skills"]) == 1
        assert body["active_skills"][0]["skill_id"] == "bash"

    def test_active_skills_excludes_pending_status(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_skill_activation(
            storage_root,
            "skillact_pend",
            {
                "id": "skillact_pend",
                "agent_id": "agent-z",
                "skill_id": "fs",
                "status": "pending",
                "requested_at": "2024-01-01T00:00:00+00:00",
            },
        )
        _write_session(storage_root, "sk-sess3", _default_session("sk-sess3", "agent-z"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sk-sess3/wake").json()
        assert body["active_skills"] == []

    def test_active_skills_excludes_revoked_status(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_skill_activation(
            storage_root,
            "skillact_rev",
            {
                "id": "skillact_rev",
                "agent_id": "agent-w",
                "skill_id": "net",
                "status": "revoked",
                "requested_at": "2024-01-01T00:00:00+00:00",
                "revoked_at": "2024-01-02T00:00:00+00:00",
            },
        )
        _write_session(storage_root, "sk-sess4", _default_session("sk-sess4", "agent-w"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sk-sess4/wake").json()
        assert body["active_skills"] == []

    def test_active_skills_excludes_other_agents(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_skill_activation(
            storage_root,
            "skillact_other",
            {
                "id": "skillact_other",
                "agent_id": "other-agent",
                "skill_id": "bash",
                "status": "active",
                "requested_at": "2024-01-01T00:00:00+00:00",
            },
        )
        _write_session(storage_root, "sk-sess5", _default_session("sk-sess5", "my-agent"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sk-sess5/wake").json()
        assert body["active_skills"] == []

    def test_active_skills_empty_when_no_agent_id(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_skill_activation(
            storage_root,
            "skillact_any",
            {
                "id": "skillact_any",
                "agent_id": "some-agent",
                "skill_id": "bash",
                "status": "active",
            },
        )
        _write_session(
            storage_root,
            "sk-sess6",
            {"session_id": "sk-sess6", "status": "active"},  # no agent_id
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/sk-sess6/wake").json()
        assert body["active_skills"] == []


# ---------------------------------------------------------------------------
# Phase determination from event log
# ---------------------------------------------------------------------------


class TestWakePhase:
    def test_phase_is_created_when_no_events(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "ph-sess1", _default_session("ph-sess1"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ph-sess1/wake").json()
        assert body["phase"] == "created"

    def test_phase_reflects_last_phase_change_event(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "ph-sess2", _default_session("ph-sess2"))
        _write_phase_change(storage_root, "ph-sess2", "idle")
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ph-sess2/wake").json()
        assert body["phase"] == "idle"

    def test_phase_tracks_multiple_transitions(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "ph-sess3", _default_session("ph-sess3"))
        _write_phase_change(storage_root, "ph-sess3", "waiting_for_model")
        _write_phase_change(storage_root, "ph-sess3", "thinking")
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/ph-sess3/wake").json()
        assert body["phase"] == "thinking"


# ---------------------------------------------------------------------------
# Thread and message context reconstruction
# ---------------------------------------------------------------------------


class TestWakeThreadMessages:
    def test_thread_id_none_when_no_threads(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "th-sess1", _default_session("th-sess1"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/th-sess1/wake").json()
        assert body["thread_id"] is None

    def test_messages_empty_when_no_threads(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "th-sess2", _default_session("th-sess2"))
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/th-sess2/wake").json()
        assert body["messages"] == []

    def test_thread_id_returned_when_thread_exists(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "th-sess3", _default_session("th-sess3"))
        _write_thread(
            storage_root,
            "th-sess3",
            "thread-001",
            "2024-01-01T00:00:00+00:00",
            messages=[],
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/th-sess3/wake").json()
        assert body["thread_id"] == "thread-001"

    def test_messages_loaded_from_thread(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "th-sess4", _default_session("th-sess4"))
        msgs = [
            {"id": "msg-1", "role": "user", "content": "hello", "sequence": 1},
            {"id": "msg-2", "role": "assistant", "content": "hi", "sequence": 2},
        ]
        _write_thread(
            storage_root, "th-sess4", "thread-abc", "2024-01-01T00:00:00+00:00", messages=msgs
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/th-sess4/wake").json()
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][1]["role"] == "assistant"

    def test_most_recent_thread_selected_by_created_at(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "th-sess5", _default_session("th-sess5"))
        _write_thread(
            storage_root,
            "th-sess5",
            "old-thread",
            "2024-01-01T00:00:00+00:00",
            messages=[{"id": "old", "role": "user", "content": "old", "sequence": 1}],
        )
        _write_thread(
            storage_root,
            "th-sess5",
            "new-thread",
            "2024-06-01T00:00:00+00:00",
            messages=[{"id": "new", "role": "user", "content": "new", "sequence": 1}],
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/th-sess5/wake").json()
        assert body["thread_id"] == "new-thread"
        assert body["messages"][0]["content"] == "new"

    def test_messages_sorted_by_sequence(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "th-sess6", _default_session("th-sess6"))
        msgs = [
            {"id": "msg-3", "role": "assistant", "content": "reply", "sequence": 3},
            {"id": "msg-1", "role": "user", "content": "q1", "sequence": 1},
            {"id": "msg-2", "role": "user", "content": "q2", "sequence": 2},
        ]
        _write_thread(
            storage_root, "th-sess6", "thread-xyz", "2024-01-01T00:00:00+00:00", messages=msgs
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/th-sess6/wake").json()
        sequences = [m["sequence"] for m in body["messages"]]
        assert sequences == [1, 2, 3]

    def test_empty_messages_ndjson_returns_empty_list(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "th-sess7", _default_session("th-sess7"))
        _write_thread(
            storage_root,
            "th-sess7",
            "thread-empty",
            "2024-01-01T00:00:00+00:00",
            messages=[],
        )
        client = _make_client(storage_root, audit)
        body = client.post("/v1/x/sessions/th-sess7/wake").json()
        assert body["messages"] == []


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


class TestWakeRouterWiring:
    def test_wake_route_exists_with_storage_root(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        _write_session(storage_root, "wire-sess1", _default_session("wire-sess1"))
        client = _make_client(storage_root, audit)
        resp = client.post("/v1/x/sessions/wire-sess1/wake")
        assert resp.status_code != 404

    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/any/wake")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestWakeOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _make_client(self, storage_root: Path) -> TestClient:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        return TestClient(app, raise_server_exceptions=False)

    def test_success_emits_session_wake_span(self, storage_root: Path) -> None:
        client = self._make_client(storage_root)
        _write_session(storage_root, "otel-wake-sess", _default_session("otel-wake-sess"))
        client.post("/v1/x/sessions/otel-wake-sess/wake")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.wake" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = self._make_client(storage_root)
        client.post("/v1/x/sessions/no-such-otel-sess/wake")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        wake_span = spans.get("session.wake")
        assert wake_span is not None
        assert wake_span.status.status_code == StatusCode.ERROR
