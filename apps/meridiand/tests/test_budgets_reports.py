"""
Budget cost breakdown reports: GET /v1/x/budgets/reports

Tests cover:
  - group_by=agent aggregates usage.delta tokens by agent_id across sessions.
  - group_by=session aggregates usage.delta tokens per session with agent_id.
  - group_by=model aggregates usage.delta tokens by provider/model pair.
  - group_by=tool counts tool_call.requested events by tool_name.
  - Invalid group_by returns 422 with budget_report_invalid_group_by code.
  - since / until filters events by the ts field.
  - Empty events directory returns empty items list.
  - OTel span "budgets.reports" is emitted on each invocation.
  - On failure, error message is surfaced in response and audit log entry written.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    return TestClient(
        create_app(FileAuditLog(storage_root), storage_root=storage_root),
        raise_server_exceptions=False,
    )


def _write_events(storage_root: Path, session_id: str, events: list[dict]) -> None:
    """Write a NDJSON event log file for session_id under events/2026/01/01/."""
    events_dir = storage_root / "events" / "2026" / "01" / "01"
    events_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(e) for e in events) + "\n"
    (events_dir / f"{session_id}.ndjson").write_text(lines)


def _write_manifest(storage_root: Path, session_id: str, agent_id: str | None) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "session_id": session_id,
        "agent_id": agent_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest))


def _usage_delta(
    seq: int,
    ts: str = "2026-01-01T12:00:00+00:00",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    provider: str = "anthropic",
    model: str = "claude-3",
) -> dict:
    return {
        "seq": seq,
        "ts": ts,
        "type": "usage.delta",
        "data": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "provider": provider,
            "model": model,
        },
    }


def _tool_requested(seq: int, tool_name: str, ts: str = "2026-01-01T12:00:00+00:00") -> dict:
    return {
        "seq": seq,
        "ts": ts,
        "type": "tool_call.requested",
        "data": {"tool_id": f"tool-{seq}", "tool_name": tool_name, "args": {}},
    }


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# group_by=agent
# ---------------------------------------------------------------------------


class TestGroupByAgent:
    def test_single_session_single_agent(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "sess-a1", "agent-1")
        _write_events(
            storage_root, "sess-a1", [_usage_delta(0, prompt_tokens=200, completion_tokens=100)]
        )
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=agent")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_by"] == "agent"
        items = body["items"]
        assert len(items) == 1
        assert items[0]["agent_id"] == "agent-1"
        assert items[0]["input_tokens"] == 200
        assert items[0]["output_tokens"] == 100
        assert items[0]["cache_tokens"] == 0

    def test_multiple_sessions_same_agent_are_summed(self, storage_root: Path) -> None:
        for i in range(3):
            sid = f"sess-multi-{i}"
            _write_manifest(storage_root, sid, "agent-shared")
            _write_events(storage_root, sid, [_usage_delta(0, prompt_tokens=100)])
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=agent")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["input_tokens"] == 300

    def test_sessions_with_different_agents_produce_separate_rows(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "s-ag1", "agent-A")
        _write_events(storage_root, "s-ag1", [_usage_delta(0, prompt_tokens=100)])
        _write_manifest(storage_root, "s-ag2", "agent-B")
        _write_events(storage_root, "s-ag2", [_usage_delta(0, prompt_tokens=200)])
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=agent")
        assert resp.status_code == 200
        items = {r["agent_id"]: r for r in resp.json()["items"]}
        assert items["agent-A"]["input_tokens"] == 100
        assert items["agent-B"]["input_tokens"] == 200

    def test_session_without_manifest_groups_under_empty_agent_id(self, storage_root: Path) -> None:
        _write_events(storage_root, "sess-no-manifest", [_usage_delta(0, prompt_tokens=50)])
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=agent")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert any(r["agent_id"] == "" for r in items)

    def test_cache_tokens_are_summed(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "sess-cache", "agent-c")
        _write_events(
            storage_root,
            "sess-cache",
            [_usage_delta(0, cache_creation_tokens=30, cache_read_tokens=20)],
        )
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=agent")
        assert resp.status_code == 200
        item = resp.json()["items"][0]
        assert item["cache_tokens"] == 50


# ---------------------------------------------------------------------------
# group_by=session
# ---------------------------------------------------------------------------


class TestGroupBySession:
    def test_each_session_has_own_row(self, storage_root: Path) -> None:
        for i in range(2):
            sid = f"sess-s{i}"
            _write_manifest(storage_root, sid, f"agent-{i}")
            _write_events(storage_root, sid, [_usage_delta(0, prompt_tokens=10 * (i + 1))])
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=session")
        assert resp.status_code == 200
        items = {r["session_id"]: r for r in resp.json()["items"]}
        assert items["sess-s0"]["input_tokens"] == 10
        assert items["sess-s1"]["input_tokens"] == 20

    def test_agent_id_is_included_in_session_row(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "sess-with-agent", "my-agent")
        _write_events(storage_root, "sess-with-agent", [_usage_delta(0)])
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=session")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert items[0]["agent_id"] == "my-agent"

    def test_multiple_deltas_in_same_session_are_summed(self, storage_root: Path) -> None:
        _write_manifest(storage_root, "sess-multi-delta", "agent-x")
        _write_events(
            storage_root,
            "sess-multi-delta",
            [_usage_delta(0, prompt_tokens=100), _usage_delta(1, prompt_tokens=200)],
        )
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=session")
        assert resp.status_code == 200
        assert resp.json()["items"][0]["input_tokens"] == 300

    def test_response_envelope_includes_group_by_since_until(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).get(
            "/v1/x/budgets/reports?group_by=session&since=2026-01-01T00:00:00Z&until=2026-12-31T23:59:59Z"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["group_by"] == "session"
        assert body["since"] == "2026-01-01T00:00:00Z"
        assert body["until"] == "2026-12-31T23:59:59Z"


# ---------------------------------------------------------------------------
# group_by=model
# ---------------------------------------------------------------------------


class TestGroupByModel:
    def test_different_models_produce_separate_rows(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess-models",
            [
                _usage_delta(0, provider="anthropic", model="claude-3", prompt_tokens=100),
                _usage_delta(1, provider="openai", model="gpt-4o", prompt_tokens=200),
            ],
        )
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=model")
        assert resp.status_code == 200
        items = {(r["provider"], r["model"]): r for r in resp.json()["items"]}
        assert items[("anthropic", "claude-3")]["input_tokens"] == 100
        assert items[("openai", "gpt-4o")]["input_tokens"] == 200

    def test_same_model_across_sessions_is_summed(self, storage_root: Path) -> None:
        for i in range(3):
            _write_events(
                storage_root,
                f"sess-model-sum-{i}",
                [_usage_delta(0, provider="anthropic", model="claude-3", prompt_tokens=50)],
            )
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=model")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["input_tokens"] == 150

    def test_delta_without_provider_model_groups_under_empty_strings(
        self, storage_root: Path
    ) -> None:
        events_dir = storage_root / "events" / "2026" / "01" / "01"
        events_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "seq": 0,
            "ts": "2026-01-01T12:00:00+00:00",
            "type": "usage.delta",
            "data": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        (events_dir / "sess-no-model.ndjson").write_text(json.dumps(record) + "\n")
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=model")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert any(r["provider"] == "" and r["model"] == "" for r in items)


# ---------------------------------------------------------------------------
# group_by=tool
# ---------------------------------------------------------------------------


class TestGroupByTool:
    def test_tool_call_counts_are_per_tool_name(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess-tools",
            [
                _tool_requested(0, "bash"),
                _tool_requested(1, "bash"),
                _tool_requested(2, "read_file"),
            ],
        )
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=tool")
        assert resp.status_code == 200
        items = {r["tool_name"]: r["call_count"] for r in resp.json()["items"]}
        assert items["bash"] == 2
        assert items["read_file"] == 1

    def test_tool_calls_across_sessions_are_summed(self, storage_root: Path) -> None:
        for i in range(3):
            _write_events(storage_root, f"sess-tool-sum-{i}", [_tool_requested(0, "bash")])
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=tool")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["tool_name"] == "bash"
        assert items[0]["call_count"] == 3


# ---------------------------------------------------------------------------
# Time window filtering
# ---------------------------------------------------------------------------


class TestTimeWindowFiltering:
    def test_since_excludes_older_events(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess-time",
            [
                _usage_delta(0, ts="2026-01-01T06:00:00+00:00", prompt_tokens=100),
                _usage_delta(1, ts="2026-01-01T14:00:00+00:00", prompt_tokens=200),
            ],
        )
        resp = _make_client(storage_root).get(
            "/v1/x/budgets/reports?group_by=session&since=2026-01-01T10:00:00+00:00"
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["input_tokens"] == 200

    def test_until_excludes_newer_events(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess-until",
            [
                _usage_delta(0, ts="2026-01-01T06:00:00+00:00", prompt_tokens=100),
                _usage_delta(1, ts="2026-01-01T14:00:00+00:00", prompt_tokens=200),
            ],
        )
        resp = _make_client(storage_root).get(
            "/v1/x/budgets/reports?group_by=session&until=2026-01-01T10:00:00+00:00"
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["input_tokens"] == 100

    def test_since_and_until_together_narrow_window(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess-window",
            [
                _usage_delta(0, ts="2026-01-01T04:00:00+00:00", prompt_tokens=10),
                _usage_delta(1, ts="2026-01-01T12:00:00+00:00", prompt_tokens=20),
                _usage_delta(2, ts="2026-01-01T20:00:00+00:00", prompt_tokens=30),
            ],
        )
        resp = _make_client(storage_root).get(
            "/v1/x/budgets/reports?group_by=session"
            "&since=2026-01-01T06:00:00+00:00"
            "&until=2026-01-01T18:00:00+00:00"
        )
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert items[0]["input_tokens"] == 20

    def test_no_events_in_window_returns_empty_items(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess-out",
            [_usage_delta(0, ts="2026-06-01T00:00:00+00:00", prompt_tokens=100)],
        )
        resp = _make_client(storage_root).get(
            "/v1/x/budgets/reports?group_by=session&until=2026-01-01T00:00:00+00:00"
        )
        assert resp.status_code == 200
        assert resp.json()["items"] == []


# ---------------------------------------------------------------------------
# Empty storage
# ---------------------------------------------------------------------------


class TestEmptyStorage:
    def test_no_events_directory_returns_empty_items(self, storage_root: Path) -> None:
        for group_by in ("agent", "session", "tool", "model"):
            resp = _make_client(storage_root).get(f"/v1/x/budgets/reports?group_by={group_by}")
            assert resp.status_code == 200
            assert resp.json()["items"] == [], f"expected empty items for group_by={group_by}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_group_by_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).get("/v1/x/budgets/reports")
        assert resp.status_code == 422

    def test_invalid_group_by_returns_422(self, storage_root: Path) -> None:
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=invalid")
        assert resp.status_code == 422

    def test_invalid_group_by_error_code(self, storage_root: Path) -> None:
        body = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=xyz").json()
        assert body["error"]["code"] == "budget_report_invalid_group_by"


# ---------------------------------------------------------------------------
# OTel span
# ---------------------------------------------------------------------------


class TestOtelSpan:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_span_emitted_on_success(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/x/budgets/reports?group_by=session")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "budgets.reports" in span_names

    def test_span_emitted_on_invalid_group_by(self, storage_root: Path) -> None:
        _make_client(storage_root).get("/v1/x/budgets/reports?group_by=bad")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "budgets.reports" in span_names


# ---------------------------------------------------------------------------
# Audit log on failure
# ---------------------------------------------------------------------------


class TestAuditLogOnFailure:
    def test_failure_writes_audit_entry(
        self, storage_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import meridiand._budgets_reports as _mod

        def _boom(*_a: object, **_kw: object) -> list:
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(_mod, "_build_session_report", _boom)
        _make_client(storage_root).get("/v1/x/budgets/reports?group_by=session")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "budgets.reports.failed" for r in records)

    def test_failure_error_is_surfaced_in_response(
        self, storage_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import meridiand._budgets_reports as _mod

        def _boom(*_a: object, **_kw: object) -> list:
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(_mod, "_build_session_report", _boom)
        resp = _make_client(storage_root).get("/v1/x/budgets/reports?group_by=session")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["code"] == "budget_report_failed"
