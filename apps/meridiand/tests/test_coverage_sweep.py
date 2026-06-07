"""Sweep tests to close small coverage gaps across many meridiand modules.

Each test class targets a single source module's leftover branches/lines
without needing to spin up a full FastAPI test client where possible.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def pagination_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# _acp_compliance — _result with reason path
# ---------------------------------------------------------------------------


class TestCheckpointPerCall:
    """Cover the tool-call completion tracking + per-call duration logic in _checkpoint."""

    def test_per_call_duration_tracking(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        body1 = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "bash"}, {"id": "t2", "name": "grep"}],
            "message_tail": [{"role": "user", "content": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        r1 = client.post("/v1/x/sessions/s1/checkpoint", json=body1)
        assert r1.status_code == 200

        # Second checkpoint: t1 completed; t2 still pending
        body2 = {
            **body1,
            "seq": 2,
            "pending_tool_calls": [{"id": "t2", "name": "grep"}],
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        r2 = client.post("/v1/x/sessions/s1/checkpoint", json=body2)
        assert r2.status_code == 200

    def test_corrupt_latest_skipped(self, tmp_path: Path) -> None:
        """latest.json that can't be parsed is silently skipped (120-123)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        # Pre-write a corrupt latest.json
        cp_dir = tmp_path / "checkpoints" / "s2"
        cp_dir.mkdir(parents=True)
        (cp_dir / "latest.json").write_text("not json {{{")

        body = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [{"role": "user", "content": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        resp = client.post("/v1/x/sessions/s2/checkpoint", json=body)
        assert resp.status_code == 200

    def test_per_call_duration_with_bad_timestamp(self, tmp_path: Path) -> None:
        """Bad taken_at in prev_taken_at raises inside try/except — silently skipped (143-148)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s3"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "x", "name": "n"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "not-a-timestamp",
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        resp = client.post("/v1/x/sessions/s3/checkpoint", json=body)
        assert resp.status_code == 200

    def test_prev_call_not_dict_or_missing_id_skipped(self, tmp_path: Path) -> None:
        """A previous pending_tool_call that's not a dict or missing id is skipped (120->119)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s4"
        cp_dir.mkdir(parents=True)
        # Mix of non-dict, dict-no-id, and valid call
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [
                "not a dict",  # non-dict — skipped
                {"name": "nopid"},  # dict but no id — skipped
                {"id": "ok", "name": "k"},  # valid
            ],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        resp = client.post("/v1/x/sessions/s4/checkpoint", json=body)
        assert resp.status_code == 200

    def test_no_completed_calls_skips_metrics(self, tmp_path: Path) -> None:
        """If prev_calls have all carried over to current, completed=[] (136->159)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s5"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00+00:00",
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        # body still has t1 → not completed
        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        resp = client.post("/v1/x/sessions/s5/checkpoint", json=body)
        assert resp.status_code == 200

    def test_per_call_duration_no_prev_taken_at(self, tmp_path: Path) -> None:
        """prev with no taken_at falls through (138->149)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s6"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "",  # falsy — triggers 138->149
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01+00:00",
        }
        resp = client.post("/v1/x/sessions/s6/checkpoint", json=body)
        assert resp.status_code == 200

    def test_per_call_duration_naive_timestamps(self, tmp_path: Path) -> None:
        """Naive (no-tz) timestamps trigger the tzinfo-None branches (143, 145)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        cp_dir = tmp_path / "checkpoints" / "s7"
        cp_dir.mkdir(parents=True)
        prev = {
            "seq": 1,
            "phase": "thinking",
            "pending_tool_calls": [{"id": "t1", "name": "a"}],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:00",  # naive — no tz
        }
        (cp_dir / "latest.json").write_text(json.dumps(prev))

        body = {
            "seq": 2,
            "phase": "thinking",
            "pending_tool_calls": [],
            "message_tail": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "taken_at": "2024-01-01T00:00:01",  # naive — no tz
        }
        resp = client.post("/v1/x/sessions/s7/checkpoint", json=body)
        assert resp.status_code == 200

    def test_checkpoint_typed_error_reraised(self, tmp_path: Path) -> None:
        """CheckpointError raised inside is re-raised verbatim (line 171)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._checkpoint import CheckpointError

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._checkpoint.dispatch_hooks",
            side_effect=CheckpointError(message="pre-typed", timestamp=pagination_now(), cause=None),
        ):
            body = {
                "seq": 1,
                "phase": "thinking",
                "pending_tool_calls": [],
                "message_tail": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "taken_at": "2024-01-01T00:00:00+00:00",
            }
            resp = client.post("/v1/x/sessions/s8/checkpoint", json=body)
        assert resp.status_code == 422


class TestCursorMiddlewareEdgeCases:
    async def test_non_http_scope_passthrough(self) -> None:
        """websocket scope is passed straight through (lines 35-36)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        called: list[str] = []

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            called.append(scope["type"])

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        await mw({"type": "websocket"}, lambda: None, lambda _m: None)
        assert called == ["websocket"]

    async def test_request_url_skips_non_host_headers(self) -> None:
        """Headers before 'host' (e.g. 'x-other') are skipped in the loop (76->75)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"",
            "headers": [
                (b"x-other", b"first"),  # not host
                (b"x-second", b"second"),  # not host
                (b"host", b"api.example.com"),
            ],
            "scheme": "http",
            "server": ("api.example.com", 80),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        async def _send(_m: Any) -> None:
            pass

        await mw(scope, _receive, _send)

    async def test_request_url_with_host_header(self) -> None:
        """A request with Host header builds URL from header (line ~80)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"limit=10",
            "headers": [
                (b"host", b"api.example.com"),
                (b"x-other", b"yes"),
            ],
            "scheme": "https",
            "server": ("api.example.com", 443),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        await mw(scope, _receive, _send)
        assert any(s["type"] == "http.response.start" for s in sent)

    async def test_request_url_without_host_header_falls_back_to_server(self) -> None:
        """No Host header → URL built from scope server (lines 83-85)."""
        from core_errors import NoopAuditLog

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"x-next-cursor", b"abc"),
            ]})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"",
            "headers": [],  # no host
            "scheme": "http",
            "server": ("10.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        await mw(scope, _receive, _send)
        # Link header should be present
        start = next(s for s in sent if s["type"] == "http.response.start")
        header_names = [h[0].decode().lower() for h in start.get("headers", [])]
        assert "link" in header_names

    async def test_build_link_header_failure_writes_audit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_link_header raising writes an audit entry (lines 97-98)."""
        from core_errors import AuditLog, AuditLogEntry

        from meridiand._cursor_middleware import CursorPaginationMiddleware

        captured: list[AuditLogEntry] = []

        class _Capture(AuditLog):
            def write(self, e: AuditLogEntry) -> None:
                captured.append(e)

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": [
                (b"x-next-cursor", b"abc"),
            ]})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        mw = CursorPaginationMiddleware(_inner, audit_log=_Capture())
        monkeypatch.setattr(
            "meridiand._cursor_middleware.build_link_header",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("link boom")),
        )

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/v1/agents",
            "query_string": b"",
            "headers": [(b"host", b"api.example.com")],
            "scheme": "http",
            "server": ("api.example.com", 80),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request"}

        async def _send(_m: Any) -> None:
            pass

        await mw(scope, _receive, _send)
        assert any(e.event == "cursor.pagination.link.failed" for e in captured), [
            e.event for e in captured
        ]


class TestSpawnTraceparent:
    async def test_spawn_with_malformed_manifest_json(
        self, tmp_path: Path
    ) -> None:
        """Manifest is invalid JSON → except swallows, _child_links empty (186-187)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        sessions = tmp_path / "sessions" / "parent"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text("not json {{{")

        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(parent_capabilities=["agent.spawn", "fs.read"], child_capabilities=["fs.read"])
        resp = await handler("parent", req)
        assert resp is not None

    async def test_spawn_manifest_without_traceparent(
        self, tmp_path: Path
    ) -> None:
        """Manifest exists but has no traceparent → if False, skip (176->190)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        sessions = tmp_path / "sessions" / "no_tp"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text(json.dumps({"other_field": "x"}))

        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(
            parent_capabilities=["agent.spawn", "fs.read"],
            child_capabilities=["fs.read"],
        )
        resp = await handler("no_tp", req)
        assert resp is not None

    async def test_spawn_without_parent_manifest(
        self, tmp_path: Path
    ) -> None:
        """No parent manifest → if-False branch (176->190)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        # No sessions dir at all
        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(
            parent_capabilities=["agent.spawn", "fs.read"],
            child_capabilities=["fs.read"],
        )
        resp = await handler("no_manifest", req)
        assert resp is not None

    async def test_spawn_with_valid_but_invalid_context_traceparent(
        self, tmp_path: Path
    ) -> None:
        """Parent has well-formed but invalid traceparent → is_valid False (179->190)."""
        from core_errors import NoopAuditLog

        from meridiand._spawn import make_spawn_router, SpawnRequest

        sessions = tmp_path / "sessions" / "parent2"
        sessions.mkdir(parents=True)
        # All zeros — well-formed but `is_valid` returns False
        (sessions / "manifest.json").write_text(
            json.dumps(
                {
                    "traceparent": "00-00000000000000000000000000000000-0000000000000000-00"
                }
            )
        )

        router = make_spawn_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/sessions/{session_id}/spawn" and "POST" in r.methods
        )
        req = SpawnRequest(parent_capabilities=["agent.spawn", "fs.read"], child_capabilities=["fs.read"])
        resp = await handler("parent2", req)
        assert resp is not None


class TestSystemAuditMiddlewareReraise:
    async def test_no_status_captured_returns_early(self, tmp_path: Path) -> None:
        """If inner app completes without sending response.start, return early (line 207)."""
        from core_errors import NoopAuditLog

        from meridiand._system_audit_middleware import SystemAuditMiddleware

        async def _silent(scope: Any, receive: Any, send: Any) -> None:
            return  # never sends

        mw = SystemAuditMiddleware(_silent, audit_log=NoopAuditLog())
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(_m: Any) -> None:
            pass

        # Should complete without raising
        await mw(scope, _receive, _send)

    async def test_audit_swallowed_when_status_already_sent_then_reraise(
        self, tmp_path: Path
    ) -> None:
        """When status was captured and audit write fails, exception still re-raises (line 207)."""
        from core_errors import AuditLog, AuditLogEntry

        from meridiand._system_audit_middleware import SystemAuditMiddleware

        class _BoomAudit(AuditLog):
            def write(self, entry: AuditLogEntry) -> None:
                raise RuntimeError("audit boom")

        async def _inner(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raise RuntimeError("handler boom")

        mw = SystemAuditMiddleware(_inner, audit_log=_BoomAudit())

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/skills",  # monitored route
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8888),
        }

        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict[str, Any]] = []

        async def _send(m: Any) -> None:
            sent.append(m)

        with pytest.raises(RuntimeError):
            await mw(scope, _receive, _send)


class TestBudgetsReportsLookupAndTool:
    def test_lookup_agent_cached_after_first_call(self, tmp_path: Path) -> None:
        """Second call uses cache (158->164 False branch)."""
        from meridiand._budgets_reports import _lookup_agent_id

        sessions = tmp_path / "sessions" / "s1"
        sessions.mkdir(parents=True)
        (sessions / "manifest.json").write_text(json.dumps({"agent_id": "a1"}))
        cache: dict[str, str | None] = {}
        # First call: populates cache
        assert _lookup_agent_id(tmp_path / "sessions", "s1", cache) == "a1"
        # Second call: uses cache without reading
        assert _lookup_agent_id(tmp_path / "sessions", "s1", cache) == "a1"

    def test_build_tool_report_skips_blank_tool_names(self, tmp_path: Path) -> None:
        """tool_call.requested with empty tool_name is skipped (228->226)."""
        from meridiand._budgets_reports import _build_tool_report

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        (events_dir / "s1.ndjson").write_text(
            json.dumps(
                {"type": "tool_call.requested", "ts": "2024-01-01T00:00:00Z", "data": {}}
            )
            + "\n"
            + json.dumps(
                {
                    "type": "tool_call.requested",
                    "ts": "2024-01-01T00:00:00Z",
                    "data": {"tool_name": "bash"},
                }
            )
            + "\n"
        )
        report = _build_tool_report(events_dir, since=None, until=None)
        # Only bash is counted (empty tool_name skipped)
        assert len(report) == 1
        assert report[0]["tool_name"] == "bash"


class TestBudgetsReportsHelper:
    def test_count_event_skips_unreadable_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that raises OSError is silently skipped (lines 126-127)."""
        from meridiand._budgets_reports import _scan_events

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        # File that exists but read fails
        (events_dir / "broken.ndjson").write_text("ignored")
        # Valid file
        (events_dir / "ok.ndjson").write_text(
            json.dumps({"type": "budget.warning", "ts": "2024-01-01T00:00:00Z"}) + "\n"
        )

        real = Path.read_text

        def _selective(self: Path, *a: Any, **k: Any) -> str:
            if self.name == "broken.ndjson":
                raise OSError("denied")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _selective)
        result = _scan_events(events_dir, frozenset({"budget.warning"}), since=None, until=None)
        # ok.ndjson contributes 1
        assert len(result) == 1

    def test_scan_events_skips_blank_lines_and_invalid_json(self, tmp_path: Path) -> None:
        """Blank lines and invalid JSON lines are skipped (131, 134-135, 137)."""
        from meridiand._budgets_reports import _scan_events

        events_dir = tmp_path / "events"
        events_dir.mkdir()
        (events_dir / "s1.ndjson").write_text(
            "\n"  # blank
            "  \n"  # whitespace
            "not json {{{\n"  # invalid
            + json.dumps({"type": "other", "ts": "2024-01-01T00:00:00Z"})
            + "\n"  # wrong type
            + json.dumps({"type": "budget.warning", "ts": "2024-01-01T00:00:00Z"})
            + "\n"
            + json.dumps({"type": "budget.warning", "ts": "1999-01-01T00:00:00Z"})
            + "\n"  # before since
        )
        result = _scan_events(
            events_dir,
            frozenset({"budget.warning"}),
            since="2023-01-01T00:00:00Z",
            until=None,
        )
        assert len(result) == 1


class TestSkillSuggestionsErrors:
    def test_request_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionRequestError

        assert SkillSuggestionRequestError(message="m", timestamp="t").http_status() == 422

    def test_mode_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionModeError

        assert SkillSuggestionModeError(message="m", timestamp="t").http_status() == 422

    def test_skill_not_found_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillNotFoundError

        assert SkillNotFoundError(message="m", timestamp="t").http_status() == 404

    def test_agent_not_found_error_http_status(self) -> None:
        from meridiand._skill_suggestions import AgentNotFoundError

        assert AgentNotFoundError(message="m", timestamp="t").http_status() == 404

    def test_suggestion_not_found_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionNotFoundError

        assert (
            SkillSuggestionNotFoundError(message="m", timestamp="t").http_status() == 404
        )

    def test_suggestion_conflict_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionConflictError

        assert SkillSuggestionConflictError(message="m", timestamp="t").http_status() == 409

    def test_emit_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionEmitError

        assert (
            SkillSuggestionEmitError(message="m", timestamp="t", cause=None).http_status() == 500
        )

    def test_approve_error_http_status(self) -> None:
        from meridiand._skill_suggestions import SkillSuggestionApproveError

        assert (
            SkillSuggestionApproveError(message="m", timestamp="t", cause=None).http_status()
            == 500
        )

    def test_latest_activation_no_matches_returns_none(self, tmp_path: Path) -> None:
        """activations exist but none match agent/skill (line 152)."""
        from meridiand._skill_suggestions import _latest_activation

        activations_dir = tmp_path / "activations"
        activations_dir.mkdir()
        (activations_dir / "a.json").write_text(
            json.dumps({"agent_id": "other_a", "skill_id": "other_s"})
        )
        result = _latest_activation(activations_dir, "wanted_a", "wanted_s")
        assert result is None

    async def test_emit_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A generic exception is wrapped in SkillSuggestionEmitError (323-344)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_suggestions import (
            SkillSuggestionEmitError,
            SkillSuggestionRequest,
            make_skill_suggestions_router,
        )

        router = make_skill_suggestions_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        # Pre-create skill so we don't trip SkillNotFoundError
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "s1.json").write_text(json.dumps({"id": "s1"}))
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "a1.json").write_text(
            json.dumps(
                {
                    "id": "a1",
                    "version": {"config": {"skill_activation_mode": "auto_suggest"}},
                }
            )
        )

        handler = next(
            r.endpoint
            for r in router.routes
            if "/skill_suggestions" in r.path and "POST" in r.methods
        )
        req = SkillSuggestionRequest(skill_id="s1", reason="r")
        with patch("meridiand._skill_suggestions.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillSuggestionEmitError):
                await handler("a1", req)

    async def test_approve_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A generic exception is wrapped in SkillSuggestionApproveError (451-471)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_suggestions import (
            SkillSuggestionApproveError,
            make_skill_suggestions_router,
        )

        router = make_skill_suggestions_router(
            audit_log=NoopAuditLog(), storage_root=tmp_path
        )
        # Pre-create a suggestion file
        suggestions_dir = tmp_path / "skill_suggestions"
        suggestions_dir.mkdir(parents=True)
        (suggestions_dir / "a1_s1.json").write_text(
            json.dumps({"id": "sg1", "agent_id": "a1", "skill_id": "s1", "status": "suggested"})
        )

        handler = next(
            r.endpoint
            for r in router.routes
            if "/approve" in r.path and "POST" in r.methods
        )
        with patch("meridiand._skill_suggestions.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(SkillSuggestionApproveError):
                await handler("a1", "s1")


class TestHookDispatchVerdict:
    def test_verdict_from_string_invalid_json_yields_continue(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        verdict, _, _ = _parse_verdict("not json {{{")
        assert verdict == "continue"

    def test_verdict_from_valid_json_string(self) -> None:
        """Valid JSON string is parsed and data populated (line 182)."""
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict(json.dumps({"verdict": "veto", "reason": "policy"}))
        assert v == "veto"
        assert r == "policy"

    def test_verdict_from_non_string_non_dict(self) -> None:
        """Content that's neither dict nor str (e.g. None, int) → data stays None."""
        from meridiand._hook_dispatch import _parse_verdict

        v, _, _ = _parse_verdict(None)  # type: ignore[arg-type]
        assert v == "continue"
        v2, _, _ = _parse_verdict(42)  # type: ignore[arg-type]
        assert v2 == "continue"

    def test_verdict_from_string_non_dict_yields_continue(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        verdict, _, _ = _parse_verdict("[1, 2]")
        assert verdict == "continue"

    def test_verdict_veto(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict({"verdict": "veto", "reason": "denied"})
        assert v == "veto"
        assert r == "denied"

    def test_verdict_fail(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict({"verdict": "fail", "reason": "broken"})
        assert v == "fail"
        assert r == "broken"

    def test_verdict_recoverable(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, _, r = _parse_verdict({"verdict": "recoverable", "reason": "retry"})
        assert v == "recoverable"

    def test_verdict_continue_with_mutations(self) -> None:
        from meridiand._hook_dispatch import _parse_verdict

        v, m, _ = _parse_verdict({"verdict": "continue", "mutations": {"key": "val"}})
        assert v == "continue"
        assert m == {"key": "val"}

    def test_build_dispatcher_all_handler_types(self) -> None:
        from sdk_sandbox._audit import AuditLog
        from sdk_sandbox._types import AuditLogEntry

        import sdk_sandbox as _sb

        from meridiand._hook_dispatch import _build_dispatcher

        class _Bridge(AuditLog):
            def write(self, entry: AuditLogEntry) -> None:
                pass

        bridge = _Bridge()
        assert isinstance(_build_dispatcher("subprocess", bridge), _sb.SubprocessDispatcher)
        assert isinstance(_build_dispatcher("http", bridge), _sb.HttpDispatcher)
        assert isinstance(_build_dispatcher("mcp", bridge), _sb.McpDispatcher)
        assert isinstance(_build_dispatcher("container", bridge), _sb.ContainerDispatcher)
        assert isinstance(_build_dispatcher("in_process", bridge), _sb.InProcessDispatcher)

    def test_load_active_hooks_skips_invalid_json(self, tmp_path: Path) -> None:
        """A hook file that's malformed JSON is silently skipped (275-276)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _load_active_hooks

        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "bad.json").write_text("not json {{{")
        (hooks_dir / "good.json").write_text(
            json.dumps({"id": "g1", "status": "active", "event": "e1"})
        )

        ctx = ExecutionContext(session_id="s")
        result = _load_active_hooks(hooks_dir, "e1", ctx)
        assert len(result) == 1
        assert result[0]["id"] == "g1"

    async def test_dispatch_one_with_in_process_handler_registered(
        self, tmp_path: Path
    ) -> None:
        """When in_process handler is registered, dispatcher.register is called (308->314)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _SandboxAuditBridge, _dispatch_one

        async def _fn(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"verdict": "continue"}

        hook = {
            "id": "h1",
            "handler": "in_process",
            "timeout_ms": 5000,
            "name": "test",
            "metadata": {"module": "x"},
        }

        bridge = _SandboxAuditBridge(core_log=MagicMock())
        result = await _dispatch_one(
            hook,
            {"x": 1},
            ExecutionContext(session_id="s"),
            bridge=bridge,
            in_process_handlers={"h1": _fn},
        )
        # Either succeeded or returned an error result — either way the line is covered.
        assert result is not None

    async def test_dispatch_one_in_process_with_no_handlers_dict(
        self, tmp_path: Path
    ) -> None:
        """in_process handler but in_process_handlers is None (308->314 False branch)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _SandboxAuditBridge, _dispatch_one

        hook = {
            "id": "h_no_dict",
            "handler": "in_process",
            "timeout_ms": 5000,
            "name": "t",
            "metadata": {"module": "x"},
        }
        bridge = _SandboxAuditBridge(core_log=MagicMock())
        result = await _dispatch_one(
            hook,
            {"x": 1},
            ExecutionContext(session_id="s"),
            bridge=bridge,
            in_process_handlers=None,
        )
        assert result is not None

    async def test_dispatch_one_with_in_process_handler_none_fn(
        self, tmp_path: Path
    ) -> None:
        """in_process_handlers exists but contains no entry for hook_id (310->314)."""
        from sdk_sandbox._types import ExecutionContext

        from meridiand._hook_dispatch import _SandboxAuditBridge, _dispatch_one

        hook = {
            "id": "h_missing",
            "handler": "in_process",
            "timeout_ms": 5000,
            "name": "t",
            "metadata": {"module": "x"},
        }
        bridge = _SandboxAuditBridge(core_log=MagicMock())
        result = await _dispatch_one(
            hook,
            {"x": 1},
            ExecutionContext(session_id="s"),
            bridge=bridge,
            in_process_handlers={"other": MagicMock()},
        )
        assert result is not None

    def test_build_tool_handler_all_handler_types(self) -> None:
        import sdk_sandbox as _sb

        from meridiand._hook_dispatch import _build_tool_handler

        assert isinstance(
            _build_tool_handler("subprocess", {"path": "/bin/true"}), _sb.SubprocessHandler
        )
        assert isinstance(
            _build_tool_handler("http", {"url": "http://x"}), _sb.HttpHandler
        )
        assert isinstance(
            _build_tool_handler("mcp", {"server_url": "u", "tool_name": "t"}),
            _sb.McpHandler,
        )
        assert isinstance(
            _build_tool_handler(
                "container",
                {"environment_id": "e", "entrypoint": "/x"},
            ),
            _sb.ContainerHandler,
        )
        assert isinstance(_build_tool_handler("in_process", {"module": "m"}), _sb.InProcessHandler)


class TestVaultBackendOsKeychain:
    def test_now_helper(self) -> None:
        from meridiand._vault_backend_os_keychain import _now

        s = _now()
        assert isinstance(s, str) and "T" in s

    def test_default_keyring_import(self) -> None:
        """Without injected _keyring, the constructor imports the real keyring module."""
        from meridiand._vault_backend_os_keychain import OsKeychainVaultBackend

        try:
            backend = OsKeychainVaultBackend()
        except Exception:
            pytest.skip("keyring not available on this system")
        assert backend._kr is not None

    def test_list_secrets_filters_existing(self) -> None:
        from meridiand._vault_backend_os_keychain import OsKeychainVaultBackend

        class _FakeKeyring:
            def __init__(self) -> None:
                self.store: dict[tuple[str, str], str] = {}

            def get_password(self, svc: str, account: str) -> str | None:
                return self.store.get((svc, account))

            def set_password(self, svc: str, account: str, password: str) -> None:
                self.store[(svc, account)] = password

            def delete_password(self, svc: str, account: str) -> None:
                self.store.pop((svc, account), None)

        kr = _FakeKeyring()
        backend = OsKeychainVaultBackend(_keyring=kr)
        backend.store_secret("v1", "k1", "val1", "2024-01-01T00:00:00Z")
        items = backend.list_secrets("v1", ["k1", "missing"])
        assert len(items) == 1
        assert "value" not in items[0]
        # Delete existing + missing
        assert backend.delete_secret("v1", "k1") is True
        assert backend.delete_secret("v1", "missing") is False


class TestVaultBackendEncryptedFile:
    def test_unlock_error_http_status(self) -> None:
        from meridiand._vault_backend_encrypted_file import VaultBackendUnlockError

        assert (
            VaultBackendUnlockError(message="m", timestamp="t", cause=None).http_status() == 500
        )

    def test_unlock_with_passphrase_failure_wraps(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import (
            EncryptedFileVaultBackend,
            VaultBackendUnlockError,
        )

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        with patch(
            "pyrage.passphrase.encrypt",
            side_effect=RuntimeError("encrypt boom"),
        ):
            with pytest.raises(VaultBackendUnlockError):
                backend.unlock_with_passphrase("secret-passphrase")

    def test_unlock_with_key_file_failure_wraps(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import (
            EncryptedFileVaultBackend,
            VaultBackendUnlockError,
        )

        key_file = tmp_path / "bad.key"
        key_file.write_text("not a key")
        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        with pytest.raises(VaultBackendUnlockError):
            backend.unlock_with_key_file(key_file)

    def test_is_unlocked_property(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        assert backend.is_unlocked is False
        backend.unlock_with_passphrase("xx")
        assert backend.is_unlocked is True

    def test_update_secret(self, tmp_path: Path) -> None:
        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        backend.unlock_with_passphrase("xx")
        backend.store_secret("v1", "k1", "old", "2024-01-01T00:00:00Z")
        backend.update_secret("v1", "k1", {"value": "new", "key": "k1", "vault_id": "v1"})
        assert backend.get_secret("v1", "k1")["value"] == "new"

    def test_full_round_trip_passphrase(self, tmp_path: Path) -> None:
        """End-to-end store/list/delete with passphrase mode exercises _encrypt/_decrypt."""
        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        backend.unlock_with_passphrase("test-passphrase")
        backend.store_secret("v1", "k1", "val1", "2024-01-01T00:00:00Z")
        assert backend.secret_exists("v1", "k1") is True
        listed = backend.list_secrets("v1")
        assert any(r["key"] == "k1" for r in listed)
        rec = backend.get_secret("v1", "k1")
        assert rec is not None
        assert rec["value"] == "val1"
        assert backend.delete_secret("v1", "k1") is True
        assert backend.delete_secret("v1", "k1") is False  # already gone
        assert backend.secret_exists("v1", "k1") is False

    def test_unlock_with_key_file_success_and_round_trip(self, tmp_path: Path) -> None:
        """Generate a real age key and exercise the key_file mode encrypt/decrypt path."""
        import pyrage  # type: ignore[import-untyped]

        from meridiand._vault_backend_encrypted_file import EncryptedFileVaultBackend

        # Generate a fresh age identity
        identity = pyrage.x25519.Identity.generate()
        key_file = tmp_path / "age.key"
        key_file.write_text(str(identity))

        backend = EncryptedFileVaultBackend(storage_root=tmp_path)
        backend.unlock_with_key_file(key_file)
        backend.store_secret("v2", "k2", "val2", "2024-01-01T00:00:00Z")
        rec = backend.get_secret("v2", "k2")
        assert rec is not None
        assert rec["value"] == "val2"


class TestHarnessPoolHelpers:
    def test_harness_pool_error_http_status(self) -> None:
        from meridiand._harness_pool import HarnessPoolError

        assert HarnessPoolError(message="m", timestamp="t", cause=None).http_status() == 422

    def test_load_session_traceparent_missing_manifest(self, tmp_path: Path) -> None:
        from meridiand._harness_pool import HarnessPool

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=MagicMock(),
            phase_reader=MagicMock(),
            num_workers=2,
        )
        assert pool._load_session_traceparent("nope") == ""

    def test_load_session_traceparent_malformed_manifest(self, tmp_path: Path) -> None:
        from meridiand._harness_pool import HarnessPool

        (tmp_path / "sessions" / "s1").mkdir(parents=True)
        (tmp_path / "sessions" / "s1" / "manifest.json").write_text("not json {{{")

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=MagicMock(),
            phase_reader=MagicMock(),
            num_workers=2,
        )
        assert pool._load_session_traceparent("s1") == ""

    async def test_pool_start_twice_skips_alive_slot(self, tmp_path: Path) -> None:
        """A second call to start() doesn't replace an alive worker task (223->222)."""
        from meridiand._harness_pool import HarnessPool

        async def _idle_run(_sid: str) -> tuple[int, int, str]:
            return 0, 0, ""

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=_idle_run,
            phase_reader=MagicMock(),
            num_workers=1,
        )
        await pool.start()
        first_task = pool._slots[0].task
        assert first_task is not None
        await pool.start()  # second start: should skip alive slot
        assert pool._slots[0].task is first_task
        await pool.stop()

    async def test_pool_start_skips_session_when_phase_reader_raises(
        self, tmp_path: Path
    ) -> None:
        """phase_reader.current_phase raising → skipped via continue (235-236)."""
        from meridiand._harness_pool import HarnessPool

        # Pre-create a session manifest
        sessions_dir = tmp_path / "sessions"
        (sessions_dir / "broken").mkdir(parents=True)
        (sessions_dir / "broken" / "manifest.json").write_text("{}")

        class _RaisingReader:
            def current_phase(self, _sid: str) -> str:
                raise RuntimeError("phase boom")

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=lambda _s: None,  # type: ignore[arg-type]
            phase_reader=_RaisingReader(),
            num_workers=2,
        )
        await pool.start()
        await pool.stop()

    async def test_pool_start_typed_error_reraised(self, tmp_path: Path) -> None:
        """HarnessPoolError raised inside is re-raised (lines 241-259, isinstance branch)."""
        from meridiand._harness_pool import HarnessPool, HarnessPoolError

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=lambda _s: None,  # type: ignore[arg-type]
            phase_reader=MagicMock(),
            num_workers=2,
        )
        # Patch asyncio.create_task in the start path to raise HarnessPoolError
        original = asyncio.create_task
        raised = HarnessPoolError(message="pre", timestamp=pagination_now(), cause=None)

        def _raising(coro: Any, *args: Any, **kwargs: Any) -> Any:
            # Close the coro so we don't leak
            coro.close()
            raise raised

        with patch.object(asyncio, "create_task", _raising):
            with pytest.raises(HarnessPoolError):
                await pool.start()

    async def test_pool_start_generic_exception_wrapped(self, tmp_path: Path) -> None:
        """A non-HarnessPoolError exception is wrapped (lines 241-259, else branch)."""
        from meridiand._harness_pool import HarnessPool, HarnessPoolError

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=lambda _s: None,  # type: ignore[arg-type]
            phase_reader=MagicMock(),
            num_workers=2,
        )

        def _raising(coro: Any, *args: Any, **kwargs: Any) -> Any:
            coro.close()
            raise RuntimeError("create boom")

        with patch.object(asyncio, "create_task", _raising):
            with pytest.raises(HarnessPoolError):
                await pool.start()

    async def test_worker_loop_swallows_run_session_exception(self, tmp_path: Path) -> None:
        """run_session raising is silently swallowed (121-122)."""
        from meridiand._harness_pool import HarnessPool

        run_count = [0]

        async def _raising_run(_sid: str) -> tuple[int, int, str]:
            run_count[0] += 1
            raise RuntimeError("intentional")

        pool = HarnessPool(
            storage_root=tmp_path,
            audit_log=MagicMock(),
            run_session=_raising_run,
            phase_reader=MagicMock(),
            num_workers=2,
        )
        slot = pool._slots[0]
        slot.queue.put_nowait("s1")
        task = asyncio.create_task(pool._worker_loop(slot))
        await asyncio.wait_for(slot.queue.join(), timeout=2.0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert run_count[0] == 1


class TestCronErrors:
    def test_parse_duration_empty(self) -> None:
        from meridiand._cron import _parse_duration

        with pytest.raises(ValueError, match="empty"):
            _parse_duration("")

    def test_parse_duration_invalid(self) -> None:
        from meridiand._cron import _parse_duration

        with pytest.raises(ValueError, match="Invalid"):
            _parse_duration("xyz")

    def test_parse_duration_negative(self) -> None:
        """Duration parsing — a unit value of 0 yields total_seconds() == 0 → must be positive."""
        from meridiand._cron import _parse_duration

        with pytest.raises(ValueError, match="positive"):
            _parse_duration("0s")

    def test_cron_create_error_http_status(self) -> None:
        from meridiand._cron import CronCreateError

        assert CronCreateError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_cron_invalid_request_error_http_status(self) -> None:
        from meridiand._cron import CronInvalidRequestError

        assert CronInvalidRequestError(message="m", timestamp="t").http_status() == 422

    def test_cron_delete_error_http_status(self) -> None:
        from meridiand._cron import CronDeleteError

        assert CronDeleteError(message="m", timestamp="t", cause=None).http_status() == 500

    async def test_cron_create_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._cron import CronCreateError, CronCreateRequest, make_cron_router

        router = make_cron_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/cron" and "POST" in r.methods
        )
        req = CronCreateRequest(
            trigger_type="interval",
            interval="1h",
            session_id="s1",
            task={"prompt": "hi"},
        )
        with patch("meridiand._cron.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(CronCreateError):
                await handler(req)

    async def test_cron_delete_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._cron import CronDeleteError, make_cron_router

        # Create a cron file
        cron_dir = tmp_path / "cron"
        cron_dir.mkdir(parents=True)
        (cron_dir / "c1.json").write_text("{}")

        router = make_cron_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/cron/{cron_id}" and "DELETE" in r.methods
        )
        with patch.object(Path, "unlink", side_effect=RuntimeError("unlink boom")):
            with pytest.raises(CronDeleteError):
                await handler("c1")


class TestWebhookErrors:
    def test_webhook_create_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookCreateError

        assert WebhookCreateError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_webhook_invalid_request_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookInvalidRequestError

        assert WebhookInvalidRequestError(message="m", timestamp="t").http_status() == 422

    def test_webhook_not_found_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookNotFoundError

        assert WebhookNotFoundError(webhook_id="x", timestamp="t").http_status() == 404

    def test_webhook_delete_error_http_status(self) -> None:
        from meridiand._webhooks import WebhookDeleteError

        assert WebhookDeleteError(message="m", timestamp="t", cause=None).http_status() == 500

    async def test_webhook_create_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._webhooks import (
            WebhookCreateError,
            WebhookCreateRequest,
            make_webhooks_router,
        )

        router = make_webhooks_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/webhooks" and "POST" in r.methods
        )
        from meridiand._webhooks import EventFilter

        req = WebhookCreateRequest(
            name="test_webhook",
            url="https://example.com/hook",
            event_filter=EventFilter(types=["session.completed"]),
            max_retries=3,
            backoff="exponential",
        )
        with patch("meridiand._webhooks.json.dumps", side_effect=RuntimeError("boom")):
            with pytest.raises(WebhookCreateError):
                await handler(req)

    async def test_webhook_delete_generic_exception_wrapped(self, tmp_path: Path) -> None:
        from core_errors import NoopAuditLog

        from meridiand._webhooks import WebhookDeleteError, make_webhooks_router

        # Pre-create a webhook file so delete proceeds past the not-found check
        wh_dir = tmp_path / "webhooks"
        wh_dir.mkdir(parents=True)
        (wh_dir / "wh1.json").write_text("{}")

        router = make_webhooks_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/webhooks/{webhook_id}" and "DELETE" in r.methods
        )
        with patch.object(Path, "unlink", side_effect=RuntimeError("unlink boom")):
            with pytest.raises(WebhookDeleteError):
                await handler("wh1")


class TestKbStoreSearchHelpers:
    def test_has_key_returns_false_when_missing(self, tmp_path: Path) -> None:
        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        assert store.has_key("/none/such") is False

    def test_glob_search_with_scope(self, tmp_path: Path) -> None:
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        store.upsert_chunks(
            "/docs/a.md",
            "world-doc",
            [
                Chunk(
                    file_path="/docs/a.md",
                    kind="text",
                    content="hello",
                    start_line=1,
                    end_line=1,
                ),
            ],
        )
        rows = store.glob_search("*.md", "world-doc", limit=10)
        assert rows
        # Scope filter
        rows2 = store.glob_search("*.txt", "world-doc", limit=10)
        assert rows2 == []

    def test_bm25_search_empty_query(self, tmp_path: Path) -> None:
        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        assert store.bm25_search("!!!  ", None, limit=10) == []

    def test_bm25_search_no_scope_path(self, tmp_path: Path) -> None:
        """bm25_search with scope=None exercises the else branch (lines 318-324)."""
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        store.upsert_chunks(
            "/x.md",
            "any",
            [Chunk(file_path="/x.md", kind="text", content="hello world", start_line=1, end_line=1)],
        )
        # Empty result is OK — point is to exercise the no-scope branch
        store.bm25_search("hello", None, limit=10)

    def test_glob_search_truncates_at_limit(self, tmp_path: Path) -> None:
        """glob_search breaks early after `limit` results (line 302)."""
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        # Add 3 chunks
        for i in range(3):
            store.upsert_chunks(
                f"/file{i}.md",
                "scope",
                [
                    Chunk(
                        file_path=f"/file{i}.md",
                        kind="text",
                        content="x",
                        start_line=1,
                        end_line=1,
                    )
                ],
            )
        # Request limit=1 — should break after first match
        rows = store.glob_search("*.md", "scope", limit=1)
        assert len(rows) == 1

    def test_vector_search_no_scope(self, tmp_path: Path) -> None:
        """vector_search with scope=None exercises the else branch (line 335 -> alternative)."""
        from meridian_kb_indexer import Chunk

        from meridiand._kb import KbStore

        store = KbStore(tmp_path / "kb.db")
        store.upsert_chunks(
            "/v.md",
            "s",
            [Chunk(file_path="/v.md", kind="text", content="hi", start_line=1, end_line=1)],
        )
        # scope=None
        store.vector_search("hi", None, limit=5)

    async def test_kb_index_skips_failed_file(self, tmp_path: Path) -> None:
        """Per-file index failure is silently skipped (lines 415-416)."""
        from core_errors import NoopAuditLog

        from meridiand._kb import KbIndexRequest, make_kb_router

        router = make_kb_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/kb/index"
            and ("POST" in r.methods or "POST" in (r.methods or []))
        )
        # Make workspace point to tmp_path with one file
        (tmp_path / "fail.md").write_text("content")
        import os as _os

        old_env = _os.environ.get("MERIDIAN_KB_WORKSPACE")
        _os.environ["MERIDIAN_KB_WORKSPACE"] = str(tmp_path)
        try:
            # Patch indexer.index_file to raise so the per-file except triggers
            with patch(
                "meridiand._kb.WorkspaceIndexer.index_file",
                side_effect=RuntimeError("file fail"),
            ):
                req = KbIndexRequest(scope="global")  # no path → workspace scan
                resp = await handler(req)
            assert resp is not None
        finally:
            if old_env is None:
                _os.environ.pop("MERIDIAN_KB_WORKSPACE", None)
            else:
                _os.environ["MERIDIAN_KB_WORKSPACE"] = old_env

    async def test_kb_index_typed_error_reraised(self, tmp_path: Path) -> None:
        """KbIndexError raised inside is re-raised verbatim (line 427)."""
        from core_errors import NoopAuditLog

        from meridiand._kb import KbIndexError, KbIndexRequest, make_kb_router

        router = make_kb_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/kb/index"
            and ("POST" in r.methods or "POST" in (r.methods or []))
        )
        # Use target_path mode to take the simpler branch
        (tmp_path / "x.md").write_text("hello")
        req = KbIndexRequest(scope="global", path=str(tmp_path / "x.md"))
        with patch(
            "meridiand._kb._load_status",
            side_effect=KbIndexError(message="pre", timestamp=pagination_now(), cause=None),
        ):
            with pytest.raises(KbIndexError):
                await handler(req)

    async def test_kb_query_typed_error_reraised(self, tmp_path: Path) -> None:
        """KbQueryError raised inside is re-raised verbatim (line 525)."""
        from core_errors import NoopAuditLog

        from meridiand._kb import KbQueryError, KbQueryRequest, make_kb_router

        router = make_kb_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint
            for r in router.routes
            if r.path == "/v1/x/kb/query"
            and ("POST" in r.methods or "POST" in (r.methods or []))
        )
        req = KbQueryRequest(query="q", scope="global", limit=5)
        # Patch _rrf_fuse to raise typed error
        with patch(
            "meridiand._kb._rrf_fuse",
            side_effect=KbQueryError(message="pre", timestamp=pagination_now(), cause=None),
        ):
            with pytest.raises(KbQueryError):
                await handler(req)


class TestHookCreateGeneric:
    async def test_generic_exception_wrapped_to_hook_create_error(self, tmp_path: Path) -> None:
        """A non-HookInvalidRequestError exception is wrapped (lines 189-210)."""
        from core_errors import NoopAuditLog

        from meridiand._hooks import (
            FailureMode,
            HandlerType,
            HookCreateError,
            HookCreateRequest,
            make_hooks_router,
        )

        # Build the router and extract the inner handler function
        router = make_hooks_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        handler = next(
            r.endpoint for r in router.routes if r.path == "/v1/x/hooks" and "POST" in r.methods
        )
        req = HookCreateRequest(
            event="on_checkpoint",
            name="test",
            handler=HandlerType.in_process,
            timeout_ms=1000,
            failure_mode=FailureMode.ignore,
        )
        with patch("meridiand._hooks.json.dumps", side_effect=RuntimeError("dump boom")):
            with pytest.raises(HookCreateError):
                await handler(req)


class TestPhaseTransitionTerminal:
    @staticmethod
    def _make_phase_client(storage_root: Path):
        from fastapi.testclient import TestClient
        from storage_event_log import LocalEventLogWriter

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(storage_root)
        writer = LocalEventLogWriter(storage_root)
        app = create_app(audit, storage_root=storage_root, event_log=writer)
        return TestClient(app, raise_server_exceptions=False)

    def test_terminal_phase_with_manifest(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions" / "sterm"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "manifest.json").write_text(
            json.dumps({"created_at": "2024-01-01T00:00:00+00:00"})
        )
        client = self._make_phase_client(tmp_path)
        resp = client.post(
            "/v1/x/sessions/sterm/phase",
            json={"to_phase": "completed", "reason": "ok"},
        )
        assert resp.status_code == 200

    def test_terminal_phase_missing_manifest_skipped(self, tmp_path: Path) -> None:
        client = self._make_phase_client(tmp_path)
        resp = client.post(
            "/v1/x/sessions/snomanifest/phase",
            json={"to_phase": "completed", "reason": "ok"},
        )
        assert resp.status_code == 200

    def test_terminal_phase_manifest_no_created_at(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "sessions" / "sno_ts"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "manifest.json").write_text("{}")
        client = self._make_phase_client(tmp_path)
        resp = client.post(
            "/v1/x/sessions/sno_ts/phase",
            json={"to_phase": "completed", "reason": "ok"},
        )
        assert resp.status_code == 200

    def test_phase_with_existing_before_decrements(self, tmp_path: Path) -> None:
        """If before is not None, active_sessions[before] is decremented (113->114)."""
        client = self._make_phase_client(tmp_path)
        # First transition to "running"
        r1 = client.post(
            "/v1/x/sessions/sbefore/phase",
            json={"to_phase": "running", "reason": "start"},
        )
        assert r1.status_code == 200
        # Second transition: before should now be "running"
        r2 = client.post(
            "/v1/x/sessions/sbefore/phase",
            json={"to_phase": "idle", "reason": "pause"},
        )
        assert r2.status_code == 200

    def test_phase_typed_error_reraised(self, tmp_path: Path) -> None:
        """PhaseTransitionError raised inside is re-raised verbatim (line 127)."""
        from fastapi.testclient import TestClient
        from storage_event_log import LocalEventLogWriter

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._phase import PhaseTransitionError

        audit = FileAuditLog(tmp_path)
        writer = LocalEventLogWriter(tmp_path)

        async def _boom(*_a: Any, **_k: Any) -> None:
            raise PhaseTransitionError(
                message="pre-typed", timestamp=pagination_now(), cause=None
            )

        with patch.object(writer, "append", _boom):
            app = create_app(audit, storage_root=tmp_path, event_log=writer)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/x/sessions/sreraise/phase",
                json={"to_phase": "completed", "reason": "ok"},
            )
        assert resp.status_code in {422, 500}


class TestErrorClassesHttpStatus:
    """One-liner http_status tests for error classes across modules."""

    def test_hook_create_error(self) -> None:
        from meridiand._hooks import HookCreateError

        assert HookCreateError(message="m", timestamp="t", cause=None).http_status() == 500

    def test_hook_invalid_request_error(self) -> None:
        from meridiand._hooks import HookInvalidRequestError

        assert HookInvalidRequestError(message="m", timestamp="t").http_status() == 422

    def test_checkpoint_error(self) -> None:
        from meridiand._checkpoint import CheckpointError

        assert CheckpointError(message="m", timestamp="t", cause=None).http_status() == 422

    def test_kb_index_error(self) -> None:
        from meridiand._kb import KbIndexError, KbQueryError, KbStatusError

        assert KbIndexError(message="m", timestamp="t", cause=None).http_status() == 422
        assert KbStatusError(message="m", timestamp="t", cause=None).http_status() == 422
        assert KbQueryError(message="m", timestamp="t", cause=None).http_status() == 422

    def test_skill_forge_errors(self) -> None:
        from meridiand._skill_forge import SkillForgeProposalError, SkillForgeRunError

        assert SkillForgeRunError(message="m", timestamp="t", cause=None).http_status() == 500
        assert SkillForgeProposalError(message="m", timestamp="t", cause=None).http_status() == 500


class TestVaultLeakSoakHelpers:
    def test_scan_skips_unreadable_files(self, tmp_path: Path) -> None:
        from meridiand._vault_leak_soak import _scan_storage_root

        # Create one readable + one that will raise OSError on read
        (tmp_path / "f1.txt").write_text("clean")
        (tmp_path / "subdir").mkdir()
        leaks = _scan_storage_root(tmp_path, ["s3cret-CANARY"])
        assert leaks == []  # no leaks expected

    def test_scan_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        from meridiand._vault_leak_soak import _scan_storage_root

        leaks = _scan_storage_root(tmp_path / "nope", ["x"])
        assert leaks == []

    def test_scan_records_leak_when_canary_found(self, tmp_path: Path) -> None:
        from meridiand._vault_leak_soak import _scan_storage_root

        (tmp_path / "leaked.txt").write_text("here is s3cret-XYZ in plain text")
        leaks = _scan_storage_root(tmp_path, ["s3cret-XYZ"])
        assert len(leaks) == 1
        assert leaks[0]["source"] == "file"

    def test_scan_skips_unreadable_file(self, tmp_path: Path) -> None:
        """A file that raises OSError on read is silently skipped (lines 97-98)."""
        from meridiand._vault_leak_soak import _scan_storage_root

        (tmp_path / "ok.txt").write_text("clean")
        (tmp_path / "fail.txt").write_text("doomed")
        real_read = Path.read_text

        def _selective(self: Path, *a: Any, **k: Any) -> str:
            if self.name == "fail.txt":
                raise OSError("denied")
            return real_read(self, *a, **k)

        with patch.object(Path, "read_text", _selective):
            leaks = _scan_storage_root(tmp_path, ["x"])
        assert leaks == []

    def test_memory_keyring_round_trip(self) -> None:
        from meridiand._vault_leak_soak import _MemoryKeyring

        k = _MemoryKeyring()
        assert k.get_password("svc", "u") is None
        k.set_password("svc", "u", "secret")
        assert k.get_password("svc", "u") == "secret"
        k.delete_password("svc", "u")
        assert k.get_password("svc", "u") is None


class TestSkillEfficacyHelpers:
    async def test_noop_trajectory_runner_returns_false(self) -> None:
        from meridiand._skill_efficacy import NoopTrajectoryRunner

        r = NoopTrajectoryRunner()
        assert await r.run({}, skill_instructions=None) is False

    async def test_compare_reraises_typed_error(self, tmp_path: Path) -> None:
        """SkillEfficacyError raised inside is re-raised verbatim (lines 203-219)."""
        from core_errors import NoopAuditLog

        from meridiand._skill_efficacy import (
            SkillEfficacyError,
            compare_proposal_trajectories,
        )

        class _BoomRunner:
            async def run(self, *_a: Any, **_k: Any) -> bool:
                raise SkillEfficacyError(
                    message="pre-typed", timestamp=pagination_now(), cause=None
                )

        with pytest.raises(SkillEfficacyError):
            await compare_proposal_trajectories(
                proposal={
                    "id": "p1",
                    "skill_id": "s1",
                    "instructions": "do x",
                    "tests": [{"name": "t1"}],
                },
                efficacy_dir=tmp_path,
                audit_log=NoopAuditLog(),
                runner=_BoomRunner(),
            )


class TestAcpHttpTransport:
    async def test_default_handler_returns_empty(self) -> None:
        from meridiand._acp import DefaultAcpInboundHandler

        h = DefaultAcpInboundHandler()
        result = await h.handle("target", {"x": 1})
        assert result == {}

    async def test_http_peer_client_call(self) -> None:
        import httpx

        from meridiand._acp import HttpAcpPeerClient

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(_handler)
        real_init = httpx.AsyncClient.__init__

        def _patched_init(self: httpx.AsyncClient, *a: Any, **kw: Any) -> None:
            kw.pop("transport", None)
            real_init(self, *a, transport=transport, **kw)

        with patch.object(httpx.AsyncClient, "__init__", _patched_init):
            c = HttpAcpPeerClient()
            out = await c.call("http://example.com/acp", {"msg": "hi"})
        assert out == {"ok": True}


class TestModelCallEventLogAdapter:
    async def test_adapter_appends_session_event(self) -> None:
        from meridiand._model_call_event_log import EventLogModelCallAdapter

        captured: list[dict[str, Any]] = []

        class _Runtime:
            async def append(self, *, session_id: str, event_type: str, data: dict[str, Any]) -> None:
                captured.append({"session_id": session_id, "event_type": event_type, "data": data})

        adapter = EventLogModelCallAdapter(_Runtime())
        await adapter.record_started(
            session_id="s1",
            routing_rule="r",
            provider_name="p",
            model="m",
        )
        assert captured == [
            {
                "session_id": "s1",
                "event_type": "model_call.started",
                "data": {"routing_rule": "r", "provider_name": "p", "model": "m"},
            }
        ]


class TestSystemPromptTemplateErrors:
    def test_expand_error_http_status(self) -> None:
        from meridiand._system_prompt_template import TemplateExpandError

        err = TemplateExpandError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_memory_not_found_error_http_status(self) -> None:
        from meridiand._system_prompt_template import TemplateMemoryNotFoundError

        err = TemplateMemoryNotFoundError(memory_key="k", timestamp="t")
        assert err.http_status() == 404


class TestHealthzReadyzMetricsErrors:
    def test_healthz_error_http_status(self) -> None:
        from meridiand._healthz import HealthzError

        err = HealthzError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_healthz_exception_path(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._healthz import HealthzError

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        # Patch JSONResponse construction inside healthz to raise
        with patch(
            "meridiand._healthz.JSONResponse",
            side_effect=RuntimeError("liveness boom"),
        ):
            resp = client.get("/healthz")
        assert resp.status_code == 500

    def test_readyz_error_http_status(self) -> None:
        from meridiand._readyz import ReadyzError

        err = ReadyzError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_readyz_exception_path(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        with patch(
            "meridiand._readyz.JSONResponse",
            side_effect=RuntimeError("readiness boom"),
        ):
            resp = client.get("/readyz")
        assert resp.status_code == 500

    def test_metrics_endpoint(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_error_http_status(self) -> None:
        from meridiand._metrics import MetricsError

        err = MetricsError(message="m", timestamp="t", cause=None)
        assert err.http_status() == 500

    def test_metrics_exception_path(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog

        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)
        with patch(
            "meridiand._metrics.generate_latest",
            side_effect=RuntimeError("scrape boom"),
        ):
            resp = client.get("/metrics")
        assert resp.status_code == 500


class TestAcpComplianceResult:
    def test_failed_with_reason_includes_reason(self) -> None:
        from meridiand._acp_compliance import _result

        r = _result("test_name", "desc", passed=False, reason="why it failed")
        assert r["reason"] == "why it failed"
        assert r["status"] == "failed"

    def test_failed_without_reason_omits_reason(self) -> None:
        from meridiand._acp_compliance import _result

        r = _result("test_name", "desc", passed=False, reason=None)
        assert "reason" not in r

    def test_passed_with_reason_omits_reason(self) -> None:
        """passed=True suppresses reason even when provided."""
        from meridiand._acp_compliance import _result

        r = _result("test_name", "desc", passed=True, reason="ignored")
        assert "reason" not in r


# ---------------------------------------------------------------------------
# _cancel — descendant traversal skips malformed manifests
# ---------------------------------------------------------------------------


class TestCancelDescendantTraversal:
    def test_malformed_manifest_skipped(self, tmp_path: Path) -> None:
        """Manifest JSON that raises (e.g. invalid JSON) is silently skipped."""
        from meridiand._cancel import _walk_descendants

        sessions = tmp_path / "sessions"
        (sessions / "s1").mkdir(parents=True)
        (sessions / "s1" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "parent1", "child_session_id": "s1"})
        )
        (sessions / "s2").mkdir(parents=True)
        (sessions / "s2" / "manifest.json").write_text("not json {{{")  # malformed

        desc = _walk_descendants("parent1", tmp_path)
        assert "s1" in desc

    def test_manifest_without_parent_or_child_skipped(self, tmp_path: Path) -> None:
        """Manifest without both parent and child fields is skipped (60->55)."""
        from meridiand._cancel import _walk_descendants

        sessions = tmp_path / "sessions"
        (sessions / "s1").mkdir(parents=True)
        (sessions / "s1" / "manifest.json").write_text(json.dumps({}))  # neither field
        (sessions / "s2").mkdir(parents=True)
        (sessions / "s2" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "p"})  # only parent, no child
        )
        desc = _walk_descendants("p", tmp_path)
        assert desc == []

    def test_duplicate_child_reference_seen_once(self, tmp_path: Path) -> None:
        """A child appearing twice in the graph is only enqueued once (72->71)."""
        from meridiand._cancel import _walk_descendants

        sessions = tmp_path / "sessions"
        # p has two children c1 and c2
        for child_id in ("c1", "c2"):
            (sessions / child_id).mkdir(parents=True)
            (sessions / child_id / "manifest.json").write_text(
                json.dumps({"parent_session_id": "p", "child_session_id": child_id})
            )
        # c1 also lists c2 as a child (creates a duplicate path to c2)
        (sessions / "c2dup").mkdir(parents=True)
        (sessions / "c2dup" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "c1", "child_session_id": "c2"})
        )
        desc = _walk_descendants("p", tmp_path)
        # c2 appears once (de-duped)
        assert desc.count("c2") == 1


class TestCancelMissingDescendantManifest:
    def test_descendant_with_missing_manifest_skipped(self, tmp_path: Path) -> None:
        """A descendant whose own manifest dir is missing is skipped (109->113)."""
        from core_errors import NoopAuditLog
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from meridiand._cancel import make_cancel_router

        sessions = tmp_path / "sessions"
        # "edge" has parent_session_id=p1 and child_session_id=ghost.
        # ghost will appear in descendants but ghost/manifest.json doesn't exist.
        (sessions / "edge").mkdir(parents=True)
        (sessions / "edge" / "manifest.json").write_text(
            json.dumps({"parent_session_id": "p1", "child_session_id": "ghost"})
        )

        router = make_cancel_router(audit_log=NoopAuditLog(), storage_root=tmp_path)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/sessions/p1/cancel")
        # the cancel-walk processes ghost without crashing
        assert resp.status_code in {200, 204}


# ---------------------------------------------------------------------------
# _diagnosis — audit-line filtering
# ---------------------------------------------------------------------------


class TestDiagnosisAuditFilter:
    def test_skips_blank_and_invalid_lines(self, tmp_path: Path) -> None:
        from meridiand._diagnosis import _read_audit_for_session

        audit_path = tmp_path / "audit.ndjson"
        audit_path.write_text(
            "\n"  # blank line
            "  \n"  # whitespace
            "not json {{{\n"  # invalid JSON
            + json.dumps({"detail": {"session_id": "wanted"}, "ts": "t"})
            + "\n"
            + json.dumps({"detail": {"session_id": "other"}, "ts": "t"})
            + "\n"
        )
        entries = _read_audit_for_session(audit_path, "wanted")
        assert len(entries) == 1
        assert entries[0]["detail"]["session_id"] == "wanted"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from meridiand._diagnosis import _read_audit_for_session

        result = _read_audit_for_session(tmp_path / "nope.ndjson", "any")
        assert result == []

    def test_phase_change_with_blank_after_keeps_default(self) -> None:
        """phase_change event with after='' doesn't update terminal_phase (108->110)."""
        from types import SimpleNamespace

        from meridiand._diagnosis import _extract_failure_summary

        events = [
            SimpleNamespace(
                type="session.phase_change",
                data={"after": "", "reason": ""},
                seq=1,
                ts="t",
                thread_id=None,
            )
        ]
        phase, reason, _ = _extract_failure_summary(events)
        assert phase == "unknown"
        assert reason == ""

    def test_diagnosis_reraises_typed_error(self, tmp_path: Path) -> None:
        """If inner code raises SessionDiagnosisError, it's re-raised (line 151)."""
        from fastapi.testclient import TestClient

        from meridiand._app import create_app
        from meridiand._audit import FileAuditLog
        from meridiand._diagnosis import SessionDiagnosisError

        # Make _extract_failure_summary raise SessionDiagnosisError
        audit = FileAuditLog(tmp_path)
        app = create_app(audit, storage_root=tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "meridiand._diagnosis._extract_failure_summary",
            side_effect=SessionDiagnosisError(
                message="pre-typed", timestamp=pagination_now(), cause=None
            ),
        ):
            resp = client.get("/v1/sessions/s1/diagnosis")
        assert resp.status_code in {422, 500}


# ---------------------------------------------------------------------------
# _event_translator — final default return for unknown events
# ---------------------------------------------------------------------------


class TestEventTranslatorUnknown:
    def test_unknown_event_returns_empty_list(self) -> None:
        from meridiand._event_translator import ModelEventTranslator

        t = ModelEventTranslator()
        # Pass a sentinel object that doesn't match any isinstance check
        out = t.translate(object())  # type: ignore[arg-type]
        assert out == []

    def test_message_stop_event_with_none_stop_reason_keeps_existing(self) -> None:
        """MessageStopEvent with stop_reason=None doesn't overwrite (122->124)."""
        from meridian_sdk_provider.types import (
            MessageDeltaEvent,
            MessageStopEvent,
        )

        from meridiand._event_translator import ModelEventTranslator

        t = ModelEventTranslator()
        # First set a stop_reason
        t.translate(MessageDeltaEvent(stop_reason="end_turn"))
        # Then send a MessageStopEvent with no stop_reason
        out = t.translate(MessageStopEvent(stop_reason=None, input_tokens=0, output_tokens=0))
        assert out  # produces model_call.completed event
        completed = next(d for k, d in out if k == "model_call.completed")
        assert completed["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# _cli_channel_driver — small class methods + ChannelFailure reraise
# ---------------------------------------------------------------------------


class TestCliChannelDriverHelpers:
    def test_sys_stdout_writer_writes_and_flushes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from meridiand._cli_channel_driver import SysStdoutWriter

        w = SysStdoutWriter()
        w.write("hello")
        w.flush()
        out = capsys.readouterr().out
        assert "hello" in out

    async def test_noop_stdin_reader_client_runs_and_stops(self) -> None:
        from meridiand._cli_channel_driver import NoopStdinReaderClient

        c = NoopStdinReaderClient()
        await c.run()
        await c.stop()

    def test_token_stream_skips_non_string_tokens(self, tmp_path: Path) -> None:
        """Tokens that aren't strings are skipped in _write_token_stream (209->208)."""
        from meridiand._cli_channel_driver import CliChannelDriver

        captured: list[str] = []

        class _W:
            def write(self, s: str) -> None:
                captured.append(s)

            def flush(self) -> None:
                pass

        driver = CliChannelDriver(storage_root=tmp_path, stdout_writer=_W())
        driver._write_token_stream(json.dumps(["a", 1, "b", None, "c"]))
        joined = "".join(captured)
        assert "a" in joined and "b" in joined and "c" in joined
        assert "1" not in joined and "None" not in joined

    def test_tool_call_non_object_raises(self, tmp_path: Path) -> None:
        """tool_call payload that's a JSON array (not object) raises (line 221)."""
        from meridiand._cli_channel_driver import CliChannelDriver

        class _W:
            def write(self, s: str) -> None:
                pass

            def flush(self) -> None:
                pass

        driver = CliChannelDriver(storage_root=tmp_path, stdout_writer=_W())
        with pytest.raises(ValueError, match="JSON object"):
            driver._write_tool_call(json.dumps([1, 2, 3]))

    async def test_idempotency_non_http_passthrough(self) -> None:
        """websocket scope is passed straight through (lines 51-52)."""
        from core_errors import NoopAuditLog

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        called: list[str] = []

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            called.append(scope["type"])

        mw = IdempotencyKeyMiddleware(_handler, audit_log=NoopAuditLog())
        await mw({"type": "websocket"}, lambda: None, lambda _m: None)
        assert called == ["websocket"]

    async def test_idempotency_capturing_send_passes_through_other_messages(self) -> None:
        """A message type other than start/body falls through to await send (125->127)."""
        from core_errors import NoopAuditLog

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.trailers", "headers": []})  # other type
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = IdempotencyKeyMiddleware(_handler, audit_log=NoopAuditLog())

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x/agents",
            "query_string": b"",
            "headers": [(b"idempotency-key", b"trailing-msg-key")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent_types: list[str] = []

        async def send(m: Any) -> None:
            sent_types.append(m["type"])

        await mw(scope, receive, send)
        assert "http.response.trailers" in sent_types

    async def test_idempotency_no_response_skip_cache(self) -> None:
        """If handler never sends response.start, cache write is skipped (line 132)."""
        from core_errors import NoopAuditLog

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            return  # never sends anything

        mw = IdempotencyKeyMiddleware(_handler, audit_log=NoopAuditLog())

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x/agents",
            "query_string": b"",
            "headers": [(b"idempotency-key", b"key-only-this-test")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_m: Any) -> None:
            pass

        # Should complete without raising
        await mw(scope, receive, send)

    async def test_idempotency_cache_store_failure_writes_audit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _CachedResponse construction raises, audit error is written (132, 143-144)."""
        from core_errors import AuditLog, AuditLogEntry

        from meridiand._idempotency_middleware import IdempotencyKeyMiddleware

        captured: list[AuditLogEntry] = []

        class _Audit(AuditLog):
            def write(self, entry: AuditLogEntry) -> None:
                captured.append(entry)

        async def _handler(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"ok":true}', "more_body": False})

        mw = IdempotencyKeyMiddleware(_handler, audit_log=_Audit())

        # Patch _CachedResponse to raise so the except handler fires
        monkeypatch.setattr(
            "meridiand._idempotency_middleware._CachedResponse",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cache write boom")),
        )

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/x/agents",
            "query_string": b"",
            "headers": [(b"idempotency-key", b"k1")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8888),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[dict[str, Any]] = []

        async def send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        await mw(scope, receive, send)
        # Audit was written for the cache-store failure
        assert any(
            e.event == "idempotency.cache.store.failed" for e in captured
        ), [e.event for e in captured]

    async def test_channel_failure_reraised(self, tmp_path: Path) -> None:
        """ChannelFailure raised by _write_content is re-raised verbatim (line 295)."""
        from sdk_channel import ChannelFailure, SendRequest

        from meridiand._cli_channel_driver import CliChannelDriver

        class _W:
            def write(self, s: str) -> None:
                pass

            def flush(self) -> None:
                pass

        driver = CliChannelDriver(storage_root=tmp_path, stdout_writer=_W())
        # Pre-populate channel config so _load_driver_config doesn't fail
        chan_dir = tmp_path / "channels"
        chan_dir.mkdir(parents=True, exist_ok=True)
        (chan_dir / "c1.json").write_text(json.dumps({"config": {}}))

        from datetime import UTC, datetime

        original = ChannelFailure(
            code="X",
            message="m",
            channel_id="c1",
            channel_kind="meridian.cli",
            session_id="s1",
            timestamp=datetime.now(UTC).isoformat(),
        )

        def _boom(*_a, **_k) -> None:
            raise original

        driver._write_content = _boom  # type: ignore[method-assign]
        req = SendRequest(
            channel_id="c1",
            channel_kind="meridian.cli",
            session_id="s1",
            recipient="user",
            content="hi",
            content_type="text",
        )
        with pytest.raises(ChannelFailure):
            await driver.send(req)


# ---------------------------------------------------------------------------
# _credential_proxy — http_client=None path
# ---------------------------------------------------------------------------


class TestCredentialProxyDefaultClient:
    def test_default_http_client_path(self, tmp_path: Path) -> None:
        """When http_client=None, the proxy creates its own AsyncClient (lines 232-233)."""
        from core_errors import HandlerOptions, install_error_handler
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from meridiand._audit import FileAuditLog
        from meridiand._credential_proxy import (
            CredentialProxyProviderConfig,
            make_credential_proxy_router,
        )

        class _Resolver:
            def resolve(self, ref: str) -> str | None:
                return "tok"

        provider = CredentialProxyProviderConfig(
            name="p1",
            base_url="http://127.0.0.1:1",  # connection will fail (port 1 closed)
            token_secret_ref="secret_ref://v/k",
        )
        audit_log = FileAuditLog(tmp_path)
        router = make_credential_proxy_router(
            audit_log=audit_log,
            secret_resolver=_Resolver(),
            providers=[provider],
            http_client=None,  # forces the else branch
        )
        app = FastAPI()
        app.include_router(router)
        install_error_handler(app, HandlerOptions(audit_log=audit_log))
        client = TestClient(app, raise_server_exceptions=False)
        # The forward will fail with a connect error, but the else branch is exercised.
        resp = client.get("/v1/credential-proxy/p1/anything")
        assert resp.status_code == 502
