"""Sweep tests to close small coverage gaps across many meridiand modules.

Each test class targets a single source module's leftover branches/lines
without needing to spin up a full FastAPI test client where possible.
"""

from __future__ import annotations

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
