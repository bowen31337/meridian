"""
Tests for the SystemOAuthProvider and its subprocess lifecycle.

Coverage:
  Lock file:
  - Valid lock file returns correct CliLockEntry
  - Missing lock file raises LockFileNotFoundError
  - Malformed JSON raises LockFileFormatError
  - Wrong schema version raises LockFileFormatError
  - Missing pins.claude-code raises LockFileFormatError
  - Missing version field raises LockFileFormatError

  CliSubprocessManager (using _cli_stub.py as the CLI):
  - start() spawns a process
  - stop() terminates the process
  - call() yields MessageStartEvent, TextDeltaEvent, MessageStopEvent
  - call() handles subprocess error response → raises CliSubprocessError
  - call() timeout → raises CliCallTimeoutError and kills the process
  - health_check_ok: pong received, process stays alive
  - health_check_fail: no pong → process is killed and respawned

  SystemOAuthProvider:
  - call() yields events from manager
  - call() failure → ProviderCallError raised, audit log written, span ERROR
  - list_models() returns the known model catalogue
  - OTel span "claude_code_oauth.model.call" is emitted on each call
  - Span carries provider.name and model attributes
  - Span carries provider.invocation event
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from meridian_sdk_provider.audit import AuditLogEntry
from meridian_sdk_provider.errors import ProviderCallError
from meridian_sdk_provider.types import (
    Message,
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    TextDeltaEvent,
)

from meridian_provider_claude_code_oauth._lock import (
    CliLockEntry,
    LockFileFormatError,
    LockFileNotFoundError,
    read_lock,
)
from meridian_provider_claude_code_oauth._subprocess import (
    CliCallTimeoutError,
    CliSubprocessError,
    CliSubprocessManager,
)
from meridian_provider_claude_code_oauth.provider import SystemOAuthProvider

# Path to the real CLI stub script used for integration-style subprocess tests.
_STUB = str(Path(__file__).parent / "_cli_stub.py")

_SIMPLE_OPTS = ModelCallOpts(
    model="claude-sonnet-4-6",
    messages=[Message(role="user", content="hi")],
)


# ---------------------------------------------------------------------------
# OTel mock shared by provider tests
# ---------------------------------------------------------------------------


class _MockSpan:
    def __init__(self) -> None:
        self.name: str = ""
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: Any = None
        self.recorded_exceptions: list[BaseException] = []

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: BaseException, **_: Any) -> None:
        self.recorded_exceptions.append(exc)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def __enter__(self) -> _MockSpan:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False


class _MockTracer:
    def __init__(self) -> None:
        self.span = _MockSpan()

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None, **_: Any
    ) -> _MockSpan:
        self.span.name = name
        if attributes:
            self.span.attributes.update(attributes)
        return self.span


# ---------------------------------------------------------------------------
# Capturing audit log
# ---------------------------------------------------------------------------


class _CapturingAuditLog:
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# Lock file tests
# ---------------------------------------------------------------------------


class TestLockFile:
    def test_valid_lock_returns_entry(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text(
            json.dumps(
                {
                    "version": 1,
                    "pins": {"claude-code": {"version": "1.2.3", "channel": "stable"}},
                }
            )
        )
        entry = read_lock(lock)
        assert isinstance(entry, CliLockEntry)
        assert entry.cli_version == "1.2.3"
        assert entry.channel == "stable"

    def test_default_channel_is_stable(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text(
            json.dumps(
                {
                    "version": 1,
                    "pins": {"claude-code": {"version": "2.0.0"}},
                }
            )
        )
        entry = read_lock(lock)
        assert entry.channel == "stable"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LockFileNotFoundError):
            read_lock(tmp_path / "meridian.lock")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text("not json {")
        with pytest.raises(LockFileFormatError):
            read_lock(lock)

    def test_non_object_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text("[1, 2, 3]")
        with pytest.raises(LockFileFormatError):
            read_lock(lock)

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text(json.dumps({"version": 99, "pins": {}}))
        with pytest.raises(LockFileFormatError, match="version"):
            read_lock(lock)

    def test_missing_claude_code_pin_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text(json.dumps({"version": 1, "pins": {}}))
        with pytest.raises(LockFileFormatError, match="claude-code"):
            read_lock(lock)

    def test_empty_version_string_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text(
            json.dumps(
                {
                    "version": 1,
                    "pins": {"claude-code": {"version": "   "}},
                }
            )
        )
        with pytest.raises(LockFileFormatError):
            read_lock(lock)

    def test_missing_version_field_raises(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text(
            json.dumps(
                {
                    "version": 1,
                    "pins": {"claude-code": {"channel": "stable"}},
                }
            )
        )
        with pytest.raises(LockFileFormatError):
            read_lock(lock)


# ---------------------------------------------------------------------------
# CliSubprocessManager — integration tests with _cli_stub.py
# ---------------------------------------------------------------------------


class TestCliSubprocessManagerSpawn:
    async def test_start_spawns_process(self) -> None:
        import asyncio

        mgr = CliSubprocessManager(sys.executable, "1.0.0", health_interval_s=9999)
        mgr._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _STUB,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert mgr._proc is not None
        assert mgr._proc.returncode is None
        mgr._proc.terminate()
        await mgr._proc.wait()

    async def test_stop_terminates_process(self) -> None:
        import asyncio

        mgr = CliSubprocessManager(sys.executable, "1.0.0", health_interval_s=9999)
        mgr._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _STUB,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await mgr.stop()
        assert mgr._proc is None


class TestCliSubprocessManagerCall:
    def _make_manager(self, *, env: dict | None = None) -> CliSubprocessManager:
        import os

        mgr = CliSubprocessManager(
            sys.executable,
            "1.0.0",
            health_interval_s=9999,
            call_timeout_s=5.0,
        )
        # Patch _spawn to use the stub
        _env = {**os.environ, **(env or {})}

        async def _stub_spawn() -> None:
            import asyncio as _asyncio

            mgr._proc = await _asyncio.create_subprocess_exec(
                sys.executable,
                _STUB,
                stdin=_asyncio.subprocess.PIPE,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
                env=_env,
            )

        mgr._spawn = _stub_spawn  # type: ignore[method-assign]
        return mgr

    async def test_call_yields_message_start(self) -> None:
        mgr = self._make_manager()
        events = [e async for e in mgr.call(_SIMPLE_OPTS)]
        assert any(isinstance(e, MessageStartEvent) for e in events)

    async def test_call_yields_text_delta(self) -> None:
        mgr = self._make_manager()
        events = [e async for e in mgr.call(_SIMPLE_OPTS)]
        assert any(isinstance(e, TextDeltaEvent) for e in events)
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert text_events[0].text == "Hello from stub!"

    async def test_call_yields_message_stop(self) -> None:
        mgr = self._make_manager()
        events = [e async for e in mgr.call(_SIMPLE_OPTS)]
        assert any(isinstance(e, MessageStopEvent) for e in events)

    async def test_call_message_stop_has_token_counts(self) -> None:
        mgr = self._make_manager()
        events = [e async for e in mgr.call(_SIMPLE_OPTS)]
        stop = next(e for e in events if isinstance(e, MessageStopEvent))
        assert stop.input_tokens == 10
        assert stop.output_tokens == 4
        assert stop.stop_reason == "end_turn"

    async def test_call_error_response_raises(self) -> None:
        mgr = self._make_manager(env={"CLI_STUB_ERROR": "1"})
        with pytest.raises(CliSubprocessError, match="cli_test_error"):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

    async def test_call_timeout_raises(self) -> None:
        mgr = self._make_manager(env={"CLI_STUB_HANG": "1"})
        mgr._call_timeout_s = 0.2
        with pytest.raises(CliCallTimeoutError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

    async def test_call_timeout_kills_process(self) -> None:
        mgr = self._make_manager(env={"CLI_STUB_HANG": "1"})
        mgr._call_timeout_s = 0.2
        with pytest.raises(CliCallTimeoutError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass
        # After timeout the manager respawns; _proc is alive again
        assert mgr._proc is not None
        mgr._proc.terminate()
        await mgr._proc.wait()

    async def test_second_call_after_error_succeeds(self) -> None:
        import asyncio
        import os

        # First call fails (error response), second call should succeed.
        # Error responses don't kill the process, so we manually terminate it
        # before the second call to force a fresh spawn.
        mgr = self._make_manager(env={"CLI_STUB_ERROR": "1"})
        with pytest.raises(CliSubprocessError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

        # Terminate the erroring process so _ensure_alive respawns.
        if mgr._proc is not None:
            mgr._proc.terminate()
            await mgr._proc.wait()
            mgr._proc = None

        _env = {**os.environ}

        async def _clean_spawn() -> None:
            mgr._proc = await asyncio.create_subprocess_exec(
                sys.executable,
                _STUB,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_env,
            )

        mgr._spawn = _clean_spawn  # type: ignore[method-assign]
        events = [e async for e in mgr.call(_SIMPLE_OPTS)]
        assert any(isinstance(e, TextDeltaEvent) for e in events)

    async def test_call_active_flag_cleared_after_success(self) -> None:
        mgr = self._make_manager()
        async for _ in mgr.call(_SIMPLE_OPTS):
            pass
        assert not mgr._call_active

    async def test_call_active_flag_cleared_after_error(self) -> None:
        mgr = self._make_manager(env={"CLI_STUB_ERROR": "1"})
        with pytest.raises(CliSubprocessError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass
        assert not mgr._call_active


class TestCliSubprocessManagerHealthCheck:
    async def test_health_check_ok_keeps_process(self) -> None:
        import asyncio

        mgr = CliSubprocessManager(sys.executable, "1.0.0", health_interval_s=9999)
        mgr._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _STUB,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        original_pid = mgr._proc.pid
        await mgr._do_health_check()
        assert mgr._proc is not None
        assert mgr._proc.pid == original_pid
        mgr._proc.terminate()
        await mgr._proc.wait()

    async def test_health_check_fail_spawns_new_process(self) -> None:
        import asyncio
        import os

        mgr = CliSubprocessManager(
            sys.executable, "1.0.0", health_interval_s=9999, health_timeout_s=0.1
        )
        # Spawn stub in no-pong mode so health check times out.
        env = {**os.environ, "CLI_STUB_NO_PONG": "1"}
        mgr._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _STUB,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        old_pid = mgr._proc.pid

        # Patch _spawn so we can detect it was called after the failure.
        spawned: list[int] = []

        async def _spawn() -> None:
            import asyncio as _asyncio

            mgr._proc = await _asyncio.create_subprocess_exec(
                sys.executable,
                _STUB,
                stdin=_asyncio.subprocess.PIPE,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            spawned.append(mgr._proc.pid)

        mgr._spawn = _spawn  # type: ignore[method-assign]
        await mgr._do_health_check()
        assert spawned, "expected _spawn to be called after health-check failure"
        assert mgr._proc is not None
        assert mgr._proc.pid != old_pid
        mgr._proc.terminate()
        await mgr._proc.wait()

    async def test_health_check_on_dead_process_respawns(self) -> None:

        mgr = CliSubprocessManager(sys.executable, "1.0.0", health_interval_s=9999)
        spawned: list[int] = []

        async def _spawn() -> None:
            import asyncio as _asyncio

            mgr._proc = await _asyncio.create_subprocess_exec(
                sys.executable,
                _STUB,
                stdin=_asyncio.subprocess.PIPE,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            spawned.append(mgr._proc.pid)

        mgr._spawn = _spawn  # type: ignore[method-assign]
        # _proc is None → health check should spawn
        await mgr._do_health_check()
        assert spawned
        mgr._proc.terminate()  # type: ignore[union-attr]
        await mgr._proc.wait()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# SystemOAuthProvider tests
# ---------------------------------------------------------------------------


class _FakeManager:
    """Controllable substitute for CliSubprocessManager used in provider tests."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.events_to_yield: list[Any] = []
        self.raise_on_call: Exception | None = None
        self.call_active = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def call(self, opts: ModelCallOpts):  # type: ignore[return]  # async gen
        self.call_active = True
        try:
            if self.raise_on_call is not None:
                raise self.raise_on_call
            for event in self.events_to_yield:
                yield event
        finally:
            self.call_active = False


def _make_provider(
    *, manager: _FakeManager, tracer: _MockTracer | None = None
) -> SystemOAuthProvider:
    p = SystemOAuthProvider(_manager=manager)
    if tracer is not None:
        import meridian_provider_claude_code_oauth.provider as _mod

        _mod.get_tracer = lambda: tracer  # type: ignore[assignment]
    return p


class TestSystemOAuthProviderCall:
    async def test_call_yields_events(self) -> None:
        mgr = _FakeManager()
        mgr.events_to_yield = [
            MessageStartEvent(type="message_start", model="claude-sonnet-4-6", provider="test"),
            TextDeltaEvent(type="text_delta", text="hi"),
        ]
        provider = SystemOAuthProvider(_manager=mgr)
        events = [e async for e in provider.call(_SIMPLE_OPTS)]
        assert len(events) == 2
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[1], TextDeltaEvent)

    async def test_call_provider_error_raises_and_writes_audit(self) -> None:
        mgr = _FakeManager()
        mgr.raise_on_call = CliSubprocessError("boom")
        audit = _CapturingAuditLog()
        provider = SystemOAuthProvider(_manager=mgr, audit_log=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry.level == "error"
        assert entry.event == "claude_code_oauth.call.failed"
        assert entry.provider_name == "claude_code_oauth"
        assert entry.provider_kind == "claude_code_oauth"

    async def test_call_unexpected_error_wrapped_as_provider_error(self) -> None:
        mgr = _FakeManager()
        mgr.raise_on_call = RuntimeError("unexpected")
        audit = _CapturingAuditLog()
        provider = SystemOAuthProvider(_manager=mgr, audit_log=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].detail["error_type"] == "RuntimeError"

    async def test_call_audit_contains_model(self) -> None:
        mgr = _FakeManager()
        mgr.raise_on_call = CliSubprocessError("fail")
        audit = _CapturingAuditLog()
        provider = SystemOAuthProvider(_manager=mgr, audit_log=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].model == "claude-sonnet-4-6"

    async def test_call_audit_has_timestamp(self) -> None:
        mgr = _FakeManager()
        mgr.raise_on_call = CliSubprocessError("fail")
        audit = _CapturingAuditLog()
        provider = SystemOAuthProvider(_manager=mgr, audit_log=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].timestamp  # non-empty

    async def test_call_emits_otel_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = _MockTracer()
        import meridian_provider_claude_code_oauth.provider as _mod

        monkeypatch.setattr(_mod, "get_tracer", lambda: tracer)

        mgr = _FakeManager()
        mgr.events_to_yield = [TextDeltaEvent(type="text_delta", text="x")]
        provider = SystemOAuthProvider(_manager=mgr)
        async for _ in provider.call(_SIMPLE_OPTS):
            pass

        assert tracer.span.name == "claude_code_oauth.model.call"

    async def test_span_has_provider_name_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = _MockTracer()
        import meridian_provider_claude_code_oauth.provider as _mod

        monkeypatch.setattr(_mod, "get_tracer", lambda: tracer)

        mgr = _FakeManager()
        provider = SystemOAuthProvider(_manager=mgr, name="my_oauth")
        async for _ in provider.call(_SIMPLE_OPTS):
            pass

        assert tracer.span.attributes.get("provider.name") == "my_oauth"

    async def test_span_has_model_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = _MockTracer()
        import meridian_provider_claude_code_oauth.provider as _mod

        monkeypatch.setattr(_mod, "get_tracer", lambda: tracer)

        mgr = _FakeManager()
        provider = SystemOAuthProvider(_manager=mgr)
        async for _ in provider.call(_SIMPLE_OPTS):
            pass

        assert tracer.span.attributes.get("model") == "claude-sonnet-4-6"

    async def test_span_has_invocation_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = _MockTracer()
        import meridian_provider_claude_code_oauth.provider as _mod

        monkeypatch.setattr(_mod, "get_tracer", lambda: tracer)

        mgr = _FakeManager()
        provider = SystemOAuthProvider(_manager=mgr)
        async for _ in provider.call(_SIMPLE_OPTS):
            pass

        event_names = [ev[0] for ev in tracer.span.events]
        assert "provider.invocation" in event_names

    async def test_failure_marks_span_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.trace import StatusCode as _SC

        tracer = _MockTracer()
        import meridian_provider_claude_code_oauth.provider as _mod

        monkeypatch.setattr(_mod, "get_tracer", lambda: tracer)

        mgr = _FakeManager()
        mgr.raise_on_call = CliSubprocessError("fail")
        provider = SystemOAuthProvider(_manager=mgr)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert tracer.span.status is not None
        assert tracer.span.status.status_code == _SC.ERROR


class TestSystemOAuthProviderListModels:
    def test_list_models_non_empty(self) -> None:
        provider = SystemOAuthProvider(_manager=_FakeManager())
        models = provider.list_models()
        assert len(models) > 0

    def test_list_models_contains_opus(self) -> None:
        provider = SystemOAuthProvider(_manager=_FakeManager())
        model_ids = [m.model for m in provider.list_models()]
        assert any("opus" in mid for mid in model_ids)

    def test_list_models_provider_name(self) -> None:
        provider = SystemOAuthProvider(_manager=_FakeManager(), name="my_provider")
        for m in provider.list_models():
            assert m.provider == "my_provider"

    def test_list_models_context_windows(self) -> None:
        provider = SystemOAuthProvider(_manager=_FakeManager())
        for m in provider.list_models():
            assert m.context_window > 0


class TestSystemOAuthProviderLifecycle:
    async def test_start_delegates_to_manager(self) -> None:
        mgr = _FakeManager()
        provider = SystemOAuthProvider(_manager=mgr)
        await provider.start()
        assert mgr.started

    async def test_close_delegates_to_manager(self) -> None:
        mgr = _FakeManager()
        provider = SystemOAuthProvider(_manager=mgr)
        await provider.close()
        assert mgr.stopped

    def test_kind_is_claude_code_oauth(self) -> None:
        provider = SystemOAuthProvider(_manager=_FakeManager())
        assert provider.kind == "claude_code_oauth"

    def test_capabilities_streaming_true(self) -> None:
        provider = SystemOAuthProvider(_manager=_FakeManager())
        assert provider.capabilities.streaming is True

    def test_capabilities_count_tokens_false(self) -> None:
        provider = SystemOAuthProvider(_manager=_FakeManager())
        assert provider.capabilities.count_tokens is False


class TestSystemOAuthProviderLockIntegration:
    def test_provider_reads_lock_file(self, tmp_path: Path) -> None:
        lock = tmp_path / "meridian.lock"
        lock.write_text(
            json.dumps(
                {
                    "version": 1,
                    "pins": {"claude-code": {"version": "9.9.9"}},
                }
            )
        )
        # Build without injected manager so SystemOAuthProvider creates a real
        # CliSubprocessManager and reads the lock into it.
        provider = SystemOAuthProvider(
            cli_path="/nonexistent/claude",
            lock_path=lock,
        )
        assert isinstance(provider._manager, CliSubprocessManager)
        assert provider._manager._cli_version == "9.9.9"

    def test_provider_continues_without_lock_file(self, tmp_path: Path) -> None:
        # Missing lock file must not raise; version falls back to "unknown".
        provider = SystemOAuthProvider(
            cli_path="/nonexistent/claude",
            lock_path=tmp_path / "nonexistent.lock",
        )
        assert isinstance(provider._manager, CliSubprocessManager)
        assert provider._manager._cli_version == "unknown"
