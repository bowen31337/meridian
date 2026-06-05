"""
Hook dispatch via Sandbox conformance suite.

Tests cover:
  - dispatch_hooks: empty hooks_dir → returns empty list.
  - dispatch_hooks: hooks_dir does not exist → returns empty list.
  - dispatch_hooks: no hooks match event → returns empty list.
  - dispatch_hooks: inactive hook → skipped.
  - dispatch_hooks: matching hook succeeds → HookDispatchResult.is_error is False.
  - dispatch_hooks: result list has one entry per dispatched hook.
  - dispatch_hooks: multiple hooks dispatched in order.
  - dispatch_hooks: failure_mode=ignore, hook fails → no raise.
  - dispatch_hooks: failure_mode=ignore, hook fails → no audit entry written.
  - dispatch_hooks: failure_mode=warn, hook fails → no raise.
  - dispatch_hooks: failure_mode=warn, hook fails → audit entry written.
  - dispatch_hooks: failure_mode=warn, audit entry level is "warn".
  - dispatch_hooks: failure_mode=warn, audit entry event is "hook.dispatch.failed".
  - dispatch_hooks: failure_mode=warn, audit entry detail has hook_id.
  - dispatch_hooks: failure_mode=warn, audit entry detail has message.
  - dispatch_hooks: failure_mode=block, hook fails → raises HookDispatchBlockedError.
  - dispatch_hooks: failure_mode=block → HookDispatchBlockedError.code is "hook_dispatch_blocked".
  - dispatch_hooks: failure_mode=block → error message surfaced on exception.
  - dispatch_hooks: failure_mode=block → audit entry written before raise.
  - dispatch_hooks: failure_mode=block → audit entry level is "error".
  - dispatch_hooks: failure_mode=abort, hook fails → raises HookDispatchBlockedError.
  - dispatch_hooks: timeout_ms enforced — slow handler returns is_error with timeout error.
  - dispatch_hooks: timeout error triggers block-mode raise.
  - dispatch_hooks: match filter session_id — hook with wrong session_id is skipped.
  - dispatch_hooks: match filter session_id — hook with matching session_id is dispatched.
  - dispatch_hooks: match filter null — hook without match is dispatched for any session.
  - dispatch_hooks: hook with wrong event skipped even if status=active.
  - HookDispatchBlockedError.http_status returns 502.
  - _SandboxAuditBridge bridges sandbox audit entries to core_errors AuditLog.
  - OTel span "hook.dispatch" emitted on every call.
  - OTel span "hook.dispatch.single" emitted per dispatched hook.
  - OTel span "hook.dispatch.single" has ERROR status when failure_mode=block.
  - OTel span "hook.dispatch" has hook.event attribute.
  - OTel span "hook.dispatch" has hook.count attribute.
  - dispatch_hooks: sandbox dispatch via same Sandbox.execute() surface — OTel child spans present.
  Verdict types:
  - verdict=continue (no-op): result.verdict is "continue", is_error is False.
  - verdict=continue (no JSON output): treated as continue.
  - verdict=continue with mutations: result.mutations carries the mutations dict.
  - verdict=continue with non-dict mutations ignored: result.mutations is None.
  - verdict=veto: raises HookVetoError.
  - verdict=veto: HookVetoError.code is "hook_veto".
  - verdict=veto: HookVetoError.message carries the reason.
  - verdict=veto: HookVetoError.http_status() is 422.
  - verdict=veto: audit entry written at info level with event "hook.verdict.veto".
  - verdict=veto: audit entry detail contains hook_id.
  - verdict=veto: audit entry detail contains reason.
  - verdict=fail + failure_mode=block: raises HookDispatchBlockedError.
  - verdict=fail + failure_mode=warn: audit entry written, no raise.
  - verdict=fail + failure_mode=ignore: no raise, no audit entry.
  - verdict=fail: error_message carries the reason from the hook.
  - verdict=fail: HookDispatchResult.verdict is "fail".
  - sandbox error (is_error=True): HookDispatchResult.verdict is "fail".
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
import uuid

from core_errors import AuditLog, AuditLogEntry
from meridiand._hook_dispatch import (
    HookDispatchBlockedError,
    HookVetoError,
    _SandboxAuditBridge,
    dispatch_hooks,
)
from opentelemetry.trace import StatusCode
import pytest
from sdk_sandbox import ExecutionContext

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


def _context(session_id: str = "sess-test") -> ExecutionContext:
    return ExecutionContext(session_id=session_id)


def _write_hook(
    hooks_dir: Path,
    *,
    event: str = "tool_call.requested",
    name: str = "test-hook",
    handler: str = "in_process",
    timeout_ms: int = 5000,
    failure_mode: str = "ignore",
    status: str = "active",
    match: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_id = f"hook_{uuid.uuid4().hex}"
    resource = {
        "id": hook_id,
        "event": event,
        "name": name,
        "handler": handler,
        "match": match,
        "timeout_ms": timeout_ms,
        "failure_mode": failure_mode,
        "secret_reads": None,
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": metadata,
    }
    (hooks_dir / f"{hook_id}.json").write_text(json.dumps(resource))
    return resource


async def _ok_handler(input: dict, context: ExecutionContext) -> Any:
    return "ok"


async def _fail_handler(input: dict, context: ExecutionContext) -> Any:
    raise RuntimeError("deliberate failure")


async def _slow_handler(input: dict, context: ExecutionContext) -> Any:
    await asyncio.sleep(30)
    return "never"


async def _continue_handler(input: dict, context: ExecutionContext) -> Any:
    return {"verdict": "continue"}


async def _mutate_handler(input: dict, context: ExecutionContext) -> Any:
    return {"verdict": "continue", "mutations": {"args": {"key": "new_value"}}}


async def _bad_mutations_handler(input: dict, context: ExecutionContext) -> Any:
    return {"verdict": "continue", "mutations": "not-a-dict"}


async def _veto_handler(input: dict, context: ExecutionContext) -> Any:
    return {"verdict": "veto", "reason": "operation not permitted by policy"}


async def _fail_verdict_handler(input: dict, context: ExecutionContext) -> Any:
    return {"verdict": "fail", "reason": "hook encountered an internal error"}


# ---------------------------------------------------------------------------
# No matching hooks
# ---------------------------------------------------------------------------


class TestNoHooks:
    async def test_missing_hooks_dir_returns_empty(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
        )
        assert results == []

    async def test_empty_hooks_dir_returns_empty(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
        )
        assert results == []

    async def test_wrong_event_returns_empty(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, event="session.created")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert results == []

    async def test_inactive_hook_skipped(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, status="inactive")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert results == []


# ---------------------------------------------------------------------------
# Success dispatch
# ---------------------------------------------------------------------------


class TestSuccessDispatch:
    async def test_success_returns_result(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert len(results) == 1
        assert results[0].is_error is False

    async def test_result_has_hook_id(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert results[0].hook_id == hook["id"]

    async def test_result_has_hook_name(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, name="my-hook")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert results[0].hook_name == "my-hook"

    async def test_multiple_hooks_all_dispatched(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook1 = _write_hook(hooks_dir, name="hook-a")
        hook2 = _write_hook(hooks_dir, name="hook-b")
        handlers = {hook1["id"]: _ok_handler, hook2["id"]: _ok_handler}
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers=handlers,
        )
        assert len(results) == 2
        assert all(r.is_error is False for r in results)


# ---------------------------------------------------------------------------
# failure_mode=ignore
# ---------------------------------------------------------------------------


class TestFailureModeIgnore:
    async def test_ignore_hook_failure_no_raise(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="ignore")
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _fail_handler},
        )

    async def test_ignore_hook_failure_no_audit_entry(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="ignore")
        audit = _CapturingAuditLog()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=audit,
            in_process_handlers={hook["id"]: _fail_handler},
        )
        hook_dispatch_entries = [e for e in audit.entries if e.event == "hook.dispatch.failed"]
        assert hook_dispatch_entries == []

    async def test_ignore_returns_error_result(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="ignore")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _fail_handler},
        )
        assert results[0].is_error is True


# ---------------------------------------------------------------------------
# failure_mode=warn
# ---------------------------------------------------------------------------


class TestFailureModeWarn:
    async def test_warn_hook_failure_no_raise(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="warn")
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _fail_handler},
        )

    async def test_warn_writes_audit_entry(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="warn")
        audit = _CapturingAuditLog()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=audit,
            in_process_handlers={hook["id"]: _fail_handler},
        )
        assert any(e.event == "hook.dispatch.failed" for e in audit.entries)

    async def test_warn_audit_level_is_warn(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="warn")
        audit = _CapturingAuditLog()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=audit,
            in_process_handlers={hook["id"]: _fail_handler},
        )
        entry = next(e for e in audit.entries if e.event == "hook.dispatch.failed")
        assert entry.level == "warn"

    async def test_warn_audit_detail_has_hook_id(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="warn")
        audit = _CapturingAuditLog()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=audit,
            in_process_handlers={hook["id"]: _fail_handler},
        )
        entry = next(e for e in audit.entries if e.event == "hook.dispatch.failed")
        assert entry.detail is not None
        assert entry.detail["hook_id"] == hook["id"]

    async def test_warn_audit_detail_has_message(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="warn")
        audit = _CapturingAuditLog()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=audit,
            in_process_handlers={hook["id"]: _fail_handler},
        )
        entry = next(e for e in audit.entries if e.event == "hook.dispatch.failed")
        assert entry.detail is not None
        assert len(entry.detail["message"]) > 0


# ---------------------------------------------------------------------------
# failure_mode=block
# ---------------------------------------------------------------------------


class TestFailureModeBlock:
    async def test_block_raises_on_hook_failure(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_handler},
            )

    async def test_block_error_code_is_hook_dispatch_blocked(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError) as exc_info:
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_handler},
            )
        assert exc_info.value.code == "hook_dispatch_blocked"

    async def test_block_error_message_is_non_empty(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError) as exc_info:
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_handler},
            )
        assert len(exc_info.value.message) > 0

    async def test_block_error_has_hook_id(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError) as exc_info:
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_handler},
            )
        assert exc_info.value.hook_id == hook["id"]

    async def test_block_audit_written_before_raise(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        audit = _CapturingAuditLog()
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=audit,
                in_process_handlers={hook["id"]: _fail_handler},
            )
        assert any(e.event == "hook.dispatch.failed" for e in audit.entries)

    async def test_block_audit_level_is_error(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        audit = _CapturingAuditLog()
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=audit,
                in_process_handlers={hook["id"]: _fail_handler},
            )
        entry = next(e for e in audit.entries if e.event == "hook.dispatch.failed")
        assert entry.level == "error"

    async def test_block_http_status_is_502(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError) as exc_info:
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_handler},
            )
        assert exc_info.value.http_status() == 502


# ---------------------------------------------------------------------------
# failure_mode=abort (same raising semantics as block)
# ---------------------------------------------------------------------------


class TestFailureModeAbort:
    async def test_abort_raises_on_hook_failure(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="abort")
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_handler},
            )

    async def test_abort_audit_written(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="abort")
        audit = _CapturingAuditLog()
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=audit,
                in_process_handlers={hook["id"]: _fail_handler},
            )
        assert any(e.event == "hook.dispatch.failed" for e in audit.entries)


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


class TestTimeoutEnforcement:
    async def test_slow_handler_returns_error(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, timeout_ms=100, failure_mode="ignore")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _slow_handler},
        )
        assert results[0].is_error is True

    async def test_slow_handler_error_code_is_timeout(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, timeout_ms=100, failure_mode="ignore")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _slow_handler},
        )
        assert results[0].error_code == "timeout"

    async def test_timeout_with_block_mode_raises(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, timeout_ms=100, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _slow_handler},
            )


# ---------------------------------------------------------------------------
# Match filter
# ---------------------------------------------------------------------------


class TestMatchFilter:
    async def test_hook_with_matching_session_id_dispatched(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(
            hooks_dir,
            match={"session_id": "sess-abc", "agent_id": None},
        )
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(session_id="sess-abc"),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert len(results) == 1

    async def test_hook_with_wrong_session_id_skipped(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(
            hooks_dir,
            match={"session_id": "sess-xyz", "agent_id": None},
        )
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(session_id="sess-abc"),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert results == []

    async def test_hook_without_match_dispatched_for_any_session(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, match=None)
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(session_id="any-session"),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# SandboxAuditBridge
# ---------------------------------------------------------------------------


class TestSandboxAuditBridge:
    def test_bridge_writes_to_core_audit_log(self) -> None:
        import sdk_sandbox as _sb

        audit = _CapturingAuditLog()
        bridge = _SandboxAuditBridge(audit)
        bridge.write(
            _sb.AuditLogEntry(
                level="error",
                event="sandbox.tool.timeout",
                tool_name="hook_123",
                session_id="sess-test",
                timestamp="2026-01-01T00:00:00+00:00",
                detail={"code": "timeout", "message": "timed out", "timeout_ms": 100},
            )
        )
        assert len(audit.entries) == 1

    def test_bridge_extracts_code_from_detail(self) -> None:
        import sdk_sandbox as _sb

        audit = _CapturingAuditLog()
        bridge = _SandboxAuditBridge(audit)
        bridge.write(
            _sb.AuditLogEntry(
                level="error",
                event="sandbox.tool.timeout",
                tool_name="hook_123",
                session_id="sess-test",
                timestamp="2026-01-01T00:00:00+00:00",
                detail={"code": "timeout", "message": "timed out"},
            )
        )
        assert audit.entries[0].code == "timeout"

    def test_bridge_preserves_event(self) -> None:
        import sdk_sandbox as _sb

        audit = _CapturingAuditLog()
        bridge = _SandboxAuditBridge(audit)
        bridge.write(
            _sb.AuditLogEntry(
                level="warn",
                event="dispatch.overhead.target_breached",
                tool_name="hook_abc",
                session_id="sess-x",
                timestamp="2026-01-01T00:00:00+00:00",
                detail={"code": "overhead", "message": "slow"},
            )
        )
        assert audit.entries[0].event == "dispatch.overhead.target_breached"

    def test_bridge_uses_sandbox_error_fallback_code(self) -> None:
        import sdk_sandbox as _sb

        audit = _CapturingAuditLog()
        bridge = _SandboxAuditBridge(audit)
        bridge.write(
            _sb.AuditLogEntry(
                level="error",
                event="sandbox.execute.failed",
                tool_name="hook_x",
                session_id="sess-y",
                timestamp="2026-01-01T00:00:00+00:00",
                detail=None,
            )
        )
        assert audit.entries[0].code == "sandbox_error"


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_hook_dispatch_span_emitted(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "hook.dispatch" in span_names

    async def test_hook_dispatch_span_emitted_even_with_no_hooks(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "hook.dispatch" in span_names

    async def test_hook_dispatch_span_has_event_attribute(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        await dispatch_hooks(
            "session.created",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
        )
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "hook.dispatch")
        assert span.attributes["hook.event"] == "session.created"

    async def test_hook_dispatch_span_has_count_attribute(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook1 = _write_hook(hooks_dir, name="h1")
        hook2 = _write_hook(hooks_dir, name="h2")
        handlers = {hook1["id"]: _ok_handler, hook2["id"]: _ok_handler}
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers=handlers,
        )
        span = next(s for s in _otel_exporter.get_finished_spans() if s.name == "hook.dispatch")
        assert span.attributes["hook.count"] == 2

    async def test_hook_dispatch_single_span_emitted_per_hook(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook1 = _write_hook(hooks_dir, name="h1")
        hook2 = _write_hook(hooks_dir, name="h2")
        handlers = {hook1["id"]: _ok_handler, hook2["id"]: _ok_handler}
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers=handlers,
        )
        single_spans = [
            s for s in _otel_exporter.get_finished_spans() if s.name == "hook.dispatch.single"
        ]
        assert len(single_spans) == 2

    async def test_block_failure_span_has_error_status(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_handler},
            )
        single_span = next(
            s for s in _otel_exporter.get_finished_spans() if s.name == "hook.dispatch.single"
        )
        assert single_span.status.status_code == StatusCode.ERROR

    async def test_sandbox_execute_span_present(self, tmp_path: Path) -> None:
        # Verifies that hooks go through Sandbox.execute() — its OTel child
        # span "sandbox.execute" must appear when a hook is dispatched.
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "sandbox.execute" in span_names


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------


class TestVerdictTypes:
    async def test_continue_noop_verdict_is_continue(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _continue_handler},
        )
        assert results[0].verdict == "continue"
        assert results[0].is_error is False

    async def test_continue_noop_no_json_treated_as_continue(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _ok_handler},
        )
        assert results[0].verdict == "continue"
        assert results[0].is_error is False

    async def test_continue_with_mutations_propagated(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _mutate_handler},
        )
        assert results[0].mutations == {"args": {"key": "new_value"}}

    async def test_continue_with_non_dict_mutations_ignored(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _bad_mutations_handler},
        )
        assert results[0].mutations is None

    async def test_veto_raises_hook_veto_error(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        with pytest.raises(HookVetoError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _veto_handler},
            )

    async def test_veto_error_code_is_hook_veto(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        with pytest.raises(HookVetoError) as exc_info:
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _veto_handler},
            )
        assert exc_info.value.code == "hook_veto"

    async def test_veto_error_message_carries_reason(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        with pytest.raises(HookVetoError) as exc_info:
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _veto_handler},
            )
        assert exc_info.value.message == "operation not permitted by policy"

    async def test_veto_http_status_is_422(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        with pytest.raises(HookVetoError) as exc_info:
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _veto_handler},
            )
        assert exc_info.value.http_status() == 422

    async def test_veto_audit_entry_at_info_level(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        audit = _CapturingAuditLog()
        with pytest.raises(HookVetoError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=audit,
                in_process_handlers={hook["id"]: _veto_handler},
            )
        entry = next(e for e in audit.entries if e.event == "hook.verdict.veto")
        assert entry.level == "info"

    async def test_veto_audit_entry_detail_has_hook_id(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        audit = _CapturingAuditLog()
        with pytest.raises(HookVetoError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=audit,
                in_process_handlers={hook["id"]: _veto_handler},
            )
        entry = next(e for e in audit.entries if e.event == "hook.verdict.veto")
        assert entry.detail is not None
        assert entry.detail["hook_id"] == hook["id"]

    async def test_veto_audit_entry_detail_has_reason(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir)
        audit = _CapturingAuditLog()
        with pytest.raises(HookVetoError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=audit,
                in_process_handlers={hook["id"]: _veto_handler},
            )
        entry = next(e for e in audit.entries if e.event == "hook.verdict.veto")
        assert entry.detail is not None
        assert entry.detail["reason"] == "operation not permitted by policy"

    async def test_fail_verdict_block_raises(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="block")
        with pytest.raises(HookDispatchBlockedError):
            await dispatch_hooks(
                "tool_call.requested",
                {},
                _context(),
                hooks_dir=hooks_dir,
                audit_log=_CapturingAuditLog(),
                in_process_handlers={hook["id"]: _fail_verdict_handler},
            )

    async def test_fail_verdict_warn_audit_no_raise(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="warn")
        audit = _CapturingAuditLog()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=audit,
            in_process_handlers={hook["id"]: _fail_verdict_handler},
        )
        assert any(e.event == "hook.dispatch.failed" for e in audit.entries)

    async def test_fail_verdict_ignore_no_raise_no_audit(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="ignore")
        audit = _CapturingAuditLog()
        await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=audit,
            in_process_handlers={hook["id"]: _fail_verdict_handler},
        )
        hook_dispatch_entries = [e for e in audit.entries if e.event == "hook.dispatch.failed"]
        assert hook_dispatch_entries == []

    async def test_fail_verdict_error_message_carries_reason(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="ignore")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _fail_verdict_handler},
        )
        assert results[0].error_message == "hook encountered an internal error"

    async def test_fail_verdict_result_verdict_is_fail(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="ignore")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _fail_verdict_handler},
        )
        assert results[0].verdict == "fail"

    async def test_sandbox_error_result_verdict_is_fail(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / "hooks"
        hook = _write_hook(hooks_dir, failure_mode="ignore")
        results = await dispatch_hooks(
            "tool_call.requested",
            {},
            _context(),
            hooks_dir=hooks_dir,
            audit_log=_CapturingAuditLog(),
            in_process_handlers={hook["id"]: _fail_handler},
        )
        assert results[0].verdict == "fail"
