"""
Tests for dispatch overhead target tracking.

Verifies that InProcessDispatcher, SubprocessDispatcher, and ContainerDispatcher:
  - Emit a "dispatch.overhead" event on every successful dispatch with the correct
    attributes (kind, overhead_ms, target_ms, target_breached).
  - Do NOT write an audit entry when overhead is within the target.
  - Write a "warn"-level audit entry when overhead exceeds the target.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from sdk_sandbox import (
    ContainerHandler,
    ExecutionContext,
    InProcessHandler,
    SubprocessHandler,
    ToolDefinition,
)
from sdk_sandbox._dispatchers import (
    ContainerDispatcher,
    InProcessDispatcher,
    SubprocessDispatcher,
)

from .conftest import CapturingAuditLog, MockSpan, MockTracer

# ---------------------------------------------------------------------------
# OTel mock — targets sdk_sandbox._dispatchers.get_tracer
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("sdk_sandbox._dispatchers.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CTX = ExecutionContext(
    session_id="sess-overhead", workspace="/workspace", scratch_dir="/tmp/scratch"
)

_ECHO_SCRIPT = """\
import sys, json
req = json.load(sys.stdin)
json.dump({"result": req["args"]}, sys.stdout)
"""

_SLOW_SCRIPT = """\
import sys, json, time
time.sleep(0.35)
req = json.load(sys.stdin)
json.dump({"result": req["args"]}, sys.stdout)
"""


def _tool(handler: Any, name: str = "test.tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Test tool",
        input_schema={"type": "object"},
        handler=handler,
    )


def _overhead_event(span: MockSpan) -> tuple[str, dict[str, Any]] | None:
    for name, attrs in span.events:
        if name == "dispatch.overhead":
            return (name, attrs)
    return None


# ---------------------------------------------------------------------------
# InProcessDispatcher — overhead tracking
# ---------------------------------------------------------------------------


class TestInProcessOverhead:
    async def test_overhead_event_emitted_on_success(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert _overhead_event(mock_span) is not None

    async def test_overhead_event_kind_attribute(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        _, attrs = _overhead_event(mock_span)
        assert attrs["dispatch.kind"] == "in_process"

    async def test_overhead_event_target_ms(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        _, attrs = _overhead_event(mock_span)
        assert attrs["dispatch.overhead.target_ms"] == 20.0

    async def test_overhead_event_overhead_ms_present(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        _, attrs = _overhead_event(mock_span)
        assert isinstance(attrs["dispatch.overhead_ms"], float)
        assert attrs["dispatch.overhead_ms"] >= 0.0

    async def test_overhead_event_breached_is_bool(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        _, attrs = _overhead_event(mock_span)
        assert isinstance(attrs["dispatch.overhead.target_breached"], bool)

    async def test_overhead_within_target_no_audit_breach(self, mock_span: MockSpan) -> None:
        audit = CapturingAuditLog()
        d = InProcessDispatcher(audit_log=audit)
        d.register("test.tool", AsyncMock(return_value="ok"))
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        breach_entries = [e for e in audit.entries if e.event == "dispatch.overhead.target_breached"]
        assert breach_entries == []

    async def test_overhead_breach_writes_warn_audit(self, mock_span: MockSpan) -> None:
        async def slow_fn(inp: dict, ctx: ExecutionContext) -> str:
            await asyncio.sleep(0.025)  # 25ms > 20ms target
            return "done"

        audit = CapturingAuditLog()
        d = InProcessDispatcher(audit_log=audit)
        d.register("test.tool", slow_fn)
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)

        breach_entries = [e for e in audit.entries if e.event == "dispatch.overhead.target_breached"]
        assert len(breach_entries) == 1
        assert breach_entries[0].level == "warn"

    async def test_overhead_breach_audit_detail(self, mock_span: MockSpan) -> None:
        async def slow_fn(inp: dict, ctx: ExecutionContext) -> str:
            await asyncio.sleep(0.025)
            return "done"

        audit = CapturingAuditLog()
        d = InProcessDispatcher(audit_log=audit)
        d.register("test.tool", slow_fn)
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)

        entry = next(e for e in audit.entries if e.event == "dispatch.overhead.target_breached")
        assert entry.detail is not None
        assert entry.detail["kind"] == "in_process"
        assert entry.detail["target_ms"] == 20.0
        assert "overhead_ms" in entry.detail

    async def test_overhead_breach_event_target_breached_true(self, mock_span: MockSpan) -> None:
        async def slow_fn(inp: dict, ctx: ExecutionContext) -> str:
            await asyncio.sleep(0.025)
            return "done"

        d = InProcessDispatcher()
        d.register("test.tool", slow_fn)
        await d.dispatch(_tool(InProcessHandler()), {}, CTX)

        _, attrs = _overhead_event(mock_span)
        assert attrs["dispatch.overhead.target_breached"] is True

    async def test_no_overhead_event_on_handler_not_found(self, mock_span: MockSpan) -> None:
        d = InProcessDispatcher()
        with pytest.raises(Exception):
            await d.dispatch(_tool(InProcessHandler()), {}, CTX)
        assert _overhead_event(mock_span) is None


# ---------------------------------------------------------------------------
# SubprocessDispatcher — overhead tracking
# ---------------------------------------------------------------------------


def _write_exe(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(f"#!{sys.executable}\n{content}")
    p.chmod(0o755)
    return str(p)


class TestSubprocessOverhead:
    async def test_overhead_event_emitted_on_success(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        d = SubprocessDispatcher()
        await d.dispatch(_tool(SubprocessHandler(path=exe)), {}, CTX)
        assert _overhead_event(mock_span) is not None

    async def test_overhead_event_kind_attribute(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        d = SubprocessDispatcher()
        await d.dispatch(_tool(SubprocessHandler(path=exe)), {}, CTX)
        _, attrs = _overhead_event(mock_span)
        assert attrs["dispatch.kind"] == "subprocess"

    async def test_overhead_event_target_ms(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        d = SubprocessDispatcher()
        await d.dispatch(_tool(SubprocessHandler(path=exe)), {}, CTX)
        _, attrs = _overhead_event(mock_span)
        assert attrs["dispatch.overhead.target_ms"] == 200.0

    async def test_overhead_within_target_no_audit_breach(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path=exe)), {}, CTX)
        breach_entries = [e for e in audit.entries if e.event == "dispatch.overhead.target_breached"]
        assert breach_entries == []

    async def test_overhead_breach_writes_warn_audit(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "slow_tool", _SLOW_SCRIPT)  # sleeps 350ms > 200ms target
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path=exe)), {}, CTX)

        breach_entries = [e for e in audit.entries if e.event == "dispatch.overhead.target_breached"]
        assert len(breach_entries) == 1
        assert breach_entries[0].level == "warn"

    async def test_overhead_breach_audit_detail(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "slow_tool", _SLOW_SCRIPT)
        audit = CapturingAuditLog()
        d = SubprocessDispatcher(audit_log=audit)
        await d.dispatch(_tool(SubprocessHandler(path=exe)), {}, CTX)

        entry = next(e for e in audit.entries if e.event == "dispatch.overhead.target_breached")
        assert entry.detail is not None
        assert entry.detail["kind"] == "subprocess"
        assert entry.detail["target_ms"] == 200.0

    async def test_no_overhead_event_on_binary_not_found(self, mock_span: MockSpan) -> None:
        d = SubprocessDispatcher()
        result = await d.dispatch(_tool(SubprocessHandler(path="/no/such/binary")), {}, CTX)
        assert result.is_error is True
        assert _overhead_event(mock_span) is None


# ---------------------------------------------------------------------------
# ContainerDispatcher — overhead tracking
# ---------------------------------------------------------------------------


class TestContainerOverhead:
    async def test_overhead_event_emitted_on_success(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(exe, **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            d = ContainerDispatcher()
            await d.dispatch(
                _tool(ContainerHandler(environment_id="c1", entrypoint=exe)), {}, CTX
            )

        assert _overhead_event(mock_span) is not None

    async def test_overhead_event_kind_attribute(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(exe, **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await ContainerDispatcher().dispatch(
                _tool(ContainerHandler(environment_id="c1", entrypoint=exe)), {}, CTX
            )

        _, attrs = _overhead_event(mock_span)
        assert attrs["dispatch.kind"] == "container"

    async def test_overhead_event_target_ms(self, tmp_path: Path, mock_span: MockSpan) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(exe, **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await ContainerDispatcher().dispatch(
                _tool(ContainerHandler(environment_id="c1", entrypoint=exe)), {}, CTX
            )

        _, attrs = _overhead_event(mock_span)
        assert attrs["dispatch.overhead.target_ms"] == 500.0

    async def test_overhead_within_target_no_audit_breach(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        exe = _write_exe(tmp_path, "echo_tool", _ECHO_SCRIPT)
        audit = CapturingAuditLog()
        orig = asyncio.create_subprocess_exec

        async def fake_exec(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            return await orig(exe, **kwargs)

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await ContainerDispatcher(audit_log=audit).dispatch(
                _tool(ContainerHandler(environment_id="c1", entrypoint=exe)), {}, CTX
            )

        breach_entries = [e for e in audit.entries if e.event == "dispatch.overhead.target_breached"]
        assert breach_entries == []

    async def test_no_overhead_event_on_docker_not_found(self, mock_span: MockSpan) -> None:
        d = ContainerDispatcher(docker_executable="/no/such/docker")
        result = await d.dispatch(
            _tool(ContainerHandler(environment_id="c1", entrypoint="/ep")), {}, CTX
        )
        assert result.is_error is True
        assert _overhead_event(mock_span) is None
