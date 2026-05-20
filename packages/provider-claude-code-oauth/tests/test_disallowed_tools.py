"""
Tests for Contract 1 — disallowed_tools (Architecture §13.4).

Coverage:
  ALL_CLAUDE_CODE_BUILTIN_TOOLS constant:
  - contains Read, Write, Bash, Edit
  - does not contain meridian_tool_proxy
  - is a frozenset

  _opts_to_dict (call payload):
  - disallowed_tools field always present
  - disallowed_tools contains all four built-ins
  - disallowed_tools is a sorted list

  DisallowedToolError:
  - is a subclass of CliSubprocessError
  - is a subclass of ProviderCallError

  CliSubprocessManager._read_events (Contract 1 violation detection):
  - raises DisallowedToolError when tool_use_start names a disallowed tool
  - error message includes the offending tool name
  - non-disallowed tool_use_start passes through normally

  SystemOAuthProvider.call() (audit + span on violation):
  - DisallowedToolError writes audit entry with event "claude_code_oauth.disallowed_tool.attempted"
  - audit entry level is "error"
  - audit entry carries correct provider_name, provider_kind, model
  - audit entry detail contains error_type "DisallowedToolError"
  - span is marked ERROR on DisallowedToolError
  - DisallowedToolError is re-raised as ProviderCallError

  Package public API:
  - ALL_CLAUDE_CODE_BUILTIN_TOOLS exported from package __init__
  - DisallowedToolError exported from package __init__
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from meridian_provider_claude_code_oauth._disallowed_tools import ALL_CLAUDE_CODE_BUILTIN_TOOLS
from meridian_provider_claude_code_oauth._subprocess import (
    CliSubprocessError,
    CliSubprocessManager,
    DisallowedToolError,
    _opts_to_dict,
)
from meridian_provider_claude_code_oauth.provider import SystemOAuthProvider
from meridian_sdk_provider.audit import AuditLogEntry
from meridian_sdk_provider.errors import ProviderCallError
from meridian_sdk_provider.types import Message, ModelCallOpts

_STUB = str(Path(__file__).parent / "_cli_stub.py")

_SIMPLE_OPTS = ModelCallOpts(
    model="claude-sonnet-4-6",
    messages=[Message(role="user", content="hi")],
)


# ---------------------------------------------------------------------------
# OTel + audit helpers (shared with test_provider.py pattern)
# ---------------------------------------------------------------------------


class _MockSpan:
    def __init__(self) -> None:
        self.name: str = ""
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: Any = None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: BaseException, **_: Any) -> None:
        pass

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


class _CapturingAuditLog:
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


class _FakeManager:
    def __init__(self) -> None:
        self.raise_on_call: Exception | None = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def call(self, opts: ModelCallOpts):  # type: ignore[return]
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return
        yield  # make it an async generator


# ---------------------------------------------------------------------------
# ALL_CLAUDE_CODE_BUILTIN_TOOLS constant
# ---------------------------------------------------------------------------


class TestAllClaudeCodeBuiltinTools:
    def test_contains_read(self) -> None:
        assert "Read" in ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_contains_write(self) -> None:
        assert "Write" in ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_contains_bash(self) -> None:
        assert "Bash" in ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_contains_edit(self) -> None:
        assert "Edit" in ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_does_not_contain_meridian_proxy(self) -> None:
        assert "meridian_tool_proxy" not in ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_is_frozenset(self) -> None:
        assert isinstance(ALL_CLAUDE_CODE_BUILTIN_TOOLS, frozenset)


# ---------------------------------------------------------------------------
# _opts_to_dict — disallowed_tools in call payload
# ---------------------------------------------------------------------------


class TestOptsToDictDisallowedTools:
    def test_disallowed_tools_present(self) -> None:
        d = _opts_to_dict(_SIMPLE_OPTS)
        assert "disallowed_tools" in d

    def test_disallowed_tools_contains_all_builtins(self) -> None:
        d = _opts_to_dict(_SIMPLE_OPTS)
        assert set(d["disallowed_tools"]) == set(ALL_CLAUDE_CODE_BUILTIN_TOOLS)

    def test_disallowed_tools_is_sorted_list(self) -> None:
        d = _opts_to_dict(_SIMPLE_OPTS)
        tools = d["disallowed_tools"]
        assert isinstance(tools, list)
        assert tools == sorted(tools)

    def test_disallowed_tools_not_affected_by_opts_tools(self) -> None:
        from meridian_sdk_provider.types import ToolDefinition

        opts = ModelCallOpts(
            model="claude-sonnet-4-6",
            messages=[Message(role="user", content="hi")],
            tools=[ToolDefinition(name="my_tool", description="x", input_schema={})],
        )
        d = _opts_to_dict(opts)
        assert set(d["disallowed_tools"]) == set(ALL_CLAUDE_CODE_BUILTIN_TOOLS)


# ---------------------------------------------------------------------------
# DisallowedToolError class hierarchy
# ---------------------------------------------------------------------------


class TestDisallowedToolError:
    def test_is_subclass_of_cli_subprocess_error(self) -> None:
        assert issubclass(DisallowedToolError, CliSubprocessError)

    def test_is_subclass_of_provider_call_error(self) -> None:
        assert issubclass(DisallowedToolError, ProviderCallError)

    def test_message_preserved(self) -> None:
        err = DisallowedToolError("test message")
        assert "test message" in str(err)

    def test_default_provider_name(self) -> None:
        err = DisallowedToolError("msg")
        assert err.provider_name == "claude_code_oauth"

    def test_custom_provider_name(self) -> None:
        err = DisallowedToolError("msg", provider_name="my_provider")
        assert err.provider_name == "my_provider"


# ---------------------------------------------------------------------------
# CliSubprocessManager — Contract 1 violation detection via _cli_stub.py
# ---------------------------------------------------------------------------


def _make_manager_with_stub(*, env: dict | None = None) -> CliSubprocessManager:
    import asyncio
    import os

    mgr = CliSubprocessManager(
        sys.executable,
        "1.0.0",
        health_interval_s=9999,
        call_timeout_s=5.0,
    )
    _env = {**os.environ, **(env or {})}

    async def _stub_spawn() -> None:
        mgr._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            _STUB,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_env,
        )

    mgr._spawn = _stub_spawn  # type: ignore[method-assign]
    return mgr


class TestCliSubprocessManagerDisallowedToolDetection:
    async def test_disallowed_read_raises_error(self) -> None:
        mgr = _make_manager_with_stub(env={"CLI_STUB_DISALLOWED_TOOL": "Read"})
        with pytest.raises(DisallowedToolError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

    async def test_disallowed_write_raises_error(self) -> None:
        mgr = _make_manager_with_stub(env={"CLI_STUB_DISALLOWED_TOOL": "Write"})
        with pytest.raises(DisallowedToolError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

    async def test_disallowed_bash_raises_error(self) -> None:
        mgr = _make_manager_with_stub(env={"CLI_STUB_DISALLOWED_TOOL": "Bash"})
        with pytest.raises(DisallowedToolError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

    async def test_disallowed_edit_raises_error(self) -> None:
        mgr = _make_manager_with_stub(env={"CLI_STUB_DISALLOWED_TOOL": "Edit"})
        with pytest.raises(DisallowedToolError):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

    async def test_error_message_contains_tool_name(self) -> None:
        mgr = _make_manager_with_stub(env={"CLI_STUB_DISALLOWED_TOOL": "Read"})
        with pytest.raises(DisallowedToolError, match="Read"):
            async for _ in mgr.call(_SIMPLE_OPTS):
                pass

    async def test_non_disallowed_tool_passes_through(self) -> None:
        from meridian_sdk_provider.types import ToolUseStartEvent

        mgr = _make_manager_with_stub(env={"CLI_STUB_DISALLOWED_TOOL": "meridian_tool_proxy"})
        events = [e async for e in mgr.call(_SIMPLE_OPTS)]
        tool_events = [e for e in events if isinstance(e, ToolUseStartEvent)]
        assert any(e.name == "meridian_tool_proxy" for e in tool_events)


# ---------------------------------------------------------------------------
# SystemOAuthProvider — audit + span on DisallowedToolError
# ---------------------------------------------------------------------------


class TestSystemOAuthProviderDisallowedTool:
    def _make_provider(
        self,
        *,
        audit: _CapturingAuditLog | None = None,
        tracer: _MockTracer | None = None,
        tool_name: str = "Read",
    ) -> tuple[SystemOAuthProvider, _FakeManager]:
        mgr = _FakeManager()
        mgr.raise_on_call = DisallowedToolError(
            f"inner loop attempted disallowed built-in tool {tool_name!r}; "
            "capability boundary violation"
        )
        provider = SystemOAuthProvider(_manager=mgr, audit_log=audit)
        if tracer is not None:
            import meridian_provider_claude_code_oauth.provider as _mod
            _mod.get_tracer = lambda: tracer  # type: ignore[assignment]
        return provider, mgr

    async def test_writes_specific_audit_event(self) -> None:
        audit = _CapturingAuditLog()
        provider, _ = self._make_provider(audit=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert len(audit.entries) == 1
        assert audit.entries[0].event == "claude_code_oauth.disallowed_tool.attempted"

    async def test_audit_level_is_error(self) -> None:
        audit = _CapturingAuditLog()
        provider, _ = self._make_provider(audit=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].level == "error"

    async def test_audit_carries_provider_name(self) -> None:
        audit = _CapturingAuditLog()
        provider = SystemOAuthProvider(
            _manager=_FakeManager(),
            audit_log=audit,
            name="my_oauth",
        )
        provider._manager.raise_on_call = DisallowedToolError("Read")  # type: ignore[attr-defined]

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].provider_name == "my_oauth"

    async def test_audit_carries_provider_kind(self) -> None:
        audit = _CapturingAuditLog()
        provider, _ = self._make_provider(audit=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].provider_kind == "claude_code_oauth"

    async def test_audit_carries_model(self) -> None:
        audit = _CapturingAuditLog()
        provider, _ = self._make_provider(audit=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].model == "claude-sonnet-4-6"

    async def test_audit_detail_contains_error_type(self) -> None:
        audit = _CapturingAuditLog()
        provider, _ = self._make_provider(audit=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert audit.entries[0].detail["error_type"] == "DisallowedToolError"

    async def test_span_marked_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tracer = _MockTracer()
        import meridian_provider_claude_code_oauth.provider as _mod
        monkeypatch.setattr(_mod, "get_tracer", lambda: tracer)

        mgr = _FakeManager()
        mgr.raise_on_call = DisallowedToolError("Read")
        provider = SystemOAuthProvider(_manager=mgr)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        assert tracer.span.status is not None
        assert tracer.span.status.status_code == StatusCode.ERROR

    async def test_disallowed_tool_reraises_as_provider_call_error(self) -> None:
        audit = _CapturingAuditLog()
        provider, _ = self._make_provider(audit=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

    async def test_generic_call_failed_event_not_written_for_disallowed_tool(self) -> None:
        audit = _CapturingAuditLog()
        provider, _ = self._make_provider(audit=audit)

        with pytest.raises(ProviderCallError):
            async for _ in provider.call(_SIMPLE_OPTS):
                pass

        events = [e.event for e in audit.entries]
        assert "claude_code_oauth.call.failed" not in events


# ---------------------------------------------------------------------------
# Package public API exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_all_claude_code_builtin_tools_exported(self) -> None:
        import meridian_provider_claude_code_oauth as pkg
        assert hasattr(pkg, "ALL_CLAUDE_CODE_BUILTIN_TOOLS")
        assert pkg.ALL_CLAUDE_CODE_BUILTIN_TOOLS is ALL_CLAUDE_CODE_BUILTIN_TOOLS

    def test_disallowed_tool_error_exported(self) -> None:
        import meridian_provider_claude_code_oauth as pkg
        assert hasattr(pkg, "DisallowedToolError")
        assert pkg.DisallowedToolError is DisallowedToolError
