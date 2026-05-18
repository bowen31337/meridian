"""
Sandbox conformance suite.

Every implementation of Sandbox must satisfy these tests. The suite covers:

  - Successful execute: span emitted, invocation event attached, no audit
    entries, correct result returned.
  - Tool not registered (TOOL_NOT_REGISTERED): SandboxFailure raised, audit
    entry written at level "error", span status set to ERROR.
  - Dispatcher not registered (DISPATCHER_KIND_NOT_REGISTERED): same.
  - Dispatcher raises (TOOL_DISPATCH_FAILED): wrapped in SandboxFailure with
    cause, audit entry written, span marked ERROR, on_error callback called.
  - Duplicate registration guard (tools and dispatchers).
  - on_error callback invocation.
  - Span lifecycle: span ended on both success and failure paths.
"""

from __future__ import annotations

import pytest
from opentelemetry.trace import StatusCode
from sdk_sandbox import (
    AuditLogEntry,
    ExecutionContext,
    InProcessHandler,
    RuntimeOptions,
    Sandbox,
    SandboxFailure,
    SandboxResult,
    ToolDefinition,
    ToolDispatcher,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Stub dispatcher
# ---------------------------------------------------------------------------


class StubDispatcher(ToolDispatcher):
    kind = "in_process"

    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises
        self.calls: list[tuple[ToolDefinition, dict, ExecutionContext]] = []

    async def dispatch(
        self, tool: ToolDefinition, input: dict, context: ExecutionContext
    ) -> SandboxResult:
        if self._raises:
            raise self._raises
        self.calls.append((tool, input, context))
        return SandboxResult(content="ok", duration_ms=1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_DEF = ToolDefinition(
    name="test.echo",
    description="Echo input",
    input_schema={"type": "object"},
    handler=InProcessHandler(),
)

CTX = ExecutionContext(session_id="sess1", workspace="/tmp")


def make_options(
    audit: CapturingAuditLog,
    errors: list[SandboxFailure] | None = None,
) -> RuntimeOptions:
    return RuntimeOptions(
        audit_log=audit,
        on_error=(lambda e: errors.append(e)) if errors is not None else None,
    )


def registered_sandbox() -> Sandbox:
    sb = Sandbox()
    sb.register_dispatcher(StubDispatcher())
    sb.register_tool(TOOL_DEF)
    return sb


# ---------------------------------------------------------------------------
# execute — success
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    async def test_returns_result(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        result = await registered_sandbox().execute("test.echo", {}, CTX, make_options(audit_log))
        assert result.content == "ok"

    async def test_span_name(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_sandbox().execute("test.echo", {}, CTX, make_options(audit_log))
        assert mock_span.name == "sandbox.execute"

    async def test_span_attributes(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_sandbox().execute("test.echo", {}, CTX, make_options(audit_log))
        assert mock_span.attributes["tool.name"] == "test.echo"
        assert mock_span.attributes["session.id"] == "sess1"

    async def test_invocation_event_attached(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_sandbox().execute("test.echo", {}, CTX, make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "sandbox.invocation" in event_names

    async def test_invocation_event_operation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_sandbox().execute("test.echo", {}, CTX, make_options(audit_log))
        inv = next(e for e in mock_span.events if e[0] == "sandbox.invocation")
        assert inv[1]["operation"] == "execute"

    async def test_no_audit_entries_on_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        await registered_sandbox().execute("test.echo", {}, CTX, make_options(audit_log))
        assert audit_log.entries == []

    async def test_span_ended(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        await registered_sandbox().execute("test.echo", {}, CTX, make_options(audit_log))
        assert mock_span.ended

    async def test_dispatches_input_and_context(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        dispatcher = StubDispatcher()
        sb = Sandbox()
        sb.register_dispatcher(dispatcher)
        sb.register_tool(TOOL_DEF)
        payload = {"msg": "hello"}
        await sb.execute("test.echo", payload, CTX, make_options(audit_log))
        assert len(dispatcher.calls) == 1
        _, dispatched_input, dispatched_ctx = dispatcher.calls[0]
        assert dispatched_input == payload
        assert dispatched_ctx is CTX


# ---------------------------------------------------------------------------
# execute — TOOL_NOT_REGISTERED
# ---------------------------------------------------------------------------


class TestExecuteToolNotRegistered:
    async def test_raises_sandbox_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher())
        with pytest.raises(SandboxFailure) as exc_info:
            await sb.execute("acme.unknown", {}, CTX, make_options(audit_log))
        assert exc_info.value.code == "TOOL_NOT_REGISTERED"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher())
        with pytest.raises(SandboxFailure):
            await sb.execute("acme.unknown", {}, CTX, make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "sandbox.execute.failed"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher())
        with pytest.raises(SandboxFailure):
            await sb.execute("acme.unknown", {}, CTX, make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_error_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher())
        with pytest.raises(SandboxFailure):
            await sb.execute("acme.unknown", {}, CTX, make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "sandbox.error" in event_names

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher())
        with pytest.raises(SandboxFailure):
            await sb.execute("acme.unknown", {}, CTX, make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "TOOL_NOT_REGISTERED"

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher())
        with pytest.raises(SandboxFailure):
            await sb.execute("acme.unknown", {}, CTX, make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# execute — DISPATCHER_KIND_NOT_REGISTERED
# ---------------------------------------------------------------------------


class TestExecuteDispatcherNotRegistered:
    async def test_raises_sandbox_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_tool(TOOL_DEF)  # no dispatcher registered for "in_process"
        with pytest.raises(SandboxFailure) as exc_info:
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert exc_info.value.code == "DISPATCHER_KIND_NOT_REGISTERED"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].event == "sandbox.execute.failed"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb = Sandbox()
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "DISPATCHER_KIND_NOT_REGISTERED"

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# execute — dispatcher raises (TOOL_DISPATCH_FAILED)
# ---------------------------------------------------------------------------


class TestExecuteDispatcherRaises:
    async def test_wraps_as_dispatch_failed(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher(raises=RuntimeError("timeout")))
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure) as exc_info:
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert exc_info.value.code == "TOOL_DISPATCH_FAILED"

    async def test_cause_preserved(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        orig = RuntimeError("timeout")
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher(raises=orig))
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure) as exc_info:
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert exc_info.value.cause is orig

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher(raises=RuntimeError("boom")))
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher(raises=RuntimeError("boom")))
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_exception_recorded_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        orig = RuntimeError("boom")
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher(raises=orig))
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert orig in mock_span.recorded_exceptions

    async def test_on_error_callback(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher(raises=RuntimeError("bang")))
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "TOOL_DISPATCH_FAILED"

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher(raises=RuntimeError("crash")))
        sb.register_tool(TOOL_DEF)
        with pytest.raises(SandboxFailure):
            await sb.execute("test.echo", {}, CTX, make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# Registry guards
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_duplicate_tool_raises(self) -> None:
        sb = Sandbox()
        sb.register_tool(TOOL_DEF)
        with pytest.raises(ValueError, match="already registered"):
            sb.register_tool(TOOL_DEF)

    def test_duplicate_dispatcher_raises(self) -> None:
        sb = Sandbox()
        sb.register_dispatcher(StubDispatcher())
        with pytest.raises(ValueError, match="already registered"):
            sb.register_dispatcher(StubDispatcher())

    def test_get_tool_returns_definition(self) -> None:
        sb = Sandbox()
        sb.register_tool(TOOL_DEF)
        assert sb.get_tool("test.echo") is TOOL_DEF

    def test_get_tool_returns_none_for_unknown(self) -> None:
        sb = Sandbox()
        assert sb.get_tool("acme.unknown") is None

    def test_get_dispatcher_returns_dispatcher(self) -> None:
        sb = Sandbox()
        dispatcher = StubDispatcher()
        sb.register_dispatcher(dispatcher)
        assert sb.get_dispatcher("in_process") is dispatcher

    def test_get_dispatcher_returns_none_for_unknown(self) -> None:
        sb = Sandbox()
        assert sb.get_dispatcher("acme.unknown") is None


# ---------------------------------------------------------------------------
# execute — capability denial (CAPABILITY_DENIED)
# ---------------------------------------------------------------------------


CAPPED_TOOL = ToolDefinition(
    name="test.restricted",
    description="Requires fs.read capability",
    input_schema={"type": "object"},
    handler=InProcessHandler(),
    required_capabilities=frozenset({"fs.read", "net.outbound"}),
)

CTX_NO_CAPS = ExecutionContext(session_id="sess1", workspace="/tmp", granted_capabilities=frozenset())
CTX_PARTIAL_CAPS = ExecutionContext(
    session_id="sess1", workspace="/tmp", granted_capabilities=frozenset({"fs.read"})
)
CTX_FULL_CAPS = ExecutionContext(
    session_id="sess1", workspace="/tmp", granted_capabilities=frozenset({"fs.read", "net.outbound"})
)


def registered_sandbox_with_capped_tool() -> tuple[Sandbox, StubDispatcher]:
    dispatcher = StubDispatcher()
    sb = Sandbox()
    sb.register_dispatcher(dispatcher)
    sb.register_tool(CAPPED_TOOL)
    return sb, dispatcher


class TestExecuteCapabilityDenied:
    async def test_returns_result_not_raises(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Denial must return, never raise — no orchestrator crash."""
        sb, _ = registered_sandbox_with_capped_tool()
        result = await sb.execute(
            "test.restricted", {}, CTX_NO_CAPS, make_options(audit_log)
        )
        assert isinstance(result, SandboxResult)

    async def test_is_error_true(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        result = await sb.execute(
            "test.restricted", {}, CTX_NO_CAPS, make_options(audit_log)
        )
        assert result.is_error is True

    async def test_error_code_capability_denied(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        result = await sb.execute(
            "test.restricted", {}, CTX_NO_CAPS, make_options(audit_log)
        )
        assert result.error_code == "capability_denied"

    async def test_error_message_names_missing_caps(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        result = await sb.execute(
            "test.restricted", {}, CTX_NO_CAPS, make_options(audit_log)
        )
        assert result.error_message is not None
        assert "fs.read" in result.error_message
        assert "net.outbound" in result.error_message

    async def test_partial_caps_still_denied(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        result = await sb.execute(
            "test.restricted", {}, CTX_PARTIAL_CAPS, make_options(audit_log)
        )
        assert result.is_error is True
        assert result.error_code == "capability_denied"
        assert result.error_message is not None
        assert "net.outbound" in result.error_message

    async def test_full_caps_dispatches_normally(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """All required caps granted → dispatch proceeds, no error."""
        sb, dispatcher = registered_sandbox_with_capped_tool()
        result = await sb.execute(
            "test.restricted", {}, CTX_FULL_CAPS, make_options(audit_log)
        )
        assert result.is_error is False
        assert result.content == "ok"
        assert len(dispatcher.calls) == 1

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        from opentelemetry.trace import StatusCode

        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute("test.restricted", {}, CTX_NO_CAPS, make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_capability_denied_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute("test.restricted", {}, CTX_NO_CAPS, make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "capability.denied" in event_names

    async def test_capability_denied_event_attributes(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute("test.restricted", {}, CTX_NO_CAPS, make_options(audit_log))
        ev = next(e for e in mock_span.events if e[0] == "capability.denied")
        assert ev[1]["error.code"] == "capability_denied"
        assert "fs.read" in ev[1]["capability.missing"]
        assert ev[1]["tool.name"] == "test.restricted"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute("test.restricted", {}, CTX_NO_CAPS, make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "sandbox.capability.denied"

    async def test_audit_entry_detail(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute("test.restricted", {}, CTX_NO_CAPS, make_options(audit_log))
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["code"] == "capability_denied"
        assert "fs.read" in detail["missing"]
        assert "net.outbound" in detail["missing"]

    async def test_on_error_callback_called(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute(
            "test.restricted", {}, CTX_NO_CAPS, make_options(audit_log, errors)
        )
        assert len(errors) == 1
        assert errors[0].code == "capability_denied"

    async def test_on_error_callback_message(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute(
            "test.restricted", {}, CTX_NO_CAPS, make_options(audit_log, errors)
        )
        assert "test.restricted" in errors[0].message

    async def test_dispatcher_not_called_on_denial(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, dispatcher = registered_sandbox_with_capped_tool()
        await sb.execute("test.restricted", {}, CTX_NO_CAPS, make_options(audit_log))
        assert dispatcher.calls == []

    async def test_span_ended_on_denial(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_capped_tool()
        await sb.execute("test.restricted", {}, CTX_NO_CAPS, make_options(audit_log))
        assert mock_span.ended

    async def test_no_caps_required_always_passes(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Tool with no required_capabilities always dispatches regardless of granted."""
        sb, dispatcher = registered_sandbox(), StubDispatcher()
        # TOOL_DEF has empty required_capabilities
        result = await registered_sandbox().execute(
            "test.echo", {}, CTX_NO_CAPS, make_options(audit_log)
        )
        assert result.is_error is False
        assert result.content == "ok"


# ---------------------------------------------------------------------------
# execute — environment mismatch (ENV_MISMATCH)
# ---------------------------------------------------------------------------


ENV_TOOL = ToolDefinition(
    name="test.docker_only",
    description="Requires docker environment",
    input_schema={"type": "object"},
    handler=InProcessHandler(),
    requires_env="docker",
)

CTX_NO_ENV = ExecutionContext(session_id="sess1", workspace="/tmp")
CTX_WRONG_ENV = ExecutionContext(session_id="sess1", workspace="/tmp", environment="podman")
CTX_RIGHT_ENV = ExecutionContext(session_id="sess1", workspace="/tmp", environment="docker")


def registered_sandbox_with_env_tool() -> tuple[Sandbox, StubDispatcher]:
    dispatcher = StubDispatcher()
    sb = Sandbox()
    sb.register_dispatcher(dispatcher)
    sb.register_tool(ENV_TOOL)
    return sb, dispatcher


class TestExecuteEnvMismatch:
    async def test_returns_result_not_raises(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Env mismatch must return, never raise — no orchestrator crash."""
        sb, _ = registered_sandbox_with_env_tool()
        result = await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert isinstance(result, SandboxResult)

    async def test_is_error_true(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        result = await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert result.is_error is True

    async def test_error_code_env_mismatch(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        result = await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert result.error_code == "env_mismatch"

    async def test_error_message_names_required_env(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        result = await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert result.error_message is not None
        assert "docker" in result.error_message

    async def test_wrong_env_still_denied(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        result = await sb.execute(
            "test.docker_only", {}, CTX_WRONG_ENV, make_options(audit_log)
        )
        assert result.is_error is True
        assert result.error_code == "env_mismatch"
        assert result.error_message is not None
        assert "podman" in result.error_message

    async def test_matching_env_dispatches_normally(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Correct environment → dispatch proceeds, no error."""
        sb, dispatcher = registered_sandbox_with_env_tool()
        result = await sb.execute(
            "test.docker_only", {}, CTX_RIGHT_ENV, make_options(audit_log)
        )
        assert result.is_error is False
        assert result.content == "ok"
        assert len(dispatcher.calls) == 1

    async def test_no_requires_env_always_passes(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Tool with requires_env=None dispatches in any environment."""
        result = await registered_sandbox().execute(
            "test.echo", {}, CTX_NO_ENV, make_options(audit_log)
        )
        assert result.is_error is False

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        from opentelemetry.trace import StatusCode

        sb, _ = registered_sandbox_with_env_tool()
        await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_env_mismatch_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "env.mismatch" in event_names

    async def test_env_mismatch_event_attributes(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        ev = next(e for e in mock_span.events if e[0] == "env.mismatch")
        assert ev[1]["error.code"] == "env_mismatch"
        assert ev[1]["env.required"] == "docker"
        assert ev[1]["tool.name"] == "test.docker_only"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "sandbox.env.mismatch"

    async def test_audit_entry_detail(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["code"] == "env_mismatch"
        assert detail["requires_env"] == "docker"

    async def test_on_error_callback_called(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb, _ = registered_sandbox_with_env_tool()
        await sb.execute(
            "test.docker_only", {}, CTX_NO_ENV, make_options(audit_log, errors)
        )
        assert len(errors) == 1
        assert errors[0].code == "env_mismatch"

    async def test_dispatcher_not_called_on_mismatch(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, dispatcher = registered_sandbox_with_env_tool()
        await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert dispatcher.calls == []

    async def test_span_ended_on_mismatch(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_env_tool()
        await sb.execute("test.docker_only", {}, CTX_NO_ENV, make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# execute — timeout (TIMEOUT)
# ---------------------------------------------------------------------------


class SlowDispatcher(StubDispatcher):
    """Dispatcher that sleeps for a configurable duration before returning."""

    def __init__(self, sleep_seconds: float) -> None:
        super().__init__()
        self._sleep_seconds = sleep_seconds

    async def dispatch(
        self, tool: ToolDefinition, input: dict, context: ExecutionContext
    ) -> SandboxResult:
        import asyncio as _asyncio

        await _asyncio.sleep(self._sleep_seconds)
        self.calls.append((tool, input, context))
        return SandboxResult(content="ok", duration_ms=1.0)


TIMEOUT_TOOL = ToolDefinition(
    name="test.slow",
    description="Slow tool with short timeout",
    input_schema={"type": "object"},
    handler=InProcessHandler(),
    timeout_ms=50,  # 50 ms — will be exceeded by SlowDispatcher(sleep=1)
)


def registered_sandbox_with_timeout_tool(sleep: float = 1.0) -> tuple[Sandbox, SlowDispatcher]:
    dispatcher = SlowDispatcher(sleep)
    sb = Sandbox()
    sb.register_dispatcher(dispatcher)
    sb.register_tool(TIMEOUT_TOOL)
    return sb, dispatcher


class TestExecuteTimeout:
    async def test_returns_result_not_raises(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Timeout must return, never raise — no orchestrator crash."""
        sb, _ = registered_sandbox_with_timeout_tool()
        result = await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert isinstance(result, SandboxResult)

    async def test_is_error_true(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        result = await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert result.is_error is True

    async def test_error_code_timeout(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        result = await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert result.error_code == "timeout"

    async def test_error_message_contains_tool_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        result = await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert result.error_message is not None
        assert "test.slow" in result.error_message

    async def test_error_message_contains_timeout_ms(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        result = await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert result.error_message is not None
        assert "50" in result.error_message

    async def test_within_timeout_dispatches_normally(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Dispatcher completes before deadline → no error."""
        fast_tool = ToolDefinition(
            name="test.fast",
            description="Fast tool with generous timeout",
            input_schema={"type": "object"},
            handler=InProcessHandler(),
            timeout_ms=5_000,
        )
        dispatcher = SlowDispatcher(0.0)
        sb = Sandbox()
        sb.register_dispatcher(dispatcher)
        sb.register_tool(fast_tool)
        result = await sb.execute("test.fast", {}, CTX, make_options(audit_log))
        assert result.is_error is False
        assert result.content == "ok"

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        from opentelemetry.trace import StatusCode

        sb, _ = registered_sandbox_with_timeout_tool()
        await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_timeout_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "tool.timeout" in event_names

    async def test_timeout_event_attributes(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        ev = next(e for e in mock_span.events if e[0] == "tool.timeout")
        assert ev[1]["error.code"] == "timeout"
        assert ev[1]["timeout.ms"] == 50
        assert ev[1]["tool.name"] == "test.slow"

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "sandbox.tool.timeout"

    async def test_audit_entry_detail(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["code"] == "timeout"
        assert detail["timeout_ms"] == 50

    async def test_on_error_callback_called(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb, _ = registered_sandbox_with_timeout_tool()
        await sb.execute("test.slow", {}, CTX, make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "timeout"

    async def test_span_ended_on_timeout(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_timeout_tool()
        await sb.execute("test.slow", {}, CTX, make_options(audit_log))
        assert mock_span.ended


# ---------------------------------------------------------------------------
# execute — post-dispatch output schema validation
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 0},
        "label": {"type": "string"},
    },
    "required": ["count", "label"],
}

SCHEMA_TOOL = ToolDefinition(
    name="test.schema",
    description="Tool with output_schema",
    input_schema={"type": "object"},
    handler=InProcessHandler(),
    output_schema=OUTPUT_SCHEMA,
)


class SchemaDispatcher(ToolDispatcher):
    """Dispatcher whose returned content is configurable per call."""

    kind = "in_process"

    def __init__(self, content: object) -> None:
        self._content = content

    async def dispatch(
        self, tool: ToolDefinition, input: dict, context: ExecutionContext
    ) -> SandboxResult:
        return SandboxResult(content=self._content, duration_ms=1.0)


def registered_sandbox_with_schema_tool(content: object) -> Sandbox:
    sb = Sandbox()
    sb.register_dispatcher(SchemaDispatcher(content))
    sb.register_tool(SCHEMA_TOOL)
    return sb


class TestOutputSchemaValidation:
    async def test_valid_output_returns_success(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool({"count": 3, "label": "ok"})
        result = await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert result.is_error is False
        assert audit_log.entries == []

    async def test_invalid_output_returns_is_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool("not an object")
        result = await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert result.is_error is True

    async def test_invalid_output_error_code(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool("not an object")
        result = await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert result.error_code == "output_validation_failed"

    async def test_error_message_contains_offending_field_path(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """error_message must name the offending field path (JSON Path format)."""
        sb = registered_sandbox_with_schema_tool({"count": -1, "label": "x"})
        result = await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert result.error_message is not None
        assert "$.count" in result.error_message

    async def test_error_message_root_path_for_type_mismatch(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Root-level type mismatch should report '$' as the offending path."""
        sb = registered_sandbox_with_schema_tool("not an object")
        result = await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert result.error_message is not None
        assert "$" in result.error_message

    async def test_does_not_raise_returns_result(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Output schema failure must return SandboxResult, never raise."""
        sb = registered_sandbox_with_schema_tool("bad")
        result = await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert isinstance(result, SandboxResult)

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool("bad")
        await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "sandbox.output.schema_failed"

    async def test_audit_entry_detail_contains_validation_errors(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool({"count": -1, "label": "x"})
        await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["code"] == "output_validation_failed"
        assert isinstance(detail["validation_errors"], list)
        assert any("$.count" in e for e in detail["validation_errors"])

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        from opentelemetry.trace import StatusCode

        sb = registered_sandbox_with_schema_tool("bad")
        await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_output_schema_failed_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool("bad")
        await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        event_names = [e[0] for e in mock_span.events]
        assert "output.schema.failed" in event_names

    async def test_output_schema_failed_event_attributes(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool({"count": -1, "label": "x"})
        await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        ev = next(e for e in mock_span.events if e[0] == "output.schema.failed")
        assert ev[1]["error.code"] == "output_validation_failed"
        assert ev[1]["tool.name"] == "test.schema"
        assert "$.count" in ev[1]["schema.offending_path"]

    async def test_on_error_callback_called(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb = registered_sandbox_with_schema_tool("bad")
        await sb.execute("test.schema", {}, CTX, make_options(audit_log, errors))
        assert len(errors) == 1
        assert errors[0].code == "output_validation_failed"

    async def test_on_error_callback_message_contains_tool_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb = registered_sandbox_with_schema_tool("bad")
        await sb.execute("test.schema", {}, CTX, make_options(audit_log, errors))
        assert "test.schema" in errors[0].message

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb = registered_sandbox_with_schema_tool("bad")
        await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        assert mock_span.ended

    async def test_no_output_schema_skips_validation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Tool with output_schema=None must pass through any content without error."""
        result = await registered_sandbox().execute(
            "test.echo", {}, CTX, make_options(audit_log)
        )
        assert result.is_error is False

    async def test_dispatcher_error_skips_output_validation(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """If the dispatcher returns is_error=True, output schema is not checked."""

        class ErrorDispatcher(ToolDispatcher):
            kind = "in_process"

            async def dispatch(
                self, tool: ToolDefinition, input: dict, context: ExecutionContext
            ) -> SandboxResult:
                return SandboxResult(
                    content="handler error",
                    is_error=True,
                    error_code="handler_failed",
                    error_message="handler error",
                )

        sb = Sandbox()
        sb.register_dispatcher(ErrorDispatcher())
        sb.register_tool(SCHEMA_TOOL)
        result = await sb.execute("test.schema", {}, CTX, make_options(audit_log))
        # Passes through the handler error without re-wrapping as schema failure
        assert result.error_code == "handler_failed"
        assert audit_log.entries == []


# ---------------------------------------------------------------------------
# execute — pre-dispatch input schema validation
# ---------------------------------------------------------------------------

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 0},
        "label": {"type": "string"},
    },
    "required": ["count", "label"],
}

INPUT_SCHEMA_TOOL = ToolDefinition(
    name="test.input_schema",
    description="Tool with strict input_schema",
    input_schema=INPUT_SCHEMA,
    handler=InProcessHandler(),
)


def registered_sandbox_with_input_schema_tool() -> tuple[Sandbox, StubDispatcher]:
    dispatcher = StubDispatcher()
    sb = Sandbox()
    sb.register_dispatcher(dispatcher)
    sb.register_tool(INPUT_SCHEMA_TOOL)
    return sb, dispatcher


class TestInputSchemaValidation:
    async def test_valid_input_dispatches_normally(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, dispatcher = registered_sandbox_with_input_schema_tool()
        result = await sb.execute(
            "test.input_schema", {"count": 3, "label": "ok"}, CTX, make_options(audit_log)
        )
        assert result.is_error is False
        assert len(dispatcher.calls) == 1
        assert audit_log.entries == []

    async def test_invalid_input_returns_is_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Schema failure must return SandboxResult(is_error=True), never raise."""
        sb, _ = registered_sandbox_with_input_schema_tool()
        result = await sb.execute(
            "test.input_schema", "not an object", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        assert result.is_error is True

    async def test_invalid_input_error_code(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_input_schema_tool()
        result = await sb.execute(
            "test.input_schema", {"count": -1, "label": "x"}, CTX, make_options(audit_log)
        )
        assert result.error_code == "input_validation_failed"

    async def test_error_message_contains_offending_field_path(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """error_message must name the offending field path (JSON Path format)."""
        sb, _ = registered_sandbox_with_input_schema_tool()
        result = await sb.execute(
            "test.input_schema", {"count": -1, "label": "x"}, CTX, make_options(audit_log)
        )
        assert result.error_message is not None
        assert "$.count" in result.error_message

    async def test_error_message_root_path_for_type_mismatch(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Root-level type mismatch should report '$' as the offending path."""
        sb, _ = registered_sandbox_with_input_schema_tool()
        result = await sb.execute(
            "test.input_schema", "not an object", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        assert result.error_message is not None
        assert "$" in result.error_message

    async def test_does_not_raise_returns_result(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Input schema failure must return SandboxResult, never raise."""
        sb, _ = registered_sandbox_with_input_schema_tool()
        result = await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        assert isinstance(result, SandboxResult)

    async def test_dispatcher_not_called_on_invalid_input(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Dispatcher must not be invoked when input schema validation fails."""
        sb, dispatcher = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        assert dispatcher.calls == []

    async def test_audit_entry_written(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "sandbox.input.schema_failed"

    async def test_audit_entry_detail_contains_validation_errors(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", {"count": -1, "label": "x"}, CTX, make_options(audit_log)
        )
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["code"] == "input_validation_failed"
        assert isinstance(detail["validation_errors"], list)
        assert any("$.count" in e for e in detail["validation_errors"])

    async def test_span_marked_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        from opentelemetry.trace import StatusCode

        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_input_schema_failed_event_on_span(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        event_names = [e[0] for e in mock_span.events]
        assert "input.schema.failed" in event_names

    async def test_input_schema_failed_event_attributes(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", {"count": -1, "label": "x"}, CTX, make_options(audit_log)
        )
        ev = next(e for e in mock_span.events if e[0] == "input.schema.failed")
        assert ev[1]["error.code"] == "input_validation_failed"
        assert ev[1]["tool.name"] == "test.input_schema"
        assert "$.count" in ev[1]["schema.offending_path"]

    async def test_on_error_callback_called(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log, errors)  # type: ignore[arg-type]
        )
        assert len(errors) == 1
        assert errors[0].code == "input_validation_failed"

    async def test_on_error_callback_message_contains_tool_name(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        errors: list[SandboxFailure] = []
        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log, errors)  # type: ignore[arg-type]
        )
        assert "test.input_schema" in errors[0].message

    async def test_span_ended_on_failure(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        sb, _ = registered_sandbox_with_input_schema_tool()
        await sb.execute(
            "test.input_schema", "bad", CTX, make_options(audit_log)  # type: ignore[arg-type]
        )
        assert mock_span.ended

    async def test_missing_required_field_returns_error(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        """Missing required field is caught pre-dispatch."""
        sb, dispatcher = registered_sandbox_with_input_schema_tool()
        result = await sb.execute(
            "test.input_schema", {"count": 1}, CTX, make_options(audit_log)
        )
        assert result.is_error is True
        assert result.error_code == "input_validation_failed"
        assert dispatcher.calls == []


# ---------------------------------------------------------------------------
# SandboxResult helpers
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_to_mcp_content_blocks_string(self) -> None:
        result = SandboxResult(content="hello")
        blocks = result.to_mcp_content_blocks()
        assert blocks == [{"type": "text", "text": "hello"}]

    def test_to_mcp_content_blocks_non_string(self) -> None:
        result = SandboxResult(content={"key": "val"})
        blocks = result.to_mcp_content_blocks()
        assert blocks[0]["type"] == "text"
        assert "key" in blocks[0]["text"]
