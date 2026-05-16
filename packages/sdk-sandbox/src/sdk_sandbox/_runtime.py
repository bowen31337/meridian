from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ._audit import AuditLog, NoopAuditLog
from ._contract import ToolDispatcher
from ._telemetry import get_tracer, record_invocation_event, record_sandbox_failure
from ._types import (
    AuditLogEntry,
    ExecutionContext,
    SandboxFailure,
    SandboxResult,
    StructuredEvent,
    ToolDefinition,
)


@dataclass
class RuntimeOptions:
    """Options supplied by the host application for each Sandbox call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[SandboxFailure], None] | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Sandbox:
    """
    Single dispatch surface for every executable action.

    Register tools (ToolDefinition) by name and dispatchers (ToolDispatcher)
    by handler kind once at startup. execute() opens an OTel span, attaches a
    structured invocation event, routes to the matching dispatcher, and on
    failure records the span error, writes the audit log, and calls on_error.

    The harness does not branch on backend — it calls execute(name, input, context)
    and the Sandbox routes to the right dispatcher via the tool's handler kind.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._dispatchers: dict[str, ToolDispatcher] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool by name. Raises ValueError on duplicate."""
        if tool.name in self._tools:
            raise ValueError(f'Tool "{tool.name}" is already registered')
        self._tools[tool.name] = tool

    def register_dispatcher(self, dispatcher: ToolDispatcher) -> None:
        """Register a dispatcher by handler kind. Raises ValueError on duplicate."""
        kind = dispatcher.kind
        if kind in self._dispatchers:
            raise ValueError(f'Dispatcher for kind "{kind}" is already registered')
        self._dispatchers[kind] = dispatcher

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Return the registered tool by name, or None."""
        return self._tools.get(name)

    def get_dispatcher(self, kind: str) -> ToolDispatcher | None:
        """Return the registered dispatcher for a handler kind, or None."""
        return self._dispatchers.get(kind)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fail(
        self,
        span: object,
        failure: SandboxFailure,
        options: RuntimeOptions,
        audit_event: str,
    ) -> None:
        record_sandbox_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                tool_name=failure.tool_name,
                session_id=failure.session_id,
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        input: dict[str, Any],
        context: ExecutionContext,
        options: RuntimeOptions | None = None,
    ) -> SandboxResult:
        """
        Single dispatch surface for every executable action.

        Per-invocation:
          1. Opens OTel span "sandbox.execute" with tool name and session attributes.
          2. Attaches a "sandbox.invocation" structured event to the span.
          3. Validates the tool is registered (TOOL_NOT_REGISTERED on failure).
          4. Validates the dispatcher for the tool's handler kind is registered
             (DISPATCHER_KIND_NOT_REGISTERED on failure).
          5. Dispatches to the dispatcher; wraps exceptions as TOOL_DISPATCH_FAILED.
        Returns SandboxResult on success. On any failure, records span error,
        writes audit log, calls on_error, and raises SandboxFailure.
        """
        opts = options or RuntimeOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "sandbox.execute",
            attributes={
                "tool.name": name,
                "session.id": context.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="sandbox.invocation",
                    tool_name=name,
                    session_id=context.session_id,
                    timestamp=now,
                    operation="execute",
                ),
            )

            tool = self._tools.get(name)
            if tool is None:
                failure = SandboxFailure(
                    code="TOOL_NOT_REGISTERED",
                    message=f'No tool registered with name "{name}"',
                    tool_name=name,
                    session_id=context.session_id,
                    timestamp=now,
                )
                self._fail(span, failure, opts, "sandbox.execute.failed")
                raise failure

            dispatcher = self._dispatchers.get(tool.handler.kind)
            if dispatcher is None:
                failure = SandboxFailure(
                    code="DISPATCHER_KIND_NOT_REGISTERED",
                    message=f'No dispatcher registered for handler kind "{tool.handler.kind}"',
                    tool_name=name,
                    session_id=context.session_id,
                    timestamp=now,
                )
                self._fail(span, failure, opts, "sandbox.execute.failed")
                raise failure

            try:
                return await dispatcher.dispatch(tool, input, context)
            except SandboxFailure:
                raise
            except Exception as exc:
                failure = SandboxFailure(
                    code="TOOL_DISPATCH_FAILED",
                    message=str(exc),
                    tool_name=name,
                    session_id=context.session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "sandbox.execute.failed")
                raise failure from exc


default_sandbox = Sandbox()
