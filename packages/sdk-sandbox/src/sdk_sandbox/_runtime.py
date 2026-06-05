from __future__ import annotations

import asyncio
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._audit import AuditLog, NoopAuditLog
from ._contract import ToolDispatcher
from ._schema import InputSchemaError, OutputSchemaError, validate_input, validate_output
from ._secret_refs import SecretRefResolveError, substitute_secret_refs
from ._telemetry import (
    get_tracer,
    record_capability_denial,
    record_env_mismatch,
    record_env_routing,
    record_input_schema_failure,
    record_invocation_event,
    record_output_schema_failure,
    record_sandbox_failure,
    record_tool_timeout,
)
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
    storage_root: Path | None = None  # enables secret_ref://vault/{id}/{name} substitution


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
        self._tool_env_routes: dict[str, str] = {}

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

    def configure_tool_env(self, tool_name: str, environment: str) -> None:
        """Route a tool to a specific environment via configuration, not code change."""
        self._tool_env_routes[tool_name] = environment

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

            # Per-tool environment routing: switch context.environment from
            # configuration, not from a code change to the tool definition.
            # Sandbox dispatch surface is identical across backends — the harness
            # calls execute(name, input, context) unchanged; routing is internal.
            routed_env = self._tool_env_routes.get(name)
            if routed_env is not None:
                record_env_routing(
                    span,
                    tool_name=name,
                    session_id=context.session_id,
                    original_env=context.environment,
                    routed_env=routed_env,
                )
                opts.audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="sandbox.env.routed",
                        tool_name=name,
                        session_id=context.session_id,
                        timestamp=now,
                        detail={
                            "routed_env": routed_env,
                            "original_env": context.environment,
                        },
                    )
                )
                context = dataclasses.replace(context, environment=routed_env)

            # Capability check: required caps must be a subset of granted caps.
            # On denial, return a synthetic SandboxResult (is_error=True) so the
            # orchestrator surfaces it to the model — never raise, never silent.
            missing_caps = tool.required_capabilities - context.granted_capabilities
            if missing_caps:
                missing_str = ", ".join(sorted(missing_caps))
                denial_message = f'Capability denied for tool "{name}"; missing: {missing_str}'
                record_capability_denial(
                    span,
                    tool_name=name,
                    session_id=context.session_id,
                    required=tool.required_capabilities,
                    missing=missing_caps,
                    granted=context.granted_capabilities,
                    message=denial_message,
                )
                opts.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sandbox.capability.denied",
                        tool_name=name,
                        session_id=context.session_id,
                        timestamp=now,
                        detail={
                            "code": "capability_denied",
                            "message": denial_message,
                            "required": sorted(tool.required_capabilities),
                            "missing": sorted(missing_caps),
                            "granted": sorted(context.granted_capabilities),
                        },
                    )
                )
                if opts.on_error is not None:
                    opts.on_error(
                        SandboxFailure(
                            code="capability_denied",
                            message=denial_message,
                            tool_name=name,
                            session_id=context.session_id,
                            timestamp=now,
                        )
                    )
                return SandboxResult(
                    content=denial_message,
                    is_error=True,
                    error_code="capability_denied",
                    error_message=denial_message,
                )

            # Environment check: requires_env must match context.environment.
            # On mismatch, return a synthetic SandboxResult (is_error=True) —
            # never raise, never silent.
            if tool.requires_env is not None and context.environment != tool.requires_env:
                env_message = (
                    f'Environment mismatch for tool "{name}"; '
                    f'requires "{tool.requires_env}", '
                    f'got "{context.environment}"'
                )
                record_env_mismatch(
                    span,
                    tool_name=name,
                    session_id=context.session_id,
                    requires_env=tool.requires_env,
                    actual_env=context.environment,
                    message=env_message,
                )
                opts.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sandbox.env.mismatch",
                        tool_name=name,
                        session_id=context.session_id,
                        timestamp=now,
                        detail={
                            "code": "env_mismatch",
                            "message": env_message,
                            "requires_env": tool.requires_env,
                            "actual_env": context.environment,
                        },
                    )
                )
                if opts.on_error is not None:
                    opts.on_error(
                        SandboxFailure(
                            code="env_mismatch",
                            message=env_message,
                            tool_name=name,
                            session_id=context.session_id,
                            timestamp=now,
                        )
                    )
                return SandboxResult(
                    content=env_message,
                    is_error=True,
                    error_code="env_mismatch",
                    error_message=env_message,
                )

            # Secret-ref substitution.  Scan tool args for
            # secret_ref://vault/{id}/{name} strings and replace with plaintext
            # values fetched from disk.  Only enabled when storage_root is set.
            # The original ref strings (not values) are logged; plaintext never
            # enters any log.  On failure return is_error=True to surface the
            # error to the model as a tool_result.
            if opts.storage_root is not None:
                try:
                    input, _refs = substitute_secret_refs(
                        input,
                        storage_root=opts.storage_root,
                        audit_log=opts.audit_log,
                        tool_name=name,
                        session_id=context.session_id,
                    )
                except SecretRefResolveError as exc:
                    ref_error_message = exc.message
                    if opts.on_error is not None:
                        opts.on_error(
                            SandboxFailure(
                                code=exc.code,
                                message=ref_error_message,
                                tool_name=name,
                                session_id=context.session_id,
                                timestamp=now,
                            )
                        )
                    return SandboxResult(
                        content=ref_error_message,
                        is_error=True,
                        error_code=exc.code,
                        error_message=ref_error_message,
                    )

            # Pre-dispatch input schema validation.  Always validates — even
            # when input_schema is a bare {"type": "object"} — so callers
            # receive a structured error instead of a silent type mismatch.
            try:
                validate_input(tool.input_schema, input)
            except InputSchemaError as exc:
                offending_path = exc.errors[0] if exc.errors else str(exc)
                schema_message = (
                    f'Input schema validation failed for tool "{name}": {offending_path}'
                )
                record_input_schema_failure(
                    span,
                    tool_name=name,
                    session_id=context.session_id,
                    offending_path=offending_path,
                    message=schema_message,
                )
                opts.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sandbox.input.schema_failed",
                        tool_name=name,
                        session_id=context.session_id,
                        timestamp=now,
                        detail={
                            "code": "input_validation_failed",
                            "message": schema_message,
                            "validation_errors": exc.errors,
                        },
                    )
                )
                if opts.on_error is not None:
                    opts.on_error(
                        SandboxFailure(
                            code="input_validation_failed",
                            message=schema_message,
                            tool_name=name,
                            session_id=context.session_id,
                            timestamp=now,
                        )
                    )
                return SandboxResult(
                    content=schema_message,
                    is_error=True,
                    error_code="input_validation_failed",
                    error_message=schema_message,
                )

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
                dispatch_result = await asyncio.wait_for(
                    dispatcher.dispatch(tool, input, context),
                    timeout=tool.timeout_ms / 1000,
                )
            except TimeoutError:
                timeout_message = f'Tool "{name}" timed out after {tool.timeout_ms}ms'
                record_tool_timeout(
                    span,
                    tool_name=name,
                    session_id=context.session_id,
                    timeout_ms=tool.timeout_ms,
                    message=timeout_message,
                )
                opts.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="sandbox.tool.timeout",
                        tool_name=name,
                        session_id=context.session_id,
                        timestamp=now,
                        detail={
                            "code": "timeout",
                            "message": timeout_message,
                            "timeout_ms": tool.timeout_ms,
                        },
                    )
                )
                if opts.on_error is not None:
                    opts.on_error(
                        SandboxFailure(
                            code="timeout",
                            message=timeout_message,
                            tool_name=name,
                            session_id=context.session_id,
                            timestamp=now,
                        )
                    )
                return SandboxResult(
                    content=timeout_message,
                    is_error=True,
                    error_code="timeout",
                    error_message=timeout_message,
                )
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

            # Post-dispatch output schema validation.  Only validates successful
            # results — errors returned by the dispatcher are passed through as-is.
            if tool.output_schema is not None and not dispatch_result.is_error:
                try:
                    validate_output(tool.output_schema, dispatch_result.content)
                except OutputSchemaError as exc:
                    offending_path = exc.errors[0] if exc.errors else str(exc)
                    schema_message = (
                        f'Output schema validation failed for tool "{name}": {offending_path}'
                    )
                    record_output_schema_failure(
                        span,
                        tool_name=name,
                        session_id=context.session_id,
                        offending_path=offending_path,
                        message=schema_message,
                    )
                    opts.audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="sandbox.output.schema_failed",
                            tool_name=name,
                            session_id=context.session_id,
                            timestamp=now,
                            detail={
                                "code": "output_validation_failed",
                                "message": schema_message,
                                "validation_errors": exc.errors,
                            },
                        )
                    )
                    if opts.on_error is not None:
                        opts.on_error(
                            SandboxFailure(
                                code="output_validation_failed",
                                message=schema_message,
                                tool_name=name,
                                session_id=context.session_id,
                                timestamp=now,
                            )
                        )
                    return SandboxResult(
                        content=schema_message,
                        is_error=True,
                        error_code="output_validation_failed",
                        error_message=schema_message,
                    )

            return dispatch_result


default_sandbox = Sandbox()
