from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

from ._audit import write_audit_event
from ._otel import tool_span
from ._schema import SchemaValidationError, validate_input, validate_output
from ._types import ToolContext, ToolDefinition, ToolError, ToolResult

logger = logging.getLogger("meridian.sdk_tool")


async def execute_tool(
    definition: ToolDefinition,
    args: Any,
    ctx: ToolContext,
    handler: Callable[[Any, ToolContext], Awaitable[Any]],
    audit_log_path: str | None = None,
) -> ToolResult:
    """Wrap a tool invocation with the full SDK pipeline.

    Pipeline (Architecture §11):
    1. Pre-dispatch: validate *args* against input_schema.
    2. Open an OTel span for the call.
    3. Call *handler(args, ctx)*.
    4. Post-dispatch: validate the returned value against output_schema.
    5. Emit a structured log event with outcome + duration.
    6. On any failure: write to the audit log and return ToolResult(is_error=True).

    Schema failures and handler exceptions are never re-raised — they produce
    an is_error=true ToolResult so the model can decide what to do next
    (Architecture §11.4, PRD F-SB-3).
    """
    start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Pre-dispatch input validation
    # ------------------------------------------------------------------
    try:
        validate_input(definition.input_schema, args)
    except SchemaValidationError as exc:
        return _fail(
            definition.name,
            ctx.session_id,
            "input_validation_failed",
            str(exc),
            {"validation_errors": exc.errors},
            start,
            audit_log_path,
        )

    # ------------------------------------------------------------------
    # 2–4. OTel span + dispatch + output validation
    # ------------------------------------------------------------------
    async with tool_span(
        definition.name,
        session_id=ctx.session_id,
        extra_attrs={"meridian.workspace": ctx.workspace},
    ):
        try:
            raw_result = await handler(args, ctx)
        except Exception as exc:  # noqa: BLE001
            return _fail(
                definition.name,
                ctx.session_id,
                "execution_failed",
                str(exc),
                {"exception_type": type(exc).__name__},
                start,
                audit_log_path,
            )

        if definition.output_schema is not None:
            try:
                validate_output(definition.output_schema, raw_result)
            except SchemaValidationError as exc:
                return _fail(
                    definition.name,
                    ctx.session_id,
                    "output_validation_failed",
                    str(exc),
                    {"validation_errors": exc.errors},
                    start,
                    audit_log_path,
                )

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "tool.executed",
        extra={
            "tool_name": definition.name,
            "session_id": ctx.session_id,
            "duration_ms": duration_ms,
            "success": True,
        },
    )
    return ToolResult.ok(raw_result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fail(
    tool_name: str,
    session_id: str,
    code: str,
    message: str,
    details: dict[str, Any],
    start: float,
    audit_log_path: str | None,
) -> ToolResult:
    duration_ms = int((time.monotonic() - start) * 1000)
    error = ToolError(code=code, message=message, details=details)

    logger.warning(
        "tool.%s",
        code,
        extra={
            "tool_name": tool_name,
            "session_id": session_id,
            "duration_ms": duration_ms,
            "error": error.model_dump(),
        },
    )

    write_audit_event(
        event_type=f"tool.{code}",
        tool_name=tool_name,
        session_id=session_id,
        error=error.model_dump(),
        audit_log_path=audit_log_path,
    )

    return ToolResult.err(code, message, **details)
