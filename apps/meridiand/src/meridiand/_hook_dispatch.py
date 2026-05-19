"""
Hook dispatch via the Sandbox surface.

Hooks are dispatched through the same Sandbox.execute() path as tools:
timeouts are enforced via asyncio.wait_for, capability scoping via the
ExecutionContext, and subprocess isolation via the SubprocessDispatcher.

failure_mode semantics
----------------------
block / abort : writes audit log at error level, raises HookDispatchBlockedError
                so the caller can surface the error message.
warn          : writes audit log at warn level, continues.
ignore        : continues silently (no audit write).

On any failure: the error message is available on HookDispatchBlockedError.message
and/or in HookDispatchResult.error_message; the audit log is written before the
error is raised.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sdk_sandbox as _sb
from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Bridge: sdk_sandbox.AuditLog → core_errors.AuditLog
# ---------------------------------------------------------------------------


class _SandboxAuditBridge(_sb.AuditLog):
    """Adapts sdk_sandbox.AuditLog writes to core_errors.AuditLog."""

    def __init__(self, core_log: AuditLog) -> None:
        self._log = core_log

    def write(self, entry: _sb.AuditLogEntry) -> None:
        detail: dict[str, Any] = dict(entry.detail or {})
        code = str(detail.pop("code", "sandbox_error"))
        detail["tool_name"] = entry.tool_name
        detail["session_id"] = entry.session_id
        self._log.write(
            AuditLogEntry(
                level=entry.level,
                event=entry.event,
                code=code,
                timestamp=entry.timestamp,
                detail=detail if detail else None,
            )
        )


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class HookDispatchBlockedError(MeridianError):
    """Raised when failure_mode=block (or abort) and the hook dispatch fails."""

    def __init__(
        self,
        *,
        hook_id: str,
        hook_name: str,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="hook_dispatch_blocked",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )
        self.hook_id = hook_id
        self.hook_name = hook_name

    def http_status(self) -> int:
        return 502


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookDispatchResult:
    hook_id: str
    hook_name: str
    is_error: bool
    error_code: str | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Internal: build ToolHandler + ToolDispatcher from hook definition
# ---------------------------------------------------------------------------


def _build_dispatcher(
    handler_type: str,
    bridge: _SandboxAuditBridge,
) -> _sb.ToolDispatcher:
    if handler_type == "subprocess":
        return _sb.SubprocessDispatcher(audit_log=bridge)
    if handler_type == "http":
        return _sb.HttpDispatcher(audit_log=bridge)
    if handler_type == "mcp":
        return _sb.McpDispatcher(audit_log=bridge)
    if handler_type == "container":
        return _sb.ContainerDispatcher(audit_log=bridge)
    return _sb.InProcessDispatcher(audit_log=bridge)


def _build_tool_handler(handler_type: str, metadata: dict[str, Any]) -> Any:
    if handler_type == "subprocess":
        return _sb.SubprocessHandler(path=metadata.get("path", ""))
    if handler_type == "http":
        return _sb.HttpHandler(url=metadata.get("url", ""))
    if handler_type == "mcp":
        return _sb.McpHandler(
            server_url=metadata.get("server_url", ""),
            tool_name=metadata.get("tool_name", ""),
            transport=metadata.get("transport", "http"),
            command=tuple(metadata.get("command", [])),
        )
    if handler_type == "container":
        return _sb.ContainerHandler(
            environment_id=metadata.get("environment_id", ""),
            entrypoint=metadata.get("entrypoint", ""),
        )
    return _sb.InProcessHandler(module=metadata.get("module", ""))


# ---------------------------------------------------------------------------
# Internal: match filter
# ---------------------------------------------------------------------------


def _matches_filter(hook: dict[str, Any], context: _sb.ExecutionContext) -> bool:
    match_filter = hook.get("match")
    if match_filter is None:
        return True
    session_id_filter = match_filter.get("session_id")
    if session_id_filter is not None and session_id_filter != context.session_id:
        return False
    return True


# ---------------------------------------------------------------------------
# Internal: load active hooks for event
# ---------------------------------------------------------------------------


def _load_active_hooks(
    hooks_dir: Path,
    event: str,
    context: _sb.ExecutionContext,
) -> list[dict[str, Any]]:
    if not hooks_dir.exists():
        return []
    matched = []
    for path in sorted(hooks_dir.glob("*.json")):
        try:
            hook = json.loads(path.read_text())
        except Exception:
            continue
        if hook.get("status") != "active":
            continue
        if hook.get("event") != event:
            continue
        if not _matches_filter(hook, context):
            continue
        matched.append(hook)
    return matched


# ---------------------------------------------------------------------------
# Internal: dispatch single hook via a fresh Sandbox instance
# ---------------------------------------------------------------------------


async def _dispatch_one(
    hook: dict[str, Any],
    payload: dict[str, Any],
    context: _sb.ExecutionContext,
    *,
    bridge: _SandboxAuditBridge,
    in_process_handlers: dict[str, Callable[..., Awaitable[Any]]] | None,
) -> _sb.SandboxResult:
    hook_id = hook["id"]
    handler_type = hook["handler"]
    timeout_ms = hook["timeout_ms"]
    metadata = hook.get("metadata") or {}

    dispatcher = _build_dispatcher(handler_type, bridge)
    tool_handler = _build_tool_handler(handler_type, metadata)

    if handler_type == "in_process" and in_process_handlers is not None:
        fn = in_process_handlers.get(hook_id)
        if fn is not None:
            assert isinstance(dispatcher, _sb.InProcessDispatcher)
            dispatcher.register(hook_id, fn)

    tool = _sb.ToolDefinition(
        name=hook_id,
        description=f"Hook: {hook.get('name', hook_id)}",
        input_schema={"type": "object"},
        handler=tool_handler,
        timeout_ms=timeout_ms,
    )

    sandbox = _sb.Sandbox()
    sandbox.register_dispatcher(dispatcher)
    sandbox.register_tool(tool)

    try:
        return await sandbox.execute(
            hook_id,
            payload,
            context,
            _sb.RuntimeOptions(audit_log=bridge),
        )
    except _sb.SandboxFailure as sf:
        return _sb.SandboxResult(
            content=sf.message,
            is_error=True,
            error_code=sf.code,
            error_message=sf.message,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def dispatch_hooks(
    event: str,
    payload: dict[str, Any],
    context: _sb.ExecutionContext,
    *,
    hooks_dir: Path,
    audit_log: AuditLog,
    in_process_handlers: dict[str, Callable[..., Awaitable[Any]]] | None = None,
) -> list[HookDispatchResult]:
    """
    Load all active hooks subscribed to *event*, dispatch each via the Sandbox
    surface (same as tools), and return per-hook results.

    Timeouts, capability scoping, and subprocess isolation are all enforced
    transparently by Sandbox.execute().

    failure_mode=block (or abort): writes audit log at error level, raises
    HookDispatchBlockedError — the error message is surfaced to the caller.
    failure_mode=warn: writes audit log at warn level, continues.
    failure_mode=ignore: continues silently.

    in_process_handlers: optional dict mapping hook_id → async callable for
    in_process hooks (callables cannot be stored in JSON).
    """
    now = _now()
    tracer = get_tracer()
    bridge = _SandboxAuditBridge(audit_log)
    hooks = _load_active_hooks(hooks_dir, event, context)
    results: list[HookDispatchResult] = []

    with tracer.start_as_current_span(
        "hook.dispatch",
        attributes={
            "hook.event": event,
            "hook.count": len(hooks),
            "session.id": context.session_id,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="hook.dispatch.invocation",
                code="hook_dispatch",
                timestamp=now,
            ),
        )

        for hook in hooks:
            hook_id = hook["id"]
            hook_name = hook["name"]
            failure_mode = hook.get("failure_mode", "ignore")
            hook_now = _now()

            with tracer.start_as_current_span(
                "hook.dispatch.single",
                attributes={
                    "hook.id": hook_id,
                    "hook.name": hook_name,
                    "hook.failure_mode": failure_mode,
                    "session.id": context.session_id,
                },
            ) as hook_span:
                record_invocation_event(
                    hook_span,
                    StructuredEvent(
                        name="hook.dispatch.single.invocation",
                        code="hook_dispatch_single",
                        timestamp=hook_now,
                    ),
                )

                result = await _dispatch_one(
                    hook,
                    payload,
                    context,
                    bridge=bridge,
                    in_process_handlers=in_process_handlers,
                )

                hook_result = HookDispatchResult(
                    hook_id=hook_id,
                    hook_name=hook_name,
                    is_error=result.is_error,
                    error_code=result.error_code,
                    error_message=result.error_message,
                )

                if result.is_error:
                    err_msg = result.error_message or "Hook dispatch failed"
                    err_code = result.error_code or "hook_dispatch_failed"

                    if failure_mode in ("block", "abort"):
                        audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="hook.dispatch.failed",
                                code=err_code,
                                timestamp=hook_now,
                                detail={
                                    "hook_id": hook_id,
                                    "hook_name": hook_name,
                                    "message": err_msg,
                                    "failure_mode": failure_mode,
                                },
                            )
                        )
                        exc = HookDispatchBlockedError(
                            hook_id=hook_id,
                            hook_name=hook_name,
                            message=err_msg,
                            timestamp=hook_now,
                        )
                        record_error(hook_span, exc)
                        raise exc

                    elif failure_mode == "warn":
                        audit_log.write(
                            AuditLogEntry(
                                level="warn",
                                event="hook.dispatch.failed",
                                code=err_code,
                                timestamp=hook_now,
                                detail={
                                    "hook_id": hook_id,
                                    "hook_name": hook_name,
                                    "message": err_msg,
                                    "failure_mode": failure_mode,
                                },
                            )
                        )
                    # ignore: no audit write, no raise

                results.append(hook_result)

    return results
