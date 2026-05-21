from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jsonschema

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    NoopAuditLog,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from meridian_sdk_provider import (
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelRouter,
)
from sdk_sandbox import ExecutionContext
from storage_event_log import EventLogWriter

from ._event_translator import ModelEventTranslator
from ._hook_dispatch import HookVetoError, dispatch_hooks


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class UsageDelta:
    """One usage increment emitted by the harness after each model call completes."""

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class IterationBudget:
    """Per-iteration budget thresholds checked after each model call in the harness loop.

    soft: warn + pause for user approval when model_calls reaches this value.
    hard: terminate the session when model_calls reaches this value.
    """

    hard: int | None = None
    soft: int | None = None


@dataclass
class MaxTokensPolicy:
    """Policy applied when the model response has stop_reason="max_tokens".

    continue_allowed: if True the loop re-calls the model; if False (default)
    the session transitions to waiting_for_user.
    """

    continue_allowed: bool = False


# Phases in which the harness releases the session so any harness can re-wake.
_STOP_PHASES: frozenset[str] = frozenset({"idle", "paused", "terminated"})


@runtime_checkable
class _PhaseReader(Protocol):
    def current_phase(self, session_id: str) -> str: ...


def _extract_canvas_op(content: Any) -> dict[str, Any] | None:
    """Return the CanvasOp dict when *content* is a canvas_op content block, else None."""
    if isinstance(content, dict) and content.get("type") == "canvas_op":
        op = content.get("canvas_op")
        return op if isinstance(op, dict) else None
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("type") == "canvas_op":
                op = parsed.get("canvas_op")
                return op if isinstance(op, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _validate_tool_args(schema: dict[str, Any], data: Any) -> list[str]:
    """Return schema validation error strings for *data* against *schema*, or [] if valid."""
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for err in validator.iter_errors(data):
        path = "".join(
            f"[{p!r}]" if isinstance(p, str) else f"[{p}]"
            for p in err.absolute_path
        )
        errors.append(f"${path}: {err.message}" if path else err.message)
    return errors


__all__ = [
    "FakeModelAdapter",
    "FakeSandboxAdapter",
    "HarnessLoopError",
    "IterationBudget",
    "MaxTokensPolicy",
    "UsageDelta",
    "_run_harness",
    "_run_harness_capturing",
    "_find_divergence",
    "run_harness_loop",
]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ReplayError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="replay_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


class FakeModelAdapter:
    """
    ModelProvider that plays back canned model responses from an NDJSON fixture file.

    Each line is a JSON array of event dicts representing one model call.
    """

    name = "fake"
    kind = "fake"

    def __init__(self, fixture_path: Path) -> None:
        self._calls: list[list[dict[str, Any]]] = []
        if fixture_path.exists():
            for raw in fixture_path.read_text().splitlines():
                line = raw.strip()
                if line:
                    self._calls.append(json.loads(line))
        self._index = 0

    async def call(self) -> AsyncIterator[dict[str, Any]]:
        if self._index < len(self._calls):
            events = self._calls[self._index]
            self._index += 1
            for event in events:
                yield event

    @property
    def call_count(self) -> int:
        return self._index


class FakeSandboxAdapter:
    """
    Tool dispatcher that plays back canned tool results from an NDJSON fixture file.

    Each line is a JSON object with at least a "content" field.
    """

    def __init__(self, fixture_path: Path) -> None:
        self._responses: list[dict[str, Any]] = []
        if fixture_path.exists():
            for raw in fixture_path.read_text().splitlines():
                line = raw.strip()
                if line:
                    self._responses.append(json.loads(line))
        self._index = 0

    def next_result(self) -> dict[str, Any]:
        if self._index < len(self._responses):
            result = self._responses[self._index]
            self._index += 1
            return result
        return {"content": ""}

    @property
    def dispatch_count(self) -> int:
        return self._index


# ---------------------------------------------------------------------------
# Harness loop
# ---------------------------------------------------------------------------


async def _run_harness(
    model_adapter: FakeModelAdapter,
    sandbox_adapter: FakeSandboxAdapter,
    on_usage_delta: Callable[[UsageDelta], None] | None = None,
    cancel_event: asyncio.Event | None = None,
    *,
    session_id: str = "",
    hooks_dir: Path | None = None,
    audit_log: AuditLog | None = None,
) -> tuple[int, int]:
    """Run the agent harness loop with fake adapters. Returns (model_calls, tool_calls).

    After each model call completes, calls on_usage_delta with synthetic token counts so
    the parent can accumulate usage.delta events in real time.  Checks cancel_event at the
    top of every loop iteration so a sibling's budget breach can stop this worker before its
    next model call.
    """
    model_calls = 0
    tool_calls = 0
    _hooks_log = audit_log or NoopAuditLog()
    _ctx = ExecutionContext(session_id=session_id)

    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError

        model_calls += 1
        tool_use_blocks: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        stop_reason = "end_turn"

        if hooks_dir is not None:
            await dispatch_hooks(
                "on_model_call",
                {"session_id": session_id, "model_call_number": model_calls},
                _ctx,
                hooks_dir=hooks_dir,
                audit_log=_hooks_log,
            )

        async for event in model_adapter.call():
            etype = event.get("type", "")
            if etype == "tool_use_start":
                current_tool = {
                    "id": event.get("id", ""),
                    "name": event.get("name", ""),
                    "input_json": "",
                }
                tool_use_blocks.append(current_tool)
            elif etype == "tool_input_delta" and current_tool is not None:
                current_tool["input_json"] += event.get("partial_json", "")
            elif etype == "message_stop":
                stop_reason = event.get("stop_reason") or "end_turn"

        if on_usage_delta is not None:
            on_usage_delta(UsageDelta(input_tokens=100, output_tokens=50))

        if not tool_use_blocks or stop_reason != "tool_use":
            if hooks_dir is not None:
                await dispatch_hooks(
                    "on_stop",
                    {
                        "session_id": session_id,
                        "stop_reason": stop_reason,
                        "model_calls": model_calls,
                        "tool_calls": tool_calls,
                    },
                    _ctx,
                    hooks_dir=hooks_dir,
                    audit_log=_hooks_log,
                )
            break

        for block in tool_use_blocks:
            tool_calls += 1
            _block_input = block.get("input_json", "")
            try:
                _block_args: Any = json.loads(_block_input) if _block_input else {}
            except json.JSONDecodeError:
                _block_args = {}
            if hooks_dir is not None:
                _pre_results = await dispatch_hooks(
                    "pre_tool_call",
                    {
                        "session_id": session_id,
                        "tool_id": block["id"],
                        "tool_name": block["name"],
                        "tool_args": _block_args,
                    },
                    _ctx,
                    hooks_dir=hooks_dir,
                    audit_log=_hooks_log,
                )
                for _pre_r in _pre_results:
                    if _pre_r.mutations and isinstance(_pre_r.mutations.get("args"), dict):
                        _block_args = _pre_r.mutations["args"]
                        break
            result = sandbox_adapter.next_result()
            if hooks_dir is not None:
                await dispatch_hooks(
                    "post_tool_call",
                    {
                        "session_id": session_id,
                        "tool_id": block["id"],
                        "tool_name": block["name"],
                        "tool_result": result.get("content", ""),
                    },
                    _ctx,
                    hooks_dir=hooks_dir,
                    audit_log=_hooks_log,
                )

    return model_calls, tool_calls


async def _run_harness_capturing(
    model_adapter: FakeModelAdapter,
    sandbox_adapter: FakeSandboxAdapter,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Run the agent harness and capture the full event sequence.

    Returns (model_calls, tool_calls, captured_events).  Tool dispatches are
    represented as synthetic {"type": "tool_result", ...} entries so that the
    baseline can detect changes to tool-dispatch ordering or content.
    """
    model_calls = 0
    tool_calls = 0
    captured: list[dict[str, Any]] = []

    while True:
        model_calls += 1
        tool_use_blocks: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        stop_reason = "end_turn"

        async for event in model_adapter.call():
            captured.append(event)
            etype = event.get("type", "")
            if etype == "tool_use_start":
                current_tool = {
                    "id": event.get("id", ""),
                    "name": event.get("name", ""),
                    "input_json": "",
                }
                tool_use_blocks.append(current_tool)
            elif etype == "tool_input_delta" and current_tool is not None:
                current_tool["input_json"] += event.get("partial_json", "")
            elif etype == "message_stop":
                stop_reason = event.get("stop_reason") or "end_turn"

        if not tool_use_blocks or stop_reason != "tool_use":
            break

        for block in tool_use_blocks:
            tool_calls += 1
            result = sandbox_adapter.next_result()
            captured.append(
                {
                    "type": "tool_result",
                    "tool_id": block["id"],
                    "content": result.get("content", ""),
                }
            )

    return model_calls, tool_calls, captured


def _find_divergence(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None] | None:
    """Return (seq, expected_event, actual_event) at the first divergence, or None if equal."""
    for i, (exp, act) in enumerate(zip(expected, actual, strict=False)):
        if exp != act:
            return (i, exp, act)
    if len(expected) != len(actual):
        seq = min(len(expected), len(actual))
        return (
            seq,
            expected[seq] if seq < len(expected) else None,
            actual[seq] if seq < len(actual) else None,
        )
    return None


# ---------------------------------------------------------------------------
# Phase-aware harness run loop
# ---------------------------------------------------------------------------


class HarnessLoopError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="harness_loop_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


async def run_harness_loop(
    session_id: str,
    *,
    model_adapter: FakeModelAdapter,
    sandbox_adapter: FakeSandboxAdapter,
    phase_reader: _PhaseReader,
    audit_log: AuditLog,
    on_usage_delta: Callable[[UsageDelta], None] | None = None,
    hooks_dir: Path | None = None,
    iteration_budget: IterationBudget | None = None,
    event_log: EventLogWriter | None = None,
    model_router: ModelRouter | None = None,
    model_call_opts: ModelCallOpts | None = None,
    max_tokens_policy: MaxTokensPolicy | None = None,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
    tool_capabilities: dict[str, frozenset[str]] | None = None,
    granted_capabilities: frozenset[str] | None = None,
) -> tuple[int, int, str]:
    """Run harness while phase is not in {idle, paused, terminated}.

    Iterates the model-call / tool-dispatch cycle while the session is in an active
    phase. Releases the session (returns) when phase enters the stop set so any
    harness can re-wake. Emits OTel span "harness.run_loop" with session.id attribute
    and logs a structured invocation event on each call. On failure surfaces an error
    message to the caller and writes the failure to the audit log.

    When iteration_budget is provided, a per-iteration check runs after each model call
    that returns tool_use (i.e., would otherwise continue). A hard breach (model_calls
    >= budget.hard) writes a session.phase_change event to terminated with
    reason="budget_exceeded" and stops the loop. A soft breach (model_calls >= budget.soft)
    writes a budget.warning event plus a session.phase_change event to waiting_for_user and
    stops the loop pending user approval. event_log must be provided to persist these
    transitions; without it the loop still stops but no events are written.

    When model_router and model_call_opts are both provided and the current phase is
    "waiting_for_model", the model call uses the router instead of model_adapter:
      1. Dispatches "pre_message" hooks (if hooks_dir is set) before the call.
      2. Streams events from model_router.call(opts) through ModelEventTranslator.
      3. Emits "message.delta" (kind="text") for each TextDeltaEvent.
      4. Emits "message.delta" (kind="thinking") for each ThinkingDeltaEvent.
      5. Emits "model_call.completed" for each MessageStopEvent.

    When the current phase is "waiting_for_tool", one pending tool call is dispatched via
    sandbox_adapter without making a model call:
      1. Dispatches "pre_tool_call" hooks (if hooks_dir is set) before dispatch.
      2. Calls sandbox_adapter.next_result() to obtain the tool result.
      3. Emits a "tool_call.result" event to event_log (if provided) with tool_id and content.
      4. Dispatches "post_tool_call" hooks (if hooks_dir is set) after dispatch.
      5. Continues the loop without incrementing model_calls; the phase_reader drives
         transition back to waiting_for_model.

    When stop_reason is "max_tokens":
      - Partial message chunks were already emitted as "message.delta" events during streaming.
      - Emits a "message.truncated" event to event_log (if provided) with model_call_number.
      - If max_tokens_policy.continue_allowed is True, loops and re-calls the model.
      - Otherwise (policy absent or continue_allowed=False), emits a "session.phase_change"
        event with after="waiting_for_user" and reason="max_tokens", sets final_phase to
        "waiting_for_user", and stops the loop. event_log must be provided to persist these
        events; without it the loop still stops or loops correctly but no events are written.

    When stop_reason is "end_turn":
      1. Emits a "message.appended" event to event_log (if provided) with model_call_number,
         signalling the complete assistant message has been appended to the conversation.
      2. Emits a "session.phase_change" event to event_log (if provided) with after="idle"
         and reason="end_turn".
      3. Dispatches "post_message" hooks (if hooks_dir is set).
      Sets final_phase to "idle" and stops the loop. event_log must be provided to persist
      these events; without it the loop still stops but no events are written. On failure
      surfaces an error message to the caller and writes the failure to the audit log.

    When stop_reason is "tool_use":
      For each tool use block in order:
        1. Schema-validates the tool's JSON args against tool_schemas[name] (if provided) or
           model_call_opts.tools[name].input_schema (if model_call_opts is set).  tool_schemas
           takes precedence; schemas from model_call_opts are used as fallback.  On validation
           failure raises HarnessLoopError which is surfaced to the caller and written to the
           audit log.
        2. Runs a capability intersection check: if tool_capabilities[name] and
           granted_capabilities are both provided, verifies every required capability is
           present in the granted set.  Missing capabilities raise HarnessLoopError.
        3. Dispatches "pre_tool_call" hooks (if hooks_dir is set).
        4. Emits a "tool_call.requested" event to event_log (if provided) with tool_id,
           tool_name, and the parsed args dict.
      Then emits a "session.phase_change" event with after="waiting_for_tool" and
      reason="tool_use" and continues the loop; the waiting_for_tool branch handles the
      actual sandbox dispatch on the next iteration.

    Returns (model_calls, tool_calls, final_phase).
    """
    now = _now()
    tracer = get_tracer()
    _ctx = ExecutionContext(session_id=session_id)

    with tracer.start_as_current_span(
        "harness.run_loop",
        attributes={"session.id": session_id},
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="harness.run_loop.invocation",
                code="harness_run_loop",
                timestamp=now,
            ),
        )

        model_calls = 0
        tool_calls = 0
        final_phase = "created"

        try:
            while True:
                final_phase = phase_reader.current_phase(session_id)
                if final_phase in _STOP_PHASES:
                    break  # release — any harness can re-wake

                if final_phase == "waiting_for_tool":
                    tool_calls += 1
                    result = sandbox_adapter.next_result()
                    tool_id = result.get("tool_id", "")
                    tool_name = result.get("tool_name", "")
                    _content = result.get("content", "")
                    if event_log is not None:
                        await event_log.append(
                            session_id,
                            "tool_call.result",
                            {
                                "tool_id": tool_id,
                                "content": _content,
                            },
                        )
                        # Emit a canvas_op event so the UI can rebuild canvas state
                        # from the event log on page reload (replay semantics).
                        _canvas_op_data = _extract_canvas_op(_content)
                        if _canvas_op_data is not None:
                            await event_log.append(
                                session_id,
                                "canvas_op",
                                _canvas_op_data,
                            )
                    if hooks_dir is not None:
                        await dispatch_hooks(
                            "post_tool_call",
                            {
                                "session_id": session_id,
                                "tool_id": tool_id,
                                "tool_name": tool_name,
                                "tool_result": result.get("content", ""),
                            },
                            _ctx,
                            hooks_dir=hooks_dir,
                            audit_log=audit_log,
                        )
                    continue

                model_calls += 1
                tool_use_blocks: list[dict[str, Any]] = []
                current_tool: dict[str, Any] | None = None
                stop_reason = "end_turn"
                _stop_event: MessageStopEvent | None = None

                _model_call_exc: Exception | None = None
                _start_event: MessageStartEvent | None = None

                if (
                    final_phase == "waiting_for_model"
                    and model_router is not None
                    and model_call_opts is not None
                ):
                    if hooks_dir is not None:
                        await dispatch_hooks(
                            "pre_message",
                            {"session_id": session_id, "model_call_number": model_calls},
                            _ctx,
                            hooks_dir=hooks_dir,
                            audit_log=audit_log,
                        )
                    opts = model_call_opts.model_copy(update={"session_id": session_id})
                    _translator = ModelEventTranslator(model_call_number=model_calls)
                    try:
                        async for event in model_router.call(opts):
                            pairs = _translator.translate(event)
                            if event_log is not None:
                                for _etype, _edata in pairs:
                                    await event_log.append(session_id, _etype, _edata)
                    except Exception as _exc:
                        _model_call_exc = _exc
                    tool_use_blocks = _translator.tool_blocks
                    stop_reason = _translator.stop_reason
                    _stop_event = _translator.stop_event
                    _start_event = _translator.start_event
                else:
                    if hooks_dir is not None:
                        await dispatch_hooks(
                            "on_model_call",
                            {"session_id": session_id, "model_call_number": model_calls},
                            _ctx,
                            hooks_dir=hooks_dir,
                            audit_log=audit_log,
                        )
                    try:
                        async for event in model_adapter.call():
                            etype = event.get("type", "")
                            if etype == "tool_use_start":
                                current_tool = {
                                    "id": event.get("id", ""),
                                    "name": event.get("name", ""),
                                    "input_json": "",
                                }
                                tool_use_blocks.append(current_tool)
                            elif etype == "tool_input_delta" and current_tool is not None:
                                current_tool["input_json"] += event.get("partial_json", "")
                            elif etype == "message_stop":
                                stop_reason = event.get("stop_reason") or "end_turn"
                    except Exception as _exc:
                        _model_call_exc = _exc

                if _model_call_exc is not None:
                    _err_now = _now()
                    with tracer.start_as_current_span(
                        "harness.model_call_error",
                        attributes={"session.id": session_id},
                    ) as _err_span:
                        record_invocation_event(
                            _err_span,
                            StructuredEvent(
                                name="harness.model_call_error.invocation",
                                code="harness_model_call_error",
                                timestamp=_err_now,
                            ),
                        )
                        _recoverable = False
                        if hooks_dir is not None:
                            try:
                                _hook_results = await dispatch_hooks(
                                    "on_error",
                                    {
                                        "session_id": session_id,
                                        "error": str(_model_call_exc),
                                        "error_type": type(_model_call_exc).__name__,
                                    },
                                    _ctx,
                                    hooks_dir=hooks_dir,
                                    audit_log=audit_log,
                                )
                                _recoverable = any(
                                    r.verdict == "recoverable" for r in _hook_results
                                )
                            except Exception as _hook_exc:
                                _err = HarnessLoopError(
                                    message=f"on_error hook dispatch failed for session {session_id!r}: {_hook_exc}",
                                    timestamp=_err_now,
                                    cause=_hook_exc,
                                )
                                record_error(_err_span, _err)
                                audit_log.write(
                                    AuditLogEntry(
                                        level="error",
                                        event="harness.model_call_error.failed",
                                        code=_err.code,
                                        timestamp=_err.timestamp,
                                        detail={
                                            "session_id": session_id,
                                            "message": _err.message,
                                        },
                                    )
                                )
                                raise _err
                    if not _recoverable:
                        final_phase = "terminated"
                        break
                    continue  # recoverable: retry model call on next iteration

                if on_usage_delta is not None:
                    if _stop_event is not None:
                        on_usage_delta(UsageDelta(
                            input_tokens=_stop_event.input_tokens or 0,
                            output_tokens=_stop_event.output_tokens or 0,
                            cache_creation_tokens=_stop_event.cache_creation_input_tokens,
                            cache_read_tokens=_stop_event.cache_read_input_tokens,
                        ))
                    else:
                        on_usage_delta(UsageDelta(input_tokens=100, output_tokens=50))

                if event_log is not None and _stop_event is not None:
                    _delta_data: dict[str, Any] = {
                        "prompt_tokens": _stop_event.input_tokens or 0,
                        "completion_tokens": _stop_event.output_tokens or 0,
                        "cache_creation_tokens": _stop_event.cache_creation_input_tokens,
                        "cache_read_tokens": _stop_event.cache_read_input_tokens,
                    }
                    if _start_event is not None:
                        _delta_data["model"] = _start_event.model
                        _delta_data["provider"] = _start_event.provider
                    await event_log.append(session_id, "usage.delta", _delta_data)

                if stop_reason == "max_tokens":
                    if event_log is not None:
                        await event_log.append(
                            session_id,
                            "message.truncated",
                            {"model_call_number": model_calls, "timestamp": _now()},
                        )
                    if max_tokens_policy is not None and max_tokens_policy.continue_allowed:
                        continue
                    if event_log is not None:
                        await event_log.append(
                            session_id,
                            "session.phase_change",
                            {
                                "before": final_phase,
                                "after": "waiting_for_user",
                                "timestamp": _now(),
                                "reason": "max_tokens",
                            },
                        )
                    final_phase = "waiting_for_user"
                    break

                if not tool_use_blocks or stop_reason != "tool_use":
                    before_phase = final_phase
                    final_phase = "idle"
                    if stop_reason == "end_turn":
                        # 1. Append final message.
                        if event_log is not None:
                            await event_log.append(
                                session_id,
                                "message.appended",
                                {"model_call_number": model_calls, "timestamp": _now()},
                            )
                        # 2. Transition phase to idle.
                        if event_log is not None:
                            await event_log.append(
                                session_id,
                                "session.phase_change",
                                {
                                    "before": before_phase,
                                    "after": "idle",
                                    "timestamp": _now(),
                                    "reason": "end_turn",
                                },
                            )
                        # 3. Run post_message hooks.
                        if hooks_dir is not None:
                            await dispatch_hooks(
                                "post_message",
                                {
                                    "session_id": session_id,
                                    "model_call_number": model_calls,
                                },
                                _ctx,
                                hooks_dir=hooks_dir,
                                audit_log=audit_log,
                            )
                    else:
                        if hooks_dir is not None:
                            await dispatch_hooks(
                                "on_stop",
                                {
                                    "session_id": session_id,
                                    "stop_reason": stop_reason,
                                    "model_calls": model_calls,
                                    "tool_calls": tool_calls,
                                },
                                _ctx,
                                hooks_dir=hooks_dir,
                                audit_log=audit_log,
                            )
                    break  # release session

                # Per-iteration budget check: only reached when tool_use would continue.
                if iteration_budget is not None:
                    before_phase = final_phase
                    if (
                        iteration_budget.hard is not None
                        and model_calls >= iteration_budget.hard
                    ):
                        if event_log is not None:
                            await event_log.append(
                                session_id,
                                "session.phase_change",
                                {
                                    "before": before_phase,
                                    "after": "terminated",
                                    "timestamp": _now(),
                                    "reason": "budget_exceeded",
                                },
                            )
                        final_phase = "terminated"
                        break
                    elif (
                        iteration_budget.soft is not None
                        and model_calls >= iteration_budget.soft
                    ):
                        if event_log is not None:
                            await event_log.append(
                                session_id,
                                "budget.warning",
                                {
                                    "model_calls": model_calls,
                                    "budget_soft": iteration_budget.soft,
                                    "timestamp": _now(),
                                },
                            )
                            await event_log.append(
                                session_id,
                                "session.phase_change",
                                {
                                    "before": before_phase,
                                    "after": "waiting_for_user",
                                    "timestamp": _now(),
                                    "reason": "budget_warning",
                                },
                            )
                        final_phase = "waiting_for_user"
                        break

                # Build schema map: model_call_opts.tools as base, tool_schemas overrides.
                _schema_map: dict[str, dict[str, Any]] = {}
                if model_call_opts is not None:
                    for _td in model_call_opts.tools:
                        _schema_map[_td.name] = _td.input_schema
                if tool_schemas is not None:
                    _schema_map.update(tool_schemas)

                for block in tool_use_blocks:
                    _tool_name = block["name"]
                    _tool_id = block["id"]
                    _input_json = block["input_json"]

                    # 1. Schema-validate args per tool.
                    try:
                        _args: Any = json.loads(_input_json) if _input_json else {}
                    except json.JSONDecodeError as _jex:
                        raise ValueError(
                            f"Tool {_tool_name!r} ({_tool_id!r}) args are not valid JSON: {_jex}"
                        ) from _jex
                    _schema = _schema_map.get(_tool_name)
                    if _schema is not None:
                        _schema_errs = _validate_tool_args(_schema, _args)
                        if _schema_errs:
                            raise ValueError(
                                f"Tool {_tool_name!r} ({_tool_id!r}) args failed schema "
                                f"validation: {_schema_errs[0]}"
                            )

                    # 2. Capability intersection check.
                    if tool_capabilities is not None and granted_capabilities is not None:
                        _required_caps = tool_capabilities.get(_tool_name, frozenset())
                        _missing_caps = _required_caps - granted_capabilities
                        if _missing_caps:
                            raise ValueError(
                                f"Capability denied for tool {_tool_name!r} ({_tool_id!r}): "
                                f"missing {sorted(_missing_caps)}"
                            )

                    # 3. pre_tool_call hooks — verdict round-trip (Contract 3).
                    if hooks_dir is not None:
                        try:
                            _pre_hook_results = await dispatch_hooks(
                                "pre_tool_call",
                                {
                                    "session_id": session_id,
                                    "tool_id": _tool_id,
                                    "tool_name": _tool_name,
                                    "tool_args": _args,
                                },
                                _ctx,
                                hooks_dir=hooks_dir,
                                audit_log=audit_log,
                            )
                            for _phr in _pre_hook_results:
                                if _phr.mutations and isinstance(
                                    _phr.mutations.get("args"), dict
                                ):
                                    _args = _phr.mutations["args"]
                                    break
                        except HookVetoError as _veto:
                            _veto_now = _now()
                            if event_log is not None:
                                await event_log.append(
                                    session_id,
                                    "tool_call.vetoed",
                                    {
                                        "tool_id": _tool_id,
                                        "tool_name": _tool_name,
                                        "reason": _veto.message,
                                    },
                                )
                            audit_log.write(
                                AuditLogEntry(
                                    level="info",
                                    event="hook.pre_tool_call.vetoed",
                                    code="hook_veto",
                                    timestamp=_veto_now,
                                    detail={
                                        "session_id": session_id,
                                        "tool_id": _tool_id,
                                        "tool_name": _tool_name,
                                        "reason": _veto.message,
                                    },
                                )
                            )
                            raise HarnessLoopError(
                                message=f"Tool call {_tool_name!r} vetoed by hook"
                                f" {_veto.hook_name!r}: {_veto.message}",
                                timestamp=_veto_now,
                                cause=_veto,
                            )

                    # 4. Write tool_call.requested event.
                    if event_log is not None:
                        await event_log.append(
                            session_id,
                            "tool_call.requested",
                            {
                                "tool_id": _tool_id,
                                "tool_name": _tool_name,
                                "args": _args,
                            },
                        )

                # 5. Transition to waiting_for_tool; actual dispatch happens on the next
                # iteration via the waiting_for_tool branch.
                if event_log is not None:
                    await event_log.append(
                        session_id,
                        "session.phase_change",
                        {
                            "before": final_phase,
                            "after": "waiting_for_tool",
                            "timestamp": _now(),
                            "reason": "tool_use",
                        },
                    )
                final_phase = "waiting_for_tool"
                continue

        except HarnessLoopError:
            raise
        except Exception as exc:
            err = HarnessLoopError(
                message=f"Harness loop failed for session {session_id!r}: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="harness.run_loop.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "session_id": session_id,
                        "message": err.message,
                    },
                )
            )
            raise err

    return model_calls, tool_calls, final_phase


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_replay_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/replay")
    async def replay_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        run_id = str(uuid.uuid4())
        fixture_dir = storage_root / "fixtures" / session_id

        with tracer.start_as_current_span(
            "replay.run",
            attributes={
                "session.id": session_id,
                "replay.run_id": run_id,
                "replay.fixture_dir": str(fixture_dir),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="replay.run.invocation",
                    code="replay_run",
                    timestamp=now,
                ),
            )

            model_fixture = fixture_dir / "model_responses.ndjson"
            if not model_fixture.exists():
                err = ReplayError(
                    message=f"Fixture not found for session {session_id!r}: {model_fixture}",
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="replay.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            try:
                model_adapter = FakeModelAdapter(model_fixture)
                sandbox_adapter = FakeSandboxAdapter(fixture_dir / "tool_responses.ndjson")
                expected_path = fixture_dir / "expected_events.ndjson"

                if expected_path.exists():
                    expected_events = [
                        json.loads(line)
                        for line in expected_path.read_text().splitlines()
                        if line.strip()
                    ]
                    model_calls, tool_calls, actual_events = await _run_harness_capturing(
                        model_adapter, sandbox_adapter
                    )
                    divergence = _find_divergence(expected_events, actual_events)
                    if divergence is not None:
                        seq, exp_ev, act_ev = divergence
                        err = ReplayError(
                            message=(
                                f"Replay diverged at event seq {seq} for session {session_id!r}"
                            ),
                            timestamp=_now(),
                        )
                        record_error(span, err)
                        audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="replay.run.failed",
                                code=err.code,
                                timestamp=err.timestamp,
                                detail={
                                    "session_id": session_id,
                                    "run_id": run_id,
                                    "first_deviating_seq": seq,
                                    "expected_event": exp_ev,
                                    "actual_event": act_ev,
                                    "message": err.message,
                                },
                            )
                        )
                        raise err
                else:
                    model_calls, tool_calls = await _run_harness(model_adapter, sandbox_adapter)
            except ReplayError:
                raise
            except Exception as exc:
                err = ReplayError(
                    message=f"Replay failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="replay.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "run_id": run_id,
                "session_id": session_id,
                "model_call_count": model_calls,
                "tool_call_count": tool_calls,
                "status": "completed",
            }
        )

    return router
