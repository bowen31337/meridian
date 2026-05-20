from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
    MessageDeltaEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelRouter,
    TextDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)
from sdk_sandbox import ExecutionContext
from storage_event_log import EventLogWriter

from ._hook_dispatch import dispatch_hooks


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


# Phases in which the harness releases the session so any harness can re-wake.
_STOP_PHASES: frozenset[str] = frozenset({"idle", "paused", "terminated"})


@runtime_checkable
class _PhaseReader(Protocol):
    def current_phase(self, session_id: str) -> str: ...


__all__ = [
    "FakeModelAdapter",
    "FakeSandboxAdapter",
    "HarnessLoopError",
    "IterationBudget",
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
            if hooks_dir is not None:
                await dispatch_hooks(
                    "pre_tool_call",
                    {
                        "session_id": session_id,
                        "tool_id": block["id"],
                        "tool_name": block["name"],
                    },
                    _ctx,
                    hooks_dir=hooks_dir,
                    audit_log=_hooks_log,
                )
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
      2. Streams events from model_router.call(opts).
      3. Emits a "message.delta" event to event_log for each TextDeltaEvent received.

    When the current phase is "waiting_for_tool", one pending tool call is dispatched via
    sandbox_adapter without making a model call:
      1. Dispatches "pre_tool_call" hooks (if hooks_dir is set) before dispatch.
      2. Calls sandbox_adapter.next_result() to obtain the tool result.
      3. Emits a "tool_call.result" event to event_log (if provided) with tool_id and content.
      4. Dispatches "post_tool_call" hooks (if hooks_dir is set) after dispatch.
      5. Continues the loop without incrementing model_calls; the phase_reader drives
         transition back to waiting_for_model.

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
                    if hooks_dir is not None:
                        await dispatch_hooks(
                            "pre_tool_call",
                            {
                                "session_id": session_id,
                                "tool_id": tool_id,
                                "tool_name": tool_name,
                            },
                            _ctx,
                            hooks_dir=hooks_dir,
                            audit_log=audit_log,
                        )
                    if event_log is not None:
                        await event_log.append(
                            session_id,
                            "tool_call.result",
                            {
                                "tool_id": tool_id,
                                "content": result.get("content", ""),
                            },
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
                    try:
                        async for event in model_router.call(opts):
                            if isinstance(event, TextDeltaEvent):
                                if event_log is not None:
                                    await event_log.append(
                                        session_id,
                                        "message.delta",
                                        {"text": event.text, "model_call_number": model_calls},
                                    )
                            elif isinstance(event, ToolUseStartEvent):
                                current_tool = {
                                    "id": event.id,
                                    "name": event.name,
                                    "input_json": "",
                                }
                                tool_use_blocks.append(current_tool)
                            elif isinstance(event, ToolInputDeltaEvent) and current_tool is not None:
                                if current_tool["id"] == event.id:
                                    current_tool["input_json"] += event.partial_json
                            elif isinstance(event, MessageStopEvent):
                                if event.stop_reason is not None:
                                    stop_reason = event.stop_reason
                                _stop_event = event
                            elif isinstance(event, MessageDeltaEvent):
                                if event.stop_reason is not None:
                                    stop_reason = event.stop_reason
                    except Exception as _exc:
                        _model_call_exc = _exc
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
                    await event_log.append(
                        session_id,
                        "usage.delta",
                        {
                            "prompt_tokens": _stop_event.input_tokens or 0,
                            "completion_tokens": _stop_event.output_tokens or 0,
                            "cache_creation_tokens": _stop_event.cache_creation_input_tokens,
                            "cache_read_tokens": _stop_event.cache_read_input_tokens,
                        },
                    )

                if not tool_use_blocks or stop_reason != "tool_use":
                    final_phase = "idle"
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
                    break  # end_turn: release session

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

                for block in tool_use_blocks:
                    tool_calls += 1
                    if hooks_dir is not None:
                        await dispatch_hooks(
                            "pre_tool_call",
                            {
                                "session_id": session_id,
                                "tool_id": block["id"],
                                "tool_name": block["name"],
                            },
                            _ctx,
                            hooks_dir=hooks_dir,
                            audit_log=audit_log,
                        )
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
                            audit_log=audit_log,
                        )

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
