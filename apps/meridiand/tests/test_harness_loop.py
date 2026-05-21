"""
Harness run loop conformance suite.

Tests cover:
  - run_harness_loop exits immediately when phase is "idle" (model_calls=0).
  - run_harness_loop exits immediately when phase is "paused" (model_calls=0).
  - run_harness_loop exits immediately when phase is "terminated" (model_calls=0).
  - run_harness_loop runs when phase is "created" (default — no phase_change events).
  - run_harness_loop runs when phase is "running".
  - Final phase is "idle" after model returns end_turn.
  - model_calls=1, tool_calls=0 for a single end_turn call.
  - model_calls=2, tool_calls=1 for a tool_use call followed by end_turn.
  - Loop exits mid-run when phase transitions to a stop phase between iterations.
  - final_phase is "paused" when an external pause interrupts after tool dispatch.
  - final_phase is "terminated" when an external terminate interrupts after tool dispatch.
  - OTel span "harness.run_loop" is emitted on success.
  - OTel span has session.id attribute.
  - OTel span carries a structured invocation event on each call.
  - OTel span is set to ERROR status on failure.
  - Audit log entry is written on failure with event "harness.run_loop.failed".
  - Audit log detail includes session_id on failure.
  - Audit log detail includes message on failure.
  - Error message is surfaced to the caller (HarnessLoopError raised).
  - HarnessLoopError has code "harness_loop_failed".
  - on_usage_delta callback is called once per model call.
  - release: returns cleanly so another harness can re-wake the session.
  Per-iteration budget check:
  - Hard breach (model_calls >= hard) transitions final_phase to "terminated".
  - Hard breach writes session.phase_change event with reason "budget_exceeded" to event log.
  - Hard breach stops the loop (model_calls does not exceed hard limit).
  - Soft breach (model_calls >= soft, < hard) emits budget.warning event to event log.
  - Soft breach transitions final_phase to "waiting_for_user".
  - Soft breach writes session.phase_change event with after "waiting_for_user" to event log.
  - Soft breach stops the loop (model_calls does not exceed soft limit).
  - No breach below soft threshold: loop runs to completion normally.
  - Hard breach without event_log: still transitions final_phase to "terminated".
  - Soft breach without event_log: still transitions final_phase to "waiting_for_user".
  - Event log failure during budget check raises HarnessLoopError.
  - Event log failure during budget check writes failure to audit log.
  waiting_for_model branch (model_router path):
  - message.delta events emitted to event_log per TextDeltaEvent chunk.
  - message.delta data contains the correct text from each chunk.
  - Multiple text chunks produce multiple message.delta events in order.
  - model_calls incremented when router path is used.
  - No message.delta emitted when event_log is None.
  - pre_message hook dispatched when hooks_dir is set.
  - on_model_call hook NOT dispatched on router path (pre_message used instead).
  - Tool use blocks collected from router ToolUseStartEvent/ToolInputDeltaEvent.
  - stop_reason parsed from MessageDeltaEvent and MessageStopEvent.
  - Router failure (no hooks) transitions final_phase to "terminated" without raising.
  - Without model_router, waiting_for_model phase falls back to fake adapter.
  waiting_for_tool branch (sandbox dispatch path):
  - tool_call.result event emitted to event_log per tool dispatched.
  - tool_call.result data contains content from sandbox result.
  - tool_call.result data contains tool_id from sandbox result.
  - Multiple waiting_for_tool phases produce multiple tool_call.result events in sequence.
  - No tool_call.result event emitted when event_log is None.
  - model_calls not incremented during waiting_for_tool (no model call made).
  - tool_calls incremented once per waiting_for_tool iteration.
  - Final phase is idle after tool dispatch followed by model end_turn.
  - pre_tool_call hook dispatched when hooks_dir is set.
  - post_tool_call hook dispatched when hooks_dir is set.
  - Sandbox failure raises HarnessLoopError (error surfaced to caller).
  - Sandbox failure writes harness.run_loop.failed to audit log.
  - Sandbox failure audit detail includes session_id.
  - Sandbox failure error message is non-empty.
  On model call error (router and fake adapter paths):
  - on_error hooks dispatched when hooks_dir is set.
  - on_error hook payload includes session_id, error, and error_type.
  - No on_error hooks dispatched when hooks_dir is None.
  - Hook returning "recoverable" verdict allows loop to continue.
  - Hook NOT returning "recoverable" transitions final_phase to "terminated".
  - OTel span "harness.model_call_error" emitted on model call error.
  - Span has session.id attribute.
  - Span carries structured invocation event (meridian.error.invocation).
  - on_error hook dispatch failure raises HarnessLoopError.
  - on_error hook dispatch failure writes harness.model_call_error.failed to audit log.
  - Audit detail includes session_id on hook dispatch failure.
  - Audit detail includes message on hook dispatch failure.
  - Fake adapter model call error transitions to "terminated" (no hooks).
  - Fake adapter model call error dispatches on_error hooks when hooks_dir is set.
  On stop_reason end_turn:
  - Emits message.appended event to event_log with model_call_number.
  - message.appended event has model_call_number field.
  - No message.appended event emitted when event_log is None.
  - Emits session.phase_change event with after="idle" and reason="end_turn".
  - session.phase_change has reason="end_turn".
  - No session.phase_change event emitted when event_log is None.
  - post_message hook dispatched when hooks_dir is set.
  - post_message hook NOT dispatched when hooks_dir is None.
  - Event log failure raises HarnessLoopError surfaced to caller.
  - Event log failure writes harness.run_loop.failed to audit log.
  - post_message hook dispatch failure raises HarnessLoopError surfaced to caller.
  - post_message hook dispatch failure writes harness.run_loop.failed to audit log.
  On stop_reason max_tokens:
  - Without policy (None): transitions final_phase to "waiting_for_user".
  - With continue_allowed=False: transitions final_phase to "waiting_for_user".
  - With continue_allowed=True: loop continues (model_calls incremented again).
  - Emits message.truncated event to event_log with model_call_number.
  - No message.truncated event emitted when event_log is None.
  - Emits session.phase_change event with after="waiting_for_user" and reason="max_tokens".
  - session.phase_change event not emitted when continue_allowed=True (loop continues).
  - Without event_log: still transitions to waiting_for_user when not continuing.
  - Without event_log: still loops when continue_allowed=True.
  - Event log failure on message.truncated raises HarnessLoopError.
  - Event log failure on message.truncated writes harness.run_loop.failed to audit log.
  On stop_reason tool_use:
  - tool_call.requested event written to event_log per tool block.
  - tool_call.requested data contains tool_id, tool_name, and parsed args.
  - No tool_call.requested event emitted when event_log is None.
  - session.phase_change event written with after="waiting_for_tool" and reason="tool_use".
  - session.phase_change event not emitted when event_log is None.
  - pre_tool_call hook dispatched per tool block (before tool_call.requested).
  - pre_tool_call not dispatched in waiting_for_tool branch (already fired on tool_use).
  - Schema-validates args per tool when tool_schemas provided.
  - Schema validation failure raises HarnessLoopError surfaced to caller.
  - Schema validation failure writes harness.run_loop.failed to audit log.
  - Invalid JSON args raises HarnessLoopError surfaced to caller.
  - Invalid JSON args writes harness.run_loop.failed to audit log.
  - Schemas from model_call_opts.tools used when tool_schemas not set.
  - tool_schemas takes precedence over model_call_opts.tools for same tool name.
  - Capability intersection check: all required caps present passes through.
  - Capability intersection check: missing cap raises HarnessLoopError surfaced to caller.
  - Capability intersection check: missing cap writes harness.run_loop.failed to audit log.
  - No capability check when tool_capabilities is None.
  - No capability check when granted_capabilities is None.
  Contract 4 — event translation (model_router path):
  - ThinkingDeltaEvent emits message.delta with kind="thinking".
  - Thinking delta data["thinking"] carries the thinking text.
  - Thinking delta data["model_call_number"] is correct.
  - No thinking delta event emitted when event_log is None.
  - model_call.completed event written to event_log on MessageStopEvent.
  - model_call.completed data["stop_reason"] matches the stream stop reason.
  - model_call.completed data["input_tokens"] and data["output_tokens"] are correct.
  - model_call.completed data["model_call_number"] matches the call number.
  - No model_call.completed event emitted when event_log is None.
  - text message.delta now includes data["kind"] == "text".
  - Mixed thinking + text deltas produce message.delta rows in order.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from meridiand._audit import FileAuditLog
from meridiand._replay import (
    FakeModelAdapter,
    FakeSandboxAdapter,
    HarnessLoopError,
    IterationBudget,
    MaxTokensPolicy,
    UsageDelta,
    run_harness_loop,
)
from meridian_sdk_provider import (
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolInputDeltaEvent,
    ToolUseStartEvent,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Fake phase reader
# ---------------------------------------------------------------------------


class _FakePhaseReader:
    """Returns successive phase values from a pre-configured list."""

    def __init__(self, phases: list[str]) -> None:
        self._phases = phases
        self._idx = 0

    def current_phase(self, session_id: str) -> str:
        phase = self._phases[min(self._idx, len(self._phases) - 1)]
        self._idx += 1
        return phase


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _end_turn_call() -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "text_delta", "text": "Done."},
        {"type": "message_stop", "stop_reason": "end_turn"},
    ]


def _tool_use_call(tool_id: str = "tu_1", tool_name: str = "bash") -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "tool_use_start", "id": tool_id, "name": tool_name},
        {"type": "tool_input_delta", "id": tool_id, "partial_json": '{"cmd":"ls"}'},
        {"type": "message_stop", "stop_reason": "tool_use"},
    ]


def _write_model_fixture(fixture_dir: Path, calls: list[list[dict[str, Any]]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(events) for events in calls]
    (fixture_dir / "model_responses.ndjson").write_text("\n".join(lines) + "\n")


def _write_tool_fixture(fixture_dir: Path, results: list[dict[str, Any]]) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in results]
    (fixture_dir / "tool_responses.ndjson").write_text("\n".join(lines) + "\n")


def _adapters(
    fixture_dir: Path,
    model_calls: list[list[dict[str, Any]]],
    tool_results: list[dict[str, Any]] | None = None,
) -> tuple[FakeModelAdapter, FakeSandboxAdapter]:
    _write_model_fixture(fixture_dir, model_calls)
    if tool_results is not None:
        _write_tool_fixture(fixture_dir, tool_results)
    return (
        FakeModelAdapter(fixture_dir / "model_responses.ndjson"),
        FakeSandboxAdapter(fixture_dir / "tool_responses.ndjson"),
    )


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Tests: stop on pre-existing stop phases
# ---------------------------------------------------------------------------


class TestHarnessLoopStopPhases:
    def test_idle_phase_stops_loop_immediately(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["idle"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 0

    def test_paused_phase_stops_loop_immediately(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["paused"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 0

    def test_terminated_phase_stops_loop_immediately(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["terminated"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 0

    def test_idle_returns_zero_tool_calls(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["idle"])
        audit = FileAuditLog(tmp_path)
        _, tool_calls, _ = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert tool_calls == 0

    def test_idle_returns_idle_final_phase(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["idle"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert final_phase == "idle"

    def test_paused_returns_paused_final_phase(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["paused"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert final_phase == "paused"

    def test_terminated_returns_terminated_final_phase(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["terminated"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert final_phase == "terminated"


# ---------------------------------------------------------------------------
# Tests: loop runs on active phases
# ---------------------------------------------------------------------------


class TestHarnessLoopActivePhases:
    def test_created_phase_runs_model(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 1

    def test_running_phase_runs_model(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 1

    def test_end_turn_sets_final_phase_to_idle(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert final_phase == "idle"

    def test_single_end_turn_model_calls_one_tool_calls_zero(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, _ = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 1
        assert tool_calls == 0

    def test_tool_use_then_end_turn_model_calls_two_tool_calls_one(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["created", "waiting_for_tool", "created"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, _ = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 2
        assert tool_calls == 1

    def test_tool_use_end_turn_final_phase_idle(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["created", "waiting_for_tool", "created"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert final_phase == "idle"


# ---------------------------------------------------------------------------
# Tests: mid-run phase interruption
# ---------------------------------------------------------------------------


class TestHarnessLoopMidRunInterruption:
    def test_paused_after_tool_dispatch_stops_loop(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        # "running": model call returns tool_use → validate/events/phase_change(waiting_for_tool)
        # "waiting_for_tool": dispatch tool, then "paused" stops the loop
        reader = _FakePhaseReader(["running", "waiting_for_tool", "paused"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 1
        assert tool_calls == 1
        assert final_phase == "paused"

    def test_terminated_after_tool_dispatch_stops_loop(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "terminated"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 1
        assert tool_calls == 1
        assert final_phase == "terminated"

    def test_idle_after_tool_dispatch_stops_loop(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "idle"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert model_calls == 1
        assert tool_calls == 1
        assert final_phase == "idle"


# ---------------------------------------------------------------------------
# Tests: usage delta callback
# ---------------------------------------------------------------------------


class TestHarnessLoopUsageDelta:
    def test_on_usage_delta_called_once_per_model_call(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        deltas: list[UsageDelta] = []
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                on_usage_delta=deltas.append,
            )
        )
        assert len(deltas) == 1

    def test_on_usage_delta_called_twice_for_two_model_calls(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["created", "waiting_for_tool", "created"])
        audit = FileAuditLog(tmp_path)
        deltas: list[UsageDelta] = []
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                on_usage_delta=deltas.append,
            )
        )
        assert len(deltas) == 2

    def test_on_usage_delta_not_called_when_phase_stops_immediately(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["idle"])
        audit = FileAuditLog(tmp_path)
        deltas: list[UsageDelta] = []
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                on_usage_delta=deltas.append,
            )
        )
        assert len(deltas) == 0


# ---------------------------------------------------------------------------
# Tests: OTel instrumentation
# ---------------------------------------------------------------------------


class TestHarnessLoopOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_harness_run_loop_span(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "harness.run_loop" in span_names

    def test_span_has_session_id_attribute(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop("sess-otel-1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.run_loop")
        assert span is not None
        assert span.attributes["session.id"] == "sess-otel-1"

    def test_span_carries_invocation_event(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.run_loop")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_span_emitted_on_stop_phase(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["idle"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "harness.run_loop" in span_names

    def test_failure_span_has_error_status(self, tmp_path: Path) -> None:
        from opentelemetry.trace import StatusCode

        class _ExplodingPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise RuntimeError("phase reader broken")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                                 phase_reader=_ExplodingPhaseReader(), audit_log=audit)
            )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        loop_span = spans.get("harness.run_loop")
        assert loop_span is not None
        assert loop_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Tests: error handling and audit log
# ---------------------------------------------------------------------------


class TestHarnessLoopErrorHandling:
    def test_raises_harness_loop_error_on_failure(self, tmp_path: Path) -> None:
        class _BrokenPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise ValueError("storage error")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                                 phase_reader=_BrokenPhaseReader(), audit_log=audit)
            )

    def test_error_code_is_harness_loop_failed(self, tmp_path: Path) -> None:
        class _BrokenPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise ValueError("storage error")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError) as exc_info:
            asyncio.run(
                run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                                 phase_reader=_BrokenPhaseReader(), audit_log=audit)
            )
        assert exc_info.value.code == "harness_loop_failed"

    def test_error_message_is_surfaced(self, tmp_path: Path) -> None:
        class _BrokenPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise ValueError("storage error")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError) as exc_info:
            asyncio.run(
                run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                                 phase_reader=_BrokenPhaseReader(), audit_log=audit)
            )
        assert len(exc_info.value.message) > 0

    def test_failure_writes_audit_log(self, tmp_path: Path) -> None:
        class _BrokenPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise OSError("disk full")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                                 phase_reader=_BrokenPhaseReader(), audit_log=audit)
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_audit_log_level_is_error(self, tmp_path: Path) -> None:
        class _BrokenPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise OSError("disk full")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                                 phase_reader=_BrokenPhaseReader(), audit_log=audit)
            )
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.run_loop.failed")
        assert record["level"] == "error"

    def test_audit_detail_has_session_id(self, tmp_path: Path) -> None:
        class _BrokenPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise OSError("disk full")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop("sess-audit-1", model_adapter=model,
                                 sandbox_adapter=sandbox,
                                 phase_reader=_BrokenPhaseReader(), audit_log=audit)
            )
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.run_loop.failed")
        assert record["detail"]["session_id"] == "sess-audit-1"

    def test_audit_detail_has_message(self, tmp_path: Path) -> None:
        class _BrokenPhaseReader:
            def current_phase(self, session_id: str) -> str:
                raise OSError("disk full")

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                                 phase_reader=_BrokenPhaseReader(), audit_log=audit)
            )
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.run_loop.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Tests: session release (re-wake contract)
# ---------------------------------------------------------------------------


class TestHarnessLoopRelease:
    def test_returns_cleanly_on_idle_for_re_wake(self, tmp_path: Path) -> None:
        """Session released on idle so any harness can re-wake without exception."""
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["idle"])
        audit = FileAuditLog(tmp_path)
        result = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert result is not None  # no exception raised; session cleanly released

    def test_returns_cleanly_on_paused_for_re_wake(self, tmp_path: Path) -> None:
        """Session released on paused so any harness can re-wake without exception."""
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["paused"])
        audit = FileAuditLog(tmp_path)
        result = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert result is not None

    def test_returns_cleanly_on_terminated_for_re_wake(self, tmp_path: Path) -> None:
        """Session released on terminated so any harness can re-wake without exception."""
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["terminated"])
        audit = FileAuditLog(tmp_path)
        result = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        assert result is not None

    def test_returns_tuple_of_three_on_release(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["idle"])
        audit = FileAuditLog(tmp_path)
        result = asyncio.run(
            run_harness_loop("s1", model_adapter=model, sandbox_adapter=sandbox,
                             phase_reader=reader, audit_log=audit)
        )
        model_calls, tool_calls, final_phase = result
        assert isinstance(model_calls, int)
        assert isinstance(tool_calls, int)
        assert isinstance(final_phase, str)


# ---------------------------------------------------------------------------
# Helpers: fake event log writer
# ---------------------------------------------------------------------------


class _FakeEventLogWriter:
    """In-memory EventLogWriter that records appended events for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []
        self._seq = 0

    async def append(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> int:
        self.events.append((session_id, event_type, data))
        seq = self._seq
        self._seq += 1
        return seq


class _FailingEventLogWriter:
    """EventLogWriter that always raises on append."""

    async def append(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> int:
        raise OSError("event log write failed")


class _FakeModelRouter:
    """Fake ModelRouter that replays configured structured events per call."""

    def __init__(self, events_per_call: list[list]) -> None:
        self._calls = events_per_call
        self._idx = 0

    async def call(self, opts: Any):
        if self._idx < len(self._calls):
            events = self._calls[self._idx]
            self._idx += 1
            for event in events:
                yield event


class _ErrorModelRouter:
    """Fake ModelRouter that raises RuntimeError on the first iteration."""

    async def call(self, opts: Any):
        raise RuntimeError("router exploded")
        yield  # makes this an async generator


# ---------------------------------------------------------------------------
# Tests: per-iteration budget check
# ---------------------------------------------------------------------------


class TestHarnessLoopIterationBudget:
    # --- Hard breach ---

    def test_hard_breach_sets_final_phase_to_terminated(self, tmp_path: Path) -> None:
        # hard=1: after 1 tool_use model call the hard budget is hit
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(hard=1),
            )
        )
        assert final_phase == "terminated"

    def test_hard_breach_writes_phase_change_event_to_terminated(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(hard=1),
                event_log=event_log,
            )
        )
        phase_changes = [
            (sid, et, d) for sid, et, d in event_log.events if et == "session.phase_change"
        ]
        assert len(phase_changes) == 1
        _, _, data = phase_changes[0]
        assert data["after"] == "terminated"
        assert data["reason"] == "budget_exceeded"

    def test_hard_breach_stops_loop_at_hard_limit(self, tmp_path: Path) -> None:
        # hard=1: loop must stop after 1 model call even though fixtures have more
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _tool_use_call(), _end_turn_call()],
            [{"content": "r1"}, {"content": "r2"}],
        )
        reader = _FakePhaseReader(["running", "running", "running"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(hard=1),
            )
        )
        assert model_calls == 1

    def test_hard_breach_without_event_log_still_terminates(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(hard=1),
                event_log=None,
            )
        )
        assert final_phase == "terminated"

    def test_hard_breach_phase_change_event_session_id(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "budget-sess-1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(hard=1),
                event_log=event_log,
            )
        )
        phase_changes = [
            (sid, et, d) for sid, et, d in event_log.events if et == "session.phase_change"
        ]
        assert phase_changes[0][0] == "budget-sess-1"

    # --- Soft breach ---

    def test_soft_breach_emits_budget_warning_event(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=1),
                event_log=event_log,
            )
        )
        warnings = [(sid, et, d) for sid, et, d in event_log.events if et == "budget.warning"]
        assert len(warnings) == 1

    def test_soft_breach_sets_final_phase_to_waiting_for_user(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=1),
            )
        )
        assert final_phase == "waiting_for_user"

    def test_soft_breach_writes_phase_change_event_to_waiting_for_user(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=1),
                event_log=event_log,
            )
        )
        phase_changes = [
            (sid, et, d) for sid, et, d in event_log.events if et == "session.phase_change"
        ]
        assert len(phase_changes) == 1
        _, _, data = phase_changes[0]
        assert data["after"] == "waiting_for_user"

    def test_soft_breach_stops_loop_at_soft_limit(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _tool_use_call(), _end_turn_call()],
            [{"content": "r1"}, {"content": "r2"}],
        )
        reader = _FakePhaseReader(["running", "running", "running"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=1),
            )
        )
        assert model_calls == 1

    def test_soft_breach_without_event_log_still_pauses(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=1),
                event_log=None,
            )
        )
        assert final_phase == "waiting_for_user"

    def test_budget_warning_event_includes_model_calls_and_soft_threshold(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=1),
                event_log=event_log,
            )
        )
        warnings = [(sid, et, d) for sid, et, d in event_log.events if et == "budget.warning"]
        _, _, data = warnings[0]
        assert data["model_calls"] == 1
        assert data["budget_soft"] == 1

    # --- No breach ---

    def test_no_breach_below_soft_threshold_runs_to_completion(self, tmp_path: Path) -> None:
        # soft=5, hard=10, but only 2 model calls — loop should complete normally
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        model_calls, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=5, hard=10),
                event_log=event_log,
            )
        )
        assert final_phase == "idle"
        assert model_calls == 2
        # no budget.warning emitted — threshold not reached
        assert not any(et == "budget.warning" for _, et, _ in event_log.events)

    # --- Hard takes precedence over soft ---

    def test_hard_takes_precedence_over_soft_at_same_threshold(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                iteration_budget=IterationBudget(soft=1, hard=1),
            )
        )
        assert final_phase == "terminated"

    # --- Event log failure ---

    def test_event_log_failure_raises_harness_loop_error(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    iteration_budget=IterationBudget(hard=1),
                    event_log=_FailingEventLogWriter(),
                )
            )

    def test_event_log_failure_writes_to_audit_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    iteration_budget=IterationBudget(hard=1),
                    event_log=_FailingEventLogWriter(),
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_event_log_failure_error_message_surfaced(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError) as exc_info:
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    iteration_budget=IterationBudget(soft=1),
                    event_log=_FailingEventLogWriter(),
                )
            )
        assert len(exc_info.value.message) > 0


# ---------------------------------------------------------------------------
# Helpers for waiting_for_model tests
# ---------------------------------------------------------------------------


def _router_end_turn(*chunks: str) -> list:
    events: list = [MessageStartEvent(type="message_start", model="fake", provider="fake")]
    for chunk in chunks:
        events.append(TextDeltaEvent(type="text_delta", text=chunk))
    events.append(MessageStopEvent(type="message_stop", stop_reason="end_turn"))
    return events


def _router_tool_use(tool_id: str = "tu_1", tool_name: str = "bash") -> list:
    return [
        MessageStartEvent(type="message_start", model="fake", provider="fake"),
        ToolUseStartEvent(type="tool_use_start", id=tool_id, name=tool_name),
        ToolInputDeltaEvent(type="tool_input_delta", id=tool_id, partial_json='{"cmd":"ls"}'),
        MessageDeltaEvent(type="message_delta", stop_reason="tool_use"),
    ]


def _opts() -> ModelCallOpts:
    return ModelCallOpts(
        model="fake:model",
        messages=[Message(role="user", content="hi")],
    )


# ---------------------------------------------------------------------------
# Tests: waiting_for_model branch (model_router path)
# ---------------------------------------------------------------------------


class TestHarnessLoopWaitingForModel:
    def test_message_delta_emitted_per_text_chunk(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn("chunk1")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        deltas = [d for _, et, d in event_log.events if et == "message.delta"]
        assert len(deltas) == 1
        assert deltas[0]["text"] == "chunk1"

    def test_multiple_chunks_produce_ordered_message_delta_events(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn("hello", " ", "world")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        texts = [d["text"] for _, et, d in event_log.events if et == "message.delta"]
        assert texts == ["hello", " ", "world"]

    def test_model_calls_counted_via_router_path(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn("done")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert model_calls == 1

    def test_final_phase_idle_after_router_end_turn(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn("done")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert final_phase == "idle"

    def test_no_message_delta_without_event_log(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn("text")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
                model_router=router,
                model_call_opts=_opts(),
            )
        )

    def test_tool_use_collected_from_router_events(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_tool_use("tu_1"), _router_end_turn("ok")])
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["waiting_for_model", "waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert model_calls == 2
        assert tool_calls == 1
        assert final_phase == "idle"

    def test_stop_reason_parsed_from_message_delta_event(self, tmp_path: Path) -> None:
        events = [
            MessageStartEvent(type="message_start", model="fake", provider="fake"),
            MessageDeltaEvent(type="message_delta", stop_reason="end_turn"),
        ]
        router = _FakeModelRouter([events])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert final_phase == "idle"

    def test_pre_message_hook_dispatched(self, tmp_path: Path, monkeypatch) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        router = _FakeModelRouter([_router_end_turn("hi")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert "pre_message" in dispatched

    def test_on_model_call_hook_not_dispatched_on_router_path(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        router = _FakeModelRouter([_router_end_turn("hi")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert "on_model_call" not in dispatched

    def test_router_failure_transitions_to_terminated(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert final_phase == "terminated"

    def test_waiting_for_model_without_router_uses_fake_adapter(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
            )
        )
        assert model_calls == 1
        assert tool_calls == 0
        assert final_phase == "idle"


# ---------------------------------------------------------------------------
# Helpers: failing sandbox adapter
# ---------------------------------------------------------------------------


class _FailingSandboxAdapter:
    """Sandbox adapter that always raises on next_result."""

    def next_result(self) -> dict[str, Any]:
        raise OSError("sandbox dispatch failed")

    @property
    def dispatch_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Tests: waiting_for_tool branch (sandbox dispatch path)
# ---------------------------------------------------------------------------


class TestHarnessLoopWaitingForTool:
    def test_tool_call_result_event_emitted_when_phase_is_waiting_for_tool(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": "ok"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        results = [d for _, et, d in event_log.events if et == "tool_call.result"]
        assert len(results) == 1

    def test_tool_call_result_data_contains_content(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": "tool output here"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        results = [d for _, et, d in event_log.events if et == "tool_call.result"]
        assert results[0]["content"] == "tool output here"

    def test_tool_call_result_data_contains_tool_id(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_xyz", "content": "result"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        results = [d for _, et, d in event_log.events if et == "tool_call.result"]
        assert results[0]["tool_id"] == "tu_xyz"

    def test_multiple_waiting_for_tool_phases_produce_multiple_events(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": "r1"}, {"tool_id": "tu_2", "content": "r2"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        results = [d for _, et, d in event_log.events if et == "tool_call.result"]
        assert len(results) == 2

    def test_no_tool_call_result_event_without_event_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": "result"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )  # no exception; no events to assert

    def test_model_calls_not_incremented_during_waiting_for_tool(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
            )
        )
        assert model_calls == 1  # only from waiting_for_model iteration

    def test_tool_calls_incremented_per_waiting_for_tool(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"content": "r1"}, {"content": "r2"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, tool_calls, _ = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
            )
        )
        assert tool_calls == 2

    def test_final_phase_idle_after_tool_dispatch_and_model_end_turn(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"content": "result"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
            )
        )
        assert final_phase == "idle"

    def test_pre_tool_call_hook_not_dispatched_when_starting_from_waiting_for_tool(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # pre_tool_call is dispatched in the stop_reason=tool_use handling, not in the
        # waiting_for_tool dispatch branch; recovery from waiting_for_tool does not re-fire it.
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": "ok"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
            )
        )
        assert "pre_tool_call" not in dispatched

    def test_post_tool_call_hook_dispatched(self, tmp_path: Path, monkeypatch) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": "ok"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
            )
        )
        assert "post_tool_call" in dispatched

    def test_sandbox_failure_raises_harness_loop_error(self, tmp_path: Path) -> None:
        model, _ = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_tool"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=_FailingSandboxAdapter(),
                    phase_reader=reader,
                    audit_log=audit,
                )
            )

    def test_sandbox_failure_writes_audit_log(self, tmp_path: Path) -> None:
        model, _ = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_tool"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=_FailingSandboxAdapter(),
                    phase_reader=reader,
                    audit_log=audit,
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_sandbox_failure_audit_detail_has_session_id(self, tmp_path: Path) -> None:
        model, _ = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_tool"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "tool-sess-1",
                    model_adapter=model,
                    sandbox_adapter=_FailingSandboxAdapter(),
                    phase_reader=reader,
                    audit_log=audit,
                )
            )
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.run_loop.failed")
        assert record["detail"]["session_id"] == "tool-sess-1"

    def test_sandbox_failure_error_message_surfaced(self, tmp_path: Path) -> None:
        model, _ = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_tool"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError) as exc_info:
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=_FailingSandboxAdapter(),
                    phase_reader=reader,
                    audit_log=audit,
                )
            )
        assert len(exc_info.value.message) > 0

    # --- canvas_op content block: special message kind on the Session ---

    def test_canvas_op_event_emitted_when_content_is_canvas_op_dict(
        self, tmp_path: Path
    ) -> None:
        canvas_op_content = {
            "type": "canvas_op",
            "canvas_op": {
                "op": "set",
                "widget_id": "w1",
                "widget_kind": "meridian.text",
                "props": {"text": "hello"},
                "sequence": 1,
                "session_id": "s1",
                "timestamp": "2026-05-21T00:00:00Z",
            },
        }
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": canvas_op_content}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        canvas_events = [d for _, et, d in event_log.events if et == "canvas_op"]
        assert len(canvas_events) == 1

    def test_canvas_op_event_payload_matches_canvas_op_block(
        self, tmp_path: Path
    ) -> None:
        canvas_op_block = {
            "op": "set",
            "widget_id": "w1",
            "widget_kind": "meridian.text",
            "props": {"text": "hello"},
            "sequence": 1,
            "session_id": "s1",
            "timestamp": "2026-05-21T00:00:00Z",
        }
        canvas_op_content = {"type": "canvas_op", "canvas_op": canvas_op_block}
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": canvas_op_content}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        canvas_events = [d for _, et, d in event_log.events if et == "canvas_op"]
        assert canvas_events[0] == canvas_op_block

    def test_canvas_op_event_emitted_when_content_is_canvas_op_json_string(
        self, tmp_path: Path
    ) -> None:
        canvas_op_block = {
            "op": "patch",
            "widget_id": "w2",
            "widget_kind": "meridian.markdown",
            "props": {"content": "# Hi"},
            "sequence": 2,
            "session_id": "s1",
            "timestamp": "2026-05-21T00:01:00Z",
        }
        canvas_op_content = json.dumps({"type": "canvas_op", "canvas_op": canvas_op_block})
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": canvas_op_content}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        canvas_events = [d for _, et, d in event_log.events if et == "canvas_op"]
        assert len(canvas_events) == 1
        assert canvas_events[0] == canvas_op_block

    def test_no_canvas_op_event_for_non_canvas_tool_result(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": "plain string result"}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        canvas_events = [d for _, et, d in event_log.events if et == "canvas_op"]
        assert canvas_events == []

    def test_canvas_op_event_emitted_after_tool_call_result_event(
        self, tmp_path: Path
    ) -> None:
        canvas_op_content = {
            "type": "canvas_op",
            "canvas_op": {
                "op": "set",
                "widget_id": "w1",
                "widget_kind": "meridian.text",
                "props": {"text": "x"},
                "sequence": 1,
                "session_id": "s1",
                "timestamp": "2026-05-21T00:00:00Z",
            },
        }
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": canvas_op_content}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        types = [et for _, et, _ in event_log.events]
        result_idx = types.index("tool_call.result")
        canvas_idx = types.index("canvas_op")
        assert canvas_idx > result_idx

    def test_no_canvas_op_event_without_event_log(self, tmp_path: Path) -> None:
        canvas_op_content = {
            "type": "canvas_op",
            "canvas_op": {
                "op": "set",
                "widget_id": "w1",
                "widget_kind": "meridian.text",
                "props": {},
                "sequence": 1,
                "session_id": "s1",
                "timestamp": "2026-05-21T00:00:00Z",
            },
        }
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_end_turn_call()],
            [{"tool_id": "tu_1", "content": canvas_op_content}],
        )
        reader = _FakePhaseReader(["waiting_for_tool", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        # Must not raise even when event_log is None.
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )


# ---------------------------------------------------------------------------
# Tests: cache token capture via model_router path
# ---------------------------------------------------------------------------


def _router_end_turn_with_cache(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> list:
    return [
        MessageStartEvent(type="message_start", model="fake", provider="fake"),
        TextDeltaEvent(type="text_delta", text="done"),
        MessageStopEvent(
            type="message_stop",
            stop_reason="end_turn",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    ]


class TestHarnessLoopCacheMetrics:
    def test_usage_delta_carries_cache_creation_tokens_from_router(
        self, tmp_path: Path
    ) -> None:
        router = _FakeModelRouter([_router_end_turn_with_cache(cache_creation=150)])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        deltas: list[UsageDelta] = []
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
                on_usage_delta=deltas.append,
            )
        )
        assert len(deltas) == 1
        assert deltas[0].cache_creation_tokens == 150

    def test_usage_delta_carries_cache_read_tokens_from_router(
        self, tmp_path: Path
    ) -> None:
        router = _FakeModelRouter([_router_end_turn_with_cache(cache_read=200)])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        deltas: list[UsageDelta] = []
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
                on_usage_delta=deltas.append,
            )
        )
        assert len(deltas) == 1
        assert deltas[0].cache_read_tokens == 200

    def test_usage_delta_carries_real_input_output_tokens(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_cache(input_tokens=42, output_tokens=7)])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        deltas: list[UsageDelta] = []
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
                on_usage_delta=deltas.append,
            )
        )
        assert deltas[0].input_tokens == 42
        assert deltas[0].output_tokens == 7

    def test_usage_delta_emitted_to_event_log_on_router_path(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_cache(cache_creation=50, cache_read=100)])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        usage_events = [d for _, et, d in event_log.events if et == "usage.delta"]
        assert len(usage_events) == 1
        assert usage_events[0]["cache_creation_tokens"] == 50
        assert usage_events[0]["cache_read_tokens"] == 100

    def test_usage_delta_not_emitted_to_event_log_on_fake_adapter_path(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        usage_events = [d for _, et, d in event_log.events if et == "usage.delta"]
        assert usage_events == []

    def test_cache_tokens_zero_by_default_on_fake_adapter(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        deltas: list[UsageDelta] = []
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                on_usage_delta=deltas.append,
            )
        )
        assert deltas[0].cache_creation_tokens == 0
        assert deltas[0].cache_read_tokens == 0


# ---------------------------------------------------------------------------
# Helpers: failing model adapter
# ---------------------------------------------------------------------------


class _ErrorModelAdapter:
    """Fake model adapter that raises RuntimeError on call()."""

    name = "error"
    kind = "error"

    async def call(self):
        raise RuntimeError("model adapter exploded")
        yield  # makes this an async generator

    @property
    def call_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Tests: on model call error — on_error hooks + terminated transition
# ---------------------------------------------------------------------------


class TestHarnessLoopModelCallError:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    # --- Router path: no hooks → terminate ---

    def test_router_failure_no_hooks_final_phase_terminated(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert final_phase == "terminated"

    def test_router_failure_no_hooks_does_not_raise(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        result = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert result is not None

    # --- Fake adapter path: no hooks → terminate ---

    def test_fake_adapter_failure_no_hooks_final_phase_terminated(
        self, tmp_path: Path
    ) -> None:
        model_fixture = tmp_path / "fix" / "model_responses.ndjson"
        model_fixture.parent.mkdir(parents=True, exist_ok=True)
        model_fixture.write_text("")
        sandbox = FakeSandboxAdapter(tmp_path / "fix" / "tool_responses.ndjson")
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=_ErrorModelAdapter(),
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
            )
        )
        assert final_phase == "terminated"

    # --- on_error hooks dispatched ---

    def test_on_error_hooks_dispatched_on_router_failure(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert "on_error" in dispatched

    def test_on_error_hooks_dispatched_on_fake_adapter_failure(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model_fixture = tmp_path / "fix" / "model_responses.ndjson"
        model_fixture.parent.mkdir(parents=True, exist_ok=True)
        model_fixture.write_text("")
        sandbox = FakeSandboxAdapter(tmp_path / "fix" / "tool_responses.ndjson")
        reader = _FakePhaseReader(["created"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=_ErrorModelAdapter(),
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
            )
        )
        assert "on_error" in dispatched

    def test_no_on_error_hooks_without_hooks_dir(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert "on_error" not in dispatched

    def test_on_error_hook_payload_includes_error_and_type(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        captured_data: list[dict] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "on_error":
                captured_data.append(data)
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert len(captured_data) == 1
        assert "error" in captured_data[0]
        assert "error_type" in captured_data[0]

    # --- recoverable verdict: loop continues ---

    def test_recoverable_verdict_allows_loop_to_continue(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from meridiand._hook_dispatch import HookDispatchResult

        call_count = 0

        class _OnceErrorRouter:
            """Fails on first call, succeeds on second."""

            async def call(self, opts: Any):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("transient error")
                    yield
                else:
                    yield MessageStopEvent(type="message_stop", stop_reason="end_turn")

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "on_error":
                return [
                    HookDispatchResult(
                        hook_id="h1",
                        hook_name="recovery",
                        is_error=False,
                        verdict="recoverable",
                    )
                ]
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
                model_router=_OnceErrorRouter(),
                model_call_opts=_opts(),
            )
        )
        assert final_phase == "idle"

    def test_non_recoverable_verdict_transitions_to_terminated(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from meridiand._hook_dispatch import HookDispatchResult

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "on_error":
                return [
                    HookDispatchResult(
                        hook_id="h1",
                        hook_name="noop",
                        is_error=False,
                        verdict="continue",
                    )
                ]
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert final_phase == "terminated"

    # --- OTel span emitted on model call error ---

    def test_model_call_error_emits_otel_span(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "harness.model_call_error" in span_names

    def test_model_call_error_span_has_session_id(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "mc-err-sess-1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.model_call_error")
        assert span is not None
        assert span.attributes["session.id"] == "mc-err-sess-1"

    def test_model_call_error_span_carries_invocation_event(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.model_call_error")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    # --- on_error hook dispatch failure: surfaces error + writes audit ---

    def test_on_error_hook_failure_raises_harness_loop_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        async def _failing_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "on_error":
                raise OSError("hook system unavailable")
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _failing_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    hooks_dir=tmp_path / "hooks",
                    model_router=router,
                    model_call_opts=_opts(),
                )
            )

    def test_on_error_hook_failure_writes_audit_log(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        async def _failing_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "on_error":
                raise OSError("hook system unavailable")
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _failing_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    hooks_dir=tmp_path / "hooks",
                    model_router=router,
                    model_call_opts=_opts(),
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.model_call_error.failed" for r in records)

    def test_on_error_hook_failure_audit_detail_has_session_id(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        async def _failing_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "on_error":
                raise OSError("hook system unavailable")
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _failing_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "mc-err-sess-2",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    hooks_dir=tmp_path / "hooks",
                    model_router=router,
                    model_call_opts=_opts(),
                )
            )
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.model_call_error.failed")
        assert record["detail"]["session_id"] == "mc-err-sess-2"

    def test_on_error_hook_failure_audit_detail_has_message(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        async def _failing_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "on_error":
                raise OSError("hook system unavailable")
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _failing_dispatch)

        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    hooks_dir=tmp_path / "hooks",
                    model_router=router,
                    model_call_opts=_opts(),
                )
            )
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.model_call_error.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Tests: stop_reason end_turn
# ---------------------------------------------------------------------------


class TestHarnessLoopEndTurn:
    def test_emits_message_appended_event(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        appended = [d for _, et, d in event_log.events if et == "message.appended"]
        assert len(appended) == 1

    def test_message_appended_has_model_call_number(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        appended = [d for _, et, d in event_log.events if et == "message.appended"]
        assert appended[0]["model_call_number"] == 1

    def test_no_message_appended_without_event_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )
        assert final_phase == "idle"  # completed without error

    def test_emits_session_phase_change_to_idle(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        assert any(d["after"] == "idle" for d in phase_changes)

    def test_session_phase_change_has_reason_end_turn(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        idle_changes = [d for d in phase_changes if d["after"] == "idle"]
        assert len(idle_changes) == 1
        assert idle_changes[0]["reason"] == "end_turn"

    def test_no_phase_change_event_without_event_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )
        assert final_phase == "idle"  # completed without error

    def test_post_message_hook_dispatched(self, tmp_path: Path, monkeypatch) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
            )
        )
        assert "post_message" in dispatched

    def test_no_post_message_hook_without_hooks_dir(self, tmp_path: Path, monkeypatch) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=None,
            )
        )
        assert "post_message" not in dispatched

    def test_event_log_failure_raises_harness_loop_error(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    event_log=_FailingEventLogWriter(),
                )
            )

    def test_event_log_failure_writes_audit_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    event_log=_FailingEventLogWriter(),
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_post_message_hook_failure_raises_harness_loop_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        async def _failing_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "post_message":
                raise OSError("hook system unavailable")
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _failing_dispatch)

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    hooks_dir=tmp_path / "hooks",
                )
            )

    def test_post_message_hook_failure_writes_audit_log(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        async def _failing_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "post_message":
                raise OSError("hook system unavailable")
            return []

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _failing_dispatch)

        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    hooks_dir=tmp_path / "hooks",
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)


# ---------------------------------------------------------------------------
# Helpers: max_tokens fixtures
# ---------------------------------------------------------------------------


def _max_tokens_call(text: str = "partial") -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "text_delta", "text": text},
        {"type": "message_stop", "stop_reason": "max_tokens"},
    ]


def _router_max_tokens(*chunks: str) -> list:
    events: list = [MessageStartEvent(type="message_start", model="fake", provider="fake")]
    for chunk in chunks:
        events.append(TextDeltaEvent(type="text_delta", text=chunk))
    events.append(MessageStopEvent(type="message_stop", stop_reason="max_tokens"))
    return events


# ---------------------------------------------------------------------------
# Tests: stop_reason max_tokens
# ---------------------------------------------------------------------------


class TestHarnessLoopMaxTokens:
    def test_no_policy_transitions_to_waiting_for_user(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
            )
        )
        assert final_phase == "waiting_for_user"

    def test_continue_false_transitions_to_waiting_for_user(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                max_tokens_policy=MaxTokensPolicy(continue_allowed=False),
            )
        )
        assert final_phase == "waiting_for_user"

    def test_continue_true_loops_and_calls_model_again(self, tmp_path: Path) -> None:
        # First call returns max_tokens; second returns end_turn.
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call(), _end_turn_call()])
        reader = _FakePhaseReader(["running", "running"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                max_tokens_policy=MaxTokensPolicy(continue_allowed=True),
            )
        )
        assert model_calls == 2

    def test_continue_true_final_phase_idle_after_end_turn(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call(), _end_turn_call()])
        reader = _FakePhaseReader(["running", "running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                max_tokens_policy=MaxTokensPolicy(continue_allowed=True),
            )
        )
        assert final_phase == "idle"

    def test_emits_message_truncated_event(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        truncated = [d for _, et, d in event_log.events if et == "message.truncated"]
        assert len(truncated) == 1

    def test_message_truncated_has_model_call_number(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        truncated = [d for _, et, d in event_log.events if et == "message.truncated"]
        assert truncated[0]["model_call_number"] == 1

    def test_no_truncated_event_without_event_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )
        # No exception raised; implicit pass if reached.

    def test_emits_phase_change_event_when_not_continuing(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        assert len(phase_changes) == 1

    def test_phase_change_after_is_waiting_for_user(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        assert phase_changes[0]["after"] == "waiting_for_user"

    def test_phase_change_reason_is_max_tokens(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        assert phase_changes[0]["reason"] == "max_tokens"

    def test_no_phase_change_event_when_continue_true(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call(), _end_turn_call()])
        reader = _FakePhaseReader(["running", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                max_tokens_policy=MaxTokensPolicy(continue_allowed=True),
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        assert not any(d.get("reason") == "max_tokens" for d in phase_changes)

    def test_without_event_log_still_transitions_to_waiting_for_user(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )
        assert final_phase == "waiting_for_user"

    def test_without_event_log_continue_true_still_loops(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call(), _end_turn_call()])
        reader = _FakePhaseReader(["running", "running"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, _ = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
                max_tokens_policy=MaxTokensPolicy(continue_allowed=True),
            )
        )
        assert model_calls == 2

    def test_event_log_failure_raises_harness_loop_error(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    event_log=_FailingEventLogWriter(),
                )
            )

    def test_event_log_failure_writes_to_audit_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(tmp_path / "fix", [_max_tokens_call()])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    event_log=_FailingEventLogWriter(),
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_router_max_tokens_without_policy_transitions_to_waiting_for_user(
        self, tmp_path: Path
    ) -> None:
        router = _FakeModelRouter([_router_max_tokens("partial text")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        assert final_phase == "waiting_for_user"

    def test_router_max_tokens_continue_true_loops(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_max_tokens("partial"), _router_end_turn("done")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model", "waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        model_calls, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_router=router,
                model_call_opts=_opts(),
                max_tokens_policy=MaxTokensPolicy(continue_allowed=True),
            )
        )
        assert model_calls == 2
        assert final_phase == "idle"


# ---------------------------------------------------------------------------
# Helpers: tool_use fixtures for stop_reason=tool_use tests
# ---------------------------------------------------------------------------


def _tool_use_bad_json_call(tool_id: str = "tu_1", tool_name: str = "bash") -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "model": "fake", "provider": "fake"},
        {"type": "tool_use_start", "id": tool_id, "name": tool_name},
        {"type": "tool_input_delta", "id": tool_id, "partial_json": "not-valid-json{"},
        {"type": "message_stop", "stop_reason": "tool_use"},
    ]


_BASH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"cmd": {"type": "string"}},
    "required": ["cmd"],
}

_BASH_SCHEMA_STRICT: dict[str, Any] = {
    "type": "object",
    "properties": {"cmd": {"type": "integer"}},  # "ls" will fail: not an integer
    "required": ["cmd"],
}


# ---------------------------------------------------------------------------
# Tests: stop_reason tool_use — schema validate, capability check, hooks, events
# ---------------------------------------------------------------------------


class TestHarnessLoopStopReasonToolUse:
    # --- tool_call.requested events ---

    def test_tool_call_requested_event_written_per_tool_block(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        requested = [d for _, et, d in event_log.events if et == "tool_call.requested"]
        assert len(requested) == 1

    def test_tool_call_requested_data_contains_tool_id(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_abc", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        requested = [d for _, et, d in event_log.events if et == "tool_call.requested"]
        assert requested[0]["tool_id"] == "tu_abc"

    def test_tool_call_requested_data_contains_tool_name(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        requested = [d for _, et, d in event_log.events if et == "tool_call.requested"]
        assert requested[0]["tool_name"] == "bash"

    def test_tool_call_requested_data_contains_parsed_args(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        requested = [d for _, et, d in event_log.events if et == "tool_call.requested"]
        assert requested[0]["args"] == {"cmd": "ls"}

    def test_no_tool_call_requested_event_without_event_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )  # no exception; no events to assert

    # --- session.phase_change to waiting_for_tool ---

    def test_phase_change_event_written_with_after_waiting_for_tool(
        self, tmp_path: Path
    ) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        assert any(d["after"] == "waiting_for_tool" for d in phase_changes)

    def test_phase_change_reason_is_tool_use(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
            )
        )
        phase_changes = [d for _, et, d in event_log.events if et == "session.phase_change"]
        tool_use_changes = [d for d in phase_changes if d.get("after") == "waiting_for_tool"]
        assert tool_use_changes[0]["reason"] == "tool_use"

    def test_no_phase_change_event_without_event_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call(), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
            )
        )  # no exception

    # --- pre_tool_call hook in tool_use handling ---

    def test_pre_tool_call_hook_dispatched_on_tool_use(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        dispatched: list[str] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            dispatched.append(event)

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
            )
        )
        assert "pre_tool_call" in dispatched

    def test_pre_tool_call_hook_payload_has_tool_id_and_name(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        captured: list[dict] = []

        async def _fake_dispatch(event, data, ctx, *, hooks_dir, audit_log):
            if event == "pre_tool_call":
                captured.append(data)

        monkeypatch.setattr("meridiand._replay.dispatch_hooks", _fake_dispatch)

        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_xyz", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                hooks_dir=tmp_path / "hooks",
            )
        )
        assert len(captured) == 1
        assert captured[0]["tool_id"] == "tu_xyz"
        assert captured[0]["tool_name"] == "bash"

    # --- Schema validation ---

    def test_valid_args_with_schema_passes(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        model_calls, tool_calls, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                tool_schemas={"bash": _BASH_SCHEMA},
            )
        )
        assert final_phase == "idle"
        assert model_calls == 2
        assert tool_calls == 1

    def test_schema_mismatch_raises_harness_loop_error(self, tmp_path: Path) -> None:
        # _tool_use_call sends {"cmd": "ls"}; _BASH_SCHEMA_STRICT expects cmd as integer
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash")],
            [],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    tool_schemas={"bash": _BASH_SCHEMA_STRICT},
                )
            )

    def test_schema_mismatch_writes_audit_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash")],
            [],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    tool_schemas={"bash": _BASH_SCHEMA_STRICT},
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_invalid_json_args_raises_harness_loop_error(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_bad_json_call("tu_1", "bash")],
            [],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                )
            )

    def test_invalid_json_args_writes_audit_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_bad_json_call("tu_1", "bash")],
            [],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_schemas_from_model_call_opts_used_for_validation(self, tmp_path: Path) -> None:
        from meridian_sdk_provider import ToolDefinition

        # _tool_use_call sends {"cmd": "ls"}; BASH_SCHEMA_STRICT expects integer → fail
        opts = ModelCallOpts(
            model="fake:model",
            messages=[Message(role="user", content="hi")],
            tools=[ToolDefinition(name="bash", description="run", input_schema=_BASH_SCHEMA_STRICT)],
        )
        model, sandbox = _adapters(tmp_path / "fix", [_tool_use_call("tu_1", "bash")], [])
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    model_call_opts=opts,
                )
            )

    def test_tool_schemas_param_overrides_model_call_opts(self, tmp_path: Path) -> None:
        from meridian_sdk_provider import ToolDefinition

        # model_call_opts has strict schema (would fail), tool_schemas has permissive (should pass)
        opts = ModelCallOpts(
            model="fake:model",
            messages=[Message(role="user", content="hi")],
            tools=[ToolDefinition(name="bash", description="run", input_schema=_BASH_SCHEMA_STRICT)],
        )
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                model_call_opts=opts,
                tool_schemas={"bash": _BASH_SCHEMA},  # override: permissive schema
            )
        )
        assert final_phase == "idle"  # passed validation, no error

    # --- Capability intersection check ---

    def test_all_required_caps_granted_passes(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                tool_capabilities={"bash": frozenset({"exec.bash"})},
                granted_capabilities=frozenset({"exec.bash", "fs.read"}),
            )
        )
        assert final_phase == "idle"

    def test_missing_capability_raises_harness_loop_error(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash")],
            [],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    tool_capabilities={"bash": frozenset({"exec.bash"})},
                    granted_capabilities=frozenset(),  # exec.bash not granted
                )
            )

    def test_missing_capability_writes_audit_log(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash")],
            [],
        )
        reader = _FakePhaseReader(["running"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "s1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    tool_capabilities={"bash": frozenset({"exec.bash"})},
                    granted_capabilities=frozenset(),
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_no_capability_check_when_tool_capabilities_none(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                tool_capabilities=None,
                granted_capabilities=frozenset(),  # empty, but no check performed
            )
        )
        assert final_phase == "idle"

    def test_no_capability_check_when_granted_capabilities_none(self, tmp_path: Path) -> None:
        model, sandbox = _adapters(
            tmp_path / "fix",
            [_tool_use_call("tu_1", "bash"), _end_turn_call()],
            [{"content": "ok"}],
        )
        reader = _FakePhaseReader(["running", "waiting_for_tool", "running"])
        audit = FileAuditLog(tmp_path)
        _, _, final_phase = asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                tool_capabilities={"bash": frozenset({"exec.bash"})},
                granted_capabilities=None,  # check skipped
            )
        )
        assert final_phase == "idle"


# ---------------------------------------------------------------------------
# Helpers for Contract 4 (event translation) tests
# ---------------------------------------------------------------------------


def _router_thinking_then_text(*chunks: str) -> list:
    """Event stream with thinking block followed by text chunks and end_turn stop."""
    events: list = [MessageStartEvent(type="message_start", model="fake", provider="fake")]
    events.append(ThinkingDeltaEvent(type="thinking_delta", thinking="let me think"))
    for chunk in chunks:
        events.append(TextDeltaEvent(type="text_delta", text=chunk))
    events.append(MessageStopEvent(
        type="message_stop",
        stop_reason="end_turn",
        input_tokens=12,
        output_tokens=6,
    ))
    return events


def _router_end_turn_with_tokens(
    input_tokens: int = 10,
    output_tokens: int = 5,
    stop_reason: str = "end_turn",
) -> list:
    return [
        MessageStartEvent(type="message_start", model="fake", provider="fake"),
        TextDeltaEvent(type="text_delta", text="done"),
        MessageStopEvent(
            type="message_stop",
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: Contract 4 — event translation via model_router path
# ---------------------------------------------------------------------------


class TestHarnessLoopContract4EventTranslation:
    # --- ThinkingDeltaEvent → message.delta kind="thinking" ---

    def test_thinking_delta_emits_message_delta(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_thinking_then_text("answer")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        thinking_deltas = [
            d for _, et, d in event_log.events
            if et == "message.delta" and d.get("kind") == "thinking"
        ]
        assert len(thinking_deltas) == 1

    def test_thinking_delta_data_kind_is_thinking(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_thinking_then_text("answer")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        thinking_deltas = [
            d for _, et, d in event_log.events
            if et == "message.delta" and d.get("kind") == "thinking"
        ]
        assert thinking_deltas[0]["kind"] == "thinking"

    def test_thinking_delta_data_carries_thinking_text(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_thinking_then_text("reply")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        thinking_deltas = [
            d for _, et, d in event_log.events
            if et == "message.delta" and d.get("kind") == "thinking"
        ]
        assert thinking_deltas[0]["thinking"] == "let me think"

    def test_thinking_delta_data_has_model_call_number(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_thinking_then_text("reply")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        thinking_deltas = [
            d for _, et, d in event_log.events
            if et == "message.delta" and d.get("kind") == "thinking"
        ]
        assert thinking_deltas[0]["model_call_number"] == 1

    def test_no_thinking_delta_without_event_log(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_thinking_then_text("reply")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        # Must not raise; simply no events written
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
                model_router=router,
                model_call_opts=_opts(),
            )
        )

    # --- text message.delta now has kind="text" ---

    def test_text_delta_kind_is_text(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn("chunk1")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        text_deltas = [
            d for _, et, d in event_log.events
            if et == "message.delta" and d.get("kind") == "text"
        ]
        assert len(text_deltas) == 1
        assert text_deltas[0]["kind"] == "text"

    # --- Mixed thinking + text ordering ---

    def test_thinking_then_text_delta_order(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_thinking_then_text("the answer")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        deltas = [(sid, et, d) for sid, et, d in event_log.events if et == "message.delta"]
        kinds = [d["kind"] for _, _, d in deltas]
        assert kinds == ["thinking", "text"]

    # --- model_call.completed event ---

    def test_model_call_completed_emitted_on_message_stop(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_tokens()])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        completed = [d for _, et, d in event_log.events if et == "model_call.completed"]
        assert len(completed) == 1

    def test_model_call_completed_data_stop_reason(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_tokens(stop_reason="end_turn")])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        completed = [d for _, et, d in event_log.events if et == "model_call.completed"]
        assert completed[0]["stop_reason"] == "end_turn"

    def test_model_call_completed_data_input_tokens(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_tokens(input_tokens=99)])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        completed = [d for _, et, d in event_log.events if et == "model_call.completed"]
        assert completed[0]["input_tokens"] == 99

    def test_model_call_completed_data_output_tokens(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_tokens(output_tokens=77)])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        completed = [d for _, et, d in event_log.events if et == "model_call.completed"]
        assert completed[0]["output_tokens"] == 77

    def test_model_call_completed_data_model_call_number(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_tokens()])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        event_log = _FakeEventLogWriter()
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=event_log,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
        completed = [d for _, et, d in event_log.events if et == "model_call.completed"]
        assert completed[0]["model_call_number"] == 1

    def test_no_model_call_completed_without_event_log(self, tmp_path: Path) -> None:
        router = _FakeModelRouter([_router_end_turn_with_tokens()])
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        # Must not raise; simply no events written
        asyncio.run(
            run_harness_loop(
                "s1",
                model_adapter=model,
                sandbox_adapter=sandbox,
                phase_reader=reader,
                audit_log=audit,
                event_log=None,
                model_router=router,
                model_call_opts=_opts(),
            )
        )
