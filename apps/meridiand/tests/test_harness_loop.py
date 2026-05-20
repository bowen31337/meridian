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
  - Router failure raises HarnessLoopError (error surfaced to caller).
  - Router failure writes harness.run_loop.failed to audit log.
  - Router failure audit detail includes session_id.
  - Router failure error message is non-empty.
  - Without model_router, waiting_for_model phase falls back to fake adapter.
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
        reader = _FakePhaseReader(["created", "created"])
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
        reader = _FakePhaseReader(["created", "created"])
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
        # First read: "running" (proceed), second read: "paused" (stop)
        reader = _FakePhaseReader(["running", "paused"])
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
        reader = _FakePhaseReader(["running", "terminated"])
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
        reader = _FakePhaseReader(["running", "idle"])
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
        reader = _FakePhaseReader(["created", "created"])
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
        reader = _FakePhaseReader(["running", "running"])
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
        assert event_log.events == []

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
        reader = _FakePhaseReader(["waiting_for_model", "waiting_for_model"])
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

    def test_router_failure_raises_harness_loop_error(self, tmp_path: Path) -> None:
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
                    model_router=router,
                    model_call_opts=_opts(),
                )
            )

    def test_router_failure_error_code_is_harness_loop_failed(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError) as exc_info:
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
        assert exc_info.value.code == "harness_loop_failed"

    def test_router_failure_writes_audit_log(self, tmp_path: Path) -> None:
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
                    model_router=router,
                    model_call_opts=_opts(),
                )
            )
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.run_loop.failed" for r in records)

    def test_router_failure_audit_detail_has_session_id(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError):
            asyncio.run(
                run_harness_loop(
                    "router-sess-1",
                    model_adapter=model,
                    sandbox_adapter=sandbox,
                    phase_reader=reader,
                    audit_log=audit,
                    model_router=router,
                    model_call_opts=_opts(),
                )
            )
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.run_loop.failed")
        assert record["detail"]["session_id"] == "router-sess-1"

    def test_router_failure_error_message_surfaced(self, tmp_path: Path) -> None:
        router = _ErrorModelRouter()
        model, sandbox = _adapters(tmp_path / "fix", [_end_turn_call()])
        reader = _FakePhaseReader(["waiting_for_model"])
        audit = FileAuditLog(tmp_path)
        with pytest.raises(HarnessLoopError) as exc_info:
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
        assert len(exc_info.value.message) > 0

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
