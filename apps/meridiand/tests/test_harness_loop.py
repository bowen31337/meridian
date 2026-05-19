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
    UsageDelta,
    run_harness_loop,
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
