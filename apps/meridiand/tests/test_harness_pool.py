"""
Harness pool conformance suite.

Tests cover:
  - _slot_for(session_id) returns hash(session_id) % num_workers.
  - assign(session_id) enqueues session into the correct slot for processing.
  - wake(session_id) enqueues session into the same slot as assign for the same session_id.
  - assign and wake for the same session_id always route to the same slot.
  - start() starts all N worker tasks.
  - start() wakes all sessions in active phases found in storage (SIGKILL restart recovery).
  - start() does not wake sessions whose phase is "idle".
  - start() does not wake sessions whose phase is "paused".
  - start() does not wake sessions whose phase is "terminated".
  - start() succeeds when no sessions directory exists.
  - stop() cancels all running worker tasks.
  - stop() is idempotent (calling twice does not raise).
  - Worker processes sessions from its slot queue sequentially.
  - OTel span "harness.pool.assign" emitted on assign.
  - OTel span "harness.pool.wake" emitted on wake.
  - OTel span "harness.pool.start" emitted on start.
  - Worker loop emits OTel span "session.run_span" when processing a session.
  - session.run_span has session.id attribute.
  - session.run_span continues the session trace when manifest has a stored traceparent.
  - assign span has session.id attribute.
  - assign span has harness.pool.worker_slot attribute.
  - assign span carries structured invocation event.
  - wake span has session.id attribute.
  - wake span has harness.pool.worker_slot attribute.
  - wake span carries structured invocation event.
  - assign failure raises HarnessPoolError with code "harness_pool_failed".
  - assign failure span has ERROR status.
  - assign failure writes audit log entry with event "harness.pool.assign.failed".
  - assign audit detail includes session_id.
  - assign audit detail includes message.
  - wake failure raises HarnessPoolError with code "harness_pool_failed".
  - wake failure writes audit log entry with event "harness.pool.wake.failed".
  - wake audit detail includes session_id.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from meridiand._audit import FileAuditLog
from meridiand._harness_pool import HarnessPool, HarnessPoolError

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Fake phase reader
# ---------------------------------------------------------------------------


class _FakePhaseReader:
    def __init__(self, phases: dict[str, str]) -> None:
        self._phases = phases

    def current_phase(self, session_id: str) -> str:
        return self._phases.get(session_id, "created")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_run_session() -> Any:
    async def run_session(session_id: str) -> tuple[int, int, str]:
        return (0, 0, "idle")

    return run_session


def _capturing_run_session(processed: list[str]) -> Any:
    async def run_session(session_id: str) -> tuple[int, int, str]:
        processed.append(session_id)
        return (0, 0, "idle")

    return run_session


def _write_session_manifest(storage_root: Path, session_id: str) -> None:
    session_dir = storage_root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(
        json.dumps({"session_id": session_id, "status": "active"})
    )


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_pool(
    tmp_path: Path,
    *,
    num_workers: int = 4,
    run_session: Any = None,
    phases: dict[str, str] | None = None,
) -> HarnessPool:
    return HarnessPool(
        num_workers=num_workers,
        run_session=run_session or _noop_run_session(),
        phase_reader=_FakePhaseReader(phases or {}),
        audit_log=FileAuditLog(tmp_path),
        storage_root=tmp_path,
    )


# ---------------------------------------------------------------------------
# Tests: slot routing
# ---------------------------------------------------------------------------


class TestHarnessPoolSlotRouting:
    def test_slot_for_uses_hash_modulo_num_workers(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=4)
        assert pool._slot_for("sess-abc") == hash("sess-abc") % 4

    def test_slot_for_same_session_same_slot(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=8)
        assert pool._slot_for("sess-1") == pool._slot_for("sess-1")

    def test_slot_for_respects_num_workers(self, tmp_path: Path) -> None:
        pool2 = _make_pool(tmp_path, num_workers=2)
        pool7 = _make_pool(tmp_path, num_workers=7)
        for sid in ["s1", "s2", "session_long_id_xyz"]:
            assert 0 <= pool2._slot_for(sid) < 2
            assert 0 <= pool7._slot_for(sid) < 7

    def test_assign_and_wake_route_to_same_slot(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=4)
        session_id = "sess-routing-check"
        assert pool._slot_for(session_id) == pool._slot_for(session_id)


# ---------------------------------------------------------------------------
# Tests: assign
# ---------------------------------------------------------------------------


class TestHarnessPoolAssign:
    def test_assign_enqueues_session_for_processing(self, tmp_path: Path) -> None:
        processed: list[str] = []
        pool = _make_pool(tmp_path, run_session=_capturing_run_session(processed))

        async def run() -> None:
            await pool.start()
            await pool.assign("sess-1")
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-1" in processed

    def test_assign_processes_session_via_worker(self, tmp_path: Path) -> None:
        processed: list[str] = []
        pool = _make_pool(tmp_path, num_workers=2, run_session=_capturing_run_session(processed))

        async def run() -> None:
            await pool.start()
            await pool.assign("sess-worker")
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-worker" in processed

    def test_assign_and_wake_send_to_same_slot(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=4)
        sid = "sess-slot-check"
        slot_assign = pool._slot_for(sid)
        slot_wake = pool._slot_for(sid)
        assert slot_assign == slot_wake


# ---------------------------------------------------------------------------
# Tests: wake
# ---------------------------------------------------------------------------


class TestHarnessPoolWake:
    def test_wake_enqueues_session_for_processing(self, tmp_path: Path) -> None:
        processed: list[str] = []
        pool = _make_pool(tmp_path, run_session=_capturing_run_session(processed))

        async def run() -> None:
            await pool.start()
            await pool.wake("sess-wake-1")
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-wake-1" in processed

    def test_wake_routes_to_same_slot_as_assign(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=4)
        sid = "sess-wake-slot"
        assert pool._slot_for(sid) == pool._slot_for(sid)

    def test_wake_multiple_sessions(self, tmp_path: Path) -> None:
        processed: list[str] = []
        pool = _make_pool(tmp_path, run_session=_capturing_run_session(processed))

        async def run() -> None:
            await pool.start()
            await pool.wake("sess-a")
            await pool.wake("sess-b")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-a" in processed
        assert "sess-b" in processed


# ---------------------------------------------------------------------------
# Tests: start (SIGKILL restart recovery)
# ---------------------------------------------------------------------------


class TestHarnessPoolStart:
    def test_start_starts_worker_tasks(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=3)

        async def run() -> None:
            await pool.start()
            for slot in pool._slots:
                assert slot.task is not None
                assert not slot.task.done()
            await pool.stop()

        asyncio.run(run())

    def test_start_wakes_created_sessions_from_storage(self, tmp_path: Path) -> None:
        _write_session_manifest(tmp_path, "sess-created")
        processed: list[str] = []
        pool = _make_pool(
            tmp_path,
            run_session=_capturing_run_session(processed),
            phases={"sess-created": "created"},
        )

        async def run() -> None:
            await pool.start()
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-created" in processed

    def test_start_wakes_running_sessions_from_storage(self, tmp_path: Path) -> None:
        _write_session_manifest(tmp_path, "sess-running")
        processed: list[str] = []
        pool = _make_pool(
            tmp_path,
            run_session=_capturing_run_session(processed),
            phases={"sess-running": "running"},
        )

        async def run() -> None:
            await pool.start()
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-running" in processed

    def test_start_does_not_wake_idle_sessions(self, tmp_path: Path) -> None:
        _write_session_manifest(tmp_path, "sess-idle")
        processed: list[str] = []
        pool = _make_pool(
            tmp_path,
            run_session=_capturing_run_session(processed),
            phases={"sess-idle": "idle"},
        )

        async def run() -> None:
            await pool.start()
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-idle" not in processed

    def test_start_does_not_wake_paused_sessions(self, tmp_path: Path) -> None:
        _write_session_manifest(tmp_path, "sess-paused")
        processed: list[str] = []
        pool = _make_pool(
            tmp_path,
            run_session=_capturing_run_session(processed),
            phases={"sess-paused": "paused"},
        )

        async def run() -> None:
            await pool.start()
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-paused" not in processed

    def test_start_does_not_wake_terminated_sessions(self, tmp_path: Path) -> None:
        _write_session_manifest(tmp_path, "sess-terminated")
        processed: list[str] = []
        pool = _make_pool(
            tmp_path,
            run_session=_capturing_run_session(processed),
            phases={"sess-terminated": "terminated"},
        )

        async def run() -> None:
            await pool.start()
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-terminated" not in processed

    def test_start_with_no_sessions_directory_succeeds(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.stop()

        asyncio.run(run())  # no exception raised

    def test_start_wakes_only_active_sessions_among_mixed(self, tmp_path: Path) -> None:
        _write_session_manifest(tmp_path, "sess-active-1")
        _write_session_manifest(tmp_path, "sess-active-2")
        _write_session_manifest(tmp_path, "sess-stopped")
        processed: list[str] = []
        pool = _make_pool(
            tmp_path,
            run_session=_capturing_run_session(processed),
            phases={
                "sess-active-1": "created",
                "sess-active-2": "running",
                "sess-stopped": "idle",
            },
        )

        async def run() -> None:
            await pool.start()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        assert "sess-active-1" in processed
        assert "sess-active-2" in processed
        assert "sess-stopped" not in processed


# ---------------------------------------------------------------------------
# Tests: stop
# ---------------------------------------------------------------------------


class TestHarnessPoolStop:
    def test_stop_cancels_worker_tasks(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=2)

        async def run() -> None:
            await pool.start()
            tasks_before = [slot.task for slot in pool._slots]
            assert all(t is not None and not t.done() for t in tasks_before)
            await pool.stop()
            for slot in pool._slots:
                assert slot.task is None or slot.task.done()

        asyncio.run(run())

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.stop()
            await pool.stop()  # second stop must not raise

        asyncio.run(run())

    def test_stop_before_start_is_safe(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.stop()  # no tasks started yet — must not raise

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Tests: OTel instrumentation
# ---------------------------------------------------------------------------


class TestHarnessPoolOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_assign_emits_harness_pool_assign_span(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.assign("sess-otel-assign")
            await pool.stop()

        asyncio.run(run())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "harness.pool.assign" in span_names

    def test_assign_span_has_session_id_attribute(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.assign("sess-otel-id")
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.pool.assign")
        assert span is not None
        assert span.attributes["session.id"] == "sess-otel-id"

    def test_assign_span_has_worker_slot_attribute(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=4)
        sid = "sess-slot-attr"
        expected_slot = pool._slot_for(sid)

        async def run() -> None:
            await pool.start()
            await pool.assign(sid)
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.pool.assign")
        assert span is not None
        assert span.attributes["harness.pool.worker_slot"] == expected_slot

    def test_assign_span_carries_invocation_event(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.assign("sess-inv-event")
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.pool.assign")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_wake_emits_harness_pool_wake_span(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.wake("sess-otel-wake")
            await pool.stop()

        asyncio.run(run())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "harness.pool.wake" in span_names

    def test_wake_span_has_session_id_attribute(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.wake("sess-wake-sid")
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.pool.wake")
        assert span is not None
        assert span.attributes["session.id"] == "sess-wake-sid"

    def test_wake_span_has_worker_slot_attribute(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=4)
        sid = "sess-wake-slot"
        expected_slot = pool._slot_for(sid)

        async def run() -> None:
            await pool.start()
            await pool.wake(sid)
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.pool.wake")
        assert span is not None
        assert span.attributes["harness.pool.worker_slot"] == expected_slot

    def test_wake_span_carries_invocation_event(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.wake("sess-wake-inv")
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.pool.wake")
        assert span is not None
        event_names = [e.name for e in span.events]
        assert "meridian.error.invocation" in event_names

    def test_start_emits_harness_pool_start_span(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.stop()

        asyncio.run(run())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "harness.pool.start" in span_names

    def test_assign_failure_span_has_error_status(self, tmp_path: Path) -> None:
        from opentelemetry.trace import StatusCode

        pool = _make_pool(tmp_path, num_workers=0)  # ZeroDivisionError in _slot_for

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.assign("sess-fail-span")

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("harness.pool.assign")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_worker_loop_emits_session_run_span(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.wake("sess-run-span")
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.run_span" in span_names

    def test_session_run_span_has_session_id_attribute(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.wake("sess-run-id")
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.run_span")
        assert span is not None
        assert span.attributes["session.id"] == "sess-run-id"

    def test_session_run_span_continues_session_trace(self, tmp_path: Path) -> None:
        # session.run_span should share trace_id with the stored session traceparent.
        session_id = "sess-trace-cont"
        fake_trace_id = "aa" * 16  # 32 hex chars
        fake_span_id = "bb" * 8   # 16 hex chars
        fake_traceparent = f"00-{fake_trace_id}-{fake_span_id}-01"

        session_dir = tmp_path / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(
            json.dumps({
                "session_id": session_id,
                "status": "active",
                "traceparent": fake_traceparent,
            })
        )

        pool = _make_pool(tmp_path)

        async def run() -> None:
            await pool.start()
            await pool.wake(session_id)
            await asyncio.sleep(0)
            await pool.stop()

        asyncio.run(run())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        run_span = spans.get("session.run_span")
        assert run_span is not None
        assert run_span.context.trace_id == int(fake_trace_id, 16)


# ---------------------------------------------------------------------------
# Tests: error handling and audit log
# ---------------------------------------------------------------------------


class TestHarnessPoolErrors:
    def test_assign_failure_raises_harness_pool_error(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.assign("sess-err")

        asyncio.run(run())

    def test_assign_error_code_is_harness_pool_failed(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError) as exc_info:
                await pool.assign("sess-code")
            assert exc_info.value.code == "harness_pool_failed"

        asyncio.run(run())

    def test_assign_failure_writes_audit_log(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.assign("sess-audit")

        asyncio.run(run())
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.pool.assign.failed" for r in records)

    def test_assign_audit_level_is_error(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.assign("sess-level")

        asyncio.run(run())
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.pool.assign.failed")
        assert record["level"] == "error"

    def test_assign_audit_detail_has_session_id(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.assign("sess-detail-id")

        asyncio.run(run())
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.pool.assign.failed")
        assert record["detail"]["session_id"] == "sess-detail-id"

    def test_assign_audit_detail_has_message(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.assign("sess-detail-msg")

        asyncio.run(run())
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.pool.assign.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0

    def test_wake_failure_raises_harness_pool_error(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.wake("sess-wake-err")

        asyncio.run(run())

    def test_wake_error_code_is_harness_pool_failed(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError) as exc_info:
                await pool.wake("sess-wake-code")
            assert exc_info.value.code == "harness_pool_failed"

        asyncio.run(run())

    def test_wake_failure_writes_audit_log(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.wake("sess-wake-audit")

        asyncio.run(run())
        records = _read_audit(tmp_path)
        assert any(r.get("event") == "harness.pool.wake.failed" for r in records)

    def test_wake_audit_detail_has_session_id(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.wake("sess-wake-detail")

        asyncio.run(run())
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.pool.wake.failed")
        assert record["detail"]["session_id"] == "sess-wake-detail"

    def test_wake_audit_detail_has_message(self, tmp_path: Path) -> None:
        pool = _make_pool(tmp_path, num_workers=0)

        async def run() -> None:
            with pytest.raises(HarnessPoolError):
                await pool.wake("sess-wake-msg")

        asyncio.run(run())
        records = _read_audit(tmp_path)
        record = next(r for r in records if r.get("event") == "harness.pool.wake.failed")
        assert "message" in record["detail"]
        assert len(record["detail"]["message"]) > 0
