"""
Cron scheduler conformance suite.

Tests cover:
  - fire_cron_trigger writes a fire record JSON under cron/fires/{cron_id}/.
  - Fire record contains fire_id, cron_id, session_id, trigger_type, capabilities, fired_at, status.
  - fire_cron_trigger returns the fire_id.
  - Fire record capabilities match the resource's declared capabilities.
  - fire_cron_trigger emits OTel span "cron.scheduler.fire".
  - OTel span carries cron.id, cron.session_id, cron.trigger_type, cron.fire_id attributes.
  - fire_cron_trigger writes audit entry with event "cron.scheduler.fired" on success.
  - Fired audit entry level is "info".
  - Fired audit detail contains cron_id, session_id, fire_id, trigger_type, capabilities.
  - fire_cron_trigger writes audit entry "cron.scheduler.fire.failed" on error.
  - Failed fire audit entry level is "error".
  - run_cron_scheduler_loop fires an interval cron whose next_fire_at has arrived.
  - Scheduler updates next_fire_at after firing an interval cron.
  - Scheduler marks a timestamp cron as status="fired" after firing.
  - Scheduler sets next_fire_at=null for a fired timestamp cron.
  - Scheduler does not fire event-driven triggers (channel_event, etc.).
  - Scheduler does not fire inactive (non-active) crons.
  - Scheduler fires a cron whose next_fire_at was set in the past (daemon restart recovery).
  - Missed fires, catch_up policy: fires once per missed interval slot.
  - Missed fires, skip policy: fires once for current slot, advances schedule.
  - next_fire_at persisted to disk after each fire.
  - Scheduler ignores malformed cron JSON files.
  - Multiple fires have unique fire_ids.
  - Capabilities inherited in fire record: fire record has exactly declared capabilities.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from meridiand._audit import FileAuditLog
from meridiand._cron_scheduler import CronFireError, fire_cron_trigger, run_cron_scheduler_loop

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fire_records(storage_root: Path, cron_id: str) -> list[dict]:
    fires_dir = storage_root / "cron" / "fires" / cron_id
    if not fires_dir.exists():
        return []
    records = []
    for f in sorted(fires_dir.glob("fire_*.json")):
        records.append(json.loads(f.read_text()))
    return records


def _make_resource(
    cron_id: str = "cron_test",
    session_id: str = "sess-abc",
    trigger_type: str = "interval",
    interval: str = "5m",
    capabilities: list[str] | None = None,
    missed_fires_policy: str = "skip",
    next_fire_at: str | None = None,
    status: str = "active",
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "id": cron_id,
        "trigger_type": trigger_type,
        "session_id": session_id,
        "name": None,
        "status": status,
        "created_at": now,
        "next_fire_at": next_fire_at or (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        "missed_fires_policy": missed_fires_policy,
        "capabilities": capabilities or [],
        "interval": interval if trigger_type == "interval" else None,
        "timestamp": None,
        "channel_id": None,
        "path": None,
        "webhook_id": None,
        "memory_key": None,
        "days_before": None,
        "metadata": None,
    }


def _write_cron(storage_root: Path, resource: dict[str, Any]) -> Path:
    cron_dir = storage_root / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    cron_file = cron_dir / f"{resource['id']}.json"
    cron_file.write_text(json.dumps(resource))
    return cron_file


async def _run_one_tick(
    storage_root: Path,
    audit_log: FileAuditLog,
    missed_fires_policy: str = "skip",
) -> None:
    """Run the scheduler loop for a single tick then cancel."""
    task = asyncio.create_task(
        run_cron_scheduler_loop(
            storage_root,
            audit_log,
            missed_fires_policy=missed_fires_policy,
            check_interval_seconds=9999.0,  # won't actually sleep during test
        )
    )
    # Give the event loop a turn to execute the loop body.
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# fire_cron_trigger: fire record
# ---------------------------------------------------------------------------


class TestFireCronTriggerRecord:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_fire_record_written(self, storage_root: Path, tmp_path: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert len(records) == 1

    def test_fire_record_has_fire_id(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        fire_id = asyncio.run(
            fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log)
        )
        records = _fire_records(storage_root, "cron_test")
        assert records[0]["fire_id"] == fire_id

    def test_fire_record_has_cron_id(self, storage_root: Path) -> None:
        resource = _make_resource(cron_id="cron_abc")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_abc")
        assert records[0]["cron_id"] == "cron_abc"

    def test_fire_record_has_session_id(self, storage_root: Path) -> None:
        resource = _make_resource(session_id="sess-xyz")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert records[0]["session_id"] == "sess-xyz"

    def test_fire_record_has_trigger_type(self, storage_root: Path) -> None:
        resource = _make_resource(trigger_type="interval")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert records[0]["trigger_type"] == "interval"

    def test_fire_record_has_fired_at(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert "fired_at" in records[0]
        assert len(records[0]["fired_at"]) > 0

    def test_fire_record_status_is_pending(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert records[0]["status"] == "pending"

    def test_fire_record_returns_fire_id(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        fire_id = asyncio.run(
            fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log)
        )
        assert fire_id.startswith("fire_")

    def test_fire_ids_are_unique(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        id1 = asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        id2 = asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        assert id1 != id2


# ---------------------------------------------------------------------------
# fire_cron_trigger: capability inheritance
# ---------------------------------------------------------------------------


class TestFireCronTriggerCapabilities:
    def test_fire_record_inherits_empty_capabilities(self, storage_root: Path) -> None:
        resource = _make_resource(capabilities=[])
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert records[0]["capabilities"] == []

    def test_fire_record_inherits_declared_capabilities(self, storage_root: Path) -> None:
        caps = ["agent.read", "agent.write"]
        resource = _make_resource(capabilities=caps)
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert records[0]["capabilities"] == caps

    def test_fire_record_does_not_escalate_capabilities(self, storage_root: Path) -> None:
        """Fire record must have exactly the declared capabilities, no more."""
        caps = ["agent.read"]
        resource = _make_resource(capabilities=caps)
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _fire_records(storage_root, "cron_test")
        assert set(records[0]["capabilities"]) == set(caps)


# ---------------------------------------------------------------------------
# fire_cron_trigger: OTel
# ---------------------------------------------------------------------------


class TestFireCronTriggerOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_emits_cron_scheduler_fire_span(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "cron.scheduler.fire" in span_names

    def test_span_has_cron_id_attribute(self, storage_root: Path) -> None:
        resource = _make_resource(cron_id="cron_span_test")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.scheduler.fire")
        assert span is not None
        assert span.attributes["cron.id"] == "cron_span_test"

    def test_span_has_session_id_attribute(self, storage_root: Path) -> None:
        resource = _make_resource(session_id="sess-otel")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.scheduler.fire")
        assert span is not None
        assert span.attributes["cron.session_id"] == "sess-otel"

    def test_span_has_trigger_type_attribute(self, storage_root: Path) -> None:
        resource = _make_resource(trigger_type="interval")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.scheduler.fire")
        assert span is not None
        assert span.attributes["cron.trigger_type"] == "interval"

    def test_span_has_fire_id_attribute(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        fire_id = asyncio.run(
            fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log)
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("cron.scheduler.fire")
        assert span is not None
        assert span.attributes["cron.fire_id"] == fire_id


# ---------------------------------------------------------------------------
# fire_cron_trigger: audit log
# ---------------------------------------------------------------------------


class TestFireCronTriggerAudit:
    def test_success_writes_fired_audit_entry(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "cron.scheduler.fired" for r in records)

    def test_fired_audit_level_is_info(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.scheduler.fired")
        assert record["level"] == "info"

    def test_fired_audit_detail_has_cron_id(self, storage_root: Path) -> None:
        resource = _make_resource(cron_id="cron_audit_test")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.scheduler.fired")
        assert record["detail"]["cron_id"] == "cron_audit_test"

    def test_fired_audit_detail_has_session_id(self, storage_root: Path) -> None:
        resource = _make_resource(session_id="sess-audit")
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.scheduler.fired")
        assert record["detail"]["session_id"] == "sess-audit"

    def test_fired_audit_detail_has_fire_id(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        fire_id = asyncio.run(
            fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log)
        )
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.scheduler.fired")
        assert record["detail"]["fire_id"] == fire_id

    def test_fired_audit_detail_has_capabilities(self, storage_root: Path) -> None:
        caps = ["agent.read"]
        resource = _make_resource(capabilities=caps)
        fires_dir = storage_root / "cron" / "fires"
        audit_log = FileAuditLog(storage_root)
        asyncio.run(fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log))
        record = next(r for r in _audit_records(storage_root) if r.get("event") == "cron.scheduler.fired")
        assert record["detail"]["capabilities"] == caps

    def test_failure_writes_failed_audit_entry(self, storage_root: Path) -> None:
        resource = _make_resource()
        # Use a fires_dir path whose parent doesn't exist and can't be created.
        fires_dir = Path("/nonexistent/path/fires")
        audit_log = FileAuditLog(storage_root)
        with pytest.raises(CronFireError):
            asyncio.run(
                fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log)
            )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "cron.scheduler.fire.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        resource = _make_resource()
        fires_dir = Path("/nonexistent/path/fires")
        audit_log = FileAuditLog(storage_root)
        with pytest.raises(CronFireError):
            asyncio.run(
                fire_cron_trigger(resource, fires_dir=fires_dir, audit_log=audit_log)
            )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "cron.scheduler.fire.failed"
        )
        assert record["level"] == "error"


# ---------------------------------------------------------------------------
# Scheduler loop: interval triggers
# ---------------------------------------------------------------------------


class TestSchedulerIntervalFires:
    def test_fires_interval_cron_when_due(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        resource = _make_resource(next_fire_at=past)
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        assert len(_fire_records(storage_root, "cron_test")) >= 1

    def test_does_not_fire_interval_cron_not_yet_due(self, storage_root: Path) -> None:
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        resource = _make_resource(next_fire_at=future)
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        assert len(_fire_records(storage_root, "cron_test")) == 0

    def test_updates_next_fire_at_after_interval_fire(self, storage_root: Path) -> None:
        before = datetime.now(UTC) - timedelta(seconds=10)
        resource = _make_resource(interval="5m", next_fire_at=before.isoformat())
        cron_file = _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        updated = json.loads(cron_file.read_text())
        new_next = datetime.fromisoformat(updated["next_fire_at"])
        assert new_next > datetime.now(UTC)

    def test_next_fire_at_advances_by_interval(self, storage_root: Path) -> None:
        before = datetime.now(UTC) - timedelta(seconds=5)
        resource = _make_resource(interval="5m", next_fire_at=before.isoformat())
        cron_file = _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        updated = json.loads(cron_file.read_text())
        new_next = datetime.fromisoformat(updated["next_fire_at"])
        expected_next = before + timedelta(minutes=5)
        # next_fire_at should be close to before + 5m (±2s tolerance)
        assert abs((new_next - expected_next).total_seconds()) < 2.0

    def test_status_remains_active_after_interval_fire(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        resource = _make_resource(next_fire_at=past)
        cron_file = _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        updated = json.loads(cron_file.read_text())
        assert updated["status"] == "active"


# ---------------------------------------------------------------------------
# Scheduler loop: timestamp triggers
# ---------------------------------------------------------------------------


class TestSchedulerTimestampFires:
    def test_fires_timestamp_cron_when_due(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        resource = _make_resource(
            trigger_type="timestamp",
            next_fire_at=past,
        )
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        assert len(_fire_records(storage_root, "cron_test")) == 1

    def test_marks_timestamp_cron_as_fired(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        resource = _make_resource(trigger_type="timestamp", next_fire_at=past)
        cron_file = _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        updated = json.loads(cron_file.read_text())
        assert updated["status"] == "fired"

    def test_timestamp_next_fire_at_null_after_fire(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        resource = _make_resource(trigger_type="timestamp", next_fire_at=past)
        cron_file = _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        updated = json.loads(cron_file.read_text())
        assert updated["next_fire_at"] is None

    def test_fired_timestamp_cron_not_refired(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
        resource = _make_resource(trigger_type="timestamp", next_fire_at=past)
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        asyncio.run(_run_one_tick(storage_root, audit_log))
        assert len(_fire_records(storage_root, "cron_test")) == 1


# ---------------------------------------------------------------------------
# Scheduler loop: non-firing cases
# ---------------------------------------------------------------------------


class TestSchedulerNonFiringCases:
    def test_does_not_fire_channel_event_trigger(self, storage_root: Path) -> None:
        resource = _make_resource()
        resource["trigger_type"] = "channel_event"
        resource["next_fire_at"] = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        assert len(_fire_records(storage_root, "cron_test")) == 0

    def test_does_not_fire_inactive_cron(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        resource = _make_resource(next_fire_at=past, status="fired")
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        assert len(_fire_records(storage_root, "cron_test")) == 0

    def test_ignores_malformed_json_file(self, storage_root: Path) -> None:
        cron_dir = storage_root / "cron"
        cron_dir.mkdir(parents=True, exist_ok=True)
        (cron_dir / "cron_bad.json").write_text("not valid json{")
        audit_log = FileAuditLog(storage_root)
        # Should not raise; malformed file is silently skipped.
        asyncio.run(_run_one_tick(storage_root, audit_log))

    def test_does_not_fire_cron_without_next_fire_at(self, storage_root: Path) -> None:
        resource = _make_resource()
        resource["next_fire_at"] = None
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        assert len(_fire_records(storage_root, "cron_test")) == 0


# ---------------------------------------------------------------------------
# Scheduler loop: missed fires policy
# ---------------------------------------------------------------------------


class TestSchedulerMissedFiresPolicy:
    def test_catch_up_fires_multiple_times_for_missed_slots(
        self, storage_root: Path
    ) -> None:
        # Missed 3 intervals of 1 second each.
        past = (datetime.now(UTC) - timedelta(seconds=3)).isoformat()
        resource = _make_resource(
            interval="1s", next_fire_at=past, missed_fires_policy="catch_up"
        )
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, missed_fires_policy="catch_up"))
        fires = _fire_records(storage_root, "cron_test")
        assert len(fires) >= 3

    def test_skip_fires_once_for_missed_slots(self, storage_root: Path) -> None:
        # Missed 3 intervals of 1 second each.
        past = (datetime.now(UTC) - timedelta(seconds=3)).isoformat()
        resource = _make_resource(
            interval="1s", next_fire_at=past, missed_fires_policy="skip"
        )
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, missed_fires_policy="skip"))
        fires = _fire_records(storage_root, "cron_test")
        assert len(fires) == 1

    def test_skip_advances_next_fire_at_past_missed_slots(
        self, storage_root: Path
    ) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=3)).isoformat()
        resource = _make_resource(
            interval="1s", next_fire_at=past, missed_fires_policy="skip"
        )
        cron_file = _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log, missed_fires_policy="skip"))
        updated = json.loads(cron_file.read_text())
        new_next = datetime.fromisoformat(updated["next_fire_at"])
        # new_next should be in the future, not in the past missed range.
        assert new_next > datetime.now(UTC)

    def test_resource_policy_overrides_loop_default(self, storage_root: Path) -> None:
        # Resource says "catch_up" but loop default is "skip".
        past = (datetime.now(UTC) - timedelta(seconds=3)).isoformat()
        resource = _make_resource(
            interval="1s", next_fire_at=past, missed_fires_policy="catch_up"
        )
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        # Loop default is "skip" — resource's "catch_up" should win.
        asyncio.run(_run_one_tick(storage_root, audit_log, missed_fires_policy="skip"))
        fires = _fire_records(storage_root, "cron_test")
        assert len(fires) >= 3


# ---------------------------------------------------------------------------
# Scheduler loop: daemon restart recovery
# ---------------------------------------------------------------------------


class TestSchedulerDaemonRestartRecovery:
    def test_recovers_and_fires_cron_after_restart(self, storage_root: Path) -> None:
        # Simulate daemon restart: cron was created, next_fire_at is in the past.
        past = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        resource = _make_resource(interval="5m", next_fire_at=past)
        _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        # Should have fired at least once.
        fires = _fire_records(storage_root, "cron_test")
        assert len(fires) >= 1

    def test_next_fire_at_updated_after_restart_recovery(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        resource = _make_resource(interval="5m", next_fire_at=past)
        cron_file = _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        updated = json.loads(cron_file.read_text())
        new_next = datetime.fromisoformat(updated["next_fire_at"])
        assert new_next > datetime.now(UTC)

    def test_multiple_crons_all_fire_on_restart(self, storage_root: Path) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
        for i in range(3):
            resource = _make_resource(cron_id=f"cron_multi_{i}", next_fire_at=past)
            _write_cron(storage_root, resource)
        audit_log = FileAuditLog(storage_root)
        asyncio.run(_run_one_tick(storage_root, audit_log))
        total = sum(len(_fire_records(storage_root, f"cron_multi_{i}")) for i in range(3))
        assert total == 3
