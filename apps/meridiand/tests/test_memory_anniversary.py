"""
Memory-anniversary trigger firing conformance suite.

Tests cover:
  - fire_memory_anniversary_trigger returns fired=True when today == fire date.
  - fire_memory_anniversary_trigger returns fired=False when today is before the fire date.
  - fire_memory_anniversary_trigger returns fired=False when today is after the fire date.
  - Annual recurrence: fires on the same date the following year.
  - days_before=0 fires on the anniversary day itself.
  - days_before=1 fires the day before the anniversary.
  - Cross-year fire date: birthday in January, days_before spans into December of prior year.
  - Feb 29 birthday is mapped to Feb 28 in non-leap years.
  - Missing memory key raises MemoryNotFoundError and writes an error audit entry.
  - Non-date memory value raises MemoryValueNotDateError and writes an error audit entry.
  - OTel span "trigger.memory_anniversary.fire" emitted on a successful fire.
  - OTel span "trigger.memory_anniversary.fire" emitted on a non-fire evaluation.
  - OTel span "trigger.memory_anniversary.fire" emitted on failure, with ERROR status.
  - Successful fire writes audit entry with event "trigger.memory_anniversary.fired".
  - Fired audit entry level is "info".
  - Fired audit detail contains cron_id, memory_key, days_before, anniversary_date, fire_date.
  - Failure audit entry written with event "trigger.memory_anniversary.failed".
  - Failure audit entry level is "error".
  - Failure audit detail contains cron_id, memory_key, message.
  - result.cron_id matches the cron resource id.
  - result.memory_key matches the memory_key field.
  - result.next_fire_date is the computed fire date.
  - No audit entry written when trigger is evaluated but does not fire.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from meridiand._audit import FileAuditLog
from meridiand._memory_anniversary import (
    MemoryNotFoundError,
    MemoryValueNotDateError,
    fire_memory_anniversary_trigger,
)
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_memory(storage_root: Path, key: str, value: str, type_: str = "date") -> None:
    mem_dir = storage_root / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_").replace("\x00", "_")
    (mem_dir / f"{safe_key}.json").write_text(
        json.dumps({"key": key, "value": value, "type": type_})
    )


def _make_resource(
    memory_key: str = "user.birthday",
    days_before: int = 3,
    cron_id: str = "cron_test001",
    session_id: str = "sess-abc",
) -> dict:
    return {
        "id": cron_id,
        "trigger_type": "memory_anniversary",
        "session_id": session_id,
        "memory_key": memory_key,
        "days_before": days_before,
        "status": "active",
    }


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Fire / no-fire decisions
# ---------------------------------------------------------------------------


class TestTriggerFireDecision:
    def test_fires_on_correct_day(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),  # 3 days before Mar 15
        )
        assert result.fired is True

    def test_does_not_fire_before_fire_date(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 11),  # one day too early
        )
        assert result.fired is False

    def test_does_not_fire_after_fire_date(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 13),  # one day too late (fire was Mar 12)
        )
        assert result.fired is False

    def test_days_before_zero_fires_on_anniversary(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.anniversary", "2015-06-20")
        resource = _make_resource(memory_key="user.anniversary", days_before=0)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 6, 20),
        )
        assert result.fired is True

    def test_days_before_one_fires_day_before_anniversary(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.anniversary", "2015-06-20")
        resource = _make_resource(memory_key="user.anniversary", days_before=1)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 6, 19),
        )
        assert result.fired is True


# ---------------------------------------------------------------------------
# Annual recurrence
# ---------------------------------------------------------------------------


class TestAnnualRecurrence:
    def test_fires_same_date_next_year(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-07-04")
        resource = _make_resource(memory_key="user.birthday", days_before=5)

        result_this_year = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 6, 29),  # 5 days before Jul 4 2026
        )
        result_next_year = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2027, 6, 29),  # 5 days before Jul 4 2027
        )
        assert result_this_year.fired is True
        assert result_next_year.fired is True

    def test_next_fire_date_is_next_year_after_anniversary_passed(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        # Today is Mar 13 — fire date for this year was Mar 12, already passed
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 13),
        )
        assert result.next_fire_date == date(2027, 3, 12)

    def test_cross_year_fire_date(self, storage_root: Path) -> None:
        # Birthday Jan 3, days_before=10 → fires Dec 24 of the prior year
        _write_memory(storage_root, "user.birthday", "1990-01-03")
        resource = _make_resource(memory_key="user.birthday", days_before=10)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 12, 24),  # 10 days before Jan 3 2027
        )
        assert result.fired is True
        assert result.next_fire_date == date(2026, 12, 24)


# ---------------------------------------------------------------------------
# Feb 29 edge cases
# ---------------------------------------------------------------------------


class TestLeapYearEdgeCases:
    def test_feb29_birthday_fires_on_feb28_in_non_leap_year(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1992-02-29")
        resource = _make_resource(memory_key="user.birthday", days_before=0)
        # 2027 is not a leap year; anniversary is mapped to Feb 28
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2027, 2, 28),
        )
        assert result.fired is True

    def test_feb29_birthday_fires_on_feb29_in_leap_year(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1992-02-29")
        resource = _make_resource(memory_key="user.birthday", days_before=0)
        # 2028 is a leap year
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2028, 2, 29),
        )
        assert result.fired is True


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


class TestTriggerFireResult:
    def test_result_cron_id(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(cron_id="cron_abc123", days_before=3)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        assert result.cron_id == "cron_abc123"

    def test_result_memory_key(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(memory_key="user.birthday", days_before=3)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        assert result.memory_key == "user.birthday"

    def test_result_next_fire_date_on_fire(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        assert result.next_fire_date == date(2026, 3, 12)

    def test_result_next_fire_date_before_fire(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        result = fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 1),
        )
        assert result.next_fire_date == date(2026, 3, 12)


# ---------------------------------------------------------------------------
# Audit log — fire
# ---------------------------------------------------------------------------


class TestAuditLogFire:
    def test_fired_writes_audit_entry(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "trigger.memory_anniversary.fired" for r in records)

    def test_fired_audit_level_is_info(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.fired"
        )
        assert record["level"] == "info"

    def test_fired_audit_detail_has_cron_id(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(cron_id="cron_det001", days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.fired"
        )
        assert record["detail"]["cron_id"] == "cron_det001"

    def test_fired_audit_detail_has_memory_key(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(memory_key="user.birthday", days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.fired"
        )
        assert record["detail"]["memory_key"] == "user.birthday"

    def test_fired_audit_detail_has_days_before(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.fired"
        )
        assert record["detail"]["days_before"] == 3

    def test_fired_audit_detail_has_anniversary_date(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.fired"
        )
        assert record["detail"]["anniversary_date"] == "1990-03-15"

    def test_fired_audit_detail_has_fire_date(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.fired"
        )
        assert record["detail"]["fire_date"] == "2026-03-12"

    def test_no_audit_entry_when_not_fired(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 1),  # too early — does not fire
        )
        records = _audit_records(storage_root)
        assert not any(r.get("event") == "trigger.memory_anniversary.fired" for r in records)


# ---------------------------------------------------------------------------
# Audit log — failure
# ---------------------------------------------------------------------------


class TestAuditLogFailure:
    def test_missing_memory_writes_error_audit(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "trigger.memory_anniversary.failed" for r in records)

    def test_missing_memory_audit_level_is_error(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.failed"
        )
        assert record["level"] == "error"

    def test_missing_memory_audit_code(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.failed"
        )
        assert record["code"] == "memory_not_found"

    def test_missing_memory_audit_detail_has_cron_id(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing", cron_id="cron_fail001")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.failed"
        )
        assert record["detail"]["cron_id"] == "cron_fail001"

    def test_missing_memory_audit_detail_has_memory_key(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.failed"
        )
        assert record["detail"]["memory_key"] == "user.missing"

    def test_missing_memory_audit_detail_has_message(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "trigger.memory_anniversary.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_non_date_value_raises_and_writes_audit(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "not-a-date")
        resource = _make_resource()
        with pytest.raises(MemoryValueNotDateError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        records = _audit_records(storage_root)
        assert any(r.get("code") == "memory_value_not_date" for r in records)

    def test_missing_memory_raises_memory_not_found(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_fire_emits_span(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "trigger.memory_anniversary.fire" in span_names

    def test_no_fire_emits_span(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 1),
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "trigger.memory_anniversary.fire" in span_names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "trigger.memory_anniversary.fire" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        resource = _make_resource(memory_key="user.missing")
        with pytest.raises(MemoryNotFoundError):
            fire_memory_anniversary_trigger(
                resource,
                storage_root=storage_root,
                audit_log=FileAuditLog(storage_root),
                today=date(2026, 3, 12),
            )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("trigger.memory_anniversary.fire")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_cron_id_attribute(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(cron_id="cron_span001", days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("trigger.memory_anniversary.fire")
        assert span is not None
        assert span.attributes["cron.id"] == "cron_span001"

    def test_span_has_memory_key_attribute(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(memory_key="user.birthday", days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("trigger.memory_anniversary.fire")
        assert span is not None
        assert span.attributes["cron.memory_key"] == "user.birthday"

    def test_span_has_days_before_attribute(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("trigger.memory_anniversary.fire")
        assert span is not None
        assert span.attributes["cron.days_before"] == 3

    def test_span_has_fired_attribute_true(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 12),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("trigger.memory_anniversary.fire")
        assert span is not None
        assert span.attributes["trigger.fired"] is True

    def test_span_has_fired_attribute_false(self, storage_root: Path) -> None:
        _write_memory(storage_root, "user.birthday", "1990-03-15")
        resource = _make_resource(days_before=3)
        fire_memory_anniversary_trigger(
            resource,
            storage_root=storage_root,
            audit_log=FileAuditLog(storage_root),
            today=date(2026, 3, 1),
        )
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("trigger.memory_anniversary.fire")
        assert span is not None
        assert span.attributes["trigger.fired"] is False
