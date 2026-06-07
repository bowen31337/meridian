"""Memory-anniversary trigger firing logic.

Evaluates whether a memory_anniversary cron trigger should fire today
(i.e. whether today is exactly `days_before` days before the annual
recurrence of a date-typed memory value).  Emits an OpenTelemetry span
and writes a structured audit-log entry on every invocation, whether the
trigger fires, skips, or fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from typing import Any

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


def _today() -> date:
    return datetime.now(UTC).date()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class MemoryAnniversaryFireError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="memory_anniversary_fire_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class MemoryNotFoundError(MeridianError):
    def __init__(self, *, memory_key: str, timestamp: str) -> None:
        super().__init__(
            code="memory_not_found",
            message=f"Memory key '{memory_key}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class MemoryValueNotDateError(MeridianError):
    def __init__(self, *, memory_key: str, value: str, timestamp: str) -> None:
        super().__init__(
            code="memory_value_not_date",
            message=(
                f"Memory key '{memory_key}' value '{value}' is not a valid "
                "date (expected YYYY-MM-DD)"
            ),
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerFireResult:
    fired: bool
    cron_id: str
    memory_key: str
    next_fire_date: date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_key(key: str) -> str:
    """Return a filesystem-safe version of a memory key."""
    return key.replace("/", "_").replace("\x00", "_")


def _load_memory_date(storage_root: Path, memory_key: str, timestamp: str) -> date:
    safe_key = _sanitize_key(memory_key)
    memory_file = storage_root / "memory" / f"{safe_key}.json"

    if not memory_file.exists():
        raise MemoryNotFoundError(memory_key=memory_key, timestamp=timestamp)

    data = json.loads(memory_file.read_text())
    value = data.get("value", "")

    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        raise MemoryValueNotDateError(
            memory_key=memory_key, value=str(value), timestamp=timestamp
        ) from None


def _next_anniversary_fire_date(anniversary: date, days_before: int, today: date) -> date:
    """Return the next date on which the trigger should fire.

    The trigger fires `days_before` days before each annual recurrence of
    `anniversary`.  When `days_before` pushes the fire date across a
    year boundary (e.g. Jan 3 birthday − 10 days = Dec 24 of the prior
    year), the calculation still returns the correct calendar date.

    Feb 29 anniversaries are mapped to Feb 28 in non-leap years.
    """
    # We need the smallest fire_date >= today.
    # fire_date = anniversary.replace(year=Y) − days_before
    # ⟺ anniversary.replace(year=Y) >= today + days_before
    cutoff = today + timedelta(days=days_before)

    for year_offset in range(3):
        year = cutoff.year + year_offset
        try:
            ann = anniversary.replace(year=year)
        except ValueError:
            ann = date(year, 2, 28)
        if ann >= cutoff:
            return ann - timedelta(days=days_before)

    raise AssertionError("unreachable: 3-year window always contains a valid anniversary")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fire_memory_anniversary_trigger(
    cron_resource: dict[str, Any],
    *,
    storage_root: Path,
    audit_log: AuditLog,
    today: date | None = None,
) -> TriggerFireResult:
    """Evaluate and, if warranted, fire a memory_anniversary cron trigger.

    Reads the date-typed memory identified by ``cron_resource["memory_key"]``
    from ``storage_root/memory/{key}.json``, calculates the next annual fire
    date (``days_before`` days before the anniversary), and returns a
    :class:`TriggerFireResult` indicating whether today is that day.

    Always emits an OpenTelemetry span named
    ``trigger.memory_anniversary.fire``.  On a successful fire writes an
    ``info``-level audit entry; on any error writes an ``error``-level entry
    and re-raises the exception so the caller can surface it.
    """
    if today is None:
        today = _today()

    cron_id: str = cron_resource["id"]
    memory_key: str = cron_resource["memory_key"]
    days_before: int = cron_resource["days_before"]

    tracer = get_tracer()

    with tracer.start_as_current_span(
        "trigger.memory_anniversary.fire",
        attributes={
            "cron.id": cron_id,
            "cron.memory_key": memory_key,
            "cron.days_before": days_before,
        },
    ) as span:
        now = _now()
        record_invocation_event(
            span,
            StructuredEvent(
                name="trigger.memory_anniversary.fire.invocation",
                code="trigger_memory_anniversary_fire",
                timestamp=now,
            ),
        )

        try:
            anniversary_date = _load_memory_date(storage_root, memory_key, now)
            next_fire_date = _next_anniversary_fire_date(anniversary_date, days_before, today)
            fired = next_fire_date == today

            span.set_attribute("trigger.fired", fired)
            span.set_attribute("trigger.next_fire_date", next_fire_date.isoformat())

            if fired:
                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="trigger.memory_anniversary.fired",
                        code="trigger_memory_anniversary_fired",
                        timestamp=now,
                        detail={
                            "cron_id": cron_id,
                            "memory_key": memory_key,
                            "days_before": days_before,
                            "anniversary_date": anniversary_date.isoformat(),
                            "fire_date": today.isoformat(),
                        },
                    )
                )

            return TriggerFireResult(
                fired=fired,
                cron_id=cron_id,
                memory_key=memory_key,
                next_fire_date=next_fire_date,
            )

        except (MemoryNotFoundError, MemoryValueNotDateError) as err:
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="trigger.memory_anniversary.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "cron_id": cron_id,
                        "memory_key": memory_key,
                        "message": err.message,
                    },
                )
            )
            raise

        except Exception as exc:
            err2 = MemoryAnniversaryFireError(
                message=f"Failed to fire memory_anniversary trigger: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err2)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="trigger.memory_anniversary.failed",
                    code=err2.code,
                    timestamp=err2.timestamp,
                    detail={
                        "cron_id": cron_id,
                        "memory_key": memory_key,
                        "message": err2.message,
                    },
                )
            )
            raise err2 from exc
