"""Durable cron scheduler.

Persists ``next_fire_at`` in the cron resource JSON file so the scheduler
survives daemon restarts.  Missed fires are handled according to the
``missed_fires_policy`` stored on each cron resource:

- ``"catch_up"``: fire once per missed interval slot.
- ``"skip"``:     skip the missed slots; advance ``next_fire_at`` to the next
                  future slot without firing.

Only ``timestamp`` and ``interval`` trigger types are time-driven and handled
by this scheduler.  Event-driven triggers (``channel_event``, ``file_change``,
``webhook``, ``memory_anniversary``) are fired by their respective subsystems.

Capability contract: cron-triggered sessions inherit the capabilities declared
at cron-creation time (stored in ``resource["capabilities"]``); the scheduler
never escalates beyond those declared capabilities.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
import uuid

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)

from ._cron import _parse_duration

# A cron fire executor runs the fired cron as an agent turn and returns an
# outcome dict ({"status": "completed"|"error"|"skipped", ...}) the scheduler
# stamps onto the fire record. None -> the legacy record-only behaviour.
CronFireExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class CronFireError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="cron_fire_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Fire a single cron trigger
# ---------------------------------------------------------------------------


async def fire_cron_trigger(
    resource: dict[str, Any],
    *,
    fires_dir: Path,
    audit_log: AuditLog,
    executor: CronFireExecutor | None = None,
) -> str:
    """Fire a cron trigger: write a fire record, emit an OTel span, and log to audit.

    The fire record inherits exactly the capabilities declared in the cron
    resource at creation time; no escalation is possible.

    Returns the ``fire_id``.  Raises :class:`CronFireError` on any failure;
    the error is recorded to the span and audit log before re-raising.
    """
    cron_id: str = resource["id"]
    session_id: str = resource["session_id"]
    trigger_type: str = resource["trigger_type"]
    # Inherit declared capabilities; never escalate beyond what was declared.
    capabilities: list[str] = resource.get("capabilities") or []
    fire_id = f"fire_{uuid.uuid4().hex}"
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "cron.scheduler.fire",
        attributes={
            "cron.id": cron_id,
            "cron.trigger_type": trigger_type,
            "cron.session_id": session_id,
            "cron.fire_id": fire_id,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="cron.scheduler.fire.invocation",
                code="cron_scheduler_fire",
                timestamp=now,
            ),
        )

        try:
            cron_fire_dir = fires_dir / cron_id
            cron_fire_dir.mkdir(parents=True, exist_ok=True)
            fire_record: dict[str, Any] = {
                "fire_id": fire_id,
                "cron_id": cron_id,
                "session_id": session_id,
                "trigger_type": trigger_type,
                # Declared capabilities passed through unchanged — no escalation.
                "capabilities": capabilities,
                "fired_at": now,
                "status": "pending",
            }
            (cron_fire_dir / f"{fire_id}.json").write_text(json.dumps(fire_record))

            span.set_attribute("cron.capabilities_count", len(capabilities))
            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="cron.scheduler.fired",
                    code="cron_scheduler_fired",
                    timestamp=now,
                    detail={
                        "cron_id": cron_id,
                        "session_id": session_id,
                        "fire_id": fire_id,
                        "trigger_type": trigger_type,
                        "capabilities": capabilities,
                    },
                )
            )

            # Execute the fire as an agent turn, recording the outcome on the
            # fire record (pending -> completed / skipped / error). Without an
            # executor the fire stays a pending record (legacy behaviour).
            if executor is not None:
                outcome = await executor(resource)
                fire_record["status"] = outcome.get("status", "completed")
                for key in ("output", "error", "reason"):
                    if key in outcome:
                        fire_record[key] = outcome[key]
                completed_at = _now()
                fire_record["completed_at"] = completed_at
                (cron_fire_dir / f"{fire_id}.json").write_text(json.dumps(fire_record))
                audit_log.write(
                    AuditLogEntry(
                        level="error" if fire_record["status"] == "error" else "info",
                        event="cron.scheduler.executed",
                        code="cron_scheduler_executed",
                        timestamp=completed_at,
                        detail={
                            "cron_id": cron_id,
                            "session_id": session_id,
                            "fire_id": fire_id,
                            "status": fire_record["status"],
                            "detail": str(outcome.get("error") or outcome.get("reason") or "")[
                                :200
                            ],
                        },
                    )
                )

        except Exception as exc:
            err = CronFireError(
                message=f"Failed to fire cron {cron_id!r}: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="cron.scheduler.fire.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "cron_id": cron_id,
                        "session_id": session_id,
                        "fire_id": fire_id,
                        "message": err.message,
                    },
                )
            )
            raise err from exc

    return fire_id


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------


def _load_resource(cron_file: Path) -> dict[str, Any] | None:
    try:
        return json.loads(cron_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


async def run_cron_scheduler_loop(
    storage_root: Path,
    audit_log: AuditLog,
    *,
    missed_fires_policy: str = "skip",
    check_interval_seconds: float = 5.0,
    executor: CronFireExecutor | None = None,
) -> None:
    """Background scheduler loop for time-based cron triggers.

    Wakes every ``check_interval_seconds`` seconds and fires any
    ``timestamp`` or ``interval`` crons whose ``next_fire_at`` has arrived.

    On daemon restart, crons whose ``next_fire_at`` is in the past are treated
    as missed fires and handled per their ``missed_fires_policy`` (falling back
    to the ``missed_fires_policy`` argument when not set on the resource):

    - ``"catch_up"``: fire once for each missed interval slot.
    - ``"skip"``:     advance the schedule past missed slots without firing.

    Persists the updated ``next_fire_at`` (and ``status`` for one-shot
    triggers) back to the cron JSON file after every iteration so the state
    survives the next restart.
    """
    cron_dir = storage_root / "cron"
    fires_dir = cron_dir / "fires"

    while True:
        if cron_dir.exists():
            now_dt = datetime.now(UTC)

            for cron_file in sorted(cron_dir.glob("cron_*.json")):
                resource = _load_resource(cron_file)
                if resource is None:
                    continue
                if resource.get("status") != "active":
                    continue

                trigger_type = resource.get("trigger_type")
                if trigger_type not in ("timestamp", "interval"):
                    continue

                next_fire_at_str = resource.get("next_fire_at")
                if next_fire_at_str is None:
                    continue

                try:
                    next_fire_dt = datetime.fromisoformat(next_fire_at_str)
                except ValueError:
                    continue

                if next_fire_dt > now_dt:
                    continue  # not yet due

                policy: str = resource.get("missed_fires_policy") or missed_fires_policy

                if trigger_type == "timestamp":
                    # One-shot trigger: always fire when the time arrives.
                    # CronFireError is already logged inside fire_cron_trigger.
                    with contextlib.suppress(CronFireError):
                        await fire_cron_trigger(
                            resource, fires_dir=fires_dir, audit_log=audit_log, executor=executor
                        )
                    resource["status"] = "fired"
                    resource["fired_at"] = now_dt.isoformat()
                    resource["next_fire_at"] = None
                    cron_file.write_text(json.dumps(resource))

                else:
                    # trigger_type == "interval" (guaranteed by the guard above)
                    interval_str = resource.get("interval", "")
                    try:
                        delta = _parse_duration(interval_str)
                    except ValueError:
                        continue  # malformed interval; skip silently

                    # Always fire for the current due slot.
                    # CronFireError is already logged inside fire_cron_trigger.
                    with contextlib.suppress(CronFireError):
                        await fire_cron_trigger(
                            resource, fires_dir=fires_dir, audit_log=audit_log, executor=executor
                        )

                    new_next = next_fire_dt + delta

                    if policy == "catch_up":
                        # Fire for every additionally missed slot.
                        while new_next <= now_dt:
                            with contextlib.suppress(CronFireError):
                                await fire_cron_trigger(
                                    resource,
                                    fires_dir=fires_dir,
                                    audit_log=audit_log,
                                    executor=executor,
                                )
                            new_next += delta
                    else:
                        # Skip missed slots; advance to the next future slot.
                        while new_next <= now_dt:
                            new_next += delta

                    resource["next_fire_at"] = new_next.isoformat()
                    cron_file.write_text(json.dumps(resource))

        await asyncio.sleep(check_interval_seconds)
