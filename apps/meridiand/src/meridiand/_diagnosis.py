"""GET /v1/sessions/{id}/diagnosis — aggregated failure summary for < 5 min MTTR.

Reads the event log, audit log, and replay fixture directory for a session and
returns a structured postmortem: terminal phase, stop reason, all failure-class
events (error / phase_change / tool_call.vetoed / budget.warning / message.truncated),
session-scoped audit entries, replay fixture availability, and total event count.
On failure surfaces an error message to the caller and writes the failure to the
audit log.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from storage_reposit import LocalEventLogReader


def _now() -> str:
    return datetime.now(UTC).isoformat()


_FAILURE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "error",
        "session.phase_change",
        "tool_call.vetoed",
        "budget.warning",
        "message.truncated",
    }
)


class SessionDiagnosisError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_diagnosis_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


def _read_audit_for_session(audit_path: Path, session_id: str) -> list[dict[str, Any]]:
    """Return audit.ndjson entries whose detail.session_id matches session_id."""
    if not audit_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for raw in audit_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            record: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        detail = record.get("detail") or {}
        if detail.get("session_id") == session_id:
            entries.append(record)
    return entries


def _extract_failure_summary(
    events: list[Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    """Return (terminal_phase, stop_reason, failure_events) from the event list."""
    terminal_phase = "unknown"
    stop_reason = ""
    failure_events: list[dict[str, Any]] = []

    for event in events:
        if event.type in _FAILURE_EVENT_TYPES:
            d: dict[str, Any] = {
                "seq": event.seq,
                "ts": event.ts,
                "type": event.type,
                "data": event.data,
            }
            if event.thread_id is not None:
                d["thread_id"] = event.thread_id
            failure_events.append(d)

        if event.type == "session.phase_change":
            data = event.data or {}
            after = data.get("after", "")
            if after:
                terminal_phase = after
            reason = data.get("reason", "")
            if reason:
                stop_reason = reason

    return terminal_phase, stop_reason, failure_events


def make_diagnosis_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/sessions/{session_id}/diagnosis")
    async def diagnose_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.diagnosis",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.diagnosis.invocation",
                    code="session_diagnosis",
                    timestamp=now,
                ),
            )

            try:
                reader = LocalEventLogReader(storage_root)
                events = reader.read_after(session_id, -1)

                terminal_phase, stop_reason, failure_events = _extract_failure_summary(events)

                audit_entries = _read_audit_for_session(storage_root / "audit.ndjson", session_id)

                replay_fixture_available = (
                    storage_root / "fixtures" / session_id / "model_responses.ndjson"
                ).exists()

            except SessionDiagnosisError:
                raise
            except Exception as exc:
                err = SessionDiagnosisError(
                    message=f"Failed to diagnose session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.diagnosis.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "session_id": session_id,
                "diagnosed_at": now,
                "terminal_phase": terminal_phase,
                "stop_reason": stop_reason,
                "failure_events": failure_events,
                "audit_entries": audit_entries,
                "replay_fixture_available": replay_fixture_available,
                "event_count": len(events),
            }
        )

    return router
