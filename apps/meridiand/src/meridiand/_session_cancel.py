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
from opentelemetry import metrics
from storage_event_log import EventLogWriter
from storage_reposit import LocalEventLogReader, PhaseProjection

from ._metrics_registry import active_sessions, session_duration_seconds, sessions_total
from ._version import MERIDIAND_VERSION

_meter = metrics.get_meter("meridian.meridiand", MERIDIAND_VERSION)
_cancels_counter = _meter.create_counter(
    "meridian_session_cancels_total",
    description="Total number of session cancel invocations",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SessionCancelNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="session_cancel_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class SessionCancelError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_cancel_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _walk_descendants(session_id: str, storage_root: Path) -> list[str]:
    """BFS over manifest parent_session_id links; returns all descendant session IDs."""
    sessions_dir = storage_root / "sessions"
    if not sessions_dir.exists():
        return []

    children_map: dict[str, list[str]] = {}
    for manifest_path in sessions_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            parent = manifest.get("parent_session_id")
            child = manifest.get("child_session_id")
            if parent and child:
                children_map.setdefault(parent, []).append(child)
        except Exception:
            continue

    descendants: list[str] = []
    queue = list(children_map.get(session_id, []))
    seen: set[str] = set(queue)
    while queue:
        current = queue.pop(0)
        descendants.append(current)
        for child in children_map.get(current, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return descendants


def _load_pending_tool_calls(storage_root: Path, session_id: str) -> list[Any]:
    """Return pending_tool_calls from the latest checkpoint, or [] if absent."""
    latest_path = storage_root / "checkpoints" / session_id / "latest.json"
    if not latest_path.exists():
        return []
    try:
        checkpoint = json.loads(latest_path.read_text())
        calls = checkpoint.get("pending_tool_calls")
        return calls if isinstance(calls, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_session_cancel_router(
    *, audit_log: AuditLog, storage_root: Path, event_log: EventLogWriter
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        _cancels_counter.add(1, {"session.id": session_id})

        with tracer.start_as_current_span(
            "session.cancel",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.cancel.invocation",
                    code="session_cancel",
                    timestamp=now,
                ),
            )

            try:
                # Step 1: Verify session exists
                manifest_path = storage_root / "sessions" / session_id / "manifest.json"
                if not manifest_path.exists():
                    err = SessionCancelNotFoundError(
                        message=f"Session {session_id!r} not found",
                        timestamp=_now(),
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.cancel.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

                # Step 2: Determine current phase
                reader = LocalEventLogReader(storage_root)
                projection = PhaseProjection(reader)
                before = projection.current_phase(session_id)

                # Step 3: Propagate cancellation to active tool calls
                cancelled_tool_call_ids: list[str] = []
                if before == "waiting_for_tool":
                    pending = _load_pending_tool_calls(storage_root, session_id)
                    for call in pending:
                        tool_call_id = call.get("id") if isinstance(call, dict) else None
                        await event_log.append(
                            session_id,
                            "tool_call.cancelled",
                            {
                                "tool_call_id": tool_call_id,
                                "reason": "session_cancelled",
                                "timestamp": _now(),
                            },
                        )
                        if tool_call_id:
                            cancelled_tool_call_ids.append(tool_call_id)

                # Step 4: Transition phase to terminated
                await event_log.append(
                    session_id,
                    "session.phase_change",
                    {
                        "before": before,
                        "after": "terminated",
                        "timestamp": _now(),
                        "reason": "cancelled",
                    },
                )
                sessions_total.labels(phase="terminated").inc()
                active_sessions.labels(phase="terminated").inc()
                if before is not None:
                    active_sessions.labels(phase=before).dec()
                try:
                    manifest_content = json.loads(manifest_path.read_text())
                    created_at = manifest_content.get("created_at", "")
                    if created_at:
                        started = datetime.fromisoformat(created_at)
                        duration = (datetime.now(UTC) - started).total_seconds()
                        session_duration_seconds.labels(result="cancelled").observe(duration)
                except Exception:
                    pass

                # Step 5: Propagate cancellation to child sessions
                descendants = _walk_descendants(session_id, storage_root)
                for child_id in descendants:
                    child_path = storage_root / "sessions" / child_id / "manifest.json"
                    if child_path.exists():
                        child_manifest = json.loads(child_path.read_text())
                        child_manifest["status"] = "cancelled"
                        child_path.write_text(json.dumps(child_manifest))
                    audit_log.write(
                        AuditLogEntry(
                            level="info",
                            event="child_session.completed",
                            code="session_cancel",
                            timestamp=_now(),
                            detail={
                                "session_id": session_id,
                                "child_session_id": child_id,
                                "reason": "cancelled",
                            },
                        )
                    )

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="session.cancel.accepted",
                        code="session_cancel_accepted",
                        timestamp=_now(),
                        detail={
                            "session_id": session_id,
                            "before": before,
                            "after": "terminated",
                            "reason": "cancelled",
                        },
                    )
                )

            except (SessionCancelNotFoundError, SessionCancelError):
                raise
            except Exception as exc:
                err2 = SessionCancelError(
                    message=f"Cancel failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.cancel.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return JSONResponse(
            content={
                "session_id": session_id,
                "before": before,
                "after": "terminated",
                "reason": "cancelled",
                "cancelled_tool_call_ids": cancelled_tool_call_ids,
                "cancelled_sessions": descendants,
                "cancelled_count": len(descendants),
            }
        )

    return router
