from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

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


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class CancelError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="cancel_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Tree walk
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


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_cancel_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

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
                descendants = _walk_descendants(session_id, storage_root)

                for child_id in descendants:
                    manifest_path = storage_root / "sessions" / child_id / "manifest.json"
                    if manifest_path.exists():
                        manifest = json.loads(manifest_path.read_text())
                        manifest["status"] = "cancelled"
                        manifest_path.write_text(json.dumps(manifest))
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

            except Exception as exc:
                err = CancelError(
                    message=f"Cancel failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
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

        return JSONResponse(
            content={
                "session_id": session_id,
                "cancelled_sessions": descendants,
                "cancelled_count": len(descendants),
            }
        )

    return router
