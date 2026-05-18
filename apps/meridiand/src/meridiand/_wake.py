from __future__ import annotations

import json
from datetime import UTC, datetime
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
from storage_reposit import LocalEventLogReader, PhaseProjection


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class WakeError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="wake_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


class WakeSessionNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="wake_session_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _load_session(storage_root: Path, session_id: str) -> dict[str, Any] | None:
    manifest_path = storage_root / "sessions" / session_id / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())  # type: ignore[no-any-return]


def _load_agent_version(storage_root: Path, agent_id: str) -> dict[str, Any] | None:
    agent_path = storage_root / "agents" / f"{agent_id}.json"
    if not agent_path.exists():
        return None
    return json.loads(agent_path.read_text())  # type: ignore[no-any-return]


def _load_active_skills(storage_root: Path, agent_id: str) -> list[dict[str, Any]]:
    activations_dir = storage_root / "skill_activations"
    if not activations_dir.exists():
        return []
    active: list[dict[str, Any]] = []
    for path in activations_dir.glob("*.json"):
        try:
            record: dict[str, Any] = json.loads(path.read_text())
            if record.get("agent_id") == agent_id and record.get("status") == "active":
                active.append(record)
        except Exception:
            continue
    return active


def _load_most_recent_thread(
    storage_root: Path, session_id: str
) -> tuple[str | None, list[dict[str, Any]]]:
    """Return (thread_id, messages) from the most recent thread for session_id.

    Threads live at $storage_root/threads/{session_id}/{thread_id}/manifest.json.
    Messages live at $storage_root/threads/{session_id}/{thread_id}/messages.ndjson.
    The most recent thread is determined by the latest created_at in its manifest.
    """
    threads_dir = storage_root / "threads" / session_id
    if not threads_dir.exists():
        return None, []

    best_thread_id: str | None = None
    best_created_at: str = ""
    for manifest_path in threads_dir.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            created_at = manifest.get("created_at", "")
            if created_at >= best_created_at:
                best_created_at = created_at
                best_thread_id = manifest.get("id") or manifest_path.parent.name
        except Exception:
            continue

    if best_thread_id is None:
        return None, []

    messages_path = threads_dir / best_thread_id / "messages.ndjson"
    if not messages_path.exists():
        return best_thread_id, []

    messages: list[dict[str, Any]] = []
    for raw in messages_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    messages.sort(key=lambda m: m.get("sequence", 0))
    return best_thread_id, messages


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_wake_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/wake")
    async def wake_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.wake",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.wake.invocation",
                    code="session_wake",
                    timestamp=now,
                ),
            )

            try:
                # Step 1: Load Session
                session = _load_session(storage_root, session_id)
                if session is None:
                    err = WakeSessionNotFoundError(
                        message=f"Session {session_id!r} not found",
                        timestamp=_now(),
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.wake.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

                # Step 2: Load AgentVersion
                agent_id: str | None = session.get("agent_id")
                agent_version = _load_agent_version(storage_root, agent_id) if agent_id else None

                # Step 3: Load active skills
                active_skills = _load_active_skills(storage_root, agent_id) if agent_id else []

                # Step 4: Tail event log to determine current phase
                reader = LocalEventLogReader(storage_root)
                projection = PhaseProjection(reader)
                phase = projection.current_phase(session_id)

                # Step 5: Rebuild model context from messages in most recent Thread
                thread_id, messages = _load_most_recent_thread(storage_root, session_id)

            except (WakeSessionNotFoundError, WakeError):
                raise
            except Exception as exc:
                err2 = WakeError(
                    message=f"Wake failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.wake.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(
            content={
                "session_id": session_id,
                "status": "awake",
                "session": session,
                "agent_version": agent_version,
                "active_skills": active_skills,
                "phase": phase,
                "thread_id": thread_id,
                "messages": messages,
            }
        )

    return router
