from __future__ import annotations

import json
import uuid
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
from pydantic import BaseModel
from storage_event_log import EventLogWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SessionCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_create_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    agent_id: str | None = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_sessions_router(
    *, audit_log: AuditLog, storage_root: Path, event_log: EventLogWriter
) -> APIRouter:
    router = APIRouter()
    agents_dir = storage_root / "agents"

    @router.post("/v1/sessions", status_code=201)
    async def create_session(body: SessionCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        session_id = f"sess_{uuid.uuid4().hex}"
        thread_id = f"thread_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "session.create",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.create.invocation",
                    code="session_create",
                    timestamp=now,
                ),
            )

            agent_version_id: str | None = None

            try:
                if body.agent_id is not None:
                    agent_file = agents_dir / f"{body.agent_id}.json"
                    if agent_file.exists():
                        agent_record: dict[str, Any] = json.loads(agent_file.read_text())
                        version = agent_record.get("version") or {}
                        agent_version_id = version.get("id")

                session_dir = storage_root / "sessions" / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                manifest: dict[str, Any] = {
                    "session_id": session_id,
                    "agent_id": body.agent_id,
                    "agent_version_id": agent_version_id,
                    "thread_id": thread_id,
                    "status": "idle",
                    "created_at": now,
                }
                (session_dir / "manifest.json").write_text(json.dumps(manifest))

                threads_dir = session_dir / "threads"
                threads_dir.mkdir(parents=True, exist_ok=True)
                thread_record: dict[str, Any] = {
                    "thread_id": thread_id,
                    "session_id": session_id,
                    "created_at": now,
                }
                (threads_dir / f"{thread_id}.json").write_text(json.dumps(thread_record))

                await event_log.append(
                    session_id,
                    "session.created",
                    {
                        "session_id": session_id,
                        "agent_id": body.agent_id,
                        "agent_version_id": agent_version_id,
                        "thread_id": thread_id,
                        "created_at": now,
                    },
                    thread_id=thread_id,
                )

            except SessionCreateError:
                raise
            except Exception as exc:
                err = SessionCreateError(
                    message=f"Failed to create session: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "agent_id": body.agent_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "session_id": session_id,
                "agent_id": body.agent_id,
                "agent_version_id": agent_version_id,
                "thread_id": thread_id,
                "status": "idle",
                "created_at": now,
            },
            status_code=201,
        )

    return router
