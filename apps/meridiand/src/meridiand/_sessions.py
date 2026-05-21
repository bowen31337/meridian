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
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from storage_event_log import EventLogWriter

from ._pagination import (
    DEFAULT_PAGE_SIZE,
    CursorDecodeError,
    apply_cursor_filter,
    decode_cursor,
    make_cursor_page,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
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


class ThreadListError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_threads_list_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class ThreadCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_thread_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class SessionNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="session_not_found",
            message=message,
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    agent_id: str | None = None


class ThreadCreateRequest(BaseModel):
    branch_of_event_seq: int
    title: str | None = None


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

    @router.get("/v1/sessions/{session_id}/threads", status_code=200)
    async def list_threads(
        session_id: str,
        cursor: str | None = Query(default=None),
        limit: int = Query(default=DEFAULT_PAGE_SIZE),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.threads.list",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.threads.list.invocation",
                    code="session_threads_list",
                    timestamp=now,
                ),
            )

            try:
                threads_dir = storage_root / "sessions" / session_id / "threads"
                all_threads: list[dict[str, Any]] = []
                if threads_dir.exists():
                    for path in threads_dir.glob("*.json"):
                        record = json.loads(path.read_text())
                        if "id" not in record:
                            record["id"] = record.get("thread_id", "")
                        record.setdefault("title", None)
                        record.setdefault("branch_of_event_seq", None)
                        all_threads.append(record)

                all_threads.sort(
                    key=lambda r: (r.get("created_at", ""), r.get("id", "")),
                    reverse=True,
                )

                if cursor is not None:
                    c_created_at, c_id = decode_cursor(cursor, timestamp=now)
                    all_threads = apply_cursor_filter(all_threads, c_created_at, c_id)

                page, next_cursor = make_cursor_page(all_threads, limit)

                span.set_attribute("session.threads.list.count", len(page))
                span.set_attribute("session.threads.list.success", True)

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="session.threads.listed",
                        code="session_threads_listed",
                        timestamp=_now(),
                        detail={
                            "session_id": session_id,
                            "count": len(page),
                        },
                    )
                )

            except CursorDecodeError as err:
                span.set_attribute("session.threads.list.success", False)
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.threads.list.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"session_id": session_id, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = ThreadListError(
                    message=f"Failed to list threads for session {session_id}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                span.set_attribute("session.threads.list.success", False)
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.threads.list.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"session_id": session_id, "message": err2.message},
                    )
                )
                raise err2

        response_headers: dict[str, str] = {}
        if next_cursor is not None:
            response_headers["X-Next-Cursor"] = next_cursor

        return JSONResponse(
            content={"items": page, "next_cursor": next_cursor, "limit": limit},
            status_code=200,
            headers=response_headers,
        )

    @router.post("/v1/sessions/{session_id}/threads", status_code=201)
    async def create_thread(session_id: str, body: ThreadCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        thread_id = f"thread_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "session.thread.create",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.thread.create.invocation",
                    code="session_thread_create",
                    timestamp=now,
                ),
            )

            try:
                session_dir = storage_root / "sessions" / session_id
                if not session_dir.exists():
                    not_found = SessionNotFoundError(
                        message=f"Session {session_id} not found",
                        timestamp=_now(),
                    )
                    span.set_attribute("session.thread.create.success", False)
                    record_error(span, not_found)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.thread.create.failed",
                            code=not_found.code,
                            timestamp=not_found.timestamp,
                            detail={
                                "session_id": session_id,
                                "message": not_found.message,
                            },
                        )
                    )
                    raise not_found

                threads_dir = session_dir / "threads"
                threads_dir.mkdir(parents=True, exist_ok=True)

                thread_record: dict[str, Any] = {
                    "thread_id": thread_id,
                    "session_id": session_id,
                    "created_at": now,
                    "branch_of_event_seq": body.branch_of_event_seq,
                }
                if body.title is not None:
                    thread_record["title"] = body.title

                (threads_dir / f"{thread_id}.json").write_text(json.dumps(thread_record))

                await event_log.append(
                    session_id,
                    "thread.created",
                    {
                        "thread_id": thread_id,
                        "session_id": session_id,
                        "branch_of_event_seq": body.branch_of_event_seq,
                        "title": body.title,
                        "created_at": now,
                    },
                    thread_id=thread_id,
                )

                span.set_attribute("session.thread.create.success", True)

                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="session.thread.created",
                        code="session_thread_created",
                        timestamp=_now(),
                        detail={
                            "session_id": session_id,
                            "thread_id": thread_id,
                            "branch_of_event_seq": body.branch_of_event_seq,
                        },
                    )
                )

            except (SessionNotFoundError, ThreadCreateError):
                raise
            except Exception as exc:
                err = ThreadCreateError(
                    message=f"Failed to create thread for session {session_id}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                span.set_attribute("session.thread.create.success", False)
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.thread.create.failed",
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
                "thread_id": thread_id,
                "id": thread_id,
                "session_id": session_id,
                "created_at": now,
                "branch_of_event_seq": body.branch_of_event_seq,
                "title": body.title,
            },
            status_code=201,
        )

    return router
