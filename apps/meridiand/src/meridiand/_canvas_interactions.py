"""canvas_interactions — endpoint for surfacing Canvas widget interactions as user messages.

POST /v1/sessions/{session_id}/canvas_interactions

Accepts a form submission or button click from the Live Canvas, appends a
``canvas_interaction`` event to the session event log (for replay semantics),
and injects a structured ``message.added`` user-role event so the harness can
deliver the interaction as a new conversation turn.

Emits an OpenTelemetry span and logs a structured event on each invocation.
On failure surfaces an HTTP error response and writes the failure to the audit log.
"""

from __future__ import annotations

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
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from storage_event_log import EventLogWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class CanvasInteractionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="canvas_interaction_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class CanvasInteractionSessionNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="session_not_found",
            message=message,
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class CanvasInteractionRequest(BaseModel):
    kind: str  # "form.submit" | "button.click"
    widget_id: str
    widget_kind: str
    payload: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_canvas_interactions_router(
    *, audit_log: AuditLog, storage_root: Path, event_log: EventLogWriter
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/sessions/{session_id}/canvas_interactions", status_code=201)
    async def submit_canvas_interaction(
        session_id: str, body: CanvasInteractionRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        interaction_id = f"cxi_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "canvas.interaction.submit",
            attributes={
                "session.id": session_id,
                "canvas.interaction.id": interaction_id,
                "canvas.interaction.kind": body.kind,
                "canvas.interaction.widget_id": body.widget_id,
                "canvas.interaction.widget_kind": body.widget_kind,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="canvas.interaction.submit.invocation",
                    code="canvas_interaction_submit",
                    timestamp=now,
                ),
            )

            try:
                manifest_path = storage_root / "sessions" / session_id / "manifest.json"
                if not manifest_path.exists():
                    err = CanvasInteractionSessionNotFoundError(
                        message=f"Session {session_id!r} not found",
                        timestamp=_now(),
                    )
                    span.set_attribute("canvas.interaction.submit.success", False)
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="canvas.interaction.submit.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "interaction_id": interaction_id,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

                manifest_data: dict[str, Any] = json.loads(manifest_path.read_text())
                thread_id: str = manifest_data.get("thread_id", "")

                # Append canvas_interaction event — this is the record of truth for replay.
                interaction_payload: dict[str, Any] = {
                    "interaction_id": interaction_id,
                    "session_id": session_id,
                    "kind": body.kind,
                    "widget_id": body.widget_id,
                    "widget_kind": body.widget_kind,
                    "payload": body.payload,
                    "timestamp": now,
                }
                await event_log.append(
                    session_id,
                    "canvas_interaction",
                    interaction_payload,
                )

                # Inject a structured user message so the harness surfaces the
                # interaction as a new conversation turn (message.added event).
                message_id = f"msg_{uuid.uuid4().hex}"
                user_content = json.dumps(
                    {
                        "type": "canvas_interaction",
                        "interaction_id": interaction_id,
                        "kind": body.kind,
                        "widget_id": body.widget_id,
                        "widget_kind": body.widget_kind,
                        "payload": body.payload,
                        "timestamp": now,
                    }
                )

                thread_dir = storage_root / "threads" / session_id / thread_id
                thread_dir.mkdir(parents=True, exist_ok=True)
                thread_manifest_path = thread_dir / "manifest.json"
                if not thread_manifest_path.exists():
                    thread_manifest_path.write_text(
                        json.dumps(
                            {
                                "id": thread_id,
                                "thread_id": thread_id,
                                "session_id": session_id,
                                "created_at": now,
                            }
                        )
                    )

                messages_path = thread_dir / "messages.ndjson"
                message_record: dict[str, Any] = {
                    "message_id": message_id,
                    "id": message_id,
                    "session_id": session_id,
                    "thread_id": thread_id,
                    "role": "user",
                    "content": user_content,
                    "created_at": now,
                }
                with messages_path.open("a") as f:
                    f.write(json.dumps(message_record) + "\n")

                await event_log.append(
                    session_id,
                    "message.added",
                    {
                        "message_id": message_id,
                        "session_id": session_id,
                        "thread_id": thread_id,
                        "role": "user",
                        "content": user_content,
                        "created_at": now,
                    },
                    thread_id=thread_id,
                )

                span.set_attribute("canvas.interaction.submit.success", True)
                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="canvas.interaction.submitted",
                        code="canvas_interaction_submitted",
                        timestamp=_now(),
                        detail={
                            "session_id": session_id,
                            "interaction_id": interaction_id,
                            "kind": body.kind,
                            "widget_id": body.widget_id,
                            "widget_kind": body.widget_kind,
                        },
                    )
                )

            except (CanvasInteractionError, CanvasInteractionSessionNotFoundError):
                raise
            except Exception as exc:
                err2 = CanvasInteractionError(
                    message=(
                        f"Failed to submit canvas interaction for session {session_id!r}: {exc}"
                    ),
                    timestamp=_now(),
                    cause=exc,
                )
                span.set_attribute("canvas.interaction.submit.success", False)
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="canvas.interaction.submit.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "session_id": session_id,
                            "interaction_id": interaction_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return JSONResponse(
            content={
                "interaction_id": interaction_id,
                "session_id": session_id,
                "kind": body.kind,
                "widget_id": body.widget_id,
                "widget_kind": body.widget_kind,
                "payload": body.payload,
                "timestamp": now,
            },
            status_code=201,
        )

    return router
