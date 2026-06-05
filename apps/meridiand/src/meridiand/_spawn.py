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
from opentelemetry import context as otel_context, trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel
from sdk_capabilities import CapabilityParseError, missing, parse_set


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SpawnError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="spawn_denied", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 403


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SpawnRequest(BaseModel):
    parent_capabilities: list[str]
    child_capabilities: list[str]
    output_schema: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_spawn_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/spawn")
    async def spawn_session(session_id: str, body: SpawnRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        child_session_id = str(uuid.uuid4())

        with tracer.start_as_current_span(
            "session.spawn",
            attributes={
                "session.id": session_id,
                "session.child_id": child_session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.spawn.invocation",
                    code="session_spawn",
                    timestamp=now,
                ),
            )

            try:
                parent_caps = parse_set(body.parent_capabilities)
                child_caps = parse_set(body.child_capabilities)
            except CapabilityParseError as exc:
                err = SpawnError(
                    message=f"Invalid capability string for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.spawn.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "parent_session_id": session_id,
                            "child_session_id": child_session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

            if not any(c.namespace == "agent" and c.name == "spawn" for c in parent_caps):
                err = SpawnError(
                    message=(
                        f"Spawn denied for session {session_id!r}: "
                        "parent does not hold the agent.spawn capability"
                    ),
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.spawn.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "parent_session_id": session_id,
                            "child_session_id": child_session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            escalating = missing(child_caps, parent_caps)
            if escalating:
                escalating_strs = sorted(str(c) for c in escalating)
                err = SpawnError(
                    message=(
                        f"Capability escalation denied for child of session {session_id!r}: "
                        f"caps not held by parent: {', '.join(escalating_strs)}"
                    ),
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.spawn.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "parent_session_id": session_id,
                            "child_session_id": child_session_id,
                            "escalating_caps": escalating_strs,
                            "message": err.message,
                        },
                    )
                )
                raise err

        child_capabilities = sorted(str(c) for c in child_caps)
        session_dir = storage_root / "sessions" / child_session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Build a span link from the parent session's trace to the child trace.
        _child_links: list[trace.Link] = []
        _parent_manifest_path = storage_root / "sessions" / session_id / "manifest.json"
        if _parent_manifest_path.exists():
            try:
                _parent_tp = json.loads(_parent_manifest_path.read_text()).get("traceparent", "")
                if _parent_tp:
                    _pctx = TraceContextTextMapPropagator().extract({"traceparent": _parent_tp})
                    _pspan_ctx = trace.get_current_span(_pctx).get_span_context()
                    if _pspan_ctx.is_valid:
                        _child_links = [
                            trace.Link(
                                context=_pspan_ctx,
                                attributes={"parent.session_id": session_id},
                            )
                        ]
            except Exception:
                pass

        # Start child session root span in a new trace, linked back to parent.
        _child_traceparent = ""
        with tracer.start_as_current_span(
            "child.session",
            context=otel_context.Context(),
            links=_child_links,
            attributes={"session.id": child_session_id, "parent.session_id": session_id},
        ):
            _carrier: dict[str, str] = {}
            TraceContextTextMapPropagator().inject(_carrier)
            _child_traceparent = _carrier.get("traceparent", "")

        manifest = {
            "child_session_id": child_session_id,
            "parent_session_id": session_id,
            "capabilities": child_capabilities,
            "output_schema": body.output_schema,
            "created_at": now,
            "status": "spawned",
            "traceparent": _child_traceparent,
        }
        (session_dir / "manifest.json").write_text(json.dumps(manifest))

        return JSONResponse(
            content={
                "child_session_id": child_session_id,
                "parent_session_id": session_id,
                "capabilities": child_capabilities,
                "status": "spawned",
            },
            status_code=201,
        )

    return router
