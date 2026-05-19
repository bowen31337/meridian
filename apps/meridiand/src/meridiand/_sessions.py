from __future__ import annotations

import json
import uuid
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
from pydantic import BaseModel

from ._replay import FakeModelAdapter, FakeSandboxAdapter, _run_harness


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SessionRunError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_run_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    agent_id: str | None = None
    fixture_session_id: str


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_sessions_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/sessions", status_code=201)
    async def create_session(body: SessionCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        session_id = f"sess_{uuid.uuid4().hex}"
        fixture_dir = storage_root / "fixtures" / body.fixture_session_id

        with tracer.start_as_current_span(
            "session.run",
            attributes={
                "session.id": session_id,
                "session.fixture_session_id": body.fixture_session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.run.invocation",
                    code="session_run",
                    timestamp=now,
                ),
            )

            try:
                # Step 1: Create session manifest (POST /v1/sessions creates it)
                session_dir = storage_root / "sessions" / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "session_id": session_id,
                    "agent_id": body.agent_id,
                    "status": "active",
                    "created_at": now,
                }
                (session_dir / "manifest.json").write_text(json.dumps(manifest))

                # Step 2: Wake — locate model fixture
                model_fixture = fixture_dir / "model_responses.ndjson"
                if not model_fixture.exists():
                    err = SessionRunError(
                        message=(
                            f"Fixture not found for fixture_session_id "
                            f"{body.fixture_session_id!r}: {model_fixture}"
                        ),
                        timestamp=_now(),
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.run.failed",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "fixture_session_id": body.fixture_session_id,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

                # Step 3: model call → tool dispatch → tool result → end_turn
                model_adapter = FakeModelAdapter(model_fixture)
                sandbox_adapter = FakeSandboxAdapter(fixture_dir / "tool_responses.ndjson")
                model_calls, tool_calls = await _run_harness(model_adapter, sandbox_adapter)

            except SessionRunError:
                raise
            except Exception as exc:
                err = SessionRunError(
                    message=f"Session run failed for {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "fixture_session_id": body.fixture_session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

        # Step 4: idle
        return JSONResponse(
            content={
                "session_id": session_id,
                "status": "idle",
                "phase": "idle",
                "model_call_count": model_calls,
                "tool_call_count": tool_calls,
            },
            status_code=201,
        )

    return router
