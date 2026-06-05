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
import jsonschema
from pydantic import BaseModel
from sdk_sandbox import ExecutionContext

from ._hook_dispatch import dispatch_hooks


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _schema_errors(schema: dict[str, Any], data: Any) -> list[str]:
    validator = jsonschema.Draft7Validator(schema)
    return [e.message for e in sorted(validator.iter_errors(data), key=str)]


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class HandoffSchemaError(MeridianError):
    """Raised when a child's terminal message fails output_schema validation."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        retry_allowed: bool,
        validation_errors: list[str],
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="handoff_schema_invalid",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )
        self.retry_allowed = retry_allowed
        self.validation_errors = validation_errors

    def http_status(self) -> int:
        return 422

    def to_envelope(self) -> dict[str, Any]:
        env = super().to_envelope()
        env["error"]["retry_allowed"] = self.retry_allowed
        env["error"]["validation_errors"] = self.validation_errors
        return env


class HandoffSessionNotFoundError(MeridianError):
    """Raised when no manifest exists for the given child session."""

    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="handoff_session_not_found",
            message=message,
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class HandoffRequest(BaseModel):
    terminal_message: Any


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_handoff_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/handoff")
    async def handoff_session(session_id: str, body: HandoffRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "handoff.validate",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="handoff.validate.invocation",
                    code="handoff_validate",
                    timestamp=now,
                ),
            )

            manifest_path = storage_root / "sessions" / session_id / "manifest.json"
            if not manifest_path.exists():
                err = HandoffSessionNotFoundError(
                    message=f"No manifest found for child session {session_id!r}",
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="handoff.validate.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            manifest = json.loads(manifest_path.read_text())
            output_schema = manifest.get("output_schema")

            if output_schema is not None:
                errors = _schema_errors(output_schema, body.terminal_message)
                if errors:
                    current_status = manifest.get("status", "spawned")
                    # First failure (status="spawned") allows one retry.
                    # Subsequent failures (status="waiting_for_user") exhaust retries.
                    retry_allowed = current_status != "waiting_for_user"

                    err = HandoffSchemaError(
                        message=(
                            f"Terminal message for session {session_id!r} failed "
                            f"output_schema validation: {errors[0]}"
                        ),
                        timestamp=_now(),
                        retry_allowed=retry_allowed,
                        validation_errors=errors,
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="handoff.schema_invalid",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "parent_session_id": manifest.get("parent_session_id"),
                                "retry_allowed": retry_allowed,
                                "validation_errors": errors,
                                "message": err.message,
                            },
                        )
                    )

                    if retry_allowed:
                        manifest["status"] = "waiting_for_user"
                        manifest_path.write_text(json.dumps(manifest))

                    raise err

        manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest))

        await dispatch_hooks(
            "on_handoff",
            {
                "session_id": session_id,
                "parent_session_id": manifest.get("parent_session_id"),
                "status": "completed",
            },
            ExecutionContext(session_id=session_id),
            hooks_dir=storage_root / "hooks",
            audit_log=audit_log,
        )

        return JSONResponse(
            content={
                "session_id": session_id,
                "parent_session_id": manifest.get("parent_session_id"),
                "status": "completed",
                "validation": "passed",
            }
        )

    return router
