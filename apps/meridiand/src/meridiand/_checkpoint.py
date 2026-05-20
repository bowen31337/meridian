from __future__ import annotations

import json
import os
import tempfile
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
from sdk_sandbox import ExecutionContext

from ._hook_dispatch import dispatch_hooks


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class CheckpointError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="checkpoint_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class SessionCheckpoint(BaseModel):
    seq: int
    phase: str
    pending_tool_calls: list[Any]
    message_tail: list[Any]
    usage: dict[str, Any]
    taken_at: str


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, data: bytes) -> None:
    """Write data to path via a temp file in the same directory (atomic rename)."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def make_checkpoint_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/checkpoint")
    async def create_checkpoint(session_id: str, body: SessionCheckpoint) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        checkpoint_dir = storage_root / "checkpoints" / session_id

        with tracer.start_as_current_span(
            "checkpoint.create",
            attributes={
                "session.id": session_id,
                "checkpoint.seq": body.seq,
                "checkpoint.phase": body.phase,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="checkpoint.create.invocation",
                    code="checkpoint_create",
                    timestamp=now,
                ),
            )

            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                encoded = json.dumps(body.model_dump(), default=str).encode()
                _write_atomic(checkpoint_dir / f"{body.seq}.json", encoded)
                _write_atomic(checkpoint_dir / "latest.json", encoded)

                await dispatch_hooks(
                    "on_checkpoint",
                    {
                        "session_id": session_id,
                        "seq": body.seq,
                        "phase": body.phase,
                    },
                    ExecutionContext(session_id=session_id),
                    hooks_dir=storage_root / "hooks",
                    audit_log=audit_log,
                )
            except CheckpointError:
                raise
            except Exception as exc:
                err = CheckpointError(
                    message=f"Checkpoint failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="checkpoint.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "seq": body.seq,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "session_id": session_id,
                "seq": body.seq,
                "status": "saved",
            }
        )

    return router
