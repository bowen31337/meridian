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
from ._metrics_registry import tool_call_duration_seconds, tool_calls_total


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

                # Detect tool calls completed since the last checkpoint.
                prev_calls: dict[str, Any] = {}
                prev_taken_at: str | None = None
                latest_path = checkpoint_dir / "latest.json"
                if latest_path.exists():
                    try:
                        prev = json.loads(latest_path.read_text())
                        prev_taken_at = prev.get("taken_at")
                        for call in prev.get("pending_tool_calls", []):
                            if isinstance(call, dict) and call.get("id"):
                                prev_calls[call["id"]] = call
                    except Exception:
                        pass

                encoded = json.dumps(body.model_dump(), default=str).encode()
                _write_atomic(checkpoint_dir / f"{body.seq}.json", encoded)
                _write_atomic(checkpoint_dir / "latest.json", encoded)

                if prev_calls:
                    current_ids = {
                        call["id"]
                        for call in body.pending_tool_calls
                        if isinstance(call, dict) and call.get("id")
                    }
                    completed = [c for cid, c in prev_calls.items() if cid not in current_ids]
                    if completed:
                        per_call_secs: float | None = None
                        if prev_taken_at:
                            try:
                                t0 = datetime.fromisoformat(prev_taken_at)
                                t1 = datetime.fromisoformat(body.taken_at)
                                if t0.tzinfo is None:
                                    t0 = t0.replace(tzinfo=UTC)
                                if t1.tzinfo is None:
                                    t1 = t1.replace(tzinfo=UTC)
                                per_call_secs = (t1 - t0).total_seconds() / len(completed)
                            except Exception:
                                pass
                        for call in completed:
                            tool_name = call.get("name", "unknown") if isinstance(call, dict) else "unknown"
                            tool_calls_total.labels(
                                tool=tool_name, backend="unknown", result="success"
                            ).inc()
                            if per_call_secs is not None and per_call_secs >= 0:
                                tool_call_duration_seconds.observe(per_call_secs)

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
