from __future__ import annotations

from datetime import UTC, datetime
import json
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

from ._replay import FakeModelAdapter, FakeSandboxAdapter, _run_harness


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ResumeError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(code="resume_failed", message=message, timestamp=timestamp, cause=cause)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_resume_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/resume")
    async def resume_session(session_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        checkpoint_dir = storage_root / "checkpoints" / session_id
        latest_path = checkpoint_dir / "latest.json"
        fixture_dir = storage_root / "fixtures" / session_id

        with tracer.start_as_current_span(
            "session.resume",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.resume.invocation",
                    code="session_resume",
                    timestamp=now,
                ),
            )

            try:
                # Step 1: Load latest checkpoint (fallback: replay log)
                checkpoint: dict | None = None

                if latest_path.exists():
                    checkpoint = json.loads(latest_path.read_text())
                else:
                    model_fixture = fixture_dir / "model_responses.ndjson"
                    if not model_fixture.exists():
                        err = ResumeError(
                            message=(
                                f"No checkpoint or replay log found for session {session_id!r}"
                            ),
                            timestamp=_now(),
                        )
                        record_error(span, err)
                        audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="session.resume.failed",
                                code=err.code,
                                timestamp=err.timestamp,
                                detail={
                                    "session_id": session_id,
                                    "message": err.message,
                                },
                            )
                        )
                        raise err

                # Step 2: Re-dispatch tool calls whose result is missing
                pending_tool_calls = checkpoint.get("pending_tool_calls", []) if checkpoint else []
                sandbox_adapter = FakeSandboxAdapter(fixture_dir / "tool_responses.ndjson")
                tool_calls_dispatched = 0
                for _ in pending_tool_calls:
                    sandbox_adapter.next_result()
                    tool_calls_dispatched += 1

                # Step 3: Transition phase appropriately
                if pending_tool_calls:
                    phase = "waiting_for_model"
                elif checkpoint:
                    phase = str(checkpoint.get("phase", "waiting_for_model"))
                else:
                    phase = "waiting_for_model"

                # Step 4: Wake harness; continue
                model_adapter = FakeModelAdapter(fixture_dir / "model_responses.ndjson")
                model_calls, tool_calls_new = await _run_harness(model_adapter, sandbox_adapter)
                if model_calls > 0:
                    phase = "idle"

            except ResumeError:
                raise
            except Exception as exc:
                err = ResumeError(
                    message=f"Resume failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.resume.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "session_id": session_id,
                "status": "resumed",
                "phase": phase,
                "checkpoint_seq": checkpoint["seq"] if checkpoint else None,
                "tool_calls_dispatched": tool_calls_dispatched,
                "model_call_count": model_calls,
                "tool_call_count": tool_calls_dispatched + tool_calls_new,
            }
        )

    return router
