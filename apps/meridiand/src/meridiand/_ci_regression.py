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

from ._replay import FakeModelAdapter, FakeSandboxAdapter, _find_divergence, _run_harness_capturing


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class RegressionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="regression_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_ci_regression_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/ci/regression-run")
    async def run_ci_regression() -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        run_id = str(uuid.uuid4())
        fixtures_dir = storage_root / "fixtures"

        with tracer.start_as_current_span(
            "ci.regression.run",
            attributes={
                "regression.run_id": run_id,
                "regression.fixtures_dir": str(fixtures_dir),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="ci.regression.run.invocation",
                    code="ci_regression_run",
                    timestamp=now,
                ),
            )

            session_results: list[dict[str, Any]] = []
            first_failure: dict[str, Any] | None = None

            if fixtures_dir.exists():
                for session_dir in sorted(fixtures_dir.iterdir()):
                    if not session_dir.is_dir():
                        continue
                    session_id = session_dir.name
                    expected_path = session_dir / "expected_events.ndjson"
                    if not expected_path.exists():
                        continue

                    try:
                        expected_events = [
                            json.loads(line)
                            for line in expected_path.read_text().splitlines()
                            if line.strip()
                        ]
                        model_adapter = FakeModelAdapter(session_dir / "model_responses.ndjson")
                        sandbox_adapter = FakeSandboxAdapter(session_dir / "tool_responses.ndjson")
                        model_calls, tool_calls, actual_events = await _run_harness_capturing(
                            model_adapter, sandbox_adapter
                        )

                        divergence = _find_divergence(expected_events, actual_events)
                        if divergence is None:
                            session_results.append(
                                {
                                    "session_id": session_id,
                                    "status": "passed",
                                    "model_calls": model_calls,
                                    "tool_calls": tool_calls,
                                }
                            )
                        else:
                            seq, exp_ev, act_ev = divergence
                            if first_failure is None:
                                first_failure = {
                                    "session_id": session_id,
                                    "first_deviating_seq": seq,
                                    "expected_event": exp_ev,
                                    "actual_event": act_ev,
                                }
                            session_results.append(
                                {
                                    "session_id": session_id,
                                    "status": "failed",
                                    "model_calls": model_calls,
                                    "tool_calls": tool_calls,
                                    "first_deviating_seq": seq,
                                }
                            )
                    except RegressionError:
                        raise
                    except Exception as exc:
                        if first_failure is None:
                            first_failure = {
                                "session_id": session_id,
                                "error": str(exc),
                            }
                        session_results.append(
                            {
                                "session_id": session_id,
                                "status": "error",
                                "error": str(exc),
                            }
                        )

            if first_failure is not None:
                sid = first_failure.get("session_id", "?")
                seq = first_failure.get("first_deviating_seq")
                if seq is not None:
                    msg = (
                        f"Regression failure: session {sid!r} diverged at event seq {seq}; "
                        f"likely caused by a system or prompt change"
                    )
                else:
                    msg = (
                        f"Regression failure: session {sid!r} error: "
                        f"{first_failure.get('error', 'unknown')}"
                    )

                err = RegressionError(message=msg, timestamp=_now())
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="ci.regression.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "run_id": run_id,
                            "first_failure": first_failure,
                            "session_results": session_results,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "passed",
                "session_count": len(session_results),
                "sessions": session_results,
            }
        )

    return router
