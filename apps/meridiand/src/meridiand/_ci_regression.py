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

from ._replay import FakeModelAdapter, FakeSandboxAdapter


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
# Harness with event capture
# ---------------------------------------------------------------------------


async def _run_harness_capturing(
    model_adapter: FakeModelAdapter,
    sandbox_adapter: FakeSandboxAdapter,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Run the agent harness and capture the full event sequence.

    Returns (model_calls, tool_calls, captured_events).  Tool dispatches are
    represented as synthetic {"type": "tool_result", ...} entries so that the
    baseline can detect changes to tool-dispatch ordering or content.
    """
    model_calls = 0
    tool_calls = 0
    captured: list[dict[str, Any]] = []

    while True:
        model_calls += 1
        tool_use_blocks: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        stop_reason = "end_turn"

        async for event in model_adapter.call():
            captured.append(event)
            etype = event.get("type", "")
            if etype == "tool_use_start":
                current_tool = {
                    "id": event.get("id", ""),
                    "name": event.get("name", ""),
                    "input_json": "",
                }
                tool_use_blocks.append(current_tool)
            elif etype == "tool_input_delta" and current_tool is not None:
                current_tool["input_json"] += event.get("partial_json", "")
            elif etype == "message_stop":
                stop_reason = event.get("stop_reason") or "end_turn"

        if not tool_use_blocks or stop_reason != "tool_use":
            break

        for block in tool_use_blocks:
            tool_calls += 1
            result = sandbox_adapter.next_result()
            captured.append(
                {
                    "type": "tool_result",
                    "tool_id": block["id"],
                    "content": result.get("content", ""),
                }
            )

    return model_calls, tool_calls, captured


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------


def _find_divergence(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None] | None:
    """Return (seq, expected_event, actual_event) at the first divergence, or None if equal."""
    for i, (exp, act) in enumerate(zip(expected, actual, strict=False)):
        if exp != act:
            return (i, exp, act)
    if len(expected) != len(actual):
        seq = min(len(expected), len(actual))
        return (
            seq,
            expected[seq] if seq < len(expected) else None,
            actual[seq] if seq < len(actual) else None,
        )
    return None


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
