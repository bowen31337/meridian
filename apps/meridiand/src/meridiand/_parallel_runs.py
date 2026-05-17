from __future__ import annotations

import asyncio
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
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ._replay import FakeModelAdapter, FakeSandboxAdapter, _run_harness


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ParallelRunsError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="parallel_runs_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 422


class BudgetExceededError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="budget_exceeded", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChildSpec(BaseModel):
    fixture_session_id: str
    capabilities: list[str] = []


class ParallelRunsRequest(BaseModel):
    children: list[ChildSpec]
    budget_model_calls: int | None = None


# ---------------------------------------------------------------------------
# Parallel execution helper
# ---------------------------------------------------------------------------


async def _run_children_parallel(
    children: list[ChildSpec],
    storage_root: Path,
    budget_model_calls: int | None,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """
    Run all children concurrently. Returns (child_results, status, total_model_calls, total_tool_calls).

    When budget_model_calls is set and the running total crosses it after processing
    a batch of completed tasks, all remaining tasks are cancelled synchronously.
    """
    if not children:
        return [], "completed", 0, 0

    child_results: list[dict[str, Any] | None] = [None] * len(children)
    total_model_calls = 0
    total_tool_calls = 0
    final_status = "completed"

    async def run_one(idx: int, spec: ChildSpec) -> tuple[int, dict[str, Any]]:
        fixture_dir = storage_root / "fixtures" / spec.fixture_session_id
        model_adapter = FakeModelAdapter(fixture_dir / "model_responses.ndjson")
        sandbox_adapter = FakeSandboxAdapter(fixture_dir / "tool_responses.ndjson")
        model_calls, tool_calls = await _run_harness(model_adapter, sandbox_adapter)
        return idx, {
            "fixture_session_id": spec.fixture_session_id,
            "model_call_count": model_calls,
            "tool_call_count": tool_calls,
            "status": "completed",
        }

    task_to_idx: dict[asyncio.Task[tuple[int, dict[str, Any]]], int] = {}
    for i, spec in enumerate(children):
        t: asyncio.Task[tuple[int, dict[str, Any]]] = asyncio.create_task(run_one(i, spec))
        task_to_idx[t] = i

    pending: set[asyncio.Task[tuple[int, dict[str, Any]]]] = set(task_to_idx)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            try:
                idx, result = await task
                child_results[idx] = result
                total_model_calls += result["model_call_count"]
                total_tool_calls += result["tool_call_count"]
            except asyncio.CancelledError:
                pass

        # Check budget after processing the entire batch of completed tasks
        if budget_model_calls is not None and total_model_calls > budget_model_calls:
            # Cancel all remaining tasks synchronously
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                for t, t_idx in task_to_idx.items():
                    if child_results[t_idx] is None:
                        child_results[t_idx] = {
                            "fixture_session_id": children[t_idx].fixture_session_id,
                            "model_call_count": 0,
                            "tool_call_count": 0,
                            "status": "cancelled",
                        }
            pending.clear()
            final_status = "budget_exceeded"
            break

    # Fill in any nulls from cancelled tasks
    for i, r in enumerate(child_results):
        if r is None:
            child_results[i] = {
                "fixture_session_id": children[i].fixture_session_id,
                "model_call_count": 0,
                "tool_call_count": 0,
                "status": "cancelled",
            }

    return child_results, final_status, total_model_calls, total_tool_calls  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_parallel_runs_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/x/sessions/{session_id}/parallel_runs")
    async def parallel_runs(session_id: str, body: ParallelRunsRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        run_id = str(uuid.uuid4())

        with tracer.start_as_current_span(
            "session.parallel_runs",
            attributes={
                "session.id": session_id,
                "parallel_runs.run_id": run_id,
                "parallel_runs.child_count": len(body.children),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.parallel_runs.invocation",
                    code="session_parallel_runs",
                    timestamp=now,
                ),
            )

            try:
                child_results, status, total_model_calls, total_tool_calls = (
                    await _run_children_parallel(body.children, storage_root, body.budget_model_calls)
                )

                if status == "budget_exceeded":
                    err = BudgetExceededError(
                        message=(
                            f"Budget of {body.budget_model_calls} model calls exceeded "
                            f"for session {session_id!r}: used {total_model_calls}"
                        ),
                        timestamp=_now(),
                    )
                    record_error(span, err)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="session.parallel_runs.budget_exceeded",
                            code=err.code,
                            timestamp=err.timestamp,
                            detail={
                                "session_id": session_id,
                                "run_id": run_id,
                                "budget_model_calls": body.budget_model_calls,
                                "total_model_calls": total_model_calls,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

            except (ParallelRunsError, BudgetExceededError):
                raise
            except Exception as exc:
                err = ParallelRunsError(
                    message=f"Parallel runs failed for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.parallel_runs.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "run_id": run_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

        succeeded = sum(1 for r in child_results if r["status"] == "completed")
        cancelled = sum(1 for r in child_results if r["status"] == "cancelled")

        return JSONResponse(
            content={
                "session_id": session_id,
                "run_id": run_id,
                "status": status,
                "total_children": len(body.children),
                "succeeded": succeeded,
                "cancelled": cancelled,
                "total_model_calls": total_model_calls,
                "total_tool_calls": total_tool_calls,
                "children": child_results,
            }
        )

    return router
