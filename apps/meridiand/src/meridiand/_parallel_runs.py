from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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

from ._replay import FakeModelAdapter, FakeSandboxAdapter, UsageDelta, _run_harness


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
# Budget accumulator
# ---------------------------------------------------------------------------


class _BudgetAccumulator:
    """
    Accumulates usage.delta events emitted by descendant workers and signals a
    breach via an asyncio.Event when the parent's hard budget is crossed.

    Each usage.delta counts as one model call.  Workers check cancel_event at
    the top of every harness loop iteration so breach propagation is synchronous
    — the breaching delta sets the event, and every other worker raises
    CancelledError at its very next model-call boundary.
    """

    def __init__(self, budget_model_calls: int | None) -> None:
        self._budget = budget_model_calls
        self._model_calls = 0
        self._cancel_event: asyncio.Event = asyncio.Event()

    def record(self, delta: UsageDelta) -> None:  # noqa: ARG002 — tokens reserved for future cost accounting
        self._model_calls += 1
        if self._budget is not None and self._model_calls > self._budget:
            self._cancel_event.set()

    @property
    def exceeded(self) -> bool:
        return self._budget is not None and self._model_calls > self._budget

    @property
    def total_model_calls(self) -> int:
        return self._model_calls

    @property
    def cancel_event(self) -> asyncio.Event:
        return self._cancel_event


# ---------------------------------------------------------------------------
# Parallel execution helper
# ---------------------------------------------------------------------------


async def _run_children_parallel(
    children: list[ChildSpec],
    storage_root: Path,
    budget_model_calls: int | None,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """
    Run all children concurrently. Returns
    (child_results, status, total_model_calls, total_tool_calls).

    Each child emits usage.delta events via on_usage_delta callbacks that roll up into a
    shared _BudgetAccumulator.  When the accumulated total crosses budget_model_calls the
    accumulator sets a shared cancel_event; every worker checks that event at the top of
    each harness loop iteration and raises CancelledError before its next model call.

    After each asyncio.wait batch the caller also cancels any remaining pending tasks
    synchronously so workers that haven't started yet are also stopped immediately.
    """
    if not children:
        return [], "completed", 0, 0

    budget_acc = _BudgetAccumulator(budget_model_calls)
    child_results: list[dict[str, Any] | None] = [None] * len(children)
    total_tool_calls = 0
    final_status = "completed"

    async def run_one(idx: int, spec: ChildSpec) -> tuple[int, dict[str, Any]]:
        fixture_dir = storage_root / "fixtures" / spec.fixture_session_id
        model_adapter = FakeModelAdapter(fixture_dir / "model_responses.ndjson")
        sandbox_adapter = FakeSandboxAdapter(fixture_dir / "tool_responses.ndjson")

        def on_delta(delta: UsageDelta) -> None:
            budget_acc.record(delta)

        model_calls, tool_calls = await _run_harness(
            model_adapter,
            sandbox_adapter,
            on_usage_delta=on_delta,
            cancel_event=budget_acc.cancel_event,
        )
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
                total_tool_calls += result["tool_call_count"]
            except asyncio.CancelledError:
                pass

        # After each batch, cancel any remaining tasks if budget was breached via usage.delta
        # events (which may have been emitted by workers that just completed or mid-run).
        if budget_acc.exceeded:
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
                for _t, t_idx in task_to_idx.items():
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

    # Fill in any nulls from tasks cancelled mid-execution via cancel_event
    for i, r in enumerate(child_results):
        if r is None:
            child_results[i] = {
                "fixture_session_id": children[i].fixture_session_id,
                "model_call_count": 0,
                "tool_call_count": 0,
                "status": "cancelled",
            }

    return child_results, final_status, budget_acc.total_model_calls, total_tool_calls  # type: ignore[return-value]


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
                (
                    child_results,
                    status,
                    total_model_calls,
                    total_tool_calls,
                ) = await _run_children_parallel(
                    body.children, storage_root, body.budget_model_calls
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
                raise err from exc

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
