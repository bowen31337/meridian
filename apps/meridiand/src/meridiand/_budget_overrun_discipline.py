"""Budget overrun discipline report: verifies PRD §7.3 compliance.

Reads the session event log to compute:
- Average soft budget overrun ratio (must be < 5% of the configured soft limit).
- Fraction of hard-budget transitions tagged with the correct reason code (must be 100%).

Exposed via:
    GET /v1/x/budgets/discipline[?since=ISO][&until=ISO]

Emits one OTel span (``budgets.discipline``) with a structured
``budgets.discipline.invocation`` event per request.  On failure writes an
error-level audit entry and surfaces the error to the caller.
"""

from __future__ import annotations

import json
from collections import defaultdict
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
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from sdk_budget import (
    BudgetOverrunDiscipline,
    BudgetOverrunDisciplineError,
    BudgetOverrunDisciplineOptions,
    HardBudgetReasonCodeError,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class BudgetDisciplineReportError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="budget_discipline_report_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Event log scanner
# ---------------------------------------------------------------------------


def _scan_events(
    events_root: Path,
    event_types: frozenset[str],
    since: str | None,
    until: str | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return (session_id, raw_record) for events matching the filters."""
    results: list[tuple[str, dict[str, Any]]] = []
    if not events_root.exists():
        return results
    for ndjson_path in sorted(events_root.rglob("*.ndjson")):
        session_id = ndjson_path.stem
        try:
            text = ndjson_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record: dict[str, Any] = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if record.get("type") not in event_types:
                continue
            ts = record.get("ts", "")
            if since is not None and ts < since:
                continue
            if until is not None and ts > until:
                continue
            results.append((session_id, record))
    return results


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _build_soft_overrun_stats(
    events_root: Path,
    since: str | None,
    until: str | None,
    discipline: BudgetOverrunDiscipline,
) -> dict[str, Any]:
    """Compute average soft budget overrun ratio from budget.warning events."""
    events = _scan_events(events_root, frozenset({"budget.warning"}), since, until)
    ratios: list[float] = []
    for session_id, record in events:
        data = record.get("data") or {}
        limit = float(data.get("limit", 0))
        actual = float(data.get("actual", 0))
        if limit <= 0 or actual <= limit:
            continue
        scope_id = str(data.get("session_id") or session_id)
        dimension = str(data.get("dimension") or "unknown")
        try:
            ratio = discipline.record_soft_overrun(
                scope="session",
                scope_id=scope_id,
                dimension=dimension,
                soft_limit=limit,
                actual=actual,
            )
            ratios.append(ratio)
        except BudgetOverrunDisciplineError:
            pass

    count = len(ratios)
    average_ratio = sum(ratios) / count if count > 0 else 0.0
    return {
        "count": count,
        "average_ratio": round(average_ratio, 6),
        "compliant": average_ratio < 0.05,
    }


def _build_hard_transition_stats(
    events_root: Path,
    since: str | None,
    until: str | None,
    discipline: BudgetOverrunDiscipline,
) -> dict[str, Any]:
    """Compute hard-budget transition reason code compliance from event log."""
    all_events = _scan_events(
        events_root,
        frozenset({"budget.exceeded", "session.phase_change"}),
        since,
        until,
    )

    # Collect sessions that had a budget.exceeded event and their dimensions.
    exceeded_sessions: dict[str, str] = {}
    for session_id, record in all_events:
        if record.get("type") == "budget.exceeded":
            dimension = str((record.get("data") or {}).get("dimension") or "unknown")
            if session_id not in exceeded_sessions:
                exceeded_sessions[session_id] = dimension

    tagged_correctly = 0
    tagged_incorrectly = 0

    for session_id, record in all_events:
        if record.get("type") != "session.phase_change":
            continue
        if session_id not in exceeded_sessions:
            continue
        data = record.get("data") or {}
        if data.get("after") != "terminated":
            continue
        reason_code = str(data.get("reason") or "")
        dimension = exceeded_sessions[session_id]
        try:
            discipline.validate_hard_transition_reason(
                scope="session",
                scope_id=session_id,
                dimension=dimension,
                reason_code=reason_code,
            )
            tagged_correctly += 1
        except HardBudgetReasonCodeError:
            tagged_incorrectly += 1
        except BudgetOverrunDisciplineError:
            pass

    total = tagged_correctly + tagged_incorrectly
    compliance_ratio = tagged_correctly / total if total > 0 else 1.0
    return {
        "total": total,
        "tagged_correctly": tagged_correctly,
        "compliance_ratio": round(compliance_ratio, 6),
        "compliant": tagged_incorrectly == 0,
    }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_budget_overrun_discipline_router(
    *, audit_log: AuditLog, storage_root: Path
) -> APIRouter:
    router = APIRouter()
    events_root = storage_root / "events"

    @router.get("/v1/x/budgets/discipline")
    async def get_budget_overrun_discipline(
        since: str | None = Query(default=None),
        until: str | None = Query(default=None),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "budgets.discipline",
            attributes={
                "report.since": since or "",
                "report.until": until or "",
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="budgets.discipline.invocation",
                    code="budgets_discipline",
                    timestamp=now,
                ),
            )

            try:
                discipline = BudgetOverrunDiscipline(
                    BudgetOverrunDisciplineOptions(audit_log=audit_log)
                )
                soft_overrun = _build_soft_overrun_stats(
                    events_root, since, until, discipline
                )
                hard_transitions = _build_hard_transition_stats(
                    events_root, since, until, discipline
                )

            except BudgetDisciplineReportError:
                raise

            except Exception as exc:
                err = BudgetDisciplineReportError(
                    message=f"Failed to build budget overrun discipline report: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="budgets.discipline.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "since": since,
                            "until": until,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "since": since,
                "until": until,
                "soft_overrun": soft_overrun,
                "hard_transitions": hard_transitions,
            },
            status_code=200,
        )

    return router
