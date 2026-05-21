"""Budget cost breakdown reports: by Agent, by Session, by Tool, by Provider/Model.

Reads the NDJSON event log within a time window and aggregates usage.delta
(for agent / session / model breakdowns) and tool_call.requested (for tool
breakdowns) events into a summary report.  Queryable via:

    GET /v1/x/budgets/reports?group_by=<agent|session|tool|model>[&since=ISO][&until=ISO]

Emits one OTel span (``budgets.reports``) per invocation with a structured
``budgets.reports.invocation`` event.  On failure, records the error on the
span and writes an error-level audit entry before re-raising.
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


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class BudgetReportError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="budget_report_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class InvalidGroupByError(MeridianError):
    def __init__(self, *, value: str, timestamp: str) -> None:
        super().__init__(
            code="budget_report_invalid_group_by",
            message=f"group_by must be one of: agent, session, tool, model; got {value!r}",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


class _Totals:
    __slots__ = ("input_tokens", "output_tokens", "cache_tokens")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_tokens = 0

    def add(self, data: dict[str, Any]) -> None:
        self.input_tokens += int(data.get("prompt_tokens", 0))
        self.output_tokens += int(data.get("completion_tokens", 0))
        self.cache_tokens += (
            int(data.get("cache_creation_tokens", 0))
            + int(data.get("cache_read_tokens", 0))
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_tokens": self.cache_tokens,
        }


# ---------------------------------------------------------------------------
# Event log scanner
# ---------------------------------------------------------------------------


def _scan_events(
    events_root: Path,
    event_types: frozenset[str],
    since: str | None,
    until: str | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return (session_id, raw_record) for events matching the filters.

    Scans all *.ndjson files under *events_root*, skipping unreadable files
    and malformed JSON lines without raising.
    """
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
# Session → agent_id lookup
# ---------------------------------------------------------------------------


def _lookup_agent_id(
    sessions_root: Path,
    session_id: str,
    cache: dict[str, str | None],
) -> str | None:
    """Read agent_id from a session manifest (cached)."""
    if session_id not in cache:
        try:
            manifest = json.loads(
                (sessions_root / session_id / "manifest.json").read_text()
            )
            cache[session_id] = manifest.get("agent_id")
        except Exception:
            cache[session_id] = None
    return cache[session_id]


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _build_agent_report(
    events_root: Path,
    sessions_root: Path,
    since: str | None,
    until: str | None,
) -> list[dict[str, Any]]:
    totals: dict[str, _Totals] = defaultdict(_Totals)
    cache: dict[str, str | None] = {}
    for session_id, record in _scan_events(
        events_root, frozenset({"usage.delta"}), since, until
    ):
        agent_id = _lookup_agent_id(sessions_root, session_id, cache) or ""
        totals[agent_id].add(record.get("data", {}))
    return [{"agent_id": k, **v.to_dict()} for k, v in sorted(totals.items())]


def _build_session_report(
    events_root: Path,
    sessions_root: Path,
    since: str | None,
    until: str | None,
) -> list[dict[str, Any]]:
    totals: dict[str, _Totals] = defaultdict(_Totals)
    cache: dict[str, str | None] = {}
    for session_id, record in _scan_events(
        events_root, frozenset({"usage.delta"}), since, until
    ):
        totals[session_id].add(record.get("data", {}))
    return [
        {
            "session_id": sid,
            "agent_id": _lookup_agent_id(sessions_root, sid, cache),
            **t.to_dict(),
        }
        for sid, t in sorted(totals.items())
    ]


def _build_model_report(
    events_root: Path,
    since: str | None,
    until: str | None,
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str], _Totals] = defaultdict(_Totals)
    for _sid, record in _scan_events(
        events_root, frozenset({"usage.delta"}), since, until
    ):
        data = record.get("data", {})
        provider = str(data.get("provider") or "")
        model = str(data.get("model") or "")
        totals[(provider, model)].add(data)
    return [
        {"provider": p, "model": m, **t.to_dict()}
        for (p, m), t in sorted(totals.items())
    ]


def _build_tool_report(
    events_root: Path,
    since: str | None,
    until: str | None,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for _sid, record in _scan_events(
        events_root, frozenset({"tool_call.requested"}), since, until
    ):
        tool_name = str(record.get("data", {}).get("tool_name") or "")
        if tool_name:
            counts[tool_name] += 1
    return [{"tool_name": k, "call_count": v} for k, v in sorted(counts.items())]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

_VALID_GROUP_BY = frozenset({"agent", "session", "tool", "model"})


def make_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    events_root = storage_root / "events"
    sessions_root = storage_root / "sessions"

    @router.get("/v1/x/budgets/reports")
    async def get_budget_reports(
        group_by: str = Query(...),
        since: str | None = Query(default=None),
        until: str | None = Query(default=None),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "budgets.reports",
            attributes={"report.group_by": group_by},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="budgets.reports.invocation",
                    code="budgets_reports",
                    timestamp=now,
                ),
            )

            try:
                if group_by not in _VALID_GROUP_BY:
                    raise InvalidGroupByError(value=group_by, timestamp=now)

                if group_by == "agent":
                    items = _build_agent_report(events_root, sessions_root, since, until)
                elif group_by == "session":
                    items = _build_session_report(events_root, sessions_root, since, until)
                elif group_by == "model":
                    items = _build_model_report(events_root, since, until)
                else:
                    items = _build_tool_report(events_root, since, until)

            except (BudgetReportError, InvalidGroupByError):
                raise

            except Exception as exc:
                err = BudgetReportError(
                    message=f"Failed to build budget report (group_by={group_by!r}): {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="budgets.reports.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "group_by": group_by,
                            "since": since,
                            "until": until,
                            "message": err.message,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "group_by": group_by,
                "since": since,
                "until": until,
                "items": items,
            },
            status_code=200,
        )

    return router
