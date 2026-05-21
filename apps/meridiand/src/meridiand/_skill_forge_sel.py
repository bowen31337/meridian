"""Skill Forge Session Selector (SEL).

Watches for terminated sessions, pulls event log + tool-call summary, and
clusters trajectories by structural similarity (tool call sequence and
terminal phase).

On every invocation: emits OTel span ``"skill_forge.sel.run"`` and logs a
structured audit event.  On failure: records the error to the span, surfaces
the error message to the caller, and writes the failure to the audit log
before re-raising.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
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

from ._cluster_extraction import Cluster, ClusterMember

_TERMINAL_PHASES = frozenset({"terminated", "completed"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ForgeSelError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="forge_sel_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SessionSummary:
    """Tool-call summary for one terminated session."""

    session_id: str
    tool_calls: list[str]
    terminal_phase: str
    event_count: int


@dataclass
class ForgeSelResult:
    """Output of one SEL run: session summaries and derived clusters."""

    session_count: int
    cluster_count: int
    clusters: list[Cluster]


# ---------------------------------------------------------------------------
# Event log scanning
# ---------------------------------------------------------------------------


def _read_session_events(storage_root: Path, session_id: str) -> list[dict[str, Any]]:
    """Read all events for *session_id* from date-partitioned NDJSON files."""
    events: list[dict[str, Any]] = []
    events_dir = storage_root / "events"
    if not events_dir.exists():
        return events
    for path in sorted(events_dir.rglob(f"{session_id}.ndjson")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
                events.append(record)
            except json.JSONDecodeError:
                continue
    return sorted(events, key=lambda e: e.get("seq", 0))


def _enumerate_session_ids(storage_root: Path) -> list[str]:
    """Return all unique session IDs that have event log NDJSON files."""
    events_dir = storage_root / "events"
    if not events_dir.exists():
        return []
    seen: set[str] = set()
    session_ids: list[str] = []
    for path in events_dir.rglob("*.ndjson"):
        sid = path.stem
        if sid not in seen:
            seen.add(sid)
            session_ids.append(sid)
    return sorted(session_ids)


def collect_terminated_sessions(storage_root: Path) -> list[SessionSummary]:
    """Scan the event log for terminated sessions and return their summaries.

    A session is *terminated* when at least one ``session.phase_change`` event
    has ``data.after`` in ``{"terminated", "completed"}``.  Tool calls are
    extracted in ascending seq order from ``tool_call.requested`` events.
    """
    summaries: list[SessionSummary] = []
    for session_id in _enumerate_session_ids(storage_root):
        events = _read_session_events(storage_root, session_id)
        terminal_phase = ""
        tool_calls: list[str] = []
        for event in events:
            etype = event.get("type", "")
            data = event.get("data") or {}
            if etype == "session.phase_change":
                after = data.get("after", "")
                if after in _TERMINAL_PHASES:
                    terminal_phase = after
            elif etype == "tool_call.requested":
                tool_name = data.get("tool_name", "")
                if tool_name:
                    tool_calls.append(tool_name)
        if terminal_phase:
            summaries.append(
                SessionSummary(
                    session_id=session_id,
                    tool_calls=tool_calls,
                    terminal_phase=terminal_phase,
                    event_count=len(events),
                )
            )
    return summaries


# ---------------------------------------------------------------------------
# Trajectory clustering
# ---------------------------------------------------------------------------


def cluster_trajectories(summaries: list[SessionSummary]) -> list[Cluster]:
    """Cluster session summaries by structural similarity.

    Two sessions belong to the same cluster when their tool-call sequence and
    terminal phase are identical.  Each cluster ID is deterministically derived
    from its key so repeated runs with the same sessions produce stable IDs.
    Clusters are returned sorted by size descending (largest cluster first).
    """
    buckets: dict[tuple[str, ...], list[ClusterMember]] = {}
    for summary in summaries:
        key: tuple[str, ...] = (summary.terminal_phase, *summary.tool_calls)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(
            ClusterMember(
                session_id=summary.session_id,
                tool_calls=summary.tool_calls,
            )
        )

    clusters: list[Cluster] = []
    for key, members in sorted(
        buckets.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        cluster_id = f"cluster_{uuid.uuid5(uuid.NAMESPACE_DNS, str(key)).hex}"
        clusters.append(Cluster(id=cluster_id, members=members))
    return clusters


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_forge_session_selector(
    storage_root: Path,
    audit_log: AuditLog,
) -> ForgeSelResult:
    """Watch terminated sessions, extract tool-call summaries, cluster by similarity.

    Scans the event log for sessions that have reached a terminal phase
    (``"terminated"`` or ``"completed"``), builds a tool-call sequence for
    each, then clusters them by structural similarity (identical tool-call
    sequence + terminal phase).

    On every invocation: emits OTel span ``"skill_forge.sel.run"`` and logs a
    structured audit event.  On failure: records the error to the span,
    surfaces the error message to the caller, and writes the failure to the
    audit log before re-raising as :class:`ForgeSelError`.
    """
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span("skill_forge.sel.run") as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="skill_forge.sel.run.invocation",
                code="skill_forge_sel_run",
                timestamp=now,
            ),
        )

        try:
            summaries = collect_terminated_sessions(storage_root)
            clusters = cluster_trajectories(summaries)
            result = ForgeSelResult(
                session_count=len(summaries),
                cluster_count=len(clusters),
                clusters=clusters,
            )

            span.set_attribute("skill_forge.sel.session_count", result.session_count)
            span.set_attribute("skill_forge.sel.cluster_count", result.cluster_count)
            span.set_attribute("skill_forge.sel.run.success", True)
            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="skill_forge.sel.ran",
                    code="skill_forge_sel_ran",
                    timestamp=_now(),
                    detail={
                        "session_count": result.session_count,
                        "cluster_count": result.cluster_count,
                    },
                )
            )

        except Exception as exc:
            err = ForgeSelError(
                message=f"Forge session selector failed: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            span.set_attribute("skill_forge.sel.run.success", False)
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="skill_forge.sel.run.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={"message": err.message},
                )
            )
            raise err

    return result
