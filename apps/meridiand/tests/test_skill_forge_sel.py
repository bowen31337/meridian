"""
Forge Session Selector (SEL) conformance suite.

Tests cover:
  - collect_terminated_sessions returns empty list when events dir is absent.
  - collect_terminated_sessions returns empty list when no terminal phase event.
  - collect_terminated_sessions includes session with terminal phase "terminated".
  - collect_terminated_sessions includes session with terminal phase "completed".
  - collect_terminated_sessions excludes session whose phase never reaches terminal.
  - collect_terminated_sessions extracts tool calls in event-seq order.
  - collect_terminated_sessions returns empty tool_calls when no tool_call.requested events.
  - collect_terminated_sessions returns multiple summaries for multiple terminated sessions.
  - collect_terminated_sessions sets terminal_phase field correctly.
  - collect_terminated_sessions sets event_count to total event count for the session.
  - cluster_trajectories returns empty list for empty input.
  - cluster_trajectories returns one cluster for a single session.
  - cluster_trajectories places sessions with identical tool sequence + phase in same cluster.
  - cluster_trajectories places sessions with different tool sequences in different clusters.
  - cluster_trajectories places sessions with same tools but different terminal phase
    in different clusters.
  - cluster_trajectories cluster ID is deterministic (same key → same ID across calls).
  - cluster_trajectories cluster members contain all sessions in that cluster.
  - cluster_trajectories member tool_calls match the session summary tool_calls.
  - cluster_trajectories sorts clusters by size descending.
  - run_forge_session_selector returns ForgeSelResult.
  - ForgeSelResult session_count equals number of terminated sessions found.
  - ForgeSelResult cluster_count equals number of clusters produced.
  - ForgeSelResult clusters is a list of Cluster instances.
  - run_forge_session_selector emits OTel span "skill_forge.sel.run".
  - OTel span has skill_forge.sel.session_count attribute.
  - OTel span has skill_forge.sel.cluster_count attribute.
  - OTel span sets skill_forge.sel.run.success=True on success.
  - run_forge_session_selector writes audit entry "skill_forge.sel.ran" on success.
  - Success audit entry level is "info".
  - Success audit detail contains session_count.
  - Success audit detail contains cluster_count.
  - run_forge_session_selector raises ForgeSelError on scan failure.
  - ForgeSelError carries error code "forge_sel_failed".
  - On failure: OTel span sets skill_forge.sel.run.success=False.
  - On failure: writes audit entry "skill_forge.sel.run.failed".
  - Failure audit entry level is "error".
  - Failure audit detail contains message.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from meridiand._audit import FileAuditLog
from meridiand._cluster_extraction import Cluster
from meridiand._skill_forge_sel import (
    ForgeSelError,
    ForgeSelResult,
    SessionSummary,
    cluster_trajectories,
    collect_terminated_sessions,
    run_forge_session_selector,
)
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_events(
    storage_root: Path,
    session_id: str,
    events: list[dict[str, Any]],
) -> None:
    """Write a list of raw event dicts to the canonical NDJSON path for session_id."""
    today = datetime.now(UTC)
    event_path = (
        storage_root
        / "events"
        / today.strftime("%Y")
        / today.strftime("%m")
        / today.strftime("%d")
        / f"{session_id}.ndjson"
    )
    event_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e) for e in events]
    event_path.write_text("\n".join(lines) + "\n")


def _make_phase_change_event(seq: int, after: str, before: str = "running") -> dict[str, Any]:
    return {
        "seq": seq,
        "ts": datetime.now(UTC).isoformat(),
        "type": "session.phase_change",
        "data": {
            "before": before,
            "after": after,
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "test",
        },
        "thread_id": None,
    }


def _make_tool_call_event(seq: int, tool_name: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "ts": datetime.now(UTC).isoformat(),
        "type": "tool_call.requested",
        "data": {"tool_id": f"tool_{seq}", "tool_name": tool_name, "args": {}},
        "thread_id": None,
    }


def _make_generic_event(seq: int, event_type: str = "session.created") -> dict[str, Any]:
    return {
        "seq": seq,
        "ts": datetime.now(UTC).isoformat(),
        "type": event_type,
        "data": {},
        "thread_id": None,
    }


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# collect_terminated_sessions
# ---------------------------------------------------------------------------


class TestCollectTerminatedSessions:
    def test_empty_when_no_events_dir(self, storage_root: Path) -> None:
        result = collect_terminated_sessions(storage_root)
        assert result == []

    def test_empty_when_no_terminal_phase_event(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_nophase",
            [
                _make_generic_event(1, "session.created"),
                _make_tool_call_event(2, "Bash"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        assert result == []

    def test_includes_session_with_terminated_phase(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_term",
            [
                _make_tool_call_event(1, "Bash"),
                _make_phase_change_event(2, "terminated"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        assert any(s.session_id == "sess_term" for s in result)

    def test_includes_session_with_completed_phase(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_comp",
            [
                _make_phase_change_event(1, "completed"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        assert any(s.session_id == "sess_comp" for s in result)

    def test_excludes_non_terminal_session(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_running",
            [
                _make_generic_event(1, "session.created"),
                _make_phase_change_event(2, "running"),
            ],
        )
        _write_events(
            storage_root,
            "sess_term2",
            [
                _make_phase_change_event(1, "terminated"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        session_ids = [s.session_id for s in result]
        assert "sess_running" not in session_ids
        assert "sess_term2" in session_ids

    def test_extracts_tool_calls_in_order(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_order",
            [
                _make_tool_call_event(1, "Read"),
                _make_tool_call_event(2, "Bash"),
                _make_tool_call_event(3, "Write"),
                _make_phase_change_event(4, "terminated"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        summary = next(s for s in result if s.session_id == "sess_order")
        assert summary.tool_calls == ["Read", "Bash", "Write"]

    def test_empty_tool_calls_when_no_tool_call_events(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_notools",
            [
                _make_generic_event(1, "session.created"),
                _make_phase_change_event(2, "terminated"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        summary = next(s for s in result if s.session_id == "sess_notools")
        assert summary.tool_calls == []

    def test_returns_multiple_summaries(self, storage_root: Path) -> None:
        for i in range(3):
            _write_events(
                storage_root,
                f"sess_multi_{i}",
                [
                    _make_tool_call_event(1, "Bash"),
                    _make_phase_change_event(2, "terminated"),
                ],
            )
        result = collect_terminated_sessions(storage_root)
        assert len(result) == 3

    def test_terminal_phase_field_set(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_phase_field",
            [
                _make_phase_change_event(1, "terminated"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        summary = next(s for s in result if s.session_id == "sess_phase_field")
        assert summary.terminal_phase == "terminated"

    def test_event_count_set(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_ecount",
            [
                _make_generic_event(1),
                _make_tool_call_event(2, "Bash"),
                _make_phase_change_event(3, "terminated"),
            ],
        )
        result = collect_terminated_sessions(storage_root)
        summary = next(s for s in result if s.session_id == "sess_ecount")
        assert summary.event_count == 3


# ---------------------------------------------------------------------------
# cluster_trajectories
# ---------------------------------------------------------------------------


class TestClusterTrajectories:
    def test_empty_input_returns_empty(self) -> None:
        assert cluster_trajectories([]) == []

    def test_single_session_returns_one_cluster(self) -> None:
        summaries = [SessionSummary("s1", ["Bash"], "terminated", 2)]
        clusters = cluster_trajectories(summaries)
        assert len(clusters) == 1

    def test_same_sequence_and_phase_in_same_cluster(self) -> None:
        summaries = [
            SessionSummary("s1", ["Bash", "Read"], "terminated", 3),
            SessionSummary("s2", ["Bash", "Read"], "terminated", 4),
        ]
        clusters = cluster_trajectories(summaries)
        assert len(clusters) == 1
        member_ids = {m.session_id for m in clusters[0].members}
        assert member_ids == {"s1", "s2"}

    def test_different_sequences_in_different_clusters(self) -> None:
        summaries = [
            SessionSummary("s1", ["Bash"], "terminated", 2),
            SessionSummary("s2", ["Read", "Write"], "terminated", 3),
        ]
        clusters = cluster_trajectories(summaries)
        assert len(clusters) == 2

    def test_same_tools_different_phase_in_different_clusters(self) -> None:
        summaries = [
            SessionSummary("s1", ["Bash"], "terminated", 2),
            SessionSummary("s2", ["Bash"], "completed", 2),
        ]
        clusters = cluster_trajectories(summaries)
        assert len(clusters) == 2

    def test_cluster_id_is_deterministic(self) -> None:
        summaries = [SessionSummary("s1", ["Bash", "Edit"], "terminated", 3)]
        c1 = cluster_trajectories(summaries)
        c2 = cluster_trajectories(summaries)
        assert c1[0].id == c2[0].id

    def test_cluster_members_contain_all_matching_sessions(self) -> None:
        summaries = [
            SessionSummary("sA", ["Glob"], "terminated", 2),
            SessionSummary("sB", ["Glob"], "terminated", 2),
            SessionSummary("sC", ["Glob"], "terminated", 3),
        ]
        clusters = cluster_trajectories(summaries)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 3

    def test_member_tool_calls_match_summary(self) -> None:
        summaries = [SessionSummary("s1", ["Read", "Write"], "terminated", 3)]
        clusters = cluster_trajectories(summaries)
        assert clusters[0].members[0].tool_calls == ["Read", "Write"]

    def test_sorted_by_size_descending(self) -> None:
        summaries = [
            SessionSummary("s1", ["Bash"], "terminated", 2),
            SessionSummary("s2", ["Read"], "terminated", 2),
            SessionSummary("s3", ["Read"], "terminated", 2),
        ]
        clusters = cluster_trajectories(summaries)
        sizes = [len(c.members) for c in clusters]
        assert sizes == sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# run_forge_session_selector: result fields
# ---------------------------------------------------------------------------


class TestRunForgeSessionSelectorResult:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_returns_forge_sel_result(self, storage_root: Path) -> None:
        result = asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        assert isinstance(result, ForgeSelResult)

    def test_session_count_matches_terminated_count(self, storage_root: Path) -> None:
        for i in range(2):
            _write_events(
                storage_root,
                f"sess_cnt_{i}",
                [
                    _make_phase_change_event(1, "terminated"),
                ],
            )
        result = asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        assert result.session_count == 2

    def test_cluster_count_matches_clusters(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_c1",
            [
                _make_tool_call_event(1, "Bash"),
                _make_phase_change_event(2, "terminated"),
            ],
        )
        _write_events(
            storage_root,
            "sess_c2",
            [
                _make_tool_call_event(1, "Read"),
                _make_phase_change_event(2, "terminated"),
            ],
        )
        result = asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        assert result.cluster_count == len(result.clusters)
        assert result.cluster_count == 2

    def test_clusters_is_list_of_cluster(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_cltype",
            [
                _make_phase_change_event(1, "completed"),
            ],
        )
        result = asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        assert isinstance(result.clusters, list)
        for cluster in result.clusters:
            assert isinstance(cluster, Cluster)


# ---------------------------------------------------------------------------
# run_forge_session_selector: OTel
# ---------------------------------------------------------------------------


class TestRunForgeSessionSelectorOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.sel.run")

    def test_emits_sel_run_span(self, storage_root: Path) -> None:
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.sel.run" in span_names

    def test_span_has_session_count_attribute(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_otel1",
            [
                _make_phase_change_event(1, "terminated"),
            ],
        )
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.sel.session_count"] == 1

    def test_span_has_cluster_count_attribute(self, storage_root: Path) -> None:
        _write_events(
            storage_root,
            "sess_otel2",
            [
                _make_phase_change_event(1, "terminated"),
            ],
        )
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        span = self._get_span()
        assert span is not None
        assert "skill_forge.sel.cluster_count" in span.attributes

    def test_span_success_true_on_success(self, storage_root: Path) -> None:
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.sel.run.success"] is True

    def test_span_success_false_on_failure(self, storage_root: Path) -> None:
        with (
            patch(
                "meridiand._skill_forge_sel.collect_terminated_sessions",
                side_effect=RuntimeError("scan exploded"),
            ),
            pytest.raises(ForgeSelError),
        ):
            asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.sel.run.success"] is False


# ---------------------------------------------------------------------------
# run_forge_session_selector: audit log
# ---------------------------------------------------------------------------


class TestRunForgeSessionSelectorAudit:
    def test_success_writes_ran_audit_entry(self, storage_root: Path) -> None:
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.sel.ran" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.sel.ran"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_session_count(self, storage_root: Path) -> None:
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.sel.ran"
        )
        assert "session_count" in record["detail"]

    def test_success_audit_detail_has_cluster_count(self, storage_root: Path) -> None:
        asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.sel.ran"
        )
        assert "cluster_count" in record["detail"]

    def test_failure_raises_forge_sel_error(self, storage_root: Path) -> None:
        with (
            patch(
                "meridiand._skill_forge_sel.collect_terminated_sessions",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ForgeSelError),
        ):
            asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))

    def test_forge_sel_error_code(self, storage_root: Path) -> None:
        with (
            patch(
                "meridiand._skill_forge_sel.collect_terminated_sessions",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ForgeSelError) as exc_info,
        ):
            asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        assert exc_info.value.code == "forge_sel_failed"

    def test_failure_writes_failed_audit_entry(self, storage_root: Path) -> None:
        with (
            patch(
                "meridiand._skill_forge_sel.collect_terminated_sessions",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ForgeSelError),
        ):
            asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.sel.run.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        with (
            patch(
                "meridiand._skill_forge_sel.collect_terminated_sessions",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ForgeSelError),
        ):
            asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.sel.run.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        with (
            patch(
                "meridiand._skill_forge_sel.collect_terminated_sessions",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(ForgeSelError),
        ):
            asyncio.run(run_forge_session_selector(storage_root, FileAuditLog(storage_root)))
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.sel.run.failed"
        )
        assert "message" in record["detail"] and record["detail"]["message"]
