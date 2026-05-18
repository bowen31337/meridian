"""
GET /v1/sessions/{id}/events endpoint conformance suite.

Tests cover:
  - GET /v1/sessions/{id}/events returns 200 with JSON array by default.
  - Response body is a JSON array.
  - Unknown session returns empty array (no 404).
  - Each event object carries seq, ts, type, and data fields.
  - thread_id is included when present; omitted when None.
  - ?since=<seq> returns only events with seq > since.
  - ?type=<t> returns only events matching that type.
  - ?type=<t1,t2> returns events matching either type.
  - ?type filter excludes non-matching events.
  - Accept: application/x-ndjson returns NDJSON stream.
  - Each NDJSON line is valid JSON with correct fields.
  - On read failure, returns 500 with code "session_events_failed".
  - On read failure, an audit log entry is written with event "session.events.read.failed".
  - Audit log detail includes session_id and since.
  - OTel span "session.events.read" is emitted on success.
  - OTel span is set to ERROR status on failure.
  - create_app wires the events router when storage_root is supplied.
  - create_app omits the events route when storage_root is None.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import LocalEventLogWriter

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path, audit_log: FileAuditLog | None = None) -> TestClient:
    audit = audit_log or FileAuditLog(storage_root)
    app = create_app(audit, storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _seed_many(
    storage_root: Path,
    session_id: str,
    events: list[tuple[str, dict[str, Any]]],
    *,
    thread_ids: list[str | None] | None = None,
) -> list[int]:
    """Write multiple events for one session using a shared writer (monotonic seq)."""

    async def _go() -> list[int]:
        writer = LocalEventLogWriter(storage_root)
        seqs = []
        for i, (event_type, data) in enumerate(events):
            tid = thread_ids[i] if thread_ids else None
            seqs.append(await writer.append(session_id, event_type, data, thread_id=tid))  # type: ignore[arg-type]
        return seqs

    return asyncio.run(_go())


def _seed(
    storage_root: Path,
    session_id: str,
    event_type: str,
    data: dict[str, Any],
    *,
    thread_id: str | None = None,
) -> int:
    return _seed_many(storage_root, session_id, [(event_type, data)], thread_ids=[thread_id])[0]


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Integration tests: GET /v1/sessions/{id}/events
# ---------------------------------------------------------------------------


class TestSessionEventsEndpoint:
    def test_returns_200_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get("/v1/sessions/sess1/events")
        assert resp.status_code == 200

    def test_response_is_json_array(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/sess2/events").json()
        assert isinstance(body, list)

    def test_empty_array_for_unknown_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/no-such-session/events").json()
        assert body == []

    def test_returns_all_events(self, storage_root: Path) -> None:
        _seed_many(storage_root, "multi-sess", [
            ("session.created", {"reason": "init"}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/multi-sess/events").json()
        assert len(body) == 2

    def test_event_has_seq_field(self, storage_root: Path) -> None:
        _seed(storage_root, "seq-sess", "session.created", {})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/seq-sess/events").json()
        assert "seq" in body[0]
        assert isinstance(body[0]["seq"], int)

    def test_event_has_ts_field(self, storage_root: Path) -> None:
        _seed(storage_root, "ts-sess", "session.created", {})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/ts-sess/events").json()
        assert "ts" in body[0]

    def test_event_has_type_field(self, storage_root: Path) -> None:
        _seed(storage_root, "type-sess", "session.created", {})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/type-sess/events").json()
        assert body[0]["type"] == "session.created"

    def test_event_has_data_field(self, storage_root: Path) -> None:
        _seed(storage_root, "data-sess", "session.created", {"key": "value"})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/data-sess/events").json()
        assert body[0]["data"] == {"key": "value"}

    def test_thread_id_included_when_present(self, storage_root: Path) -> None:
        _seed(storage_root, "tid-sess", "session.created", {}, thread_id="thread-1")
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/tid-sess/events").json()
        assert body[0]["thread_id"] == "thread-1"

    def test_thread_id_omitted_when_none(self, storage_root: Path) -> None:
        _seed(storage_root, "notid-sess", "session.created", {})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/notid-sess/events").json()
        assert "thread_id" not in body[0]

    def test_seq_values_are_ascending(self, storage_root: Path) -> None:
        _seed_many(storage_root, "asc-sess", [("session.created", {"i": i}) for i in range(3)])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/asc-sess/events").json()
        seqs = [e["seq"] for e in body]
        assert seqs == sorted(seqs)


# ---------------------------------------------------------------------------
# ?since filter
# ---------------------------------------------------------------------------


class TestSinceFilter:
    def test_since_minus_one_returns_all(self, storage_root: Path) -> None:
        _seed_many(storage_root, "since-all", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/since-all/events?since=-1").json()
        assert len(body) == 2

    def test_since_zero_excludes_seq_zero(self, storage_root: Path) -> None:
        _seed_many(storage_root, "since-zero", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/since-zero/events?since=0").json()
        seqs = [e["seq"] for e in body]
        assert 0 not in seqs
        assert 1 in seqs

    def test_since_returns_only_later_events(self, storage_root: Path) -> None:
        _seed_many(storage_root, "since-later", [("session.created", {"i": i}) for i in range(5)])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/since-later/events?since=2").json()
        assert all(e["seq"] > 2 for e in body)

    def test_since_beyond_all_events_returns_empty(self, storage_root: Path) -> None:
        _seed(storage_root, "since-empty", "session.created", {})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/since-empty/events?since=99").json()
        assert body == []


# ---------------------------------------------------------------------------
# ?type filter
# ---------------------------------------------------------------------------


class TestTypeFilter:
    def test_type_filter_returns_only_matching_type(self, storage_root: Path) -> None:
        _seed_many(storage_root, "type-filt", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/type-filt/events?type=session.created").json()
        assert all(e["type"] == "session.created" for e in body)
        assert len(body) == 1

    def test_type_filter_excludes_other_types(self, storage_root: Path) -> None:
        _seed_many(storage_root, "type-excl", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/type-excl/events?type=session.phase_change").json()
        assert not any(e["type"] == "session.created" for e in body)

    def test_type_filter_multiple_types(self, storage_root: Path) -> None:
        _seed_many(storage_root, "type-multi", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
            ("error", {"msg": "boom"}),
        ])
        client = _make_client(storage_root)
        body = client.get(
            "/v1/sessions/type-multi/events?type=session.created,session.phase_change"
        ).json()
        types = {e["type"] for e in body}
        assert types == {"session.created", "session.phase_change"}
        assert len(body) == 2

    def test_type_filter_no_match_returns_empty(self, storage_root: Path) -> None:
        _seed(storage_root, "type-none", "session.created", {})
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/type-none/events?type=error").json()
        assert body == []

    def test_since_and_type_combined(self, storage_root: Path) -> None:
        _seed_many(storage_root, "combo", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
            ("session.phase_change", {"before": "running", "after": "done"}),
        ])
        client = _make_client(storage_root)
        body = client.get("/v1/sessions/combo/events?since=0&type=session.phase_change").json()
        assert all(e["type"] == "session.phase_change" for e in body)
        assert all(e["seq"] > 0 for e in body)


# ---------------------------------------------------------------------------
# NDJSON streaming
# ---------------------------------------------------------------------------


class TestNdjsonFormat:
    def test_ndjson_accept_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get(
            "/v1/sessions/ndjson-sess/events",
            headers={"accept": "application/x-ndjson"},
        )
        assert resp.status_code == 200

    def test_ndjson_content_type(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get(
            "/v1/sessions/ndjson-ct/events",
            headers={"accept": "application/x-ndjson"},
        )
        assert "application/x-ndjson" in resp.headers["content-type"]

    def test_ndjson_each_line_is_valid_json(self, storage_root: Path) -> None:
        _seed_many(storage_root, "ndjson-lines", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        resp = client.get(
            "/v1/sessions/ndjson-lines/events",
            headers={"accept": "application/x-ndjson"},
        )
        lines = [ln for ln in resp.text.splitlines() if ln.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "seq" in obj
            assert "type" in obj

    def test_ndjson_empty_for_unknown_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.get(
            "/v1/sessions/ndjson-empty/events",
            headers={"accept": "application/x-ndjson"},
        )
        assert resp.text.strip() == ""

    def test_ndjson_since_filter_applies(self, storage_root: Path) -> None:
        _seed_many(storage_root, "ndjson-since", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        resp = client.get(
            "/v1/sessions/ndjson-since/events?since=0",
            headers={"accept": "application/x-ndjson"},
        )
        lines = [ln for ln in resp.text.splitlines() if ln.strip()]
        objs = [json.loads(ln) for ln in lines]
        assert all(o["seq"] > 0 for o in objs)


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestSessionEventsFailure:
    def _corrupt_session(self, storage_root: Path, session_id: str) -> None:
        bad_file = storage_root / f"{session_id}.ndjson"
        bad_file.write_text("not-valid-json\n")

    def test_read_failure_returns_500(self, storage_root: Path) -> None:
        self._corrupt_session(storage_root, "bad-sess")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.get("/v1/sessions/bad-sess/events")
        assert resp.status_code == 500

    def test_read_failure_response_has_code(self, storage_root: Path) -> None:
        self._corrupt_session(storage_root, "bad-code")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        resp = client.get("/v1/sessions/bad-code/events")
        assert resp.json()["error"]["code"] == "session_events_failed"

    def test_read_failure_writes_audit_log(self, storage_root: Path) -> None:
        self._corrupt_session(storage_root, "bad-audit")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.get("/v1/sessions/bad-audit/events")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.events.read.failed" for r in records)

    def test_read_failure_audit_has_session_id(self, storage_root: Path) -> None:
        self._corrupt_session(storage_root, "bad-detail")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.get("/v1/sessions/bad-detail/events")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.read.failed")
        assert rec["detail"]["session_id"] == "bad-detail"

    def test_read_failure_audit_has_since(self, storage_root: Path) -> None:
        self._corrupt_session(storage_root, "bad-since")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.get("/v1/sessions/bad-since/events?since=5")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.read.failed")
        assert rec["detail"]["since"] == 5


# ---------------------------------------------------------------------------
# OTel span tests
# ---------------------------------------------------------------------------


class TestSessionEventsOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_events_read_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.get("/v1/sessions/otel-sess/events")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.events.read" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        bad_file = storage_root / "otel-bad.ndjson"
        bad_file.write_text("not-valid-json\n")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.get("/v1/sessions/otel-bad/events")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("session.events.read")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# SSE streaming (?stream=true)
# ---------------------------------------------------------------------------


def _parse_sse_frames(body: str) -> list[dict[str, Any]]:
    """Parse SSE text body into a list of frame dicts with event/id/data keys."""
    frames = []
    for block in body.split("\n\n"):
        frame: dict[str, Any] = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("event:"):
                frame["event"] = line[len("event:"):].strip()
            elif line.startswith("id:"):
                frame["id"] = line[len("id:"):].strip()
            elif line.startswith("data:"):
                frame["data"] = json.loads(line[len("data:"):].strip())
        if frame:
            frames.append(frame)
    return frames


class TestSseStream:
    def test_stream_returns_200(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-200", "session.created", {})
        client = _make_client(storage_root)
        resp = client.get("/v1/sessions/sse-200/events?stream=true")
        assert resp.status_code == 200

    def test_stream_content_type_is_event_stream(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-ct", "session.created", {})
        client = _make_client(storage_root)
        resp = client.get("/v1/sessions/sse-ct/events?stream=true")
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_frame_has_event_field(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-ev", "session.created", {})
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-ev/events?stream=true").text)
        assert frames[0]["event"] == "session.created"

    def test_stream_frame_has_id_field(self, storage_root: Path) -> None:
        seqs = _seed_many(storage_root, "sse-id", [("session.created", {})])
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-id/events?stream=true").text)
        assert frames[0]["id"] == str(seqs[0])

    def test_stream_frame_has_data_field(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-data", "session.created", {"k": "v"})
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-data/events?stream=true").text)
        assert frames[0]["data"]["data"] == {"k": "v"}

    def test_stream_data_contains_seq(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-seq", "session.created", {})
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-seq/events?stream=true").text)
        assert "seq" in frames[0]["data"]
        assert isinstance(frames[0]["data"]["seq"], int)

    def test_stream_data_contains_type(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-type", "session.phase_change", {"before": "created", "after": "running"})
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-type/events?stream=true").text)
        assert frames[0]["data"]["type"] == "session.phase_change"

    def test_stream_event_field_matches_data_type(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-match", "session.created", {})
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-match/events?stream=true").text)
        assert frames[0]["event"] == frames[0]["data"]["type"]

    def test_stream_id_field_matches_data_seq(self, storage_root: Path) -> None:
        _seed(storage_root, "sse-idseq", "session.created", {})
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-idseq/events?stream=true").text)
        assert frames[0]["id"] == str(frames[0]["data"]["seq"])

    def test_stream_multiple_events_in_order(self, storage_root: Path) -> None:
        _seed_many(storage_root, "sse-multi", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
            ("session.phase_change", {"before": "running", "after": "done"}),
        ])
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-multi/events?stream=true").text)
        seqs = [int(f["id"]) for f in frames]
        assert seqs == sorted(seqs)
        assert len(frames) == 3

    def test_stream_empty_for_unknown_session(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/sse-unknown/events?stream=true").text)
        assert frames == []

    def test_stream_since_filter(self, storage_root: Path) -> None:
        seqs = _seed_many(storage_root, "sse-since", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        frames = _parse_sse_frames(
            client.get(f"/v1/sessions/sse-since/events?stream=true&since={seqs[0]}").text
        )
        assert len(frames) == 1
        assert all(int(f["id"]) > seqs[0] for f in frames)

    def test_stream_type_filter(self, storage_root: Path) -> None:
        _seed_many(storage_root, "sse-tf", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/sse-tf/events?stream=true&type=session.created").text
        )
        assert len(frames) == 1
        assert all(f["event"] == "session.created" for f in frames)

    def test_stream_last_event_id_resumes_from_watermark(self, storage_root: Path) -> None:
        seqs = _seed_many(storage_root, "sse-lei", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
            ("error", {"msg": "oops"}),
        ])
        client = _make_client(storage_root)
        frames = _parse_sse_frames(
            client.get(
                "/v1/sessions/sse-lei/events?stream=true",
                headers={"last-event-id": str(seqs[0])},
            ).text
        )
        assert len(frames) == 2
        assert all(int(f["id"]) > seqs[0] for f in frames)

    def test_stream_last_event_id_overrides_since(self, storage_root: Path) -> None:
        seqs = _seed_many(storage_root, "sse-lei-override", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        # since=-1 would return all events; Last-Event-ID=seqs[0] narrows to events after seqs[0]
        frames = _parse_sse_frames(
            client.get(
                "/v1/sessions/sse-lei-override/events?stream=true&since=-1",
                headers={"last-event-id": str(seqs[0])},
            ).text
        )
        assert len(frames) == 1
        assert all(int(f["id"]) > seqs[0] for f in frames)

    def test_stream_failure_yields_error_event(self, storage_root: Path) -> None:
        bad_file = storage_root / "sse-fail.ndjson"
        bad_file.write_text("not-valid-json\n")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/sse-fail/events?stream=true").text
        )
        assert any(f.get("event") == "error" for f in frames)

    def test_stream_failure_error_event_has_code(self, storage_root: Path) -> None:
        bad_file = storage_root / "sse-fail-code.ndjson"
        bad_file.write_text("not-valid-json\n")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/sse-fail-code/events?stream=true").text
        )
        err_frame = next(f for f in frames if f.get("event") == "error")
        assert err_frame["data"]["code"] == "session_events_failed"

    def test_stream_failure_writes_audit_log(self, storage_root: Path) -> None:
        bad_file = storage_root / "sse-fail-audit.ndjson"
        bad_file.write_text("not-valid-json\n")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.get("/v1/sessions/sse-fail-audit/events?stream=true")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.events.stream.failed" for r in records)

    def test_stream_failure_audit_has_session_id(self, storage_root: Path) -> None:
        bad_file = storage_root / "sse-fail-sid.ndjson"
        bad_file.write_text("not-valid-json\n")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.get("/v1/sessions/sse-fail-sid/events?stream=true")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.stream.failed")
        assert rec["detail"]["session_id"] == "sse-fail-sid"

    def test_stream_failure_audit_has_since(self, storage_root: Path) -> None:
        bad_file = storage_root / "sse-fail-snc.ndjson"
        bad_file.write_text("not-valid-json\n")
        audit = FileAuditLog(storage_root)
        client = _make_client(storage_root, audit)
        client.get("/v1/sessions/sse-fail-snc/events?stream=true&since=7")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.stream.failed")
        assert rec["detail"]["since"] == 7

    def test_stream_otel_span_emitted(self, storage_root: Path) -> None:
        from tests._otel_shared import otel_exporter as _otel_exporter
        _otel_exporter.clear()
        _seed(storage_root, "sse-otel", "session.created", {})
        client = _make_client(storage_root)
        client.get("/v1/sessions/sse-otel/events?stream=true")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "session.events.stream" in span_names


# ---------------------------------------------------------------------------
# Router wiring tests
# ---------------------------------------------------------------------------


class TestEventsRouterWiring:
    def test_no_storage_root_no_route(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/sessions/any/events")
        assert resp.status_code == 404

    def test_with_storage_root_route_exists(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/sessions/any/events")
        assert resp.status_code == 200
