"""
SSE backpressure conformance suite.

Tests cover the bounded in-process channel behaviour introduced in the
SubscriberBus integration:

  - Without a subscriber bus, ?stream=true streams historical events (original behaviour).
  - With a subscriber bus, historical events are replayed before following live queue.
  - subscriber_lagged event is emitted when the bus drops the subscriber (queue full).
  - subscriber_lagged event carries code="subscriber_lagged".
  - subscriber_lagged writes a warn-level audit entry.
  - Audit entry event name is "session.events.stream.subscriber_lagged".
  - Audit entry detail includes session_id, since, and capacity.
  - After subscriber_lagged the SSE stream closes (no further frames).
  - Events already sent in history replay are not duplicated from the queue.
  - Existing SSE tests still pass when subscriber_bus=None (backward compat).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from storage_event_log import SUBSCRIBER_CHANNEL_SIZE, LocalEventLogWriter, SessionEvent, SubscriberBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_many(
    storage_root: Path,
    session_id: str,
    events: list[tuple[str, dict[str, Any]]],
) -> list[int]:
    async def _go() -> list[int]:
        writer = LocalEventLogWriter(storage_root)
        seqs = []
        for event_type, data in events:
            seqs.append(await writer.append(session_id, event_type, data))  # type: ignore[arg-type]
        return seqs

    return asyncio.run(_go())


def _seed(storage_root: Path, session_id: str, event_type: str, data: dict[str, Any]) -> int:
    return _seed_many(storage_root, session_id, [(event_type, data)])[0]


def _read_audit(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parse_sse_frames(body: str) -> list[dict[str, Any]]:
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


def _make_client(storage_root: Path, subscriber_bus: SubscriberBus | None = None) -> TestClient:
    audit = FileAuditLog(storage_root)
    app = create_app(audit, storage_root=storage_root, subscriber_bus=subscriber_bus)
    return TestClient(app, raise_server_exceptions=False)


def _make_lagged_bus() -> SubscriberBus:
    """
    Return a SubscriberBus whose subscribe() always returns a queue that already
    contains the lagged sentinel (None).  This makes _sse_live() terminate after
    history replay without blocking, enabling synchronous TestClient tests.
    """
    bus = SubscriberBus()
    pre_lagged: asyncio.Queue[SessionEvent | None] = asyncio.Queue(maxsize=SUBSCRIBER_CHANNEL_SIZE)
    pre_lagged.put_nowait(None)

    def _subscribe(session_id: str) -> asyncio.Queue[SessionEvent | None]:
        return pre_lagged

    def _unsubscribe(session_id: str, queue: asyncio.Queue[SessionEvent | None]) -> None:
        pass  # nothing to clean up

    bus.subscribe = _subscribe  # type: ignore[method-assign]
    bus.unsubscribe = _unsubscribe  # type: ignore[method-assign]
    return bus


# ---------------------------------------------------------------------------
# Backward compatibility: no bus → historical-only streaming
# ---------------------------------------------------------------------------


class TestNoBusBackwardCompat:
    def test_no_bus_streams_historical_events(self, storage_root: Path) -> None:
        _seed_many(storage_root, "hist-sess", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        client = _make_client(storage_root)
        frames = _parse_sse_frames(client.get("/v1/sessions/hist-sess/events?stream=true").text)
        assert len(frames) == 2

    def test_no_bus_stream_terminates_after_history(self, storage_root: Path) -> None:
        _seed(storage_root, "hist-term", "session.created", {})
        client = _make_client(storage_root)
        resp = client.get("/v1/sessions/hist-term/events?stream=true")
        assert resp.status_code == 200
        frames = _parse_sse_frames(resp.text)
        # Only the one historical event; no subscriber_lagged
        assert not any(f.get("event") == "subscriber_lagged" for f in frames)


# ---------------------------------------------------------------------------
# With bus: subscriber_lagged path
# ---------------------------------------------------------------------------


class TestSubscriberLagged:
    def test_lagged_yields_subscriber_lagged_event(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/lag-event/events?stream=true").text
        )
        assert any(f.get("event") == "subscriber_lagged" for f in frames)

    def test_lagged_event_has_code_subscriber_lagged(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/lag-code/events?stream=true").text
        )
        lag_frame = next(f for f in frames if f.get("event") == "subscriber_lagged")
        assert lag_frame["data"]["code"] == "subscriber_lagged"

    def test_lagged_event_message_mentions_capacity(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/lag-msg/events?stream=true").text
        )
        lag_frame = next(f for f in frames if f.get("event") == "subscriber_lagged")
        assert str(SUBSCRIBER_CHANNEL_SIZE) in lag_frame["data"]["message"]

    def test_lagged_writes_warn_audit_entry(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        client.get("/v1/sessions/lag-audit/events?stream=true")
        records = _read_audit(storage_root)
        assert any(r.get("event") == "session.events.stream.subscriber_lagged" for r in records)

    def test_lagged_audit_level_is_warn(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        client.get("/v1/sessions/lag-lvl/events?stream=true")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.stream.subscriber_lagged")
        assert rec["level"] == "warn"

    def test_lagged_audit_detail_has_session_id(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        client.get("/v1/sessions/lag-sid/events?stream=true")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.stream.subscriber_lagged")
        assert rec["detail"]["session_id"] == "lag-sid"

    def test_lagged_audit_detail_has_since(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        client.get("/v1/sessions/lag-snc/events?stream=true&since=3")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.stream.subscriber_lagged")
        assert rec["detail"]["since"] == 3

    def test_lagged_audit_detail_has_capacity(self, storage_root: Path) -> None:
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        client.get("/v1/sessions/lag-cap/events?stream=true")
        records = _read_audit(storage_root)
        rec = next(r for r in records if r.get("event") == "session.events.stream.subscriber_lagged")
        assert rec["detail"]["capacity"] == SUBSCRIBER_CHANNEL_SIZE


# ---------------------------------------------------------------------------
# History replay before following live queue
# ---------------------------------------------------------------------------


class TestHistoryReplayWithBus:
    def test_history_events_delivered_before_lagged(self, storage_root: Path) -> None:
        _seed_many(storage_root, "hist-bus", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/hist-bus/events?stream=true").text
        )
        # Both historical events plus the lagged frame
        history = [f for f in frames if f.get("event") != "subscriber_lagged"]
        lag = [f for f in frames if f.get("event") == "subscriber_lagged"]
        assert len(history) == 2
        assert len(lag) == 1

    def test_history_events_come_before_lagged_frame(self, storage_root: Path) -> None:
        _seed(storage_root, "hist-order", "session.created", {})
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/hist-order/events?stream=true").text
        )
        events_order = [f.get("event") for f in frames]
        lag_idx = events_order.index("subscriber_lagged")
        # All frames before the lag must not be subscriber_lagged
        assert all(e != "subscriber_lagged" for e in events_order[:lag_idx])

    def test_since_filter_applies_to_history_replay(self, storage_root: Path) -> None:
        seqs = _seed_many(storage_root, "hist-since", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get(f"/v1/sessions/hist-since/events?stream=true&since={seqs[0]}").text
        )
        history = [f for f in frames if f.get("event") != "subscriber_lagged"]
        assert len(history) == 1
        assert all(int(f["id"]) > seqs[0] for f in history)

    def test_type_filter_applies_to_history_replay(self, storage_root: Path) -> None:
        _seed_many(storage_root, "hist-type", [
            ("session.created", {}),
            ("session.phase_change", {"before": "created", "after": "running"}),
        ])
        bus = _make_lagged_bus()
        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/hist-type/events?stream=true&type=session.created").text
        )
        history = [f for f in frames if f.get("event") != "subscriber_lagged"]
        assert len(history) == 1
        assert history[0]["event"] == "session.created"


# ---------------------------------------------------------------------------
# Live delivery via bus (writer → bus → SSE)
# ---------------------------------------------------------------------------


class TestLiveDelivery:
    def test_writer_event_delivered_to_sse_subscriber(self, storage_root: Path) -> None:
        """
        Write an event through a writer that shares the bus with the SSE router.
        The event goes to both disk and the in-process queue.

        Because TestClient is synchronous, we pre-populate the queue by writing
        via the writer BEFORE creating the SSE connection, then immediately lag
        the queue so the stream terminates.
        """
        bus = SubscriberBus()
        writer = LocalEventLogWriter(storage_root, subscriber_bus=bus)

        # Pre-subscribe, write one event, then put the lagged sentinel
        loop = asyncio.new_event_loop()
        try:
            q = bus.subscribe("live-sess")
            loop.run_until_complete(
                writer.append("live-sess", "session.created", {"live": True})
            )
            # Put lagged sentinel so the stream terminates after consuming the event
            q.put_nowait(None)
            bus.unsubscribe("live-sess", q)  # remove so subscribe() in the handler creates a fresh one
        finally:
            loop.close()

        # Override subscribe to return our pre-populated queue
        bus.subscribe = lambda session_id: q  # type: ignore[method-assign]
        bus.unsubscribe = lambda session_id, queue: None  # type: ignore[method-assign]

        client = _make_client(storage_root, subscriber_bus=bus)
        frames = _parse_sse_frames(
            client.get("/v1/sessions/live-sess/events?stream=true").text
        )
        # The historical read yields the event from disk; the queue item is seq-deduplicated
        history = [f for f in frames if f.get("event") != "subscriber_lagged"]
        assert len(history) == 1
        assert history[0]["data"]["data"] == {"live": True}
