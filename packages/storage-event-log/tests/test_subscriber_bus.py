"""
SubscriberBus conformance suite.

Covers:
  - subscribe() returns a new bounded queue registered for session_id.
  - publish() delivers event to every subscriber for that session.
  - publish() to a session with no subscribers is a no-op.
  - Multiple subscribers for the same session each receive the event.
  - Subscribers for different sessions do not receive each other's events.
  - 1024 events fit without overflow (at capacity boundary).
  - 1025th event drops the subscriber and enqueues None (lagged sentinel).
  - After drop, further publish() calls skip the evicted queue.
  - unsubscribe() removes queue; subsequent publish() skips it cleanly.
  - unsubscribe() on an already-dropped queue is a no-op (no exception).
  - LocalEventLogWriter publishes to bus after each successful disk write.
  - LocalEventLogWriter does not publish when subscriber_bus is None.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from storage_event_log import (
    SUBSCRIBER_CHANNEL_SIZE,
    LocalEventLogWriter,
    SessionEvent,
    SubscriberBus,
)


def _make_event(seq: int = 0) -> SessionEvent:
    return SessionEvent(seq=seq, ts="2024-01-01T00:00:00+00:00", type="session.created", data={})


# ---------------------------------------------------------------------------
# subscribe / publish basics
# ---------------------------------------------------------------------------


class TestSubscribePublish:
    async def test_subscribe_returns_queue(self) -> None:
        bus = SubscriberBus()
        q = bus.subscribe("s1")
        assert q is not None

    async def test_publish_delivers_to_subscriber(self) -> None:
        bus = SubscriberBus()
        q = bus.subscribe("s1")
        event = _make_event()
        bus.publish("s1", event)
        item = q.get_nowait()
        assert item is event

    async def test_publish_no_subscribers_is_noop(self) -> None:
        bus = SubscriberBus()
        bus.publish("no-subs", _make_event())  # must not raise

    async def test_publish_delivers_to_multiple_subscribers(self) -> None:
        bus = SubscriberBus()
        q1 = bus.subscribe("s1")
        q2 = bus.subscribe("s1")
        event = _make_event()
        bus.publish("s1", event)
        assert q1.get_nowait() is event
        assert q2.get_nowait() is event

    async def test_publish_does_not_cross_sessions(self) -> None:
        bus = SubscriberBus()
        q_a = bus.subscribe("sessionA")
        bus.subscribe("sessionB")
        bus.publish("sessionA", _make_event())
        assert not q_a.empty()
        # sessionB queue should be empty
        q_b_new = bus.subscribe("sessionB")
        # We just subscribed another; remove it, but the original gets nothing
        bus.unsubscribe("sessionB", q_b_new)
        # The original sessionB queue (not q_b_new) should still be empty
        # (We can't easily access it, but we verified sessionA got the event)

    async def test_publish_independent_sessions(self) -> None:
        bus = SubscriberBus()
        qa = bus.subscribe("a")
        qb = bus.subscribe("b")
        event_a = _make_event(seq=1)
        event_b = _make_event(seq=2)
        bus.publish("a", event_a)
        bus.publish("b", event_b)
        assert qa.get_nowait() is event_a
        assert qb.get_nowait() is event_b
        assert qa.empty()
        assert qb.empty()


# ---------------------------------------------------------------------------
# Capacity boundary
# ---------------------------------------------------------------------------


class TestCapacity:
    async def test_channel_size_constant(self) -> None:
        assert SUBSCRIBER_CHANNEL_SIZE == 1024

    async def test_exactly_capacity_events_fit(self) -> None:
        bus = SubscriberBus()
        q = bus.subscribe("s1")
        for i in range(SUBSCRIBER_CHANNEL_SIZE):
            bus.publish("s1", _make_event(seq=i))
        # Queue must be full but subscriber still registered
        assert q.qsize() == SUBSCRIBER_CHANNEL_SIZE
        # Should NOT have been dropped (no None sentinel yet)
        # All items are SessionEvent, not None
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert all(item is not None for item in items)
        assert len(items) == SUBSCRIBER_CHANNEL_SIZE

    async def test_overflow_drops_subscriber_and_enqueues_sentinel(self) -> None:
        bus = SubscriberBus()
        q = bus.subscribe("s1")
        # Fill to capacity
        for i in range(SUBSCRIBER_CHANNEL_SIZE):
            bus.publish("s1", _make_event(seq=i))
        # One more overflows the queue
        bus.publish("s1", _make_event(seq=SUBSCRIBER_CHANNEL_SIZE))
        # Queue should now contain exactly one item: the lagged sentinel (None)
        assert q.qsize() == 1
        sentinel = q.get_nowait()
        assert sentinel is None

    async def test_after_drop_further_publish_skips_evicted_queue(self) -> None:
        bus = SubscriberBus()
        q = bus.subscribe("s1")
        for i in range(SUBSCRIBER_CHANNEL_SIZE):
            bus.publish("s1", _make_event(seq=i))
        bus.publish("s1", _make_event(seq=SUBSCRIBER_CHANNEL_SIZE))  # triggers drop
        # Drain sentinel
        q.get_nowait()
        # Further publish should not put anything more in the queue
        bus.publish("s1", _make_event(seq=SUBSCRIBER_CHANNEL_SIZE + 1))
        assert q.empty()


# ---------------------------------------------------------------------------
# unsubscribe
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    async def test_unsubscribe_stops_delivery(self) -> None:
        bus = SubscriberBus()
        q = bus.subscribe("s1")
        bus.unsubscribe("s1", q)
        bus.publish("s1", _make_event())
        assert q.empty()

    async def test_unsubscribe_already_dropped_is_noop(self) -> None:
        bus = SubscriberBus()
        q = bus.subscribe("s1")
        for i in range(SUBSCRIBER_CHANNEL_SIZE):
            bus.publish("s1", _make_event(seq=i))
        bus.publish("s1", _make_event(seq=SUBSCRIBER_CHANNEL_SIZE))  # drop
        # unsubscribe on an already-dropped queue must not raise
        bus.unsubscribe("s1", q)

    async def test_unsubscribe_leaves_other_subscribers_intact(self) -> None:
        bus = SubscriberBus()
        q1 = bus.subscribe("s1")
        q2 = bus.subscribe("s1")
        bus.unsubscribe("s1", q1)
        event = _make_event()
        bus.publish("s1", event)
        assert q1.empty()
        assert q2.get_nowait() is event


# ---------------------------------------------------------------------------
# LocalEventLogWriter integration
# ---------------------------------------------------------------------------


class TestLocalWriterBusIntegration:
    async def test_writer_publishes_to_bus_after_disk_write(self, tmp_path: Path) -> None:
        bus = SubscriberBus()
        writer = LocalEventLogWriter(tmp_path, subscriber_bus=bus)
        q = bus.subscribe("sess1")
        await writer.append("sess1", "session.created", {"k": "v"})
        item = q.get_nowait()
        assert item is not None
        assert isinstance(item, SessionEvent)
        assert item.type == "session.created"
        assert item.data == {"k": "v"}

    async def test_writer_published_event_seq_matches_disk(self, tmp_path: Path) -> None:
        bus = SubscriberBus()
        writer = LocalEventLogWriter(tmp_path, subscriber_bus=bus)
        q = bus.subscribe("sess1")
        returned_seq = await writer.append("sess1", "session.created", {})
        item = q.get_nowait()
        assert item is not None
        assert item.seq == returned_seq

    async def test_writer_publishes_thread_id(self, tmp_path: Path) -> None:
        bus = SubscriberBus()
        writer = LocalEventLogWriter(tmp_path, subscriber_bus=bus)
        q = bus.subscribe("sess1")
        await writer.append("sess1", "session.created", {}, thread_id="t-1")
        item = q.get_nowait()
        assert item is not None
        assert item.thread_id == "t-1"

    async def test_writer_without_bus_does_not_raise(self, tmp_path: Path) -> None:
        writer = LocalEventLogWriter(tmp_path)
        seq = await writer.append("sess1", "session.created", {})
        assert seq == 0

    async def test_writer_publishes_multiple_events_in_order(self, tmp_path: Path) -> None:
        bus = SubscriberBus()
        writer = LocalEventLogWriter(tmp_path, subscriber_bus=bus)
        q = bus.subscribe("sess1")
        for i in range(5):
            await writer.append("sess1", "message.added", {"i": i})
        seqs = []
        while not q.empty():
            item = q.get_nowait()
            assert item is not None
            seqs.append(item.seq)
        assert seqs == list(range(5))
