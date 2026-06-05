from __future__ import annotations

import asyncio
import contextlib

from ._types import SessionEvent

SUBSCRIBER_CHANNEL_SIZE = 1024
"""Maximum events buffered per live SSE subscriber before it is dropped."""


class SubscriberBus:
    """
    In-process fan-out bus for session events.

    Each subscriber gets a bounded asyncio.Queue (maxsize=SUBSCRIBER_CHANNEL_SIZE).
    When publish() overflows a subscriber's queue the subscriber is evicted and
    the lagged sentinel (None) is placed at the head of the queue so the consumer
    can detect the drop without blocking.

    All methods are synchronous and non-blocking.  The harness never waits on a
    slow subscriber; if the queue is full the subscriber is silently dropped.

    Thread safety: designed for a single asyncio event loop.  Do not call from
    multiple threads.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue[SessionEvent | None]]] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue[SessionEvent | None]:
        """Register a new bounded queue for session_id and return it."""
        q: asyncio.Queue[SessionEvent | None] = asyncio.Queue(maxsize=SUBSCRIBER_CHANNEL_SIZE)
        self._subs.setdefault(session_id, []).append(q)
        return q

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[SessionEvent | None]) -> None:
        """Remove queue from the registry.  Safe to call even after the queue was dropped."""
        bucket = self._subs.get(session_id)
        if bucket is not None:
            with contextlib.suppress(ValueError):
                bucket.remove(queue)
            if not bucket:
                del self._subs[session_id]

    def publish(self, session_id: str, event: SessionEvent) -> None:
        """Fan-out event to all registered subscribers.  Drops lagged subscribers immediately."""
        for q in list(self._subs.get(session_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self._drop(session_id, q)

    def _drop(self, session_id: str, queue: asyncio.Queue[SessionEvent | None]) -> None:
        """Evict queue from registry and enqueue the lagged sentinel (None)."""
        bucket = self._subs.get(session_id)
        if bucket is not None:
            with contextlib.suppress(ValueError):
                bucket.remove(queue)
            if not bucket:
                del self._subs[session_id]
        # Drain to make room — safe because asyncio is single-threaded and there
        # is no await between the QueueFull detection and this drain.
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(None)  # lagged sentinel
