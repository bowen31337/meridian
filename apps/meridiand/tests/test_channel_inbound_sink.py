"""
Tests for AsgiInboundSink — the in-process inbound sink for long-poll drivers.

Covers:
  - dispatch() POSTs the decoded message into the app's inbound route via the
    in-process ASGI transport (no network), reusing the real handler.
  - dispatch() is a no-op before the app is bound.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from meridiand._channel_inbound_sink import AsgiInboundSink


class TestAsgiInboundSink:
    async def test_dispatch_posts_to_inbound_route(self) -> None:
        captured: list[dict[str, Any]] = []

        app = FastAPI()

        @app.post("/v1/channels/{channel_id}/inbound")
        async def _inbound(channel_id: str, body: dict[str, Any]) -> dict[str, Any]:
            captured.append({"channel_id": channel_id, **body})
            return {"session_id": "sess_x"}

        sink = AsgiInboundSink()
        sink.bind(app)
        await sink.dispatch(
            channel_id="ch_1", sender_id="42", content="hi", content_type="text/plain"
        )

        assert captured == [
            {"channel_id": "ch_1", "sender_id": "42", "content": "hi", "content_type": "text/plain"}
        ]

    async def test_dispatch_noop_when_unbound(self) -> None:
        sink = AsgiInboundSink()
        # No bind() — must not raise.
        await sink.dispatch(
            channel_id="ch_1", sender_id="42", content="hi", content_type="text/plain"
        )
