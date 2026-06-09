"""In-process inbound sink for long-poll channel drivers.

A long-poll driver (e.g. Telegram getUpdates) decodes a platform update into a
normalized message and hands it to an UpdateSink. AsgiInboundSink delivers that
message into the daemon's own ``POST /v1/channels/{id}/inbound`` route via an
in-process ASGI transport — reusing the real inbound handler (pairing policy,
quarantine, session creation, cross-channel fan-out) with no network hop and no
duplicated logic.

The app is bound after construction (bind) to break the app <-> runtime <->
driver <-> sink construction cycle: the sink is created first, wired into the
drivers, then pointed at the finished app.
"""

from __future__ import annotations

import httpx
from starlette.types import ASGIApp


class AsgiInboundSink:
    """UpdateSink that POSTs decoded updates into the app's inbound route."""

    def __init__(self, base_url: str = "http://meridian.internal") -> None:
        self._app: ASGIApp | None = None
        self._base_url = base_url

    def bind(self, app: ASGIApp) -> None:
        """Point the sink at the finished ASGI app (called once after create_app)."""
        self._app = app

    async def dispatch(
        self, *, channel_id: str, sender_id: str, content: str, content_type: str
    ) -> None:
        if self._app is None:
            return
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url=self._base_url) as client:
            await client.post(
                f"/v1/channels/{channel_id}/inbound",
                json={"sender_id": sender_id, "content": content, "content_type": content_type},
            )
