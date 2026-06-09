"""Dev harness: drives the PROMOTED production TelegramLongPollClient (now in
src/meridiand/_telegram_channel_driver.py) through the real TelegramChannelDriver,
pulling LIVE Telegram updates and dispatching each into the running gateway's
inbound endpoint.

The client implements the LongPollClient protocol, so the driver's own
start()/stop() lifecycle launches and tears it down — this exercises the real
driver + client code path, not a side channel.

Run:  TELEGRAM_BOT_TOKEN=... uv run python run_longpoll.py <channel_id> [seconds]
The daemon (run_gateway.py) must be up so the inbound sink can POST to it.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import Any

import httpx

from meridiand._telegram_channel_driver import TelegramChannelDriver, TelegramLongPollClient
from sdk_channel import StartRequest, StopRequest

_GATEWAY_BASE = "http://127.0.0.1:8888"
_TELEGRAM_KIND = "meridian.telegram"


class GatewayInboundSink:
    """Dispatches each pulled update into the running gateway's inbound route."""

    def __init__(self, base_url: str = _GATEWAY_BASE) -> None:
        self._base = base_url

    async def dispatch(
        self, *, channel_id: str, sender_id: str, content: str, content_type: str
    ) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self._base}/v1/channels/{channel_id}/inbound",
                json={"sender_id": sender_id, "content": content, "content_type": content_type},
            )
        if resp.is_success:
            print(f"    -> gateway inbound OK: session={resp.json().get('session_id')}", flush=True)
        else:
            print(f"    -> gateway inbound FAILED: HTTP {resp.status_code} {resp.text}", flush=True)


class _EnvResolver:
    """Resolve any bot_token_ref to TELEGRAM_BOT_TOKEN from the environment."""

    def resolve(self, secret_ref: str) -> str | None:
        return os.environ.get("TELEGRAM_BOT_TOKEN")


def _log_event(name: str, detail: dict[str, Any]) -> None:
    if name == "update":
        print(
            f"[pulled] update_id={detail['update_id']} "
            f"from={detail['sender_id']} text={detail['text']!r}",
            flush=True,
        )
    else:
        print(f"[{name}] {detail}", flush=True)


async def main(channel_id: str, run_seconds: float) -> None:
    client = TelegramLongPollClient(
        channel_id=channel_id,
        sink=GatewayInboundSink(),
        on_event=_log_event,
    )
    driver = TelegramChannelDriver(
        storage_root=Path(__file__).parent / ".gateway-scratch",
        secret_resolver=_EnvResolver(),
        long_poll_client=client,
    )

    print(f"Starting driver long-poll for channel {channel_id} ...", flush=True)
    await driver.start(
        StartRequest(channel_id=channel_id, channel_kind=_TELEGRAM_KIND, session_id="lp_session")
    )
    print(
        f"Driver poll task running ({driver._poll_tasks.get(channel_id) is not None}). "
        f"Polling live for {run_seconds:.0f}s — send a message to the bot now.",
        flush=True,
    )

    await asyncio.sleep(run_seconds)

    print("Stopping driver (graceful cancel) ...", flush=True)
    await driver.stop(
        StopRequest(channel_id=channel_id, channel_kind=_TELEGRAM_KIND, session_id="lp_session")
    )
    print(f"Stopped. poll task remaining: {channel_id in driver._poll_tasks}", flush=True)


if __name__ == "__main__":
    ch = sys.argv[1] if len(sys.argv) > 1 else ""
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    asyncio.run(main(ch, secs))
