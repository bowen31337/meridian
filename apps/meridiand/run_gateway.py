"""Dev launcher: boot meridiand with the Telegram channel driver wired into
the gateway (ChannelRuntime), so POST /v1/channels/{id}/outbound delivers via
the Telegram Bot API. The stock __main__ does not register any channel runtime.

Bot token is read from the TELEGRAM_BOT_TOKEN env var (any bot_token_ref in a
channel config resolves to it). Bind + storage are local-scratch.
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from meridiand._app import create_app
from meridiand._config import BindConfig, MeridianConfig
from meridiand._services import init_services
from meridiand._telegram_channel_driver import TelegramChannelDriver
from sdk_channel import ChannelRuntime

STORAGE_ROOT = Path(__file__).parent / ".gateway-scratch"


class EnvSecretResolver:
    """Resolve any bot_token_ref to TELEGRAM_BOT_TOKEN from the environment."""

    def resolve(self, secret_ref: str) -> str | None:
        return os.environ.get("TELEGRAM_BOT_TOKEN")


def build() -> object:
    config = MeridianConfig(
        storage_root=STORAGE_ROOT,
        bind=BindConfig(host="127.0.0.1", port=8888, socket=None),
    )
    services = init_services(config)

    runtime = ChannelRuntime()
    runtime.register(
        TelegramChannelDriver(
            storage_root=config.storage_root,
            secret_resolver=EnvSecretResolver(),
            audit_log=services.audit_log,
        )
    )

    return create_app(
        services.audit_log,
        plugin_loader=services.plugin_loader,
        storage_root=config.storage_root,
        event_log=services.event_log,
        channel_runtime=runtime,
    )


if __name__ == "__main__":
    uvicorn.run(build(), host="127.0.0.1", port=8888, log_level="info", log_config=None)
