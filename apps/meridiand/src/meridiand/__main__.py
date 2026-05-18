from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from core_errors import AuditLogEntry

from ._app import create_app
from ._config import DEFAULT_CONFIG_PATH, load_config
from ._services import init_services

_LOG = logging.getLogger("meridiand")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meridiand", description="Meridian daemon")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args(argv)
    config_path: Path = args.config or DEFAULT_CONFIG_PATH

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"meridiand: config error: {exc}", file=sys.stderr)
        return 1

    try:
        services = init_services(config)
    except Exception as exc:
        print(f"meridiand: service init error: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    app = create_app(
        services.audit_log,
        plugin_loader=services.plugin_loader,
        storage_root=config.storage_root,
        event_log=services.event_log,
        cors=config.cors,
    )

    bind = config.bind
    if bind.socket:
        server_kwargs: dict[str, object] = {
            "uds": bind.socket,
        }
    else:
        server_kwargs = {
            "host": bind.host,
            "port": bind.port,
        }

    try:
        uvicorn.run(app, log_level=config.log_level.lower(), **server_kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        msg = str(exc)
        with contextlib.suppress(Exception):
            services.audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="meridiand.startup_failed",
                    code="startup_failed",
                    timestamp=_now(),
                    detail={"message": msg},
                )
            )
        print(f"meridiand: server error: {msg}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
