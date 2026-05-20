from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from core_errors import AuditLogEntry, StructuredEvent, record_invocation_event
from storage_repository import RepositoryFailure, SqliteRepositoryDriver

from ._app import create_app
from ._config import load_config, resolve_config_location, validate_config
from ._services import init_services
from ._telemetry import get_tracer, record_daemon_failure, record_daemon_start_event

_LOG = logging.getLogger("meridiand")


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def _run_db_migrations(db_path: Path) -> None:
    driver = await SqliteRepositoryDriver.open(db_path)
    try:
        await driver.migrate()
    finally:
        await driver.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meridiand", description="Meridian daemon")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to YAML config file. "
            "When omitted, searched in order: $MERIDIAN_CONFIG, "
            "~/.meridian/config.yml, /etc/meridian/config.yml"
        ),
    )
    args = parser.parse_args(argv)
    config_path: Path | None = args.config
    if config_path is None:
        try:
            config_path = resolve_config_location()
        except Exception as exc:
            print(f"meridiand: config location error: {exc}", file=sys.stderr)
            return 1

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

    try:
        validate_config(config, audit_log=services.audit_log)
    except Exception as exc:
        print(f"meridiand: config error: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        asyncio.run(_run_db_migrations(config.storage_root / "meridian.db"))
    except RepositoryFailure as exc:
        with contextlib.suppress(Exception):
            services.audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="meridiand.migration_failed",
                    code=exc.code,
                    timestamp=exc.timestamp,
                    detail={"message": exc.message},
                )
            )
        print(f"meridiand: migration failed: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:
        with contextlib.suppress(Exception):
            services.audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="meridiand.migration_failed",
                    code="migration_error",
                    timestamp=_now(),
                    detail={"message": str(exc)},
                )
            )
        print(f"meridiand: migration error: {exc}", file=sys.stderr)
        return 1

    app = create_app(
        services.audit_log,
        plugin_loader=services.plugin_loader,
        storage_root=config.storage_root,
        event_log=services.event_log,
        cors=config.cors,
        auth_config=config.auth,
    )

    bind = config.bind
    if bind.socket:
        Path(bind.socket).parent.mkdir(parents=True, exist_ok=True)
        server_kwargs: dict[str, object] = {"uds": bind.socket}
        bind_mode = "socket"
    else:
        server_kwargs = {"host": bind.host, "port": bind.port}
        bind_mode = "tcp"

    tracer = get_tracer()
    with tracer.start_as_current_span(
        "daemon.start",
        attributes={
            "daemon.bind_mode": bind_mode,
            "daemon.socket": bind.socket or "",
            "daemon.host": bind.host,
            "daemon.port": bind.port,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="daemon.start.invocation",
                code="daemon_start",
                timestamp=_now(),
            ),
        )
        record_daemon_start_event(
            span,
            bind_mode=bind_mode,
            bind_socket=bind.socket or "",
            bind_host=bind.host,
            bind_port=bind.port,
        )
        try:
            uvicorn.run(app, log_level=config.log_level.lower(), **server_kwargs)  # type: ignore[arg-type]
        except Exception as exc:
            msg = str(exc)
            record_daemon_failure(span, exc)
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
