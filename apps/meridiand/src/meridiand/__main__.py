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
from ._logging import LoggingConfigError, configure_json_logging, emit_early_error
from ._provider_factory import ProviderFactoryError, build_model_router, build_provider_registry
from ._secret_ref import SecretRefResolver
from ._services import init_services
from ._telemetry import get_tracer, record_daemon_failure, record_daemon_start_event
from ._vault_backend_os_keychain import OsKeychainVaultBackend

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
            emit_early_error("meridiand", f"config location error: {exc}")
            return 1

    try:
        config = load_config(config_path)
    except Exception as exc:
        emit_early_error("meridiand", f"config error: {exc}")
        return 1

    try:
        services = init_services(config)
    except Exception as exc:
        emit_early_error("meridiand", f"service init error: {exc}")
        return 1

    try:
        validate_config(config, audit_log=services.audit_log)
    except Exception as exc:
        emit_early_error("meridiand", f"config error: {exc}")
        return 1

    try:
        configure_json_logging(config.log_level, audit_log=services.audit_log)
    except LoggingConfigError as exc:
        emit_early_error("meridiand", f"logging config error: {exc.message}")
        return 1

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
        _LOG.error("migration failed: %s", exc.message)
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
        _LOG.error("migration error: %s", exc)
        return 1

    # Build the provider registry and model router when providers are configured.
    model_router = None
    secret_resolver: SecretRefResolver | None = None
    if config.providers:
        os_keychain = OsKeychainVaultBackend()
        secret_resolver = SecretRefResolver(
            storage_root=config.storage_root,
            os_keychain_backend=os_keychain,
            audit_log=services.audit_log,
        )
        try:
            registry = build_provider_registry(
                config,
                secret_resolver=secret_resolver,
                audit_log=services.audit_log,
            )
            model_router = build_model_router(config, registry, audit_log=services.audit_log)
        except ProviderFactoryError as exc:
            with contextlib.suppress(Exception):
                services.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="meridiand.provider_init_failed",
                        code=exc.code,
                        timestamp=exc.timestamp,
                        detail={"message": exc.message},
                    )
                )
            _LOG.error("provider init error: %s", exc.message)
            return 1

    serve_ui = config.daemon.serve_ui if config.daemon is not None else False
    ui_dist_path = Path(__file__).parent / "ui" if serve_ui else None

    app = create_app(
        services.audit_log,
        plugin_loader=services.plugin_loader,
        storage_root=config.storage_root,
        config_path=config_path,
        event_log=services.event_log,
        cors=config.cors,
        auth_config=config.auth,
        model_router=model_router,
        secret_resolver=secret_resolver,
        serve_ui=serve_ui,
        ui_dist_path=ui_dist_path,
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
            uvicorn.run(app, log_level=config.log_level.lower(), log_config=None, **server_kwargs)  # type: ignore[arg-type]
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
            _LOG.error("server error: %s", msg)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
