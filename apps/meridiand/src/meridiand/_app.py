from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from core_errors import AuditLog, AuditLogEntry, HandlerOptions, install_error_handler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from meridian_plugin_loader import PluginLoader
from storage_event_log import EventLogWriter

from ._acp import AcpPeerClient, make_acp_router
from ._cancel import make_cancel_router
from ._checkpoint import make_checkpoint_router
from ._ci_regression import make_ci_regression_router
from ._config import CorsConfig
from ._events import make_events_router
from ._handoff import make_handoff_router
from ._kb import make_kb_router
from ._parallel_runs import make_parallel_runs_router
from ._phase import make_phase_router
from ._replay import make_replay_router
from ._resume import make_resume_router
from ._spawn import make_spawn_router
from ._telemetry import get_tracer, record_create_event, record_factory_failure

_LOG = logging.getLogger("meridiand")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_app(
    audit_log: AuditLog,
    plugin_loader: PluginLoader | None = None,
    storage_root: Path | None = None,
    event_log: EventLogWriter | None = None,
    acp_targets: dict[str, str] | None = None,
    acp_peer_client: AcpPeerClient | None = None,
    cors: CorsConfig | None = None,
) -> FastAPI:
    """
    Application factory for the meridiand HTTP API.

    Mounts all routes under /v1, with Meridian extensions under /v1/x.
    Always installs GZipMiddleware.  CORSMiddleware is installed only when
    *cors* supplies at least one origin.  Emits an OpenTelemetry span and
    structured event on every invocation; on failure writes an audit log
    entry and re-raises so the caller receives the error.
    """
    cors_enabled = cors is not None and bool(cors.allow_origins)
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "app.factory.create",
        attributes={"cors.enabled": cors_enabled, "gzip.enabled": True},
    ) as span:
        try:
            @asynccontextmanager
            async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
                if plugin_loader is not None:
                    result = plugin_loader.load_all()
                    _LOG.info(
                        "meridiand plugins loaded",
                        extra={
                            "plugin_loader.loaded_count": len(result.manifests),
                            "plugin_loader.error_count": len(result.errors),
                        },
                    )
                    for err in result.errors:
                        _LOG.error(
                            "plugin load failed: %s",
                            err.message,
                            extra={"plugin.name": err.plugin_name, "error.code": err.code},
                        )
                _LOG.info("meridiand ready")
                yield

            app = FastAPI(title="meridiand", lifespan=_lifespan)

            # GZip is always enabled; minimum_size=1000 avoids compressing tiny payloads.
            app.add_middleware(GZipMiddleware, minimum_size=1000)
            if cors_enabled:
                assert cors is not None  # narrowed above
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=cors.allow_origins,
                    allow_methods=cors.allow_methods,
                    allow_headers=cors.allow_headers,
                    allow_credentials=cors.allow_credentials,
                )

            install_error_handler(app, HandlerOptions(audit_log=audit_log))

            if storage_root is not None:
                app.include_router(
                    make_events_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_replay_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_checkpoint_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_resume_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_kb_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_spawn_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_handoff_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_cancel_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_parallel_runs_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_ci_regression_router(audit_log=audit_log, storage_root=storage_root)
                )
                if event_log is not None:
                    app.include_router(
                        make_phase_router(
                            audit_log=audit_log,
                            storage_root=storage_root,
                            event_log=event_log,
                        )
                    )
            if acp_targets is not None:
                app.include_router(
                    make_acp_router(
                        audit_log=audit_log,
                        targets=acp_targets,
                        peer_client=acp_peer_client,
                    )
                )

            record_create_event(
                span,
                cors_enabled=cors_enabled,
                gzip_enabled=True,
                router_count=len(app.routes),
            )
            return app

        except Exception as exc:
            record_factory_failure(span, exc)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="app.factory.create.failed",
                    code="create_app_failed",
                    timestamp=_now(),
                    detail={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )
            raise
