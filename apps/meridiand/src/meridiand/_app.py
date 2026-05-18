from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from core_errors import AuditLog, HandlerOptions, install_error_handler
from fastapi import FastAPI
from meridian_plugin_loader import PluginLoader
from storage_event_log import EventLogWriter

from ._acp import AcpPeerClient, make_acp_router
from ._cancel import make_cancel_router
from ._checkpoint import make_checkpoint_router
from ._ci_regression import make_ci_regression_router
from ._handoff import make_handoff_router
from ._kb import make_kb_router
from ._parallel_runs import make_parallel_runs_router
from ._phase import make_phase_router
from ._replay import make_replay_router
from ._resume import make_resume_router
from ._spawn import make_spawn_router

_LOG = logging.getLogger("meridiand")


def create_app(
    audit_log: AuditLog,
    plugin_loader: PluginLoader | None = None,
    storage_root: Path | None = None,
    event_log: EventLogWriter | None = None,
    acp_targets: dict[str, str] | None = None,
    acp_peer_client: AcpPeerClient | None = None,
) -> FastAPI:
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
    install_error_handler(app, HandlerOptions(audit_log=audit_log))
    if storage_root is not None:
        app.include_router(make_replay_router(audit_log=audit_log, storage_root=storage_root))
        app.include_router(make_checkpoint_router(audit_log=audit_log, storage_root=storage_root))
        app.include_router(make_resume_router(audit_log=audit_log, storage_root=storage_root))
        app.include_router(make_kb_router(audit_log=audit_log, storage_root=storage_root))
        app.include_router(make_spawn_router(audit_log=audit_log, storage_root=storage_root))
        app.include_router(make_handoff_router(audit_log=audit_log, storage_root=storage_root))
        app.include_router(make_cancel_router(audit_log=audit_log, storage_root=storage_root))
        app.include_router(
            make_parallel_runs_router(audit_log=audit_log, storage_root=storage_root)
        )
        app.include_router(
            make_ci_regression_router(audit_log=audit_log, storage_root=storage_root)
        )
        if event_log is not None:
            app.include_router(
                make_phase_router(
                    audit_log=audit_log, storage_root=storage_root, event_log=event_log
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
    return app
