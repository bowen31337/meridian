from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from core_errors import AuditLog, HandlerOptions, install_error_handler
from fastapi import FastAPI
from meridian_plugin_loader import PluginLoader

_LOG = logging.getLogger("meridiand")


def create_app(audit_log: AuditLog, plugin_loader: PluginLoader | None = None) -> FastAPI:
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
    return app
