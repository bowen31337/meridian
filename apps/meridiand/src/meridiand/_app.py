from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from core_errors import AuditLog, HandlerOptions, install_error_handler
from fastapi import FastAPI

_LOG = logging.getLogger("meridiand")


def create_app(audit_log: AuditLog) -> FastAPI:
    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        _LOG.info("meridiand ready")
        yield

    app = FastAPI(title="meridiand", lifespan=_lifespan)
    install_error_handler(app, HandlerOptions(audit_log=audit_log))
    return app
