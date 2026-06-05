from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import FileResponse, Response


def _now() -> str:
    return datetime.now(UTC).isoformat()


class UiServeError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="ui_serve_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


def make_ui_router(*, audit_log: AuditLog, ui_dist_path: Path) -> APIRouter:
    router = APIRouter()
    _dist = ui_dist_path.resolve()

    async def _handle(rel_path: str) -> Response:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "ui.serve",
            attributes={"ui.path": rel_path or "/"},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="ui.serve.invocation",
                    code="ui_serve",
                    timestamp=now,
                ),
            )
            try:
                clean = rel_path.lstrip("/")
                if clean:
                    candidate = (_dist / clean).resolve()
                    try:
                        candidate.relative_to(_dist)
                    except ValueError:
                        raise UiServeError(
                            message=f"Path traversal denied: {rel_path!r}",
                            timestamp=_now(),
                        ) from None
                    if candidate.is_file():
                        return FileResponse(candidate)

                index = _dist / "index.html"
                if not index.is_file():
                    raise UiServeError(
                        message=f"UI dist not found at {_dist}",
                        timestamp=_now(),
                    )
                return FileResponse(index)

            except Exception as exc:
                err = (
                    exc
                    if isinstance(exc, UiServeError)
                    else UiServeError(
                        message=f"Failed to serve UI: {exc}",
                        timestamp=_now(),
                        cause=exc,
                    )
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="ui.serve.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"path": rel_path, "message": err.message},
                    )
                )
                raise err from exc

    @router.get("/ui")
    async def get_ui_root() -> Response:
        return await _handle("")

    @router.get("/ui/{path:path}")
    async def get_ui_asset(path: str) -> Response:
        return await _handle(path)

    return router
