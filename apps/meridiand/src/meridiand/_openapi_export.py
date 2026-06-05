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
from fastapi import APIRouter, Request
from fastapi.responses import Response
import yaml


def _now() -> str:
    return datetime.now(UTC).isoformat()


class OpenApiExportError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="openapi_export_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


def make_openapi_export_router(
    *,
    audit_log: AuditLog,
    dest_path: Path | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/openapi")
    async def export_openapi(request: Request) -> Response:
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "openapi.export",
            attributes={"openapi.dest": str(dest_path) if dest_path else ""},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="openapi.export.invocation",
                    code="openapi_export",
                    timestamp=now,
                ),
            )
            try:
                spec = request.app.openapi()
                yaml_content = yaml.dump(spec, default_flow_style=False, allow_unicode=True)
                if dest_path is not None:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_text(yaml_content)
                span.add_event(
                    "openapi.export.done",
                    {
                        "openapi.dest": str(dest_path) if dest_path else "",
                        "openapi.bytes": len(yaml_content),
                    },
                )
                return Response(content=yaml_content.encode(), media_type="application/yaml")
            except Exception as exc:
                err = OpenApiExportError(
                    message=f"Failed to export OpenAPI spec: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="openapi.export.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise err from exc

    return router
