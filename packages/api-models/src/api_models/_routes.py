from __future__ import annotations

import datetime

from fastapi import APIRouter
from meridian_sdk_provider import ModelRouter

from ._audit import AuditLog, NoopAuditLog
from ._telemetry import get_tracer, record_failure, record_list_event
from ._types import AuditLogEntry, ListModelsResponse, ModelCapabilityFlags, ModelInfo


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def make_router(
    model_router: ModelRouter,
    audit_log: AuditLog | None = None,
) -> APIRouter:
    """Return an APIRouter mounting GET /v1/models.

    Lists all models available across configured providers, including
    provider name, model name, context window, and capability flags
    (streaming, thinking, vision, tools, cache).  Emits an OpenTelemetry
    span and logs a structured event on every invocation; on failure the
    error is surfaced to the caller and written to *audit_log*.
    """
    _audit = audit_log or NoopAuditLog()
    router = APIRouter()

    @router.get("/v1/models", response_model=ListModelsResponse)
    def list_models() -> ListModelsResponse:
        tracer = get_tracer()
        with tracer.start_as_current_span("models.list") as span:
            try:
                entries = model_router.list_models()
                models = [
                    ModelInfo(
                        provider=entry.provider,
                        model=entry.model,
                        context_window=entry.context_window,
                        capabilities=ModelCapabilityFlags(
                            streaming=entry.capabilities.streaming,
                            thinking=entry.capabilities.thinking,
                            vision=entry.capabilities.vision,
                            tools=entry.capabilities.tools,
                            cache=entry.capabilities.cache,
                        ),
                    )
                    for entry in entries
                ]
                record_list_event(span, count=len(models))
                return ListModelsResponse(models=models)
            except Exception as exc:
                now = _now_iso()
                record_failure(span, exc, operation="list")
                _audit.write(
                    AuditLogEntry(
                        level="error",
                        event="models.list.failed",
                        operation="list",
                        timestamp=now,
                        detail={"error_type": type(exc).__name__, "error": str(exc)},
                    )
                )
                raise

    return router
