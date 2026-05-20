"""System config reload endpoint: PUT /v1/system/config.

Validates the submitted YAML against the current binary's config schema.  On
validation failure the old config stays in effect, a structured error is written
to the audit log, and a 422 response with per-field error details is returned.
On success the provider registry is hot-swapped and the routing policy updated.

Emits an OTel span ``"system.config.reload"`` and a structured invocation event
on every call.  On failure the span is marked ERROR and ``system.config.reload.failed``
is written to the audit log.  On success ``system.config.reload.ok`` is written.
"""

from __future__ import annotations

from datetime import UTC, datetime

from core_errors import (
    AuditLog,
    AuditLogEntry,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from meridian_sdk_provider import ModelRouter, ModelRoutingPolicy

from ._config import (
    ConfigLoadError,
    ConfigValidateError,
    parse_config,
    validate_config,
)
from ._provider_factory import (
    ProviderFactoryError,
    _build_provider,
    _convert_routing_policy,
    _resolve_auth,
)
from ._secret_ref import SecretRefResolver


def _now() -> str:
    return datetime.now(UTC).isoformat()


def make_system_config_router(
    *,
    audit_log: AuditLog,
    model_router: ModelRouter,
    secret_resolver: SecretRefResolver | None = None,
) -> APIRouter:
    """Return an APIRouter that mounts PUT /v1/system/config.

    Requires *model_router* to have been constructed with a ProviderRegistry
    (i.e. ``model_router.registry`` is not None) to support hot-swap on reload.
    If the registry is absent the endpoint still validates and rejects invalid
    configs, but skips the provider swap.
    """
    router = APIRouter()

    @router.put("/v1/system/config")
    async def reload_config(request: Request) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span("system.config.reload") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="system.config.reload.invocation",
                    code="system_config_reload",
                    timestamp=now,
                ),
            )

            # ── 1. Read raw body ──────────────────────────────────────────────
            try:
                raw_body = (await request.body()).decode("utf-8")
            except Exception as exc:
                ts = _now()
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="system.config.reload.failed",
                        code="config_reload_invalid",
                        timestamp=ts,
                        detail={"stage": "read", "message": str(exc)},
                    )
                )
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "config_reload_invalid",
                            "message": f"Failed to read request body: {exc}",
                            "timestamp": ts,
                            "errors": [],
                        }
                    },
                )

            # ── 2. Parse YAML + Pydantic schema validation ────────────────────
            try:
                config = parse_config(raw_body, audit_log=audit_log)
            except ConfigLoadError as exc:
                record_error(span, exc)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="system.config.reload.failed",
                        code="config_reload_invalid",
                        timestamp=exc.timestamp,
                        detail={"stage": "parse", "message": exc.message},
                    )
                )
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "config_reload_invalid",
                            "message": exc.message,
                            "timestamp": exc.timestamp,
                            "errors": [exc.message],
                        }
                    },
                )

            # ── 3. Semantic validation ────────────────────────────────────────
            try:
                validate_config(config, audit_log=audit_log)
            except ConfigValidateError as exc:
                record_error(span, exc)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="system.config.reload.failed",
                        code="config_reload_invalid",
                        timestamp=exc.timestamp,
                        detail={"stage": "validate", "message": exc.message, "errors": exc.errors},
                    )
                )
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "config_reload_invalid",
                            "message": exc.message,
                            "timestamp": exc.timestamp,
                            "errors": exc.errors,
                        }
                    },
                )

            # ── 4. Hot-swap providers (if registry is available) ──────────────
            registry = model_router.registry
            provider_count = 0

            if registry is not None:
                try:
                    new_providers = {}
                    for provider_cfg in config.providers:
                        resolved_auth = _resolve_auth(provider_cfg, secret_resolver)
                        new_providers[provider_cfg.name] = _build_provider(
                            provider_cfg, resolved_auth
                        )
                    provider_count = len(new_providers)
                    await registry.swap_all(new_providers)
                except ProviderFactoryError as exc:
                    record_error(span, exc)
                    audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="system.config.reload.failed",
                            code=exc.code,
                            timestamp=exc.timestamp,
                            detail={"stage": "provider_build", "message": exc.message},
                        )
                    )
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "code": exc.code,
                                "message": exc.message,
                                "timestamp": exc.timestamp,
                                "errors": [exc.message],
                            }
                        },
                    )

            # ── 5. Update routing policy ──────────────────────────────────────
            new_policy = (
                _convert_routing_policy(config.routing)
                if config.routing is not None
                else ModelRoutingPolicy(rules=[], fallbacks=[])
            )
            model_router.set_policy(new_policy)

            # ── 6. Success ────────────────────────────────────────────────────
            ts = _now()
            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="system.config.reload.ok",
                    code="system_config_reload_ok",
                    timestamp=ts,
                    detail={"provider_count": provider_count},
                )
            )

            return JSONResponse(
                status_code=200,
                content={
                    "reloaded": True,
                    "provider_count": provider_count,
                    "timestamp": ts,
                },
            )

    return router
