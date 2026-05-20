"""SIGHUP signal handler: reload config from disk.

Invokes the same validate-then-atomic-swap path as PUT /v1/system/config.
On success emits OTel span ``"system.config.reload"`` and writes
``system.config.reload.ok`` to the audit log.  On failure marks the span
ERROR and writes ``system.config.reload.failed`` — the old config stays
in effect and the daemon keeps running.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import UTC, datetime
from pathlib import Path

from core_errors import (
    AuditLog,
    AuditLogEntry,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from meridian_sdk_provider import ModelRouter, ModelRoutingPolicy

from ._config import ConfigLoadError, ConfigValidateError, load_config, validate_config
from ._provider_factory import (
    ProviderFactoryError,
    _build_provider,
    _convert_routing_policy,
    _resolve_auth,
)
from ._secret_ref import SecretRefResolver

_LOG = logging.getLogger("meridiand")


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def _do_reload(
    config_path: Path,
    model_router: ModelRouter,
    audit_log: AuditLog,
    secret_resolver: SecretRefResolver | None,
) -> None:
    """Run the validate-then-atomic-swap reload from the config file on disk."""
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

        # ── 1. Load + parse YAML from disk ────────────────────────────────────
        try:
            config = load_config(config_path, audit_log=audit_log)
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
            _LOG.error("SIGHUP config reload failed (parse): %s", exc.message)
            return

        # ── 2. Semantic validation ────────────────────────────────────────────
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
            _LOG.error("SIGHUP config reload failed (validate): %s", exc.message)
            return

        # ── 3. Hot-swap providers (if registry is available) ──────────────────
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
                _LOG.error("SIGHUP config reload failed (provider_build): %s", exc.message)
                return

        # ── 4. Update routing policy ──────────────────────────────────────────
        new_policy = (
            _convert_routing_policy(config.routing)
            if config.routing is not None
            else ModelRoutingPolicy(rules=[], fallbacks=[])
        )
        model_router.set_policy(new_policy)

        # ── 5. Success ────────────────────────────────────────────────────────
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
        _LOG.info("SIGHUP config reload succeeded; provider_count=%d", provider_count)


def install_sighup_handler(
    *,
    config_path: Path,
    model_router: ModelRouter,
    audit_log: AuditLog,
    secret_resolver: SecretRefResolver | None = None,
) -> None:
    """Register a SIGHUP handler on the running asyncio event loop.

    Must be called from within a running event loop (e.g. inside an
    asynccontextmanager lifespan).  Each SIGHUP schedules _do_reload()
    as a fire-and-forget task; concurrent signals each get their own task.
    """
    loop = asyncio.get_event_loop()

    def _handle() -> None:
        loop.create_task(
            _do_reload(
                config_path=config_path,
                model_router=model_router,
                audit_log=audit_log,
                secret_resolver=secret_resolver,
            )
        )

    loop.add_signal_handler(signal.SIGHUP, _handle)


def remove_sighup_handler() -> None:
    """Unregister the SIGHUP handler installed by install_sighup_handler."""
    loop = asyncio.get_event_loop()
    loop.remove_signal_handler(signal.SIGHUP)
