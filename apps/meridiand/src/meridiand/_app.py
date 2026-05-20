from __future__ import annotations

import asyncio
import contextlib
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
from storage_event_log import EventLogWriter, SubscriberBus

from meridian_sdk_provider import ModelRouter
from sdk_channel import ChannelRuntime

from ._acp import AcpInboundHandler, AcpPeerClient, make_acp_router
from ._acp_compliance import make_acp_compliance_router
from ._cancel import make_cancel_router
from ._checkpoint import make_checkpoint_router
from ._ci_regression import make_ci_regression_router
from ._crash_recovery_soak import make_crash_recovery_soak_router
from ._e8_hardening_soak import make_e8_hardening_soak_router
from ._skill_forge_soak import make_skill_forge_soak_router
from ._vault_leak_soak import make_vault_leak_soak_router
from ._compaction import make_compaction_router, run_compaction_loop
from ._config import AuthConfig, CompactionConfig, CronSchedulerConfig, CorsConfig, SkillForgeConfig, WebhookSenderConfig
from ._secret_ref import SecretRefResolver
from ._sighup import install_sighup_handler, remove_sighup_handler
from ._system_config import make_system_config_router
from ._cron import make_cron_router
from ._cron_scheduler import run_cron_scheduler_loop
from ._environments import make_environments_router
from ._memory_stores import make_memory_stores_router
from ._vault_backend_encrypted_file import EncryptedFileVaultBackend
from ._vault_backend_os_keychain import OsKeychainVaultBackend
from ._credential_proxy import CredentialProxyProviderConfig, make_credential_proxy_router
from ._vaults import make_vaults_router
from ._webhook_sender import run_webhook_sender_loop
from ._skill_forge import run_skill_forge_loop
from ._agents import make_agents_router
from ._channels import make_channels_router
from ._system_channel import make_system_channel_router
from ._webhook_channel_driver import SecretResolver
from ._skill_activations import make_skill_activations_router
from ._skill_forge_proposals import make_skill_forge_proposals_router
from ._skill_suggestions import make_skill_suggestions_router
from ._skills import make_skills_router
from ._user_profiles import make_user_profiles_router
from ._hooks import make_hooks_router
from ._webhooks import make_webhooks_router
from ._events import make_events_router
from ._files import make_files_router
from ._handoff import make_handoff_router
from ._kb import make_kb_router
from ._messages import make_messages_router
from ._model_call_event_log import EventLogModelCallAdapter
from api_models import make_router as make_models_router
from ._parallel_runs import make_parallel_runs_router
from ._phase import make_phase_router
from ._replay import make_replay_router
from ._resume import make_resume_router
from ._sessions import make_sessions_router
from ._wake import make_wake_router
from ._auth_middleware import AuthMiddleware
from ._cursor_middleware import CursorPaginationMiddleware
from ._error_envelope_middleware import ErrorEnvelopeMiddleware
from ._idempotency_middleware import IdempotencyKeyMiddleware
from ._system_audit_middleware import SystemAuditMiddleware
from ._openapi_export import make_openapi_export_router
from ._spawn import make_spawn_router
from ._telemetry import get_tracer, record_create_event, record_factory_failure

_LOG = logging.getLogger("meridiand")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_app(
    audit_log: AuditLog,
    plugin_loader: PluginLoader | None = None,
    storage_root: Path | None = None,
    config_path: Path | None = None,
    event_log: EventLogWriter | None = None,
    acp_targets: dict[str, str] | None = None,
    acp_peer_client: AcpPeerClient | None = None,
    acp_inbound_handler: AcpInboundHandler | None = None,
    cors: CorsConfig | None = None,
    model_router: ModelRouter | None = None,
    secret_resolver: SecretRefResolver | None = None,
    compaction: CompactionConfig | None = None,
    cron_scheduler: CronSchedulerConfig | None = None,
    webhook_sender: WebhookSenderConfig | None = None,
    skill_forge: SkillForgeConfig | None = None,
    auth_config: AuthConfig | None = None,
    channel_runtime: ChannelRuntime | None = None,
    channel_secret_resolver: SecretResolver | None = None,
    vault_backend: EncryptedFileVaultBackend | None = None,
    os_keychain_backend: OsKeychainVaultBackend | None = None,
    subscriber_bus: SubscriberBus | None = None,
    credential_proxy_providers: list[CredentialProxyProviderConfig] | None = None,
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
    bearer_token = auth_config.bearer_token if auth_config is not None else None
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

                compaction_task: asyncio.Task[None] | None = None
                if (
                    storage_root is not None
                    and compaction is not None
                    and compaction.enabled
                ):
                    compaction_task = asyncio.create_task(
                        run_compaction_loop(storage_root, compaction, audit_log)
                    )

                cron_scheduler_task: asyncio.Task[None] | None = None
                if storage_root is not None:
                    _cron_cfg = cron_scheduler or CronSchedulerConfig()
                    cron_scheduler_task = asyncio.create_task(
                        run_cron_scheduler_loop(
                            storage_root,
                            audit_log,
                            missed_fires_policy=_cron_cfg.missed_fires_policy,
                            check_interval_seconds=_cron_cfg.check_interval_seconds,
                        )
                    )

                webhook_sender_task: asyncio.Task[None] | None = None
                if storage_root is not None:
                    _ws_cfg = webhook_sender or WebhookSenderConfig()
                    webhook_sender_task = asyncio.create_task(
                        run_webhook_sender_loop(
                            storage_root,
                            audit_log,
                            check_interval_seconds=_ws_cfg.check_interval_seconds,
                        )
                    )

                skill_forge_task: asyncio.Task[None] | None = None
                if storage_root is not None:
                    _sf_cfg = skill_forge or SkillForgeConfig()
                    if _sf_cfg.enabled:
                        skill_forge_task = asyncio.create_task(
                            run_skill_forge_loop(
                                storage_root,
                                audit_log,
                                max_invocations_per_minute=_sf_cfg.max_invocations_per_minute,
                                check_interval_seconds=_sf_cfg.check_interval_seconds,
                            )
                        )

                if config_path is not None and model_router is not None:
                    install_sighup_handler(
                        config_path=config_path,
                        model_router=model_router,
                        audit_log=audit_log,
                        secret_resolver=secret_resolver,
                    )

                _LOG.info("meridiand ready")
                yield

                if config_path is not None and model_router is not None:
                    remove_sighup_handler()

                if compaction_task is not None:
                    compaction_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await compaction_task

                if cron_scheduler_task is not None:
                    cron_scheduler_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await cron_scheduler_task

                if webhook_sender_task is not None:
                    webhook_sender_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await webhook_sender_task

                if skill_forge_task is not None:
                    skill_forge_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await skill_forge_task

            app = FastAPI(title="meridiand", lifespan=_lifespan)

            # GZip is always enabled; minimum_size=1000 avoids compressing tiny payloads.
            app.add_middleware(IdempotencyKeyMiddleware, audit_log=audit_log)
            app.add_middleware(SystemAuditMiddleware, audit_log=audit_log)
            app.add_middleware(GZipMiddleware, minimum_size=1000)
            app.add_middleware(CursorPaginationMiddleware, audit_log=audit_log)
            if cors_enabled:
                assert cors is not None  # narrowed above
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=cors.allow_origins,
                    allow_methods=cors.allow_methods,
                    allow_headers=cors.allow_headers,
                    allow_credentials=cors.allow_credentials,
                )
            _hooks_dir = storage_root / "hooks" if storage_root is not None else None
            app.add_middleware(AuthMiddleware, audit_log=audit_log, bearer_token=bearer_token)
            app.add_middleware(ErrorEnvelopeMiddleware, audit_log=audit_log, hooks_dir=_hooks_dir)

            install_error_handler(app, HandlerOptions(audit_log=audit_log))

            app.include_router(make_openapi_export_router(audit_log=audit_log))

            if storage_root is not None:
                app.include_router(
                    make_files_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_events_router(
                        audit_log=audit_log,
                        storage_root=storage_root,
                        subscriber_bus=subscriber_bus,
                    )
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
                    make_wake_router(audit_log=audit_log, storage_root=storage_root)
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
                app.include_router(
                    make_crash_recovery_soak_router(
                        audit_log=audit_log, storage_root=storage_root
                    )
                )
                app.include_router(
                    make_skill_forge_soak_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_vault_leak_soak_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_e8_hardening_soak_router(
                        audit_log=audit_log, storage_root=storage_root
                    )
                )
                app.include_router(
                    make_cron_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_hooks_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_webhooks_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_skills_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_skill_activations_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_skill_forge_proposals_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_skill_suggestions_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_agents_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_channels_router(audit_log=audit_log, storage_root=storage_root)
                )
                if channel_runtime is not None:
                    app.include_router(
                        make_system_channel_router(
                            audit_log=audit_log,
                            storage_root=storage_root,
                            channel_runtime=channel_runtime,
                            secret_resolver=channel_secret_resolver,
                        )
                    )
                app.include_router(
                    make_user_profiles_router(audit_log=audit_log, storage_root=storage_root)
                )
                app.include_router(
                    make_memory_stores_router(
                        audit_log=audit_log,
                        storage_root=storage_root,
                        model_router=model_router,
                    )
                )
                app.include_router(
                    make_vaults_router(
                        audit_log=audit_log,
                        storage_root=storage_root,
                        vault_backend=vault_backend,
                        os_keychain_backend=os_keychain_backend,
                    )
                )
                app.include_router(
                    make_environments_router(audit_log=audit_log, storage_root=storage_root)
                )
                if credential_proxy_providers and secret_resolver is not None:
                    app.include_router(
                        make_credential_proxy_router(
                            audit_log=audit_log,
                            secret_resolver=secret_resolver,
                            providers=credential_proxy_providers,
                        )
                    )
                if compaction is not None:
                    app.include_router(
                        make_compaction_router(
                            audit_log=audit_log,
                            storage_root=storage_root,
                            policy=compaction,
                        )
                    )
                if event_log is not None:
                    app.include_router(
                        make_sessions_router(
                            audit_log=audit_log,
                            storage_root=storage_root,
                            event_log=event_log,
                        )
                    )
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
                        inbound_handler=acp_inbound_handler,
                    )
                )
                app.include_router(
                    make_acp_compliance_router(audit_log=audit_log)
                )
            if model_router is not None:
                if event_log is not None:
                    from storage_event_log import EventLogRuntime
                    model_router.set_event_log(
                        EventLogModelCallAdapter(EventLogRuntime(event_log))
                    )
                app.include_router(
                    make_messages_router(
                        audit_log=audit_log,
                        model_router=model_router,
                        hooks_dir=_hooks_dir,
                    )
                )
                app.include_router(
                    make_models_router(
                        model_router=model_router,
                        audit_log=audit_log,
                    )
                )
                app.include_router(
                    make_system_config_router(
                        audit_log=audit_log,
                        model_router=model_router,
                        secret_resolver=secret_resolver,
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
