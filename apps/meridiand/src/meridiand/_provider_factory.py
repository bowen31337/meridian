"""Provider factory: instantiate ModelProvider objects from MeridianConfig.

Reads the ``providers`` and ``routing`` sections of the config, resolves
auth secrets, and returns a ready-to-use ``ProviderRegistry`` and
``ModelRouter``.

Emits an OTel span and a structured event on each build_provider_registry()
invocation.  On failure the span is marked ERROR, an audit log entry is
written, and the exception is re-raised to the caller.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    NoopAuditLog,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from meridian_provider_anthropic_apikey import AnthropicApiKeyProvider
from meridian_provider_claude_code_oauth import SystemOAuthProvider
from meridian_sdk_provider import (
    FallbackRule,
    LoadBalancedProvider,
    ModelProvider,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ProviderRegistry,
    RoutingCondition,
    TokenRange,
)
from meridian_sdk_provider.audit import AuditLog as SdkAuditLog, AuditLogEntry as SdkAuditLogEntry

from ._config import MeridianConfig, ProviderConfig, RoutingConfig
from ._secret_ref import SecretRefResolver

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _SdkProviderAuditBridge:
    """Adapt meridian_sdk_provider AuditLogEntry writes to a core_errors AuditLog.

    The ModelRouter and ProviderRegistry emit ``meridian_sdk_provider.audit``
    entries (provider_name/provider_kind/model, no ``code``); core_errors audit
    sinks expect a ``code`` field. Without this bridge a failure-path write
    raises ``'AuditLogEntry' object has no attribute 'code'`` and masks the real
    provider error.
    """

    def __init__(self, core_log: AuditLog) -> None:
        self._core = core_log

    def write(self, entry: SdkAuditLogEntry) -> None:
        detail: dict[str, Any] = dict(entry.detail or {})
        detail.setdefault("provider_name", entry.provider_name)
        detail.setdefault("provider_kind", entry.provider_kind)
        if entry.model is not None:
            detail.setdefault("model", entry.model)
        if entry.session_id is not None:
            detail.setdefault("session_id", entry.session_id)
        level = "warn" if entry.level == "warning" else entry.level
        code = str(detail.get("error_type") or entry.event.replace(".", "_"))
        self._core.write(
            AuditLogEntry(
                level=level,
                event=entry.event,
                code=code,
                timestamp=entry.timestamp,
                detail=detail or None,
            )
        )


def _bridge_audit(audit_log: AuditLog | None) -> SdkAuditLog | None:
    """Wrap a core_errors AuditLog so sdk-provider components can write to it."""
    return _SdkProviderAuditBridge(audit_log) if audit_log is not None else None


class ProviderFactoryError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="provider_factory_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )


def _resolve_auth(cfg: ProviderConfig, resolver: SecretRefResolver | None) -> str | None:
    if cfg.auth is None:
        return None
    if cfg.auth.startswith("secret_ref://") and resolver is not None:
        return resolver.resolve(cfg.auth)
    return cfg.auth  # plaintext (already warned in validate_config)


def _build_pool(cfg: ProviderConfig, resolver: SecretRefResolver | None) -> ModelProvider:
    """Build a round-robin LoadBalancedProvider — one member per auth_pool token.

    Each member shares the pool's kind/base_url but authenticates with its own
    resolved secret, so requests are spread across the keys and a throttled key
    fails over to the next within a single call.
    """
    members: list[ModelProvider] = []
    for i, ref in enumerate(cfg.auth_pool or []):
        member_cfg = cfg.model_copy(
            update={"name": f"{cfg.name}#{i}", "auth": ref, "auth_pool": None}
        )
        members.append(_build_provider(member_cfg, _resolve_auth(member_cfg, resolver)))
    return LoadBalancedProvider(name=cfg.name, kind=cfg.kind, members=members)


def _build_provider(cfg: ProviderConfig, resolved_auth: str | None) -> ModelProvider:
    """Instantiate a single ModelProvider from ProviderConfig + resolved auth."""
    kind = cfg.kind
    name = cfg.name
    base_url = cfg.base_url

    if kind == "anthropic":
        kwargs: dict[str, Any] = {"name": name}
        if base_url:
            kwargs["base_url"] = base_url
        return AnthropicApiKeyProvider(resolved_auth or "", **kwargs)

    if kind == "openai":
        kwargs = {"name": name}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIProvider(resolved_auth or "", **kwargs)

    if kind == "openrouter":
        kwargs = {"name": name}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenRouterProvider(resolved_auth or "", **kwargs)

    if kind in ("ollama", "local"):
        effective_url = base_url or _DEFAULT_OLLAMA_BASE_URL
        return OllamaProvider(effective_url, name=name)

    if kind == "claude_code_oauth":
        kwargs_oauth: dict[str, Any] = {"name": name}
        if base_url:
            kwargs_oauth["cli_path"] = base_url  # base_url repurposed as cli_path override
        return SystemOAuthProvider(**kwargs_oauth)

    raise ProviderFactoryError(
        message=f"Unsupported provider kind {kind!r} for provider {name!r}",
        timestamp=_now(),
    )


def _convert_routing_policy(cfg: RoutingConfig) -> ModelRoutingPolicy:
    """Map config-level RoutingConfig to the sdk-provider ModelRoutingPolicy."""
    if cfg.default is None:
        return ModelRoutingPolicy(rules=[], fallbacks=[])

    rules: list[ModelRoutingRule] = []
    for r in cfg.default.rules:
        when: RoutingCondition | None = None
        if r.when is not None:
            tr: TokenRange | None = None
            if r.when.estimated_input_tokens is not None:
                t = r.when.estimated_input_tokens
                tr = TokenRange(gt=t.gt, gte=t.gte, lt=t.lt, lte=t.lte)
            when = RoutingCondition(
                skill_id=r.when.skill_id,
                estimated_input_tokens=tr,
                metadata_match=dict(r.when.metadata_match) if r.when.metadata_match else None,
                role=r.when.role,
            )
        rules.append(ModelRoutingRule(when=when, model=r.model))

    fallbacks = [FallbackRule(on=f.on, model=f.model) for f in cfg.default.fallbacks]
    return ModelRoutingPolicy(rules=rules, fallbacks=fallbacks)


def build_provider_registry(
    config: MeridianConfig,
    secret_resolver: SecretRefResolver | None = None,
    audit_log: AuditLog | None = None,
) -> ProviderRegistry:
    """Build a ProviderRegistry from the providers section of MeridianConfig.

    Resolves auth secrets via *secret_resolver* when provided.  Emits an OTel
    span and a structured event on invocation.  On failure writes to the audit
    log and re-raises.
    """
    _audit = audit_log or NoopAuditLog()
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "provider_factory.build_registry",
        attributes={"provider_count": len(config.providers)},
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="provider_factory.build_registry.invocation",
                code="provider_factory_build_registry",
                timestamp=now,
            ),
        )
        try:
            providers: dict[str, ModelProvider] = {}
            for cfg in config.providers:
                try:
                    if cfg.auth_pool:
                        provider = _build_pool(cfg, secret_resolver)
                    else:
                        provider = _build_provider(cfg, _resolve_auth(cfg, secret_resolver))
                    providers[cfg.name] = provider
                except ProviderFactoryError:
                    raise
                except Exception as exc:
                    raise ProviderFactoryError(
                        message=(
                            f"Failed to build provider {cfg.name!r} (kind={cfg.kind!r}): {exc}"
                        ),
                        timestamp=_now(),
                        cause=exc,
                    ) from exc

            return ProviderRegistry(providers=providers, audit_log=_bridge_audit(_audit))

        except ProviderFactoryError as err:
            record_error(span, err)
            _audit.write(
                AuditLogEntry(
                    level="error",
                    event="provider_factory.build_registry.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={"message": err.message},
                )
            )
            raise
        except Exception as exc:
            err2 = ProviderFactoryError(
                message=f"Failed to build provider registry: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err2)
            _audit.write(
                AuditLogEntry(
                    level="error",
                    event="provider_factory.build_registry.failed",
                    code=err2.code,
                    timestamp=err2.timestamp,
                    detail={"message": str(exc)},
                )
            )
            raise err2 from exc


def build_model_router(
    config: MeridianConfig,
    registry: ProviderRegistry,
    audit_log: AuditLog | None = None,
) -> ModelRouter:
    """Build a ModelRouter from the routing section of MeridianConfig + the registry."""
    policy = (
        _convert_routing_policy(config.routing)
        if config.routing is not None
        else ModelRoutingPolicy(rules=[], fallbacks=[])
    )
    return ModelRouter(policy=policy, registry=registry, audit_log=_bridge_audit(audit_log))
