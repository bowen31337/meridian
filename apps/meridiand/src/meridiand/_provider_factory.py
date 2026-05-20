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
from meridian_sdk_provider import (
    FallbackRule,
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
from meridian_provider_anthropic_apikey import AnthropicApiKeyProvider

from ._config import MeridianConfig, ProviderConfig, RoutingConfig
from ._secret_ref import SecretRefResolver

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _now() -> str:
    return datetime.now(UTC).isoformat()


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

    raise ProviderFactoryError(
        message=f"Unsupported provider kind {kind!r} for provider {name!r}",
        timestamp=_now(),
    )


def _convert_routing_policy(cfg: RoutingConfig) -> ModelRoutingPolicy:
    """Map config-level RoutingConfig to the sdk-provider ModelRoutingPolicy."""
    rules: list[ModelRoutingRule] = []
    for r in cfg.rules:
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

    fallbacks = [FallbackRule(on=f.on, model=f.model) for f in cfg.fallbacks]
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
                    resolved_auth = _resolve_auth(cfg, secret_resolver)
                    provider = _build_provider(cfg, resolved_auth)
                    providers[cfg.name] = provider
                except ProviderFactoryError:
                    raise
                except Exception as exc:
                    raise ProviderFactoryError(
                        message=(
                            f"Failed to build provider {cfg.name!r} "
                            f"(kind={cfg.kind!r}): {exc}"
                        ),
                        timestamp=_now(),
                        cause=exc,
                    ) from exc

            return ProviderRegistry(providers=providers, audit_log=_audit)

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
    return ModelRouter(policy=policy, registry=registry, audit_log=audit_log)
