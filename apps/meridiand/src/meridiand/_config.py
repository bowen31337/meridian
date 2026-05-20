from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import yaml
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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path.home() / ".meridian" / "config.yaml"
DEFAULT_SOCKET_PATH = Path.home() / ".meridian" / "meridiand.sock"
MERIDIAN_CONFIG_VERSION = 2

_MERIDIAN_CONFIG_ENV = "MERIDIAN_CONFIG"
_USER_CONFIG_PATH = Path.home() / ".meridian" / "config.yml"
SYSTEM_CONFIG_PATH = Path("/etc/meridian/config.yml")

_DEFAULT_CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
_DEFAULT_CORS_HEADERS = ["*"]
_VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
_VALID_VAULT_BACKENDS = {"os_keychain", "encrypted_file"}
_VALID_PROVIDER_KINDS = {"anthropic", "openai", "openrouter", "ollama", "local"}
_SECRET_REF_VAULT_PREFIX = "secret_ref://vault/"


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ConfigLoadError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="config_load_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )


class ConfigValidateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="config_validate_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )


class ConfigResolveError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="config_resolve_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class BindConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = 8888
    socket: str | None = str(DEFAULT_SOCKET_PATH)

    @field_validator("socket", mode="before")
    @classmethod
    def _expand_socket(cls, v: object) -> str | None:
        if v is None:
            return None
        return str(Path(str(v)).expanduser())


class CorsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    allow_origins: list[str] = Field(default_factory=list)
    allow_methods: list[str] = Field(default_factory=lambda: list(_DEFAULT_CORS_METHODS))
    allow_headers: list[str] = Field(default_factory=lambda: list(_DEFAULT_CORS_HEADERS))
    allow_credentials: bool = False


class CompactionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    idle_days: int = 30
    summary_strategy: str = "tail"
    tail_events: int = 50
    retention_days: int | None = None


class CronSchedulerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # What to do with fires missed during daemon downtime.
    # "catch_up": fire once per missed interval slot.
    # "skip":     advance schedule past missed slots without firing.
    missed_fires_policy: str = "skip"
    check_interval_seconds: float = 5.0


class WebhookSenderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_interval_seconds: float = 5.0


class SkillForgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    # Maximum model invocations issued per 60-second window.
    max_invocations_per_minute: int = 10
    check_interval_seconds: float = 5.0


class AuditSigningConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Optional in v1; MUST be enabled for multi-user installs in v1.1 (PRD §6.3).
    enabled: bool = False


class AuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bearer_token: str | None = None


class VaultConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    backend: str = "os_keychain"


class TokenRangeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    gt: int | None = None
    gte: int | None = None
    lt: int | None = None
    lte: int | None = None


class RoutingConditionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str | None = None
    estimated_input_tokens: TokenRangeConfig | None = None
    metadata_match: dict[str, Any] | None = None
    role: str | None = None


class RoutingRuleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    when: RoutingConditionConfig | None = None
    model: str  # "provider_name:model_id" form


class FallbackRuleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    on: Literal["rate_limit", "timeout", "5xx", "any"]
    model: str  # "provider_name:model_id" form


class RoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rules: list[RoutingRuleConfig] = Field(default_factory=list)
    fallbacks: list[FallbackRuleConfig] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: str
    mode: str | None = None
    base_url: str | None = None
    auth: str | None = None


class DaemonConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bind: BindConfig = Field(default_factory=BindConfig)
    workspace_root: Path = Field(default_factory=lambda: Path.home() / ".meridian")
    log_level: str = "info"

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _expand_workspace_root(cls, v: object) -> Path:
        return Path(str(v)).expanduser()


class StorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    database: str | None = None
    event_log: str | None = None
    blob_store: str | None = None


class MeridianConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 2
    storage_root: Path
    bind: BindConfig = Field(default_factory=BindConfig)
    log_level: str = "info"
    cors: CorsConfig = Field(default_factory=CorsConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    cron: CronSchedulerConfig = Field(default_factory=CronSchedulerConfig)
    webhook_sender: WebhookSenderConfig = Field(default_factory=WebhookSenderConfig)
    skill_forge: SkillForgeConfig = Field(default_factory=SkillForgeConfig)
    audit_signing: AuditSigningConfig = Field(default_factory=AuditSigningConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    vaults: list[VaultConfig] = Field(default_factory=list)
    providers: list[ProviderConfig] = Field(default_factory=list)
    routing: RoutingConfig | None = None
    daemon: DaemonConfig | None = None
    storage: StorageConfig | None = None

    @field_validator("storage_root", mode="before")
    @classmethod
    def _expand_storage_root(cls, v: object) -> Path:
        return Path(str(v)).expanduser()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(path: Path, audit_log: AuditLog | None = None) -> MeridianConfig:
    _audit = audit_log if audit_log is not None else NoopAuditLog()
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "config.load",
        attributes={"config.path": str(path)},
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="config.load.invocation",
                code="config_load",
                timestamp=now,
            ),
        )

        try:
            raw = yaml.safe_load(path.read_text())
            if not isinstance(raw, dict):
                raise ValueError(f"Config at {path} is not a YAML mapping")

            config = MeridianConfig.model_validate(raw)

            if config.version != MERIDIAN_CONFIG_VERSION:
                raise ValueError(
                    f"Config version {config.version!r} does not match "
                    f"binary version {MERIDIAN_CONFIG_VERSION!r}"
                )

            span.set_attribute("config.version", config.version)
            return config

        except ConfigLoadError:
            raise
        except Exception as exc:
            err = ConfigLoadError(
                message=f"Failed to load config from {path}: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err)
            _audit.write(
                AuditLogEntry(
                    level="error",
                    event="config.load.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "path": str(path),
                        "message": str(exc),
                    },
                )
            )
            raise err


def resolve_config_location(audit_log: AuditLog | None = None) -> Path:
    """Return the config file path using first-match order:
    $MERIDIAN_CONFIG → ~/.meridian/config.yml → /etc/meridian/config.yml.

    If $MERIDIAN_CONFIG is set but the path does not exist the search stops
    and ConfigResolveError is raised — it does not fall through.
    """
    _audit = audit_log if audit_log is not None else NoopAuditLog()
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span("config.resolve_location") as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="config.resolve_location.invocation",
                code="config_resolve_location",
                timestamp=now,
            ),
        )

        try:
            env_val = os.environ.get(_MERIDIAN_CONFIG_ENV)
            if env_val is not None:
                path = Path(env_val).expanduser()
                if not path.exists():
                    raise FileNotFoundError(
                        f"${_MERIDIAN_CONFIG_ENV} is set but path does not exist: {path}"
                    )
                span.set_attribute("config.source", "env")
                span.set_attribute("config.path", str(path))
                return path

            if _USER_CONFIG_PATH.exists():
                span.set_attribute("config.source", "user")
                span.set_attribute("config.path", str(_USER_CONFIG_PATH))
                return _USER_CONFIG_PATH

            if SYSTEM_CONFIG_PATH.exists():
                span.set_attribute("config.source", "system")
                span.set_attribute("config.path", str(SYSTEM_CONFIG_PATH))
                return SYSTEM_CONFIG_PATH

            searched = (
                f"${_MERIDIAN_CONFIG_ENV} (not set), "
                f"{_USER_CONFIG_PATH} (not found), "
                f"{SYSTEM_CONFIG_PATH} (not found)"
            )
            raise FileNotFoundError(f"No config file found; searched: {searched}")

        except ConfigResolveError:
            raise
        except Exception as exc:
            err = ConfigResolveError(
                message=f"Failed to resolve config location: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err)
            _audit.write(
                AuditLogEntry(
                    level="error",
                    event="config.resolve_location.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={"message": str(exc)},
                )
            )
            raise err


def validate_config(config: MeridianConfig, audit_log: AuditLog | None = None) -> None:
    """Validate vaults, providers, daemon, and storage sections; raise ConfigValidateError on failure."""
    _audit = audit_log if audit_log is not None else NoopAuditLog()
    now = _now()
    tracer = get_tracer()

    with tracer.start_as_current_span("config.validate") as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="config.validate.invocation",
                code="config_validate",
                timestamp=now,
            ),
        )

        errors: list[str] = []

        # Validate vaults section
        seen_ids: set[str] = set()
        for i, vault in enumerate(config.vaults):
            prefix = f"vaults[{i}]"
            if not vault.id.strip():
                errors.append(f"{prefix}.id must not be empty")
                continue
            if vault.id in seen_ids:
                errors.append(f"{prefix}.id: duplicate vault id {vault.id!r}")
            else:
                seen_ids.add(vault.id)
            if vault.backend not in _VALID_VAULT_BACKENDS:
                errors.append(
                    f"{prefix}.backend: {vault.backend!r} is not valid; "
                    f"expected one of {sorted(_VALID_VAULT_BACKENDS)}"
                )

        # Validate providers section
        seen_names: set[str] = set()
        for i, provider in enumerate(config.providers):
            prefix = f"providers[{i}]"
            if not provider.name.strip():
                errors.append(f"{prefix}.name must not be empty")
                continue
            if provider.name in seen_names:
                errors.append(f"{prefix}.name: duplicate provider name {provider.name!r}")
            else:
                seen_names.add(provider.name)
            if provider.kind not in _VALID_PROVIDER_KINDS:
                errors.append(
                    f"{prefix}.kind: {provider.kind!r} is not valid; "
                    f"expected one of {sorted(_VALID_PROVIDER_KINDS)}"
                )
            if provider.auth is not None and provider.auth.startswith("secret_ref://"):
                if not provider.auth.startswith(_SECRET_REF_VAULT_PREFIX):
                    errors.append(
                        f"{prefix}.auth: invalid secret_ref format; "
                        f"expected secret_ref://vault/{{vault_id}}/{{key}}"
                    )
                else:
                    remainder = provider.auth[len(_SECRET_REF_VAULT_PREFIX):]
                    slash_pos = remainder.find("/")
                    if slash_pos <= 0 or slash_pos == len(remainder) - 1:
                        errors.append(
                            f"{prefix}.auth: invalid secret_ref format; "
                            f"expected secret_ref://vault/{{vault_id}}/{{key}}"
                        )

        # Validate routing section
        if config.routing is not None:
            for i, rule in enumerate(config.routing.rules):
                if ":" not in rule.model:
                    errors.append(
                        f"routing.rules[{i}].model: {rule.model!r} must be in "
                        "'provider_name:model_id' form"
                    )
                else:
                    pname = rule.model.split(":")[0]
                    if pname not in seen_names:
                        errors.append(
                            f"routing.rules[{i}].model: provider '{pname}' "
                            "is not declared in the providers section"
                        )
            for i, fb in enumerate(config.routing.fallbacks):
                if ":" not in fb.model:
                    errors.append(
                        f"routing.fallbacks[{i}].model: {fb.model!r} must be in "
                        "'provider_name:model_id' form"
                    )
                else:
                    pname = fb.model.split(":")[0]
                    if pname not in seen_names:
                        errors.append(
                            f"routing.fallbacks[{i}].model: provider '{pname}' "
                            "is not declared in the providers section"
                        )

        # Emit config.plaintext_secret warning for each provider with plaintext auth
        for provider in config.providers:
            if provider.auth is not None and not provider.auth.startswith("secret_ref://"):
                _audit.write(
                    AuditLogEntry(
                        level="warn",
                        event="config.plaintext_secret",
                        code="config_plaintext_secret",
                        timestamp=_now(),
                        detail={"provider": provider.name},
                    )
                )

        # Validate daemon section
        if config.daemon is not None:
            if config.daemon.log_level.lower() not in _VALID_LOG_LEVELS:
                errors.append(
                    f"daemon.log_level: {config.daemon.log_level!r} is not valid; "
                    f"expected one of {sorted(_VALID_LOG_LEVELS)}"
                )
            port = config.daemon.bind.port
            if not (1 <= port <= 65535):
                errors.append(f"daemon.bind.port: {port} is not in range 1-65535")

        if errors:
            err = ConfigValidateError(
                message=f"Config validation failed: {'; '.join(errors)}",
                timestamp=_now(),
            )
            record_error(span, err)
            _audit.write(
                AuditLogEntry(
                    level="error",
                    event="config.validate.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={"errors": errors},
                )
            )
            raise err
