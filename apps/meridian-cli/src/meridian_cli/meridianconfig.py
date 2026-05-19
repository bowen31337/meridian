"""meridianconfig subcommands – validate and migrate ~/.meridian/config.yml."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import click
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ._audit import write_audit
from ._telemetry import get_tracer, record_failure, record_invocation_event

DEFAULT_CONFIG_PATH = Path.home() / ".meridian" / "config.yml"
_CONFIG_VERSION = 2


# ---------------------------------------------------------------------------
# Config schema (mirrors meridiand._config without the daemon dependency)
# ---------------------------------------------------------------------------


class _BindConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = 8888
    socket: str | None = None

    @field_validator("socket", mode="before")
    @classmethod
    def _expand_socket(cls, v: object) -> str | None:
        if v is None:
            return None
        return str(Path(str(v)).expanduser())


class _CorsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    allow_origins: list[str] = Field(default_factory=list)
    allow_methods: list[str] = Field(default_factory=list)
    allow_headers: list[str] = Field(default_factory=list)
    allow_credentials: bool = False


class _CompactionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    idle_days: int = 30
    summary_strategy: str = "tail"
    tail_events: int = 50
    retention_days: int | None = None


class _CronSchedulerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    missed_fires_policy: str = "skip"
    check_interval_seconds: float = 5.0


class _WebhookSenderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_interval_seconds: float = 5.0


class _SkillForgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_invocations_per_minute: int = 10
    check_interval_seconds: float = 5.0


class _AuthConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bearer_token: str | None = None


class _VaultConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    backend: str = "os_keychain"


class _DaemonConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    bind: _BindConfig = Field(default_factory=_BindConfig)
    workspace_root: Path = Field(default_factory=lambda: Path.home() / ".meridian")
    log_level: str = "info"

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _expand_workspace_root(cls, v: object) -> Path:
        return Path(str(v)).expanduser()


class _StorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    database: str | None = None
    event_log: str | None = None
    blob_store: str | None = None


class _MeridianConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 1
    storage_root: Path
    bind: _BindConfig = Field(default_factory=_BindConfig)
    log_level: str = "info"
    cors: _CorsConfig = Field(default_factory=_CorsConfig)
    compaction: _CompactionConfig = Field(default_factory=_CompactionConfig)
    cron: _CronSchedulerConfig = Field(default_factory=_CronSchedulerConfig)
    webhook_sender: _WebhookSenderConfig = Field(default_factory=_WebhookSenderConfig)
    skill_forge: _SkillForgeConfig = Field(default_factory=_SkillForgeConfig)
    auth: _AuthConfig = Field(default_factory=_AuthConfig)
    vaults: list[_VaultConfig] = Field(default_factory=list)
    daemon: _DaemonConfig | None = None
    storage: _StorageConfig | None = None

    @field_validator("storage_root", mode="before")
    @classmethod
    def _expand_storage_root(cls, v: object) -> Path:
        return Path(str(v)).expanduser()


# ---------------------------------------------------------------------------
# CLI group + validate command
# ---------------------------------------------------------------------------


@click.group()
def meridianconfig() -> None:
    """Manage Meridian configuration."""


@meridianconfig.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help=f"Config file path (default: {DEFAULT_CONFIG_PATH}).",
)
def validate(config_path: Path | None) -> None:
    """Validate ~/.meridian/config.yml and print OK or a list of errors."""
    path = config_path or DEFAULT_CONFIG_PATH
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "meridianconfig.validate",
        attributes={"config.path": str(path)},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": "meridianconfig.validate.invocation",
                "config.path": str(path),
            },
        )

        errors = _run_validate(path)

        if not errors:
            click.echo("OK")
            span.add_event("meridianconfig.validate.ok")
            return

        # Failure path
        _emit_failure(span, path, errors)
        sys.exit(1)


_VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
_VALID_VAULT_BACKENDS = {"os_keychain", "encrypted_file"}


def _run_validate(path: Path) -> list[str]:
    """Return a list of human-readable error strings, or [] on success."""
    if not path.exists():
        return [f"file not found: {path}"]

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        return [f"invalid YAML: {exc}"]

    if not isinstance(raw, dict):
        return ["config must be a YAML mapping, not a sequence or scalar"]

    version = raw.get("version", _CONFIG_VERSION)
    if version != _CONFIG_VERSION:
        return [
            f"version mismatch: config has version={version!r}, "
            f"expected {_CONFIG_VERSION!r}"
        ]

    try:
        config = _MeridianConfig.model_validate(raw)
    except ValidationError as exc:
        return [
            f"  {'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]

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

    return errors


def _emit_failure(span: object, path: Path, errors: list[str]) -> None:
    error_text = "\n".join(errors)
    message = f"config validation failed: {path}\n{error_text}"
    click.echo(message, err=False)
    record_failure(span, "config_validation_failed", message)  # type: ignore[arg-type]
    write_audit(
        "error",
        "meridianconfig.validate.failed",
        {
            "path": str(path),
            "errors": errors,
        },
    )


# ---------------------------------------------------------------------------
# Upgrade registry (mirrors apps/meridiand/src/meridiand/config/upgrades/)
# ---------------------------------------------------------------------------


def _upgrade_v1_to_v2(raw: dict[str, object]) -> dict[str, object]:
    result = dict(raw)
    result["version"] = 2
    return result


_UPGRADES: dict[int, Callable[[dict[str, object]], dict[str, object]]] = {
    1: _upgrade_v1_to_v2,
}


# ---------------------------------------------------------------------------
# Migrate command helpers
# ---------------------------------------------------------------------------


@dataclass
class _MigrateResult:
    errors: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    from_version: int = 0
    to_version: int = 0


def _run_migrate(path: Path) -> _MigrateResult:
    result = _MigrateResult()

    if not path.exists():
        result.errors.append(f"file not found: {path}")
        return result

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        result.errors.append(f"invalid YAML: {exc}")
        return result

    if not isinstance(raw, dict):
        result.errors.append("config must be a YAML mapping, not a sequence or scalar")
        return result

    current = int(raw.get("version", 1))
    result.from_version = current
    result.to_version = current

    if current > _CONFIG_VERSION:
        result.errors.append(
            f"config version {current} is newer than this tool supports "
            f"(max: {_CONFIG_VERSION})"
        )
        return result

    version = current
    while version < _CONFIG_VERSION:
        upgrade_fn = _UPGRADES.get(version)
        if upgrade_fn is None:
            result.errors.append(f"no upgrade path from version {version} to {version + 1}")
            return result
        raw = upgrade_fn(raw)
        result.applied.append(f"v{version} → v{version + 1}")
        version += 1

    result.to_version = version

    if result.applied:
        path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))

    return result


# ---------------------------------------------------------------------------
# Migrate command
# ---------------------------------------------------------------------------


@meridianconfig.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help=f"Config file path (default: {DEFAULT_CONFIG_PATH}).",
)
def migrate(config_path: Path | None) -> None:
    """Migrate ~/.meridian/config.yml to the latest config version."""
    path = config_path or DEFAULT_CONFIG_PATH
    tracer = get_tracer()

    with tracer.start_as_current_span(
        "meridianconfig.migrate",
        attributes={"config.path": str(path)},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": "meridianconfig.migrate.invocation",
                "config.path": str(path),
            },
        )

        result = _run_migrate(path)

        if result.errors:
            error_text = "\n".join(result.errors)
            message = f"config migration failed: {path}\n{error_text}"
            click.echo(message)
            record_failure(span, "config_migration_failed", message)  # type: ignore[arg-type]
            write_audit(
                "error",
                "meridianconfig.migrate.failed",
                {"path": str(path), "errors": result.errors},
            )
            sys.exit(1)
            return

        if not result.applied:
            click.echo(
                f"config is already at version {result.from_version}, nothing to migrate"
            )
            span.add_event("meridianconfig.migrate.noop")
            return

        summary = ", ".join(result.applied)
        click.echo(f"migrated {path}: {summary}")
        span.add_event(
            "meridianconfig.migrate.ok",
            {
                "migrations.applied": summary,
                "version.from": str(result.from_version),
                "version.to": str(result.to_version),
            },
        )
        write_audit(
            "info",
            "meridianconfig.migrate.ok",
            {
                "path": str(path),
                "from_version": result.from_version,
                "to_version": result.to_version,
                "applied": result.applied,
            },
        )


# ---------------------------------------------------------------------------
# Schema command
# ---------------------------------------------------------------------------


@meridianconfig.command()
def schema() -> None:
    """Emit the MeridianConfig JSON Schema to stdout for editor autocomplete."""
    tracer = get_tracer()

    with tracer.start_as_current_span("meridianconfig.schema") as span:
        record_invocation_event(
            span,
            {"event.name": "meridianconfig.schema.invocation"},
        )

        try:
            json_schema = _MeridianConfig.model_json_schema()
            click.echo(json.dumps(json_schema, indent=2))
            span.add_event("meridianconfig.schema.ok")
        except Exception as exc:
            message = f"failed to generate config schema: {exc}"
            click.echo(message)
            record_failure(span, "config_schema_failed", message)  # type: ignore[arg-type]
            write_audit(
                "error",
                "meridianconfig.schema.failed",
                {"error": str(exc)},
            )
            sys.exit(1)
