"""meridianconfig subcommands – validate ~/.meridian/config.yml."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ._audit import write_audit
from ._telemetry import get_tracer, record_failure, record_invocation_event

DEFAULT_CONFIG_PATH = Path.home() / ".meridian" / "config.yml"
_CONFIG_VERSION = 1


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
        _MeridianConfig.model_validate(raw)
    except ValidationError as exc:
        return [
            f"  {'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]

    return []


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
