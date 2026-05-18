from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".meridian" / "config.yaml"

_DEFAULT_CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
_DEFAULT_CORS_HEADERS = ["*"]


@dataclass(frozen=True)
class BindConfig:
    host: str = "127.0.0.1"
    port: int = 7432
    socket: str | None = None


@dataclass(frozen=True)
class CorsConfig:
    allow_origins: list[str] = field(default_factory=list)
    allow_methods: list[str] = field(default_factory=lambda: list(_DEFAULT_CORS_METHODS))
    allow_headers: list[str] = field(default_factory=lambda: list(_DEFAULT_CORS_HEADERS))
    allow_credentials: bool = False


@dataclass(frozen=True)
class DaemonConfig:
    storage_root: Path
    bind: BindConfig = field(default_factory=BindConfig)
    log_level: str = "info"
    cors: CorsConfig = field(default_factory=CorsConfig)


def _parse_bind(raw: dict[str, Any] | None) -> BindConfig:
    if raw is None:
        return BindConfig()
    return BindConfig(
        host=str(raw.get("host", "127.0.0.1")),
        port=int(raw.get("port", 7432)),
        socket=raw.get("socket"),
    )


def _parse_cors(raw: dict[str, Any] | None) -> CorsConfig:
    if raw is None:
        return CorsConfig()
    return CorsConfig(
        allow_origins=list(raw.get("allow_origins", [])),
        allow_methods=list(raw.get("allow_methods", _DEFAULT_CORS_METHODS)),
        allow_headers=list(raw.get("allow_headers", _DEFAULT_CORS_HEADERS)),
        allow_credentials=bool(raw.get("allow_credentials", False)),
    )


def load_config(path: Path) -> DaemonConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} is not a YAML mapping")

    storage_root_raw = raw.get("storage_root")
    if not storage_root_raw:
        raise ValueError("Config missing required key: storage_root")

    return DaemonConfig(
        storage_root=Path(storage_root_raw).expanduser(),
        bind=_parse_bind(raw.get("bind")),
        log_level=str(raw.get("log_level", "info")),
        cors=_parse_cors(raw.get("cors")),
    )
