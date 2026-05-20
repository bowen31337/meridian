from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


class ModelCapabilityFlags(BaseModel):
    streaming: bool
    thinking: bool
    vision: bool
    tools: bool
    cache: bool


class ModelInfo(BaseModel):
    provider: str
    model: str
    context_window: int
    capabilities: ModelCapabilityFlags


class ListModelsResponse(BaseModel):
    models: list[ModelInfo]


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every models API failure."""

    level: Literal["info", "warning", "error"]
    event: str
    operation: str
    timestamp: str
    detail: dict[str, Any] | None = None
