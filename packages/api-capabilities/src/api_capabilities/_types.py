from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


class CapabilityInfo(BaseModel):
    """A single declared capability with its namespace, name, and parameterisation flag."""

    id: str
    namespace: str
    name: str
    param_expected: bool


class ListCapabilitiesResponse(BaseModel):
    capabilities: list[CapabilityInfo]


class PluginCapabilitySpec(BaseModel):
    """One capability entry inside a plugin namespace registration request."""

    name: str
    param_expected: bool


class RegisterNamespaceRequest(BaseModel):
    namespace: str
    capabilities: list[PluginCapabilitySpec]


class RegisterNamespaceResponse(BaseModel):
    namespace: str
    capabilities: list[CapabilityInfo]


@dataclass(frozen=True)
class AuditLogEntry:
    """Append-only record written to the audit log on every capabilities API failure."""

    level: Literal["info", "warning", "error"]
    event: str
    operation: str
    timestamp: str
    detail: dict[str, Any] | None = None
