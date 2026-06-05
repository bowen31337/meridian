# Routes
# Audit
from ._audit import AuditLog, NoopAuditLog

# Registry
from ._registry import CapabilityRegistry
from ._routes import make_router

# Types
from ._types import (
    AuditLogEntry,
    CapabilityInfo,
    ListCapabilitiesResponse,
    PluginCapabilitySpec,
    RegisterNamespaceRequest,
    RegisterNamespaceResponse,
)

# Version
from ._version import API_CAPABILITIES_VERSION

__all__ = [
    # Routes
    "make_router",
    # Registry
    "CapabilityRegistry",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Types
    "AuditLogEntry",
    "CapabilityInfo",
    "ListCapabilitiesResponse",
    "PluginCapabilitySpec",
    "RegisterNamespaceRequest",
    "RegisterNamespaceResponse",
    # Version
    "API_CAPABILITIES_VERSION",
]
