# Types
# Audit
from ._audit import AuditLog, NoopAuditLog

# Handlers
from ._handlers import HandlerOptions, install_error_handler

# Telemetry
from ._telemetry import get_tracer, record_error, record_invocation_event
from ._types import (
    AuditLogEntry,
    BudgetExceededError,
    CapabilityDeniedError,
    DivergenceError,
    MeridianError,
    SchemaInvalidError,
    StructuredEvent,
    VaultUnauthorizedError,
)

# Version
from ._version import CORE_ERRORS_VERSION

__all__ = [
    # Types
    "MeridianError",
    "CapabilityDeniedError",
    "SchemaInvalidError",
    "VaultUnauthorizedError",
    "BudgetExceededError",
    "DivergenceError",
    "AuditLogEntry",
    "StructuredEvent",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Telemetry
    "get_tracer",
    "record_error",
    "record_invocation_event",
    # Handlers
    "HandlerOptions",
    "install_error_handler",
    # Version
    "CORE_ERRORS_VERSION",
]
