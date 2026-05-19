# Types
# Audit log
from ._audit import AuditLog, NoopAuditLog

# Contract
from ._contract import EnvironmentDriver

# Built-in drivers
from ._http_driver import HttpBackendDriver, VaultCredentialResolver

# Network policy enforcement
from ._enforcer import NetworkEnforcer

# Filesystem policy enforcement
from ._fs_enforcer import FilesystemEnforcer
from ._fs_gate import FilesystemGate

# Worker pool (lifecycle: provision-on-first-use, warm pool, TTL reaper, on-demand)
from ._pool import WorkerPool
from ._proxy import OutboundProxyTransport

# Runtime
from ._runtime import EnvironmentRuntime, RuntimeOptions, default_runtime

# Telemetry
from ._telemetry import (
    get_tracer,
    record_environment_failure,
    record_invocation_event,
    record_pool_event,
)
from ._types import (
    AgentFilesystemPolicy,
    AgentNetworkPolicy,
    AuditLogEntry,
    CapabilityEnvelope,
    EnvironmentFailure,
    ExecuteRequest,
    ExecuteResult,
    FilesystemPolicy,
    FilesystemViolation,
    NetworkPolicy,
    NetworkViolation,
    PoolEvent,
    PoolOptions,
    ProvisionRequest,
    ReclaimRequest,
    StructuredEvent,
)

# Version
from ._version import ENVIRONMENT_SDK_VERSION

__all__ = [
    # Built-in drivers
    "HttpBackendDriver",
    "VaultCredentialResolver",
    # Types
    "AgentFilesystemPolicy",
    "AgentNetworkPolicy",
    "AuditLogEntry",
    "CapabilityEnvelope",
    "EnvironmentFailure",
    "ExecuteRequest",
    "ExecuteResult",
    "FilesystemPolicy",
    "FilesystemViolation",
    "NetworkPolicy",
    "NetworkViolation",
    "PoolEvent",
    "PoolOptions",
    "ProvisionRequest",
    "ReclaimRequest",
    "StructuredEvent",
    # Contract
    "EnvironmentDriver",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Telemetry
    "get_tracer",
    "record_environment_failure",
    "record_invocation_event",
    "record_pool_event",
    # Network policy
    "NetworkEnforcer",
    "OutboundProxyTransport",
    # Filesystem policy
    "FilesystemEnforcer",
    "FilesystemGate",
    # Runtime
    "EnvironmentRuntime",
    "RuntimeOptions",
    "default_runtime",
    # Pool (lifecycle)
    "WorkerPool",
    # Version
    "ENVIRONMENT_SDK_VERSION",
]
