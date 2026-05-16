# Types
from ._types import AuditLogEntry, IndexerFailure, StructuredEvent

# Contract
from ._contract import EventHandler

# Core components
from ._reader import LocalEventLogReader
from ._store import SQLiteProjectionStore
from ._indexer import BackgroundIndexer

# Audit log
from ._audit import AuditLog, NoopAuditLog

# Telemetry
from ._telemetry import get_tracer, record_indexer_failure, record_invocation_event

# Runtime
from ._runtime import IndexerOptions, IndexerRuntime

# Version
from ._version import INDEXER_SDK_VERSION

__all__ = [
    # Types
    "AuditLogEntry",
    "IndexerFailure",
    "StructuredEvent",
    # Contract
    "EventHandler",
    # Core components
    "LocalEventLogReader",
    "SQLiteProjectionStore",
    "BackgroundIndexer",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Telemetry
    "get_tracer",
    "record_indexer_failure",
    "record_invocation_event",
    # Runtime
    "IndexerOptions",
    "IndexerRuntime",
    # Version
    "INDEXER_SDK_VERSION",
]
