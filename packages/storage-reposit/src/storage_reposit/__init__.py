# Types
from ._types import AuditLogEntry, EventSeq, IndexerFailure, StructuredEvent

# Contract
from ._contract import EventHandler

# Core components
from ._reader import LocalEventLogReader
from ._store import SQLiteProjectionStore
from ._indexer import BackgroundIndexer

# Audit log
from ._audit import AuditLog, NoopAuditLog

# Telemetry
from ._telemetry import get_tracer, record_indexer_failure, record_invocation_event, record_reader_failure

# Runtime
from ._runtime import IndexerOptions, IndexerRuntime
from ._reader_runtime import ReaderOptions, ReaderRuntime

# Version
from ._version import INDEXER_SDK_VERSION

__all__ = [
    # Types
    "AuditLogEntry",
    "EventSeq",
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
    "record_reader_failure",
    # Runtime
    "IndexerOptions",
    "IndexerRuntime",
    "ReaderOptions",
    "ReaderRuntime",
    # Version
    "INDEXER_SDK_VERSION",
]
