# Types
# Audit log
from ._audit import AuditLog, NoopAuditLog

# Contract
from ._contract import EventHandler
from ._indexer import BackgroundIndexer

# Phase projection
from ._phase import PhaseProjection, PhaseProjectionOptions, PhaseProjectionRuntime

# Core components
from ._reader import LocalEventLogReader
from ._reader_runtime import ReaderOptions, ReaderRuntime

# Runtime
from ._runtime import IndexerOptions, IndexerRuntime
from ._store import SQLiteProjectionStore

# Telemetry
from ._telemetry import (
    get_tracer,
    record_indexer_failure,
    record_invocation_event,
    record_phase_failure,
    record_reader_failure,
)
from ._types import AuditLogEntry, EventSeq, IndexerFailure, StructuredEvent

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
    # Phase projection
    "PhaseProjection",
    "PhaseProjectionOptions",
    "PhaseProjectionRuntime",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Telemetry
    "get_tracer",
    "record_indexer_failure",
    "record_invocation_event",
    "record_phase_failure",
    "record_reader_failure",
    # Runtime
    "IndexerOptions",
    "IndexerRuntime",
    "ReaderOptions",
    "ReaderRuntime",
    # Version
    "INDEXER_SDK_VERSION",
]
