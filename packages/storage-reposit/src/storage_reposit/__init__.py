# Types
# Audit log
from ._audit import AuditLog, NoopAuditLog

# Contract
from ._contract import EventHandler
from ._indexer import BackgroundIndexer

# Migration
from ._migration_runtime import MigrationOptions, MigrationRuntime
from ._migrations import SCHEMA_VERSION

# Phase projection
from ._phase import PhaseProjection, PhaseProjectionOptions, PhaseProjectionRuntime

# Core components
from ._reader import LocalEventLogReader
from ._reader_runtime import ReaderOptions, ReaderRuntime

# Runtime
from ._runtime import IndexerOptions, IndexerRuntime

# Phase state machine
from ._state_machine import (
    EVENTS,
    PHASES,
    PhaseStateMachine,
    PhaseStateMachineOptions,
    PhaseStateMachineRuntime,
)
from ._store import SQLiteProjectionStore

# Telemetry
from ._telemetry import (
    get_tracer,
    record_indexer_failure,
    record_invocation_event,
    record_migration_failure,
    record_phase_failure,
    record_reader_failure,
    record_state_machine_failure,
)
from ._types import AuditLogEntry, EventSeq, IndexerFailure, StructuredEvent

# Usage rollup
from ._usage_rollup import UsageRollupProjector

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
    # Migration
    "MigrationOptions",
    "MigrationRuntime",
    "SCHEMA_VERSION",
    # Phase projection
    "PhaseProjection",
    "PhaseProjectionOptions",
    "PhaseProjectionRuntime",
    # Phase state machine
    "EVENTS",
    "PHASES",
    "PhaseStateMachine",
    "PhaseStateMachineOptions",
    "PhaseStateMachineRuntime",
    # Audit
    "AuditLog",
    "NoopAuditLog",
    # Telemetry
    "get_tracer",
    "record_indexer_failure",
    "record_invocation_event",
    "record_migration_failure",
    "record_phase_failure",
    "record_reader_failure",
    "record_state_machine_failure",
    # Runtime
    "IndexerOptions",
    "IndexerRuntime",
    "ReaderOptions",
    "ReaderRuntime",
    # Usage rollup
    "UsageRollupProjector",
    # Version
    "INDEXER_SDK_VERSION",
]
