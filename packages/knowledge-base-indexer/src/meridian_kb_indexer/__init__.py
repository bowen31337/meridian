"""Meridian Workspace Indexer.

Watches ``$WORKSPACE`` via platform-native file events (FSEvents on macOS,
inotify on Linux), chunks code files into symbol-level units via tree-sitter,
and chunks markdown / plain-text files by heading boundaries.

Quick start::

    from meridian_kb_indexer import WorkspaceIndexer, IndexEvent

    async def handle(event: IndexEvent) -> None:
        for chunk in event.chunks:
            print(chunk.kind, chunk.symbol_name, chunk.start_line)

    indexer = WorkspaceIndexer(workspace="/workspace", on_index_event=handle)
    await indexer.start()
    ...
    indexer.stop()
"""

from ._audit import write_audit_event
from ._chunker import chunk_file, detect_language, should_index_path
from ._telemetry import get_tracer, record_indexer_failure, record_invocation_event
from ._types import Chunk, ChunkKind, IndexEvent, IndexEventKind, IndexerError
from ._version import KB_INDEXER_VERSION
from ._watcher import WorkspaceWatcher
from .indexer import WorkspaceIndexer

__version__ = KB_INDEXER_VERSION

__all__ = [
    # Main class
    "WorkspaceIndexer",
    # Watcher
    "WorkspaceWatcher",
    # Chunker helpers
    "chunk_file",
    "detect_language",
    "should_index_path",
    # Types
    "Chunk",
    "ChunkKind",
    "IndexEvent",
    "IndexEventKind",
    "IndexerError",
    # Telemetry
    "get_tracer",
    "record_invocation_event",
    "record_indexer_failure",
    # Audit
    "write_audit_event",
    # Version
    "__version__",
    "KB_INDEXER_VERSION",
]
