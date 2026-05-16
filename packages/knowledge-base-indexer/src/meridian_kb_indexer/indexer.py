from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Coroutine

from ._audit import write_audit_event
from ._chunker import chunk_file, should_index_path
from ._telemetry import get_tracer, record_indexer_failure, record_invocation_event
from ._types import Chunk, IndexEvent, IndexerError
from ._watcher import WorkspaceWatcher

_WORKSPACE_ENV = "WORKSPACE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceIndexer:
    """Watches ``$WORKSPACE`` and indexes files into symbol/heading/text chunks.

    On ``start()``:
      1. Full scan — every indexable file in *workspace* is read and chunked;
         one ``initial_scan`` IndexEvent is emitted per file.
      2. Watcher is started — FSEvents (macOS) or inotify (Linux) fires
         ``created`` / ``modified`` / ``deleted`` events as files change.

    Every index operation is wrapped in an OTel span.  On failure the error
    message is surfaced to the caller and written to the audit log.

    Usage::

        async def handle(event: IndexEvent) -> None:
            for chunk in event.chunks:
                store(chunk)

        indexer = WorkspaceIndexer(workspace="/workspace", on_index_event=handle)
        await indexer.start()
        ...
        indexer.stop()
    """

    def __init__(
        self,
        workspace: str | None = None,
        on_index_event: Callable[[IndexEvent], Coroutine] | None = None,  # type: ignore[type-arg]
        audit_log_path: str | None = None,
    ) -> None:
        self._workspace = workspace or os.environ.get(_WORKSPACE_ENV, os.getcwd())
        self._on_index_event = on_index_event
        self._audit_log_path = audit_log_path
        self._watcher: WorkspaceWatcher | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Full workspace scan then begin watching for changes."""
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "indexer.start",
            attributes={"indexer.workspace": self._workspace},
        ) as span:
            record_invocation_event(span, operation="start")
            try:
                await self._scan_workspace()
                loop = asyncio.get_event_loop()
                self._watcher = WorkspaceWatcher(
                    workspace=self._workspace,
                    on_change=self._handle_change,
                    loop=loop,
                )
                self._watcher.start()
            except IndexerError:
                raise
            except Exception as exc:
                record_indexer_failure(span, exc, operation="start")
                write_audit_event(
                    "indexer.start.failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    audit_log_path=self._audit_log_path,
                )
                raise IndexerError("INDEXER_START_FAILED", str(exc)) from exc

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    async def index_file(self, file_path: str) -> list[Chunk]:
        """Read and parse *file_path* into Chunks.

        Emits an OTel span.  On failure writes to the audit log and raises
        ``IndexerError`` so callers receive a structured error message.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "indexer.index_file",
            attributes={"indexer.file_path": file_path},
        ) as span:
            record_invocation_event(span, operation="index_file", file_path=file_path)
            try:
                content = Path(file_path).read_text(encoding="utf-8", errors="replace")
                chunks = chunk_file(file_path, content)
                record_invocation_event(
                    span,
                    operation="index_file",
                    file_path=file_path,
                    chunk_count=len(chunks),
                )
                return chunks
            except IndexerError:
                raise
            except Exception as exc:
                record_indexer_failure(span, exc, operation="index_file", file_path=file_path)
                write_audit_event(
                    "indexer.index_file.failed",
                    file_path=file_path,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    audit_log_path=self._audit_log_path,
                )
                raise IndexerError(
                    "INDEXER_FILE_FAILED", str(exc), file_path=file_path
                ) from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _scan_workspace(self) -> None:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "indexer.scan_workspace",
            attributes={"indexer.workspace": self._workspace},
        ) as span:
            record_invocation_event(span, operation="scan_workspace")
            total_chunks = 0

            for path in Path(self._workspace).rglob("*"):
                if not path.is_file():
                    continue
                if not should_index_path(str(path)):
                    continue
                try:
                    chunks = await self.index_file(str(path))
                    total_chunks += len(chunks)
                    if self._on_index_event:
                        await self._on_index_event(
                            IndexEvent(
                                event_kind="initial_scan",
                                file_path=str(path),
                                chunks=chunks,
                                timestamp=_now(),
                            )
                        )
                except IndexerError:
                    pass  # per-file failures already audited in index_file()

            record_invocation_event(
                span,
                operation="scan_workspace",
                chunk_count=total_chunks,
            )

    async def _handle_change(self, event: IndexEvent) -> None:
        if event.event_kind == "deleted":
            if self._on_index_event:
                await self._on_index_event(event)
            return

        try:
            chunks = await self.index_file(event.file_path)
            enriched = IndexEvent(
                event_kind=event.event_kind,
                file_path=event.file_path,
                chunks=chunks,
                timestamp=event.timestamp,
            )
        except IndexerError as exc:
            enriched = IndexEvent(
                event_kind=event.event_kind,
                file_path=event.file_path,
                error=exc.message,
                timestamp=event.timestamp,
            )

        if self._on_index_event:
            await self._on_index_event(enriched)
