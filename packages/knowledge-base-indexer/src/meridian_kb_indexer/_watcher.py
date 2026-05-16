from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Coroutine

from watchdog.events import (  # type: ignore[import-untyped]
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer  # type: ignore[import-untyped]

from ._chunker import should_index_path
from ._types import IndexEvent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _BridgeHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Converts watchdog events to IndexEvent and schedules the async callback."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        callback: Callable[[IndexEvent], Coroutine],  # type: ignore[type-arg]
    ) -> None:
        super().__init__()
        self._loop = loop
        self._callback = callback

    def _fire(self, event: IndexEvent) -> None:
        asyncio.run_coroutine_threadsafe(self._callback(event), self._loop)

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and should_index_path(str(event.src_path)):
            self._fire(
                IndexEvent(
                    event_kind="created",
                    file_path=str(event.src_path),
                    timestamp=_now(),
                )
            )

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and should_index_path(str(event.src_path)):
            self._fire(
                IndexEvent(
                    event_kind="modified",
                    file_path=str(event.src_path),
                    timestamp=_now(),
                )
            )

    def on_deleted(self, event: FileDeletedEvent) -> None:  # type: ignore[override]
        if not event.is_directory:
            self._fire(
                IndexEvent(
                    event_kind="deleted",
                    file_path=str(event.src_path),
                    timestamp=_now(),
                )
            )


class WorkspaceWatcher:
    """Watches a directory for file changes using platform-native events.

    On macOS this uses FSEvents; on Linux inotify; on Windows ReadDirectoryChangesW.
    Events are dispatched by scheduling *on_change* on the given asyncio event loop.

    Usage::

        watcher = WorkspaceWatcher("/workspace", on_change=handler, loop=loop)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(
        self,
        workspace: str,
        on_change: Callable[[IndexEvent], Coroutine],  # type: ignore[type-arg]
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._workspace = workspace
        self._on_change = on_change
        self._loop = loop
        self._observer: Observer | None = None

    def start(self) -> None:
        """Start watching *workspace* recursively."""
        loop = self._loop or asyncio.get_event_loop()
        handler = _BridgeHandler(loop, self._on_change)
        self._observer = Observer()
        self._observer.schedule(handler, self._workspace, recursive=True)
        self._observer.start()

    def stop(self) -> None:
        """Stop watching and join the observer thread."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
