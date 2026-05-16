"""Tests for WorkspaceWatcher: file-system events delivered via asyncio."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from meridian_kb_indexer import IndexEvent, WorkspaceWatcher


async def _collect_events(
    workspace: Path,
    action,
    timeout: float = 3.0,
    max_events: int = 3,
) -> list[IndexEvent]:
    """Run *action* then wait up to *timeout* seconds for *max_events* IndexEvents."""
    received: list[IndexEvent] = []
    ready = asyncio.Event()

    async def handler(event: IndexEvent) -> None:
        received.append(event)
        if len(received) >= max_events:
            ready.set()

    loop = asyncio.get_event_loop()
    watcher = WorkspaceWatcher(str(workspace), on_change=handler, loop=loop)
    watcher.start()
    try:
        action()
        try:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
    finally:
        watcher.stop()

    return received


@pytest.mark.asyncio
async def test_watcher_fires_on_file_creation(tmp_path: Path) -> None:
    new_file = tmp_path / "hello.txt"

    def create() -> None:
        time.sleep(0.1)
        new_file.write_text("hello")

    events = await _collect_events(tmp_path, create)
    paths = [e.file_path for e in events]
    assert str(new_file) in paths


@pytest.mark.asyncio
async def test_watcher_fires_on_file_modification(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    existing.write_text("initial")

    def modify() -> None:
        time.sleep(0.1)
        existing.write_text("modified")

    events = await _collect_events(tmp_path, modify)
    kinds = {e.event_kind for e in events}
    assert kinds & {"created", "modified"}


@pytest.mark.asyncio
async def test_watcher_fires_on_file_deletion(tmp_path: Path) -> None:
    to_delete = tmp_path / "gone.txt"
    to_delete.write_text("bye")

    def delete() -> None:
        time.sleep(0.2)
        to_delete.unlink()

    events = await _collect_events(tmp_path, delete, timeout=5.0, max_events=1)
    # FSEvents on macOS may report deletion as "deleted" or as a dir "modified".
    # Accept either as long as the target file path appears somewhere.
    all_paths = [e.file_path for e in events]
    deleted_events = [e for e in events if e.event_kind == "deleted"]
    target = str(to_delete)
    assert target in all_paths or any(target in e.file_path for e in deleted_events)


@pytest.mark.asyncio
async def test_watcher_ignores_pycache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()

    received: list[IndexEvent] = []

    async def handler(event: IndexEvent) -> None:
        received.append(event)

    loop = asyncio.get_event_loop()
    watcher = WorkspaceWatcher(str(tmp_path), on_change=handler, loop=loop)
    watcher.start()
    try:
        time.sleep(0.1)
        (cache_dir / "foo.pyc").write_text("bytecode")
        await asyncio.sleep(0.5)
    finally:
        watcher.stop()

    cache_events = [e for e in received if "__pycache__" in e.file_path]
    assert len(cache_events) == 0


@pytest.mark.asyncio
async def test_watcher_stop_is_idempotent(tmp_path: Path) -> None:
    async def noop(event: IndexEvent) -> None:
        pass

    watcher = WorkspaceWatcher(str(tmp_path), on_change=noop)
    watcher.start()
    watcher.stop()
    watcher.stop()  # should not raise
