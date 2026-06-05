from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import ToolResult

# In-process store keyed by (tool_name, idempotency_key).  Scoped to the
# Sandbox worker process; long-lived retries across restarts rely on the
# session event-log replay path (Architecture §5.4, §11.5).
_lock = threading.Lock()
_store: dict[tuple[str, str], ToolResult] = {}


def get_cached_result(tool_name: str, idempotency_key: str) -> ToolResult | None:
    """Return the previously stored result for (tool_name, idempotency_key), or None."""
    with _lock:
        return _store.get((tool_name, idempotency_key))


def cache_result(tool_name: str, idempotency_key: str, result: ToolResult) -> None:
    """Store result under (tool_name, idempotency_key) if the slot is empty.

    Uses setdefault so a concurrent first-write always wins — subsequent
    writes for the same key are silently discarded.
    """
    with _lock:
        _store.setdefault((tool_name, idempotency_key), result)


def clear() -> None:
    """Remove all cached results.  Test-only — do not call in production."""
    with _lock:
        _store.clear()
