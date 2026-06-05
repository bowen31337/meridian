from __future__ import annotations

import base64
from datetime import UTC, datetime
import json
from typing import Any
from urllib.parse import urlparse, urlunparse

from core_errors import MeridianError

DEFAULT_PAGE_SIZE: int = 50
MAX_PAGE_SIZE: int = 200


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CursorDecodeError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="cursor_invalid", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 400


def encode_cursor(created_at: str, record_id: str) -> str:
    """Encode (created_at, id) as an opaque URL-safe base64 cursor token."""
    payload = json.dumps({"t": created_at, "i": record_id}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str, *, timestamp: str) -> tuple[str, str]:
    """Decode a cursor token back to (created_at, id); raises CursorDecodeError on bad input."""
    try:
        padded = cursor + "=="
        payload = base64.urlsafe_b64decode(padded.encode()).decode()
        data = json.loads(payload)
        return str(data["t"]), str(data["i"])
    except Exception as exc:
        raise CursorDecodeError(
            message="Invalid cursor token",
            timestamp=timestamp,
        ) from exc


def apply_cursor_filter(
    items: list[dict[str, Any]],
    created_at: str,
    record_id: str,
) -> list[dict[str, Any]]:
    """Return items strictly after the cursor position in a descending-sorted list."""
    for i, item in enumerate(items):
        if item.get("created_at") == created_at and item.get("id") == record_id:
            return items[i + 1 :]
    # Cursor item was deleted; fall back to tuple comparison (descending order)
    return [
        item
        for item in items
        if (item.get("created_at", ""), item.get("id", "")) < (created_at, record_id)
    ]


def make_cursor_page(
    items: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Slice items to limit and return (page, next_cursor_or_None).

    next_cursor is non-None only when there are items beyond the current page.
    """
    page = items[:limit]
    next_cursor: str | None = None
    if len(items) > limit:
        last = page[-1]
        next_cursor = encode_cursor(
            last.get("created_at", ""),
            last.get("id", ""),
        )
    return page, next_cursor


def build_link_header(request_url: str, next_cursor: str, limit: int) -> str:
    """Return an RFC 8288 Link header value for rel=next."""
    parsed = urlparse(request_url)
    query = f"cursor={next_cursor}&limit={limit}"
    next_url = urlunparse(parsed._replace(query=query))
    return f'<{next_url}>; rel="next"'
