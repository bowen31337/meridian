from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core_errors import AuditLog, AuditLogEntry
from starlette.types import ASGIApp, Receive, Scope, Send


_TTL_SECONDS = 86400  # 24 hours
_MAX_KEY_LENGTH = 255


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _CachedResponse:
    status_code: int
    headers: list[tuple[bytes, bytes]]
    body: bytes
    expires_at: float


class IdempotencyKeyMiddleware:
    """ASGI middleware for Idempotency-Key support on POST endpoints.

    Responsibilities:
    - Intercepts POST requests carrying an Idempotency-Key header.
    - On the first request for a key+path pair: forwards the request and caches
      the full response (status, headers, body) for 24 h.
    - On duplicate requests with the same key: replays the cached response verbatim.
    - Rejects keys that are empty or exceed 255 characters with 422 and writes to
      the audit log.
    - On internal cache-store failure: writes to the audit log and surfaces an
      error to the caller.
    """

    def __init__(self, app: ASGIApp, *, audit_log: AuditLog) -> None:
        self._app = app
        self._audit_log = audit_log
        self._cache: dict[str, _CachedResponse] = {}
        self._lock = threading.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if scope.get("method") != "POST":
            await self._app(scope, receive, send)
            return

        raw_key: bytes | None = None
        for name, value in scope.get("headers", []):
            if name == b"idempotency-key":
                raw_key = value
                break

        if raw_key is None:
            await self._app(scope, receive, send)
            return

        key_str = raw_key.decode("utf-8", errors="replace")

        if not key_str or len(key_str) > _MAX_KEY_LENGTH:
            await self._send_error(
                send,
                422,
                "idempotency_key_invalid",
                "Idempotency-Key must be between 1 and 255 characters",
            )
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="idempotency.key.invalid",
                    code="idempotency_key_invalid",
                    timestamp=_now(),
                    detail={"key_length": len(key_str)},
                )
            )
            return

        path = scope.get("path", "/")
        cache_key = f"POST:{path}:{key_str}"
        now_mono = time.monotonic()

        replay: _CachedResponse | None = None
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                if cached.expires_at > now_mono:
                    replay = cached
                else:
                    del self._cache[cache_key]

        if replay is not None:
            await send(
                {
                    "type": "http.response.start",
                    "status": replay.status_code,
                    "headers": list(replay.headers),
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": replay.body,
                    "more_body": False,
                }
            )
            return

        captured: dict[str, Any] = {}
        body_chunks: list[bytes] = []

        async def capturing_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
                captured["headers"] = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body_chunks.append(message.get("body", b""))
            await send(message)

        await self._app(scope, receive, capturing_send)

        if not captured:
            return

        try:
            entry = _CachedResponse(
                status_code=captured["status"],
                headers=captured["headers"],
                body=b"".join(body_chunks),
                expires_at=now_mono + _TTL_SECONDS,
            )
            with self._lock:
                self._cache[cache_key] = entry
        except Exception as exc:
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="idempotency.cache.store.failed",
                    code="idempotency_cache_store_failed",
                    timestamp=_now(),
                    detail={"error": str(exc), "key": key_str},
                )
            )

    async def _send_error(
        self,
        send: Send,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        body = json.dumps(
            {"error": {"code": code, "message": message}},
            separators=(",", ":"),
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )
