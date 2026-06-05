from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

from core_errors import AuditLog, AuditLogEntry
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

from ._pagination import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, build_link_header


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CursorPaginationMiddleware:
    """ASGI middleware for cursor-based pagination.

    Responsibilities:
    - Rejects requests where limit > MAX_PAGE_SIZE or limit < 1 with 422;
      writes to the audit log on rejection.
    - Converts the internal X-Next-Cursor response header into an RFC 8288
      Link: <url>; rel="next" header and strips the internal header.
    - On link-build failure, logs the error and still strips the internal header.
    """

    def __init__(self, app: ASGIApp, *, audit_log: AuditLog) -> None:
        self._app = app
        self._audit_log = audit_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        query_string = scope.get("query_string", b"").decode()
        params: dict[str, str] = {}
        for part in query_string.split("&"):
            if "=" in part:
                k, _, v = part.partition("=")
                params[k] = v

        now = _now()

        limit_str = params.get("limit")
        if limit_str is not None:
            try:
                limit_val = int(limit_str)
            except ValueError:
                limit_val = 0
            if limit_val < 1 or limit_val > MAX_PAGE_SIZE:
                await self._send_error(
                    send,
                    422,
                    "cursor_limit_exceeded",
                    f"limit must be between 1 and {MAX_PAGE_SIZE}",
                )
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="cursor.pagination.limit.rejected",
                        code="cursor_limit_exceeded",
                        timestamp=now,
                        detail={"limit": limit_str},
                    )
                )
                return

        effective_limit = int(params.get("limit", DEFAULT_PAGE_SIZE))

        scheme = scope.get("scheme", "http")
        host_header = b""
        for name, value in scope.get("headers", []):
            if name == b"host":
                host_header = value
                break
        path = scope.get("path", "/")
        if host_header:
            request_url = f"{scheme}://{host_header.decode()}{path}"
        else:
            server = scope.get("server")
            host_str = f"{server[0]}:{server[1]}" if server else "localhost"
            request_url = f"{scheme}://{host_str}{path}"
        if query_string:
            request_url = f"{request_url}?{query_string}"

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                next_cursor = headers.get("x-next-cursor")
                if next_cursor:
                    try:
                        link = build_link_header(request_url, next_cursor, effective_limit)
                        headers.append("Link", link)
                    except Exception as exc:
                        self._audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="cursor.pagination.link.failed",
                                code="cursor_link_build_failed",
                                timestamp=_now(),
                                detail={"error": str(exc)},
                            )
                        )
                    del headers["x-next-cursor"]
            await send(message)

        await self._app(scope, receive, send_wrapper)

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
