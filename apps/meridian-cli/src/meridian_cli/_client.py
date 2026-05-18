"""HTTP client for the meridian daemon (UDS or TCP loopback)."""

from __future__ import annotations

import os
from typing import Any

import httpx

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7432


class DaemonError(Exception):
    """Raised when the daemon returns an error or is unreachable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DaemonClient:
    """Thin HTTP client that routes to the daemon via UDS or TCP."""

    def __init__(
        self,
        socket: str | None = None,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        if socket:
            self._base_url = "http://localhost"
            self._transport: httpx.HTTPTransport | None = httpx.HTTPTransport(uds=socket)
        else:
            self._base_url = f"http://{host}:{port}"
            self._transport = None

    def request(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Send a request to the daemon and return the parsed JSON body.

        Raises DaemonError on connection failure or non-2xx response.
        """
        kwargs: dict[str, Any] = {}
        if json_body is not None:
            kwargs["json"] = json_body
        if content is not None:
            kwargs["content"] = content
        if headers:
            kwargs["headers"] = headers

        try:
            with httpx.Client(
                base_url=self._base_url,
                transport=self._transport,
                timeout=30.0,
            ) as client:
                resp = client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise DaemonError(
                code="daemon_unreachable",
                message=f"cannot connect to daemon at {self._base_url}: {exc}",
            ) from exc
        except httpx.TimeoutException as exc:
            raise DaemonError(
                code="daemon_timeout",
                message=f"request to {self._base_url}{path} timed out: {exc}",
            ) from exc
        except httpx.HTTPError as exc:
            raise DaemonError(
                code="daemon_http_error",
                message=f"HTTP error talking to daemon: {exc}",
            ) from exc

        if resp.is_error:
            try:
                body = resp.json()
                err = body.get("error", {})
                code = err.get("code", f"http_{resp.status_code}")
                message = err.get("message", resp.text)
            except Exception:
                code = f"http_{resp.status_code}"
                message = resp.text
            raise DaemonError(code=code, message=message)

        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()


def client_from_env(
    socket: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> DaemonClient:
    """Build a DaemonClient from explicit args, falling back to env vars and defaults."""
    resolved_socket = socket or os.environ.get("MERIDIAN_SOCKET")
    if resolved_socket:
        return DaemonClient(socket=resolved_socket)
    resolved_host = host or os.environ.get("MERIDIAN_HOST", _DEFAULT_HOST)
    resolved_port = port or int(os.environ.get("MERIDIAN_PORT", _DEFAULT_PORT))
    return DaemonClient(host=resolved_host, port=resolved_port)
