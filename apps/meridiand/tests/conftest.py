"""Shared fixtures for the meridiand conformance suite."""

from __future__ import annotations

import pytest
import starlette.testclient as _starlette_tc


# ---------------------------------------------------------------------------
# Patch Starlette's TestClient to simulate loopback connections.
#
# TestClient hardcodes ("testclient", 50000) as the ASGI scope client, which
# is not a valid IP address.  AuthMiddleware enforces loopback-only, so every
# TestClient request would be rejected with 403.  The patch wraps the ASGI
# app inside each transport instance to replace the client tuple with
# ("127.0.0.1", 50000) before the app sees the scope.
# ---------------------------------------------------------------------------

_orig_handle_request = _starlette_tc._TestClientTransport.handle_request


def _loopback_handle_request(self, request):  # type: ignore[no-untyped-def]
    _orig_app = self.app

    class _LoopbackProxy:
        async def __call__(self_, scope, receive, send) -> None:  # noqa: N805
            if scope.get("type") == "http":
                scope = {**scope, "client": ("127.0.0.1", 50000)}
            await _orig_app(scope, receive, send)

    self.app = _LoopbackProxy()
    try:
        return _orig_handle_request(self, request)
    finally:
        self.app = _orig_app


_starlette_tc._TestClientTransport.handle_request = _loopback_handle_request  # type: ignore[method-assign]


@pytest.fixture()
def storage_root(tmp_path):
    root = tmp_path / "storage"
    root.mkdir()
    return root
