"""Unit tests for meridian_cli._client (HTTP transport + env-var resolution)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from meridian_cli._client import (
    DaemonClient,
    DaemonError,
    client_from_env,
)


def _make_mock_client(handler):
    transport = httpx.MockTransport(handler)
    return transport


def _patch_httpx_client(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    real_init = httpx.Client.__init__

    def _init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("transport", None)
        kwargs.pop("base_url", None)
        kwargs.pop("timeout", None)
        real_init(self, *args, transport=transport, base_url="http://localhost", **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", _init)


class TestDaemonClientConstruction:
    def test_uds_socket_sets_transport(self) -> None:
        c = DaemonClient(socket="/tmp/m.sock")
        assert c._base_url == "http://localhost"
        assert c._transport is not None

    def test_tcp_default_no_transport(self) -> None:
        c = DaemonClient()
        assert c._base_url == "http://127.0.0.1:7432"
        assert c._transport is None

    def test_tcp_custom_host_port(self) -> None:
        c = DaemonClient(host="example.com", port=9000)
        assert c._base_url == "http://example.com:9000"


class TestDaemonClientRequest:
    def test_request_success_returns_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        c = DaemonClient()
        assert c.request("GET", "/v1/x") == {"ok": True}

    def test_request_with_json_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content
            return httpx.Response(200, json={})

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        DaemonClient().request("POST", "/v1/x", json_body={"k": "v"})
        assert b'"k"' in seen["body"]

    def test_request_with_raw_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content
            return httpx.Response(200, json={})

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        DaemonClient().request("POST", "/v1/x", content=b"raw-bytes")
        assert seen["body"] == b"raw-bytes"

    def test_request_with_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["h"] = request.headers.get("x-test", "")
            return httpx.Response(200, json={})

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        DaemonClient().request("GET", "/v1/x", headers={"X-Test": "yes"})
        assert seen["h"] == "yes"

    def test_request_204_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(204)

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        assert DaemonClient().request("DELETE", "/v1/x") is None

    def test_request_empty_body_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        assert DaemonClient().request("GET", "/v1/x") is None

    def test_request_4xx_with_error_json_raises_daemon_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": {"code": "not_found", "message": "no such thing"}},
            )

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        with pytest.raises(DaemonError) as ei:
            DaemonClient().request("GET", "/v1/x")
        assert ei.value.code == "not_found"
        assert "no such thing" in ei.value.message

    def test_request_4xx_with_unparseable_body_falls_back_to_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b"<html>oops</html>")

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        with pytest.raises(DaemonError) as ei:
            DaemonClient().request("GET", "/v1/x")
        assert ei.value.code == "http_500"
        assert "oops" in ei.value.message

    def test_request_4xx_json_without_error_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"unrelated": True})

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        with pytest.raises(DaemonError) as ei:
            DaemonClient().request("POST", "/v1/x")
        # falls back to http_400 since "error" key is absent
        assert ei.value.code == "http_400"

    def test_request_connect_error_raises_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        with pytest.raises(DaemonError) as ei:
            DaemonClient().request("GET", "/v1/x")
        assert ei.value.code == "daemon_unreachable"

    def test_request_timeout_raises_daemon_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        with pytest.raises(DaemonError) as ei:
            DaemonClient().request("GET", "/v1/x")
        assert ei.value.code == "daemon_timeout"

    def test_request_generic_http_error_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.NetworkError("flap", request=request)

        _patch_httpx_client(monkeypatch, httpx.MockTransport(handler))
        with pytest.raises(DaemonError) as ei:
            DaemonClient().request("GET", "/v1/x")
        assert ei.value.code == "daemon_http_error"


class TestClientFromEnv:
    def test_explicit_socket_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MERIDIAN_SOCKET", raising=False)
        c = client_from_env(socket="/tmp/explicit.sock")
        assert c._transport is not None

    def test_env_socket_used_when_no_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERIDIAN_SOCKET", "/tmp/from-env.sock")
        c = client_from_env()
        assert c._transport is not None

    def test_falls_back_to_tcp_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MERIDIAN_SOCKET", raising=False)
        monkeypatch.delenv("MERIDIAN_HOST", raising=False)
        monkeypatch.delenv("MERIDIAN_PORT", raising=False)
        c = client_from_env()
        assert c._base_url == "http://127.0.0.1:7432"

    def test_env_host_port_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MERIDIAN_SOCKET", raising=False)
        monkeypatch.setenv("MERIDIAN_HOST", "10.0.0.1")
        monkeypatch.setenv("MERIDIAN_PORT", "9999")
        c = client_from_env()
        assert c._base_url == "http://10.0.0.1:9999"

    def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERIDIAN_HOST", "ignored")
        monkeypatch.setenv("MERIDIAN_PORT", "1234")
        c = client_from_env(host="explicit", port=4321)
        assert c._base_url == "http://explicit:4321"


class TestDaemonError:
    def test_error_carries_code_and_message(self) -> None:
        err = DaemonError(code="x", message="y")
        assert err.code == "x"
        assert err.message == "y"
        assert str(err) == "y"
