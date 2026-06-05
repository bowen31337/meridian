"""
Credential proxy conformance suite.

Tests cover:
  - POST /v1/credential-proxy/{provider}/{path} returns upstream status code.
  - Authorization: Bearer <token> is injected into the forwarded request.
  - Token never appears in the response body returned to the caller.
  - Request body is forwarded unchanged.
  - Query parameters are forwarded.
  - Unknown provider returns 404 with code "credential_proxy_provider_not_found".
  - Unresolvable token returns 502 with code "credential_proxy_token_unavailable".
  - Network failure returns 502 with code "credential_proxy_forward_failed".
  - Every failure writes an audit log entry at level "error".
  - Audit entry has event "credential.proxy.request.failed".
  - Audit detail contains provider_name, path, and message.
  - OTel span "credential.proxy.request" emitted on success.
  - OTel span "credential.proxy.request" emitted on failure.
  - Span set to ERROR status on failure.
  - Span carries credential.proxy.provider and credential.proxy.path attributes.
  - Route present when credential_proxy_providers configured via create_app.
  - Route absent when credential_proxy_providers is None or empty in create_app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core_errors import HandlerOptions, install_error_handler
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._credential_proxy import (
    CredentialProxyProviderConfig,
    make_credential_proxy_router,
)

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResolver:
    """SecretResolver backed by a plain dict."""

    def __init__(self, mapping: dict[str, str | None]) -> None:
        self._mapping = mapping

    def resolve(self, secret_ref: str) -> str | None:
        return self._mapping.get(secret_ref)


class _CapturingTransport(httpx.AsyncBaseTransport):
    """Captures forwarded requests and returns a configurable fixed response."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b'{"ok": true}',
        headers: dict[str, str] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._status = status
        self._body = body
        self._headers = headers or {}
        self._raise_exc = raise_exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._raise_exc is not None:
            raise self._raise_exc
        return httpx.Response(self._status, content=self._body, headers=self._headers)


_DEFAULT_PROVIDER = CredentialProxyProviderConfig(
    name="testprovider",
    base_url="https://api.example.com",
    token_secret_ref="secret_ref://vault/vault_abc/oauth_token",
)

_DEFAULT_TOKEN = "sk-test-token-abc123"
_DEFAULT_RESOLVER = _FakeResolver({_DEFAULT_PROVIDER.token_secret_ref: _DEFAULT_TOKEN})


def _make_router_client(
    storage_root: Path,
    *,
    providers: list[CredentialProxyProviderConfig] | None = None,
    resolver: _FakeResolver | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> TestClient:
    _providers = providers if providers is not None else [_DEFAULT_PROVIDER]
    _resolver = resolver if resolver is not None else _DEFAULT_RESOLVER
    _transport = transport if transport is not None else _CapturingTransport()
    audit_log = FileAuditLog(storage_root)
    router = make_credential_proxy_router(
        audit_log=audit_log,
        secret_resolver=_resolver,
        providers=_providers,
        http_client=httpx.AsyncClient(transport=_transport),
    )
    app = FastAPI()
    app.include_router(router)
    install_error_handler(app, HandlerOptions(audit_log=audit_log))
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Success: upstream forwarding
# ---------------------------------------------------------------------------


class TestCredentialProxySuccess:
    def test_returns_upstream_status_200(self, storage_root: Path) -> None:
        transport = _CapturingTransport(status=200)
        client = _make_router_client(storage_root, transport=transport)
        resp = client.post("/v1/credential-proxy/testprovider/v1/messages", json={})
        assert resp.status_code == 200

    def test_returns_upstream_status_201(self, storage_root: Path) -> None:
        transport = _CapturingTransport(status=201, body=b'{"id": "x"}')
        client = _make_router_client(storage_root, transport=transport)
        resp = client.post("/v1/credential-proxy/testprovider/v1/items", json={})
        assert resp.status_code == 201

    def test_returns_upstream_body(self, storage_root: Path) -> None:
        transport = _CapturingTransport(status=200, body=b'{"answer": 42}')
        client = _make_router_client(storage_root, transport=transport)
        resp = client.get("/v1/credential-proxy/testprovider/v1/data/info")
        assert resp.json() == {"answer": 42}

    def test_injects_authorization_header(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        client.post("/v1/credential-proxy/testprovider/v1/check", json={})
        assert len(transport.requests) == 1
        assert transport.requests[0].headers["authorization"] == f"Bearer {_DEFAULT_TOKEN}"

    def test_token_not_in_response_body(self, storage_root: Path) -> None:
        upstream_body = b'{"result": "clean"}'
        transport = _CapturingTransport(body=upstream_body)
        client = _make_router_client(storage_root, transport=transport)
        resp = client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        assert _DEFAULT_TOKEN not in resp.text

    def test_forwards_request_body(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        payload = {"model": "gpt-4", "prompt": "hello"}
        client.post(
            "/v1/credential-proxy/testprovider/v1/complete",
            json=payload,
        )
        assert len(transport.requests) == 1
        forwarded_body = json.loads(transport.requests[0].content)
        assert forwarded_body == payload

    def test_forwards_query_params(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        client.get(
            "/v1/credential-proxy/testprovider/v1/list",
            params={"limit": "10", "cursor": "abc"},
        )
        assert len(transport.requests) == 1
        url_str = str(transport.requests[0].url)
        assert "limit=10" in url_str
        assert "cursor=abc" in url_str

    def test_strips_client_authorization_header(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        client.post(
            "/v1/credential-proxy/testprovider/v1/check",
            headers={"Authorization": "Bearer client-should-not-see-this"},
            json={},
        )
        assert len(transport.requests) == 1
        assert transport.requests[0].headers["authorization"] == f"Bearer {_DEFAULT_TOKEN}"

    def test_forwards_target_url_correctly(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        client.get("/v1/credential-proxy/testprovider/v1/models/gpt-4")
        assert len(transport.requests) == 1
        url_str = str(transport.requests[0].url)
        assert "api.example.com" in url_str
        assert "v1/models/gpt-4" in url_str

    def test_get_method_forwarded(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        client.get("/v1/credential-proxy/testprovider/v1/info")
        assert transport.requests[0].method == "GET"

    def test_put_method_forwarded(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        client.put("/v1/credential-proxy/testprovider/v1/resource/1", json={})
        assert transport.requests[0].method == "PUT"

    def test_delete_method_forwarded(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(storage_root, transport=transport)
        client.delete("/v1/credential-proxy/testprovider/v1/resource/1")
        assert transport.requests[0].method == "DELETE"


# ---------------------------------------------------------------------------
# Provider not found
# ---------------------------------------------------------------------------


class TestCredentialProxyProviderNotFound:
    def test_unknown_provider_returns_404(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        resp = client.post("/v1/credential-proxy/unknownprovider/v1/op", json={})
        assert resp.status_code == 404

    def test_not_found_error_code(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        body = client.post("/v1/credential-proxy/ghost/v1/op", json={}).json()
        assert body["error"]["code"] == "credential_proxy_provider_not_found"

    def test_not_found_error_has_message(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        body = client.post("/v1/credential-proxy/ghost/v1/op", json={}).json()
        assert len(body["error"]["message"]) > 0

    def test_not_found_writes_audit(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.post("/v1/credential-proxy/ghost/v1/op", json={})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "credential.proxy.request.failed" for r in records)

    def test_not_found_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.post("/v1/credential-proxy/ghost/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["level"] == "error"

    def test_not_found_audit_code(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.post("/v1/credential-proxy/ghost/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["code"] == "credential_proxy_provider_not_found"

    def test_not_found_audit_detail_has_provider_name(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.post("/v1/credential-proxy/ghost/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["detail"]["provider_name"] == "ghost"

    def test_not_found_audit_detail_has_path(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.post("/v1/credential-proxy/ghost/v1/some/path", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["detail"]["path"] == "v1/some/path"

    def test_not_found_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.post("/v1/credential-proxy/ghost/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert len(record["detail"]["message"]) > 0


# ---------------------------------------------------------------------------
# Token unavailable
# ---------------------------------------------------------------------------


class TestCredentialProxyTokenUnavailable:
    def _noop_resolver(self) -> _FakeResolver:
        return _FakeResolver({})

    def test_missing_token_returns_502(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, resolver=self._noop_resolver())
        resp = client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        assert resp.status_code == 502

    def test_token_unavailable_error_code(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, resolver=self._noop_resolver())
        body = client.post("/v1/credential-proxy/testprovider/v1/op", json={}).json()
        assert body["error"]["code"] == "credential_proxy_token_unavailable"

    def test_token_unavailable_error_has_message(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, resolver=self._noop_resolver())
        body = client.post("/v1/credential-proxy/testprovider/v1/op", json={}).json()
        assert len(body["error"]["message"]) > 0

    def test_token_unavailable_writes_audit(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, resolver=self._noop_resolver())
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "credential.proxy.request.failed" for r in records)

    def test_token_unavailable_audit_code(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, resolver=self._noop_resolver())
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["code"] == "credential_proxy_token_unavailable"

    def test_token_unavailable_no_upstream_request(self, storage_root: Path) -> None:
        transport = _CapturingTransport()
        client = _make_router_client(
            storage_root,
            resolver=self._noop_resolver(),
            transport=transport,
        )
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        assert len(transport.requests) == 0


# ---------------------------------------------------------------------------
# Forward failure
# ---------------------------------------------------------------------------


class TestCredentialProxyForwardFailure:
    def _failing_transport(self) -> _CapturingTransport:
        return _CapturingTransport(raise_exc=ConnectionError("refused"))

    def test_network_error_returns_502(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        resp = client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        assert resp.status_code == 502

    def test_forward_failure_error_code(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        body = client.post("/v1/credential-proxy/testprovider/v1/op", json={}).json()
        assert body["error"]["code"] == "credential_proxy_forward_failed"

    def test_forward_failure_error_has_message(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        body = client.post("/v1/credential-proxy/testprovider/v1/op", json={}).json()
        assert len(body["error"]["message"]) > 0

    def test_forward_failure_writes_audit(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        records = _audit_records(storage_root)
        assert any(r.get("event") == "credential.proxy.request.failed" for r in records)

    def test_forward_failure_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["level"] == "error"

    def test_forward_failure_audit_code(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["code"] == "credential_proxy_forward_failed"

    def test_forward_failure_audit_detail_has_provider_name(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert record["detail"]["provider_name"] == "testprovider"

    def test_forward_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "credential.proxy.request.failed"
        )
        assert len(record["detail"]["message"]) > 0

    def test_token_not_in_error_response(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root, transport=self._failing_transport())
        resp = client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        assert _DEFAULT_TOKEN not in resp.text


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestCredentialProxyOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_span(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.get("/v1/credential-proxy/testprovider/v1/info")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "credential.proxy.request" in span_names

    def test_failure_emits_span(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.get("/v1/credential-proxy/ghost/v1/info")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "credential.proxy.request" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_router_client(storage_root)
        client.get("/v1/credential-proxy/ghost/v1/info")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("credential.proxy.request")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_span_has_provider_attribute(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.get("/v1/credential-proxy/testprovider/v1/info")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("credential.proxy.request")
        assert span is not None
        assert span.attributes["credential.proxy.provider"] == "testprovider"

    def test_span_has_path_attribute(self, storage_root: Path) -> None:
        client = _make_router_client(storage_root)
        client.get("/v1/credential-proxy/testprovider/v1/models")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("credential.proxy.request")
        assert span is not None
        assert span.attributes["credential.proxy.path"] == "v1/models"

    def test_token_unavailable_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_router_client(storage_root, resolver=_FakeResolver({}))
        client.get("/v1/credential-proxy/testprovider/v1/info")
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("credential.proxy.request")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_forward_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        transport = _CapturingTransport(raise_exc=ConnectionError("refused"))
        client = _make_router_client(storage_root, transport=transport)
        client.post("/v1/credential-proxy/testprovider/v1/op", json={})
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("credential.proxy.request")
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Route wiring via create_app
# ---------------------------------------------------------------------------


def _has_proxy_route(app: Any) -> bool:
    return any(hasattr(r, "path") and "credential-proxy" in r.path for r in app.routes)


class TestCredentialProxyRouteWiring:
    def test_route_present_with_providers_and_resolver(self, storage_root: Path) -> None:
        providers = [
            CredentialProxyProviderConfig(
                name="myprovider",
                base_url="http://example.com",
                token_secret_ref="ref://some/token",
            )
        ]
        resolver = _FakeResolver({"ref://some/token": "tok"})
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            secret_resolver=resolver,
            credential_proxy_providers=providers,
        )
        assert _has_proxy_route(app)

    def test_route_absent_without_providers(self, storage_root: Path) -> None:
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
        )
        assert not _has_proxy_route(app)

    def test_route_absent_when_providers_empty(self, storage_root: Path) -> None:
        resolver = _FakeResolver({})
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            secret_resolver=resolver,
            credential_proxy_providers=[],
        )
        assert not _has_proxy_route(app)

    def test_route_absent_without_secret_resolver(self, storage_root: Path) -> None:
        providers = [
            CredentialProxyProviderConfig(
                name="myprovider",
                base_url="http://example.com",
                token_secret_ref="ref://some/token",
            )
        ]
        app = create_app(
            FileAuditLog(storage_root),
            storage_root=storage_root,
            credential_proxy_providers=providers,
            # secret_resolver omitted
        )
        assert not _has_proxy_route(app)
