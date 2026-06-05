"""
In-harness credential proxy: network tools call a localhost endpoint on the
harness; the harness looks up the provider's OAuth token from the vault and
injects it as Authorization: Bearer <token> into the outbound request, so
workers never see the token even on stdin.

Route:
  ANY /v1/credential-proxy/{provider_name}/{path:path}

Per request:
  1. Opens an OTel span "credential.proxy.request" with provider/path attrs.
  2. Resolves the provider config; returns 404 if unknown.
  3. Resolves the OAuth token via SecretResolver; returns 502 if unavailable.
  4. Forwards the request to provider.base_url/{path} with Authorization injected.
  5. On forward failure: returns 502, writes audit entry, marks span ERROR.
  6. On success: streams the upstream response back to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter, Request
from fastapi.responses import Response
import httpx

from ._webhook_channel_driver import SecretResolver


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CredentialProxyProviderConfig:
    """Configuration for a single OAuth-protected provider exposed via the proxy."""

    name: str
    base_url: str
    token_secret_ref: str


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class CredentialProxyProviderNotFoundError(MeridianError):
    def __init__(self, *, provider_name: str, timestamp: str) -> None:
        super().__init__(
            code="credential_proxy_provider_not_found",
            message=f"Credential proxy provider '{provider_name}' is not configured",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class CredentialProxyTokenUnavailableError(MeridianError):
    def __init__(self, *, provider_name: str, timestamp: str) -> None:
        super().__init__(
            code="credential_proxy_token_unavailable",
            message=f"OAuth token for provider '{provider_name}' could not be resolved",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 502


class CredentialProxyForwardError(MeridianError):
    def __init__(
        self,
        *,
        provider_name: str,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="credential_proxy_forward_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 502


# ---------------------------------------------------------------------------
# Proxy router
# ---------------------------------------------------------------------------

_HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# Headers stripped from the inbound client request before forwarding upstream.
_REQUEST_STRIP_HEADERS = frozenset(
    {
        "host",
        "connection",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        "content-length",
        "authorization",  # replaced by vault-resolved token
    }
)

# Headers stripped from the upstream response before returning to the client.
_RESPONSE_STRIP_HEADERS = frozenset(
    {
        "connection",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "content-encoding",  # httpx decodes response bodies; header would be stale
        "content-length",  # recalculated by Starlette from response bytes
    }
)


def make_credential_proxy_router(
    *,
    audit_log: AuditLog,
    secret_resolver: SecretResolver,
    providers: list[CredentialProxyProviderConfig],
    http_client: httpx.AsyncClient | None = None,
) -> APIRouter:
    """
    Return a FastAPI router that proxies requests to configured OAuth providers.

    Workers call ``/v1/credential-proxy/{provider_name}/{path:path}``.  The
    harness resolves the provider's token from the vault and injects it as
    ``Authorization: Bearer <token>`` before forwarding.  The raw token is
    never returned to the caller.
    """
    router = APIRouter()
    _providers: dict[str, CredentialProxyProviderConfig] = {p.name: p for p in providers}

    @router.api_route(
        "/v1/credential-proxy/{provider_name}/{path:path}",
        methods=_HTTP_METHODS,
    )
    async def proxy_request(
        provider_name: str,
        path: str,
        request: Request,
    ) -> Response:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "credential.proxy.request",
            attributes={
                "credential.proxy.provider": provider_name,
                "credential.proxy.path": path,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="credential.proxy.request.invocation",
                    code="credential_proxy_request",
                    timestamp=now,
                ),
            )

            try:
                # 1. Resolve provider config
                config = _providers.get(provider_name)
                if config is None:
                    raise CredentialProxyProviderNotFoundError(
                        provider_name=provider_name,
                        timestamp=now,
                    )

                # 2. Resolve token — never expose it to the caller
                token = secret_resolver.resolve(config.token_secret_ref)
                if token is None:
                    raise CredentialProxyTokenUnavailableError(
                        provider_name=provider_name,
                        timestamp=now,
                    )

                # 3. Build target URL
                base = config.base_url.rstrip("/")
                target_url = f"{base}/{path}"
                if request.url.query:
                    target_url = f"{target_url}?{request.url.query}"

                # 4. Forward headers with injected Authorization
                forward_headers = {
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in _REQUEST_STRIP_HEADERS
                }
                forward_headers["Authorization"] = f"Bearer {token}"

                body = await request.body()

                # 5. Forward to upstream
                try:
                    if http_client is not None:
                        upstream = await http_client.request(
                            method=request.method,
                            url=target_url,
                            headers=forward_headers,
                            content=body,
                        )
                    else:
                        async with httpx.AsyncClient(timeout=60.0) as _client:
                            upstream = await _client.request(
                                method=request.method,
                                url=target_url,
                                headers=forward_headers,
                                content=body,
                            )
                except Exception as exc:
                    raise CredentialProxyForwardError(
                        provider_name=provider_name,
                        message=f"Failed to forward request to '{target_url}': {exc}",
                        timestamp=_now(),
                        cause=exc,
                    ) from exc

                # 6. Return upstream response (hop-by-hop headers stripped)
                response_headers = {
                    k: v
                    for k, v in upstream.headers.items()
                    if k.lower() not in _RESPONSE_STRIP_HEADERS
                }
                return Response(
                    content=upstream.content,
                    status_code=upstream.status_code,
                    headers=response_headers,
                )

            except (
                CredentialProxyProviderNotFoundError,
                CredentialProxyTokenUnavailableError,
                CredentialProxyForwardError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="credential.proxy.request.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "provider_name": provider_name,
                            "path": path,
                            "message": err.message,
                        },
                    )
                )
                raise

    return router
