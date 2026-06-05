from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
import uuid

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sdk_capabilities import Capability, CapabilityParseError, check_grant, parse_set


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class AcpOutboundDeniedError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="acp_outbound_denied", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 403


class AcpOutboundFailedError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="acp_outbound_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 502


class AcpInboundDeniedError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="acp_inbound_denied", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 403


class AcpInboundFailedError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="acp_inbound_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Peer transport (outbound)
# ---------------------------------------------------------------------------


@runtime_checkable
class AcpPeerClient(Protocol):
    """Transport protocol for delivering ACP messages to peer systems."""

    async def call(self, url: str, message: dict[str, Any]) -> dict[str, Any]: ...


class HttpAcpPeerClient:
    """Production transport: HTTP POST to the peer's ACP endpoint."""

    async def call(self, url: str, message: dict[str, Any]) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=message, timeout=30.0)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Inbound handler
# ---------------------------------------------------------------------------


@runtime_checkable
class AcpInboundHandler(Protocol):
    """Handler protocol for processing ACP messages received from peer systems."""

    async def handle(self, target: str, message: dict[str, Any]) -> dict[str, Any]: ...


class DefaultAcpInboundHandler:
    """Default handler: accepts all messages and returns an empty acknowledgement."""

    async def handle(self, target: str, message: dict[str, Any]) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AcpOutboundRequest(BaseModel):
    session_capabilities: list[str]
    target: str
    message: dict[str, Any]


class AcpOutboundTopLevelRequest(BaseModel):
    capabilities: list[str]
    target: str
    message: dict[str, Any]


class AcpInboundRequest(BaseModel):
    session_capabilities: list[str]
    target: str
    message: dict[str, Any]


class AcpInboundTopLevelRequest(BaseModel):
    capabilities: list[str]
    target: str
    message: dict[str, Any]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_acp_router(
    *,
    audit_log: AuditLog,
    targets: dict[str, str],
    peer_client: AcpPeerClient | None = None,
    inbound_handler: AcpInboundHandler | None = None,
) -> APIRouter:
    _client: AcpPeerClient = peer_client if peer_client is not None else HttpAcpPeerClient()
    _inbound: AcpInboundHandler = (
        inbound_handler if inbound_handler is not None else DefaultAcpInboundHandler()
    )
    router = APIRouter()

    @router.post("/v1/x/acp/outbound")
    async def acp_outbound_toplevel(body: AcpOutboundTopLevelRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        call_id = str(uuid.uuid4())
        target = body.target

        with tracer.start_as_current_span(
            "acp.outbound",
            attributes={
                "acp.target": target,
                "acp.call_id": call_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="acp.outbound.invocation",
                    code="acp_outbound",
                    timestamp=now,
                ),
            )

            try:
                granted_caps = parse_set(body.capabilities)
            except CapabilityParseError as exc:
                err = AcpOutboundDeniedError(
                    message=f"Invalid capability string: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

            required_cap = Capability(namespace="acp", name="outbound", param=target)
            if not check_grant(frozenset({required_cap}), granted_caps):
                err = AcpOutboundDeniedError(
                    message=f"ACP outbound denied: does not hold acp.outbound[{target}]",
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            target_url = targets.get(target)
            if target_url is None:
                err = AcpOutboundDeniedError(
                    message=f"ACP outbound denied: target {target!r} not registered",
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            try:
                peer_response = await _client.call(target_url, body.message)
            except Exception as exc:
                err = AcpOutboundFailedError(
                    message=f"ACP outbound call to {target!r} failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "call_id": call_id,
                "target": target,
                "status": "delivered",
                "response": peer_response,
            }
        )

    @router.post("/v1/x/sessions/{session_id}/acp/outbound")
    async def acp_outbound(session_id: str, body: AcpOutboundRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        call_id = str(uuid.uuid4())
        target = body.target

        with tracer.start_as_current_span(
            "acp.outbound",
            attributes={
                "session.id": session_id,
                "acp.target": target,
                "acp.call_id": call_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="acp.outbound.invocation",
                    code="acp_outbound",
                    timestamp=now,
                ),
            )

            # Parse session capabilities
            try:
                granted_caps = parse_set(body.session_capabilities)
            except CapabilityParseError as exc:
                err = AcpOutboundDeniedError(
                    message=f"Invalid capability string for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

            # Capability gate: session must hold acp.outbound[target]
            required_cap = Capability(namespace="acp", name="outbound", param=target)
            if not check_grant(frozenset({required_cap}), granted_caps):
                err = AcpOutboundDeniedError(
                    message=(
                        f"ACP outbound denied for session {session_id!r}: "
                        f"does not hold acp.outbound[{target}]"
                    ),
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            # Target URL resolution
            target_url = targets.get(target)
            if target_url is None:
                err = AcpOutboundDeniedError(
                    message=(
                        f"ACP outbound denied for session {session_id!r}: "
                        f"target {target!r} not registered"
                    ),
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            # Transport call to peer system
            try:
                peer_response = await _client.call(target_url, body.message)
            except Exception as exc:
                err = AcpOutboundFailedError(
                    message=(
                        f"ACP outbound call to {target!r} failed for session {session_id!r}: {exc}"
                    ),
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.outbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "call_id": call_id,
                "session_id": session_id,
                "target": target,
                "status": "delivered",
                "response": peer_response,
            }
        )

    @router.post("/v1/x/acp/inbound")
    async def acp_inbound_toplevel(body: AcpInboundTopLevelRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        call_id = str(uuid.uuid4())
        target = body.target

        with tracer.start_as_current_span(
            "acp.inbound",
            attributes={
                "acp.target": target,
                "acp.call_id": call_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="acp.inbound.invocation",
                    code="acp_inbound",
                    timestamp=now,
                ),
            )

            try:
                granted_caps = parse_set(body.capabilities)
            except CapabilityParseError as exc:
                err = AcpInboundDeniedError(
                    message=f"Invalid capability string: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

            required_cap = Capability(namespace="acp", name="inbound", param=target)
            if not check_grant(frozenset({required_cap}), granted_caps):
                err = AcpInboundDeniedError(
                    message=f"ACP inbound denied: does not hold acp.inbound[{target}]",
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            if target not in targets:
                err = AcpInboundDeniedError(
                    message=f"ACP inbound denied: target {target!r} not registered",
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            try:
                await _inbound.handle(target, body.message)
            except Exception as exc:
                err = AcpInboundFailedError(
                    message=f"ACP inbound processing for target {target!r} failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "call_id": call_id,
                "target": target,
                "status": "accepted",
            }
        )

    @router.post("/v1/x/sessions/{session_id}/acp/inbound")
    async def acp_inbound(session_id: str, body: AcpInboundRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        call_id = str(uuid.uuid4())
        target = body.target

        with tracer.start_as_current_span(
            "acp.inbound",
            attributes={
                "session.id": session_id,
                "acp.target": target,
                "acp.call_id": call_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="acp.inbound.invocation",
                    code="acp_inbound",
                    timestamp=now,
                ),
            )

            # Parse session capabilities
            try:
                granted_caps = parse_set(body.session_capabilities)
            except CapabilityParseError as exc:
                err = AcpInboundDeniedError(
                    message=f"Invalid capability string for session {session_id!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

            # Capability gate: must hold acp.inbound[target]
            required_cap = Capability(namespace="acp", name="inbound", param=target)
            if not check_grant(frozenset({required_cap}), granted_caps):
                err = AcpInboundDeniedError(
                    message=(
                        f"ACP inbound denied for session {session_id!r}: "
                        f"does not hold acp.inbound[{target}]"
                    ),
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            # Target registration check
            if target not in targets:
                err = AcpInboundDeniedError(
                    message=(
                        f"ACP inbound denied for session {session_id!r}: "
                        f"target {target!r} not registered"
                    ),
                    timestamp=_now(),
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.denied",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err

            # Process the inbound message
            try:
                await _inbound.handle(target, body.message)
            except Exception as exc:
                err = AcpInboundFailedError(
                    message=(
                        f"ACP inbound processing for target {target!r} failed "
                        f"for session {session_id!r}: {exc}"
                    ),
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="acp.inbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "session_id": session_id,
                            "target": target,
                            "call_id": call_id,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

        return JSONResponse(
            content={
                "call_id": call_id,
                "session_id": session_id,
                "target": target,
                "status": "accepted",
            }
        )

    return router
