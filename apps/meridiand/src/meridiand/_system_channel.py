"""
System Channel: resolves inbound messages to (UserProfile, Agent, Session)
and routes outbound messages through a registered ChannelDriver.

Endpoints:
  POST /v1/pairing_tokens/{token}/redeem   - Link a sender_id to the UserProfile
                                             recorded in a pairing token.
  POST /v1/channels/{channel_id}/inbound   - Accept an inbound message and resolve
                                             the routing context.
  POST /v1/channels/{channel_id}/outbound  - Send a message via the registered
                                             ChannelDriver and confirm delivery.

Instrumentation:
  Every endpoint opens an OTel span and attaches a structured invocation event.
  On failure the span is marked ERROR and an audit-log entry is written before
  the error response is returned to the caller.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sdk_channel import (
    ChannelFailure,
    ChannelRuntime,
    NoopAuditLog,
    RuntimeOptions,
    SendRequest,
)

from ._webhook_channel_driver import SecretResolver, NoopSecretResolver, _sign_payload


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class PairingTokenNotFoundError(MeridianError):
    def __init__(self, *, token: str, timestamp: str) -> None:
        super().__init__(
            code="pairing_token_not_found",
            message=f"Pairing token '{token}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class PairingTokenAlreadyRedeemedError(MeridianError):
    def __init__(self, *, token: str, timestamp: str) -> None:
        super().__init__(
            code="pairing_token_already_redeemed",
            message=f"Pairing token '{token}' has already been redeemed",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


class ChannelInboundNotFoundError(MeridianError):
    def __init__(self, *, channel_id: str, timestamp: str) -> None:
        super().__init__(
            code="channel_inbound_not_found",
            message=f"Channel '{channel_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class ChannelInboundPolicyRejectedError(MeridianError):
    def __init__(self, *, channel_id: str, sender_id: str, timestamp: str) -> None:
        super().__init__(
            code="channel_inbound_policy_rejected",
            message=f"Sender '{sender_id}' is not paired with channel '{channel_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 403


class ChannelInboundError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="channel_inbound_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class ChannelOutboundNotFoundError(MeridianError):
    def __init__(self, *, channel_id: str, timestamp: str) -> None:
        super().__init__(
            code="channel_outbound_not_found",
            message=f"Channel '{channel_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class ChannelOutboundDisabledError(MeridianError):
    def __init__(self, *, channel_id: str, timestamp: str) -> None:
        super().__init__(
            code="channel_outbound_disabled",
            message=f"Egress is disabled for channel '{channel_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 422


class ChannelOutboundError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="channel_outbound_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class ChannelInboundHmacError(MeridianError):
    def __init__(self, *, channel_id: str, timestamp: str) -> None:
        super().__init__(
            code="channel_inbound_hmac_invalid",
            message=f"HMAC signature verification failed for channel '{channel_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 401


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


def _check_hmac_signature(raw_body: bytes, secret: str, signature_header: str | None) -> bool:
    """Return True if signature_header is a valid sha256 HMAC of raw_body."""
    if signature_header is None:
        return False
    prefix = "sha256="
    if not signature_header.startswith(prefix):
        return False
    provided = signature_header[len(prefix):]
    expected = _sign_payload(raw_body, secret)
    return hmac.compare_digest(provided.encode(), expected.encode())


class RedeemPairingTokenRequest(BaseModel):
    sender_id: str


class InboundMessageRequest(BaseModel):
    sender_id: str
    content: str
    content_type: str = "text/plain"


class OutboundMessageRequest(BaseModel):
    session_id: str
    recipient: str
    content: str
    content_type: str = "text/plain"


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_system_channel_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    channel_runtime: ChannelRuntime,
    secret_resolver: SecretResolver | None = None,
) -> APIRouter:
    router = APIRouter()
    _resolver: SecretResolver = secret_resolver if secret_resolver is not None else NoopSecretResolver()

    channels_dir = storage_root / "channels"
    pairing_tokens_dir = storage_root / "pairing_tokens"
    channel_pairings_dir = storage_root / "channel_pairings"
    channel_sessions_dir = storage_root / "channel_sessions"
    channel_quarantine_dir = storage_root / "channel_quarantine"

    # -----------------------------------------------------------------------
    # POST /v1/pairing_tokens/{token}/redeem
    # -----------------------------------------------------------------------

    @router.post("/v1/pairing_tokens/{token}/redeem", status_code=200)
    async def redeem_pairing_token(
        token: str, body: RedeemPairingTokenRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "channel.pair.redeem",
            attributes={"pairing.token": token, "channel.sender_id": body.sender_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.pair.redeem.invocation",
                    code="channel_pair_redeem",
                    timestamp=now,
                ),
            )

            try:
                token_file = pairing_tokens_dir / f"{token}.json"
                if not token_file.exists():
                    raise PairingTokenNotFoundError(token=token, timestamp=now)

                token_record: dict[str, Any] = json.loads(token_file.read_text())
                if token_record.get("redeemed"):
                    raise PairingTokenAlreadyRedeemedError(token=token, timestamp=now)

                channel_id: str = token_record["channel_id"]
                user_profile_id: str | None = token_record.get("user_profile_id")

                pairing_dir = channel_pairings_dir / channel_id
                pairing_dir.mkdir(parents=True, exist_ok=True)
                pairing_record: dict[str, Any] = {
                    "sender_id": body.sender_id,
                    "user_profile_id": user_profile_id,
                    "channel_id": channel_id,
                    "token": token,
                    "created_at": now,
                }
                (pairing_dir / f"{body.sender_id}.json").write_text(
                    json.dumps(pairing_record)
                )

                token_record["redeemed"] = True
                token_record["redeemed_at"] = now
                token_record["sender_id"] = body.sender_id
                token_file.write_text(json.dumps(token_record))

            except (PairingTokenNotFoundError, PairingTokenAlreadyRedeemedError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.pair.redeem.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"token": token, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = ChannelInboundError(
                    message=f"Failed to redeem pairing token: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.pair.redeem.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"token": token, "message": err2.message},
                    )
                )
                raise err2

        return JSONResponse(
            content={
                "token": token,
                "channel_id": channel_id,
                "sender_id": body.sender_id,
                "user_profile_id": user_profile_id,
                "redeemed_at": now,
            },
            status_code=200,
        )

    # -----------------------------------------------------------------------
    # POST /v1/channels/{channel_id}/inbound
    # -----------------------------------------------------------------------

    @router.post("/v1/channels/{channel_id}/inbound", status_code=200)
    async def channel_inbound(
        channel_id: str, body: InboundMessageRequest, request: Request
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "channel.inbound",
            attributes={
                "channel.id": channel_id,
                "channel.sender_id": body.sender_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.inbound.invocation",
                    code="channel_inbound",
                    timestamp=now,
                ),
            )

            try:
                channel_file = channels_dir / f"{channel_id}.json"
                if not channel_file.exists():
                    raise ChannelInboundNotFoundError(channel_id=channel_id, timestamp=now)

                channel: dict[str, Any] = json.loads(channel_file.read_text())

                # HMAC verification for webhook channels with hmac_secret_ref configured.
                if channel.get("kind") == "meridian.webhook":
                    hmac_secret_ref: str | None = channel.get("config", {}).get("hmac_secret_ref")
                    if hmac_secret_ref is not None:
                        secret = _resolver.resolve(hmac_secret_ref)
                        if secret is not None:
                            raw_body = await request.body()
                            sig_header = request.headers.get("X-Meridian-Signature")
                            if not _check_hmac_signature(raw_body, secret, sig_header):
                                raise ChannelInboundHmacError(
                                    channel_id=channel_id, timestamp=now
                                )

                inbound_policy: str = channel.get("inbound_policy", "open")

                user_profile_id: str | None = None
                pairing_file = channel_pairings_dir / channel_id / f"{body.sender_id}.json"

                if pairing_file.exists():
                    pairing: dict[str, Any] = json.loads(pairing_file.read_text())
                    user_profile_id = pairing.get("user_profile_id")
                elif inbound_policy == "quarantine":
                    quarantine_id = f"quar_{uuid.uuid4().hex}"
                    q_dir = channel_quarantine_dir / channel_id
                    q_dir.mkdir(parents=True, exist_ok=True)
                    q_record: dict[str, Any] = {
                        "id": quarantine_id,
                        "channel_id": channel_id,
                        "sender_id": body.sender_id,
                        "content": body.content,
                        "content_type": body.content_type,
                        "quarantined_at": now,
                    }
                    (q_dir / f"{quarantine_id}.json").write_text(json.dumps(q_record))
                    return JSONResponse(
                        content={"quarantined": True, "quarantine_id": quarantine_id},
                        status_code=200,
                    )
                elif inbound_policy == "paired_only":
                    raise ChannelInboundPolicyRejectedError(
                        channel_id=channel_id,
                        sender_id=body.sender_id,
                        timestamp=now,
                    )
                else:
                    user_profile_id = channel.get("default_user_profile_id")

                agent_id: str | None = channel.get("default_agent_id")

                session_id = f"sess_{uuid.uuid4().hex}"
                s_dir = channel_sessions_dir / channel_id
                s_dir.mkdir(parents=True, exist_ok=True)
                session_record: dict[str, Any] = {
                    "id": session_id,
                    "channel_id": channel_id,
                    "user_profile_id": user_profile_id,
                    "agent_id": agent_id,
                    "sender_id": body.sender_id,
                    "created_at": now,
                }
                (s_dir / f"{session_id}.json").write_text(json.dumps(session_record))

            except (ChannelInboundNotFoundError, ChannelInboundHmacError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.inbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"channel_id": channel_id, "message": err.message},
                    )
                )
                raise

            except ChannelInboundPolicyRejectedError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.inbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "channel_id": channel_id,
                            "sender_id": body.sender_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = ChannelInboundError(
                    message=f"Inbound routing failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.inbound.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"channel_id": channel_id, "message": err2.message},
                    )
                )
                raise err2

        return JSONResponse(
            content={
                "user_profile_id": user_profile_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "quarantined": False,
            },
            status_code=200,
        )

    # -----------------------------------------------------------------------
    # POST /v1/channels/{channel_id}/outbound
    # -----------------------------------------------------------------------

    @router.post("/v1/channels/{channel_id}/outbound", status_code=200)
    async def channel_outbound(
        channel_id: str, body: OutboundMessageRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "channel.outbound",
            attributes={
                "channel.id": channel_id,
                "session.id": body.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.outbound.invocation",
                    code="channel_outbound",
                    timestamp=now,
                ),
            )

            try:
                channel_file = channels_dir / f"{channel_id}.json"
                if not channel_file.exists():
                    raise ChannelOutboundNotFoundError(channel_id=channel_id, timestamp=now)

                channel = json.loads(channel_file.read_text())

                if channel.get("egress_policy") == "disabled":
                    raise ChannelOutboundDisabledError(channel_id=channel_id, timestamp=now)

                send_request = SendRequest(
                    channel_id=channel_id,
                    channel_kind=channel["kind"],
                    session_id=body.session_id,
                    recipient=body.recipient,
                    content=body.content,
                    content_type=body.content_type,
                )

                result = await channel_runtime.send(
                    send_request,
                    RuntimeOptions(audit_log=NoopAuditLog()),
                )

            except (
                ChannelOutboundNotFoundError,
                ChannelOutboundDisabledError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.outbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"channel_id": channel_id, "message": err.message},
                    )
                )
                raise

            except ChannelFailure as cf:
                err2 = ChannelOutboundError(
                    message=cf.message,
                    timestamp=now,
                    cause=cf,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.outbound.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "channel_id": channel_id,
                            "channel_kind": cf.channel_kind,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from cf

            except Exception as exc:
                err3 = ChannelOutboundError(
                    message=f"Outbound delivery failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err3)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.outbound.failed",
                        code=err3.code,
                        timestamp=err3.timestamp,
                        detail={"channel_id": channel_id, "message": err3.message},
                    )
                )
                raise err3

        return JSONResponse(
            content={
                "message_id": result.message_id,
                "delivered": result.delivered,
                "timestamp": result.timestamp,
            },
            status_code=200,
        )

    return router
