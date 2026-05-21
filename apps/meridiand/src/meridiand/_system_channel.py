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
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# PRD §6.1: channel inbound → harness wake must be < 1 s p95 end-to-end.
_INBOUND_LATENCY_TARGET_MS = 1000.0

# Quarantine sessions expire after this many minutes of silence.
_QUARANTINE_SILENCE_TIMEOUT_MINUTES = 15

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

from sdk_sandbox import ExecutionContext

from ._hook_dispatch import dispatch_hooks
from ._metrics_registry import channel_inbound_total, channel_outbound_total
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


class SessionOutboundNotFoundError(MeridianError):
    def __init__(self, *, session_id: str, timestamp: str) -> None:
        super().__init__(
            code="session_outbound_not_found",
            message=f"No channels attached to session '{session_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class SessionOutboundError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="session_outbound_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class ChannelRemoteNotFoundError(MeridianError):
    def __init__(self, *, channel_id: str, timestamp: str) -> None:
        super().__init__(
            code="channel_remote_not_found",
            message=f"Channel '{channel_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class ChannelPairingNotFoundError(MeridianError):
    def __init__(self, *, channel_id: str, remote_id: str, timestamp: str) -> None:
        super().__init__(
            code="channel_pairing_not_found",
            message=f"No pairing found for remote '{remote_id}' on channel '{channel_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


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


class SessionOutboundRequest(BaseModel):
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
    hooks_dir = storage_root / "hooks"
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
    # GET /v1/channels/{channel_id}/remote/{remote_id}
    # -----------------------------------------------------------------------

    @router.get("/v1/channels/{channel_id}/remote/{remote_id}", status_code=200)
    async def resolve_channel_pairing(channel_id: str, remote_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "channel.pairing.resolve",
            attributes={"channel.id": channel_id, "channel.remote_id": remote_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.pairing.resolve.invocation",
                    code="channel_pairing_resolve",
                    timestamp=now,
                ),
            )

            try:
                channel_file = channels_dir / f"{channel_id}.json"
                if not channel_file.exists():
                    raise ChannelRemoteNotFoundError(channel_id=channel_id, timestamp=now)

                pairing_file = channel_pairings_dir / channel_id / f"{remote_id}.json"
                if not pairing_file.exists():
                    raise ChannelPairingNotFoundError(
                        channel_id=channel_id, remote_id=remote_id, timestamp=now
                    )

                pairing: dict[str, Any] = json.loads(pairing_file.read_text())
                user_profile_id: str | None = pairing.get("user_profile_id")

            except (ChannelRemoteNotFoundError, ChannelPairingNotFoundError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.pairing.resolve.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"channel_id": channel_id, "remote_id": remote_id, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = ChannelInboundError(
                    message=f"Pairing resolve failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="channel.pairing.resolve.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"channel_id": channel_id, "remote_id": remote_id, "message": err2.message},
                    )
                )
                raise err2

        return JSONResponse(
            content={
                "channel_id": channel_id,
                "remote_id": remote_id,
                "user_profile_id": user_profile_id,
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
            t0 = time.perf_counter()
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

                # HMAC verification for any channel kind with hmac_secret_ref configured.
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

                await dispatch_hooks(
                    "pre_message",
                    {
                        "channel_id": channel_id,
                        "sender_id": body.sender_id,
                        "content": body.content,
                        "content_type": body.content_type,
                    },
                    ExecutionContext(session_id=""),
                    hooks_dir=hooks_dir,
                    audit_log=audit_log,
                )

                inbound_policy: str = channel.get("inbound_policy", "open")

                user_profile_id: str | None = None
                quarantined = False
                quarantine_id: str | None = None
                caps_envelope: dict[str, Any] | None = None
                expires_at: str | None = None
                pairing_file = channel_pairings_dir / channel_id / f"{body.sender_id}.json"

                if pairing_file.exists():
                    # Known sender — use their linked UserProfile.
                    pairing: dict[str, Any] = json.loads(pairing_file.read_text())
                    user_profile_id = pairing.get("user_profile_id")
                elif inbound_policy == "paired_only":
                    raise ChannelInboundPolicyRejectedError(
                        channel_id=channel_id,
                        sender_id=body.sender_id,
                        timestamp=now,
                    )
                elif inbound_policy == "quarantine":
                    # Route to the per-channel quarantine UserProfile (minimal caps).
                    q_dir = channel_quarantine_dir / channel_id
                    q_dir.mkdir(parents=True, exist_ok=True)
                    qp_file = q_dir / "_profile.json"
                    if qp_file.exists():
                        qp: dict[str, Any] = json.loads(qp_file.read_text())
                        user_profile_id = qp["id"]
                    else:
                        qp_id = f"qup_{uuid.uuid4().hex}"
                        qp = {
                            "id": qp_id,
                            "channel_id": channel_id,
                            "username": f"quarantine_{channel_id}",
                            "metadata": json.dumps({"capabilities": ["minimal"]}),
                            "created_at": now,
                        }
                        qp_file.write_text(json.dumps(qp))
                        user_profile_id = qp_id

                    # Minimal caps envelope: fs.read on dedicated sandbox dir only,
                    # no exec.*, no net.fetch.
                    sandbox_dir = f"channel_quarantine/{channel_id}"
                    caps_envelope = {
                        "can_exec_subprocesses": False,
                        "can_write_filesystem": False,
                        "network": {"egress_allowed": False},
                        "filesystem": {"read_globs": [f"{sandbox_dir}/**"]},
                    }
                    expires_at = (
                        datetime.now(UTC)
                        + timedelta(minutes=_QUARANTINE_SILENCE_TIMEOUT_MINUTES)
                    ).isoformat()

                    quarantine_id = f"quar_{uuid.uuid4().hex}"
                    q_record: dict[str, Any] = {
                        "id": quarantine_id,
                        "channel_id": channel_id,
                        "sender_id": body.sender_id,
                        "content": body.content,
                        "content_type": body.content_type,
                        "quarantined_at": now,
                    }

                    with tracer.start_as_current_span(
                        "channel.inbound.quarantine",
                        attributes={
                            "channel.id": channel_id,
                            "channel.sender_id": body.sender_id,
                            "quarantine.sandbox_dir": sandbox_dir,
                            "quarantine.expires_at": expires_at,
                        },
                    ) as q_span:
                        record_invocation_event(
                            q_span,
                            StructuredEvent(
                                name="channel.inbound.quarantine.invocation",
                                code="channel_inbound_quarantine",
                                timestamp=now,
                            ),
                        )
                        (q_dir / f"{quarantine_id}.json").write_text(json.dumps(q_record))

                    quarantined = True
                    span.set_attribute("channel.inbound.quarantined", True)
                    span.set_attribute("channel.inbound.quarantine.expires_at", expires_at)
                else:
                    # open policy — auto-create a UserProfile for this sender so their
                    # identity is stable across sessions.
                    new_profile_id = f"up_{uuid.uuid4().hex}"
                    pairing_dir = channel_pairings_dir / channel_id
                    pairing_dir.mkdir(parents=True, exist_ok=True)
                    auto_pairing: dict[str, Any] = {
                        "sender_id": body.sender_id,
                        "user_profile_id": new_profile_id,
                        "channel_id": channel_id,
                        "created_at": now,
                        "auto_created": True,
                    }
                    (pairing_dir / f"{body.sender_id}.json").write_text(json.dumps(auto_pairing))
                    user_profile_id = new_profile_id

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
                if caps_envelope is not None:
                    session_record["caps_envelope"] = caps_envelope
                if expires_at is not None:
                    session_record["expires_at"] = expires_at
                (s_dir / f"{session_id}.json").write_text(json.dumps(session_record))

                await dispatch_hooks(
                    "post_message",
                    {
                        "channel_id": channel_id,
                        "sender_id": body.sender_id,
                        "session_id": session_id,
                        "user_profile_id": user_profile_id,
                        "agent_id": agent_id,
                    },
                    ExecutionContext(session_id=session_id),
                    hooks_dir=hooks_dir,
                    audit_log=audit_log,
                )

                # Fan out: register this session on every other channel where the same
                # UserProfile is paired, so the Session is reachable from all paired channels.
                if user_profile_id and channel_pairings_dir.exists():
                    for ch_dir in channel_pairings_dir.iterdir():
                        if not ch_dir.is_dir() or ch_dir.name == channel_id:
                            continue
                        for pairing_f in ch_dir.glob("*.json"):
                            p: dict[str, Any] = json.loads(pairing_f.read_text())
                            if p.get("user_profile_id") == user_profile_id:
                                other_ch_id: str = ch_dir.name
                                other_s_dir = channel_sessions_dir / other_ch_id
                                other_s_dir.mkdir(parents=True, exist_ok=True)
                                other_sess: dict[str, Any] = {
                                    "id": session_id,
                                    "channel_id": other_ch_id,
                                    "user_profile_id": user_profile_id,
                                    "agent_id": agent_id,
                                    "sender_id": p["sender_id"],
                                    "created_at": now,
                                }
                                (other_s_dir / f"{session_id}.json").write_text(
                                    json.dumps(other_sess)
                                )

                channel_inbound_total.labels(kind=channel.get("kind", "unknown")).inc()

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

            finally:
                latency_ms = (time.perf_counter() - t0) * 1000
                span.set_attribute("channel.inbound.latency_ms", latency_ms)
                if latency_ms > _INBOUND_LATENCY_TARGET_MS:
                    span.set_attribute("channel.inbound.latency_target_exceeded", True)

        await dispatch_hooks(
            "on_channel_inbound",
            {
                "channel_id": channel_id,
                "sender_id": body.sender_id,
                "session_id": session_id,
                "user_profile_id": user_profile_id,
                "quarantined": quarantined,
            },
            ExecutionContext(session_id=session_id),
            hooks_dir=hooks_dir,
            audit_log=audit_log,
        )

        response_body: dict[str, Any] = {
            "user_profile_id": user_profile_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "quarantined": quarantined,
        }
        if quarantine_id is not None:
            response_body["quarantine_id"] = quarantine_id
        return JSONResponse(content=response_body, status_code=200)

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

                channel_outbound_total.labels(kind=channel["kind"]).inc()

                await dispatch_hooks(
                    "on_channel_outbound",
                    {
                        "channel_id": channel_id,
                        "session_id": body.session_id,
                        "recipient": body.recipient,
                        "content_type": body.content_type,
                        "message_id": result.message_id,
                        "delivered": result.delivered,
                    },
                    ExecutionContext(session_id=body.session_id),
                    hooks_dir=hooks_dir,
                    audit_log=audit_log,
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

    # -----------------------------------------------------------------------
    # POST /v1/sessions/{session_id}/outbound
    # -----------------------------------------------------------------------

    @router.post("/v1/sessions/{session_id}/outbound", status_code=200)
    async def session_outbound(
        session_id: str, body: SessionOutboundRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "session.outbound",
            attributes={"session.id": session_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="session.outbound.invocation",
                    code="session_outbound",
                    timestamp=now,
                ),
            )

            try:
                # Scan all channel_sessions dirs to find channels attached to this session.
                session_records: list[dict[str, Any]] = []
                if channel_sessions_dir.exists():
                    for channel_dir in channel_sessions_dir.iterdir():
                        if channel_dir.is_dir():
                            sess_file = channel_dir / f"{session_id}.json"
                            if sess_file.exists():
                                session_records.append(json.loads(sess_file.read_text()))

                if not session_records:
                    raise SessionOutboundNotFoundError(session_id=session_id, timestamp=now)

                span.set_attribute("session.channel_count", len(session_records))

                results: list[dict[str, Any]] = []
                channels_skipped = 0

                for sess_rec in session_records:
                    ch_id: str = sess_rec["channel_id"]
                    recipient: str = sess_rec["sender_id"]

                    channel_file = channels_dir / f"{ch_id}.json"
                    if not channel_file.exists():
                        channels_skipped += 1
                        continue

                    channel = json.loads(channel_file.read_text())

                    if channel.get("egress_policy") == "disabled":
                        channels_skipped += 1
                        continue

                    send_request = SendRequest(
                        channel_id=ch_id,
                        channel_kind=channel["kind"],
                        session_id=session_id,
                        recipient=recipient,
                        content=body.content,
                        content_type=body.content_type,
                    )

                    try:
                        result = await channel_runtime.send(
                            send_request,
                            RuntimeOptions(audit_log=NoopAuditLog()),
                        )
                        channel_outbound_total.labels(kind=channel["kind"]).inc()
                        results.append(
                            {
                                "channel_id": ch_id,
                                "message_id": result.message_id,
                                "delivered": result.delivered,
                                "timestamp": result.timestamp,
                            }
                        )
                    except ChannelFailure as cf:
                        send_err = SessionOutboundError(
                            message=cf.message,
                            timestamp=now,
                            cause=cf,
                        )
                        record_error(span, send_err)
                        audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="session.outbound.failed",
                                code=send_err.code,
                                timestamp=send_err.timestamp,
                                detail={
                                    "session_id": session_id,
                                    "channel_id": ch_id,
                                    "channel_kind": cf.channel_kind,
                                    "message": send_err.message,
                                },
                            )
                        )
                        results.append(
                            {"channel_id": ch_id, "delivered": False, "error": send_err.code}
                        )
                    except Exception as exc:
                        send_err = SessionOutboundError(
                            message=f"Outbound delivery failed: {exc}",
                            timestamp=_now(),
                            cause=exc,
                        )
                        record_error(span, send_err)
                        audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="session.outbound.failed",
                                code=send_err.code,
                                timestamp=send_err.timestamp,
                                detail={
                                    "session_id": session_id,
                                    "channel_id": ch_id,
                                    "message": send_err.message,
                                },
                            )
                        )
                        results.append(
                            {"channel_id": ch_id, "delivered": False, "error": send_err.code}
                        )

            except SessionOutboundNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.outbound.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"session_id": session_id, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = SessionOutboundError(
                    message=f"Session outbound failed: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="session.outbound.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"session_id": session_id, "message": err2.message},
                    )
                )
                raise err2

        return JSONResponse(
            content={
                "session_id": session_id,
                "results": results,
                "channels_skipped": channels_skipped,
            },
            status_code=200,
        )

    return router
