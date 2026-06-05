"""Webhook sender: delivers filtered session events to registered webhook endpoints.

Scans all active webhooks, reads undelivered session events matching each
webhook's event_filter (types + optional session_id), and POSTs them to the
configured URL.  Each request is signed with HMAC-SHA256 when a secret is
available.  Transient failures are retried with exponential or linear backoff;
permanent failures (all retries exhausted or non-retryable HTTP status) are
written to the dead-letter queue.

Emits an OpenTelemetry span and writes a structured audit-log entry on every
delivery attempt.  On failure, the error is surfaced to the caller and written
to the audit log before re-raising.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
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
import httpx
from storage_event_log import SessionEvent
from storage_reposit import LocalEventLogReader


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Secret resolution
# ---------------------------------------------------------------------------


@runtime_checkable
class SecretResolver(Protocol):
    """Resolves a secret_ref string to the raw secret value, or None if unavailable."""

    def resolve(self, secret_ref: str) -> str | None: ...


class NoopSecretResolver:
    """Always returns None — no signing performed."""

    def resolve(self, secret_ref: str) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class WebhookDeliveryError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="webhook_delivery_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# HMAC-SHA256 signing
# ---------------------------------------------------------------------------


def _sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Return the HMAC-SHA256 hex digest of payload_bytes keyed by secret."""
    mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
    return mac.hexdigest()


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def _backoff_delay(backoff_type: str, attempt: int, *, base_seconds: float = 1.0) -> float:
    """Return delay in seconds before retry attempt N (1-indexed).

    exponential: 1s, 2s, 4s, 8s, …
    linear:      1s, 2s, 3s, 4s, …
    """
    if backoff_type == "exponential":
        return base_seconds * (2 ** (attempt - 1))
    return base_seconds * attempt


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def _discover_session_ids(storage_root: Path) -> list[str]:
    """Return all unique session IDs found in the event log directory."""
    events_dir = storage_root / "events"
    if not events_dir.exists():
        return []
    return list({f.stem for f in events_dir.rglob("*.ndjson")})


# ---------------------------------------------------------------------------
# Watermark persistence
# ---------------------------------------------------------------------------


def _load_watermarks(watermark_file: Path) -> dict[str, int]:
    if not watermark_file.exists():
        return {}
    try:
        return json.loads(watermark_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_watermarks(watermark_file: Path, watermarks: dict[str, int]) -> None:
    watermark_file.parent.mkdir(parents=True, exist_ok=True)
    watermark_file.write_text(json.dumps(watermarks))


# ---------------------------------------------------------------------------
# Single-event delivery (with retry + DLQ)
# ---------------------------------------------------------------------------


async def deliver_webhook_event(
    webhook: dict[str, Any],
    event: SessionEvent,
    session_id: str,
    *,
    client: httpx.AsyncClient,
    secret_resolver: SecretResolver,
    audit_log: AuditLog,
    dlq_dir: Path,
) -> None:
    """Deliver one session event to a webhook URL.

    Retries on transient failures (5xx, network errors) using the webhook's
    backoff policy.  Permanent failures (retries exhausted or 4xx) are written
    to the dead-letter queue under dlq_dir/{webhook_id}/{delivery_id}.json.

    Always emits an OTel span "webhook.sender.deliver" and writes an audit
    entry.  Raises WebhookDeliveryError on any failure after logging.
    """
    webhook_id: str = webhook["id"]
    url: str = webhook["url"]
    max_retries: int = webhook["max_retries"]
    backoff_type: str = webhook["backoff"]
    secret_ref: str | None = webhook.get("secret_ref")

    delivery_id = f"delivery_{uuid.uuid4().hex}"
    now = _now()
    tracer = get_tracer()

    event_dict: dict[str, Any] = {
        "seq": event.seq,
        "ts": event.ts,
        "type": event.type,
        "data": event.data,
    }
    if event.thread_id is not None:
        event_dict["thread_id"] = event.thread_id

    payload: dict[str, Any] = {
        "webhook_id": webhook_id,
        "session_id": session_id,
        "event": event_dict,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    secret: str | None = None
    if secret_ref is not None:
        secret = secret_resolver.resolve(secret_ref)

    with tracer.start_as_current_span(
        "webhook.sender.deliver",
        attributes={
            "webhook.id": webhook_id,
            "webhook.url": url,
            "webhook.delivery_id": delivery_id,
            "session.id": session_id,
            "event.seq": event.seq,
            "event.type": event.type,
        },
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="webhook.sender.deliver.invocation",
                code="webhook_sender_deliver",
                timestamp=now,
            ),
        )

        try:
            last_error: str = ""
            success = False

            for attempt in range(max_retries + 1):
                if attempt > 0:
                    delay = _backoff_delay(backoff_type, attempt)
                    await asyncio.sleep(delay)

                headers: dict[str, str] = {"Content-Type": "application/json"}
                if secret is not None:
                    sig = _sign_payload(payload_bytes, secret)
                    headers["X-Meridian-Signature"] = f"sha256={sig}"

                try:
                    response = await client.post(url, content=payload_bytes, headers=headers)
                    if response.is_success:
                        success = True
                        break
                    last_error = f"HTTP {response.status_code}"
                    if not response.is_server_error:
                        # 4xx is a permanent client error — do not retry
                        break
                    # 5xx is transient — retry
                except httpx.RequestError as exc:
                    last_error = str(exc)
                    # network error is transient — retry

            if success:
                span.set_attribute("webhook.delivery.success", True)
                audit_log.write(
                    AuditLogEntry(
                        level="info",
                        event="webhook.sender.delivered",
                        code="webhook_sender_delivered",
                        timestamp=_now(),
                        detail={
                            "webhook_id": webhook_id,
                            "delivery_id": delivery_id,
                            "session_id": session_id,
                            "event_seq": event.seq,
                            "event_type": event.type,
                        },
                    )
                )
                return

            # Permanent failure: write to DLQ then raise
            err = WebhookDeliveryError(
                message=(
                    f"Webhook {webhook_id!r} delivery failed after "
                    f"{max_retries + 1} attempt(s): {last_error}"
                ),
                timestamp=_now(),
            )
            span.set_attribute("webhook.delivery.success", False)
            record_error(span, err)

            dlq_webhook_dir = dlq_dir / webhook_id
            dlq_webhook_dir.mkdir(parents=True, exist_ok=True)
            dlq_record: dict[str, Any] = {
                "delivery_id": delivery_id,
                "webhook_id": webhook_id,
                "session_id": session_id,
                "event": event_dict,
                "attempts": max_retries + 1,
                "last_error": last_error,
                "failed_at": err.timestamp,
            }
            (dlq_webhook_dir / f"{delivery_id}.json").write_text(json.dumps(dlq_record))

            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="webhook.sender.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={
                        "webhook_id": webhook_id,
                        "delivery_id": delivery_id,
                        "session_id": session_id,
                        "event_seq": event.seq,
                        "event_type": event.type,
                        "message": err.message,
                    },
                )
            )
            raise err

        except WebhookDeliveryError:
            raise
        except Exception as exc:
            err2 = WebhookDeliveryError(
                message=f"Unexpected error delivering webhook {webhook_id!r}: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err2)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="webhook.sender.failed",
                    code=err2.code,
                    timestamp=err2.timestamp,
                    detail={
                        "webhook_id": webhook_id,
                        "delivery_id": delivery_id,
                        "session_id": session_id,
                        "event_seq": event.seq,
                        "event_type": event.type,
                        "message": err2.message,
                    },
                )
            )
            raise err2 from exc


# ---------------------------------------------------------------------------
# Background sender loop
# ---------------------------------------------------------------------------


async def run_webhook_sender_loop(
    storage_root: Path,
    audit_log: AuditLog,
    *,
    secret_resolver: SecretResolver | None = None,
    check_interval_seconds: float = 5.0,
    _http_client: httpx.AsyncClient | None = None,
) -> None:
    """Background loop that polls active webhooks and delivers matching events.

    For each active webhook, scans the relevant sessions for undelivered events
    matching the webhook's event_filter, delivers them (with retry/backoff), and
    advances the per-session delivery watermark.  Permanent failures are written
    to the DLQ.

    Watermarks are persisted to:
        storage_root/webhooks/watermarks/{webhook_id}.json

    Dead-letter queue entries are written to:
        storage_root/webhooks/dlq/{webhook_id}/{delivery_id}.json
    """
    _resolver: SecretResolver = (
        secret_resolver if secret_resolver is not None else NoopSecretResolver()
    )
    webhooks_dir = storage_root / "webhooks"
    watermarks_dir = webhooks_dir / "watermarks"
    dlq_dir = webhooks_dir / "dlq"
    reader = LocalEventLogReader(storage_root)

    async def _run(client: httpx.AsyncClient) -> None:
        while True:
            if webhooks_dir.exists():
                for webhook_file in sorted(webhooks_dir.glob("webhook_*.json")):
                    try:
                        webhook = json.loads(webhook_file.read_text())
                    except (json.JSONDecodeError, OSError):
                        continue

                    if webhook.get("status") != "active":
                        continue

                    event_filter = webhook.get("event_filter", {})
                    filter_types: set[str] = set(event_filter.get("types", []))
                    filter_session_id: str | None = event_filter.get("session_id")
                    webhook_id: str = webhook["id"]

                    watermark_file = watermarks_dir / f"{webhook_id}.json"
                    watermarks = _load_watermarks(watermark_file)

                    sessions = (
                        [filter_session_id]
                        if filter_session_id is not None
                        else _discover_session_ids(storage_root)
                    )

                    for session_id in sessions:
                        watermark = watermarks.get(session_id, -1)
                        try:
                            events = reader.read_after(session_id, watermark)
                        except Exception:
                            continue

                        matching = [e for e in events if e.type in filter_types]

                        for event in matching:
                            try:
                                await deliver_webhook_event(
                                    webhook,
                                    event,
                                    session_id,
                                    client=client,
                                    secret_resolver=_resolver,
                                    audit_log=audit_log,
                                    dlq_dir=dlq_dir,
                                )
                            except WebhookDeliveryError:
                                pass  # already logged; fall through to advance watermark
                            except Exception:
                                break  # unexpected; stop this webhook/session

                            # Advance watermark whether success or DLQ to avoid re-delivery.
                            watermarks[session_id] = event.seq
                            _save_watermarks(watermark_file, watermarks)

            await asyncio.sleep(check_interval_seconds)

    if _http_client is not None:
        await _run(_http_client)
    else:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await _run(client)
