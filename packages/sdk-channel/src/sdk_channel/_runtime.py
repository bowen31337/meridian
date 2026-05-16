from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from ._audit import AuditLog, NoopAuditLog
from ._contract import ChannelDriver
from ._telemetry import get_tracer, record_channel_failure, record_invocation_event
from ._types import (
    AuditLogEntry,
    ChannelCapabilities,
    ChannelFailure,
    SendRequest,
    SendResult,
    StartRequest,
    StopRequest,
    StructuredEvent,
)


@dataclass
class RuntimeOptions:
    """Options supplied by the host application for each runtime call."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)
    on_error: Callable[[ChannelFailure], None] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChannelRuntime:
    """
    Central dispatcher for channel operations.

    Register drivers once at startup; the runtime dispatches each
    start / send / stop call to the matching driver by kind,
    wrapping every invocation with an OTel span, a structured event, and
    audit-log writes on failure.
    """

    def __init__(self) -> None:
        self._drivers: dict[str, ChannelDriver] = {}

    def register(self, driver: ChannelDriver) -> None:
        """Register a driver by its kind. Raises ValueError on duplicate."""
        kind = driver.kind
        if kind in self._drivers:
            raise ValueError(f'Channel kind "{kind}" is already registered')
        self._drivers[kind] = driver

    def get(self, kind: str) -> ChannelDriver | None:
        """Return the registered driver for a kind, or None."""
        return self._drivers.get(kind)

    def capabilities(self, kind: str, options: RuntimeOptions | None = None) -> ChannelCapabilities:
        """
        Return the capability declaration for a registered kind.
        Raises ChannelFailure(CHAN_KIND_NOT_REGISTERED) if unknown.
        """
        driver = self._drivers.get(kind)
        if driver is None:
            raise ChannelFailure(
                code="CHAN_KIND_NOT_REGISTERED",
                message=f'No driver registered for kind "{kind}"',
                channel_id="",
                channel_kind=kind,
                session_id="",
                timestamp=_now(),
            )
        return driver.capabilities()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fail(
        self,
        span: object,
        failure: ChannelFailure,
        options: RuntimeOptions,
        audit_event: str,
    ) -> None:
        record_channel_failure(span, failure)  # type: ignore[arg-type]
        options.audit_log.write(
            AuditLogEntry(
                level="error",
                event=audit_event,
                channel_id=failure.channel_id,
                channel_kind=failure.channel_kind,
                session_id=failure.session_id,
                timestamp=failure.timestamp,
                detail={"code": failure.code, "message": failure.message},
            )
        )
        if options.on_error is not None:
            options.on_error(failure)

    def _not_registered(self, kind: str, channel_id: str, session_id: str) -> ChannelFailure:
        return ChannelFailure(
            code="CHAN_KIND_NOT_REGISTERED",
            message=f'No driver registered for kind "{kind}"',
            channel_id=channel_id,
            channel_kind=kind,
            session_id=session_id,
            timestamp=_now(),
        )

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    async def start(self, request: StartRequest, options: RuntimeOptions | None = None) -> None:
        """
        Connect a channel instance and begin accepting messages.

        Per-invocation:
          1. Opens OTel span "channel.start" with channel/session attributes.
          2. Attaches a "channel.invocation" structured event to the span.
          3. Validates that the requested kind is registered.
             On failure: records span error, writes audit log, calls on_error, raises ChannelFailure.
          4. Dispatches to the driver. Driver exceptions are wrapped in ChannelFailure
             (code CHAN_START_FAILED) and handled identically to step 3.
        """
        opts = options or RuntimeOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "channel.start",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.invocation",
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    operation="start",
                ),
            )

            driver = self._drivers.get(request.channel_kind)
            if driver is None:
                failure = self._not_registered(
                    request.channel_kind, request.channel_id, request.session_id
                )
                self._fail(span, failure, opts, "channel.start.failed")
                raise failure

            try:
                await driver.start(request)
            except ChannelFailure:
                raise
            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_START_FAILED",
                    message=str(exc),
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "channel.start.failed")
                raise failure from exc

    async def send(self, request: SendRequest, options: RuntimeOptions | None = None) -> SendResult:
        """
        Send a message over an active channel connection.

        Per-invocation:
          1. Opens OTel span "channel.send" with channel/session attributes.
          2. Attaches a "channel.invocation" structured event to the span.
          3. Validates that the requested kind is registered.
          4. Dispatches to the driver; wraps exceptions as CHAN_SEND_FAILED.
        Returns SendResult on success.
        """
        opts = options or RuntimeOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "channel.send",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.invocation",
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    operation="send",
                ),
            )

            driver = self._drivers.get(request.channel_kind)
            if driver is None:
                failure = self._not_registered(
                    request.channel_kind, request.channel_id, request.session_id
                )
                self._fail(span, failure, opts, "channel.send.failed")
                raise failure

            try:
                return await driver.send(request)
            except ChannelFailure:
                raise
            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_SEND_FAILED",
                    message=str(exc),
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "channel.send.failed")
                raise failure from exc

    async def stop(self, request: StopRequest, options: RuntimeOptions | None = None) -> None:
        """
        Disconnect a channel instance and release resources.

        Per-invocation:
          1. Opens OTel span "channel.stop" with channel/session attributes.
          2. Attaches a "channel.invocation" structured event to the span.
          3. Validates that the requested kind is registered.
          4. Dispatches to the driver; wraps exceptions as CHAN_STOP_FAILED.
        """
        opts = options or RuntimeOptions()
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "channel.stop",
            attributes={
                "channel.id": request.channel_id,
                "channel.kind": request.channel_kind,
                "session.id": request.session_id,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="channel.invocation",
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    operation="stop",
                ),
            )

            driver = self._drivers.get(request.channel_kind)
            if driver is None:
                failure = self._not_registered(
                    request.channel_kind, request.channel_id, request.session_id
                )
                self._fail(span, failure, opts, "channel.stop.failed")
                raise failure

            try:
                await driver.stop(request)
            except ChannelFailure:
                raise
            except Exception as exc:
                failure = ChannelFailure(
                    code="CHAN_STOP_FAILED",
                    message=str(exc),
                    channel_id=request.channel_id,
                    channel_kind=request.channel_kind,
                    session_id=request.session_id,
                    timestamp=now,
                    cause=exc,
                )
                self._fail(span, failure, opts, "channel.stop.failed")
                raise failure from exc


default_runtime = ChannelRuntime()
