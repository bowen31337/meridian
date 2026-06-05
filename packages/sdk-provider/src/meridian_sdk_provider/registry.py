"""Provider registry: holds instantiated ModelProvider instances keyed by config-level name.

Supports hot-swap on config reload via atomic pointer swap with drain of in-flight calls.
Each swap() / swap_all() emits an OTel span and a structured "provider_registry.invocation"
span event.  On failure the span is marked ERROR, an audit log entry is written, and the
exception is re-raised to the caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
from typing import Any

from opentelemetry.trace import Status, StatusCode

from .audit import AuditLog, AuditLogEntry, NoopAuditLog
from .protocol import ModelProvider
from .telemetry import get_tracer


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


class _ProviderSlot:
    """Reference-counted holder for a single ModelProvider instance.

    The registry atomically replaces the slot pointer on hot-swap.  Callers
    that already hold a direct reference to an old slot continue using it;
    the registry drains by waiting for the in-flight count to reach zero
    before calling provider.close().
    """

    __slots__ = ("provider", "_refcount", "_drained")

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider
        self._refcount: int = 0
        self._drained: asyncio.Event = asyncio.Event()
        self._drained.set()  # set = "no in-flight calls"

    def acquire(self) -> None:
        """Mark one more in-flight call on this slot."""
        if self._refcount == 0:
            self._drained.clear()
        self._refcount += 1

    def release(self) -> None:
        """Mark one in-flight call finished; signal drain when count hits zero."""
        self._refcount -= 1
        if self._refcount <= 0:
            self._refcount = 0
            self._drained.set()

    async def wait_drained(self, timeout: float) -> None:
        """Block until all in-flight calls complete, or *timeout* seconds elapse."""
        # proceed with close even if some calls are still running
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(self._drained.wait()), timeout)


class ProviderRegistry:
    """Holds instantiated ModelProvider instances keyed by config-level name.

    Supports hot-swap on config reload via atomic pointer swap with drain of
    in-flight calls.

    Per swap() / swap_all():

    1. Opens an OTel span ``"provider_registry.swap"``.
    2. Attaches a structured ``"provider_registry.invocation"`` span event.
    3. Atomically replaces the registry entry / entire dict under an asyncio lock.
    4. Waits for in-flight calls on the displaced slot(s) to complete (drain).
    5. Calls ``close()`` on each displaced provider.
    6. On failure: sets span status to ERROR, writes an audit log entry, and
       re-raises the exception so the caller receives it.

    The ModelRouter acquires a slot via ``get_slot()`` before calling a provider
    and releases it after the call completes (or fails), enabling the drain step.
    """

    def __init__(
        self,
        providers: dict[str, ModelProvider] | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._slots: dict[str, _ProviderSlot] = {}
        self._audit: AuditLog = audit_log or NoopAuditLog()
        self._lock: asyncio.Lock = asyncio.Lock()
        for name, provider in (providers or {}).items():
            self._slots[name] = _ProviderSlot(provider)

    # ── Read path ─────────────────────────────────────────────────────────────

    def get_slot(self, name: str) -> _ProviderSlot | None:
        """Return the current slot for *name*, or None if not registered."""
        return self._slots.get(name)

    def names(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._slots)

    def providers(self) -> list[ModelProvider]:
        """Return all registered providers as a flat list."""
        return [slot.provider for slot in self._slots.values()]

    # ── Synchronous registration (startup / test) ──────────────────────────────

    def register(self, provider: ModelProvider) -> None:
        """Register a provider synchronously without drain (startup or test only).

        For hot-swap at runtime use swap() which drains in-flight calls first.
        """
        self._slots[provider.name] = _ProviderSlot(provider)

    # ── Hot-swap write path ────────────────────────────────────────────────────

    async def swap(
        self,
        name: str,
        provider: ModelProvider,
        *,
        drain_timeout: float = 30.0,
    ) -> None:
        """Hot-swap a single provider instance.

        Atomically replaces the slot for *name*, drains all in-flight calls on
        the displaced slot, then closes the displaced provider.
        """
        tracer = get_tracer()
        now = _now_iso()
        with tracer.start_as_current_span(
            "provider_registry.swap",
            attributes={
                "registry.operation": "swap",
                "provider.name": name,
                "provider.kind": provider.kind,
            },
        ) as span:
            span.add_event(
                "provider_registry.invocation",
                {
                    "operation": "swap",
                    "provider.name": name,
                    "provider.kind": provider.kind,
                },
            )
            try:
                old_slot = await self._atomic_replace(name, _ProviderSlot(provider))
                if old_slot is not None:
                    await self._drain_and_close(old_slot, drain_timeout)
            except Exception as exc:
                self._record_failure(span, exc, name, provider.kind, now)
                raise

    async def swap_all(
        self,
        providers: dict[str, ModelProvider],
        *,
        drain_timeout: float = 30.0,
    ) -> None:
        """Atomically replace the entire registry (used on full config reload).

        Builds new slots, atomically swaps the internal dict, then drains and
        closes all displaced provider instances.
        """
        tracer = get_tracer()
        now = _now_iso()
        provider_names = sorted(providers)
        with tracer.start_as_current_span(
            "provider_registry.swap",
            attributes={
                "registry.operation": "swap_all",
                "registry.provider_count": len(providers),
                "registry.provider_names": ",".join(provider_names),
            },
        ) as span:
            span.add_event(
                "provider_registry.invocation",
                {
                    "operation": "swap_all",
                    "provider_count": len(providers),
                    "provider_names": ",".join(provider_names),
                },
            )
            try:
                new_slots = {n: _ProviderSlot(p) for n, p in providers.items()}
                old_slots = await self._atomic_replace_all(new_slots)
                for old_slot in old_slots:
                    await self._drain_and_close(old_slot, drain_timeout)
            except Exception as exc:
                self._record_failure(span, exc, "<registry>", "<all>", now)
                raise

    async def close_all(self, *, drain_timeout: float = 30.0) -> None:
        """Drain all in-flight calls and close all providers (daemon shutdown)."""
        async with self._lock:
            slots = list(self._slots.values())
            self._slots.clear()
        for slot in slots:
            await self._drain_and_close(slot, drain_timeout)

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _atomic_replace(self, name: str, new_slot: _ProviderSlot) -> _ProviderSlot | None:
        async with self._lock:
            old = self._slots.get(name)
            self._slots[name] = new_slot
            return old

    async def _atomic_replace_all(self, new_slots: dict[str, _ProviderSlot]) -> list[_ProviderSlot]:
        async with self._lock:
            old = list(self._slots.values())
            self._slots = new_slots
            return old

    @staticmethod
    async def _drain_and_close(slot: _ProviderSlot, timeout: float) -> None:
        await slot.wait_drained(timeout)
        await slot.provider.close()

    def _record_failure(
        self,
        span: Any,
        exc: Exception,
        name: str,
        kind: str,
        timestamp: str,
    ) -> None:
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.add_event(
            "provider_registry.error",
            {
                "provider.name": name,
                "provider.kind": kind,
                "error.type": type(exc).__name__,
                "error.message": str(exc),
            },
        )
        span.record_exception(exc)
        self._audit.write(
            AuditLogEntry(
                level="error",
                event="provider_registry.swap.failed",
                provider_name=name,
                provider_kind=kind,
                timestamp=timestamp,
                detail={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        )
