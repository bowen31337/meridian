"""Shared test fixtures and fake provider implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from meridian_sdk_provider import (
    AuditLogEntry,
    MessageStartEvent,
    MessageStopEvent,
    ModelCallOpts,
    ModelCountReq,
    ModelEvent,
    ProviderCapabilities,
    TextDeltaEvent,
    TokenCount,
)


class FakeProvider:
    """Minimal ModelProvider that yields a fixed event sequence."""

    def __init__(
        self,
        name: str,
        kind: str = "fake",
        capabilities: ProviderCapabilities | None = None,
        events: list[ModelEvent] | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.capabilities = capabilities or ProviderCapabilities()
        self._events: list[ModelEvent] = events or [
            MessageStartEvent(type="message_start", model="test", provider=name),
            TextDeltaEvent(type="text_delta", text="hello"),
            MessageStopEvent(type="message_stop", stop_reason="end_turn"),
        ]
        self._raise_on_call = raise_on_call
        self.call_count = 0
        self.last_opts: ModelCallOpts | None = None

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        self.call_count += 1
        self.last_opts = opts
        if self._raise_on_call is not None:
            raise self._raise_on_call
        for event in self._events:
            yield event

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        return TokenCount(input_tokens=42)

    async def close(self) -> None:
        pass


class CollectingAuditLog:
    """AuditLog that records all written entries for assertion in tests."""

    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider(name="test-provider")


@pytest.fixture
def audit_log() -> CollectingAuditLog:
    return CollectingAuditLog()


def make_opts(**kwargs: object) -> ModelCallOpts:
    defaults: dict = {
        "model": "test-provider:test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    defaults.update(kwargs)
    return ModelCallOpts(**defaults)
