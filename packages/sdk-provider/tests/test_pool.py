"""Tests for LoadBalancedProvider — round-robin + intra-pool failover."""

from __future__ import annotations

import pytest

from meridian_sdk_provider import (
    LoadBalancedProvider,
    ProviderCapabilities,
    ProviderRateLimitError,
)
from tests.conftest import FakeProvider, make_opts


def _pool(members) -> LoadBalancedProvider:  # type: ignore[no-untyped-def]
    return LoadBalancedProvider(name="zai", kind="anthropic", members=members)


async def _drain(prov, opts):  # type: ignore[no-untyped-def]
    return [e async for e in prov.call(opts)]


def test_requires_at_least_one_member() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        LoadBalancedProvider(name="p", kind="anthropic", members=[])


def test_name_kind_and_capabilities_exposed() -> None:
    m = FakeProvider(name="m0", capabilities=ProviderCapabilities(count_tokens=True))
    pool = _pool([m])
    assert pool.name == "zai"
    assert pool.kind == "anthropic"
    assert pool.capabilities.count_tokens is True  # inherited from the first member


async def test_round_robins_across_members() -> None:
    a = FakeProvider(name="a")
    b = FakeProvider(name="b")
    pool = _pool([a, b])
    for _ in range(4):
        await _drain(pool, make_opts())
    # Four calls spread evenly: 2 to each member.
    assert a.call_count == 2
    assert b.call_count == 2


async def test_each_member_serves_full_event_stream() -> None:
    pool = _pool([FakeProvider(name="a"), FakeProvider(name="b")])
    events = await _drain(pool, make_opts())
    assert len(events) == 3


async def test_failover_rotates_to_next_member() -> None:
    # First member (by round-robin start) fails pre-stream -> next serves it.
    a = FakeProvider(name="a", raise_on_call=ProviderRateLimitError("rl", "a"))
    b = FakeProvider(name="b")
    pool = _pool([a, b])
    events = await _drain(pool, make_opts())
    assert len(events) == 3
    assert a.call_count == 1
    assert b.call_count == 1


async def test_all_members_fail_raises_last_error() -> None:
    a = FakeProvider(name="a", raise_on_call=ProviderRateLimitError("rl-a", "a"))
    b = FakeProvider(name="b", raise_on_call=ProviderRateLimitError("rl-b", "b"))
    pool = _pool([a, b])
    with pytest.raises(ProviderRateLimitError):
        await _drain(pool, make_opts())
    assert a.call_count == 1
    assert b.call_count == 1


async def test_empty_member_stream_returns_cleanly() -> None:
    class _Empty:
        name = "e"
        kind = "anthropic"
        capabilities = ProviderCapabilities()

        async def call(self, opts):  # type: ignore[no-untyped-def]
            return
            yield  # unreachable; makes call() an async generator

    pool = _pool([_Empty()])
    assert await _drain(pool, make_opts()) == []


async def test_close_closes_all_members() -> None:
    a = FakeProvider(name="a")
    b = FakeProvider(name="b")
    pool = _pool([a, b])
    await pool.close()
    assert a.closed and b.closed


async def test_count_tokens_delegates_to_first_member() -> None:
    a = FakeProvider(name="a")
    b = FakeProvider(name="b")
    pool = _pool([a, b])
    from meridian_sdk_provider import Message, ModelCountReq

    tc = await pool.count_tokens(
        ModelCountReq(model="m", messages=[Message(role="user", content="hi")])
    )
    assert tc.input_tokens == 42  # FakeProvider.count_tokens returns 42


def test_list_models_delegates_to_first_member() -> None:
    pool = _pool([FakeProvider(name="a"), FakeProvider(name="b")])
    assert pool.list_models() == []  # FakeProvider returns []
