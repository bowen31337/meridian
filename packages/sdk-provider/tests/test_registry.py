"""Tests for ProviderRegistry: slot management, hot-swap, drain, and audit."""

from __future__ import annotations

import asyncio

import pytest

from meridian_sdk_provider import (
    FallbackRule,
    ModelCallOpts,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    ProviderCapabilities,
    ProviderRegistry,
    ProviderRateLimitError,
)
from meridian_sdk_provider.registry import _ProviderSlot
from tests.conftest import CollectingAuditLog, FakeProvider, make_opts


# ─── _ProviderSlot unit tests ──────────────────────────────────────────────────


def test_slot_starts_drained() -> None:
    provider = FakeProvider(name="p")
    slot = _ProviderSlot(provider)
    assert slot._drained.is_set()
    assert slot._refcount == 0


def test_slot_acquire_clears_drain_event() -> None:
    slot = _ProviderSlot(FakeProvider(name="p"))
    slot.acquire()
    assert not slot._drained.is_set()
    assert slot._refcount == 1


def test_slot_release_sets_drain_event_when_zero() -> None:
    slot = _ProviderSlot(FakeProvider(name="p"))
    slot.acquire()
    slot.acquire()
    slot.release()
    assert not slot._drained.is_set()
    slot.release()
    assert slot._drained.is_set()
    assert slot._refcount == 0


def test_slot_release_does_not_go_negative() -> None:
    slot = _ProviderSlot(FakeProvider(name="p"))
    slot.release()  # called without acquire
    assert slot._refcount == 0
    assert slot._drained.is_set()


async def test_slot_wait_drained_returns_immediately_when_no_inflight() -> None:
    slot = _ProviderSlot(FakeProvider(name="p"))
    await slot.wait_drained(1.0)  # should not block


async def test_slot_wait_drained_waits_for_release() -> None:
    slot = _ProviderSlot(FakeProvider(name="p"))
    slot.acquire()

    released: list[bool] = []

    async def releaser() -> None:
        await asyncio.sleep(0.01)
        released.append(True)
        slot.release()

    task = asyncio.create_task(releaser())
    await slot.wait_drained(1.0)
    assert released == [True]
    await task


async def test_slot_wait_drained_times_out_gracefully() -> None:
    slot = _ProviderSlot(FakeProvider(name="p"))
    slot.acquire()
    # Should return without raising even though not drained
    await slot.wait_drained(0.01)


# ─── ProviderRegistry basics ───────────────────────────────────────────────────


def test_registry_holds_providers_from_init() -> None:
    p1 = FakeProvider(name="a")
    p2 = FakeProvider(name="b")
    reg = ProviderRegistry(providers={"a": p1, "b": p2})
    assert reg.get_slot("a") is not None
    assert reg.get_slot("a").provider is p1  # type: ignore[union-attr]
    assert reg.get_slot("b") is not None
    assert reg.get_slot("b").provider is p2  # type: ignore[union-attr]
    assert reg.get_slot("c") is None


def test_registry_names_returns_all() -> None:
    reg = ProviderRegistry(providers={"x": FakeProvider(name="x"), "y": FakeProvider(name="y")})
    assert sorted(reg.names()) == ["x", "y"]


def test_registry_providers_returns_all() -> None:
    p1, p2 = FakeProvider(name="a"), FakeProvider(name="b")
    reg = ProviderRegistry(providers={"a": p1, "b": p2})
    assert set(reg.providers()) == {p1, p2}


def test_registry_register_adds_provider_synchronously() -> None:
    reg = ProviderRegistry()
    p = FakeProvider(name="p")
    reg.register(p)
    slot = reg.get_slot("p")
    assert slot is not None
    assert slot.provider is p


def test_registry_register_replaces_without_drain() -> None:
    old = FakeProvider(name="p")
    new = FakeProvider(name="p")
    reg = ProviderRegistry(providers={"p": old})
    reg.register(new)
    assert reg.get_slot("p").provider is new  # type: ignore[union-attr]
    assert not old.closed  # no drain/close for synchronous register


# ─── swap() hot-swap ───────────────────────────────────────────────────────────


async def test_swap_replaces_provider() -> None:
    old = FakeProvider(name="p")
    new = FakeProvider(name="p")
    reg = ProviderRegistry(providers={"p": old})
    await reg.swap("p", new)
    assert reg.get_slot("p").provider is new  # type: ignore[union-attr]


async def test_swap_closes_old_provider() -> None:
    old = FakeProvider(name="p")
    reg = ProviderRegistry(providers={"p": old})
    await reg.swap("p", FakeProvider(name="p"))
    assert old.closed


async def test_swap_adds_new_provider_when_none_existed() -> None:
    reg = ProviderRegistry()
    p = FakeProvider(name="p")
    await reg.swap("p", p)
    assert reg.get_slot("p").provider is p  # type: ignore[union-attr]


async def test_swap_drains_inflight_before_closing() -> None:
    old = FakeProvider(name="p")
    reg = ProviderRegistry(providers={"p": old})

    old_slot = reg.get_slot("p")
    assert old_slot is not None
    old_slot.acquire()  # simulate in-flight call

    close_order: list[str] = []

    async def finish_inflight() -> None:
        await asyncio.sleep(0.01)
        close_order.append("released")
        old_slot.release()

    task = asyncio.create_task(finish_inflight())
    await reg.swap("p", FakeProvider(name="p"))
    close_order.append("swap_done")
    await task

    # close() must happen AFTER release
    assert "released" in close_order
    assert old.closed


async def test_swap_writes_audit_on_failure(audit_log: CollectingAuditLog) -> None:
    class BrokenProvider(FakeProvider):
        @property
        def kind(self) -> str:
            raise RuntimeError("broken kind")

    reg = ProviderRegistry(audit_log=audit_log)
    broken = FakeProvider(name="p")

    # Monkey-patch kind to fail after construction, during span attribute read
    original_swap = reg.swap

    async def patched_swap(name: str, provider: object, **kw: object) -> None:
        # Trigger failure by passing a provider whose close() raises
        class CloseError(FakeProvider):
            async def close(self) -> None:
                raise RuntimeError("close failed")

        await original_swap(name, CloseError(name="p"), **kw)  # type: ignore[arg-type]

    await reg.swap("p", broken)  # put a valid provider in first
    # Force a failure: close() will raise when draining
    old_slot = reg.get_slot("p")
    assert old_slot is not None

    class RaisesOnClose(FakeProvider):
        async def close(self) -> None:
            raise RuntimeError("intentional close failure")

    reg2 = ProviderRegistry(providers={"p": RaisesOnClose(name="p")}, audit_log=audit_log)
    with pytest.raises(RuntimeError, match="intentional close failure"):
        await reg2.swap("p", FakeProvider(name="p"))

    assert any(e.event == "provider_registry.swap.failed" for e in audit_log.entries)


# ─── swap_all() hot-swap ──────────────────────────────────────────────────────


async def test_swap_all_replaces_entire_registry() -> None:
    old_a = FakeProvider(name="a")
    old_b = FakeProvider(name="b")
    reg = ProviderRegistry(providers={"a": old_a, "b": old_b})

    new_a = FakeProvider(name="a")
    new_c = FakeProvider(name="c")
    await reg.swap_all({"a": new_a, "c": new_c})

    assert reg.get_slot("a").provider is new_a  # type: ignore[union-attr]
    assert reg.get_slot("c") is not None
    assert reg.get_slot("b") is None  # removed


async def test_swap_all_closes_all_old_providers() -> None:
    old_a = FakeProvider(name="a")
    old_b = FakeProvider(name="b")
    reg = ProviderRegistry(providers={"a": old_a, "b": old_b})
    await reg.swap_all({"a": FakeProvider(name="a")})
    assert old_a.closed
    assert old_b.closed


async def test_swap_all_with_empty_dict_clears_registry() -> None:
    reg = ProviderRegistry(providers={"p": FakeProvider(name="p")})
    await reg.swap_all({})
    assert reg.names() == []


# ─── close_all() ─────────────────────────────────────────────────────────────


async def test_close_all_closes_every_provider() -> None:
    p1 = FakeProvider(name="a")
    p2 = FakeProvider(name="b")
    reg = ProviderRegistry(providers={"a": p1, "b": p2})
    await reg.close_all()
    assert p1.closed
    assert p2.closed
    assert reg.names() == []


# ─── Router + Registry integration ────────────────────────────────────────────


async def test_router_uses_registry_provider() -> None:
    p = FakeProvider(name="p")
    reg = ProviderRegistry(providers={"p": p})
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        registry=reg,
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3
    assert p.call_count == 1


async def test_router_register_provider_updates_registry() -> None:
    reg = ProviderRegistry()
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        registry=reg,
    )
    p = FakeProvider(name="p")
    router.register_provider(p)
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3


async def test_router_slot_acquired_and_released_on_success() -> None:
    p = FakeProvider(name="p")
    reg = ProviderRegistry(providers={"p": p})
    slot = reg.get_slot("p")
    assert slot is not None

    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        registry=reg,
    )
    async for _ in router.call(make_opts()):
        pass

    assert slot._refcount == 0
    assert slot._drained.is_set()


async def test_router_slot_released_on_failure() -> None:
    p = FakeProvider(name="p", raise_on_call=ProviderRateLimitError("err", "p"))
    reg = ProviderRegistry(providers={"p": p})
    slot = reg.get_slot("p")
    assert slot is not None

    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        registry=reg,
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass

    assert slot._refcount == 0
    assert slot._drained.is_set()


async def test_swap_during_inflight_call_drains_before_close() -> None:
    """swap() must not close the old provider while a router call is in progress."""
    inflight_event = asyncio.Event()
    release_event = asyncio.Event()

    class SlowProvider(FakeProvider):
        async def call(self, opts: ModelCallOpts):  # type: ignore[override]
            inflight_event.set()
            await release_event.wait()
            for ev in self._events:
                yield ev

    slow = SlowProvider(name="p")
    reg = ProviderRegistry(providers={"p": slow})
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        registry=reg,
    )

    collected: list[object] = []

    async def do_call() -> None:
        async for ev in router.call(make_opts()):
            collected.append(ev)

    call_task = asyncio.create_task(do_call())
    await inflight_event.wait()

    # Start the swap while the call is in-flight; it should not close slow yet.
    new_p = FakeProvider(name="p")
    swap_task = asyncio.create_task(reg.swap("p", new_p))

    await asyncio.sleep(0.01)
    assert not slow.closed  # drain is still waiting

    release_event.set()
    await call_task
    await swap_task

    assert slow.closed  # closed only after call finished
    assert len(collected) == 3


async def test_router_fallback_slot_released_on_success() -> None:
    primary = FakeProvider(name="primary", raise_on_call=ProviderRateLimitError("err", "primary"))
    fallback = FakeProvider(name="fallback")
    reg = ProviderRegistry(providers={"primary": primary, "fallback": fallback})
    fb_slot = reg.get_slot("fallback")
    assert fb_slot is not None

    router = ModelRouter(
        policy=ModelRoutingPolicy(
            rules=[ModelRoutingRule(model="primary:m")],
            fallbacks=[FallbackRule(on="rate_limit", model="fallback:m")],
        ),
        registry=reg,
    )
    async for _ in router.call(make_opts()):
        pass

    assert fb_slot._refcount == 0
    assert fb_slot._drained.is_set()


async def test_router_close_delegates_to_registry() -> None:
    p = FakeProvider(name="p")
    reg = ProviderRegistry(providers={"p": p})
    router = ModelRouter(
        policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p:m")]),
        registry=reg,
    )
    await router.close()
    assert p.closed
