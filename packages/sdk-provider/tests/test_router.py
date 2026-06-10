"""Tests for ModelRouter: routing rules, capability constraints, fallback, audit log."""

from __future__ import annotations

import pytest

from meridian_sdk_provider import (
    FallbackRule,
    ModelCallOpts,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    NoProviderFoundError,
    ProviderCapabilities,
    ProviderRateLimitError,
    ProviderServerError,
    RoutingCondition,
    RoutingError,
    TokenRange,
)
from tests.conftest import CollectingAuditLog, CollectingModelCallEventLog, FakeProvider, make_opts


def _router(
    rules: list[ModelRoutingRule],
    providers: dict,
    fallbacks: list[FallbackRule] | None = None,
    audit_log: CollectingAuditLog | None = None,
    event_log: CollectingModelCallEventLog | None = None,
) -> ModelRouter:
    policy = ModelRoutingPolicy(rules=rules, fallbacks=fallbacks or [])
    return ModelRouter(
        policy=policy,
        providers=providers,
        audit_log=audit_log,
        event_log=event_log,
    )


# ─── Basic routing ────────────────────────────────────────────────────────────


async def test_router_delivers_all_events() -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3
    assert provider.call_count == 1


async def test_router_passes_model_to_provider() -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:specific-model")],
        providers={"p": provider},
    )
    async for _ in router.call(make_opts()):
        pass
    assert provider.last_opts is not None
    assert provider.last_opts.model == "specific-model"


async def test_router_raises_when_no_rule_matches() -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(when=RoutingCondition(role="worker"), model="p:m")],
        providers={"p": provider},
    )
    with pytest.raises(NoProviderFoundError):
        async for _ in router.call(make_opts(role="planner")):
            pass


async def test_router_catch_all_rule_matches_any_opts() -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
    )
    for role in ("worker", "planner", None):
        async for _ in router.call(make_opts(role=role)):
            pass
    assert provider.call_count == 3


# ─── Routing conditions ───────────────────────────────────────────────────────


async def test_role_condition_matches() -> None:
    worker_prov = FakeProvider(name="worker")
    planner_prov = FakeProvider(name="planner")
    router = _router(
        rules=[
            ModelRoutingRule(when=RoutingCondition(role="worker"), model="worker:m"),
            ModelRoutingRule(model="planner:m"),
        ],
        providers={"worker": worker_prov, "planner": planner_prov},
    )
    async for _ in router.call(make_opts(role="worker")):
        pass
    async for _ in router.call(make_opts(role="planner")):
        pass

    assert worker_prov.call_count == 1
    assert planner_prov.call_count == 1


async def test_token_range_gt_condition() -> None:
    big_prov = FakeProvider(name="big")
    small_prov = FakeProvider(name="small")
    router = _router(
        rules=[
            ModelRoutingRule(
                when=RoutingCondition(estimated_input_tokens=TokenRange(gt=100_000)),
                model="big:m",
            ),
            ModelRoutingRule(model="small:m"),
        ],
        providers={"big": big_prov, "small": small_prov},
    )
    async for _ in router.call(make_opts(estimated_input_tokens=200_000)):
        pass
    async for _ in router.call(make_opts(estimated_input_tokens=50_000)):
        pass

    assert big_prov.call_count == 1
    assert small_prov.call_count == 1


async def test_skill_id_condition() -> None:
    provider = FakeProvider(name="p")
    other = FakeProvider(name="o")
    router = _router(
        rules=[
            ModelRoutingRule(when=RoutingCondition(skill_id="summarize"), model="p:m"),
            ModelRoutingRule(model="o:m"),
        ],
        providers={"p": provider, "o": other},
    )
    async for _ in router.call(make_opts(skill_id="summarize")):
        pass
    async for _ in router.call(make_opts(skill_id="other")):
        pass

    assert provider.call_count == 1
    assert other.call_count == 1


async def test_metadata_match_condition() -> None:
    provider = FakeProvider(name="p")
    other = FakeProvider(name="o")
    router = _router(
        rules=[
            ModelRoutingRule(
                when=RoutingCondition(metadata_match={"tier": "pro"}),
                model="p:m",
            ),
            ModelRoutingRule(model="o:m"),
        ],
        providers={"p": provider, "o": other},
    )
    async for _ in router.call(make_opts(metadata={"tier": "pro"})):
        pass
    async for _ in router.call(make_opts(metadata={"tier": "free"})):
        pass

    assert provider.call_count == 1
    assert other.call_count == 1


# ─── Capability constraints ───────────────────────────────────────────────────


async def test_streaming_flag_cleared_when_not_supported() -> None:
    provider = FakeProvider(name="p", capabilities=ProviderCapabilities(streaming=False))
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
    )
    async for _ in router.call(make_opts(stream=True)):
        pass
    assert provider.last_opts is not None
    assert provider.last_opts.stream is False


async def test_thinking_flag_cleared_when_not_supported() -> None:
    provider = FakeProvider(name="p", capabilities=ProviderCapabilities(thinking=False))
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
    )
    async for _ in router.call(make_opts(enable_thinking=True, thinking_budget_tokens=1000)):
        pass
    assert provider.last_opts is not None
    assert provider.last_opts.enable_thinking is False
    assert provider.last_opts.thinking_budget_tokens is None


async def test_cache_control_stripped_when_not_supported() -> None:
    from meridian_sdk_provider import CacheControl, Message, TextBlock

    msg = Message(
        role="user",
        content=[TextBlock(type="text", text="hi", cache_control=CacheControl())],
    )
    provider = FakeProvider(name="p", capabilities=ProviderCapabilities(cache_control=False))
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
    )
    opts = ModelCallOpts(model="p:m", messages=[msg])
    async for _ in router.call(opts):
        pass

    assert provider.last_opts is not None
    block = provider.last_opts.messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert block.cache_control is None


async def test_cache_control_preserved_when_supported() -> None:
    from meridian_sdk_provider import CacheControl, Message, TextBlock

    msg = Message(
        role="user",
        content=[TextBlock(type="text", text="hi", cache_control=CacheControl())],
    )
    provider = FakeProvider(name="p", capabilities=ProviderCapabilities(cache_control=True))
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
    )
    opts = ModelCallOpts(model="p:m", messages=[msg])
    async for _ in router.call(opts):
        pass

    block = provider.last_opts.messages[0].content[0]  # type: ignore[index]
    assert isinstance(block, TextBlock)
    assert block.cache_control is not None


# ─── Fallback ─────────────────────────────────────────────────────────────────


async def test_fallback_triggered_on_rate_limit() -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderRateLimitError("rate limited", "primary"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="rate_limit", model="fallback:m")],
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3
    assert fallback.call_count == 1


async def test_fallback_not_triggered_on_unmatched_error() -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderRateLimitError("rate limited", "primary"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="timeout", model="fallback:m")],
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass
    assert fallback.call_count == 0


async def test_fallback_any_catches_all_errors() -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ValueError("unexpected"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="any", model="fallback:m")],
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3


async def test_fallback_failure_raises_and_logs(audit_log: CollectingAuditLog) -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderRateLimitError("rate limited", "primary"),
    )
    fallback = FakeProvider(
        name="fallback",
        raise_on_call=ProviderRateLimitError("also rate limited", "fallback"),
    )
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="any", model="fallback:m")],
        audit_log=audit_log,
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass

    # Expect: one router.failover decision entry + one provider.call.failed entry.
    assert len(audit_log.entries) == 2
    failover_entry = next(e for e in audit_log.entries if e.event == "router.failover")
    failure_entry = next(e for e in audit_log.entries if e.event == "provider.call.failed")
    assert failover_entry.provider_name == "primary"
    assert failure_entry.provider_name == "fallback"


# ─── Multi-hop fallback cascade ────────────────────────────────────────────────


async def test_multihop_cascades_to_second_fallback() -> None:
    # primary fails, fb1 fails pre-stream, fb2 streams: the call still succeeds.
    primary = FakeProvider(name="primary", raise_on_call=ProviderRateLimitError("rl", "primary"))
    fb1 = FakeProvider(name="fb1", raise_on_call=ProviderRateLimitError("rl", "fb1"))
    fb2 = FakeProvider(name="fb2")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fb1": fb1, "fb2": fb2},
        fallbacks=[FallbackRule(on="any", model="fb1:m"), FallbackRule(on="any", model="fb2:m")],
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3
    assert fb1.call_count == 1
    assert fb2.call_count == 1


async def test_multihop_exhausts_bench_then_raises(audit_log: CollectingAuditLog) -> None:
    primary = FakeProvider(name="primary", raise_on_call=ProviderRateLimitError("rl", "primary"))
    fb1 = FakeProvider(name="fb1", raise_on_call=ProviderRateLimitError("rl", "fb1"))
    fb2 = FakeProvider(name="fb2", raise_on_call=ProviderRateLimitError("rl", "fb2"))
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fb1": fb1, "fb2": fb2},
        fallbacks=[FallbackRule(on="any", model="fb1:m"), FallbackRule(on="any", model="fb2:m")],
        audit_log=audit_log,
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass
    # Two failover hops logged (primary->fb1, fb1->fb2) + one final failure (fb2).
    failovers = [e for e in audit_log.entries if e.event == "router.failover"]
    failures = [e for e in audit_log.entries if e.event == "provider.call.failed"]
    assert len(failovers) == 2
    assert len(failures) == 1
    assert failures[0].provider_name == "fb2"
    assert fb1.call_count == 1 and fb2.call_count == 1


async def test_multihop_reevaluates_error_category_per_hop() -> None:
    # primary rate_limits -> fb1 (on=rate_limit) raises 5xx -> fb2 (on=5xx) streams.
    # fb2 is only reachable if eligibility is recomputed against fb1's failure.
    primary = FakeProvider(name="primary", raise_on_call=ProviderRateLimitError("rl", "primary"))
    fb1 = FakeProvider(name="fb1", raise_on_call=ProviderServerError("500", "fb1"))
    fb2 = FakeProvider(name="fb2")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fb1": fb1, "fb2": fb2},
        fallbacks=[
            FallbackRule(on="rate_limit", model="fb1:m"),
            FallbackRule(on="5xx", model="fb2:m"),
        ],
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3
    assert fb1.call_count == 1
    assert fb2.call_count == 1


async def test_multihop_skips_nonmatching_fallback() -> None:
    # primary rate_limits; fb1 (on=timeout) is not eligible and must be skipped;
    # fb2 (on=any) handles it. fb1 is never called.
    primary = FakeProvider(name="primary", raise_on_call=ProviderRateLimitError("rl", "primary"))
    fb1 = FakeProvider(name="fb1")
    fb2 = FakeProvider(name="fb2")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fb1": fb1, "fb2": fb2},
        fallbacks=[
            FallbackRule(on="timeout", model="fb1:m"),
            FallbackRule(on="any", model="fb2:m"),
        ],
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3
    assert fb1.call_count == 0
    assert fb2.call_count == 1


async def test_multihop_each_fallback_tried_at_most_once() -> None:
    # Two on=any fallbacks that both fail -> each attempted exactly once, no loop.
    primary = FakeProvider(name="primary", raise_on_call=ValueError("boom"))
    fb1 = FakeProvider(name="fb1", raise_on_call=ValueError("boom1"))
    fb2 = FakeProvider(name="fb2", raise_on_call=ValueError("boom2"))
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fb1": fb1, "fb2": fb2},
        fallbacks=[FallbackRule(on="any", model="fb1:m"), FallbackRule(on="any", model="fb2:m")],
    )
    with pytest.raises(ValueError):
        async for _ in router.call(make_opts()):
            pass
    assert fb1.call_count == 1
    assert fb2.call_count == 1


class _StreamThenFail:
    """Provider that yields one event, then fails — to exercise a committed hop."""

    def __init__(self, name: str, fail: Exception) -> None:
        self.name = name
        self.kind = "fake"
        self.capabilities = ProviderCapabilities()
        self._fail = fail
        self.call_count = 0

    async def call(self, opts):  # type: ignore[no-untyped-def]
        from meridian_sdk_provider import MessageStartEvent

        self.call_count += 1
        yield MessageStartEvent(type="message_start", model="m", provider=self.name)
        raise self._fail


async def test_multihop_committed_fallback_midstream_failure_raises(
    audit_log: CollectingAuditLog,
) -> None:
    # Once a fallback streams its first event the router commits to it; a later
    # mid-stream failure surfaces (no further hop) and is audited.
    primary = FakeProvider(name="primary", raise_on_call=ProviderRateLimitError("rl", "primary"))
    fb = _StreamThenFail("fb", ProviderRateLimitError("mid", "fb"))
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fb": fb},
        fallbacks=[FallbackRule(on="any", model="fb:m")],
        audit_log=audit_log,
    )
    seen = []
    with pytest.raises(ProviderRateLimitError):
        async for e in router.call(make_opts()):
            seen.append(e)
    assert len(seen) == 1  # the committed first event was delivered
    assert fb.call_count == 1
    failure = next(e for e in audit_log.entries if e.event == "provider.call.failed")
    assert failure.provider_name == "fb"


class _EmptyStream:
    """Provider whose call yields no events at all."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.kind = "fake"
        self.capabilities = ProviderCapabilities()
        self.call_count = 0

    async def call(self, opts):  # type: ignore[no-untyped-def]
        self.call_count += 1
        return
        yield  # unreachable; makes call() an async generator


async def test_multihop_fallback_yields_no_events_returns() -> None:
    # A fallback that streams nothing ends the call cleanly (no error, no events).
    primary = FakeProvider(name="primary", raise_on_call=ProviderRateLimitError("rl", "primary"))
    empty = _EmptyStream("empty")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "empty": empty},
        fallbacks=[FallbackRule(on="any", model="empty:m")],
    )
    events = [e async for e in router.call(make_opts())]
    assert events == []
    assert empty.call_count == 1


async def test_failover_decision_logged_on_rate_limit(audit_log: CollectingAuditLog) -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderRateLimitError("rate limited", "primary"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="rate_limit", model="fallback:fb-model")],
        audit_log=audit_log,
    )
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.event == "router.failover"
    assert entry.level == "info"
    assert entry.provider_name == "primary"
    assert entry.detail["error_category"] == "rate_limit"
    assert entry.detail["fallback_model"] == "fallback:fb-model"
    assert entry.detail["fallback_on"] == "rate_limit"
    assert entry.detail["error_type"] == "ProviderRateLimitError"


async def test_failover_decision_logged_on_timeout(audit_log: CollectingAuditLog) -> None:
    from meridian_sdk_provider import ProviderTimeoutError

    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderTimeoutError("timed out", "primary"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="timeout", model="fallback:fb-model")],
        audit_log=audit_log,
    )
    [e async for e in router.call(make_opts())]

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.event == "router.failover"
    assert entry.detail["error_category"] == "timeout"


async def test_failover_decision_logged_on_5xx(audit_log: CollectingAuditLog) -> None:
    from meridian_sdk_provider import ProviderServerError

    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderServerError("server error", "primary"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="5xx", model="fallback:fb-model")],
        audit_log=audit_log,
    )
    [e async for e in router.call(make_opts())]

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.event == "router.failover"
    assert entry.detail["error_category"] == "5xx"


async def test_failover_decision_logged_on_any(audit_log: CollectingAuditLog) -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ValueError("unexpected"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="any", model="fallback:fb-model")],
        audit_log=audit_log,
    )
    [e async for e in router.call(make_opts())]

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.event == "router.failover"
    assert entry.detail["error_category"] == "any"
    assert entry.detail["fallback_on"] == "any"


async def test_failover_decision_not_logged_when_no_fallback_matches(
    audit_log: CollectingAuditLog,
) -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderRateLimitError("rate limited", "primary"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="timeout", model="fallback:fb-model")],
        audit_log=audit_log,
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass

    # Only the failure entry; no router.failover since no rule matched.
    assert len(audit_log.entries) == 1
    assert audit_log.entries[0].event == "provider.call.failed"


async def test_failover_decision_not_logged_on_success(audit_log: CollectingAuditLog) -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
        fallbacks=[FallbackRule(on="any", model="p:m")],
        audit_log=audit_log,
    )
    [e async for e in router.call(make_opts())]
    assert len(audit_log.entries) == 0


async def test_failover_decision_includes_session_id(audit_log: CollectingAuditLog) -> None:
    primary = FakeProvider(
        name="primary",
        raise_on_call=ProviderRateLimitError("rate limited", "primary"),
    )
    fallback = FakeProvider(name="fallback")
    router = _router(
        rules=[ModelRoutingRule(model="primary:m")],
        providers={"primary": primary, "fallback": fallback},
        fallbacks=[FallbackRule(on="any", model="fallback:fb-model")],
        audit_log=audit_log,
    )
    [e async for e in router.call(make_opts(session_id="sess-456"))]

    entry = next(e for e in audit_log.entries if e.event == "router.failover")
    assert entry.session_id == "sess-456"


# ─── Audit log ────────────────────────────────────────────────────────────────


async def test_audit_log_written_on_primary_failure(audit_log: CollectingAuditLog) -> None:
    provider = FakeProvider(
        name="p",
        raise_on_call=ProviderRateLimitError("rate limited", "p"),
    )
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
        audit_log=audit_log,
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts()):
            pass

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.level == "error"
    assert entry.provider_name == "p"
    assert entry.event == "provider.call.failed"


async def test_audit_log_not_written_on_success(audit_log: CollectingAuditLog) -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
        audit_log=audit_log,
    )
    async for _ in router.call(make_opts()):
        pass
    assert len(audit_log.entries) == 0


async def test_audit_log_includes_session_id(audit_log: CollectingAuditLog) -> None:
    provider = FakeProvider(name="p", raise_on_call=ProviderRateLimitError("err", "p"))
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
        audit_log=audit_log,
    )
    with pytest.raises(ProviderRateLimitError):
        async for _ in router.call(make_opts(session_id="sess-123")):
            pass

    assert audit_log.entries[0].session_id == "sess-123"


# ─── Token counting ───────────────────────────────────────────────────────────


async def test_count_tokens_delegates_to_capable_provider() -> None:
    provider = FakeProvider(name="p", capabilities=ProviderCapabilities(count_tokens=True))
    router = _router(rules=[], providers={"p": provider})
    from meridian_sdk_provider import ModelCountReq

    result = await router.count_tokens(ModelCountReq(model="p:m", messages=[]))
    assert result.input_tokens == 42


async def test_count_tokens_falls_back_to_estimate() -> None:
    provider = FakeProvider(name="p", capabilities=ProviderCapabilities(count_tokens=False))
    router = _router(rules=[], providers={"p": provider})
    from meridian_sdk_provider import Message, ModelCountReq

    req = ModelCountReq(
        model="p:m",
        messages=[Message(role="user", content="hello world")],
    )
    result = await router.count_tokens(req)
    assert result.input_tokens >= 1


# ─── Provider registration ────────────────────────────────────────────────────


async def test_register_provider_after_init() -> None:
    router = _router(rules=[ModelRoutingRule(model="p:m")], providers={})
    provider = FakeProvider(name="p")
    router.register_provider(provider)
    events = [e async for e in router.call(make_opts())]
    assert len(events) == 3


# ─── model_call.started event log ────────────────────────────────────────────


async def test_model_call_started_recorded_on_success(
    event_log: CollectingModelCallEventLog,
) -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:specific")],
        providers={"p": provider},
        event_log=event_log,
    )
    [e async for e in router.call(make_opts(session_id="sess-1"))]

    assert len(event_log.started) == 1
    rec = event_log.started[0]
    assert rec.session_id == "sess-1"
    assert rec.routing_rule == "p:specific"
    assert rec.provider_name == "p"
    assert rec.model == "specific"


async def test_model_call_started_not_recorded_without_session(
    event_log: CollectingModelCallEventLog,
) -> None:
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
        event_log=event_log,
    )
    [e async for e in router.call(make_opts())]  # no session_id

    assert len(event_log.started) == 0


async def test_model_call_started_routing_rule_is_model_ref(
    event_log: CollectingModelCallEventLog,
) -> None:
    provider = FakeProvider(name="myp")
    router = _router(
        rules=[ModelRoutingRule(model="myp:claude-3")],
        providers={"myp": provider},
        event_log=event_log,
    )
    [e async for e in router.call(make_opts(session_id="sess-2"))]

    rec = event_log.started[0]
    assert rec.routing_rule == "myp:claude-3"
    assert rec.provider_name == "myp"
    assert rec.model == "claude-3"


async def test_model_call_started_failure_raises_routing_error_and_writes_audit(
    audit_log: CollectingAuditLog,
) -> None:
    failing_event_log = CollectingModelCallEventLog(
        raise_on_record=RuntimeError("event log unavailable")
    )
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
        audit_log=audit_log,
        event_log=failing_event_log,
    )
    with pytest.raises(RoutingError, match="model_call.started"):
        [e async for e in router.call(make_opts(session_id="sess-3"))]

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.event == "provider.call.failed"
    assert entry.provider_name == "<event_log>"


async def test_model_call_started_failure_provider_not_called(
    event_log: CollectingModelCallEventLog,
) -> None:
    failing_event_log = CollectingModelCallEventLog(raise_on_record=RuntimeError("unavailable"))
    provider = FakeProvider(name="p")
    router = _router(
        rules=[ModelRoutingRule(model="p:m")],
        providers={"p": provider},
        event_log=failing_event_log,
    )
    with pytest.raises(RoutingError):
        [e async for e in router.call(make_opts(session_id="sess-4"))]

    assert provider.call_count == 0


async def test_set_event_log_replaces_noop(event_log: CollectingModelCallEventLog) -> None:
    provider = FakeProvider(name="p")
    router = _router(rules=[ModelRoutingRule(model="p:m")], providers={"p": provider})
    router.set_event_log(event_log)

    [e async for e in router.call(make_opts(session_id="sess-5"))]

    assert len(event_log.started) == 1


# ─── capability-constraint helpers ──────────────────────────────────────────


class TestCapConstraintHelpers:
    def test_strip_skips_string_content_and_non_cache_blocks(self) -> None:
        from meridian_sdk_provider import CacheControl, Message, TextBlock
        from meridian_sdk_provider.router import _strip_cache_control

        string_msg = Message(role="system", content="plain text")
        block_msg = Message(
            role="user",
            content=[
                TextBlock(type="text", text="cached", cache_control=CacheControl()),
                TextBlock(type="text", text="plain", cache_control=None),
            ],
        )
        result = _strip_cache_control([string_msg, block_msg])

        assert result[0].content == "plain text"  # string content left untouched
        assert result[1].content[0].cache_control is None  # cached block stripped
        assert result[1].content[1].cache_control is None  # plain block unchanged

    def test_apply_constraints_keeps_thinking_when_supported(self) -> None:
        from meridian_sdk_provider.router import _apply_cap_constraints

        opts = ModelCallOpts(
            model="p:m",
            messages=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
            thinking_budget_tokens=512,
            stream=True,
        )
        caps = ProviderCapabilities(streaming=False, thinking=True)
        result = _apply_cap_constraints(opts, caps)

        assert result.stream is False  # streaming cleared
        assert result.enable_thinking is True  # thinking preserved
        assert result.thinking_budget_tokens == 512
