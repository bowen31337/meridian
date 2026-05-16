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
    RoutingCondition,
    TextDeltaEvent,
    TokenRange,
)
from tests.conftest import CollectingAuditLog, FakeProvider, make_opts


def _router(
    rules: list[ModelRoutingRule],
    providers: dict,
    fallbacks: list[FallbackRule] | None = None,
    audit_log: CollectingAuditLog | None = None,
) -> ModelRouter:
    policy = ModelRoutingPolicy(rules=rules, fallbacks=fallbacks or [])
    return ModelRouter(policy=policy, providers=providers, audit_log=audit_log)


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

    assert len(audit_log.entries) == 1
    assert audit_log.entries[0].provider_name == "fallback"
    assert audit_log.entries[0].event == "provider.call.failed"


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
    provider = FakeProvider(
        name="p", raise_on_call=ProviderRateLimitError("err", "p")
    )
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
    provider = FakeProvider(
        name="p", capabilities=ProviderCapabilities(count_tokens=True)
    )
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
