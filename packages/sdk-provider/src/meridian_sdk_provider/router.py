from __future__ import annotations

import datetime
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

from .audit import AuditLog, AuditLogEntry, NoopAuditLog
from .errors import (
    NoProviderFoundError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    RoutingError,
)
from .protocol import ModelProvider, ProviderCapabilities
from .telemetry import get_tracer, record_invocation_event, record_provider_failure
from .types import (
    ContentBlock,
    Message,
    ModelCallOpts,
    ModelCountReq,
    ModelEvent,
    TextBlock,
    TokenCount,
)

# ─── Routing policy models ────────────────────────────────────────────────────


class TokenRange(BaseModel):
    """Inclusive/exclusive token-count bounds for a routing condition."""

    gt: int | None = None
    gte: int | None = None
    lt: int | None = None
    lte: int | None = None


class RoutingCondition(BaseModel):
    skill_id: str | None = None
    estimated_input_tokens: TokenRange | None = None
    metadata_match: dict[str, Any] | None = None
    role: str | None = None


class ModelRoutingRule(BaseModel):
    """A single declarative routing rule.

    ``model`` must be in ``provider_name:model_id`` form.
    When ``when`` is absent the rule matches every call (catch-all).
    """

    when: RoutingCondition | None = None
    model: str


class FallbackRule(BaseModel):
    """Fallback target tried when a primary call fails with the matching error class."""

    on: Literal["rate_limit", "timeout", "5xx", "any"]
    model: str


class ModelRoutingPolicy(BaseModel):
    """Full routing policy: ordered rules plus optional fallback list."""

    rules: list[ModelRoutingRule]
    fallbacks: list[FallbackRule] = Field(default_factory=list)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _parse_model_ref(model_ref: str) -> tuple[str, str]:
    """Split ``"provider:model"`` into ``(provider_name, model_id)``.

    Raises ``RoutingError`` if the ref is not in the required two-part form.
    """
    if ":" not in model_ref:
        raise RoutingError(
            f"Model ref '{model_ref}' must be in 'provider_name:model_id' form."
        )
    provider_name, _, model_id = model_ref.partition(":")
    return provider_name, model_id


def _condition_matches(cond: RoutingCondition, opts: ModelCallOpts) -> bool:
    if cond.skill_id is not None and opts.skill_id != cond.skill_id:
        return False
    if cond.role is not None and opts.role != cond.role:
        return False
    if cond.estimated_input_tokens is not None:
        r = cond.estimated_input_tokens
        t = opts.estimated_input_tokens
        if t is None:
            return False
        if r.gt is not None and not (t > r.gt):
            return False
        if r.gte is not None and not (t >= r.gte):
            return False
        if r.lt is not None and not (t < r.lt):
            return False
        if r.lte is not None and not (t <= r.lte):
            return False
    if cond.metadata_match is not None:
        for k, v in cond.metadata_match.items():
            if opts.metadata.get(k) != v:
                return False
    return True


def _find_rule(
    policy: ModelRoutingPolicy, opts: ModelCallOpts
) -> ModelRoutingRule | None:
    for rule in policy.rules:
        if rule.when is None or _condition_matches(rule.when, opts):
            return rule
    return None


def _error_category(exc: Exception) -> Literal["rate_limit", "timeout", "5xx"] | None:
    if isinstance(exc, ProviderRateLimitError):
        return "rate_limit"
    if isinstance(exc, ProviderTimeoutError):
        return "timeout"
    if isinstance(exc, ProviderServerError):
        return "5xx"
    return None


def _strip_cache_control(messages: list[Message]) -> list[Message]:
    """Return a copy of messages with all TextBlock.cache_control fields removed."""
    result: list[Message] = []
    for msg in messages:
        if isinstance(msg.content, list):
            cleaned: list[ContentBlock] = []
            for block in msg.content:
                if isinstance(block, TextBlock) and block.cache_control is not None:
                    block = block.model_copy(update={"cache_control": None})
                cleaned.append(block)
            msg = msg.model_copy(update={"content": cleaned})
        result.append(msg)
    return result


def _apply_cap_constraints(
    opts: ModelCallOpts, caps: ProviderCapabilities
) -> ModelCallOpts:
    """Return a copy of *opts* with unsupported capability flags cleared."""
    changes: dict[str, Any] = {}

    if not caps.streaming:
        changes["stream"] = False

    if not caps.thinking:
        changes["enable_thinking"] = False
        changes["thinking_budget_tokens"] = None

    if not caps.cache_control:
        needs_strip = any(
            isinstance(msg.content, list)
            and any(
                isinstance(b, TextBlock) and b.cache_control is not None
                for b in msg.content
            )
            for msg in opts.messages
        )
        if needs_strip:
            changes["messages"] = _strip_cache_control(opts.messages)

    return opts.model_copy(update=changes) if changes else opts


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


# ─── ModelRouter ──────────────────────────────────────────────────────────────


class ModelRouter:
    """Routes model calls to registered providers based on a declarative policy.

    Per invocation:
      1. Opens an OTel span ``"provider.call"`` with routing context.
      2. Evaluates routing rules to select the primary provider + model.
      3. Applies capability constraints (clears opts unsupported by provider).
      4. Calls the provider; on pre-stream failure, attempts configured fallbacks.
      5. On terminal failure: sets span ERROR status, writes an audit log entry,
         and re-raises the error to the caller.

    The span closes when the last event is yielded, when an exception escapes,
    or when the caller closes the generator early (``aclose()``).
    """

    def __init__(
        self,
        policy: ModelRoutingPolicy,
        providers: dict[str, ModelProvider] | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._policy = policy
        self._providers: dict[str, ModelProvider] = providers or {}
        self._audit = audit_log or NoopAuditLog()

    def register_provider(self, provider: ModelProvider) -> None:
        """Add or replace a provider in the registry by its ``name``."""
        self._providers[provider.name] = provider

    def _resolve(self, model_ref: str) -> tuple[ModelProvider, str]:
        provider_name, model_id = _parse_model_ref(model_ref)
        provider = self._providers.get(provider_name)
        if provider is None:
            raise NoProviderFoundError(
                f"Provider '{provider_name}' not registered "
                f"(known: {sorted(self._providers)})."
            )
        return provider, model_id

    def _write_audit_failure(
        self,
        exc: Exception,
        provider_name: str,
        provider_kind: str,
        model: str | None,
        opts: ModelCallOpts,
        routing_rule: str,
    ) -> None:
        self._audit.write(
            AuditLogEntry(
                level="error",
                event="provider.call.failed",
                provider_name=provider_name,
                provider_kind=provider_kind,
                model=model,
                session_id=opts.session_id,
                timestamp=_now_iso(),
                detail={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "routing_rule": routing_rule,
                },
            )
        )

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        """Stream model events for *opts* according to the routing policy.

        This is an async generator.  Use it as::

            async for event in router.call(opts):
                handle(event)
        """
        tracer = get_tracer()
        rule = _find_rule(self._policy, opts)
        rule_label = rule.model if rule is not None else "<no-match>"

        span = tracer.start_span(
            "provider.call",
            attributes={
                "routing.rule": rule_label,
                "model.role": opts.role or "",
                "model.skill_id": opts.skill_id or "",
                **({"session.id": opts.session_id} if opts.session_id else {}),
            },
        )
        try:
            if rule is None:
                err = NoProviderFoundError(
                    f"No routing rule matched: role={opts.role!r} "
                    f"skill_id={opts.skill_id!r} "
                    f"tokens={opts.estimated_input_tokens}"
                )
                record_provider_failure(span, err, provider_name="<none>", model="<none>")
                self._write_audit_failure(err, "<none>", "<none>", None, opts, rule_label)
                raise err

            provider, model_id = self._resolve(rule.model)
            effective_opts = _apply_cap_constraints(
                opts.model_copy(update={"model": model_id}),
                provider.capabilities,
            )

            record_invocation_event(
                span,
                provider_name=provider.name,
                provider_kind=provider.kind,
                model=model_id,
                session_id=opts.session_id,
                routing_rule=rule_label,
            )

            # Peek at the first event before yielding anything.  This lets us
            # attempt a fallback when the provider errors before sending data.
            gen = provider.call(effective_opts)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                return
            except Exception as primary_exc:
                record_provider_failure(
                    span, primary_exc, provider_name=provider.name, model=model_id
                )

                cat = _error_category(primary_exc)
                fb_rule = next(
                    (fb for fb in self._policy.fallbacks if fb.on in ("any", cat)),
                    None,
                )

                if fb_rule is None:
                    self._write_audit_failure(
                        primary_exc, provider.name, provider.kind, model_id, opts, rule_label
                    )
                    raise

                # Attempt the fallback provider.
                fb_provider, fb_model_id = self._resolve(fb_rule.model)
                fb_opts = _apply_cap_constraints(
                    opts.model_copy(update={"model": fb_model_id}),
                    fb_provider.capabilities,
                )
                record_invocation_event(
                    span,
                    provider_name=fb_provider.name,
                    provider_kind=fb_provider.kind,
                    model=fb_model_id,
                    session_id=opts.session_id,
                    routing_rule=f"fallback:{fb_rule.model}",
                )
                try:
                    async for event in fb_provider.call(fb_opts):
                        yield event
                    return
                except Exception as fb_exc:
                    record_provider_failure(
                        span, fb_exc, provider_name=fb_provider.name, model=fb_model_id
                    )
                    self._write_audit_failure(
                        fb_exc,
                        fb_provider.name,
                        fb_provider.kind,
                        fb_model_id,
                        opts,
                        rule_label,
                    )
                    raise fb_exc from primary_exc

            # Primary returned its first event — commit to this stream.
            yield first
            try:
                async for event in gen:
                    yield event
            except Exception as exc:
                record_provider_failure(
                    span, exc, provider_name=provider.name, model=model_id
                )
                self._write_audit_failure(
                    exc, provider.name, provider.kind, model_id, opts, rule_label
                )
                raise

        finally:
            span.end()

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        """Delegate to the first provider that declares ``count_tokens`` capability.

        Falls back to a character-based estimate when no such provider is registered.
        """
        for provider in self._providers.values():
            if provider.capabilities.count_tokens:
                return await provider.count_tokens(req)

        total_chars = sum(
            len(m.content)
            if isinstance(m.content, str)
            else sum(
                len(getattr(b, "text", "") or "") + len(getattr(b, "thinking", "") or "")
                for b in m.content
            )
            for m in req.messages
        )
        return TokenCount(input_tokens=max(1, total_chars // 4))

    async def close(self) -> None:
        """Close all registered provider adapters."""
        for provider in self._providers.values():
            await provider.close()
