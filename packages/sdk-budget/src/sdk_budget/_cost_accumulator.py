"""CostAccumulator — folds per-call usage.delta events into per-scope counters.

Each call to ``accumulate()`` opens an OTel span (``cost.accumulate``), appends
a ``cost.accumulate.invocation`` event, updates the in-memory counters for the
requested scope, and returns the updated snapshot.

On any internal failure the span is marked ERROR, a ``cost.accumulate.failure``
event is attached, an error-level audit entry is written, and a
``CostAccumulatorError`` is raised so callers can surface the message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from core_errors import AuditLog, AuditLogEntry, MeridianError, NoopAuditLog

from ._price_book import PriceBook
from ._telemetry import (
    get_tracer,
    record_cost_accumulate_failure,
    record_cost_accumulate_invocation,
)
from ._types import BudgetScope


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CostAccumulatorError(MeridianError):
    """Raised when CostAccumulator.accumulate() fails."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="cost_accumulate_error",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )


@dataclass
class ScopeCounters:
    """Accumulated usage for a single (scope, scope_id) pair."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    dollars: float = 0.0


@dataclass
class CostAccumulatorOptions:
    """Options injected by the host into CostAccumulator."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)


class CostAccumulator:
    """Accumulates per-call usage.delta events into per-scope counters.

    Maintains one ``ScopeCounters`` entry per ``(scope, scope_id)`` pair in
    memory.  Callers should provide the same accumulator instance across all
    model calls that belong to the same agent / session / run-span lifetime.

    ``accumulate()`` emits one OTel span per call so that distributed traces
    carry a complete cost trail.  On unexpected failure the error is surfaced
    to the caller via ``CostAccumulatorError`` *and* written to the audit log.
    """

    def __init__(
        self,
        price_book: PriceBook,
        options: CostAccumulatorOptions | None = None,
    ) -> None:
        self._price_book = price_book
        self._opts = options or CostAccumulatorOptions()
        self._counters: dict[tuple[str, str], ScopeCounters] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def accumulate(
        self,
        *,
        scope: BudgetScope,
        scope_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        timestamp: str | None = None,
    ) -> ScopeCounters:
        """Process one usage.delta and return the updated scope snapshot.

        Opens a ``cost.accumulate`` OTel span, records an invocation event,
        updates counters, and closes the span.  On failure sets span ERROR,
        writes to the audit log, and raises ``CostAccumulatorError``.
        """
        ts = timestamp or _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "cost.accumulate",
            attributes={
                "cost.scope": scope,
                "cost.scope_id": scope_id,
                "provider": provider,
                "model": model,
            },
        ) as span:
            try:
                dollars = self._price_book.cost_for_delta(
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                )

                key = (scope, scope_id)
                if key not in self._counters:
                    self._counters[key] = ScopeCounters()
                c = self._counters[key]
                c.input_tokens += prompt_tokens
                c.output_tokens += completion_tokens
                c.cache_tokens += cache_creation_tokens + cache_read_tokens
                c.dollars += dollars

                record_cost_accumulate_invocation(
                    span,
                    scope=scope,
                    scope_id=scope_id,
                    provider=provider,
                    model=model,
                    dollars=dollars,
                    timestamp=ts,
                )

                return ScopeCounters(
                    input_tokens=c.input_tokens,
                    output_tokens=c.output_tokens,
                    cache_tokens=c.cache_tokens,
                    dollars=c.dollars,
                )

            except CostAccumulatorError:
                raise
            except Exception as exc:
                err = CostAccumulatorError(
                    message=(
                        f"cost accumulation failed for {scope} {scope_id!r} "
                        f"({provider}/{model}): {exc}"
                    ),
                    timestamp=ts,
                    cause=exc,
                )
                record_cost_accumulate_failure(span, err)
                self._opts.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="cost.accumulate.failed",
                        code=err.code,
                        timestamp=ts,
                        detail={
                            "scope": scope,
                            "scope_id": scope_id,
                            "provider": provider,
                            "model": model,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

    def snapshot(self, scope: BudgetScope, scope_id: str) -> ScopeCounters:
        """Return a copy of the current counters for (scope, scope_id).

        Returns zero-valued counters if no delta has been accumulated yet.
        """
        c = self._counters.get((scope, scope_id))
        if c is None:
            return ScopeCounters()
        return ScopeCounters(
            input_tokens=c.input_tokens,
            output_tokens=c.output_tokens,
            cache_tokens=c.cache_tokens,
            dollars=c.dollars,
        )
