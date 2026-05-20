"""
CostAccumulator conformance suite.

Covers:

  accumulate():
    - Opens a "cost.accumulate" span per call.
    - Span carries cost.scope, cost.scope_id, provider, model attributes.
    - Attaches a "cost.accumulate.invocation" event with cost.dollars.
    - Accumulates input_tokens across multiple calls (same scope).
    - Accumulates output_tokens across multiple calls (same scope).
    - Accumulates cache_tokens across multiple calls (same scope).
    - Accumulates dollars using the price book.
    - Returns updated snapshot after each call.
    - Separate (scope, scope_id) pairs are tracked independently.
    - All three scope types (agent, session, run) are accepted.
    - cache_creation_tokens and cache_read_tokens default to 0.
    - Unknown (provider, model) yields 0.0 dollars (no error).
    - Span ends on success path.

  accumulate() — failure path:
    - Raises CostAccumulatorError when price_book.cost_for_delta raises.
    - Sets span status to ERROR on failure.
    - Attaches "cost.accumulate.failure" event on failure.
    - Writes error-level audit entry on failure.
    - CostAccumulatorError has code "cost_accumulate_error".
    - Span ends on failure path.

  snapshot():
    - Returns zero-valued ScopeCounters for unseen scope.
    - Returns a copy (mutations do not affect accumulator state).
    - Reflects accumulated state after multiple accumulate() calls.

  default options:
    - CostAccumulatorOptions defaults to NoopAuditLog (no side effects).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from core_errors import AuditLog, AuditLogEntry
from opentelemetry.trace import StatusCode

from sdk_budget import (
    CostAccumulator,
    CostAccumulatorError,
    CostAccumulatorOptions,
    ModelPricing,
    PriceBook,
    ScopeCounters,
)

from .conftest import CapturingAuditLog, MockSpan


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_price_book() -> PriceBook:
    return PriceBook.from_dict(
        {
            "openai": {
                "gpt-4o": {
                    "input": 2.50,
                    "output": 10.00,
                    "cache_creation": 3.75,
                    "cache_read": 1.25,
                }
            }
        }
    )


def make_accumulator(
    audit: CapturingAuditLog | None = None,
    price_book: PriceBook | None = None,
) -> CostAccumulator:
    opts = CostAccumulatorOptions(audit_log=audit) if audit is not None else None
    return CostAccumulator(price_book or make_price_book(), opts)


# ---------------------------------------------------------------------------
# Span lifecycle and invocation event
# ---------------------------------------------------------------------------


class TestSpanAndInvocationEvent:
    def test_opens_cost_accumulate_span(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(
            scope="session",
            scope_id="s1",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert mock_span.name == "cost.accumulate"

    def test_span_carries_scope_attributes(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(
            scope="agent",
            scope_id="a1",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert mock_span.attributes["cost.scope"] == "agent"
        assert mock_span.attributes["cost.scope_id"] == "a1"
        assert mock_span.attributes["provider"] == "openai"
        assert mock_span.attributes["model"] == "gpt-4o"

    def test_attaches_invocation_event(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(
            scope="session",
            scope_id="s1",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=0,
        )
        inv_events = [e for e in mock_span.events if e[0] == "cost.accumulate.invocation"]
        assert len(inv_events) == 1

    def test_invocation_event_carries_dollars(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(
            scope="session",
            scope_id="s1",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1000,  # $2.50
            completion_tokens=0,
        )
        inv_events = [e for e in mock_span.events if e[0] == "cost.accumulate.invocation"]
        attrs = inv_events[0][1]
        assert attrs["cost.dollars"] == pytest.approx(2.50)

    def test_span_ends_on_success(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(
            scope="run",
            scope_id="r1",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert mock_span.ended is True


# ---------------------------------------------------------------------------
# Counter accumulation
# ---------------------------------------------------------------------------


class TestCounterAccumulation:
    def test_accumulates_input_tokens(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=100, completion_tokens=0)
        result = acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                                prompt_tokens=200, completion_tokens=0)
        assert result.input_tokens == 300

    def test_accumulates_output_tokens(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=0, completion_tokens=50)
        result = acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                                prompt_tokens=0, completion_tokens=75)
        assert result.output_tokens == 125

    def test_accumulates_cache_tokens(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=0, completion_tokens=0,
                       cache_creation_tokens=100, cache_read_tokens=50)
        result = acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                                prompt_tokens=0, completion_tokens=0,
                                cache_creation_tokens=200, cache_read_tokens=25)
        assert result.cache_tokens == 375

    def test_accumulates_dollars(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        # call 1: 1000 prompt → $2.50
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=1000, completion_tokens=0)
        # call 2: 500 completion → $5.00
        result = acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                                prompt_tokens=0, completion_tokens=500)
        assert result.dollars == pytest.approx(7.50)

    def test_returns_updated_snapshot(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        result = acc.accumulate(scope="run", scope_id="r1", provider="openai", model="gpt-4o",
                                prompt_tokens=100, completion_tokens=50)
        assert isinstance(result, ScopeCounters)
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    def test_unknown_provider_model_yields_zero_dollars(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        result = acc.accumulate(scope="session", scope_id="s1",
                                provider="mystery", model="unknown",
                                prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert result.dollars == 0.0
        assert result.input_tokens == 1_000_000

    def test_cache_tokens_default_to_zero(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        result = acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                                prompt_tokens=0, completion_tokens=0)
        assert result.cache_tokens == 0


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------


class TestScopeIsolation:
    def test_different_scope_ids_tracked_independently(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=100, completion_tokens=0)
        result_s2 = acc.accumulate(scope="session", scope_id="s2", provider="openai", model="gpt-4o",
                                   prompt_tokens=200, completion_tokens=0)
        assert result_s2.input_tokens == 200

    def test_agent_session_run_scopes_independent(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(scope="agent", scope_id="id1", provider="openai", model="gpt-4o",
                       prompt_tokens=10, completion_tokens=0)
        acc.accumulate(scope="session", scope_id="id1", provider="openai", model="gpt-4o",
                       prompt_tokens=20, completion_tokens=0)
        acc.accumulate(scope="run", scope_id="id1", provider="openai", model="gpt-4o",
                       prompt_tokens=30, completion_tokens=0)
        assert acc.snapshot("agent", "id1").input_tokens == 10
        assert acc.snapshot("session", "id1").input_tokens == 20
        assert acc.snapshot("run", "id1").input_tokens == 30


# ---------------------------------------------------------------------------
# Scope type acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["agent", "session", "run"])
def test_all_scope_types_accepted(scope: str, mock_span: MockSpan) -> None:
    acc = make_accumulator()
    result = acc.accumulate(
        scope=scope,  # type: ignore[arg-type]
        scope_id=f"{scope}-id",
        provider="openai",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert result.input_tokens == 100
    assert result.output_tokens == 50


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_returns_zero_counters_for_unseen_scope(self) -> None:
        acc = make_accumulator()
        result = acc.snapshot("agent", "never-seen")
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cache_tokens == 0
        assert result.dollars == 0.0

    def test_reflects_accumulated_state(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=500, completion_tokens=250)
        snap = acc.snapshot("session", "s1")
        assert snap.input_tokens == 500
        assert snap.output_tokens == 250

    def test_returns_copy_mutations_do_not_affect_state(self, mock_span: MockSpan) -> None:
        acc = make_accumulator()
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=100, completion_tokens=0)
        snap = acc.snapshot("session", "s1")
        snap.input_tokens = 9999
        assert acc.snapshot("session", "s1").input_tokens == 100


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestFailurePath:
    def _broken_price_book(self) -> PriceBook:
        pb = MagicMock(spec=PriceBook)
        pb.cost_for_delta.side_effect = RuntimeError("injected failure")
        return pb

    def test_raises_cost_accumulator_error(self, mock_span: MockSpan) -> None:
        acc = CostAccumulator(self._broken_price_book())
        with pytest.raises(CostAccumulatorError):
            acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                           prompt_tokens=100, completion_tokens=0)

    def test_error_code_is_cost_accumulate_error(self, mock_span: MockSpan) -> None:
        acc = CostAccumulator(self._broken_price_book())
        with pytest.raises(CostAccumulatorError) as exc_info:
            acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                           prompt_tokens=100, completion_tokens=0)
        assert exc_info.value.code == "cost_accumulate_error"

    def test_span_marked_error_on_failure(self, mock_span: MockSpan) -> None:
        acc = CostAccumulator(self._broken_price_book())
        with pytest.raises(CostAccumulatorError):
            acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                           prompt_tokens=100, completion_tokens=0)
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    def test_failure_event_attached_on_failure(self, mock_span: MockSpan) -> None:
        acc = CostAccumulator(self._broken_price_book())
        with pytest.raises(CostAccumulatorError):
            acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                           prompt_tokens=100, completion_tokens=0)
        failure_events = [e for e in mock_span.events if e[0] == "cost.accumulate.failure"]
        assert len(failure_events) == 1

    def test_audit_log_written_on_failure(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        acc = CostAccumulator(self._broken_price_book(), CostAccumulatorOptions(audit_log=audit_log))
        with pytest.raises(CostAccumulatorError):
            acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                           prompt_tokens=100, completion_tokens=0)
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "cost.accumulate.failed"
        assert entry.code == "cost_accumulate_error"

    def test_span_ends_on_failure(self, mock_span: MockSpan) -> None:
        acc = CostAccumulator(self._broken_price_book())
        with pytest.raises(CostAccumulatorError):
            acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                           prompt_tokens=100, completion_tokens=0)
        assert mock_span.ended is True


# ---------------------------------------------------------------------------
# Default options (NoopAuditLog)
# ---------------------------------------------------------------------------


def test_default_options_no_audit_side_effects(mock_span: MockSpan) -> None:
    acc = CostAccumulator(MagicMock(spec=PriceBook, **{"cost_for_delta.side_effect": RuntimeError("x")}))
    with pytest.raises(CostAccumulatorError):
        acc.accumulate(scope="session", scope_id="s1", provider="openai", model="gpt-4o",
                       prompt_tokens=0, completion_tokens=0)
