"""Unit coverage for sdk-budget telemetry cause-recording and re-raise branches."""

from __future__ import annotations

import pytest
from core_errors import BudgetExceededError, MeridianError
from opentelemetry import trace
from sdk_budget import CostAccumulator, CostAccumulatorOptions, PriceBook
from sdk_budget._cost_accumulator import CostAccumulatorError
from sdk_budget._overrun_discipline import (
    BudgetOverrunDiscipline,
    BudgetOverrunDisciplineError,
    BudgetOverrunDisciplineOptions,
)
from sdk_budget._telemetry import (
    get_tracer,
    record_budget_exceeded,
    record_cost_accumulate_failure,
    record_hard_transition_failure,
    record_soft_overrun_failure,
)

from .conftest import CapturingAuditLog, MockSpan

TS = "2026-01-01T00:00:00Z"


def test_get_tracer_returns_tracer() -> None:
    assert isinstance(get_tracer(), trace.Tracer)


# --- telemetry: span.record_exception(error.cause) branches ---


def test_record_budget_exceeded_records_cause() -> None:
    span = MockSpan()
    cause = ValueError("root")
    err = BudgetExceededError(message="exceeded", timestamp=TS, cause=cause)
    record_budget_exceeded(span, err, dimension="usd", limit=1.0, actual=2.0)
    assert cause in span.recorded_exceptions


def test_record_soft_overrun_failure_records_cause() -> None:
    span = MockSpan()
    cause = ValueError("root")
    err = MeridianError(code="x", message="m", timestamp=TS, cause=cause)
    record_soft_overrun_failure(span, err)
    assert cause in span.recorded_exceptions


def test_record_hard_transition_failure_records_cause() -> None:
    span = MockSpan()
    cause = ValueError("root")
    err = MeridianError(code="x", message="m", timestamp=TS, cause=cause)
    record_hard_transition_failure(span, err)
    assert cause in span.recorded_exceptions
    assert any(e[0] == "budget.overrun.hard.failure" for e in span.events)


# --- telemetry: error.cause is None (no record_exception) branches ---


def test_record_cost_accumulate_failure_without_cause() -> None:
    span = MockSpan()
    err = MeridianError(code="x", message="m", timestamp=TS)
    record_cost_accumulate_failure(span, err)
    assert span.recorded_exceptions == []
    assert any(e[0] == "cost.accumulate.failure" for e in span.events)


def test_record_soft_overrun_failure_without_cause() -> None:
    span = MockSpan()
    err = MeridianError(code="x", message="m", timestamp=TS)
    record_soft_overrun_failure(span, err)
    assert span.recorded_exceptions == []


def test_record_hard_transition_failure_without_cause() -> None:
    span = MockSpan()
    err = MeridianError(code="x", message="m", timestamp=TS)
    record_hard_transition_failure(span, err)
    assert span.recorded_exceptions == []


# --- re-raise / generic-except branches in the runtimes ---


def _price_book() -> PriceBook:
    return PriceBook.from_dict(
        {"openai": {"gpt-4o": {"input": 2.5, "output": 10.0, "cache_read": 0.0}}}
    )


def test_accumulate_reraises_cost_accumulator_error(
    mock_span: MockSpan, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise CostAccumulatorError(message="inner", timestamp=TS)

    monkeypatch.setattr(
        "sdk_budget._cost_accumulator.record_cost_accumulate_invocation", _boom
    )
    audit = CapturingAuditLog()
    acc = CostAccumulator(_price_book(), CostAccumulatorOptions(audit_log=audit))
    with pytest.raises(CostAccumulatorError) as exc_info:
        acc.accumulate(
            scope="session",
            scope_id="s1",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1,
            completion_tokens=1,
        )
    assert exc_info.value.message == "inner"
    assert audit.entries == []


def test_record_soft_overrun_reraises_discipline_error(
    mock_span: MockSpan, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise BudgetOverrunDisciplineError(message="inner", timestamp=TS)

    monkeypatch.setattr(
        "sdk_budget._overrun_discipline.record_soft_overrun_invocation", _boom
    )
    audit = CapturingAuditLog()
    disc = BudgetOverrunDiscipline(BudgetOverrunDisciplineOptions(audit_log=audit))
    with pytest.raises(BudgetOverrunDisciplineError) as exc_info:
        disc.record_soft_overrun(
            scope="session", scope_id="s1", dimension="usd", soft_limit=1.0, actual=2.0
        )
    assert exc_info.value.message == "inner"
    assert audit.entries == []


def test_validate_hard_transition_wraps_unexpected_error(
    mock_span: MockSpan, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(
        "sdk_budget._overrun_discipline.record_hard_transition_invocation", _boom
    )
    audit = CapturingAuditLog()
    disc = BudgetOverrunDiscipline(BudgetOverrunDisciplineOptions(audit_log=audit))
    with pytest.raises(BudgetOverrunDisciplineError):
        disc.validate_hard_transition_reason(
            scope="session", scope_id="s1", dimension="usd", reason_code="budget_exceeded"
        )
    assert any(e.event == "budget.overrun.hard.failed" for e in audit.entries)
