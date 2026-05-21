from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from core_errors import BudgetExceededError, MeridianError

from ._version import SDK_BUDGET_VERSION

_TRACER_NAME = "meridian.sdk-budget"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, SDK_BUDGET_VERSION)


def record_invocation_event(span: Span, scope: str, scope_id: str, timestamp: str) -> None:
    """Attaches a structured "budget.check.invocation" event to the active span."""
    span.add_event(
        "budget.check.invocation",
        {
            "budget.scope": scope,
            "budget.scope_id": scope_id,
            "timestamp": timestamp,
        },
    )


def record_budget_warning(
    span: Span,
    scope: str,
    scope_id: str,
    dimension: str,
    limit: float,
    actual: float,
    timestamp: str,
) -> None:
    """Attaches a structured "budget.warning" event for a soft-threshold breach."""
    span.add_event(
        "budget.warning",
        {
            "budget.scope": scope,
            "budget.scope_id": scope_id,
            "budget.dimension": dimension,
            "budget.limit": limit,
            "budget.actual": actual,
            "timestamp": timestamp,
        },
    )


def record_budget_exceeded(
    span: Span,
    error: BudgetExceededError,
    dimension: str,
    limit: float,
    actual: float,
) -> None:
    """Sets span status to ERROR, adds a "budget.exceeded" event, and records the cause."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    span.add_event(
        "budget.exceeded",
        {
            "error.code": error.code,
            "error.message": error.message,
            "budget.dimension": dimension,
            "budget.limit": limit,
            "budget.actual": actual,
        },
    )
    if error.cause is not None:
        span.record_exception(error.cause)


def record_cost_accumulate_invocation(
    span: Span,
    *,
    scope: str,
    scope_id: str,
    provider: str,
    model: str,
    dollars: float,
    timestamp: str,
) -> None:
    """Attach a structured ``cost.accumulate.invocation`` event to the active span."""
    span.add_event(
        "cost.accumulate.invocation",
        {
            "cost.scope": scope,
            "cost.scope_id": scope_id,
            "provider": provider,
            "model": model,
            "cost.dollars": dollars,
            "timestamp": timestamp,
        },
    )


def record_cost_accumulate_failure(span: Span, error: MeridianError) -> None:
    """Set span status to ERROR and attach a ``cost.accumulate.failure`` event."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    span.add_event(
        "cost.accumulate.failure",
        {
            "error.code": error.code,
            "error.message": error.message,
        },
    )
    if error.cause is not None:
        span.record_exception(error.cause)


def record_soft_overrun_invocation(
    span: Span,
    *,
    scope: str,
    scope_id: str,
    dimension: str,
    soft_limit: float,
    actual: float,
    overrun_ratio: float,
    timestamp: str,
) -> None:
    """Attach a structured ``budget.overrun.soft.invocation`` event to the active span."""
    span.add_event(
        "budget.overrun.soft.invocation",
        {
            "budget.scope": scope,
            "budget.scope_id": scope_id,
            "budget.dimension": dimension,
            "budget.soft_limit": soft_limit,
            "budget.actual": actual,
            "budget.overrun_ratio": overrun_ratio,
            "timestamp": timestamp,
        },
    )


def record_soft_overrun_failure(span: Span, error: MeridianError) -> None:
    """Set span status to ERROR and attach a ``budget.overrun.soft.failure`` event."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    span.add_event(
        "budget.overrun.soft.failure",
        {
            "error.code": error.code,
            "error.message": error.message,
        },
    )
    if error.cause is not None:
        span.record_exception(error.cause)


def record_hard_transition_invocation(
    span: Span,
    *,
    scope: str,
    scope_id: str,
    dimension: str,
    reason_code: str,
    timestamp: str,
) -> None:
    """Attach a structured ``budget.overrun.hard.invocation`` event to the active span."""
    span.add_event(
        "budget.overrun.hard.invocation",
        {
            "budget.scope": scope,
            "budget.scope_id": scope_id,
            "budget.dimension": dimension,
            "budget.reason_code": reason_code,
            "timestamp": timestamp,
        },
    )


def record_hard_transition_reason_code_invalid(
    span: Span,
    error: MeridianError,
    *,
    reason_code: str,
    expected: str,
) -> None:
    """Set span status to ERROR and attach a ``budget.overrun.hard.reason_code_invalid`` event."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    span.add_event(
        "budget.overrun.hard.reason_code_invalid",
        {
            "error.code": error.code,
            "error.message": error.message,
            "budget.reason_code": reason_code,
            "budget.expected_reason_code": expected,
        },
    )


def record_hard_transition_failure(span: Span, error: MeridianError) -> None:
    """Set span status to ERROR and attach a ``budget.overrun.hard.failure`` event."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    span.add_event(
        "budget.overrun.hard.failure",
        {
            "error.code": error.code,
            "error.message": error.message,
        },
    )
    if error.cause is not None:
        span.record_exception(error.cause)
