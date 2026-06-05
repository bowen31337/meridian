"""BudgetOverrunDiscipline — validates soft overrun ratios and hard-budget reason codes.

Enforces PRD §7.3:
- Average soft budget overrun < 5% of the configured soft limit.
- 100% of hard-budget transitions tagged with reason code "budget_exceeded".

Each public method opens an OTel span, records a structured invocation event, and
returns / raises.  On unexpected failure the span is marked ERROR, an error-level
audit entry is written, and the error is surfaced to the caller.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from core_errors import AuditLog, AuditLogEntry, MeridianError, NoopAuditLog

from ._telemetry import (
    get_tracer,
    record_hard_transition_failure,
    record_hard_transition_invocation,
    record_hard_transition_reason_code_invalid,
    record_soft_overrun_failure,
    record_soft_overrun_invocation,
)

CORRECT_HARD_REASON_CODE = "budget_exceeded"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class HardBudgetReasonCodeError(MeridianError):
    """Raised when a hard-budget transition carries an incorrect reason code."""

    def __init__(
        self,
        *,
        scope: str,
        scope_id: str,
        dimension: str,
        reason_code: str,
        timestamp: str,
    ) -> None:
        super().__init__(
            code="budget_hard_reason_code_invalid",
            message=(
                f"Hard-budget transition for {scope} {scope_id!r} on {dimension!r} "
                f"has incorrect reason code {reason_code!r}; "
                f"expected {CORRECT_HARD_REASON_CODE!r}"
            ),
            timestamp=timestamp,
        )


class BudgetOverrunDisciplineError(MeridianError):
    """Raised when BudgetOverrunDiscipline encounters an internal failure."""

    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="budget_overrun_discipline_error",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )


@dataclass
class BudgetOverrunDisciplineOptions:
    """Options injected by the host into BudgetOverrunDiscipline."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)


class BudgetOverrunDiscipline:
    """Validates soft overrun ratios and hard-budget transition reason codes.

    ``record_soft_overrun()`` computes (actual - soft_limit) / soft_limit and emits
    a ``budget.overrun.soft`` span.  ``validate_hard_transition_reason()`` asserts
    the reason code and emits a ``budget.overrun.hard`` span.

    Both methods raise on failure and write to the audit log before re-raising so
    callers can surface the error message to the end user.
    """

    def __init__(self, options: BudgetOverrunDisciplineOptions | None = None) -> None:
        self._opts = options or BudgetOverrunDisciplineOptions()

    def record_soft_overrun(
        self,
        *,
        scope: str,
        scope_id: str,
        dimension: str,
        soft_limit: float,
        actual: float,
    ) -> float:
        """Record a soft budget overrun and return the overrun ratio.

        overrun_ratio = (actual − soft_limit) / soft_limit.

        Emits a ``budget.overrun.soft`` OTel span with a structured invocation event.
        On internal failure marks the span ERROR, writes an audit entry, and raises
        ``BudgetOverrunDisciplineError``.
        """
        ts = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "budget.overrun.soft",
            attributes={
                "budget.scope": scope,
                "budget.scope_id": scope_id,
                "budget.dimension": dimension,
            },
        ) as span:
            try:
                overrun_ratio = (actual - soft_limit) / soft_limit if soft_limit > 0 else 0.0
                record_soft_overrun_invocation(
                    span,
                    scope=scope,
                    scope_id=scope_id,
                    dimension=dimension,
                    soft_limit=soft_limit,
                    actual=actual,
                    overrun_ratio=overrun_ratio,
                    timestamp=ts,
                )
                return overrun_ratio

            except BudgetOverrunDisciplineError:
                raise
            except Exception as exc:
                err = BudgetOverrunDisciplineError(
                    message=(
                        f"soft overrun recording failed for {scope} {scope_id!r} "
                        f"on {dimension!r}: {exc}"
                    ),
                    timestamp=ts,
                    cause=exc,
                )
                with contextlib.suppress(Exception):
                    record_soft_overrun_failure(span, err)
                self._opts.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="budget.overrun.soft.failed",
                        code=err.code,
                        timestamp=ts,
                        detail={
                            "scope": scope,
                            "scope_id": scope_id,
                            "dimension": dimension,
                            "message": err.message,
                        },
                    )
                )
                raise err from exc

    def validate_hard_transition_reason(
        self,
        *,
        scope: str,
        scope_id: str,
        dimension: str,
        reason_code: str,
    ) -> None:
        """Validate that a hard-budget transition uses the correct reason code.

        Emits a ``budget.overrun.hard`` OTel span with a structured invocation event.
        Raises ``HardBudgetReasonCodeError`` if *reason_code* is not
        ``"budget_exceeded"``.  On unexpected internal failure raises
        ``BudgetOverrunDisciplineError``.
        """
        ts = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "budget.overrun.hard",
            attributes={
                "budget.scope": scope,
                "budget.scope_id": scope_id,
                "budget.dimension": dimension,
                "budget.reason_code": reason_code,
            },
        ) as span:
            try:
                record_hard_transition_invocation(
                    span,
                    scope=scope,
                    scope_id=scope_id,
                    dimension=dimension,
                    reason_code=reason_code,
                    timestamp=ts,
                )

                if reason_code != CORRECT_HARD_REASON_CODE:
                    err = HardBudgetReasonCodeError(
                        scope=scope,
                        scope_id=scope_id,
                        dimension=dimension,
                        reason_code=reason_code,
                        timestamp=ts,
                    )
                    record_hard_transition_reason_code_invalid(
                        span,
                        err,
                        reason_code=reason_code,
                        expected=CORRECT_HARD_REASON_CODE,
                    )
                    self._opts.audit_log.write(
                        AuditLogEntry(
                            level="error",
                            event="budget.hard_transition.reason_code_invalid",
                            code=err.code,
                            timestamp=ts,
                            detail={
                                "scope": scope,
                                "scope_id": scope_id,
                                "dimension": dimension,
                                "reason_code": reason_code,
                                "expected": CORRECT_HARD_REASON_CODE,
                                "message": err.message,
                            },
                        )
                    )
                    raise err

            except (HardBudgetReasonCodeError, BudgetOverrunDisciplineError):
                raise
            except Exception as exc:
                err2 = BudgetOverrunDisciplineError(
                    message=(
                        f"hard transition validation failed for {scope} {scope_id!r} "
                        f"on {dimension!r}: {exc}"
                    ),
                    timestamp=ts,
                    cause=exc,
                )
                with contextlib.suppress(Exception):
                    record_hard_transition_failure(span, err2)
                self._opts.audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="budget.overrun.hard.failed",
                        code=err2.code,
                        timestamp=ts,
                        detail={
                            "scope": scope,
                            "scope_id": scope_id,
                            "dimension": dimension,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc
