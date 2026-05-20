from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from core_errors import AuditLog, AuditLogEntry, BudgetExceededError, NoopAuditLog

from ._telemetry import (
    get_tracer,
    record_budget_exceeded,
    record_budget_warning,
    record_invocation_event,
)
from ._types import (
    BUDGET_DIMENSIONS,
    BudgetCheckResult,
    BudgetConfig,
    BudgetViolation,
    UsageSnapshot,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class BudgetCheckerOptions:
    """Options injected by the host into BudgetChecker."""

    audit_log: AuditLog = field(default_factory=NoopAuditLog)


class BudgetChecker:
    """Synchronous budget enforcement engine.

    Call check() before each model invocation (or at any metering point) to
    enforce per-agent / per-session / per-run-span budgets across five dimensions:
    input_tokens, output_tokens, cache_tokens, dollars, wall_clock_seconds.

    Soft violations emit a "budget.warning" span event and are collected in the
    returned BudgetCheckResult.  Hard violations emit a "budget.exceeded" span
    event, write an error-level audit entry, and raise BudgetExceededError — the
    caller is expected to surface this to the end user.
    """

    def __init__(self, options: BudgetCheckerOptions | None = None) -> None:
        self._opts = options or BudgetCheckerOptions()

    def check(self, config: BudgetConfig, usage: UsageSnapshot) -> BudgetCheckResult:
        """Evaluate *usage* against *config* thresholds.

        Returns BudgetCheckResult (possibly with soft warnings) when all hard
        limits pass.  Raises BudgetExceededError on the first hard violation.
        """
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "budget.check",
            attributes={
                "budget.scope": config.scope,
                "budget.scope_id": config.scope_id,
            },
        ) as span:
            record_invocation_event(span, config.scope, config.scope_id, now)
            warnings: list[BudgetViolation] = []

            for dim in BUDGET_DIMENSIONS:
                actual = float(getattr(usage, dim))

                if config.soft is not None:
                    soft_limit = getattr(config.soft, dim)
                    if soft_limit is not None and actual >= float(soft_limit):
                        violation = BudgetViolation(
                            scope=config.scope,
                            scope_id=config.scope_id,
                            dimension=dim,
                            threshold_type="soft",
                            limit=float(soft_limit),
                            actual=actual,
                        )
                        warnings.append(violation)
                        record_budget_warning(
                            span,
                            config.scope,
                            config.scope_id,
                            dim,
                            float(soft_limit),
                            actual,
                            now,
                        )
                        self._opts.audit_log.write(
                            AuditLogEntry(
                                level="warn",
                                event="budget.warning",
                                code="budget_warning",
                                timestamp=now,
                                detail={
                                    "scope": config.scope,
                                    "scope_id": config.scope_id,
                                    "dimension": dim,
                                    "limit": float(soft_limit),
                                    "actual": actual,
                                },
                            )
                        )

                if config.hard is not None:
                    hard_limit = getattr(config.hard, dim)
                    if hard_limit is not None and actual >= float(hard_limit):
                        message = (
                            f"{config.scope} {config.scope_id!r} exceeded hard "
                            f"{dim} budget: {actual} >= {float(hard_limit)}"
                        )
                        error = BudgetExceededError(message=message, timestamp=now)
                        record_budget_exceeded(span, error, dim, float(hard_limit), actual)
                        self._opts.audit_log.write(
                            AuditLogEntry(
                                level="error",
                                event="budget.exceeded",
                                code="budget_exceeded",
                                timestamp=now,
                                detail={
                                    "scope": config.scope,
                                    "scope_id": config.scope_id,
                                    "dimension": dim,
                                    "limit": float(hard_limit),
                                    "actual": actual,
                                },
                            )
                        )
                        raise error

            return BudgetCheckResult(warnings=warnings)
