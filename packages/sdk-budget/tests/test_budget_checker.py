"""
BudgetChecker conformance suite.

Covers:
  - clean usage (no thresholds exceeded): span emitted, invocation event attached,
    no audit entries, empty warnings list returned.
  - soft threshold exceeded (per dimension): warning collected in result, audit
    entry at "warn" level, "budget.warning" event on span.
  - hard threshold exceeded (per dimension): BudgetExceededError raised, audit
    entry at "error" level, span marked ERROR with "budget.exceeded" event.
  - both soft and hard exceeded simultaneously: warning emitted then hard raises.
  - hard raises on the first violated dimension and stops iteration.
  - all three scope types (agent, session, run).
  - no thresholds configured: always passes, no audit entries.
  - span lifecycle: span ended on both success and failure paths.
  - BudgetCheckerOptions defaults to NoopAuditLog (no side-effects).
"""

from __future__ import annotations

import pytest
from core_errors import BudgetExceededError
from opentelemetry.trace import StatusCode

from sdk_budget import (
    BudgetChecker,
    BudgetCheckerOptions,
    BudgetConfig,
    BudgetThreshold,
    UsageSnapshot,
)

from .conftest import CapturingAuditLog, MockSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_checker(audit: CapturingAuditLog | None = None) -> BudgetChecker:
    opts = BudgetCheckerOptions(audit_log=audit) if audit is not None else None
    return BudgetChecker(opts)


# ---------------------------------------------------------------------------
# Clean usage — no thresholds exceeded
# ---------------------------------------------------------------------------


def test_clean_usage_returns_empty_result(mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(
        scope="session",
        scope_id="s1",
        soft=BudgetThreshold(input_tokens=1000),
        hard=BudgetThreshold(input_tokens=2000),
    )
    result = checker.check(config, UsageSnapshot(input_tokens=500))

    assert result.warnings == []
    assert audit_log.entries == []


def test_clean_usage_emits_invocation_event(mock_span: MockSpan) -> None:
    checker = make_checker()
    config = BudgetConfig(scope="session", scope_id="s1", hard=BudgetThreshold(dollars=10.0))
    checker.check(config, UsageSnapshot(dollars=1.0))

    assert mock_span.name == "budget.check"
    assert mock_span.attributes["budget.scope"] == "session"
    assert mock_span.attributes["budget.scope_id"] == "s1"
    invocation_events = [e for e in mock_span.events if e[0] == "budget.check.invocation"]
    assert len(invocation_events) == 1
    attrs = invocation_events[0][1]
    assert attrs["budget.scope"] == "session"
    assert attrs["budget.scope_id"] == "s1"


def test_clean_usage_span_ends(mock_span: MockSpan) -> None:
    checker = make_checker()
    checker.check(
        BudgetConfig(scope="agent", scope_id="a1"),
        UsageSnapshot(),
    )
    assert mock_span.ended is True


# ---------------------------------------------------------------------------
# Soft threshold exceeded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dimension", "snapshot_kwargs", "threshold_kwargs"),
    [
        ("input_tokens", {"input_tokens": 900}, {"input_tokens": 800}),
        ("output_tokens", {"output_tokens": 600}, {"output_tokens": 500}),
        ("cache_tokens", {"cache_tokens": 200}, {"cache_tokens": 100}),
        ("dollars", {"dollars": 5.5}, {"dollars": 5.0}),
        ("wall_clock_seconds", {"wall_clock_seconds": 31.0}, {"wall_clock_seconds": 30.0}),
    ],
)
def test_soft_threshold_exceeded_per_dimension(
    dimension: str,
    snapshot_kwargs: dict,
    threshold_kwargs: dict,
    mock_span: MockSpan,
    audit_log: CapturingAuditLog,
) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(
        scope="session",
        scope_id="sess-soft",
        soft=BudgetThreshold(**threshold_kwargs),
    )
    result = checker.check(config, UsageSnapshot(**snapshot_kwargs))

    assert len(result.warnings) == 1
    v = result.warnings[0]
    assert v.dimension == dimension
    assert v.threshold_type == "soft"
    assert v.scope == "session"
    assert v.scope_id == "sess-soft"

    warning_events = [e for e in mock_span.events if e[0] == "budget.warning"]
    assert len(warning_events) == 1
    assert warning_events[0][1]["budget.dimension"] == dimension

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.level == "warn"
    assert entry.event == "budget.warning"
    assert entry.code == "budget_warning"
    assert entry.detail is not None
    assert entry.detail["dimension"] == dimension


def test_soft_threshold_no_hard_does_not_raise(mock_span: MockSpan) -> None:
    checker = make_checker()
    config = BudgetConfig(
        scope="agent",
        scope_id="a1",
        soft=BudgetThreshold(dollars=1.0),
    )
    result = checker.check(config, UsageSnapshot(dollars=99.0))
    assert len(result.warnings) == 1
    assert result.warnings[0].threshold_type == "soft"


# ---------------------------------------------------------------------------
# Hard threshold exceeded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dimension", "snapshot_kwargs", "threshold_kwargs"),
    [
        ("input_tokens", {"input_tokens": 1001}, {"input_tokens": 1000}),
        ("output_tokens", {"output_tokens": 501}, {"output_tokens": 500}),
        ("cache_tokens", {"cache_tokens": 101}, {"cache_tokens": 100}),
        ("dollars", {"dollars": 10.01}, {"dollars": 10.0}),
        ("wall_clock_seconds", {"wall_clock_seconds": 60.1}, {"wall_clock_seconds": 60.0}),
    ],
)
def test_hard_threshold_exceeded_per_dimension(
    dimension: str,
    snapshot_kwargs: dict,
    threshold_kwargs: dict,
    mock_span: MockSpan,
    audit_log: CapturingAuditLog,
) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(
        scope="session",
        scope_id="sess-hard",
        hard=BudgetThreshold(**threshold_kwargs),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        checker.check(config, UsageSnapshot(**snapshot_kwargs))

    err = exc_info.value
    assert err.code == "budget_exceeded"
    assert "sess-hard" in err.message
    assert dimension in err.message

    exceeded_events = [e for e in mock_span.events if e[0] == "budget.exceeded"]
    assert len(exceeded_events) == 1
    assert exceeded_events[0][1]["budget.dimension"] == dimension

    assert mock_span.status is not None
    assert mock_span.status.status_code == StatusCode.ERROR

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.level == "error"
    assert entry.event == "budget.exceeded"
    assert entry.code == "budget_exceeded"
    assert entry.detail is not None
    assert entry.detail["dimension"] == dimension


def test_hard_exceeded_span_ends(mock_span: MockSpan) -> None:
    checker = make_checker()
    config = BudgetConfig(scope="run", scope_id="r1", hard=BudgetThreshold(dollars=1.0))
    with pytest.raises(BudgetExceededError):
        checker.check(config, UsageSnapshot(dollars=2.0))
    assert mock_span.ended is True


# ---------------------------------------------------------------------------
# Both soft and hard exceeded on same dimension
# ---------------------------------------------------------------------------


def test_soft_and_hard_both_exceeded(mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(
        scope="session",
        scope_id="s2",
        soft=BudgetThreshold(input_tokens=500),
        hard=BudgetThreshold(input_tokens=1000),
    )

    with pytest.raises(BudgetExceededError):
        checker.check(config, UsageSnapshot(input_tokens=1500))

    warning_events = [e for e in mock_span.events if e[0] == "budget.warning"]
    exceeded_events = [e for e in mock_span.events if e[0] == "budget.exceeded"]
    assert len(warning_events) == 1
    assert len(exceeded_events) == 1

    assert len(audit_log.entries) == 2
    assert audit_log.entries[0].level == "warn"
    assert audit_log.entries[1].level == "error"


# ---------------------------------------------------------------------------
# Hard raises on first violated dimension (stops iteration)
# ---------------------------------------------------------------------------


def test_hard_raises_on_first_dimension(mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(
        scope="session",
        scope_id="s3",
        hard=BudgetThreshold(input_tokens=100, output_tokens=100),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        checker.check(config, UsageSnapshot(input_tokens=200, output_tokens=200))

    assert "input_tokens" in exc_info.value.message
    assert len(audit_log.entries) == 1
    assert audit_log.entries[0].detail is not None
    assert audit_log.entries[0].detail["dimension"] == "input_tokens"


# ---------------------------------------------------------------------------
# Scope types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["agent", "session", "run"])
def test_all_scope_types_accepted(scope: str, mock_span: MockSpan) -> None:
    checker = make_checker()
    config = BudgetConfig(
        scope=scope,  # type: ignore[arg-type]
        scope_id=f"{scope}-id-1",
        hard=BudgetThreshold(dollars=100.0),
    )
    result = checker.check(config, UsageSnapshot(dollars=1.0))
    assert result.warnings == []
    assert mock_span.attributes["budget.scope"] == scope
    assert mock_span.attributes["budget.scope_id"] == f"{scope}-id-1"


# ---------------------------------------------------------------------------
# No thresholds configured
# ---------------------------------------------------------------------------


def test_no_thresholds_always_passes(mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(scope="run", scope_id="r0")
    result = checker.check(
        config,
        UsageSnapshot(
            input_tokens=10**9,
            output_tokens=10**9,
            cache_tokens=10**9,
            dollars=1_000_000.0,
            wall_clock_seconds=86400.0,
        ),
    )
    assert result.warnings == []
    assert audit_log.entries == []


# ---------------------------------------------------------------------------
# Default options (NoopAuditLog)
# ---------------------------------------------------------------------------


def test_default_options_no_audit_side_effects(mock_span: MockSpan) -> None:
    checker = BudgetChecker()
    config = BudgetConfig(scope="agent", scope_id="a-noop", soft=BudgetThreshold(dollars=1.0))
    result = checker.check(config, UsageSnapshot(dollars=5.0))
    assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# Boundary: exactly at threshold is exceeded (>=)
# ---------------------------------------------------------------------------


def test_exact_boundary_triggers_soft(mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(scope="session", scope_id="s-boundary", soft=BudgetThreshold(dollars=10.0))
    result = checker.check(config, UsageSnapshot(dollars=10.0))
    assert len(result.warnings) == 1


def test_one_below_boundary_does_not_trigger(mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(scope="session", scope_id="s-below", hard=BudgetThreshold(input_tokens=1000))
    result = checker.check(config, UsageSnapshot(input_tokens=999))
    assert result.warnings == []
    assert audit_log.entries == []


# ---------------------------------------------------------------------------
# Multiple soft warnings in one check
# ---------------------------------------------------------------------------


def test_multiple_soft_warnings_collected(mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
    checker = make_checker(audit_log)
    config = BudgetConfig(
        scope="agent",
        scope_id="a-multi",
        soft=BudgetThreshold(input_tokens=500, output_tokens=300, dollars=5.0),
    )
    result = checker.check(
        config,
        UsageSnapshot(input_tokens=600, output_tokens=400, dollars=6.0),
    )
    assert len(result.warnings) == 3
    dims = {v.dimension for v in result.warnings}
    assert dims == {"input_tokens", "output_tokens", "dollars"}
    assert len(audit_log.entries) == 3
    assert all(e.level == "warn" for e in audit_log.entries)
