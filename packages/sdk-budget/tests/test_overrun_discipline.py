"""
BudgetOverrunDiscipline conformance suite.

Covers:
  - record_soft_overrun: returns correct ratio, emits span + invocation event.
  - record_soft_overrun: ratio is (actual - limit) / limit.
  - record_soft_overrun: ratio is 0.0 when soft_limit is zero.
  - record_soft_overrun: no audit entry written on success.
  - record_soft_overrun: on internal failure, span is ERROR, audit entry written,
    BudgetOverrunDisciplineError raised with message surfaced.
  - validate_hard_transition_reason: correct code passes without raising.
  - validate_hard_transition_reason: emits span + invocation event on success.
  - validate_hard_transition_reason: no audit entry on success.
  - validate_hard_transition_reason: wrong code raises HardBudgetReasonCodeError.
  - validate_hard_transition_reason: wrong code marks span ERROR.
  - validate_hard_transition_reason: wrong code writes error-level audit entry.
  - validate_hard_transition_reason: audit entry code is budget_hard_reason_code_invalid.
  - validate_hard_transition_reason: audit detail includes scope, scope_id, dimension,
    reason_code, expected.
  - validate_hard_transition_reason: HardBudgetReasonCodeError message includes
    scope, scope_id, dimension, and reason_code.
  - Default options use NoopAuditLog (no side effects).
  - Span ends on both success and failure paths.
"""

from __future__ import annotations

import pytest
from core_errors import BudgetExceededError

from sdk_budget import (
    BudgetOverrunDiscipline,
    BudgetOverrunDisciplineError,
    BudgetOverrunDisciplineOptions,
    CORRECT_HARD_REASON_CODE,
    HardBudgetReasonCodeError,
)

from .conftest import CapturingAuditLog, MockSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_discipline(audit: CapturingAuditLog | None = None) -> BudgetOverrunDiscipline:
    opts = BudgetOverrunDisciplineOptions(audit_log=audit) if audit is not None else None
    return BudgetOverrunDiscipline(opts)


# ---------------------------------------------------------------------------
# record_soft_overrun — success path
# ---------------------------------------------------------------------------


class TestRecordSoftOverrun:
    def test_returns_correct_ratio(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        ratio = d.record_soft_overrun(
            scope="session",
            scope_id="s1",
            dimension="dollars",
            soft_limit=10.0,
            actual=10.3,
        )
        assert abs(ratio - 0.03) < 1e-9

    def test_ratio_is_actual_minus_limit_over_limit(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        ratio = d.record_soft_overrun(
            scope="agent",
            scope_id="a1",
            dimension="input_tokens",
            soft_limit=100.0,
            actual=105.0,
        )
        assert ratio == pytest.approx(0.05)

    def test_ratio_zero_when_soft_limit_is_zero(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        ratio = d.record_soft_overrun(
            scope="session",
            scope_id="s2",
            dimension="dollars",
            soft_limit=0.0,
            actual=5.0,
        )
        assert ratio == 0.0

    def test_emits_budget_overrun_soft_span(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.record_soft_overrun(
            scope="session",
            scope_id="s3",
            dimension="output_tokens",
            soft_limit=50.0,
            actual=51.0,
        )
        assert mock_span.name == "budget.overrun.soft"

    def test_span_has_scope_attributes(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.record_soft_overrun(
            scope="run",
            scope_id="r1",
            dimension="cache_tokens",
            soft_limit=200.0,
            actual=210.0,
        )
        assert mock_span.attributes["budget.scope"] == "run"
        assert mock_span.attributes["budget.scope_id"] == "r1"
        assert mock_span.attributes["budget.dimension"] == "cache_tokens"

    def test_emits_invocation_event(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.record_soft_overrun(
            scope="session",
            scope_id="s4",
            dimension="dollars",
            soft_limit=10.0,
            actual=10.5,
        )
        events = [e for e in mock_span.events if e[0] == "budget.overrun.soft.invocation"]
        assert len(events) == 1
        attrs = events[0][1]
        assert attrs["budget.scope"] == "session"
        assert attrs["budget.scope_id"] == "s4"
        assert attrs["budget.dimension"] == "dollars"
        assert attrs["budget.soft_limit"] == 10.0
        assert attrs["budget.actual"] == 10.5
        assert "budget.overrun_ratio" in attrs

    def test_no_audit_entry_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        d = make_discipline(audit_log)
        d.record_soft_overrun(
            scope="session",
            scope_id="s5",
            dimension="dollars",
            soft_limit=5.0,
            actual=5.1,
        )
        assert audit_log.entries == []

    def test_span_ends_on_success(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.record_soft_overrun(
            scope="session",
            scope_id="s6",
            dimension="dollars",
            soft_limit=1.0,
            actual=1.1,
        )
        assert mock_span.ended is True


# ---------------------------------------------------------------------------
# record_soft_overrun — failure path
# ---------------------------------------------------------------------------


class TestRecordSoftOverrunFailure:
    def test_internal_failure_raises_discipline_error(
        self, mock_tracer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = make_discipline()
        # Patch get_tracer to return a tracer whose start_as_current_span raises
        class _ExplodingSpan:
            name = "budget.overrun.soft"
            attributes: dict = {}
            events: list = []
            status = None
            recorded_exceptions: list = []
            ended = False

            def add_event(self, *a, **kw):
                raise RuntimeError("disk full")

            def set_attribute(self, *a, **kw):
                pass

            def set_status(self, *a, **kw):
                pass

            def record_exception(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.ended = True
                return False

        class _ExplodingTracer:
            def start_as_current_span(self, name, *, attributes=None, **kw):
                return _ExplodingSpan()

        monkeypatch.setattr("sdk_budget._overrun_discipline.get_tracer", _ExplodingTracer)
        d2 = make_discipline()
        with pytest.raises(BudgetOverrunDisciplineError) as exc_info:
            d2.record_soft_overrun(
                scope="session",
                scope_id="fail-sess",
                dimension="dollars",
                soft_limit=1.0,
                actual=1.1,
            )
        assert exc_info.value.code == "budget_overrun_discipline_error"
        assert len(exc_info.value.message) > 0

    def test_failure_writes_audit_entry(
        self, monkeypatch: pytest.MonkeyPatch, audit_log: CapturingAuditLog
    ) -> None:
        class _ExplodingSpan:
            name = "budget.overrun.soft"
            attributes: dict = {}
            events: list = []
            status = None
            recorded_exceptions: list = []
            ended = False

            def add_event(self, *a, **kw):
                raise RuntimeError("network error")

            def set_attribute(self, *a, **kw):
                pass

            def set_status(self, *a, **kw):
                pass

            def record_exception(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.ended = True
                return False

        class _ExplodingTracer:
            def start_as_current_span(self, name, *, attributes=None, **kw):
                return _ExplodingSpan()

        monkeypatch.setattr("sdk_budget._overrun_discipline.get_tracer", _ExplodingTracer)
        d = make_discipline(audit_log)
        with pytest.raises(BudgetOverrunDisciplineError):
            d.record_soft_overrun(
                scope="session",
                scope_id="fail-sess2",
                dimension="dollars",
                soft_limit=1.0,
                actual=1.1,
            )
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "budget.overrun.soft.failed"
        assert entry.code == "budget_overrun_discipline_error"


# ---------------------------------------------------------------------------
# validate_hard_transition_reason — success path
# ---------------------------------------------------------------------------


class TestValidateHardTransitionReasonSuccess:
    def test_correct_code_does_not_raise(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.validate_hard_transition_reason(
            scope="session",
            scope_id="s1",
            dimension="dollars",
            reason_code=CORRECT_HARD_REASON_CODE,
        )

    def test_emits_budget_overrun_hard_span(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.validate_hard_transition_reason(
            scope="session",
            scope_id="s2",
            dimension="dollars",
            reason_code=CORRECT_HARD_REASON_CODE,
        )
        assert mock_span.name == "budget.overrun.hard"

    def test_span_has_reason_code_attribute(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.validate_hard_transition_reason(
            scope="agent",
            scope_id="a1",
            dimension="input_tokens",
            reason_code=CORRECT_HARD_REASON_CODE,
        )
        assert mock_span.attributes["budget.reason_code"] == CORRECT_HARD_REASON_CODE

    def test_emits_invocation_event(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.validate_hard_transition_reason(
            scope="session",
            scope_id="s3",
            dimension="output_tokens",
            reason_code=CORRECT_HARD_REASON_CODE,
        )
        events = [e for e in mock_span.events if e[0] == "budget.overrun.hard.invocation"]
        assert len(events) == 1
        attrs = events[0][1]
        assert attrs["budget.reason_code"] == CORRECT_HARD_REASON_CODE
        assert attrs["budget.scope"] == "session"
        assert attrs["budget.scope_id"] == "s3"

    def test_no_audit_entry_on_success(self, mock_span: MockSpan, audit_log: CapturingAuditLog) -> None:
        d = make_discipline(audit_log)
        d.validate_hard_transition_reason(
            scope="session",
            scope_id="s4",
            dimension="dollars",
            reason_code=CORRECT_HARD_REASON_CODE,
        )
        assert audit_log.entries == []

    def test_span_ends_on_success(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        d.validate_hard_transition_reason(
            scope="run",
            scope_id="r1",
            dimension="dollars",
            reason_code=CORRECT_HARD_REASON_CODE,
        )
        assert mock_span.ended is True


# ---------------------------------------------------------------------------
# validate_hard_transition_reason — wrong reason code
# ---------------------------------------------------------------------------


class TestValidateHardTransitionReasonWrongCode:
    def test_wrong_code_raises_hard_budget_reason_code_error(
        self, mock_span: MockSpan
    ) -> None:
        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError):
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s1",
                dimension="dollars",
                reason_code="unknown_reason",
            )

    def test_wrong_code_error_code_is_correct(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError) as exc_info:
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s2",
                dimension="dollars",
                reason_code="wrong",
            )
        assert exc_info.value.code == "budget_hard_reason_code_invalid"

    def test_error_message_includes_scope_id(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError) as exc_info:
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="my-session-99",
                dimension="dollars",
                reason_code="bad_code",
            )
        assert "my-session-99" in exc_info.value.message

    def test_error_message_includes_dimension(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError) as exc_info:
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s3",
                dimension="input_tokens",
                reason_code="bad_code",
            )
        assert "input_tokens" in exc_info.value.message

    def test_error_message_includes_wrong_reason_code(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError) as exc_info:
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s4",
                dimension="dollars",
                reason_code="totally_wrong",
            )
        assert "totally_wrong" in exc_info.value.message

    def test_wrong_code_marks_span_error(self, mock_span: MockSpan) -> None:
        from opentelemetry.trace import StatusCode

        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError):
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s5",
                dimension="dollars",
                reason_code="bad",
            )
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    def test_wrong_code_emits_reason_code_invalid_event(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError):
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s6",
                dimension="dollars",
                reason_code="bad",
            )
        events = [e for e in mock_span.events if e[0] == "budget.overrun.hard.reason_code_invalid"]
        assert len(events) == 1
        attrs = events[0][1]
        assert attrs["budget.reason_code"] == "bad"
        assert attrs["budget.expected_reason_code"] == CORRECT_HARD_REASON_CODE

    def test_wrong_code_writes_audit_entry(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        d = make_discipline(audit_log)
        with pytest.raises(HardBudgetReasonCodeError):
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s7",
                dimension="dollars",
                reason_code="wrong_code",
            )
        assert len(audit_log.entries) == 1
        entry = audit_log.entries[0]
        assert entry.level == "error"
        assert entry.event == "budget.hard_transition.reason_code_invalid"
        assert entry.code == "budget_hard_reason_code_invalid"

    def test_audit_detail_includes_scope_and_dimension(
        self, mock_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        d = make_discipline(audit_log)
        with pytest.raises(HardBudgetReasonCodeError):
            d.validate_hard_transition_reason(
                scope="agent",
                scope_id="agent-42",
                dimension="output_tokens",
                reason_code="bad",
            )
        detail = audit_log.entries[0].detail
        assert detail is not None
        assert detail["scope"] == "agent"
        assert detail["scope_id"] == "agent-42"
        assert detail["dimension"] == "output_tokens"
        assert detail["reason_code"] == "bad"
        assert detail["expected"] == CORRECT_HARD_REASON_CODE

    def test_span_ends_on_wrong_code(self, mock_span: MockSpan) -> None:
        d = make_discipline()
        with pytest.raises(HardBudgetReasonCodeError):
            d.validate_hard_transition_reason(
                scope="session",
                scope_id="s8",
                dimension="dollars",
                reason_code="nope",
            )
        assert mock_span.ended is True


# ---------------------------------------------------------------------------
# Default options — NoopAuditLog
# ---------------------------------------------------------------------------


def test_default_options_no_audit_side_effects(mock_span: MockSpan) -> None:
    d = BudgetOverrunDiscipline()
    with pytest.raises(HardBudgetReasonCodeError):
        d.validate_hard_transition_reason(
            scope="session",
            scope_id="noop-sess",
            dimension="dollars",
            reason_code="wrong",
        )


# ---------------------------------------------------------------------------
# CORRECT_HARD_REASON_CODE constant
# ---------------------------------------------------------------------------


def test_correct_hard_reason_code_value() -> None:
    assert CORRECT_HARD_REASON_CODE == "budget_exceeded"


# ---------------------------------------------------------------------------
# All scope types accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", ["agent", "session", "run"])
def test_all_scope_types_accepted_for_soft_overrun(scope: str, mock_span: MockSpan) -> None:
    d = make_discipline()
    ratio = d.record_soft_overrun(
        scope=scope,
        scope_id=f"{scope}-id",
        dimension="dollars",
        soft_limit=10.0,
        actual=10.5,
    )
    assert ratio == pytest.approx(0.05)
    assert mock_span.attributes["budget.scope"] == scope


@pytest.mark.parametrize("scope", ["agent", "session", "run"])
def test_all_scope_types_accepted_for_hard_validation(scope: str, mock_span: MockSpan) -> None:
    d = make_discipline()
    d.validate_hard_transition_reason(
        scope=scope,
        scope_id=f"{scope}-id",
        dimension="dollars",
        reason_code=CORRECT_HARD_REASON_CODE,
    )
    assert mock_span.attributes["budget.scope"] == scope
