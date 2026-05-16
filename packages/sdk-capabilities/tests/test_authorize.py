"""
authorize() conformance test suite.

Covers:
  Glob satisfaction (_satisfies_glob via authorize)
    - unrestricted grant (no param) covers any required param.
    - glob grant covers matching concrete param.
    - glob grant does not cover non-matching concrete param.
    - scoped grant does not cover unscoped requirement.
    - exact literal param works as degenerate glob.

  Intersection check
    - all required satisfied → allowed.
    - one required missing → denied.
    - multiple missing → CapabilityDenied.missing contains all missing.
    - empty required set → always allowed.

  OTel span
    - span name is "capability.authorize".
    - span attributes include agent.id, session.id, capability.required.
    - span emits "capability.authorize" event with required + granted + timestamp.
    - span ended on success.
    - span ended on denial.
    - span set to ERROR on denial.

  Audit log
    - level="info" on success.
    - level="error" on denial.
    - event is "capability.authorize".
    - detail["allowed"] mirrors outcome.
    - detail["required"] lists required caps.
    - detail["missing"] is empty on success, non-empty on denial.
    - detail["args"] echoes the args mapping.
    - agent_id and session_id appear on the entry.

  Error surface
    - CapabilityDenied is raised on denial.
    - CapabilityDenied.missing is the frozenset of unsatisfied caps.
    - CapabilityDenied message includes the missing cap name.
"""
from __future__ import annotations

import pytest
from opentelemetry.trace import StatusCode

from sdk_capabilities import (
    Capability,
    CapabilityDenied,
    CapabilitySet,
    authorize,
    parse,
    parse_set,
)
from sdk_capabilities._audit import AuditLogEntry

from .conftest import CapturingAuditLog, MockSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cap(text: str) -> Capability:
    return parse(text)


def caps(*texts: str) -> CapabilitySet:
    return parse_set(texts)


# ===========================================================================
# Glob satisfaction — grant param treated as glob
# ===========================================================================

class TestGlobSatisfaction:
    def test_unrestricted_grant_covers_concrete_param(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("fs.read"),
            caps("fs.read[/workspace/foo.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_glob_grant_covers_matching_param(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("fs.read[/workspace/**]"),
            caps("fs.read[/workspace/src/main.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_glob_grant_covers_nested_path(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("fs.read[/workspace/**]"),
            caps("fs.read[/workspace/a/b/c/deep.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_glob_single_star_matches_within_segment(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("fs.read[/workspace/*.py]"),
            caps("fs.read[/workspace/main.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_glob_grant_does_not_cover_non_matching_param(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("fs.read[/workspace/**]"),
                caps("fs.read[/tmp/secret.txt]"),
                {},
                audit_log=audit_log,
            )
        assert audit_log.entries[0].level == "error"

    def test_single_star_does_not_cross_separator(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("fs.read[/workspace/*.py]"),
                caps("fs.read[/workspace/sub/main.py]"),
                {},
                audit_log=audit_log,
            )

    def test_scoped_grant_does_not_cover_unscoped_required(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("fs.read[/workspace/**]"),
                caps("fs.read"),
                {},
                audit_log=audit_log,
            )

    def test_literal_param_exact_match(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("secret.read[db_password]"),
            caps("secret.read[db_password]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_literal_param_mismatch_denied(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("secret.read[db_password]"),
                caps("secret.read[api_key]"),
                {},
                audit_log=audit_log,
            )


# ===========================================================================
# Intersection check
# ===========================================================================

class TestIntersectionCheck:
    def test_all_required_satisfied_passes(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell", "net.listen", "fs.read[/workspace/**]"),
            caps("exec.shell", "net.listen", "fs.read[/workspace/main.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_one_required_missing_raises(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(
                caps("exec.shell"),
                caps("exec.shell", "exec.sudo"),
                {},
                audit_log=audit_log,
            )
        assert cap("exec.sudo") in exc_info.value.missing

    def test_multiple_missing_all_reported(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(
                frozenset(),
                caps("exec.shell", "exec.sudo", "net.listen"),
                {},
                audit_log=audit_log,
            )
        assert exc_info.value.missing == caps("exec.shell", "exec.sudo", "net.listen")

    def test_empty_required_always_passes(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(frozenset(), frozenset(), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "info"

    def test_empty_agent_caps_denies_all(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)

    def test_superset_agent_caps_satisfies_required(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell", "exec.sudo", "net.listen", "fs.read"),
            caps("exec.shell"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"


# ===========================================================================
# OTel span
# ===========================================================================

class TestOTelSpan:
    def test_span_name_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.name == "capability.authorize"

    def test_span_name_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.name == "capability.authorize"

    def test_span_attributes_agent_id(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell"), caps("exec.shell"), {},
            agent_id="agent-42", audit_log=audit_log,
        )
        assert mock_authorize_span.attributes["agent.id"] == "agent-42"

    def test_span_attributes_session_id(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell"), caps("exec.shell"), {},
            session_id="sess-7", audit_log=audit_log,
        )
        assert mock_authorize_span.attributes["session.id"] == "sess-7"

    def test_span_attributes_capability_required(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log
        )
        assert "exec.shell" in mock_authorize_span.attributes["capability.required"]

    def test_span_emits_invocation_event(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        event_names = [e[0] for e in mock_authorize_span.events]
        assert "capability.authorize" in event_names

    def test_span_event_includes_required_and_granted(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell", "net.listen"),
            caps("exec.shell"),
            {},
            audit_log=audit_log,
        )
        event_attrs = dict(mock_authorize_span.events[0][1])
        assert "exec.shell" in event_attrs["capability.required"]
        assert "exec.shell" in event_attrs["capability.granted"]

    def test_span_event_includes_timestamp(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        event_attrs = dict(mock_authorize_span.events[0][1])
        assert "timestamp" in event_attrs

    def test_span_ended_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.ended

    def test_span_ended_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.ended

    def test_span_set_to_error_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.status is not None
        assert mock_authorize_span.status.status_code == StatusCode.ERROR

    def test_span_not_set_to_error_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.status is None


# ===========================================================================
# Audit log
# ===========================================================================

class TestAuditLog:
    def test_exactly_one_entry_written_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert len(audit_log.entries) == 1

    def test_exactly_one_entry_written_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert len(audit_log.entries) == 1

    def test_level_info_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "info"

    def test_level_error_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "error"

    def test_event_name(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].event == "capability.authorize"

    def test_detail_allowed_true_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        entry = audit_log.entries[0]
        assert entry.detail is not None
        assert entry.detail["allowed"] is True

    def test_detail_allowed_false_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        entry = audit_log.entries[0]
        assert entry.detail is not None
        assert entry.detail["allowed"] is False

    def test_detail_required_lists_caps(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell", "net.listen"),
            caps("exec.shell"),
            {},
            audit_log=audit_log,
        )
        entry = audit_log.entries[0]
        assert entry.detail is not None
        assert "exec.shell" in entry.detail["required"]

    def test_detail_missing_empty_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].detail["missing"] == []  # type: ignore[index]

    def test_detail_missing_non_empty_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.sudo"), {}, audit_log=audit_log)
        entry = audit_log.entries[0]
        assert entry.detail is not None
        assert "exec.sudo" in entry.detail["missing"]

    def test_detail_args_echoed(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        args = {"path": "/workspace/foo.py", "mode": "read"}
        authorize(caps("fs.read"), caps("fs.read[/workspace/foo.py]"), args, audit_log=audit_log)
        entry = audit_log.entries[0]
        assert entry.detail is not None
        assert entry.detail["args"] == args

    def test_agent_id_on_entry(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell"), caps("exec.shell"), {},
            agent_id="my-agent", audit_log=audit_log,
        )
        assert audit_log.entries[0].agent_id == "my-agent"

    def test_session_id_on_entry(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell"), caps("exec.shell"), {},
            session_id="sess-99", audit_log=audit_log,
        )
        assert audit_log.entries[0].session_id == "sess-99"

    def test_timestamp_present(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].timestamp != ""

    def test_no_audit_log_does_not_raise(
        self, mock_authorize_span: MockSpan
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {})  # no audit_log — NoopAuditLog used


# ===========================================================================
# Error surface — CapabilityDenied
# ===========================================================================

class TestCapabilityDeniedError:
    def test_raises_capability_denied_type(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)

    def test_missing_attribute_is_frozenset(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert isinstance(exc_info.value.missing, frozenset)

    def test_missing_contains_unsatisfied_cap(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(frozenset(), caps("exec.sudo"), {}, audit_log=audit_log)
        assert cap("exec.sudo") in exc_info.value.missing

    def test_missing_contains_all_unsatisfied(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(
                caps("exec.shell"),
                caps("exec.shell", "exec.sudo", "net.listen"),
                {},
                audit_log=audit_log,
            )
        assert exc_info.value.missing == caps("exec.sudo", "net.listen")

    def test_message_includes_missing_cap_name(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(frozenset(), caps("exec.sudo"), {}, audit_log=audit_log)
        assert "exec.sudo" in str(exc_info.value)

    def test_no_exception_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)

    def test_glob_mismatch_raises_with_correct_missing(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        required_cap = cap("fs.read[/tmp/secret]")
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(
                caps("fs.read[/workspace/**]"),
                frozenset({required_cap}),
                {},
                audit_log=audit_log,
            )
        assert required_cap in exc_info.value.missing
