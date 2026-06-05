"""
Capability intersection conformance suite.

Proves the four cross-cutting behavioural properties that every implementation
of the capability system must satisfy:

  Glob narrowing (via authorize)
    - ``**`` crosses path separators; ``*`` does not.
    - ``?`` matches exactly one non-separator character.
    - Literal params behave as degenerate (exact) glob patterns.
    - Multiple wildcards in one pattern.
    - Unrestricted grant (no param) covers any required param, including paths.

  Parameter matching (via satisfies / _satisfies_glob)
    - Unrestricted grant (no param) covers any required param.
    - Exact param match covers; any mismatch fails.
    - Scoped grant cannot cover an unscoped (no-param) requirement.
    - authorize uses glob matching; satisfies uses exact matching — same cap,
      different semantics depending on call site.

  Missing-cap error path (via authorize)
    - CapabilityDenied is raised with .missing listing every unsatisfied cap.
    - Error message surfaces the missing cap names to the caller.
    - Audit log entry is written at level "error" with detail["allowed"]=False
      and detail["missing"] naming the denied caps.
    - OTel span status is set to ERROR; span is always ended.

  Subset enforcement on spawn (via is_subset)
    - Child caps that are an exact subset of parent caps are accepted.
    - Unrestricted parent caps (no param) cover any child parameterised cap.
    - Child caps that escalate beyond parent are rejected.
    - agent.spawn[id]: unrestricted parent covers any specific spawn request.
    - agent.spawn[id]: parameterised parent covers only the identical spawn id.
    - agent.spawn[id]: child cannot gain a spawn right the parent lacks.
    - Mixed escalation: one cap escalates ⟹ is_subset returns False.
"""

from __future__ import annotations

from opentelemetry.trace import StatusCode
import pytest
from sdk_capabilities import (
    Capability,
    CapabilityDenied,
    CapabilitySet,
    authorize,
    is_subset,
    parse,
    parse_set,
    satisfies,
)

from .conftest import CapturingAuditLog, MockSpan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cap(text: str) -> Capability:
    return parse(text)


def caps(*texts: str) -> CapabilitySet:
    return parse_set(texts)


# ===========================================================================
# Glob narrowing — grant param treated as a glob pattern
# ===========================================================================


class TestGlobNarrowing:
    """authorize() must use glob matching on grant params."""

    # --- double-star crosses path separators --------------------------------

    def test_double_star_covers_direct_child(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("fs.read[/w/**]"), caps("fs.read[/w/foo.py]"), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "info"

    def test_double_star_covers_deeply_nested(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("fs.read[/workspace/**]"),
            caps("fs.read[/workspace/a/b/c/d.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_double_star_covers_root_level_file(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("fs.read[/**]"), caps("fs.read[/etc/passwd]"), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "info"

    # --- single-star does NOT cross path separators -------------------------

    def test_single_star_matches_within_one_segment(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("fs.read[/workspace/*.py]"),
            caps("fs.read[/workspace/main.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_single_star_rejects_nested_path(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("fs.read[/workspace/*.py]"),
                caps("fs.read[/workspace/sub/main.py]"),
                {},
                audit_log=audit_log,
            )
        assert audit_log.entries[0].level == "error"

    def test_single_star_rejects_crossing_path_separator(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        # * matches any non-'/' sequence; grant /a/*/c does not cover /a/b/d/c
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("fs.read[/a/*/c]"),
                caps("fs.read[/a/b/d/c]"),
                {},
                audit_log=audit_log,
            )

    # --- question-mark matches one non-separator character ------------------

    def test_question_mark_matches_single_char(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("secret.read[vault?key]"),
            caps("secret.read[vault_key]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_question_mark_rejects_separator(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("secret.read[vault?key]"),
                caps("secret.read[vault/key]"),
                {},
                audit_log=audit_log,
            )

    def test_question_mark_rejects_empty_or_multi_char(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("secret.read[db?]"),
                caps("secret.read[database]"),
                {},
                audit_log=audit_log,
            )

    # --- literal param behaves as exact glob --------------------------------

    def test_literal_param_exact_match_allowed(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("secret.read[prod_db_password]"),
            caps("secret.read[prod_db_password]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_literal_param_mismatch_denied(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("secret.read[prod_db_password]"),
                caps("secret.read[staging_db_password]"),
                {},
                audit_log=audit_log,
            )

    # --- multiple wildcards in one pattern ----------------------------------

    def test_double_star_and_extension_wildcard(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("fs.read[/workspace/**/*.py]"),
            caps("fs.read[/workspace/src/util/helper.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_double_star_and_extension_rejects_wrong_ext(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(
                caps("fs.read[/workspace/**/*.py]"),
                caps("fs.read[/workspace/src/config.json]"),
                {},
                audit_log=audit_log,
            )

    # --- unrestricted grant covers parameterised required -------------------

    def test_unrestricted_grant_covers_any_path(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        for path in ["/etc/hosts", "/workspace/deep/nested/file.py", "/tmp/scratch"]:
            al: CapturingAuditLog = CapturingAuditLog()
            authorize(caps("fs.read"), caps(f"fs.read[{path}]"), {}, audit_log=al)
            assert al.entries[0].level == "info"

    def test_unrestricted_grant_covers_unparameterised_required(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "info"


# ===========================================================================
# Parameter matching — satisfies vs _satisfies_glob
# ===========================================================================


class TestParameterMatching:
    """satisfies() uses exact param equality; authorize() uses glob matching."""

    def test_satisfies_exact_param_match(self) -> None:
        assert satisfies(cap("fs.read[/home/user]"), cap("fs.read[/home/user]"))

    def test_satisfies_param_mismatch_fails(self) -> None:
        assert not satisfies(cap("fs.read[/home/user]"), cap("fs.read[/home/other]"))

    def test_satisfies_glob_in_param_not_expanded(self) -> None:
        # satisfies() treats the param literally — no glob expansion
        assert not satisfies(cap("fs.read[/home/user]"), cap("fs.read[/home/*]"))

    def test_satisfies_unrestricted_covers_parameterised(self) -> None:
        assert satisfies(cap("fs.read[/home/user]"), cap("fs.read"))

    def test_satisfies_unrestricted_covers_unparameterised(self) -> None:
        assert satisfies(cap("exec.shell"), cap("exec.shell"))

    def test_satisfies_scoped_does_not_cover_unscoped(self) -> None:
        # A narrower grant cannot cover a broader (unscoped) requirement
        assert not satisfies(cap("fs.read"), cap("fs.read[/home/*]"))

    def test_authorize_uses_glob_where_satisfies_does_not(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        # satisfies(fs.read[/workspace/main.py], fs.read[/workspace/**]) → False (exact)
        assert not satisfies(cap("fs.read[/workspace/main.py]"), cap("fs.read[/workspace/**]"))
        # authorize(...) → True (glob)
        authorize(
            caps("fs.read[/workspace/**]"),
            caps("fs.read[/workspace/main.py]"),
            {},
            audit_log=audit_log,
        )
        assert audit_log.entries[0].level == "info"

    def test_namespace_mismatch_always_fails(self) -> None:
        assert not satisfies(cap("fs.read"), cap("net.read"))
        assert not satisfies(cap("fs.read[/etc]"), cap("net.read[/etc]"))

    def test_name_mismatch_always_fails(self) -> None:
        assert not satisfies(cap("fs.read"), cap("fs.write"))
        assert not satisfies(cap("fs.read[/tmp]"), cap("fs.write[/tmp]"))


# ===========================================================================
# Missing-cap error path
# ===========================================================================


class TestMissingCapErrorPath:
    """On denial, error must be raised, audited at error level, and span must be ERROR."""

    # --- CapabilityDenied raised with correct .missing ----------------------

    def test_raises_capability_denied(
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

    def test_single_missing_cap_in_exception(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(frozenset(), caps("exec.sudo"), {}, audit_log=audit_log)
        assert cap("exec.sudo") in exc_info.value.missing

    def test_all_missing_caps_in_exception(
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

    # --- error message surfaces missing cap names ---------------------------

    def test_error_message_includes_missing_cap(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(frozenset(), caps("exec.sudo"), {}, audit_log=audit_log)
        assert "exec.sudo" in str(exc_info.value)

    def test_error_message_includes_all_missing_caps(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            authorize(frozenset(), caps("exec.sudo", "net.listen"), {}, audit_log=audit_log)
        msg = str(exc_info.value)
        assert "exec.sudo" in msg
        assert "net.listen" in msg

    # --- audit log written at error level -----------------------------------

    def test_audit_entry_written_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert len(audit_log.entries) == 1

    def test_audit_level_error_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "error"

    def test_audit_detail_allowed_false(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].detail["allowed"] is False  # type: ignore[index]

    def test_audit_detail_missing_names_present(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.sudo"), {}, audit_log=audit_log)
        assert "exec.sudo" in audit_log.entries[0].detail["missing"]  # type: ignore[index]

    def test_audit_event_name(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].event == "capability.authorize"

    # --- OTel span set to ERROR and ended -----------------------------------

    def test_span_status_error_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.status is not None
        assert mock_authorize_span.status.status_code == StatusCode.ERROR

    def test_span_status_message_includes_missing_cap(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.sudo"), {}, audit_log=audit_log)
        assert "exec.sudo" in mock_authorize_span.status.description

    def test_span_ended_on_denial(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        with pytest.raises(CapabilityDenied):
            authorize(frozenset(), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.ended

    def test_span_ended_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.ended

    def test_span_not_set_to_error_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.status is None

    # --- audit level info on success ----------------------------------------

    def test_audit_level_info_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].level == "info"

    def test_audit_detail_allowed_true_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].detail["allowed"] is True  # type: ignore[index]

    def test_audit_detail_missing_empty_on_success(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert audit_log.entries[0].detail["missing"] == []  # type: ignore[index]

    # --- OTel span carries structured event ---------------------------------

    def test_span_emits_invocation_event(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        event_names = [e[0] for e in mock_authorize_span.events]
        assert "capability.authorize" in event_names

    def test_span_event_includes_required(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        attrs = dict(mock_authorize_span.events[0][1])
        assert "exec.shell" in attrs["capability.required"]

    def test_span_event_includes_granted(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        attrs = dict(mock_authorize_span.events[0][1])
        assert "exec.shell" in attrs["capability.granted"]

    def test_span_event_includes_timestamp(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        attrs = dict(mock_authorize_span.events[0][1])
        assert attrs.get("timestamp", "") != ""

    def test_span_attributes_carry_agent_id(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell"),
            caps("exec.shell"),
            {},
            agent_id="agent-77",
            audit_log=audit_log,
        )
        assert mock_authorize_span.attributes["agent.id"] == "agent-77"

    def test_span_attributes_carry_session_id(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(
            caps("exec.shell"),
            caps("exec.shell"),
            {},
            session_id="sess-42",
            audit_log=audit_log,
        )
        assert mock_authorize_span.attributes["session.id"] == "sess-42"

    def test_span_name_is_capability_authorize(
        self, mock_authorize_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        authorize(caps("exec.shell"), caps("exec.shell"), {}, audit_log=audit_log)
        assert mock_authorize_span.name == "capability.authorize"


# ===========================================================================
# Subset enforcement on spawn — is_subset / agent.spawn
# ===========================================================================


class TestSubsetEnforcementOnSpawn:
    """
    is_subset(child, parent) must be True only when every cap in child is
    exactly satisfied by some element of parent.  The no-upward-escalation rule
    is absolute: a child session must never receive a capability the parent
    does not already hold.
    """

    # --- basic subset checks ------------------------------------------------

    def test_empty_child_is_always_valid(self) -> None:
        assert is_subset(frozenset(), caps("exec.shell", "fs.read"))

    def test_empty_child_against_empty_parent(self) -> None:
        assert is_subset(frozenset(), frozenset())

    def test_equal_sets_are_valid_subset(self) -> None:
        s = caps("exec.shell", "net.listen")
        assert is_subset(s, s)

    def test_proper_child_subset_allowed(self) -> None:
        assert is_subset(caps("exec.shell"), caps("exec.shell", "exec.sudo", "net.listen"))

    def test_child_escalation_rejected(self) -> None:
        assert not is_subset(caps("exec.shell", "exec.sudo"), caps("exec.shell"))

    def test_fully_disjoint_child_rejected(self) -> None:
        assert not is_subset(caps("exec.sudo"), caps("exec.shell"))

    # --- unrestricted parent covers any child parameterised cap -------------

    def test_unrestricted_parent_covers_scoped_child(self) -> None:
        assert is_subset(caps("fs.read[/workspace]"), caps("fs.read"))

    def test_unrestricted_parent_covers_deeply_scoped_child(self) -> None:
        assert is_subset(caps("fs.read[/workspace/src/main.py]"), caps("fs.read"))

    def test_scoped_parent_does_not_cover_unscoped_child(self) -> None:
        # Child requests unrestricted fs.read; parent only has fs.read[/workspace]
        assert not is_subset(caps("fs.read"), caps("fs.read[/workspace]"))

    def test_scoped_parent_covers_exact_same_scoped_child(self) -> None:
        assert is_subset(caps("fs.read[/workspace]"), caps("fs.read[/workspace]"))

    def test_scoped_parent_does_not_cover_different_scoped_child(self) -> None:
        # is_subset uses exact param matching — glob patterns not expanded
        assert not is_subset(caps("fs.read[/workspace/src]"), caps("fs.read[/workspace]"))

    # --- agent.spawn enforcement --------------------------------------------

    def test_unrestricted_spawn_parent_covers_any_spawn_child(self) -> None:
        # Parent has agent.spawn (unrestricted) → child can spawn any specific agent
        assert is_subset(caps("agent.spawn[worker-1]"), caps("agent.spawn"))

    def test_unrestricted_spawn_parent_covers_multiple_spawn_children(self) -> None:
        assert is_subset(
            caps("agent.spawn[worker-1]", "agent.spawn[worker-2]"),
            caps("agent.spawn"),
        )

    def test_exact_spawn_id_parent_covers_same_spawn_id_child(self) -> None:
        assert is_subset(caps("agent.spawn[worker-1]"), caps("agent.spawn[worker-1]"))

    def test_exact_spawn_id_parent_does_not_cover_different_id(self) -> None:
        assert not is_subset(caps("agent.spawn[worker-2]"), caps("agent.spawn[worker-1]"))

    def test_child_cannot_gain_spawn_right_parent_lacks(self) -> None:
        # Parent has no agent.spawn at all → child cannot spawn anything
        assert not is_subset(caps("agent.spawn[worker-1]"), caps("exec.shell"))

    def test_spawn_glob_in_parent_not_expanded_by_is_subset(self) -> None:
        # is_subset uses satisfies(), not _satisfies_glob → glob in parent not expanded
        # Parent has agent.spawn[worker-*] (a glob-looking string but treated literally)
        # Child wants agent.spawn[worker-1]
        # satisfies(child=agent.spawn[worker-1], parent=agent.spawn[worker-*]) → param mismatch
        assert not is_subset(caps("agent.spawn[worker-1]"), caps("agent.spawn[worker-*]"))

    # --- mixed escalation: one cap over the line rejects the whole set ------

    def test_one_escalating_cap_rejects_entire_child_set(self) -> None:
        # Parent grants exec.shell and fs.read; child also wants exec.sudo
        assert not is_subset(
            caps("exec.shell", "fs.read", "exec.sudo"),
            caps("exec.shell", "fs.read"),
        )

    def test_all_caps_satisfied_returns_true(self) -> None:
        assert is_subset(
            caps("exec.shell", "fs.read", "net.listen"),
            caps("exec.shell", "fs.read", "net.listen", "exec.sudo"),
        )

    # --- spawn enforcement with mixed caps ----------------------------------

    def test_child_spawn_within_parent_unrestricted_caps(self) -> None:
        # Realistic: parent has broad grants; child is a narrowed subset
        assert is_subset(
            caps("exec.shell", "fs.read[/workspace]", "agent.spawn[sub-agent]"),
            caps("exec.shell", "fs.read", "agent.spawn"),
        )

    def test_child_spawn_escalation_beyond_parent_caps(self) -> None:
        # Parent never granted exec.sudo; child cannot escalate
        assert not is_subset(
            caps("exec.shell", "exec.sudo", "agent.spawn[sub-agent]"),
            caps("exec.shell", "agent.spawn"),
        )
