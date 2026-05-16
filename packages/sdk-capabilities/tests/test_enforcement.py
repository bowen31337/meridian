"""Enforcement tests: satisfies, check_grant, assert_grant, missing, intersect, is_subset."""
from __future__ import annotations

import pytest

from sdk_capabilities import (
    Capability,
    CapabilityDenied,
    CapabilitySet,
    assert_grant,
    check_grant,
    intersect,
    is_subset,
    missing,
    parse,
    parse_set,
    satisfies,
)


def cap(text: str) -> Capability:
    return parse(text)


def caps(*texts: str) -> CapabilitySet:
    return parse_set(texts)


# ---------------------------------------------------------------------------
# satisfies
# ---------------------------------------------------------------------------

class TestSatisfies:
    def test_exact_match_no_param(self) -> None:
        assert satisfies(cap("exec.shell"), cap("exec.shell"))

    def test_exact_match_with_same_param(self) -> None:
        assert satisfies(cap("fs.read[/home/*]"), cap("fs.read[/home/*]"))

    def test_unrestricted_grant_covers_parameterized_required(self) -> None:
        # granted fs.read (no param) → unrestricted → satisfies fs.read[anything]
        assert satisfies(cap("fs.read[/etc/passwd]"), cap("fs.read"))

    def test_unrestricted_grant_covers_no_param_required(self) -> None:
        assert satisfies(cap("exec.shell"), cap("exec.shell"))

    def test_different_namespace_not_satisfied(self) -> None:
        assert not satisfies(cap("fs.read"), cap("net.read"))

    def test_different_name_not_satisfied(self) -> None:
        assert not satisfies(cap("fs.read"), cap("fs.write"))

    def test_param_mismatch_not_satisfied(self) -> None:
        assert not satisfies(cap("fs.read[/etc/*]"), cap("fs.read[/home/*]"))

    def test_restricted_grant_does_not_cover_unrestricted_required(self) -> None:
        # required fs.read (unrestricted) cannot be covered by granted fs.read[/home/*]
        assert not satisfies(cap("fs.read"), cap("fs.read[/home/*]"))

    def test_unrestricted_grant_covers_any_param(self) -> None:
        for path in ["/tmp", "/var/log", "/etc/passwd"]:
            assert satisfies(cap(f"fs.read[{path}]"), cap("fs.read"))


# ---------------------------------------------------------------------------
# check_grant
# ---------------------------------------------------------------------------

class TestCheckGrant:
    def test_empty_required_always_satisfied(self) -> None:
        assert check_grant(frozenset(), caps("exec.shell"))

    def test_empty_required_empty_granted(self) -> None:
        assert check_grant(frozenset(), frozenset())

    def test_all_required_present_in_granted(self) -> None:
        assert check_grant(
            caps("exec.shell", "fs.read[/workspace]"),
            caps("exec.shell", "fs.read[/workspace]", "net.listen"),
        )

    def test_missing_one_cap_fails(self) -> None:
        assert not check_grant(
            caps("exec.shell", "exec.sudo"),
            caps("exec.shell"),
        )

    def test_empty_granted_fails(self) -> None:
        assert not check_grant(caps("exec.shell"), frozenset())

    def test_unrestricted_grant_covers_parameterized_required(self) -> None:
        assert check_grant(
            caps("fs.read[/home/user]"),
            caps("fs.read"),
        )

    def test_restricted_grant_does_not_cover_unrestricted_required(self) -> None:
        assert not check_grant(
            caps("fs.read"),
            caps("fs.read[/home/user]"),
        )


# ---------------------------------------------------------------------------
# missing
# ---------------------------------------------------------------------------

class TestMissing:
    def test_nothing_missing_when_all_granted(self) -> None:
        assert missing(caps("exec.shell"), caps("exec.shell")) == frozenset()

    def test_returns_unsatisfied_caps(self) -> None:
        gap = missing(
            caps("exec.shell", "exec.sudo"),
            caps("exec.shell"),
        )
        assert gap == caps("exec.sudo")

    def test_empty_required_gives_empty_gap(self) -> None:
        assert missing(frozenset(), caps("exec.shell")) == frozenset()

    def test_empty_granted_gives_full_required(self) -> None:
        required = caps("exec.shell", "fs.read[/home]")
        assert missing(required, frozenset()) == required

    def test_multiple_missing(self) -> None:
        gap = missing(
            caps("exec.shell", "exec.sudo", "exec.pty"),
            caps("exec.shell"),
        )
        assert gap == caps("exec.sudo", "exec.pty")


# ---------------------------------------------------------------------------
# assert_grant
# ---------------------------------------------------------------------------

class TestAssertGrant:
    def test_no_exception_when_all_satisfied(self) -> None:
        assert_grant(caps("exec.shell"), caps("exec.shell", "net.listen"))

    def test_no_exception_when_required_empty(self) -> None:
        assert_grant(frozenset(), frozenset())

    def test_raises_capability_denied(self) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            assert_grant(caps("exec.sudo"), caps("exec.shell"))
        assert cap("exec.sudo") in exc_info.value.missing

    def test_denied_message_includes_cap_name(self) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            assert_grant(caps("exec.sudo"), frozenset())
        assert "exec.sudo" in str(exc_info.value)

    def test_denied_missing_attribute_is_frozenset(self) -> None:
        with pytest.raises(CapabilityDenied) as exc_info:
            assert_grant(caps("exec.sudo", "exec.pty"), frozenset())
        assert isinstance(exc_info.value.missing, frozenset)
        assert exc_info.value.missing == caps("exec.sudo", "exec.pty")

    def test_unrestricted_grant_suppresses_denial(self) -> None:
        assert_grant(caps("fs.read[/home/user]"), caps("fs.read"))


# ---------------------------------------------------------------------------
# intersect
# ---------------------------------------------------------------------------

class TestIntersect:
    def test_empty_sets(self) -> None:
        assert intersect(frozenset(), frozenset()) == frozenset()

    def test_disjoint_sets_yield_empty(self) -> None:
        assert intersect(caps("exec.shell"), caps("net.listen")) == frozenset()

    def test_common_elements_retained(self) -> None:
        result = intersect(
            caps("exec.shell", "exec.sudo", "net.listen"),
            caps("exec.shell", "net.listen"),
        )
        assert result == caps("exec.shell", "net.listen")

    def test_unrestricted_grant_covers_parameterized_in_a(self) -> None:
        result = intersect(
            caps("fs.read[/home/*]"),
            caps("fs.read"),
        )
        assert result == caps("fs.read[/home/*]")

    def test_restricted_grant_does_not_cover_unrestricted_in_a(self) -> None:
        result = intersect(
            caps("fs.read"),
            caps("fs.read[/home/*]"),
        )
        assert result == frozenset()

    def test_full_overlap(self) -> None:
        s = caps("exec.shell", "net.listen", "fs.read[/tmp]")
        assert intersect(s, s) == s

    def test_subset_preserved(self) -> None:
        result = intersect(
            caps("exec.shell"),
            caps("exec.shell", "exec.sudo", "net.listen"),
        )
        assert result == caps("exec.shell")


# ---------------------------------------------------------------------------
# is_subset (no-upward-escalation rule)
# ---------------------------------------------------------------------------

class TestIsSubset:
    def test_empty_child_is_always_subset(self) -> None:
        assert is_subset(frozenset(), caps("exec.shell"))

    def test_empty_child_empty_parent(self) -> None:
        assert is_subset(frozenset(), frozenset())

    def test_child_is_subset_of_parent(self) -> None:
        assert is_subset(
            caps("exec.shell"),
            caps("exec.shell", "exec.sudo"),
        )

    def test_child_escalates_beyond_parent(self) -> None:
        assert not is_subset(
            caps("exec.shell", "exec.sudo"),
            caps("exec.shell"),
        )

    def test_unrestricted_parent_covers_restricted_child(self) -> None:
        assert is_subset(
            caps("fs.read[/workspace]"),
            caps("fs.read"),
        )

    def test_restricted_parent_does_not_cover_unrestricted_child(self) -> None:
        assert not is_subset(
            caps("fs.read"),
            caps("fs.read[/workspace]"),
        )

    def test_equal_sets_are_subset(self) -> None:
        s = caps("exec.shell", "net.listen")
        assert is_subset(s, s)


# ---------------------------------------------------------------------------
# Registry integration: is_known + param_expected
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_known_caps_are_recognized(self) -> None:
        from sdk_capabilities import is_known
        assert is_known("fs", "read")
        assert is_known("exec", "shell")
        assert is_known("acp", "outbound")

    def test_unknown_cap_not_recognized(self) -> None:
        from sdk_capabilities import is_known
        assert not is_known("custom", "action")

    def test_param_expected_true(self) -> None:
        from sdk_capabilities import param_expected
        assert param_expected("fs", "read") is True
        assert param_expected("net", "fetch") is True

    def test_param_expected_false(self) -> None:
        from sdk_capabilities import param_expected
        assert param_expected("exec", "shell") is False
        assert param_expected("net", "listen") is False

    def test_param_expected_unknown(self) -> None:
        from sdk_capabilities import param_expected
        assert param_expected("custom", "action") is None
