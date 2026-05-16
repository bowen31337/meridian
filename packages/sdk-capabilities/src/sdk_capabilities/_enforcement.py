from __future__ import annotations

from ._types import Capability, CapabilityDenied, CapabilitySet


def satisfies(required: Capability, granted: Capability) -> bool:
    """
    Return True if `granted` covers `required`.

    Matching rules:
    - namespace and name must be identical.
    - If granted.param is None (unrestricted), it covers any required param.
    - If granted.param is set, it only covers the identical required param.

    This means a narrower grant (with param) never covers a broader requirement
    (without param): granting fs.read[/home] does not satisfy requiring fs.read.
    """
    if (granted.namespace, granted.name) != (required.namespace, required.name):
        return False
    if granted.param is None:
        return True
    return granted.param == required.param


def missing(required: CapabilitySet, granted: CapabilitySet) -> CapabilitySet:
    """Return the subset of `required` not satisfied by any element of `granted`."""
    return frozenset(req for req in required if not any(satisfies(req, g) for g in granted))


def check_grant(required: CapabilitySet, granted: CapabilitySet) -> bool:
    """Return True if every capability in `required` is satisfied by `granted`."""
    return not missing(required, granted)


def assert_grant(required: CapabilitySet, granted: CapabilitySet) -> None:
    """Raise CapabilityDenied listing every required capability not covered by granted."""
    gap = missing(required, granted)
    if gap:
        raise CapabilityDenied(gap)


def intersect(a: CapabilitySet, b: CapabilitySet) -> CapabilitySet:
    """
    Return the elements of `a` that are satisfied by some element of `b`.

    Used at dispatch time: given agent-granted caps (`b`) and tool-required
    caps (`a`), intersect() yields the caps from `a` that the agent actually
    holds.  check_grant(a, b) is equivalent to intersect(a, b) == a.
    """
    return frozenset(cap for cap in a if any(satisfies(cap, g) for g in b))


def is_subset(child: CapabilitySet, parent: CapabilitySet) -> bool:
    """
    Return True if `parent` grants every capability in `child`.

    Enforces the no-upward-escalation rule: a child session may only be
    granted capabilities the parent already holds.
    """
    return check_grant(child, parent)
