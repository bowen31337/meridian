from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import re
from typing import Any

from opentelemetry.trace import Status, StatusCode

from ._audit import AuditLog, AuditLogEntry, NoopAuditLog
from ._telemetry import get_tracer
from ._types import Capability, CapabilityDenied, CapabilitySet


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _glob_to_regex(pattern: str) -> str:
    """
    Convert a glob pattern to a regex string.

    - ``**`` matches any sequence of characters including path separators.
    - ``*``  matches any sequence of characters except ``/``.
    - ``?``  matches any single character except ``/``.
    - All other characters are matched literally.
    """
    regex = ""
    i = 0
    while i < len(pattern):
        if pattern[i : i + 2] == "**":
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            regex += "[^/]"
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return regex


def _glob_matches(pattern: str, value: str) -> bool:
    """Return True if *value* matches *pattern* (glob with ** support)."""
    return bool(re.fullmatch(_glob_to_regex(pattern), value))


def _satisfies_glob(required: Capability, granted: Capability) -> bool:
    """
    Return True if `granted` covers `required`, with glob matching on params.

    Matching rules:
    - namespace and name must be identical.
    - If granted.param is None (unrestricted), it covers any required param.
    - If granted.param is set, it is treated as a glob pattern matched against
      required.param; a scoped grant cannot cover an unscoped requirement.
    """
    if (granted.namespace, granted.name) != (required.namespace, required.name):
        return False
    if granted.param is None:
        return True
    if required.param is None:
        return False
    return _glob_matches(granted.param, required.param)


def _missing_glob(required: CapabilitySet, granted: CapabilitySet) -> CapabilitySet:
    """Return the subset of `required` not covered by any element of `granted` (glob matching)."""
    return frozenset(req for req in required if not any(_satisfies_glob(req, g) for g in granted))


def authorize(
    agent_caps: CapabilitySet,
    required: CapabilitySet,
    args: Mapping[str, Any],
    *,
    agent_id: str = "",
    session_id: str = "",
    audit_log: AuditLog | None = None,
) -> None:
    """
    Verify that every cap in `required` is satisfied by at least one grant in `agent_caps`.

    Grant params are treated as glob patterns matched against required cap params,
    so fs.read[/workspace/**] satisfies fs.read[/workspace/foo/bar.py].
    An unrestricted grant (no param) covers any required cap with the same namespace
    and name regardless of the required param.

    Emits an OTel span ``"capability.authorize"`` and writes an audit log entry on
    every call. On denial the span is set to ERROR, an audit entry at level="error"
    is written, and CapabilityDenied is raised with the missing caps — surfacing the
    error message to the caller.
    """
    now = _now()
    tracer = get_tracer()
    _audit = audit_log or NoopAuditLog()

    required_strs = sorted(str(c) for c in required)
    granted_strs = sorted(str(c) for c in agent_caps)

    with tracer.start_as_current_span(
        "capability.authorize",
        attributes={
            "agent.id": agent_id,
            "session.id": session_id,
            "capability.required": ", ".join(required_strs),
        },
    ) as span:
        span.add_event(
            "capability.authorize",
            {
                "agent.id": agent_id,
                "session.id": session_id,
                "capability.required": ", ".join(required_strs),
                "capability.granted": ", ".join(granted_strs),
                "timestamp": now,
            },
        )

        gap = _missing_glob(required, agent_caps)
        allowed = not gap

        _audit.write(
            AuditLogEntry(
                level="info" if allowed else "error",
                event="capability.authorize",
                agent_id=agent_id,
                session_id=session_id,
                timestamp=now,
                detail={
                    "required": required_strs,
                    "missing": sorted(str(c) for c in gap),
                    "args": dict(args),
                    "allowed": allowed,
                },
            )
        )

        if not allowed:
            missing_str = ", ".join(sorted(str(c) for c in gap))
            span.set_status(Status(StatusCode.ERROR, f"Capability denied; missing: {missing_str}"))
            raise CapabilityDenied(gap)
