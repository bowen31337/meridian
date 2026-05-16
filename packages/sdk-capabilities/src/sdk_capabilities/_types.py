from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    """A single system capability declared as a dotted-name with optional parameter."""

    namespace: str
    name: str
    param: str | None = None

    def __str__(self) -> str:
        base = f"{self.namespace}.{self.name}"
        return f"{base}[{self.param}]" if self.param is not None else base


CapabilitySet = frozenset[Capability]


class CapabilityParseError(ValueError):
    """Raised when a capability string does not conform to the grammar."""

    def __init__(self, text: str, reason: str) -> None:
        super().__init__(f"Invalid capability {text!r}: {reason}")
        self.text = text
        self.reason = reason


class CapabilityDenied(Exception):
    """Raised when a required capability is not satisfied by the granted set."""

    def __init__(self, missing: CapabilitySet) -> None:
        names = ", ".join(sorted(str(c) for c in missing))
        super().__init__(f"Capability denied; missing: {names}")
        self.missing = missing
