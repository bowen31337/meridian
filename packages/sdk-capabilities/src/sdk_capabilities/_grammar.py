from __future__ import annotations

import re
from collections.abc import Iterable

from ._types import Capability, CapabilityParseError, CapabilitySet

# Grammar:
#   capability ::= namespace "." name ( "[" param "]" )?
#   namespace  ::= [a-z][a-z0-9_]*
#   name       ::= [a-z][a-z0-9_]*
#   param      ::= [^\[\]]+
_IDENT = r"[a-z][a-z0-9_]*"
_PARAM = r"[^\[\]]+"
_PATTERN = re.compile(rf"^(?P<ns>{_IDENT})\.(?P<name>{_IDENT})(?:\[(?P<param>{_PARAM})\])?$")


def parse(text: str) -> Capability:
    """Parse a capability string. Raises CapabilityParseError if the string is invalid."""
    stripped = text.strip()
    if not stripped:
        raise CapabilityParseError(text, "empty string")
    m = _PATTERN.match(stripped)
    if not m:
        raise CapabilityParseError(
            text,
            "expected namespace.name or namespace.name[param] "
            "(lowercase identifiers only; param must be non-empty if brackets present)",
        )
    return Capability(
        namespace=m.group("ns"),
        name=m.group("name"),
        param=m.group("param"),
    )


def parse_set(texts: Iterable[str]) -> CapabilitySet:
    """Parse an iterable of capability strings. Raises CapabilityParseError on first error."""
    return frozenset(parse(t) for t in texts)
