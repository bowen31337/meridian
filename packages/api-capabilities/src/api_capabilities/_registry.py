from __future__ import annotations

import re
import threading

from ._types import CapabilityInfo

_IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def is_valid_identifier(s: str) -> bool:
    """Return True if *s* is a valid lowercase dotted-name identifier segment."""
    return bool(_IDENT_RE.match(s))


class CapabilityRegistry:
    """Thread-safe in-memory store for plugin-registered capability namespaces."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._namespaces: dict[str, list[CapabilityInfo]] = {}

    def register(self, namespace: str, capabilities: list[CapabilityInfo]) -> None:
        with self._lock:
            self._namespaces[namespace] = list(capabilities)

    def is_registered(self, namespace: str) -> bool:
        with self._lock:
            return namespace in self._namespaces

    def all_capabilities(self) -> list[CapabilityInfo]:
        with self._lock:
            result: list[CapabilityInfo] = []
            for caps in self._namespaces.values():
                result.extend(caps)
            return result
