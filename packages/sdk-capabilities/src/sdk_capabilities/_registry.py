from __future__ import annotations

# Registry of known system capabilities.
# True  → a parameter is expected (e.g. fs.read[glob])
# False → no parameter expected  (e.g. exec.shell)
#
# parse() accepts any well-formed dotted-name; this registry is purely
# informational and used by is_known() / param_expected().
_KNOWN: dict[tuple[str, str], bool] = {
    ("fs", "read"): True,
    ("fs", "write"): True,
    ("fs", "delete"): True,
    ("net", "fetch"): True,
    ("net", "listen"): False,
    ("exec", "shell"): False,
    ("exec", "sudo"): False,
    ("exec", "pty"): False,
    ("kb", "read"): True,
    ("kb", "write"): True,
    ("memory", "read"): True,
    ("memory", "write"): True,
    ("agent", "spawn"): True,
    ("agent", "cancel"): False,
    ("secret", "read"): True,
    ("hook", "invoke"): True,
    ("channel", "send"): True,
    ("channel", "receive"): True,
    ("acp", "outbound"): True,
    ("acp", "inbound"): True,
}

KNOWN_CAPABILITIES: frozenset[tuple[str, str]] = frozenset(_KNOWN.keys())


def is_known(namespace: str, name: str) -> bool:
    """Return True if (namespace, name) is a known system capability."""
    return (namespace, name) in _KNOWN


def param_expected(namespace: str, name: str) -> bool | None:
    """
    Return True if a parameter is normally expected, False if none is expected,
    or None if the (namespace, name) pair is not in the known registry.
    """
    return _KNOWN.get((namespace, name))
