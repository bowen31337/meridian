"""Idempotent upgrade from Meridian config version 1 to version 2.

Version 2 is schema-identical to version 1; the only change is the explicit
``version: 2`` field, which establishes a stable baseline for future upgrades.
"""

from __future__ import annotations


def upgrade(raw: dict[str, object]) -> dict[str, object]:
    """Return a copy of *raw* upgraded to version 2."""
    result = dict(raw)
    result["version"] = 2
    return result
