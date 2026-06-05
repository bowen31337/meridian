"""Config upgrade registry.

Each entry maps a from-version integer to an idempotent upgrade function that
transforms a raw config dict to the next version.  The functions are applied
in sequence by the ``meridian config migrate`` CLI command.
"""

from __future__ import annotations

from collections.abc import Callable

from . import v1_to_v2

# Maps from_version -> upgrade function (raw dict -> raw dict, idempotent).
UPGRADES: dict[int, Callable[[dict[str, object]], dict[str, object]]] = {
    1: v1_to_v2.upgrade,
}

LATEST_VERSION: int = max(UPGRADES) + 1  # currently 2
