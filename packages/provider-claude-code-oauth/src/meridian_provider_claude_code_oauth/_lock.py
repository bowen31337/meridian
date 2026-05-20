"""meridian.lock reader — resolves the pinned Claude Code CLI version.

The lock file is a JSON document at the workspace or storage root that pins the
CLI binary version, preventing silent upgrades from changing the subprocess
protocol between deployments.

Lock file format::

    {
      "version": 1,
      "pins": {
        "claude-code": {
          "version": "1.2.3",
          "channel": "stable"
        }
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_LOCK_TOOL_KEY = "claude-code"
_LOCK_VERSION = 1


@dataclass(frozen=True)
class CliLockEntry:
    """Pinned CLI entry read from meridian.lock."""

    cli_version: str
    channel: str = "stable"


class LockFileNotFoundError(FileNotFoundError):
    """Raised when meridian.lock does not exist at the expected path."""


class LockFileFormatError(ValueError):
    """Raised when meridian.lock is malformed or missing required fields."""


def read_lock(lock_path: Path) -> CliLockEntry:
    """Read meridian.lock and return the pinned Claude Code CLI version.

    Raises
    ------
    LockFileNotFoundError
        If the file does not exist.
    LockFileFormatError
        If the file is not valid JSON, has the wrong schema version, or is
        missing the required ``pins.claude-code`` entry.
    """
    if not lock_path.exists():
        raise LockFileNotFoundError(
            f"meridian.lock not found at {lock_path}; "
            "run 'meridian lock' to generate it"
        )

    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LockFileFormatError(f"Invalid JSON in meridian.lock: {exc}") from exc

    if not isinstance(data, dict):
        raise LockFileFormatError("meridian.lock must be a JSON object")

    if data.get("version") != _LOCK_VERSION:
        raise LockFileFormatError(
            f"Unsupported meridian.lock version {data.get('version')!r}; "
            f"expected {_LOCK_VERSION}"
        )

    pins = data.get("pins", {})
    if not isinstance(pins, dict):
        raise LockFileFormatError("meridian.lock 'pins' field must be a JSON object")

    entry = pins.get(_LOCK_TOOL_KEY)
    if not isinstance(entry, dict):
        raise LockFileFormatError(
            f"meridian.lock is missing required pin '{_LOCK_TOOL_KEY}' under 'pins'"
        )

    cli_version = entry.get("version")
    if not isinstance(cli_version, str) or not cli_version.strip():
        raise LockFileFormatError(
            f"meridian.lock pins.{_LOCK_TOOL_KEY}.version must be a non-empty string"
        )

    channel = entry.get("channel", "stable")

    return CliLockEntry(cli_version=cli_version.strip(), channel=str(channel))
