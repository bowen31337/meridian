from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

from ._types import ChannelCapabilities, ChannelFailure

_MANIFEST_FILENAME = "channel.json"


@dataclass(frozen=True)
class ChannelManifest:
    """
    Metadata descriptor for a channel driver package.

    Loaded from channel.json in the package directory during
    ``meridian channel install ./pkg``.
    """

    kind: str
    version: str
    display_name: str
    platforms: tuple[str, ...]
    auth_schemes: tuple[str, ...]
    capabilities: ChannelCapabilities
    rate_limit_per_minute: int | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def validate_manifest(manifest: ChannelManifest) -> None:
    """
    Validate a ChannelManifest for required fields.
    Raises ChannelFailure(CHAN_MANIFEST_INVALID) on the first validation error.
    """
    errors: list[str] = []
    if not manifest.kind:
        errors.append("kind must not be empty")
    if not manifest.version:
        errors.append("version must not be empty")
    if not manifest.display_name:
        errors.append("display_name must not be empty")
    if not manifest.platforms:
        errors.append("platforms must not be empty")
    if errors:
        raise ChannelFailure(
            code="CHAN_MANIFEST_INVALID",
            message="; ".join(errors),
            channel_id="",
            channel_kind=manifest.kind,
            session_id="",
            timestamp=_now(),
        )


def load_manifest(directory: str | Path) -> ChannelManifest:
    """
    Load channel.json from *directory* and return a validated ChannelManifest.

    Raises ChannelFailure(CHAN_MANIFEST_INVALID) when:
      - channel.json is not found in the directory.
      - The file contains invalid JSON.
      - Required top-level fields (kind, version, display_name) are absent.
      - Validation via validate_manifest() fails.
    """
    path = Path(directory) / _MANIFEST_FILENAME
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ChannelFailure(
            code="CHAN_MANIFEST_INVALID",
            message=f"{_MANIFEST_FILENAME} not found in {directory}",
            channel_id="",
            channel_kind="",
            session_id="",
            timestamp=_now(),
        ) from None
    except json.JSONDecodeError as exc:
        raise ChannelFailure(
            code="CHAN_MANIFEST_INVALID",
            message=f"{_MANIFEST_FILENAME} is not valid JSON: {exc}",
            channel_id="",
            channel_kind="",
            session_id="",
            timestamp=_now(),
        ) from exc

    try:
        caps_raw = data.get("capabilities", {})
        capabilities = ChannelCapabilities(
            can_send_text=caps_raw.get("can_send_text", True),
            can_send_files=caps_raw.get("can_send_files", False),
            can_receive_reactions=caps_raw.get("can_receive_reactions", False),
            can_thread=caps_raw.get("can_thread", False),
            max_message_length=caps_raw.get("max_message_length"),
            rate_limit_per_minute=caps_raw.get("rate_limit_per_minute"),
        )
        manifest = ChannelManifest(
            kind=data["kind"],
            version=data["version"],
            display_name=data["display_name"],
            platforms=tuple(data.get("platforms", [])),
            auth_schemes=tuple(data.get("auth_schemes", [])),
            capabilities=capabilities,
            rate_limit_per_minute=data.get("rate_limit_per_minute"),
        )
    except KeyError as exc:
        raise ChannelFailure(
            code="CHAN_MANIFEST_INVALID",
            message=f"{_MANIFEST_FILENAME} missing required field: {exc}",
            channel_id="",
            channel_kind="",
            session_id="",
            timestamp=_now(),
        ) from exc

    validate_manifest(manifest)
    return manifest
