"""Secret-ref inline substitution for tool arguments.

At dispatch time the harness scans every string value in the tool input for
``secret_ref://vault/{vault_id}/{secret_name}`` tokens and replaces them with
the plaintext secret stored on disk.  The original ref strings (not values)
are written to the OTel event and audit log so plaintext never enters the log.

Raises :class:`SecretRefResolveError` (or a subclass) on any lookup failure
after writing the audit log entry; callers convert this to a
``SandboxResult(is_error=True)`` surfaced to the model as a tool_result.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from opentelemetry.trace import Status, StatusCode

from ._audit import AuditLog
from ._telemetry import get_tracer
from ._types import AuditLogEntry

_SECRET_REF_RE = re.compile(r"^secret_ref://vault/([^/]+)/([^/]+)$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SecretRefResolveError(Exception):
    """Base for all secret-ref resolution failures."""

    def __init__(self, *, code: str, message: str, ref: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.ref = ref


class SecretRefVaultNotFoundError(SecretRefResolveError):
    def __init__(self, *, vault_id: str, ref: str) -> None:
        super().__init__(
            code="secret_ref_vault_not_found",
            message=f"Vault '{vault_id}' not found for ref '{ref}'",
            ref=ref,
        )


class SecretRefNotFoundError(SecretRefResolveError):
    def __init__(self, *, vault_id: str, secret_name: str, ref: str) -> None:
        super().__init__(
            code="secret_ref_not_found",
            message=f"Secret '{secret_name}' not found in vault '{vault_id}' for ref '{ref}'",
            ref=ref,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_refs(value: Any, out: list[str]) -> None:
    """Recursively collect every secret_ref:// string from a tool arg value."""
    if isinstance(value, str):
        if _SECRET_REF_RE.match(value):
            out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _collect_refs(v, out)
    elif isinstance(value, list):
        for item in value:
            _collect_refs(item, out)


def _substitute_value(value: Any, resolved: dict[str, str]) -> Any:
    """Return value with all secret_ref strings replaced from the resolved cache."""
    if isinstance(value, str):
        return resolved.get(value, value)
    if isinstance(value, dict):
        return {k: _substitute_value(v, resolved) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_value(item, resolved) for item in value]
    return value


def _load_secret(ref: str, storage_root: Path) -> str:
    """Fetch the plaintext value for one ref from disk. Raises SecretRefResolveError."""
    m = _SECRET_REF_RE.match(ref)
    assert m  # guaranteed: only called for strings that already matched
    vault_id, secret_name = m.group(1), m.group(2)

    vault_file = storage_root / "vaults" / f"{vault_id}.json"
    if not vault_file.exists():
        raise SecretRefVaultNotFoundError(vault_id=vault_id, ref=ref)

    secret_file = storage_root / "vaults" / vault_id / "secrets" / f"{secret_name}.json"
    if not secret_file.exists():
        raise SecretRefNotFoundError(vault_id=vault_id, secret_name=secret_name, ref=ref)

    try:
        record = json.loads(secret_file.read_text())
    except Exception as exc:
        raise SecretRefResolveError(
            code="secret_ref_read_failed",
            message=f"Failed to read secret for ref '{ref}': {exc}",
            ref=ref,
        ) from exc

    return str(record.get("value", ""))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def substitute_secret_refs(
    input: dict[str, Any],
    *,
    storage_root: Path,
    audit_log: AuditLog,
    tool_name: str,
    session_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(substituted_input, refs)`` with all secret_ref URIs resolved.

    ``refs`` is the list of original ref strings found (not plaintext values) —
    safe to write to any log.  Raises :class:`SecretRefResolveError` on any
    lookup failure; the audit log entry is written before raising so the caller
    only needs to return ``SandboxResult(is_error=True)``.
    """
    now = _now()
    tracer = get_tracer()

    refs: list[str] = []
    _collect_refs(input, refs)
    unique_refs = list(dict.fromkeys(refs))  # stable dedup, preserve order

    with tracer.start_as_current_span(
        "secret_ref.substitute",
        attributes={
            "tool.name": tool_name,
            "session.id": session_id,
            "ref_count": len(unique_refs),
        },
    ) as span:
        span.add_event(
            "secret_ref.substitute",
            {
                "tool.name": tool_name,
                "session.id": session_id,
                "ref_count": len(unique_refs),
                "timestamp": now,
            },
        )

        if not unique_refs:
            return input, []

        try:
            resolved: dict[str, str] = {}
            for ref in unique_refs:
                resolved[ref] = _load_secret(ref, storage_root)
        except SecretRefResolveError as exc:
            span.set_status(Status(StatusCode.ERROR, exc.message))
            span.add_event(
                "secret_ref.substitute.failed",
                {
                    "tool.name": tool_name,
                    "session.id": session_id,
                    "error.code": exc.code,
                    "error.message": exc.message,
                    "ref": exc.ref,
                    "timestamp": now,
                },
            )
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="secret_ref.substitute.failed",
                    tool_name=tool_name,
                    session_id=session_id,
                    timestamp=now,
                    detail={
                        "code": exc.code,
                        "message": exc.message,
                        "ref": exc.ref,
                    },
                )
            )
            raise

        span.add_event(
            "secret_ref.substituted",
            {
                "tool.name": tool_name,
                "session.id": session_id,
                "refs": refs,
                "ref_count": len(unique_refs),
                "timestamp": now,
            },
        )
        return _substitute_value(input, resolved), refs
