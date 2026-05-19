from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    NoopAuditLog,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)

from ._secret_ref import SecretRefResolver

_SECRET_REF_RE = re.compile(r"^secret_ref://vault/([^/]+)/(.+)$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class HookStdinRedactionError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="hook_stdin_redaction_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Internal recursive walker
# ---------------------------------------------------------------------------


def _walk(
    value: Any,
    *,
    allowed_keys: frozenset[str],
    resolver: SecretRefResolver,
) -> Any:
    """Recursively substitute only the allowed vault refs; leave all others unsubstituted."""
    if isinstance(value, str):
        m = _SECRET_REF_RE.match(value)
        if m is None:
            return value
        key = m.group(2)
        if key in allowed_keys:
            return resolver.resolve(value)
        return value  # unsubstituted — ref URI is not the secret value
    if isinstance(value, dict):
        return {k: _walk(v, allowed_keys=allowed_keys, resolver=resolver) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(item, allowed_keys=allowed_keys, resolver=resolver) for item in value]
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def redact_vault_refs(
    payload: Any,
    *,
    allowed_keys: frozenset[str],
    resolver: SecretRefResolver,
    audit_log: AuditLog | None = None,
) -> Any:
    """
    Walk *payload* and ensure vault refs are not resolved unless the hook
    declared ``secret.read[key]``.

    - ``secret_ref://vault/{vault_id}/{key}`` where *key* is in *allowed_keys*
      is resolved to its plaintext value via *resolver*.
    - All other vault refs are left unsubstituted (the URI string is preserved,
      not the plaintext value) — Risk R7 mitigation.

    Emits OTel span ``hook.stdin.redact`` and a structured invocation event on
    every call.  On failure, writes to the audit log and raises
    :class:`HookStdinRedactionError` so the caller can surface the error
    message.
    """
    now = _now()
    tracer = get_tracer()
    _audit = audit_log if audit_log is not None else NoopAuditLog()

    with tracer.start_as_current_span(
        "hook.stdin.redact",
        attributes={"hook.stdin.allowed_key_count": len(allowed_keys)},
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="hook.stdin.redact.invocation",
                code="hook_stdin_redact",
                timestamp=now,
            ),
        )

        try:
            return _walk(payload, allowed_keys=allowed_keys, resolver=resolver)
        except Exception as exc:
            err = HookStdinRedactionError(
                message=f"Failed to redact hook stdin: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err)
            _audit.write(
                AuditLogEntry(
                    level="error",
                    event="hook.stdin.redact.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={"message": err.message},
                )
            )
            raise err from exc
