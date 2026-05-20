from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

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

from ._vault_backend_encrypted_file import EncryptedFileVaultBackend
from ._vault_backend_os_keychain import OsKeychainVaultBackend

_SECRET_REF_RE = re.compile(r"^secret_ref://vault/([^/]+)/(.+)$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SecretRefParseError(MeridianError):
    def __init__(self, *, ref: str, timestamp: str) -> None:
        super().__init__(
            code="secret_ref_parse_failed",
            message=f"Invalid secret_ref URI: {ref!r}",
            timestamp=timestamp,
        )


class SecretRefResolveError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="secret_ref_resolve_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )


class SecretRefNotFoundError(MeridianError):
    def __init__(self, *, vault_id: str, key: str, timestamp: str) -> None:
        super().__init__(
            code="secret_ref_not_found",
            message=f"Secret '{key}' not found in vault '{vault_id}'",
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class SecretRefResolver:
    """
    Resolves secret_ref://vault/{vault_id}/{key} URIs against vault backends.

    Lazy: secrets are fetched at first use, not at config load, so rotation
    in Vault takes effect without a daemon restart.  Each call to resolve()
    goes directly to the backend — no in-memory caching — so a rotated secret
    is visible on the very next resolution.
    """

    def __init__(
        self,
        *,
        storage_root: Path,
        vault_backend: EncryptedFileVaultBackend | None = None,
        os_keychain_backend: OsKeychainVaultBackend | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._storage_root = storage_root
        self._vault_backend = vault_backend
        self._os_keychain_backend = os_keychain_backend
        self._audit = audit_log if audit_log is not None else NoopAuditLog()

    def resolve(
        self,
        ref: str,
        *,
        agent_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        """Resolve a secret_ref URI, fetching fresh from the vault on every call."""
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "vault.secret.resolve",
            attributes={"secret.ref": ref},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.secret.resolve.invocation",
                    code="vault_secret_resolve",
                    timestamp=now,
                ),
            )

            try:
                m = _SECRET_REF_RE.match(ref)
                if m is None:
                    raise SecretRefParseError(ref=ref, timestamp=now)

                vault_id, key = m.group(1), m.group(2)
                span.set_attribute("vault.id", vault_id)
                span.set_attribute("secret.key", key)

                vault_file = self._storage_root / "vaults" / f"{vault_id}.json"
                if not vault_file.exists():
                    raise SecretRefResolveError(
                        message=f"Vault '{vault_id}' not found",
                        timestamp=now,
                    )

                vault_meta = json.loads(vault_file.read_text())
                backend_name = vault_meta.get("backend", "os_keychain")

                if backend_name == "encrypted_file":
                    if self._vault_backend is None:
                        raise SecretRefResolveError(
                            message=(
                                "encrypted_file backend is not configured; "
                                "start the daemon with a passphrase or key file"
                            ),
                            timestamp=now,
                        )
                    record = self._vault_backend.get_secret(vault_id, key)
                else:
                    if self._os_keychain_backend is None:
                        raise SecretRefResolveError(
                            message=(
                                "os_keychain backend is not configured; "
                                "pass an OsKeychainVaultBackend to the resolver"
                            ),
                            timestamp=now,
                        )
                    record = self._os_keychain_backend.get_secret(vault_id, key)

                if record is None:
                    raise SecretRefNotFoundError(vault_id=vault_id, key=key, timestamp=now)

                self._audit.write(
                    AuditLogEntry(
                        level="info",
                        event="audit.secret_access",
                        code="vault_secret_access",
                        timestamp=now,
                        detail={
                            "vault_id": vault_id,
                            "name": key,
                            "requester_agent_id": agent_id,
                            "requester_tool_call_id": tool_call_id,
                        },
                    )
                )
                return str(record["value"])

            except (SecretRefParseError, SecretRefNotFoundError, SecretRefResolveError) as err:
                record_error(span, err)
                self._audit.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.secret.resolve.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "ref": ref,
                            "message": err.message,
                            "requester_agent_id": agent_id,
                            "requester_tool_call_id": tool_call_id,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SecretRefResolveError(
                    message=f"Failed to resolve secret ref {ref!r}: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                self._audit.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.secret.resolve.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "ref": ref,
                            "message": err2.message,
                            "requester_agent_id": agent_id,
                            "requester_tool_call_id": tool_call_id,
                        },
                    )
                )
                raise err2 from exc
