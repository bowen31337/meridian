from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core_errors import (
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class VaultBackendUnlockError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_backend_unlock_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# Sentinel values used to dispatch between passphrase and key-file modes.
_MODE_PASSPHRASE = "passphrase"
_MODE_KEY_FILE = "key_file"


class EncryptedFileVaultBackend:
    """
    Age-encrypted JSON secret storage for vaults with backend="encrypted_file".

    Must be unlocked at daemon start (unlock_with_passphrase or
    unlock_with_key_file) before any operation is attempted.  Intended for
    headless servers where the OS keychain is unavailable.

    Each vault's secrets live in a single age-encrypted file:
        storage_root/vaults/{vault_id}/secrets.age

    The file contains a JSON object mapping secret key → secret record.
    Encryption uses the pyrage library (age spec v1).
    """

    def __init__(self, storage_root: Path) -> None:
        self._storage_root = storage_root
        self._mode: str | None = None
        # passphrase mode
        self._passphrase: str | None = None
        # key-file mode
        self._identity: Any = None
        self._recipient: Any = None

    @property
    def is_unlocked(self) -> bool:
        return self._mode is not None

    # ------------------------------------------------------------------
    # Unlock
    # ------------------------------------------------------------------

    def unlock_with_passphrase(self, passphrase: str) -> None:
        """Unlock using a passphrase. Emits vault.backend.unlock OTel span."""
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "vault.backend.unlock",
            attributes={
                "vault.backend": "encrypted_file",
                "vault.backend.auth": _MODE_PASSPHRASE,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.backend.unlock.invocation",
                    code="vault_backend_unlock",
                    timestamp=now,
                ),
            )
            try:
                # Validate the passphrase is usable by doing a round-trip on a
                # tiny payload — pyrage raises on bad passphrase during decrypt.
                import pyrage  # type: ignore[import-untyped]

                _probe = pyrage.passphrase.encrypt(b"probe", passphrase)
                pyrage.passphrase.decrypt(_probe, passphrase)
                self._passphrase = passphrase
                self._mode = _MODE_PASSPHRASE
            except Exception as exc:
                err = VaultBackendUnlockError(
                    message=f"Failed to configure passphrase backend: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                raise err from exc

    def unlock_with_key_file(self, key_file: Path) -> None:
        """Unlock using an age private key file. Emits vault.backend.unlock OTel span."""
        now = _now()
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "vault.backend.unlock",
            attributes={
                "vault.backend": "encrypted_file",
                "vault.backend.auth": _MODE_KEY_FILE,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.backend.unlock.invocation",
                    code="vault_backend_unlock",
                    timestamp=now,
                ),
            )
            try:
                import pyrage  # type: ignore[import-untyped]

                key_text = key_file.read_text().strip()
                identity = pyrage.x25519.Identity.from_str(key_text)
                self._identity = identity
                self._recipient = identity.to_public()
                self._mode = _MODE_KEY_FILE
            except Exception as exc:
                err = VaultBackendUnlockError(
                    message=f"Failed to load age key file '{key_file}': {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                raise err from exc

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _secrets_file(self, vault_id: str) -> Path:
        return self._storage_root / "vaults" / vault_id / "secrets.age"

    def _encrypt(self, plaintext: bytes) -> bytes:
        import pyrage  # type: ignore[import-untyped]

        if self._mode == _MODE_PASSPHRASE:
            return pyrage.passphrase.encrypt(plaintext, self._passphrase)
        return pyrage.encrypt(plaintext, [self._recipient])

    def _decrypt(self, ciphertext: bytes) -> bytes:
        import pyrage  # type: ignore[import-untyped]

        if self._mode == _MODE_PASSPHRASE:
            return pyrage.passphrase.decrypt(ciphertext, self._passphrase)
        return pyrage.decrypt(ciphertext, [self._identity])

    def _read_secrets(self, vault_id: str) -> dict[str, Any]:
        f = self._secrets_file(vault_id)
        if not f.exists():
            return {}
        return json.loads(self._decrypt(f.read_bytes()))

    def _write_secrets(self, vault_id: str, data: dict[str, Any]) -> None:
        f = self._secrets_file(vault_id)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(self._encrypt(json.dumps(data).encode()))

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def secret_exists(self, vault_id: str, key: str) -> bool:
        return key in self._read_secrets(vault_id)

    def store_secret(
        self, vault_id: str, key: str, value: str, now: str
    ) -> dict[str, Any]:
        data = self._read_secrets(vault_id)
        record: dict[str, Any] = {
            "vault_id": vault_id,
            "key": key,
            "value": value,
            "created_at": now,
            "last_accessed_at": None,
            "requester_counts": {},
        }
        data[key] = record
        self._write_secrets(vault_id, data)
        return record

    def get_secret(self, vault_id: str, name: str) -> dict[str, Any] | None:
        return self._read_secrets(vault_id).get(name)

    def update_secret(self, vault_id: str, name: str, record: dict[str, Any]) -> None:
        data = self._read_secrets(vault_id)
        data[name] = record
        self._write_secrets(vault_id, data)
