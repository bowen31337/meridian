from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


_SERVICE = "meridian"


class OsKeychainVaultBackend:
    """
    OS keychain vault backend (macOS Keychain / Windows Credential Manager / libsecret).

    Default backend for workstations. No unlock step is required — the OS
    manages access via the user's login session.

    Secrets are stored under service="meridian", account="<vault_id>/<key>".
    The keychain item value is a JSON-encoded record containing the secret
    value and metadata (created_at, last_accessed_at, requester_counts).

    Requires the ``keyring`` package. Pass a *_keyring* adapter in tests
    to avoid touching the real system keychain.
    """

    def __init__(self, *, _keyring: Any = None) -> None:
        if _keyring is None:
            import keyring as kr  # type: ignore[import-untyped]

            self._kr: Any = kr
        else:
            self._kr = _keyring

    def _account(self, vault_id: str, key: str) -> str:
        return f"{vault_id}/{key}"

    def secret_exists(self, vault_id: str, key: str) -> bool:
        return self._kr.get_password(_SERVICE, self._account(vault_id, key)) is not None

    def store_secret(self, vault_id: str, key: str, value: str, now: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "vault_id": vault_id,
            "key": key,
            "value": value,
            "created_at": now,
            "last_accessed_at": None,
            "requester_counts": {},
        }
        self._kr.set_password(_SERVICE, self._account(vault_id, key), json.dumps(record))
        return record

    def get_secret(self, vault_id: str, name: str) -> dict[str, Any] | None:
        raw = self._kr.get_password(_SERVICE, self._account(vault_id, name))
        if raw is None:
            return None
        return json.loads(raw)

    def update_secret(self, vault_id: str, name: str, record: dict[str, Any]) -> None:
        self._kr.set_password(_SERVICE, self._account(vault_id, name), json.dumps(record))

    def list_secrets(self, vault_id: str, keys: list[str]) -> list[dict[str, Any]]:
        items = []
        for key in keys:
            rec = self.get_secret(vault_id, key)
            if rec is not None:
                items.append({k: v for k, v in rec.items() if k != "value"})
        return items

    def delete_secret(self, vault_id: str, name: str) -> bool:
        if not self.secret_exists(vault_id, name):
            return False
        self._kr.delete_password(_SERVICE, self._account(vault_id, name))
        return True
