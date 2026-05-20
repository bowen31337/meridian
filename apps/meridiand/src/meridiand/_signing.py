from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


class DaemonSigningKey:
    """Ed25519 private key used to sign each audit log line for tamper-evidence.

    Key files:
      - audit_signing.key  (32-byte raw seed, mode 0o600)
      - audit_signing.pub  (32-byte raw public key, mode 0o644)

    The signed payload for each audit entry is the canonical JSON of the record
    (sort_keys=True, without the "sig" field) encoded as UTF-8 bytes.
    """

    _KEY_FILE = "audit_signing.key"
    _PUB_FILE = "audit_signing.pub"

    def __init__(self, storage_root: Path) -> None:
        self._key_path = storage_root / self._KEY_FILE
        self._pub_path = storage_root / self._PUB_FILE
        self._private_key: Ed25519PrivateKey = self._load_or_generate()

    def _load_or_generate(self) -> Ed25519PrivateKey:
        if self._key_path.exists():
            key = Ed25519PrivateKey.from_private_bytes(self._key_path.read_bytes())
        else:
            key = Ed25519PrivateKey.generate()
            seed = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            try:
                fd = os.open(str(self._key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.write(fd, seed)
                finally:
                    os.close(fd)
            except FileExistsError:
                # Another process wrote the key concurrently; load theirs.
                key = Ed25519PrivateKey.from_private_bytes(self._key_path.read_bytes())

        pub = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        fd = os.open(
            str(self._pub_path),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o644,
        )
        try:
            os.write(fd, pub)
        finally:
            os.close(fd)
        return key

    def sign(self, data: bytes) -> str:
        """Return a base64-encoded Ed25519 signature over data."""
        return base64.b64encode(self._private_key.sign(data)).decode()

    @property
    def public_key_path(self) -> Path:
        return self._pub_path
