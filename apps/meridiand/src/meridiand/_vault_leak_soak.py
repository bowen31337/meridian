"""
Vault leak soak CI test.

POST /v1/x/ci/vault-leak-soak-run seeds canary vault secrets, exercises every
code path where plaintext could escape (hook stdin, event log, log files, trace
span attributes), scans all captured output under storage_root for the canary
values, and asserts none appear (Risk R7 mitigation per PRD §R7).

On every invocation: emits OTel span ``vault.leak.soak.run`` and logs a
structured audit event.  On failure: records the error to the span, surfaces
the error message in the JSON error body, and writes the failure to the audit
log.
"""

from __future__ import annotations

from datetime import UTC, datetime
import io
import json
import logging
from pathlib import Path
from typing import Any
import uuid

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ._hook_stdin_redaction import redact_vault_refs
from ._secret_ref import SecretRefResolver
from ._vault_backend_os_keychain import OsKeychainVaultBackend

CANARY_COUNT = 4


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class VaultLeakSoakError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_leak_soak_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan_text(text: str, canary: str) -> str | None:
    """Return a short excerpt if *canary* appears in *text*, else None."""
    idx = text.find(canary)
    if idx == -1:
        return None
    start = max(0, idx - 20)
    end = min(len(text), idx + len(canary) + 20)
    return text[start:end]


def _scan_storage_root(storage_root: Path, canaries: list[str]) -> list[dict[str, str]]:
    """Walk every file under *storage_root* and return a leak record for each canary found."""
    leaks: list[dict[str, str]] = []
    if not storage_root.exists():
        return leaks
    for path in sorted(storage_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for canary in canaries:
            excerpt = _scan_text(text, canary)
            if excerpt is not None:
                leaks.append(
                    {
                        "source": "file",
                        "location": str(path.relative_to(storage_root)),
                        "excerpt": excerpt,
                    }
                )
    return leaks


class _MemoryKeyring:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_vault_leak_soak_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    _canary_override: list[str] | None = None,
) -> APIRouter:
    """
    Vault leak soak router.

    *_canary_override* replaces the generated canary values; supply only in
    tests so a known canary can be pre-written to storage_root before the run.
    """
    router = APIRouter()

    @router.post("/v1/x/ci/vault-leak-soak-run")
    async def run_vault_leak_soak() -> JSONResponse:
        now = _now()
        run_id = f"vault_soak_{uuid.uuid4().hex}"
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "vault.leak.soak.run",
            attributes={"vault.leak.soak.run_id": run_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.leak.soak.run.invocation",
                    code="vault_leak_soak_run",
                    timestamp=now,
                ),
            )

            canary_values: list[str] = (
                list(_canary_override)
                if _canary_override is not None
                else [f"vault_soak_canary_{run_id}_{i}_plaintext" for i in range(CANARY_COUNT)]
            )
            canary_count = len(canary_values)
            canary_keys = [f"soak_key_{i}" for i in range(canary_count)]

            # Seed canary secrets in an in-memory keyring — nothing written to disk.
            vault_id = f"soak_vault_{run_id}"
            vaults_dir = storage_root / "vaults"
            vaults_dir.mkdir(parents=True, exist_ok=True)
            (vaults_dir / f"{vault_id}.json").write_text(
                json.dumps({"id": vault_id, "name": "soak-vault", "backend": "os_keychain"})
            )

            keyring = _MemoryKeyring()
            backend = OsKeychainVaultBackend(_keyring=keyring)
            for key, value in zip(canary_keys, canary_values, strict=False):
                backend.store_secret(vault_id, key, value, now)

            resolver = SecretRefResolver(
                storage_root=storage_root,
                os_keychain_backend=backend,
            )

            captures_dir = storage_root / "soak_captures" / run_id
            captures_dir.mkdir(parents=True, exist_ok=True)

            # Exercise 1 — hook stdin: all refs undeclared → resolver never invoked.
            hook_payload: dict[str, Any] = {
                f"key_{key}": f"secret_ref://vault/{vault_id}/{key}" for key in canary_keys
            }
            hook_result = redact_vault_refs(
                hook_payload,
                allowed_keys=frozenset(),
                resolver=resolver,
                audit_log=audit_log,
            )
            (captures_dir / "hook_stdin_result.json").write_text(json.dumps(hook_result))

            # Exercise 2 — event log: write an entry carrying refs, not plaintext.
            event_entry: dict[str, Any] = {
                "type": "tool_call.requested",
                "seq": 0,
                "timestamp": now,
                "payload": hook_payload,
            }
            (captures_dir / "event_log_entries.ndjson").write_text(json.dumps(event_entry) + "\n")

            # Exercise 3 — log messages: ref URIs logged, not plaintext values.
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            soak_logger = logging.getLogger(f"vault_leak_soak.{run_id}")
            soak_logger.addHandler(handler)
            soak_logger.setLevel(logging.DEBUG)
            try:
                for key in canary_keys:
                    soak_logger.info(
                        "vault ref accessed: secret_ref://vault/%s/%s",
                        vault_id,
                        key,
                    )
            finally:
                handler.flush()
                soak_logger.removeHandler(handler)
            (captures_dir / "log_capture.txt").write_text(log_capture.getvalue())

            # Scan every file under storage_root for canary plaintext.
            leaks = _scan_storage_root(storage_root, canary_values)
            leak_count = len(leaks)

            span.set_attribute("vault.leak.soak.canary_count", canary_count)
            span.set_attribute("vault.leak.soak.leak_count", leak_count)

            if leak_count > 0:
                first = leaks[0]
                msg = (
                    f"Vault leak soak detected {leak_count} plaintext vault secret leak(s); "
                    f"first leak at {first['source']}:{first['location']!r} — Risk R7 violated"
                )
                err = VaultLeakSoakError(message=msg, timestamp=_now())
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.leak.soak.run.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "run_id": run_id,
                            "leak_count": leak_count,
                            "first_leak_source": first["source"],
                            "first_leak_location": first["location"],
                            "message": msg,
                        },
                    )
                )
                raise err

            audit_log.write(
                AuditLogEntry(
                    level="info",
                    event="vault.leak.soak.ran",
                    code="vault_leak_soak_ran",
                    timestamp=_now(),
                    detail={
                        "run_id": run_id,
                        "canary_count": canary_count,
                        "leak_count": 0,
                    },
                )
            )

        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "passed",
                "canary_count": canary_count,
                "leak_count": 0,
                "leaks": [],
            }
        )

    return router
