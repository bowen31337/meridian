from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from ._vault_backend_encrypted_file import EncryptedFileVaultBackend


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class VaultCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class VaultInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="vault_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class VaultNotFoundError(MeridianError):
    def __init__(self, *, vault_id: str, timestamp: str) -> None:
        super().__init__(
            code="vault_not_found",
            message=f"Vault '{vault_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class VaultInUseError(MeridianError):
    def __init__(self, *, vault_id: str, timestamp: str) -> None:
        super().__init__(
            code="vault_in_use",
            message=f"Vault '{vault_id}' is still referenced by a provider or channel",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


class VaultDeleteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_delete_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class VaultListError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_list_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class VaultSecretStoreError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_secret_store_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class VaultSecretInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="vault_secret_invalid_request",
            message=message,
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 422


class VaultSecretConflictError(MeridianError):
    def __init__(self, *, vault_id: str, key: str, timestamp: str) -> None:
        super().__init__(
            code="vault_secret_conflict",
            message=f"Secret '{key}' already exists in vault '{vault_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


class VaultSecretNotFoundError(MeridianError):
    def __init__(self, *, vault_id: str, name: str, timestamp: str) -> None:
        super().__init__(
            code="vault_secret_not_found",
            message=f"Secret '{name}' not found in vault '{vault_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class VaultSecretMetaError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="vault_secret_meta_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

VaultBackend = Literal["os_keychain", "encrypted_file"]


class VaultCreateRequest(BaseModel):
    name: str
    backend: VaultBackend


def _validate_request(body: VaultCreateRequest) -> VaultInvalidRequestError | None:
    if not body.name.strip():
        return VaultInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    return None


class VaultSecretStoreRequest(BaseModel):
    key: str
    value: str


def _validate_secret_request(
    body: VaultSecretStoreRequest,
) -> VaultSecretInvalidRequestError | None:
    if not body.key.strip():
        return VaultSecretInvalidRequestError(
            message="'key' must not be empty",
            timestamp=_now(),
        )
    return None


def _vault_is_referenced(vault_id: str, storage_root: Path) -> bool:
    channels_dir = storage_root / "channels"
    if channels_dir.exists():
        for path in channels_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except Exception:
                continue
            config = record.get("config") or {}
            tvr = config.get("token_vault_ref", "")
            if isinstance(tvr, str) and (
                tvr == vault_id or tvr.startswith(f"{vault_id}/")
            ):
                return True

    providers_dir = storage_root / "providers"
    if providers_dir.exists():
        for path in providers_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except Exception:
                continue
            for field in ("vault_id", "vault_ref"):
                ref = record.get(field, "")
                if isinstance(ref, str) and (
                    ref == vault_id or ref.startswith(f"{vault_id}/")
                ):
                    return True

    return False


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_vaults_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    vault_backend: EncryptedFileVaultBackend | None = None,
) -> APIRouter:
    router = APIRouter()
    vaults_dir = storage_root / "vaults"

    @router.post("/v1/vaults", status_code=201)
    async def create_vault(body: VaultCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        vault_id = f"vault_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "vault.create",
            attributes={
                "vault.id": vault_id,
                "vault.name": body.name,
                "vault.backend": body.backend,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.create.invocation",
                    code="vault_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                vaults_dir.mkdir(parents=True, exist_ok=True)

                vault_record: dict[str, Any] = {
                    "id": vault_id,
                    "name": body.name,
                    "backend": body.backend,
                    "created_at": now,
                }
                (vaults_dir / f"{vault_id}.json").write_text(json.dumps(vault_record))

            except VaultInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = VaultCreateError(
                    message=f"Failed to create vault: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=vault_record, status_code=201)

    @router.get("/v1/vaults", status_code=200)
    async def list_vaults() -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span("vault.list") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.list.invocation",
                    code="vault_list",
                    timestamp=now,
                ),
            )

            try:
                items: list[dict[str, Any]] = []
                if vaults_dir.exists():
                    for path in sorted(vaults_dir.glob("*.json")):
                        try:
                            items.append(json.loads(path.read_text()))
                        except Exception:
                            continue

            except Exception as exc:
                err2 = VaultListError(
                    message=f"Failed to list vaults: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.list.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"message": err2.message},
                    )
                )
                raise err2

        return JSONResponse(content={"items": items}, status_code=200)

    @router.delete("/v1/vaults/{vault_id}", status_code=204)
    async def delete_vault(vault_id: str) -> Response:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "vault.delete",
            attributes={"vault.id": vault_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.delete.invocation",
                    code="vault_delete",
                    timestamp=now,
                ),
            )

            try:
                vault_file = vaults_dir / f"{vault_id}.json"
                if not vault_file.exists():
                    raise VaultNotFoundError(vault_id=vault_id, timestamp=now)

                if _vault_is_referenced(vault_id, storage_root):
                    raise VaultInUseError(vault_id=vault_id, timestamp=now)

                vault_file.unlink()

            except (VaultNotFoundError, VaultInUseError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.delete.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = VaultDeleteError(
                    message=f"Failed to delete vault: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.delete.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return Response(status_code=204)

    @router.post("/v1/vaults/{vault_id}/secrets", status_code=201)
    async def store_secret(
        vault_id: str, body: VaultSecretStoreRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "vault.secret.store",
            attributes={"vault.id": vault_id, "secret.key": body.key},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.secret.store.invocation",
                    code="vault_secret_store",
                    timestamp=now,
                ),
            )

            secret_record: dict[str, Any] = {}
            try:
                vault_file = vaults_dir / f"{vault_id}.json"
                if not vault_file.exists():
                    raise VaultNotFoundError(vault_id=vault_id, timestamp=now)

                validation_err = _validate_secret_request(body)
                if validation_err is not None:
                    raise validation_err

                vault_meta = json.loads(vault_file.read_text())
                if vault_meta.get("backend") == "encrypted_file":
                    if vault_backend is None:
                        raise VaultSecretStoreError(
                            message="encrypted_file backend is not configured; "
                            "start the daemon with a passphrase or key file",
                            timestamp=now,
                        )
                    if vault_backend.secret_exists(vault_id, body.key):
                        raise VaultSecretConflictError(
                            vault_id=vault_id, key=body.key, timestamp=now
                        )
                    secret_record = vault_backend.store_secret(
                        vault_id, body.key, body.value, now
                    )
                else:
                    secrets_dir = vaults_dir / vault_id / "secrets"
                    secret_file = secrets_dir / f"{body.key}.json"
                    if secret_file.exists():
                        raise VaultSecretConflictError(
                            vault_id=vault_id, key=body.key, timestamp=now
                        )
                    secrets_dir.mkdir(parents=True, exist_ok=True)
                    secret_record = {
                        "vault_id": vault_id,
                        "key": body.key,
                        "value": body.value,
                        "created_at": now,
                        "last_accessed_at": None,
                        "requester_counts": {},
                    }
                    secret_file.write_text(json.dumps(secret_record))

            except (
                VaultNotFoundError,
                VaultSecretInvalidRequestError,
                VaultSecretConflictError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.secret.store.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "key": body.key,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = VaultSecretStoreError(
                    message=f"Failed to store secret: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.secret.store.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "key": body.key,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(
            content={
                "vault_id": secret_record["vault_id"],
                "key": secret_record["key"],
                "created_at": secret_record["created_at"],
            },
            status_code=201,
        )

    @router.get("/v1/vaults/{vault_id}/secrets/{name}/meta", status_code=200)
    async def get_secret_meta(
        vault_id: str, name: str, request: Request
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        requester = (
            request.client.host if request.client else None
        ) or "unknown"

        with tracer.start_as_current_span(
            "vault.secret.meta",
            attributes={"vault.id": vault_id, "secret.key": name},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="vault.secret.meta.invocation",
                    code="vault_secret_meta",
                    timestamp=now,
                ),
            )

            meta: dict[str, Any] = {}
            try:
                vault_file = vaults_dir / f"{vault_id}.json"
                if not vault_file.exists():
                    raise VaultNotFoundError(vault_id=vault_id, timestamp=now)

                vault_meta = json.loads(vault_file.read_text())
                if vault_meta.get("backend") == "encrypted_file":
                    if vault_backend is None:
                        raise VaultSecretMetaError(
                            message="encrypted_file backend is not configured; "
                            "start the daemon with a passphrase or key file",
                            timestamp=now,
                        )
                    enc_record = vault_backend.get_secret(vault_id, name)
                    if enc_record is None:
                        raise VaultSecretNotFoundError(
                            vault_id=vault_id, name=name, timestamp=now
                        )
                    record: dict[str, Any] = dict(enc_record)
                    record["last_accessed_at"] = now
                    counts: dict[str, int] = dict(record.get("requester_counts") or {})
                    counts[requester] = counts.get(requester, 0) + 1
                    record["requester_counts"] = counts
                    vault_backend.update_secret(vault_id, name, record)
                else:
                    secret_file = vaults_dir / vault_id / "secrets" / f"{name}.json"
                    if not secret_file.exists():
                        raise VaultSecretNotFoundError(
                            vault_id=vault_id, name=name, timestamp=now
                        )
                    record = json.loads(secret_file.read_text())
                    record["last_accessed_at"] = now
                    counts = record.get("requester_counts") or {}
                    counts[requester] = counts.get(requester, 0) + 1
                    record["requester_counts"] = counts
                    secret_file.write_text(json.dumps(record))

                meta = {
                    "vault_id": vault_id,
                    "key": record["key"],
                    "created_at": record["created_at"],
                    "last_accessed_at": record["last_accessed_at"],
                    "requester_counts": record["requester_counts"],
                }

            except (VaultNotFoundError, VaultSecretNotFoundError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.secret.meta.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "name": name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = VaultSecretMetaError(
                    message=f"Failed to get secret metadata: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="vault.secret.meta.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "vault_id": vault_id,
                            "name": name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=meta, status_code=200)

    return router
