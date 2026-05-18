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
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


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


def make_vaults_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
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

    return router
