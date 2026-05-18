from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from fastapi.responses import Response
from pydantic import BaseModel
from sdk_capabilities import CapabilityParseError
from sdk_capabilities import parse as parse_capability


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class UserProfileCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="user_profile_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class UserProfileInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="user_profile_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class UserProfileNotFoundError(MeridianError):
    def __init__(self, *, user_profile_id: str, timestamp: str) -> None:
        super().__init__(
            code="user_profile_not_found",
            message=f"User profile '{user_profile_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class UserProfileIsPrimaryError(MeridianError):
    def __init__(self, *, user_profile_id: str, timestamp: str) -> None:
        super().__init__(
            code="user_profile_is_primary",
            message=f"Cannot delete primary user profile '{user_profile_id}'",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


class UserProfileHasActiveSessionsError(MeridianError):
    def __init__(self, *, user_profile_id: str, timestamp: str) -> None:
        super().__init__(
            code="user_profile_has_active_sessions",
            message=f"Cannot delete user profile '{user_profile_id}' with active sessions",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


class UserProfileDeleteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="user_profile_delete_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class UserProfileUpdateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="user_profile_update_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class UserProfileCreateRequest(BaseModel):
    username: str
    display_name: str | None = None
    email: str | None = None
    metadata: dict[str, Any] | None = None


class UserProfileUpdateRequest(BaseModel):
    display_name: str | None = None
    capabilities: list[str] | None = None
    memories: list[str] | None = None


def _validate_request(body: UserProfileCreateRequest) -> UserProfileInvalidRequestError | None:
    if not body.username.strip():
        return UserProfileInvalidRequestError(
            message="'username' must not be empty",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_user_profiles_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    profiles_dir = storage_root / "user_profiles"

    @router.post("/v1/user_profiles", status_code=201)
    async def create_user_profile(body: UserProfileCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        user_id = f"user_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "user_profile.create",
            attributes={
                "user_profile.id": user_id,
                "user_profile.username": body.username,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="user_profile.create.invocation",
                    code="user_profile_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                profiles_dir.mkdir(parents=True, exist_ok=True)

                existing = list(profiles_dir.glob("*.json"))
                is_primary = len(existing) == 0

                profile_record: dict[str, Any] = {
                    "id": user_id,
                    "username": body.username,
                    "display_name": body.display_name,
                    "email": body.email,
                    "metadata": body.metadata,
                    "capabilities": [],
                    "memories": [],
                    "is_primary": is_primary,
                    "created_at": now,
                    "updated_at": now,
                }
                (profiles_dir / f"{user_id}.json").write_text(json.dumps(profile_record))

            except UserProfileInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "user_profile_id": user_id,
                            "username": body.username,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = UserProfileCreateError(
                    message=f"Failed to create user profile: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "user_profile_id": user_id,
                            "username": body.username,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=profile_record, status_code=201)

    @router.delete("/v1/user_profiles/{user_profile_id}", status_code=204)
    async def delete_user_profile(user_profile_id: str) -> Response:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "user_profile.delete",
            attributes={"user_profile.id": user_profile_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="user_profile.delete.invocation",
                    code="user_profile_delete",
                    timestamp=now,
                ),
            )

            try:
                profile_file = profiles_dir / f"{user_profile_id}.json"
                if not profile_file.exists():
                    raise UserProfileNotFoundError(
                        user_profile_id=user_profile_id, timestamp=now
                    )

                profile = json.loads(profile_file.read_text())

                if profile.get("is_primary"):
                    raise UserProfileIsPrimaryError(
                        user_profile_id=user_profile_id, timestamp=now
                    )

                sessions_dir = storage_root / "sessions"
                if sessions_dir.exists():
                    for manifest_path in sessions_dir.glob("*/manifest.json"):
                        try:
                            manifest = json.loads(manifest_path.read_text())
                        except Exception:
                            continue
                        if (
                            manifest.get("user_profile_id") == user_profile_id
                            and manifest.get("status") == "active"
                        ):
                            raise UserProfileHasActiveSessionsError(
                                user_profile_id=user_profile_id, timestamp=now
                            )

                profile_file.unlink()

            except (
                UserProfileNotFoundError,
                UserProfileIsPrimaryError,
                UserProfileHasActiveSessionsError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.delete.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "user_profile_id": user_profile_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = UserProfileDeleteError(
                    message=f"Failed to delete user profile: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.delete.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "user_profile_id": user_profile_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return Response(status_code=204)

    @router.patch("/v1/user_profiles/{user_profile_id}", status_code=200)
    async def update_user_profile(user_profile_id: str, body: UserProfileUpdateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "user_profile.update",
            attributes={"user_profile.id": user_profile_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="user_profile.update.invocation",
                    code="user_profile_update",
                    timestamp=now,
                ),
            )

            try:
                profile_file = profiles_dir / f"{user_profile_id}.json"
                if not profile_file.exists():
                    raise UserProfileNotFoundError(
                        user_profile_id=user_profile_id, timestamp=now
                    )

                profile = json.loads(profile_file.read_text())
                fields_set = body.model_fields_set

                if "display_name" in fields_set:
                    profile["display_name"] = body.display_name

                if "capabilities" in fields_set:
                    caps = body.capabilities or []
                    for cap in caps:
                        try:
                            parse_capability(cap)
                        except CapabilityParseError as exc:
                            raise UserProfileInvalidRequestError(
                                message=f"Invalid capability '{cap}': {exc}",
                                timestamp=now,
                            ) from exc
                    profile["capabilities"] = caps

                if "memories" in fields_set:
                    profile["memories"] = body.memories or []

                profile["updated_at"] = now
                profile_file.write_text(json.dumps(profile))

            except (UserProfileNotFoundError, UserProfileInvalidRequestError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.update.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "user_profile_id": user_profile_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = UserProfileUpdateError(
                    message=f"Failed to update user profile: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="user_profile.update.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "user_profile_id": user_profile_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=profile, status_code=200)

    return router
