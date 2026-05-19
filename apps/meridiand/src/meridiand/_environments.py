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
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class EnvironmentCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="environment_create_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class EnvironmentInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="environment_invalid_request", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class EnvironmentNotFoundError(MeridianError):
    def __init__(self, *, environment_id: str, timestamp: str) -> None:
        super().__init__(
            code="environment_not_found",
            message=f"Environment '{environment_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class EnvironmentListError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="environment_list_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class EnvironmentGetError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="environment_get_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class EnvironmentUpdateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="environment_update_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class EnvironmentDeleteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="environment_delete_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class EnvironmentInUseError(MeridianError):
    def __init__(self, *, environment_id: str, timestamp: str) -> None:
        super().__init__(
            code="environment_in_use",
            message=f"Environment '{environment_id}' is still referenced by an agent",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


class EnvironmentActiveSessionError(MeridianError):
    def __init__(self, *, environment_id: str, timestamp: str) -> None:
        super().__init__(
            code="environment_active_session",
            message=f"Environment '{environment_id}' has active sessions",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 409


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class EnvironmentCreateRequest(BaseModel):
    name: str
    backend: str
    image: str | None = None
    template: str | None = None
    workspace_path: str | None = None
    env_passthrough: list[str] | None = None
    network_policy: dict[str, Any] | None = None
    caps_envelope: dict[str, Any] | None = None
    default_timeout_ms: int | None = None


class EnvironmentUpdateRequest(BaseModel):
    name: str | None = None
    image: str | None = None
    template: str | None = None
    workspace_path: str | None = None
    env_passthrough: list[str] | None = None
    network_policy: dict[str, Any] | None = None
    caps_envelope: dict[str, Any] | None = None
    default_timeout_ms: int | None = None


def _validate_create(body: EnvironmentCreateRequest) -> EnvironmentInvalidRequestError | None:
    if not body.name.strip():
        return EnvironmentInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    if not body.backend.strip():
        return EnvironmentInvalidRequestError(
            message="'backend' must not be empty",
            timestamp=_now(),
        )
    return None


def _validate_update(body: EnvironmentUpdateRequest) -> EnvironmentInvalidRequestError | None:
    if body.name is not None and not body.name.strip():
        return EnvironmentInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _referenced_by_agent(environment_id: str, agents_dir: Path) -> bool:
    if not agents_dir.exists():
        return False
    for path in agents_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text())
            if record.get("default_environment_id") == environment_id:
                return True
        except Exception:
            continue
    return False


def _has_active_session(environment_id: str, sessions_dir: Path, agents_dir: Path) -> bool:
    if not sessions_dir.exists():
        return False
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "active":
                continue
            agent_id = manifest.get("agent_id")
            if agent_id is None:
                continue
            agent_path = agents_dir / f"{agent_id}.json"
            if not agent_path.exists():
                continue
            agent = json.loads(agent_path.read_text())
            if agent.get("default_environment_id") == environment_id:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_environments_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    envs_dir = storage_root / "environments"
    agents_dir = storage_root / "agents"
    sessions_dir = storage_root / "sessions"

    @router.post("/v1/environments", status_code=201)
    async def create_environment(body: EnvironmentCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        env_id = f"env_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "environment.create",
            attributes={
                "environment.id": env_id,
                "environment.name": body.name,
                "environment.backend": body.backend,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.create.invocation",
                    code="environment_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_create(body)
                if validation_err is not None:
                    raise validation_err

                envs_dir.mkdir(parents=True, exist_ok=True)

                env_record: dict[str, Any] = {
                    "id": env_id,
                    "name": body.name,
                    "backend": body.backend,
                    "image": body.image,
                    "template": body.template,
                    "workspace_path": body.workspace_path,
                    "env_passthrough": body.env_passthrough,
                    "network_policy": body.network_policy,
                    "caps_envelope": body.caps_envelope,
                    "default_timeout_ms": body.default_timeout_ms,
                    "created_at": now,
                    "updated_at": now,
                }
                (envs_dir / f"{env_id}.json").write_text(json.dumps(env_record))

            except EnvironmentInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "environment_id": env_id,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = EnvironmentCreateError(
                    message=f"Failed to create environment: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "environment_id": env_id,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=env_record, status_code=201)

    @router.get("/v1/environments", status_code=200)
    async def list_environments() -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span("environment.list") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.list.invocation",
                    code="environment_list",
                    timestamp=now,
                ),
            )

            try:
                items: list[dict[str, Any]] = []
                if envs_dir.exists():
                    for path in sorted(envs_dir.glob("*.json")):
                        try:
                            items.append(json.loads(path.read_text()))
                        except Exception:
                            continue

            except Exception as exc:
                err2 = EnvironmentListError(
                    message=f"Failed to list environments: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.list.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"message": err2.message},
                    )
                )
                raise err2

        return JSONResponse(content={"items": items}, status_code=200)

    @router.get("/v1/environments/{environment_id}", status_code=200)
    async def get_environment(environment_id: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "environment.get",
            attributes={"environment.id": environment_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.get.invocation",
                    code="environment_get",
                    timestamp=now,
                ),
            )

            try:
                env_file = envs_dir / f"{environment_id}.json"
                if not env_file.exists():
                    raise EnvironmentNotFoundError(
                        environment_id=environment_id, timestamp=now
                    )
                env_record = json.loads(env_file.read_text())

            except EnvironmentNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.get.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "environment_id": environment_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = EnvironmentGetError(
                    message=f"Failed to get environment: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.get.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "environment_id": environment_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=env_record, status_code=200)

    @router.patch("/v1/environments/{environment_id}", status_code=200)
    async def update_environment(
        environment_id: str, body: EnvironmentUpdateRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "environment.update",
            attributes={"environment.id": environment_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.update.invocation",
                    code="environment_update",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_update(body)
                if validation_err is not None:
                    raise validation_err

                env_file = envs_dir / f"{environment_id}.json"
                if not env_file.exists():
                    raise EnvironmentNotFoundError(
                        environment_id=environment_id, timestamp=now
                    )

                env_record = json.loads(env_file.read_text())

                if _has_active_session(environment_id, sessions_dir, agents_dir):
                    raise EnvironmentActiveSessionError(
                        environment_id=environment_id, timestamp=now
                    )

                patch = body.model_dump(exclude_unset=True)
                env_record.update(patch)
                env_record["updated_at"] = now

                env_file.write_text(json.dumps(env_record))

            except (
                EnvironmentInvalidRequestError,
                EnvironmentNotFoundError,
                EnvironmentActiveSessionError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.update.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "environment_id": environment_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = EnvironmentUpdateError(
                    message=f"Failed to update environment: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.update.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "environment_id": environment_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=env_record, status_code=200)

    @router.delete("/v1/environments/{environment_id}", status_code=204)
    async def delete_environment(environment_id: str) -> Response:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "environment.delete",
            attributes={"environment.id": environment_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="environment.delete.invocation",
                    code="environment_delete",
                    timestamp=now,
                ),
            )

            try:
                env_file = envs_dir / f"{environment_id}.json"
                if not env_file.exists():
                    raise EnvironmentNotFoundError(
                        environment_id=environment_id, timestamp=now
                    )

                if _referenced_by_agent(environment_id, agents_dir):
                    raise EnvironmentInUseError(
                        environment_id=environment_id, timestamp=now
                    )

                env_file.unlink()

            except (EnvironmentNotFoundError, EnvironmentInUseError) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.delete.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "environment_id": environment_id,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = EnvironmentDeleteError(
                    message=f"Failed to delete environment: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="environment.delete.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "environment_id": environment_id,
                            "message": err2.message,
                        },
                    )
                )
                raise err2 from exc

        return Response(status_code=204)

    return router
