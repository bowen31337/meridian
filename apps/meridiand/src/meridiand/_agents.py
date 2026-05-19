from __future__ import annotations

import hashlib
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


class AgentCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="agent_create_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class AgentInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="agent_invalid_request", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 422


class AgentNotFoundError(MeridianError):
    def __init__(self, *, agent_id: str, timestamp: str) -> None:
        super().__init__(
            code="agent_not_found",
            message=f"Agent '{agent_id}' not found",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


class AgentDeleteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="agent_delete_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class AgentVersionCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="agent_version_create_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AgentCreateRequest(BaseModel):
    name: str
    kind: str
    config: dict[str, Any] = {}
    capabilities: list[str] = []
    default_environment_id: str | None = None


class AgentVersionCreateRequest(BaseModel):
    name: str
    kind: str
    config: dict[str, Any] = {}
    capabilities: list[str] = []


def _validate_request(body: AgentCreateRequest) -> AgentInvalidRequestError | None:
    if not body.name.strip():
        return AgentInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    if not body.kind.strip():
        return AgentInvalidRequestError(
            message="'kind' must not be empty",
            timestamp=_now(),
        )
    return None


def _validate_version_request(body: AgentVersionCreateRequest) -> AgentInvalidRequestError | None:
    if not body.name.strip():
        return AgentInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    if not body.kind.strip():
        return AgentInvalidRequestError(
            message="'kind' must not be empty",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Content-addressed version ID
# ---------------------------------------------------------------------------


def _content_version_id(
    *,
    agent_id: str,
    name: str,
    kind: str,
    config: dict[str, Any],
    capabilities: list[str],
) -> str:
    body = {
        "agent_id": agent_id,
        "capabilities": capabilities,
        "config": config,
        "kind": kind,
        "name": name,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"agentver_{digest}"


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_agents_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    agents_dir = storage_root / "agents"
    versions_dir = storage_root / "agent_versions"

    @router.post("/v1/agents", status_code=201)
    async def create_agent(body: AgentCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        agent_id = f"agent_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "agent.create",
            attributes={
                "agent.id": agent_id,
                "agent.name": body.name,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="agent.create.invocation",
                    code="agent_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                agents_dir.mkdir(parents=True, exist_ok=True)
                versions_dir.mkdir(parents=True, exist_ok=True)

                version_id = _content_version_id(
                    agent_id=agent_id,
                    name=body.name,
                    kind=body.kind,
                    config=body.config,
                    capabilities=body.capabilities,
                )

                version_record: dict[str, Any] = {
                    "id": version_id,
                    "agent_id": agent_id,
                    "version_number": 1,
                    "name": body.name,
                    "kind": body.kind,
                    "config": body.config,
                    "capabilities": body.capabilities,
                    "created_at": now,
                }
                (versions_dir / f"{version_id}.json").write_text(json.dumps(version_record))

                agent_record: dict[str, Any] = {
                    "id": agent_id,
                    "name": body.name,
                    "kind": body.kind,
                    "default_environment_id": body.default_environment_id,
                    "created_at": now,
                    "version": version_record,
                }
                (agents_dir / f"{agent_id}.json").write_text(json.dumps(agent_record))

            except AgentInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="agent.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "agent_id": agent_id,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = AgentCreateError(
                    message=f"Failed to create agent: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="agent.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "agent_id": agent_id,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=agent_record, status_code=201)

    @router.delete("/v1/agents/{agent_id}", status_code=204)
    async def delete_agent(agent_id: str) -> Response:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "agent.delete",
            attributes={"agent.id": agent_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="agent.delete.invocation",
                    code="agent_delete",
                    timestamp=now,
                ),
            )

            try:
                agent_file = agents_dir / f"{agent_id}.json"
                if not agent_file.exists():
                    raise AgentNotFoundError(agent_id=agent_id, timestamp=now)

                agent_record = json.loads(agent_file.read_text())
                agent_record["deleted_at"] = now
                agent_file.write_text(json.dumps(agent_record))

            except AgentNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="agent.delete.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"agent_id": agent_id, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = AgentDeleteError(
                    message=f"Failed to delete agent: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="agent.delete.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"agent_id": agent_id, "message": err2.message},
                    )
                )
                raise err2

        return Response(status_code=204)

    @router.post("/v1/agents/{agent_id}/versions")
    async def create_agent_version(
        agent_id: str, body: AgentVersionCreateRequest
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        version_record: dict[str, Any] = {}
        status_code = 201

        with tracer.start_as_current_span(
            "agent.version.create",
            attributes={"agent.id": agent_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="agent.version.create.invocation",
                    code="agent_version_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_version_request(body)
                if validation_err is not None:
                    raise validation_err

                agent_file = agents_dir / f"{agent_id}.json"
                if not agent_file.exists():
                    raise AgentNotFoundError(agent_id=agent_id, timestamp=now)

                version_id = _content_version_id(
                    agent_id=agent_id,
                    name=body.name,
                    kind=body.kind,
                    config=body.config,
                    capabilities=body.capabilities,
                )

                version_file = versions_dir / f"{version_id}.json"
                if version_file.exists():
                    version_record = json.loads(version_file.read_text())
                    status_code = 200
                else:
                    versions_dir.mkdir(parents=True, exist_ok=True)
                    existing_numbers = []
                    for vf in versions_dir.glob("*.json"):
                        try:
                            vr = json.loads(vf.read_text())
                            if vr.get("agent_id") == agent_id:
                                existing_numbers.append(vr.get("version_number", 0))
                        except Exception:
                            pass
                    next_version_number = max(existing_numbers, default=0) + 1

                    version_record = {
                        "id": version_id,
                        "agent_id": agent_id,
                        "version_number": next_version_number,
                        "name": body.name,
                        "kind": body.kind,
                        "config": body.config,
                        "capabilities": body.capabilities,
                        "created_at": now,
                    }
                    version_file.write_text(json.dumps(version_record))
                    status_code = 201

            except MeridianError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="agent.version.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"agent_id": agent_id, "message": err.message},
                    )
                )
                raise

            except Exception as exc:
                err2 = AgentVersionCreateError(
                    message=f"Failed to create agent version: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="agent.version.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={"agent_id": agent_id, "message": err2.message},
                    )
                )
                raise err2

        return JSONResponse(content=version_record, status_code=status_code)

    return router
