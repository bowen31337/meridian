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
from fastapi.responses import JSONResponse
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


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class AgentCreateRequest(BaseModel):
    name: str
    kind: str
    config: dict[str, Any] = {}
    capabilities: list[str] = []
    default_environment_id: str | None = None


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

    return router
