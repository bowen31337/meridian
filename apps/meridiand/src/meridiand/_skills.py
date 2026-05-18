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
from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class SkillCreateError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_create_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class SkillInvalidRequestError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_invalid_request", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request model (agentskills.io schema)
# ---------------------------------------------------------------------------


class SkillTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class SkillTest(BaseModel):
    name: str
    input: dict[str, Any]
    expected_output: str | None = None


class SkillCreateRequest(BaseModel):
    name: str
    description: str
    instructions: str
    tools: list[SkillTool]
    tests: list[SkillTest] | None = None
    metadata: dict[str, Any] | None = None


def _validate_request(body: SkillCreateRequest) -> SkillInvalidRequestError | None:
    if not body.name.strip():
        return SkillInvalidRequestError(
            message="'name' must not be empty",
            timestamp=_now(),
        )
    if not body.instructions.strip():
        return SkillInvalidRequestError(
            message="'instructions' must not be empty",
            timestamp=_now(),
        )
    if not body.tools:
        return SkillInvalidRequestError(
            message="'tools' must contain at least one tool",
            timestamp=_now(),
        )
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_skills_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    skills_dir = storage_root / "skills"
    versions_dir = storage_root / "skill_versions"

    @router.post("/v1/skills", status_code=201)
    async def create_skill(body: SkillCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        skill_id = f"skill_{uuid.uuid4().hex}"
        version_id = f"skillver_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "skill.create",
            attributes={
                "skill.id": skill_id,
                "skill.name": body.name,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.create.invocation",
                    code="skill_create",
                    timestamp=now,
                ),
            )

            try:
                validation_err = _validate_request(body)
                if validation_err is not None:
                    raise validation_err

                skills_dir.mkdir(parents=True, exist_ok=True)
                versions_dir.mkdir(parents=True, exist_ok=True)

                version_record: dict[str, Any] = {
                    "id": version_id,
                    "skill_id": skill_id,
                    "version_number": 1,
                    "instructions": body.instructions,
                    "tools": [t.model_dump() for t in body.tools],
                    "tests": [t.model_dump() for t in body.tests] if body.tests else [],
                    "created_at": now,
                }
                (versions_dir / f"{version_id}.json").write_text(json.dumps(version_record))

                skill_record: dict[str, Any] = {
                    "id": skill_id,
                    "name": body.name,
                    "description": body.description,
                    "created_at": now,
                    "metadata": body.metadata,
                    "version": version_record,
                }
                (skills_dir / f"{skill_id}.json").write_text(json.dumps(skill_record))

            except SkillInvalidRequestError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.create.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "skill_id": skill_id,
                            "name": body.name,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillCreateError(
                    message=f"Failed to create skill: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.create.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "skill_id": skill_id,
                            "name": body.name,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=skill_record, status_code=201)

    return router
