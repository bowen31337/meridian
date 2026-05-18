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
from fastapi import APIRouter, Query
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


class SkillVersionNotFoundError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(code="skill_version_not_found", message=message, timestamp=timestamp)

    def http_status(self) -> int:
        return 404


class SkillListError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_list_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class SkillVersionsListError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_versions_list_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


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

    @router.get("/v1/skills/{skill_id}/versions/{ver}")
    async def get_skill_version(skill_id: str, ver: str) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill.version.get",
            attributes={
                "skill.id": skill_id,
                "skill.version.id": ver,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.version.get.invocation",
                    code="skill_version_get",
                    timestamp=now,
                ),
            )

            try:
                version_path = versions_dir / f"{ver}.json"
                if not version_path.exists():
                    raise SkillVersionNotFoundError(
                        message=f"Skill version '{ver}' not found",
                        timestamp=now,
                    )

                version_record = json.loads(version_path.read_text())

                if version_record.get("skill_id") != skill_id:
                    raise SkillVersionNotFoundError(
                        message=f"Skill version '{ver}' not found",
                        timestamp=now,
                    )

            except SkillVersionNotFoundError as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.version.get.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "skill_id": skill_id,
                            "version_id": ver,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillCreateError(
                    message=f"Failed to retrieve skill version: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.version.get.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "skill_id": skill_id,
                            "version_id": ver,
                            "message": err2.message,
                        },
                    )
                )
                raise err2

        return JSONResponse(content=version_record, status_code=200)

    @router.get("/v1/skills")
    async def list_skills(
        limit: int = Query(default=20),
        offset: int = Query(default=0),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span("skill.list") as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.list.invocation",
                    code="skill_list",
                    timestamp=now,
                ),
            )

            try:
                all_skills: list[dict[str, Any]] = []
                if skills_dir.exists():
                    for path in skills_dir.glob("*.json"):
                        all_skills.append(json.loads(path.read_text()))

                all_skills.sort(key=lambda r: r.get("created_at", ""), reverse=True)
                total = len(all_skills)
                page = all_skills[offset : offset + limit]

            except Exception as exc:
                err = SkillListError(
                    message=f"Failed to list skills: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.list.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message},
                    )
                )
                raise err

        return JSONResponse(
            content={"items": page, "total": total, "limit": limit, "offset": offset},
            status_code=200,
        )

    @router.get("/v1/skills/{skill_id}/versions")
    async def list_skill_versions(
        skill_id: str,
        limit: int = Query(default=20),
        offset: int = Query(default=0),
    ) -> JSONResponse:
        now = _now()
        tracer = get_tracer()

        with tracer.start_as_current_span(
            "skill.versions.list",
            attributes={"skill.id": skill_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.versions.list.invocation",
                    code="skill_versions_list",
                    timestamp=now,
                ),
            )

            try:
                all_versions: list[dict[str, Any]] = []
                if versions_dir.exists():
                    for path in versions_dir.glob("*.json"):
                        record = json.loads(path.read_text())
                        if record.get("skill_id") == skill_id:
                            all_versions.append(record)

                all_versions.sort(key=lambda r: r.get("created_at", ""), reverse=True)
                total = len(all_versions)
                page = all_versions[offset : offset + limit]

            except Exception as exc:
                err = SkillVersionsListError(
                    message=f"Failed to list skill versions: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.versions.list.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"skill_id": skill_id, "message": err.message},
                    )
                )
                raise err

        return JSONResponse(
            content={"items": page, "total": total, "limit": limit, "offset": offset},
            status_code=200,
        )

    return router
