from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

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


class SkillInstallError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_install_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


class SkillInstallInvalidSourceError(MeridianError):
    def __init__(self, *, message: str, timestamp: str) -> None:
        super().__init__(
            code="skill_install_invalid_source", message=message, timestamp=timestamp
        )

    def http_status(self) -> int:
        return 422


class SkillInstallSourceLoadError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="skill_install_source_load_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# Request models (agentskills.io schema)
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


class SkillInstallRequest(BaseModel):
    source: str


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
# Content-addressed version ID
# ---------------------------------------------------------------------------


def _content_version_id(
    *,
    skill_id: str,
    instructions: str,
    tools: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    source_type: str,
    source_url: str | None,
    source: str,
    derived_from_session_ids: list[str] | None,
) -> str:
    """Return ``skillver_<sha256>`` where the hash covers the canonical JSON body."""
    body = {
        "derived_from_session_ids": derived_from_session_ids,
        "instructions": instructions,
        "skill_id": skill_id,
        "source": source,
        "source_type": source_type,
        "source_url": source_url,
        "tests": tests,
        "tools": tools,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"skillver_{digest}"


# ---------------------------------------------------------------------------
# Source type detection
# ---------------------------------------------------------------------------


def _detect_source_type(source: str, timestamp: str) -> str:
    if source.startswith("file://"):
        return "file"
    if source.startswith("npm:"):
        return "npm"
    if source.startswith("git+") or source.startswith("git://"):
        return "git"
    if "agentskills.io" in source or source.startswith("agentskills://"):
        return "registry"
    raise SkillInstallInvalidSourceError(
        message=(
            f"Unrecognized source URL '{source}'. "
            "Supported schemes: file://, npm:, git+, git://, agentskills://, "
            "or a URL containing agentskills.io"
        ),
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


class SkillSourceLoader(Protocol):
    def load(self, source_url: str) -> dict[str, Any]: ...


class FileSkillLoader:
    def load(self, source_url: str) -> dict[str, Any]:
        path_str = source_url[7:]  # strip "file://"
        path = Path(path_str)
        manifest_path = path / "skill.json"
        if not manifest_path.exists():
            raise SkillInstallSourceLoadError(
                message=f"skill.json not found at '{path}'",
                timestamp=_now(),
            )
        try:
            return json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise SkillInstallSourceLoadError(
                message=f"Invalid JSON in skill.json at '{path}': {exc}",
                timestamp=_now(),
                cause=exc,
            ) from exc
        except OSError as exc:
            raise SkillInstallSourceLoadError(
                message=f"Failed to read skill.json at '{path}': {exc}",
                timestamp=_now(),
                cause=exc,
            ) from exc


class NpmSkillLoader:
    def load(self, source_url: str) -> dict[str, Any]:
        spec = source_url[4:]  # strip "npm:"
        if spec.startswith("@") and spec.count("@") > 1:
            at_idx = spec.index("@", 1)
            name = spec[:at_idx]
            version = spec[at_idx + 1:]
        elif not spec.startswith("@") and "@" in spec:
            name, version = spec.rsplit("@", 1)
        else:
            name = spec
            version = "latest"

        try:
            registry_url = f"https://registry.npmjs.org/{name}/{version}"
            req = urllib.request.Request(registry_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                pkg_meta = json.loads(resp.read())
        except Exception as exc:
            raise SkillInstallSourceLoadError(
                message=f"Failed to fetch npm metadata for '{spec}': {exc}",
                timestamp=_now(),
                cause=exc,
            ) from exc

        if "meridian-skill" in pkg_meta:
            return pkg_meta["meridian-skill"]

        tarball_url = pkg_meta.get("dist", {}).get("tarball")
        if not tarball_url:
            raise SkillInstallSourceLoadError(
                message=f"No tarball URL in npm metadata for '{spec}'",
                timestamp=_now(),
            )

        try:
            with urllib.request.urlopen(tarball_url, timeout=60) as resp:  # noqa: S310
                data = resp.read()
        except Exception as exc:
            raise SkillInstallSourceLoadError(
                message=f"Failed to download npm tarball for '{spec}': {exc}",
                timestamp=_now(),
                cause=exc,
            ) from exc

        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if member.name.endswith("/skill.json") or member.name == "skill.json":
                        f = tf.extractfile(member)
                        if f is not None:
                            return json.loads(f.read())
        except Exception as exc:
            raise SkillInstallSourceLoadError(
                message=f"Failed to extract skill.json from npm tarball for '{spec}': {exc}",
                timestamp=_now(),
                cause=exc,
            ) from exc

        raise SkillInstallSourceLoadError(
            message=f"skill.json not found in npm package '{spec}'",
            timestamp=_now(),
        )


class GitSkillLoader:
    def load(self, source_url: str) -> dict[str, Any]:
        url = source_url[4:] if source_url.startswith("git+") else source_url
        ref: str | None = None
        if "#" in url:
            url, ref = url.rsplit("#", 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = ["git", "clone", "--depth=1"]
            if ref:
                cmd += ["--branch", ref]
            cmd += [url, tmp_dir]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)  # noqa: S603
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace")
                raise SkillInstallSourceLoadError(
                    message=f"Failed to clone '{url}': {stderr}",
                    timestamp=_now(),
                    cause=exc,
                ) from exc
            except Exception as exc:
                raise SkillInstallSourceLoadError(
                    message=f"Failed to clone '{url}': {exc}",
                    timestamp=_now(),
                    cause=exc,
                ) from exc

            manifest_path = Path(tmp_dir) / "skill.json"
            if not manifest_path.exists():
                raise SkillInstallSourceLoadError(
                    message=f"skill.json not found in repository '{url}'",
                    timestamp=_now(),
                )
            try:
                return json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                raise SkillInstallSourceLoadError(
                    message=f"Failed to read skill.json from '{url}': {exc}",
                    timestamp=_now(),
                    cause=exc,
                ) from exc


class RegistrySkillLoader:
    def load(self, source_url: str) -> dict[str, Any]:
        if source_url.startswith("agentskills://"):
            skill_id = source_url[14:]
            api_url = f"https://agentskills.io/api/v1/skills/{skill_id}"
        else:
            api_url = source_url

        try:
            req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                return json.loads(resp.read())
        except Exception as exc:
            raise SkillInstallSourceLoadError(
                message=f"Failed to fetch skill from registry '{source_url}': {exc}",
                timestamp=_now(),
                cause=exc,
            ) from exc


_DEFAULT_SOURCE_LOADERS: dict[str, Any] = {
    "file": FileSkillLoader(),
    "npm": NpmSkillLoader(),
    "git": GitSkillLoader(),
    "registry": RegistrySkillLoader(),
}


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_skills_router(
    *,
    audit_log: AuditLog,
    storage_root: Path,
    source_loaders: dict[str, Any] | None = None,
) -> APIRouter:
    router = APIRouter()
    skills_dir = storage_root / "skills"
    versions_dir = storage_root / "skill_versions"
    _loaders: dict[str, Any] = {**_DEFAULT_SOURCE_LOADERS, **(source_loaders or {})}

    @router.post("/v1/skills", status_code=201)
    async def create_skill(body: SkillCreateRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        skill_id = f"skill_{uuid.uuid4().hex}"

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

                tools_data = [t.model_dump() for t in body.tools]
                tests_data = [t.model_dump() for t in body.tests] if body.tests else []
                version_id = _content_version_id(
                    skill_id=skill_id,
                    instructions=body.instructions,
                    tools=tools_data,
                    tests=tests_data,
                    source_type="api",
                    source_url=None,
                    source="authored",
                    derived_from_session_ids=None,
                )

                version_record: dict[str, Any] = {
                    "id": version_id,
                    "skill_id": skill_id,
                    "version_number": 1,
                    "instructions": body.instructions,
                    "tools": tools_data,
                    "tests": tests_data,
                    "created_at": now,
                    "source_type": "api",
                    "source_url": None,
                    "source": "authored",
                    "derived_from_session_ids": None,
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

    @router.post("/v1/skills/install", status_code=201)
    async def install_skill(body: SkillInstallRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        skill_id = f"skill_{uuid.uuid4().hex}"

        with tracer.start_as_current_span(
            "skill.install",
            attributes={
                "skill.id": skill_id,
                "skill.install.source": body.source,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="skill.install.invocation",
                    code="skill_install",
                    timestamp=now,
                ),
            )

            try:
                source_type = _detect_source_type(body.source, now)

                loader = _loaders.get(source_type)
                if loader is None:
                    raise SkillInstallError(
                        message=f"No loader configured for source type '{source_type}'",
                        timestamp=now,
                    )

                try:
                    manifest = loader.load(body.source)
                except (SkillInstallSourceLoadError, SkillInstallInvalidSourceError):
                    raise
                except Exception as exc:
                    raise SkillInstallError(
                        message=f"Unexpected error loading skill from '{body.source}': {exc}",
                        timestamp=now,
                        cause=exc,
                    ) from exc

                try:
                    req = SkillCreateRequest(**manifest)
                except Exception as exc:
                    raise SkillInstallSourceLoadError(
                        message=f"Skill manifest from '{body.source}' has invalid structure: {exc}",
                        timestamp=now,
                        cause=exc,
                    ) from exc

                validation_err = _validate_request(req)
                if validation_err is not None:
                    raise validation_err

                skills_dir.mkdir(parents=True, exist_ok=True)
                versions_dir.mkdir(parents=True, exist_ok=True)

                tools_data = [t.model_dump() for t in req.tools]
                tests_data = [t.model_dump() for t in req.tests] if req.tests else []
                version_id = _content_version_id(
                    skill_id=skill_id,
                    instructions=req.instructions,
                    tools=tools_data,
                    tests=tests_data,
                    source_type=source_type,
                    source_url=body.source,
                    source="authored",
                    derived_from_session_ids=None,
                )

                version_record: dict[str, Any] = {
                    "id": version_id,
                    "skill_id": skill_id,
                    "version_number": 1,
                    "instructions": req.instructions,
                    "tools": tools_data,
                    "tests": tests_data,
                    "created_at": now,
                    "source_type": source_type,
                    "source_url": body.source,
                    "source": "authored",
                    "derived_from_session_ids": None,
                }
                (versions_dir / f"{version_id}.json").write_text(json.dumps(version_record))

                skill_record: dict[str, Any] = {
                    "id": skill_id,
                    "name": req.name,
                    "description": req.description,
                    "created_at": now,
                    "metadata": req.metadata,
                    "version": version_record,
                }
                (skills_dir / f"{skill_id}.json").write_text(json.dumps(skill_record))

            except (
                SkillInstallInvalidSourceError,
                SkillInstallSourceLoadError,
                SkillInvalidRequestError,
            ) as err:
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.install.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "skill_id": skill_id,
                            "source": body.source,
                            "message": err.message,
                        },
                    )
                )
                raise

            except Exception as exc:
                err2 = SkillInstallError(
                    message=f"Failed to install skill: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="skill.install.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "skill_id": skill_id,
                            "source": body.source,
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
