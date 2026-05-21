"""Import endpoints for OpenClaw and Hermes migrations.

POST /v1/x/imports/openclaw  — import channel records exported from OpenClaw
POST /v1/x/imports/hermes    — import skill records exported from Hermes

Every invocation:
  - Writes a per-import audit log: storage_root/meridian-import-<timestamp>.audit.ndjson
  - Lists each record translated and any lossy field mappings
  - Generates a manual review checklist
  - Is transactional (all-or-nothing): on write failure all partial files are removed
  - On failure surfaces an error to the caller and writes to both the import audit log
    and the main daemon audit log
"""

from __future__ import annotations

import contextlib
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

from ._import_audit import ImportAuditLog

# ---------------------------------------------------------------------------
# Known channel kinds supported by Meridian channel drivers
# ---------------------------------------------------------------------------

_KNOWN_CHANNEL_KINDS = frozenset(
    {"telegram", "slack", "discord", "whatsapp", "signal", "imessage", "cli", "webhook"}
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class ImportRecordInvalidError(MeridianError):
    def __init__(self, *, message: str, timestamp: str, seq: int) -> None:
        super().__init__(code="import_record_invalid", message=message, timestamp=timestamp)
        self.seq = seq

    def http_status(self) -> int:
        return 422


class ImportWriteError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="import_write_failed", message=message, timestamp=timestamp, cause=cause
        )

    def http_status(self) -> int:
        return 500


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class OpenClawRecord(BaseModel):
    """A single channel record exported from OpenClaw."""

    id: str
    kind: str
    name: str | None = None
    config: dict[str, Any] | None = None
    webhook_url: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


class OpenClawImportRequest(BaseModel):
    records: list[OpenClawRecord]


class HermesRecord(BaseModel):
    """A single skill record exported from Hermes."""

    id: str
    name: str
    description: str
    instructions: str
    tools: list[dict[str, Any]]
    tests: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    version_tag: str | None = None
    source_url: str | None = None
    tags: list[str] | None = None
    is_public: bool | None = None


class HermesImportRequest(BaseModel):
    records: list[HermesRecord]


# ---------------------------------------------------------------------------
# OpenClaw → Meridian Channel translation
# ---------------------------------------------------------------------------


def _translate_openclaw(
    rec: OpenClawRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    """Translate one OpenClaw record to a Meridian channel record.

    Returns (channel_record, lossy_fields).  Lossy fields are symbolic names
    describing information that cannot be round-tripped without manual review.
    """
    lossy: list[str] = []
    channel_id = f"ch_{uuid.uuid4().hex}"

    # kind mapping
    mapped_kind = rec.kind
    if rec.kind not in _KNOWN_CHANNEL_KINDS:
        mapped_kind = "generic"
        lossy.append("kind_remapped_to_generic")

    # config assembly
    config: dict[str, Any] = dict(rec.config) if rec.config else {}
    if rec.webhook_url is not None:
        config["webhook_url"] = rec.webhook_url
        lossy.append("webhook_url")  # needs validation post-import

    # token_vault_ref is required by Meridian channel API but absent in imports
    if "token_vault_ref" not in config:
        lossy.append("token_vault_ref_missing")  # must be assigned manually

    # metadata
    meta: dict[str, Any] = {}
    meta["openclaw_id"] = rec.id
    if rec.kind not in _KNOWN_CHANNEL_KINDS:
        meta["openclaw_kind"] = rec.kind
    if rec.name:
        meta["openclaw_name"] = rec.name
    if rec.description:
        meta["openclaw_description"] = rec.description
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"openclaw_meta_{k}"] = v
        lossy.append("metadata")

    channel_record: dict[str, Any] = {
        "id": channel_id,
        "kind": mapped_kind,
        "config": config,
        "default_agent_id": None,
        "default_user_profile_id": None,
        "inbound_policy": "open",
        "egress_policy": "enabled",
        "created_at": now,
        "updated_at": now,
        "metadata": meta,
    }
    return channel_record, lossy


# ---------------------------------------------------------------------------
# Hermes → Meridian Skill translation
# ---------------------------------------------------------------------------


def _skill_version_id(
    skill_id: str,
    instructions: str,
    tools: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    source_url: str | None,
) -> str:
    body = {
        "derived_from_session_ids": None,
        "instructions": instructions,
        "skill_id": skill_id,
        "source": "imported",
        "source_type": "hermes",
        "source_url": source_url,
        "tests": tests,
        "tools": tools,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"skillver_{digest}"


def _translate_hermes(
    rec: HermesRecord, *, now: str
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Translate one Hermes record to Meridian skill + version records.

    Returns (skill_record, version_record, lossy_fields).
    """
    lossy: list[str] = []
    skill_id = f"skill_{uuid.uuid4().hex}"

    if not rec.name.strip():
        raise ValueError("name must not be empty")
    if not rec.instructions.strip():
        raise ValueError("instructions must not be empty")
    if not rec.tools:
        raise ValueError("tools must contain at least one entry")

    tools_data = list(rec.tools)
    tests_data = list(rec.tests) if rec.tests else []

    version_id = _skill_version_id(skill_id, rec.instructions, tools_data, tests_data, rec.source_url)

    meta: dict[str, Any] = {}
    meta["hermes_id"] = rec.id
    if rec.version_tag is not None:
        meta["hermes_version_tag"] = rec.version_tag
        lossy.append("version_tag")  # mapped to version_number=1
    if rec.tags is not None:
        meta["hermes_tags"] = rec.tags
        lossy.append("tags")  # no Meridian equivalent
    if rec.is_public is not None:
        meta["hermes_is_public"] = rec.is_public
        lossy.append("is_public")  # no Meridian equivalent
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"hermes_meta_{k}"] = v
        lossy.append("metadata")

    version_record: dict[str, Any] = {
        "id": version_id,
        "skill_id": skill_id,
        "version_number": 1,
        "instructions": rec.instructions,
        "tools": tools_data,
        "tests": tests_data,
        "created_at": now,
        "source_type": "hermes",
        "source_url": rec.source_url,
        "source": "imported",
        "derived_from_session_ids": None,
    }

    skill_record: dict[str, Any] = {
        "id": skill_id,
        "name": rec.name,
        "description": rec.description,
        "created_at": now,
        "metadata": meta,
        "version": version_record,
    }
    return skill_record, version_record, lossy


# ---------------------------------------------------------------------------
# Checklist generation
# ---------------------------------------------------------------------------


def _openclaw_checklist(
    records: list[OpenClawRecord],
    channel_records: list[dict[str, Any]],
    lossy_per_record: list[list[str]],
) -> list[str]:
    items: list[str] = []
    needs_vault_ref = [
        (i, records[i]) for i, l in enumerate(lossy_per_record) if "token_vault_ref_missing" in l
    ]
    if needs_vault_ref:
        ids = ", ".join(ch["id"] for ch in channel_records)
        items.append(
            f"Assign config.token_vault_ref to {len(needs_vault_ref)} imported channel(s) "
            f"before activating them ({ids})"
        )
    needs_webhook = [i for i, l in enumerate(lossy_per_record) if "webhook_url" in l]
    if needs_webhook:
        items.append(
            f"Validate webhook_url for {len(needs_webhook)} channel(s) — URLs may be stale"
        )
    remapped_kinds = [(i, records[i].kind) for i, l in enumerate(lossy_per_record) if "kind_remapped_to_generic" in l]
    for i, orig_kind in remapped_kinds:
        items.append(
            f"Channel {channel_records[i]['id']}: kind '{orig_kind}' is not a recognized "
            "Meridian driver — assigned 'generic'; assign the correct driver kind manually"
        )
    with_meta = [i for i, l in enumerate(lossy_per_record) if "metadata" in l]
    if with_meta:
        items.append(
            f"Review openclaw_meta_* fields in metadata for {len(with_meta)} channel(s) "
            "— non-standard keys are stored verbatim"
        )
    total = len(records)
    lossy_count = sum(1 for l in lossy_per_record if l)
    items.append(
        f"Imported {total} channel(s) from OpenClaw; "
        f"{lossy_count} had lossy field mappings requiring review"
    )
    return items


def _hermes_checklist(
    records: list[HermesRecord],
    skill_records: list[dict[str, Any]],
    lossy_per_record: list[list[str]],
) -> list[str]:
    items: list[str] = []
    with_version_tag = [(i, records[i].version_tag) for i, l in enumerate(lossy_per_record) if "version_tag" in l]
    if with_version_tag:
        items.append(
            f"Version tag not preserved for {len(with_version_tag)} skill(s) — "
            "all imported as version_number=1; original tags stored in metadata.hermes_version_tag"
        )
    with_tags = [i for i, l in enumerate(lossy_per_record) if "tags" in l]
    if with_tags:
        items.append(
            f"{len(with_tags)} skill(s) had tags in Hermes — stored under metadata.hermes_tags; "
            "no native tags field in Meridian"
        )
    public_skills = [(i, records[i].name) for i, l in enumerate(lossy_per_record) if "is_public" in l and records[i].is_public]
    if public_skills:
        names = ", ".join(name for _, name in public_skills)
        items.append(
            f"Skill(s) marked public in Hermes ({names}) — "
            "visibility is not a Meridian skill concept; review sharing policy"
        )
    with_meta = [i for i, l in enumerate(lossy_per_record) if "metadata" in l]
    if with_meta:
        items.append(
            f"Review hermes_meta_* fields in metadata for {len(with_meta)} skill(s) "
            "— non-standard keys are stored verbatim"
        )
    total = len(records)
    lossy_count = sum(1 for l in lossy_per_record if l)
    items.append(
        f"Imported {total} skill(s) from Hermes; "
        f"{lossy_count} had lossy field mappings requiring review"
    )
    return items


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_imports_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    channels_dir = storage_root / "channels"
    skills_dir = storage_root / "skills"
    versions_dir = storage_root / "skill_versions"

    @router.post("/v1/x/imports/openclaw", status_code=201)
    async def import_openclaw(body: OpenClawImportRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        import_audit = ImportAuditLog(storage_root, source="openclaw", timestamp=now)

        with tracer.start_as_current_span(
            "import.openclaw",
            attributes={
                "import.source": "openclaw",
                "import.record_count": len(body.records),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="import.openclaw.invocation",
                    code="import_openclaw",
                    timestamp=now,
                ),
            )

            import_audit.write_started(record_count=len(body.records), ts=now)

            # ----------------------------------------------------------------
            # Phase 1: Translate all records in memory
            # ----------------------------------------------------------------
            channel_records: list[dict[str, Any]] = []
            lossy_per_record: list[list[str]] = []

            try:
                for seq, rec in enumerate(body.records):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"Record at seq={seq} has empty 'id'",
                            timestamp=now,
                            seq=seq,
                        )
                    try:
                        channel_record, lossy = _translate_openclaw(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate record '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq,
                        ) from exc

                    import_audit.write_record_translated(
                        seq=seq,
                        source_id=rec.id,
                        target_id=channel_record["id"],
                        kind="channel",
                        lossy_fields=lossy,
                    )
                    channel_records.append(channel_record)
                    lossy_per_record.append(lossy)

            except ImportRecordInvalidError as err:
                record_error(span, err)
                import_audit.write_failed(code=err.code, message=err.message)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="import.openclaw.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "message": err.message,
                            "seq": err.seq,
                            "audit_path": import_audit.path.name,
                        },
                    )
                )
                raise

            # ----------------------------------------------------------------
            # Phase 2: Write all channel files transactionally
            # ----------------------------------------------------------------
            written: list[Path] = []
            try:
                channels_dir.mkdir(parents=True, exist_ok=True)
                for channel_record in channel_records:
                    path = channels_dir / f"{channel_record['id']}.json"
                    path.write_text(json.dumps(channel_record))
                    written.append(path)
            except Exception as exc:
                for p in written:
                    with contextlib.suppress(OSError):
                        p.unlink()
                err2 = ImportWriteError(
                    message=f"Failed to write channel records: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                import_audit.write_failed(code=err2.code, message=err2.message)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="import.openclaw.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "message": err2.message,
                            "audit_path": import_audit.path.name,
                        },
                    )
                )
                raise err2

            # ----------------------------------------------------------------
            # Phase 3: Finalize audit log
            # ----------------------------------------------------------------
            checklist = _openclaw_checklist(body.records, channel_records, lossy_per_record)
            import_audit.write_checklist(checklist)
            lossy_count = sum(1 for l in lossy_per_record if l)
            import_audit.write_completed(total=len(channel_records), lossy_count=lossy_count)

        return JSONResponse(
            content={
                "imported": len(channel_records),
                "lossy_count": lossy_count,
                "audit_path": import_audit.path.name,
                "channel_ids": [ch["id"] for ch in channel_records],
            },
            status_code=201,
        )

    @router.post("/v1/x/imports/hermes", status_code=201)
    async def import_hermes(body: HermesImportRequest) -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        import_audit = ImportAuditLog(storage_root, source="hermes", timestamp=now)

        with tracer.start_as_current_span(
            "import.hermes",
            attributes={
                "import.source": "hermes",
                "import.record_count": len(body.records),
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="import.hermes.invocation",
                    code="import_hermes",
                    timestamp=now,
                ),
            )

            import_audit.write_started(record_count=len(body.records), ts=now)

            # ----------------------------------------------------------------
            # Phase 1: Translate all records in memory
            # ----------------------------------------------------------------
            skill_records: list[dict[str, Any]] = []
            version_records: list[dict[str, Any]] = []
            lossy_per_record: list[list[str]] = []

            try:
                for seq, rec in enumerate(body.records):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"Record at seq={seq} has empty 'id'",
                            timestamp=now,
                            seq=seq,
                        )
                    try:
                        skill_record, version_record, lossy = _translate_hermes(rec, now=now)
                    except ImportRecordInvalidError:
                        raise
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate record '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq,
                        ) from exc

                    import_audit.write_record_translated(
                        seq=seq,
                        source_id=rec.id,
                        target_id=skill_record["id"],
                        kind="skill",
                        lossy_fields=lossy,
                    )
                    skill_records.append(skill_record)
                    version_records.append(version_record)
                    lossy_per_record.append(lossy)

            except ImportRecordInvalidError as err:
                record_error(span, err)
                import_audit.write_failed(code=err.code, message=err.message)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="import.hermes.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "message": err.message,
                            "seq": err.seq,
                            "audit_path": import_audit.path.name,
                        },
                    )
                )
                raise

            # ----------------------------------------------------------------
            # Phase 2: Write all skill + version files transactionally
            # ----------------------------------------------------------------
            written: list[Path] = []
            try:
                skills_dir.mkdir(parents=True, exist_ok=True)
                versions_dir.mkdir(parents=True, exist_ok=True)
                for skill_record, version_record in zip(skill_records, version_records):
                    vpath = versions_dir / f"{version_record['id']}.json"
                    vpath.write_text(json.dumps(version_record))
                    written.append(vpath)
                    spath = skills_dir / f"{skill_record['id']}.json"
                    spath.write_text(json.dumps(skill_record))
                    written.append(spath)
            except Exception as exc:
                for p in written:
                    with contextlib.suppress(OSError):
                        p.unlink()
                err2 = ImportWriteError(
                    message=f"Failed to write skill records: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                record_error(span, err2)
                import_audit.write_failed(code=err2.code, message=err2.message)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="import.hermes.failed",
                        code=err2.code,
                        timestamp=err2.timestamp,
                        detail={
                            "message": err2.message,
                            "audit_path": import_audit.path.name,
                        },
                    )
                )
                raise err2

            # ----------------------------------------------------------------
            # Phase 3: Finalize audit log
            # ----------------------------------------------------------------
            checklist = _hermes_checklist(body.records, skill_records, lossy_per_record)
            import_audit.write_checklist(checklist)
            lossy_count = sum(1 for l in lossy_per_record if l)
            import_audit.write_completed(total=len(skill_records), lossy_count=lossy_count)

        return JSONResponse(
            content={
                "imported": len(skill_records),
                "lossy_count": lossy_count,
                "audit_path": import_audit.path.name,
                "skill_ids": [s["id"] for s in skill_records],
            },
            status_code=201,
        )

    return router
