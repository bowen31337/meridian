"""Import endpoints for OpenClaw and Hermes migrations.

POST /v1/x/imports/openclaw         — import channel records exported from OpenClaw
POST /v1/x/imports/hermes           — import skill records exported from Hermes
POST /v1/x/imports/hermes/install   — import a full Hermes installation: skills,
                                      environments, providers, sessions, user_profiles,
                                      cron jobs, and ACP registry entries

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
from pydantic import BaseModel, Field

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


class OpenClawSessionRecord(BaseModel):
    """A single session exported from OpenClaw."""

    id: str
    title: str | None = None
    agent_id: str | None = None
    created_at: str
    events: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


class OpenClawMemoryRecord(BaseModel):
    """A single memory entry exported from an OpenClaw MEMORY.md."""

    key: str
    content: str
    metadata: dict[str, Any] | None = None


_CONSERVATIVE_TOOL_CAPS: dict[str, Any] = {
    "allow_exec": False,
    "allow_network": False,
    "allow_file_write": False,
    "allow_file_read": True,
    "sandboxed": True,
}

_KNOWN_HANDLER_KINDS = frozenset(
    {"http", "subprocess", "mcp", "container", "in_process"}
)


class OpenClawToolRecord(BaseModel):
    """A single tool definition exported from OpenClaw."""

    id: str
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    handler_kind: str | None = None
    capabilities: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class OpenClawInstallImportRequest(BaseModel):
    """Full OpenClaw installation import — all subsystems are optional."""

    channels: list[OpenClawRecord] = Field(default_factory=list)
    sessions: list[OpenClawSessionRecord] = Field(default_factory=list)
    memory: list[OpenClawMemoryRecord] = Field(default_factory=list)
    tools: list[OpenClawToolRecord] = Field(default_factory=list)


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
# Hermes installation — per-subsystem record models
# ---------------------------------------------------------------------------


class HermesEnvRecord(BaseModel):
    """A single environment record exported from Hermes."""

    id: str
    name: str
    backend: str  # e.g. "docker", "local", "nix", "kubernetes"
    image: str | None = None
    template: str | None = None
    workspace_path: str | None = None
    env_passthrough: list[str] | None = None
    network_policy: dict[str, Any] | None = None
    caps_envelope: dict[str, Any] | None = None
    default_timeout_ms: int | None = None
    metadata: dict[str, Any] | None = None


class HermesProviderRecord(BaseModel):
    """A single model provider entry exported from Hermes."""

    id: str
    name: str
    kind: str  # anthropic, openai, openrouter, ollama, local
    base_url: str | None = None
    # auth is NOT imported (security concern) — always lossy
    model_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None


class HermesSessionRecord(BaseModel):
    """A single session with event-log history exported from Hermes."""

    id: str
    title: str | None = None
    agent_id: str | None = None
    user_profile_id: str | None = None
    created_at: str
    events: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


class HermesUserProfileRecord(BaseModel):
    """A single Honcho user profile exported from Hermes."""

    id: str
    username: str
    display_name: str | None = None
    email: str | None = None
    memories: list[str] | None = None
    capabilities: list[str] | None = None
    metadata: dict[str, Any] | None = None


class HermesCronRecord(BaseModel):
    """A single cron job exported from Hermes."""

    id: str
    trigger_type: str
    session_id: str
    name: str | None = None
    interval: str | None = None
    timestamp: str | None = None
    channel_id: str | None = None
    path: str | None = None
    webhook_id: str | None = None
    memory_key: str | None = None
    days_before: int | None = None
    capabilities: list[str] | None = None
    missed_fires_policy: str | None = None
    metadata: dict[str, Any] | None = None


class HermesAcpRecord(BaseModel):
    """A single ACP peer registry entry exported from Hermes."""

    id: str
    peer_id: str
    base_url: str
    allowed_capabilities: list[str] | None = None
    metadata: dict[str, Any] | None = None


class HermesInstallImportRequest(BaseModel):
    """Full Hermes installation import — all subsystems are optional."""

    skills: list[HermesRecord] = Field(default_factory=list)
    environments: list[HermesEnvRecord] = Field(default_factory=list)
    providers: list[HermesProviderRecord] = Field(default_factory=list)
    sessions: list[HermesSessionRecord] = Field(default_factory=list)
    user_profiles: list[HermesUserProfileRecord] = Field(default_factory=list)
    cron: list[HermesCronRecord] = Field(default_factory=list)
    acp_registry: list[HermesAcpRecord] = Field(default_factory=list)


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


def _translate_openclaw_session(
    rec: OpenClawSessionRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    """Translate one OpenClaw session record to a Meridian session record."""
    lossy: list[str] = []
    if not rec.created_at.strip():
        raise ValueError("created_at must not be empty")

    session_id = f"sess_{uuid.uuid4().hex}"
    thread_id = f"thread_{uuid.uuid4().hex}"

    meta: dict[str, Any] = {"openclaw_id": rec.id}
    if rec.title:
        meta["openclaw_title"] = rec.title
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"openclaw_meta_{k}"] = v
        lossy.append("metadata")

    events: list[dict[str, Any]] = []
    if rec.events:
        for seq, ev in enumerate(rec.events):
            normalized = dict(ev)
            normalized.setdefault("session_id", session_id)
            normalized.setdefault("thread_id", thread_id)
            normalized.setdefault("seq", seq)
            events.append(normalized)
    else:
        lossy.append("events_empty")

    session_record: dict[str, Any] = {
        "session_id": session_id,
        "thread_id": thread_id,
        "manifest": {
            "session_id": session_id,
            "agent_id": rec.agent_id,
            "agent_version_id": None,
            "thread_id": thread_id,
            "status": "archived",
            "created_at": rec.created_at,
            "imported_at": now,
            "metadata": meta,
        },
        "thread_record": {
            "thread_id": thread_id,
            "session_id": session_id,
            "created_at": rec.created_at,
        },
        "events": events,
        "event_count": len(events),
    }
    return session_record, lossy


def _translate_openclaw_memory_store(
    records: list[OpenClawMemoryRecord], *, now: str
) -> tuple[dict[str, Any], list[str]]:
    """Translate a list of OpenClaw memory records into one Meridian memory_store record."""
    lossy: list[str] = []
    store_id = f"memstore_{uuid.uuid4().hex}"

    if not records:
        lossy.append("memory_empty")

    store_record: dict[str, Any] = {
        "id": store_id,
        "name": "openclaw-memory",
        "backend": "sqlite-vec",
        "scope": "agent",
        "created_at": now,
        "metadata": {
            "from": "openclaw",
            "openclaw_source": "MEMORY.md",
            "entry_count": len(records),
        },
    }
    return store_record, lossy


def _translate_openclaw_tool(
    rec: OpenClawToolRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    """Translate one OpenClaw tool definition to a Meridian tool registry record.

    Capabilities not supplied by the source are filled with conservative defaults:
    allow_exec=False, allow_network=False, allow_file_write=False, allow_file_read=True,
    sandboxed=True.
    """
    lossy: list[str] = []

    if not rec.name.strip():
        raise ValueError("name must not be empty")

    handler_kind = rec.handler_kind
    if handler_kind is not None and handler_kind not in _KNOWN_HANDLER_KINDS:
        lossy.append("handler_kind_unknown")
    if handler_kind is None:
        lossy.append("handler_kind_missing")  # needs assignment post-import

    # Merge conservative defaults with any supplied caps; source wins where supplied.
    caps: dict[str, Any] = dict(_CONSERVATIVE_TOOL_CAPS)
    if rec.capabilities:
        caps.update(rec.capabilities)
        supplied = set(rec.capabilities.keys())
        overridden = supplied & set(_CONSERVATIVE_TOOL_CAPS.keys())
        if overridden - {"allow_file_read"}:
            # Any cap that relaxes beyond read-only is lossy — needs human review.
            lossy.append("capabilities_relaxed")
    else:
        lossy.append("capabilities_defaulted")  # all caps were conservative-defaulted

    tool_id = f"tool_{uuid.uuid4().hex}"
    meta: dict[str, Any] = {"openclaw_id": rec.id}
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"openclaw_meta_{k}"] = v
        lossy.append("metadata")

    tool_record: dict[str, Any] = {
        "id": tool_id,
        "name": rec.name,
        "description": rec.description,
        "input_schema": rec.input_schema or {"type": "object", "properties": {}},
        "handler_kind": handler_kind,
        "capabilities": caps,
        "created_at": now,
        "source": "imported",
        "source_type": "openclaw",
        "metadata": meta,
    }
    return tool_record, lossy


def _openclaw_install_checklist(
    body: OpenClawInstallImportRequest,
    results: dict[str, Any],
) -> list[str]:
    items: list[str] = []

    if results["channels"]["imported"]:
        needs_vault = results["channels"].get("needs_vault_ref_count", 0)
        if needs_vault:
            items.append(
                f"Assign config.token_vault_ref to {needs_vault} imported channel(s) "
                "before activating them"
            )

    if results["sessions"]["imported"]:
        items.append(
            f"Imported {results['sessions']['imported']} session(s) with status=archived; "
            "agent_id references may not resolve if agents were not also migrated"
        )

    if results["memory_stores"]["imported"]:
        items.append(
            f"Memory store created from MEMORY.md (scope=agent, tagged from:openclaw); "
            "write content via POST /v1/memory_stores/{store_id}/write to index entries"
        )

    tool_cap_notes: list[str] = []
    for rec in body.tools:
        if rec.capabilities and any(
            rec.capabilities.get(k) for k in ("allow_exec", "allow_network", "allow_file_write")
        ):
            tool_cap_notes.append(rec.name)
    if tool_cap_notes:
        items.append(
            f"Tool(s) {tool_cap_notes} had non-conservative capabilities in source — "
            "review capabilities field before activating"
        )

    missing_handler = [rec.name for rec in body.tools if rec.handler_kind is None]
    if missing_handler:
        items.append(
            f"Tool(s) {missing_handler} had no handler_kind — assign a handler before use"
        )

    totals = [
        f"channels={results['channels']['imported']}",
        f"sessions={results['sessions']['imported']}",
        f"memory_stores={results['memory_stores']['imported']}",
        f"tools={results['tools']['imported']}",
    ]
    items.append(f"OpenClaw installation import complete: {', '.join(totals)}")
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
# Hermes installation subsystem translations
# ---------------------------------------------------------------------------

_KNOWN_BACKENDS = frozenset(
    {"docker", "local", "nix", "kubernetes", "podman", "firecracker", "wasm"}
)
_VALID_TRIGGER_TYPES = frozenset(
    {"timestamp", "interval", "channel_event", "file_change", "webhook", "memory_anniversary"}
)


def _translate_hermes_env(
    rec: HermesEnvRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    lossy: list[str] = []
    env_id = f"env_{uuid.uuid4().hex}"

    if not rec.name.strip():
        raise ValueError("name must not be empty")
    if not rec.backend.strip():
        raise ValueError("backend must not be empty")

    if rec.backend not in _KNOWN_BACKENDS:
        lossy.append("backend_unknown")  # kept verbatim; may not be functional

    meta: dict[str, Any] = {"hermes_id": rec.id}
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"hermes_meta_{k}"] = v
        lossy.append("metadata")

    env_record: dict[str, Any] = {
        "id": env_id,
        "name": rec.name,
        "backend": rec.backend,
        "image": rec.image,
        "template": rec.template,
        "workspace_path": rec.workspace_path,
        "env_passthrough": rec.env_passthrough,
        "network_policy": rec.network_policy,
        "caps_envelope": rec.caps_envelope,
        "default_timeout_ms": rec.default_timeout_ms,
        "created_at": now,
        "updated_at": now,
        "metadata": meta,
    }
    return env_record, lossy


def _translate_hermes_provider(
    rec: HermesProviderRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    lossy: list[str] = []

    if not rec.name.strip():
        raise ValueError("name must not be empty")
    if not rec.kind.strip():
        raise ValueError("kind must not be empty")

    # auth is never imported — must be set manually after import
    lossy.append("auth_not_imported")
    if rec.model_ids is not None:
        lossy.append("model_ids_advisory")  # stored for reference only

    meta: dict[str, Any] = {"hermes_id": rec.id}
    if rec.model_ids is not None:
        meta["hermes_model_ids"] = rec.model_ids
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"hermes_meta_{k}"] = v
        lossy.append("metadata")

    provider_record: dict[str, Any] = {
        "name": rec.name,
        "kind": rec.kind,
        "base_url": rec.base_url,
        "auth": None,  # deliberately omitted — must be configured post-import
        "metadata": meta,
    }
    return provider_record, lossy


def _translate_hermes_session(
    rec: HermesSessionRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    lossy: list[str] = []
    session_id = f"sess_{uuid.uuid4().hex}"
    thread_id = f"thread_{uuid.uuid4().hex}"

    if not rec.created_at.strip():
        raise ValueError("created_at must not be empty")

    meta: dict[str, Any] = {"hermes_id": rec.id}
    if rec.title:
        meta["hermes_title"] = rec.title
    if rec.user_profile_id:
        meta["hermes_user_profile_id"] = rec.user_profile_id
        lossy.append("user_profile_id")  # original profile ID mapping not preserved
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"hermes_meta_{k}"] = v
        lossy.append("metadata")

    events: list[dict[str, Any]] = []
    if rec.events:
        for seq, ev in enumerate(rec.events):
            normalized = dict(ev)
            normalized.setdefault("session_id", session_id)
            normalized.setdefault("thread_id", thread_id)
            normalized.setdefault("seq", seq)
            events.append(normalized)
    else:
        lossy.append("events_empty")

    session_record: dict[str, Any] = {
        "session_id": session_id,
        "thread_id": thread_id,
        "manifest": {
            "session_id": session_id,
            "agent_id": rec.agent_id,
            "agent_version_id": None,
            "thread_id": thread_id,
            "status": "archived",
            "created_at": rec.created_at,
            "imported_at": now,
            "metadata": meta,
        },
        "thread_record": {
            "thread_id": thread_id,
            "session_id": session_id,
            "created_at": rec.created_at,
        },
        "events": events,
        "event_count": len(events),
    }
    return session_record, lossy


def _translate_hermes_user_profile(
    rec: HermesUserProfileRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    lossy: list[str] = []
    user_id = f"user_{uuid.uuid4().hex}"

    if not rec.username.strip():
        raise ValueError("username must not be empty")

    meta: dict[str, Any] = {"hermes_id": rec.id}
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"hermes_meta_{k}"] = v
        lossy.append("metadata")

    memories = list(rec.memories) if rec.memories else []
    if memories:
        lossy.append("memories_verbatim")  # stored as-is; not memory-store objects

    profile_record: dict[str, Any] = {
        "id": user_id,
        "username": rec.username,
        "display_name": rec.display_name,
        "email": rec.email,
        "capabilities": list(rec.capabilities) if rec.capabilities else [],
        "memories": memories,
        "is_primary": False,  # imported profiles are never primary
        "created_at": now,
        "updated_at": now,
        "metadata": meta,
    }
    return profile_record, lossy


def _translate_hermes_cron(
    rec: HermesCronRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    lossy: list[str] = []
    cron_id = f"cron_{uuid.uuid4().hex}"

    if not rec.session_id.strip():
        raise ValueError("session_id must not be empty")
    if rec.trigger_type not in _VALID_TRIGGER_TYPES:
        raise ValueError(f"unknown trigger_type: {rec.trigger_type!r}")

    meta: dict[str, Any] = {"hermes_id": rec.id}
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"hermes_meta_{k}"] = v
        lossy.append("metadata")

    missed_fires_policy = rec.missed_fires_policy or "skip"
    if missed_fires_policy not in {"catch_up", "skip"}:
        missed_fires_policy = "skip"
        lossy.append("missed_fires_policy_reset")

    next_fire_at: str | None = None
    if rec.trigger_type == "timestamp":
        next_fire_at = rec.timestamp
    # interval triggers: next_fire_at left None; scheduler computes on first check

    cron_record: dict[str, Any] = {
        "id": cron_id,
        "trigger_type": rec.trigger_type,
        "session_id": rec.session_id,
        "name": rec.name,
        "status": "active",
        "created_at": now,
        "next_fire_at": next_fire_at,
        "missed_fires_policy": missed_fires_policy,
        "capabilities": list(rec.capabilities) if rec.capabilities else [],
        "timestamp": rec.timestamp,
        "interval": rec.interval,
        "channel_id": rec.channel_id,
        "path": rec.path,
        "webhook_id": rec.webhook_id,
        "memory_key": rec.memory_key,
        "days_before": rec.days_before,
        "metadata": meta,
    }
    return cron_record, lossy


def _translate_hermes_acp(
    rec: HermesAcpRecord, *, now: str
) -> tuple[dict[str, Any], list[str]]:
    lossy: list[str] = []

    if not rec.peer_id.strip():
        raise ValueError("peer_id must not be empty")
    if not rec.base_url.strip():
        raise ValueError("base_url must not be empty")

    meta: dict[str, Any] = {"hermes_id": rec.id}
    if rec.metadata:
        for k, v in rec.metadata.items():
            meta[f"hermes_meta_{k}"] = v
        lossy.append("metadata")

    acp_record: dict[str, Any] = {
        "peer_id": rec.peer_id,
        "base_url": rec.base_url,
        "allowed_capabilities": list(rec.allowed_capabilities) if rec.allowed_capabilities else [],
        "imported_at": now,
        "metadata": meta,
    }
    return acp_record, lossy


# ---------------------------------------------------------------------------
# Checklist generation for install import
# ---------------------------------------------------------------------------


def _install_checklist(
    body: HermesInstallImportRequest,
    results: dict[str, Any],
) -> list[str]:
    items: list[str] = []

    if results["providers"]["imported"]:
        items.append(
            f"Merge {results['providers']['imported']} provider(s) from "
            "providers-imported.json into your config.yml providers section; "
            "set auth for each (auth was not imported)"
        )
    if results["acp_registry"]["imported"]:
        items.append(
            f"Merge {results['acp_registry']['imported']} ACP peer(s) from "
            "acp-peers-imported.json into your ACP adapter config"
        )

    env_unknown_backends = [
        rec.backend
        for rec in body.environments
        if rec.backend not in _KNOWN_BACKENDS
    ]
    if env_unknown_backends:
        unique = sorted(set(env_unknown_backends))
        items.append(
            f"Environment(s) with unrecognized backend(s) {unique} were imported verbatim; "
            "verify the backend driver is installed"
        )

    if results["sessions"]["imported"]:
        items.append(
            f"Imported {results['sessions']['imported']} session(s) with status=archived; "
            "agent_id references may not resolve if agents were not also migrated"
        )

    if results["user_profiles"]["imported"]:
        items.append(
            f"Imported {results['user_profiles']['imported']} user profile(s) as non-primary; "
            "promote one to primary via the API if needed"
        )

    if results["cron"]["imported"]:
        items.append(
            f"Imported {results['cron']['imported']} cron job(s); "
            "session_id references must resolve to imported or existing sessions"
        )

    totals = [
        f"skills={results['skills']['imported']}",
        f"environments={results['environments']['imported']}",
        f"providers={results['providers']['imported']}",
        f"sessions={results['sessions']['imported']}",
        f"user_profiles={results['user_profiles']['imported']}",
        f"cron={results['cron']['imported']}",
        f"acp_registry={results['acp_registry']['imported']}",
    ]
    items.append(f"Hermes installation import complete: {', '.join(totals)}")
    return items


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_imports_router(*, audit_log: AuditLog, storage_root: Path) -> APIRouter:
    router = APIRouter()
    channels_dir = storage_root / "channels"
    skills_dir = storage_root / "skills"
    versions_dir = storage_root / "skill_versions"
    envs_dir = storage_root / "environments"
    profiles_dir = storage_root / "user_profiles"
    cron_dir = storage_root / "cron"
    sessions_dir = storage_root / "sessions"
    memory_stores_dir = storage_root / "memory_stores"
    tools_dir = storage_root / "tools"

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

    @router.post("/v1/x/imports/openclaw/install", status_code=201)
    async def import_openclaw_install(body: OpenClawInstallImportRequest) -> JSONResponse:  # noqa: C901
        now = _now()
        tracer = get_tracer()
        total_records = (
            len(body.channels)
            + len(body.sessions)
            + len(body.memory)
            + len(body.tools)
        )
        import_audit = ImportAuditLog(storage_root, source="openclaw_install", timestamp=now)

        with tracer.start_as_current_span(
            "import.openclaw_install",
            attributes={
                "import.source": "openclaw_install",
                "import.record_count": total_records,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="import.openclaw_install.invocation",
                    code="import_openclaw_install",
                    timestamp=now,
                ),
            )
            import_audit.write_started(record_count=total_records, ts=now)

            results: dict[str, Any] = {
                "channels": {"imported": 0, "lossy_count": 0, "ids": [], "needs_vault_ref_count": 0},
                "sessions": {"imported": 0, "lossy_count": 0, "ids": []},
                "memory_stores": {"imported": 0, "lossy_count": 0, "ids": []},
                "tools": {"imported": 0, "lossy_count": 0, "ids": []},
            }
            all_written: list[Path] = []

            def _fail(err: MeridianError, *, event: str) -> None:
                record_error(span, err)
                import_audit.write_failed(code=err.code, message=err.message)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event=event,
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message, "audit_path": import_audit.path.name},
                    )
                )

            # ----------------------------------------------------------------
            # Phase 1: Translate all records in memory
            # ----------------------------------------------------------------

            # --- channels ---
            channel_records: list[dict[str, Any]] = []
            channel_lossy: list[list[str]] = []
            try:
                for seq, rec in enumerate(body.channels):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"channels[{seq}] has empty 'id'", timestamp=now, seq=seq
                        )
                    try:
                        cr, lossy = _translate_openclaw(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate channel '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq, source_id=rec.id, target_id=cr["id"], kind="channel", lossy_fields=lossy
                    )
                    channel_records.append(cr)
                    channel_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.openclaw_install.failed")
                raise

            # --- sessions ---
            session_records: list[dict[str, Any]] = []
            session_lossy: list[list[str]] = []
            seq_offset = len(body.channels)
            try:
                for seq, rec in enumerate(body.sessions):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"sessions[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        sr, lossy = _translate_openclaw_session(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate session '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=sr["session_id"],
                        kind="session",
                        lossy_fields=lossy,
                    )
                    session_records.append(sr)
                    session_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.openclaw_install.failed")
                raise

            # --- memory ---
            memory_store_record: dict[str, Any] | None = None
            memory_lossy: list[str] = []
            seq_offset += len(body.sessions)
            if body.memory:
                try:
                    memory_store_record, memory_lossy = _translate_openclaw_memory_store(
                        body.memory, now=now
                    )
                except Exception as exc:
                    err_mem = ImportRecordInvalidError(
                        message=f"Failed to translate memory records: {exc}",
                        timestamp=now,
                        seq=seq_offset,
                    )
                    _fail(err_mem, event="import.openclaw_install.failed")
                    raise err_mem from exc
                import_audit.write_record_translated(
                    seq=seq_offset,
                    source_id="MEMORY.md",
                    target_id=memory_store_record["id"],
                    kind="memory_store",
                    lossy_fields=memory_lossy,
                )

            # --- tools ---
            tool_records: list[dict[str, Any]] = []
            tool_lossy: list[list[str]] = []
            seq_offset += (1 if body.memory else 0)
            try:
                for seq, rec in enumerate(body.tools):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"tools[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        tr, lossy = _translate_openclaw_tool(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate tool '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=tr["id"],
                        kind="tool",
                        lossy_fields=lossy,
                    )
                    tool_records.append(tr)
                    tool_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.openclaw_install.failed")
                raise

            # ----------------------------------------------------------------
            # Phase 2: Write all records transactionally
            # ----------------------------------------------------------------
            try:
                # channels
                if channel_records:
                    channels_dir.mkdir(parents=True, exist_ok=True)
                    for ch in channel_records:
                        path = channels_dir / f"{ch['id']}.json"
                        path.write_text(json.dumps(ch))
                        all_written.append(path)

                # sessions + event-log replay
                if session_records:
                    sessions_dir.mkdir(parents=True, exist_ok=True)
                    for sr in session_records:
                        sess_dir = sessions_dir / sr["session_id"]
                        sess_dir.mkdir(parents=True, exist_ok=True)
                        mpath = sess_dir / "manifest.json"
                        mpath.write_text(json.dumps(sr["manifest"]))
                        all_written.append(mpath)
                        threads_dir = sess_dir / "threads"
                        threads_dir.mkdir(parents=True, exist_ok=True)
                        tpath = threads_dir / f"{sr['thread_id']}.json"
                        tpath.write_text(json.dumps(sr["thread_record"]))
                        all_written.append(tpath)
                        if sr["events"]:
                            elog_path = sess_dir / "events.ndjson"
                            lines = (
                                "\n".join(
                                    json.dumps(ev, separators=(",", ":")) for ev in sr["events"]
                                )
                                + "\n"
                            )
                            elog_path.write_text(lines)
                            all_written.append(elog_path)

                # memory store — record + raw content files
                if memory_store_record is not None:
                    memory_stores_dir.mkdir(parents=True, exist_ok=True)
                    store_path = memory_stores_dir / f"{memory_store_record['id']}.json"
                    store_path.write_text(json.dumps(memory_store_record))
                    all_written.append(store_path)
                    raw_dir = memory_stores_dir / memory_store_record["id"] / "raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    for mem_rec in body.memory:
                        raw_path = raw_dir / f"{mem_rec.key.replace('/', '_')}.md"
                        raw_path.write_text(mem_rec.content)
                        all_written.append(raw_path)

                # tools
                if tool_records:
                    tools_dir.mkdir(parents=True, exist_ok=True)
                    for tr in tool_records:
                        tpath = tools_dir / f"{tr['id']}.json"
                        tpath.write_text(json.dumps(tr))
                        all_written.append(tpath)

            except Exception as exc:
                for p in all_written:
                    with contextlib.suppress(OSError):
                        p.unlink()
                err2 = ImportWriteError(
                    message=f"Failed to write OpenClaw install records: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                _fail(err2, event="import.openclaw_install.failed")
                raise err2

            # ----------------------------------------------------------------
            # Phase 3: Populate results and finalize audit log
            # ----------------------------------------------------------------
            needs_vault = sum(
                1 for l in channel_lossy if "token_vault_ref_missing" in l
            )
            results["channels"]["imported"] = len(channel_records)
            results["channels"]["lossy_count"] = sum(1 for l in channel_lossy if l)
            results["channels"]["ids"] = [ch["id"] for ch in channel_records]
            results["channels"]["needs_vault_ref_count"] = needs_vault

            results["sessions"]["imported"] = len(session_records)
            results["sessions"]["lossy_count"] = sum(1 for l in session_lossy if l)
            results["sessions"]["ids"] = [s["session_id"] for s in session_records]

            if memory_store_record is not None:
                results["memory_stores"]["imported"] = 1
                results["memory_stores"]["lossy_count"] = 1 if memory_lossy else 0
                results["memory_stores"]["ids"] = [memory_store_record["id"]]

            results["tools"]["imported"] = len(tool_records)
            results["tools"]["lossy_count"] = sum(1 for l in tool_lossy if l)
            results["tools"]["ids"] = [t["id"] for t in tool_records]

            checklist = _openclaw_install_checklist(body, results)
            import_audit.write_checklist(checklist)
            import_audit.write_completed(
                total=total_records,
                lossy_count=sum(r["lossy_count"] for r in results.values()),
            )

        return JSONResponse(
            content={
                "audit_path": import_audit.path.name,
                **results,
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

    @router.post("/v1/x/imports/hermes/install", status_code=201)
    async def import_hermes_install(body: HermesInstallImportRequest) -> JSONResponse:  # noqa: C901
        now = _now()
        tracer = get_tracer()
        total_records = (
            len(body.skills)
            + len(body.environments)
            + len(body.providers)
            + len(body.sessions)
            + len(body.user_profiles)
            + len(body.cron)
            + len(body.acp_registry)
        )
        import_audit = ImportAuditLog(storage_root, source="hermes_install", timestamp=now)

        with tracer.start_as_current_span(
            "import.hermes_install",
            attributes={
                "import.source": "hermes_install",
                "import.record_count": total_records,
            },
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="import.hermes_install.invocation",
                    code="import_hermes_install",
                    timestamp=now,
                ),
            )
            import_audit.write_started(record_count=total_records, ts=now)

            # Accumulated results per subsystem (populated below)
            results: dict[str, Any] = {
                "skills": {"imported": 0, "lossy_count": 0, "ids": []},
                "environments": {"imported": 0, "lossy_count": 0, "ids": []},
                "providers": {"imported": 0, "lossy_count": 0, "fragment_path": "providers-imported.json"},
                "sessions": {"imported": 0, "lossy_count": 0, "ids": []},
                "user_profiles": {"imported": 0, "lossy_count": 0, "ids": []},
                "cron": {"imported": 0, "lossy_count": 0, "ids": []},
                "acp_registry": {"imported": 0, "lossy_count": 0, "fragment_path": "acp-peers-imported.json"},
            }
            all_written: list[Path] = []

            def _fail(err: MeridianError, *, event: str) -> None:
                record_error(span, err)
                import_audit.write_failed(code=err.code, message=err.message)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event=event,
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": err.message, "audit_path": import_audit.path.name},
                    )
                )

            # ----------------------------------------------------------------
            # Phase 1: Translate all records in memory
            # ----------------------------------------------------------------

            # --- skills ---
            skill_records: list[dict[str, Any]] = []
            version_records: list[dict[str, Any]] = []
            skill_lossy: list[list[str]] = []
            try:
                for seq, rec in enumerate(body.skills):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"skills[{seq}] has empty 'id'", timestamp=now, seq=seq
                        )
                    try:
                        sr, vr, lossy = _translate_hermes(rec, now=now)
                    except ImportRecordInvalidError:
                        raise
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate skill '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq, source_id=rec.id, target_id=sr["id"], kind="skill", lossy_fields=lossy
                    )
                    skill_records.append(sr)
                    version_records.append(vr)
                    skill_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.hermes_install.failed")
                raise

            # --- environments ---
            env_records: list[dict[str, Any]] = []
            env_lossy: list[list[str]] = []
            seq_offset = len(body.skills)
            try:
                for seq, rec in enumerate(body.environments):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"environments[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        er, lossy = _translate_hermes_env(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate environment '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=er["id"],
                        kind="environment",
                        lossy_fields=lossy,
                    )
                    env_records.append(er)
                    env_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.hermes_install.failed")
                raise

            # --- providers ---
            provider_records: list[dict[str, Any]] = []
            provider_lossy: list[list[str]] = []
            seq_offset += len(body.environments)
            try:
                for seq, rec in enumerate(body.providers):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"providers[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        pr, lossy = _translate_hermes_provider(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate provider '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=pr["name"],
                        kind="provider",
                        lossy_fields=lossy,
                    )
                    provider_records.append(pr)
                    provider_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.hermes_install.failed")
                raise

            # --- sessions ---
            session_records: list[dict[str, Any]] = []
            session_lossy: list[list[str]] = []
            seq_offset += len(body.providers)
            try:
                for seq, rec in enumerate(body.sessions):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"sessions[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        sr, lossy = _translate_hermes_session(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate session '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=sr["session_id"],
                        kind="session",
                        lossy_fields=lossy,
                    )
                    session_records.append(sr)
                    session_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.hermes_install.failed")
                raise

            # --- user_profiles ---
            profile_records: list[dict[str, Any]] = []
            profile_lossy: list[list[str]] = []
            seq_offset += len(body.sessions)
            try:
                for seq, rec in enumerate(body.user_profiles):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"user_profiles[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        pr, lossy = _translate_hermes_user_profile(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate user_profile '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=pr["id"],
                        kind="user_profile",
                        lossy_fields=lossy,
                    )
                    profile_records.append(pr)
                    profile_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.hermes_install.failed")
                raise

            # --- cron ---
            cron_records: list[dict[str, Any]] = []
            cron_lossy: list[list[str]] = []
            seq_offset += len(body.user_profiles)
            try:
                for seq, rec in enumerate(body.cron):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"cron[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        cr, lossy = _translate_hermes_cron(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate cron '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=cr["id"],
                        kind="cron",
                        lossy_fields=lossy,
                    )
                    cron_records.append(cr)
                    cron_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.hermes_install.failed")
                raise

            # --- acp_registry ---
            acp_records: list[dict[str, Any]] = []
            acp_lossy: list[list[str]] = []
            seq_offset += len(body.cron)
            try:
                for seq, rec in enumerate(body.acp_registry):
                    if not rec.id.strip():
                        raise ImportRecordInvalidError(
                            message=f"acp_registry[{seq}] has empty 'id'",
                            timestamp=now,
                            seq=seq_offset + seq,
                        )
                    try:
                        ar, lossy = _translate_hermes_acp(rec, now=now)
                    except Exception as exc:
                        raise ImportRecordInvalidError(
                            message=f"Failed to translate acp_registry entry '{rec.id}': {exc}",
                            timestamp=now,
                            seq=seq_offset + seq,
                        ) from exc
                    import_audit.write_record_translated(
                        seq=seq_offset + seq,
                        source_id=rec.id,
                        target_id=ar["peer_id"],
                        kind="acp_peer",
                        lossy_fields=lossy,
                    )
                    acp_records.append(ar)
                    acp_lossy.append(lossy)
            except ImportRecordInvalidError as err:
                _fail(err, event="import.hermes_install.failed")
                raise

            # ----------------------------------------------------------------
            # Phase 2: Write all records transactionally
            # ----------------------------------------------------------------
            try:
                # skills
                if skill_records:
                    skills_dir.mkdir(parents=True, exist_ok=True)
                    versions_dir.mkdir(parents=True, exist_ok=True)
                    for sr, vr in zip(skill_records, version_records):
                        vpath = versions_dir / f"{vr['id']}.json"
                        vpath.write_text(json.dumps(vr))
                        all_written.append(vpath)
                        spath = skills_dir / f"{sr['id']}.json"
                        spath.write_text(json.dumps(sr))
                        all_written.append(spath)

                # environments
                if env_records:
                    envs_dir.mkdir(parents=True, exist_ok=True)
                    for er in env_records:
                        epath = envs_dir / f"{er['id']}.json"
                        epath.write_text(json.dumps(er))
                        all_written.append(epath)

                # providers — config fragment
                if provider_records:
                    pfrag = storage_root / "providers-imported.json"
                    pfrag.write_text(json.dumps(provider_records, indent=2))
                    all_written.append(pfrag)

                # sessions + event log replay
                if session_records:
                    sessions_dir.mkdir(parents=True, exist_ok=True)
                    for sr in session_records:
                        sess_dir = sessions_dir / sr["session_id"]
                        sess_dir.mkdir(parents=True, exist_ok=True)
                        mpath = sess_dir / "manifest.json"
                        mpath.write_text(json.dumps(sr["manifest"]))
                        all_written.append(mpath)
                        threads_dir = sess_dir / "threads"
                        threads_dir.mkdir(parents=True, exist_ok=True)
                        tpath = threads_dir / f"{sr['thread_id']}.json"
                        tpath.write_text(json.dumps(sr["thread_record"]))
                        all_written.append(tpath)
                        if sr["events"]:
                            elog_path = sess_dir / "events.ndjson"
                            lines = "\n".join(json.dumps(ev, separators=(",", ":")) for ev in sr["events"]) + "\n"
                            elog_path.write_text(lines)
                            all_written.append(elog_path)

                # user_profiles
                if profile_records:
                    profiles_dir.mkdir(parents=True, exist_ok=True)
                    for pr in profile_records:
                        ppath = profiles_dir / f"{pr['id']}.json"
                        ppath.write_text(json.dumps(pr))
                        all_written.append(ppath)

                # cron
                if cron_records:
                    cron_dir.mkdir(parents=True, exist_ok=True)
                    for cr in cron_records:
                        cpath = cron_dir / f"{cr['id']}.json"
                        cpath.write_text(json.dumps(cr))
                        all_written.append(cpath)

                # acp_registry — config fragment
                if acp_records:
                    afrag = storage_root / "acp-peers-imported.json"
                    afrag.write_text(json.dumps(acp_records, indent=2))
                    all_written.append(afrag)

            except Exception as exc:
                for p in all_written:
                    with contextlib.suppress(OSError):
                        p.unlink()
                err2 = ImportWriteError(
                    message=f"Failed to write Hermes install records: {exc}",
                    timestamp=_now(),
                    cause=exc,
                )
                _fail(err2, event="import.hermes_install.failed")
                raise err2

            # ----------------------------------------------------------------
            # Phase 3: Populate results and finalize audit log
            # ----------------------------------------------------------------
            results["skills"]["imported"] = len(skill_records)
            results["skills"]["lossy_count"] = sum(1 for l in skill_lossy if l)
            results["skills"]["ids"] = [s["id"] for s in skill_records]

            results["environments"]["imported"] = len(env_records)
            results["environments"]["lossy_count"] = sum(1 for l in env_lossy if l)
            results["environments"]["ids"] = [e["id"] for e in env_records]

            results["providers"]["imported"] = len(provider_records)
            results["providers"]["lossy_count"] = sum(1 for l in provider_lossy if l)

            results["sessions"]["imported"] = len(session_records)
            results["sessions"]["lossy_count"] = sum(1 for l in session_lossy if l)
            results["sessions"]["ids"] = [s["session_id"] for s in session_records]

            results["user_profiles"]["imported"] = len(profile_records)
            results["user_profiles"]["lossy_count"] = sum(1 for l in profile_lossy if l)
            results["user_profiles"]["ids"] = [p["id"] for p in profile_records]

            results["cron"]["imported"] = len(cron_records)
            results["cron"]["lossy_count"] = sum(1 for l in cron_lossy if l)
            results["cron"]["ids"] = [c["id"] for c in cron_records]

            results["acp_registry"]["imported"] = len(acp_records)
            results["acp_registry"]["lossy_count"] = sum(1 for l in acp_lossy if l)

            checklist = _install_checklist(body, results)
            import_audit.write_checklist(checklist)
            import_audit.write_completed(
                total=total_records,
                lossy_count=sum(r["lossy_count"] for r in results.values()),
            )

        return JSONResponse(
            content={
                "audit_path": import_audit.path.name,
                **results,
            },
            status_code=201,
        )

    return router
