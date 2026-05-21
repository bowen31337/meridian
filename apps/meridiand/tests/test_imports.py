"""
Import endpoints conformance suite.

Tests cover:
  - POST /v1/x/imports/openclaw returns 201 on success.
  - POST /v1/x/imports/openclaw response has imported, lossy_count, audit_path, channel_ids.
  - POST /v1/x/imports/openclaw creates a meridian-import-*.audit.ndjson in storage_root.
  - POST /v1/x/imports/openclaw audit file has import_started line as first entry.
  - POST /v1/x/imports/openclaw audit file has one record_translated line per input record.
  - POST /v1/x/imports/openclaw record_translated line has seq, source_id, target_id, kind, lossy_fields.
  - POST /v1/x/imports/openclaw record_translated kind is "channel".
  - POST /v1/x/imports/openclaw audit file has checklist line after all record_translated lines.
  - POST /v1/x/imports/openclaw checklist includes token_vault_ref_missing item when config absent.
  - POST /v1/x/imports/openclaw checklist includes webhook_url item when webhook_url present.
  - POST /v1/x/imports/openclaw checklist includes kind_remapped item for unknown kinds.
  - POST /v1/x/imports/openclaw audit file has import_completed line as last entry.
  - POST /v1/x/imports/openclaw import_completed has total and lossy_count.
  - POST /v1/x/imports/openclaw channels written to storage_root/channels/{id}.json.
  - POST /v1/x/imports/openclaw channel record has id with "ch_" prefix.
  - POST /v1/x/imports/openclaw channel record has kind, config, created_at.
  - POST /v1/x/imports/openclaw known kind passed through unchanged.
  - POST /v1/x/imports/openclaw unknown kind mapped to "generic".
  - POST /v1/x/imports/openclaw unknown kind produces kind_remapped_to_generic lossy field.
  - POST /v1/x/imports/openclaw webhook_url stored in config.webhook_url.
  - POST /v1/x/imports/openclaw webhook_url lossy field recorded.
  - POST /v1/x/imports/openclaw source id stored in metadata.openclaw_id.
  - POST /v1/x/imports/openclaw name stored in metadata.openclaw_name.
  - POST /v1/x/imports/openclaw metadata keys stored with openclaw_meta_ prefix.
  - POST /v1/x/imports/openclaw is transactional: no files written on validation failure.
  - POST /v1/x/imports/openclaw validation failure returns 422 with code "import_record_invalid".
  - POST /v1/x/imports/openclaw failure writes import_failed line to import audit log.
  - POST /v1/x/imports/openclaw failure writes to main audit.ndjson.
  - POST /v1/x/imports/openclaw empty records list returns 201 with imported=0.
  - POST /v1/x/imports/hermes returns 201 on success.
  - POST /v1/x/imports/hermes response has imported, lossy_count, audit_path, skill_ids.
  - POST /v1/x/imports/hermes creates a meridian-import-*.audit.ndjson in storage_root.
  - POST /v1/x/imports/hermes audit file has import_started line as first entry.
  - POST /v1/x/imports/hermes audit file has one record_translated line per input record.
  - POST /v1/x/imports/hermes record_translated kind is "skill".
  - POST /v1/x/imports/hermes audit file has checklist line after all record_translated lines.
  - POST /v1/x/imports/hermes checklist includes version_tag item when version_tag present.
  - POST /v1/x/imports/hermes checklist includes tags item when tags present.
  - POST /v1/x/imports/hermes checklist includes is_public item for public skills.
  - POST /v1/x/imports/hermes audit file has import_completed line as last entry.
  - POST /v1/x/imports/hermes import_completed has total and lossy_count.
  - POST /v1/x/imports/hermes skills written to storage_root/skills/{id}.json.
  - POST /v1/x/imports/hermes versions written to storage_root/skill_versions/{ver_id}.json.
  - POST /v1/x/imports/hermes skill record has id with "skill_" prefix.
  - POST /v1/x/imports/hermes skill version has id with "skillver_" prefix.
  - POST /v1/x/imports/hermes version_number is 1 for all imported skills.
  - POST /v1/x/imports/hermes version source_type is "hermes".
  - POST /v1/x/imports/hermes version source is "imported".
  - POST /v1/x/imports/hermes hermes id stored in metadata.hermes_id.
  - POST /v1/x/imports/hermes version_tag stored in metadata.hermes_version_tag (lossy).
  - POST /v1/x/imports/hermes tags stored in metadata.hermes_tags (lossy).
  - POST /v1/x/imports/hermes is_public stored in metadata.hermes_is_public (lossy).
  - POST /v1/x/imports/hermes is transactional: no files written on validation failure.
  - POST /v1/x/imports/hermes validation failure returns 422 with code "import_record_invalid".
  - POST /v1/x/imports/hermes failure writes import_failed line to import audit log.
  - POST /v1/x/imports/hermes failure writes to main audit.ndjson.
  - POST /v1/x/imports/hermes empty name returns 422.
  - POST /v1/x/imports/hermes empty instructions returns 422.
  - POST /v1/x/imports/hermes empty tools returns 422.
  - POST /v1/x/imports/openclaw/install returns 201 on success.
  - POST /v1/x/imports/openclaw/install response has audit_path and per-subsystem keys.
  - POST /v1/x/imports/openclaw/install channels/sessions/memory_stores/tools imported counts correct.
  - POST /v1/x/imports/openclaw/install empty subsystems returns zeros.
  - POST /v1/x/imports/openclaw/install creates meridian-import-*.audit.ndjson.
  - POST /v1/x/imports/openclaw/install audit has import_started as first entry.
  - POST /v1/x/imports/openclaw/install audit has record_translated for each record across all subsystems.
  - POST /v1/x/imports/openclaw/install record_translated kinds cover all subsystem types.
  - POST /v1/x/imports/openclaw/install audit ends with import_completed.
  - POST /v1/x/imports/openclaw/install audit has checklist entry.
  - POST /v1/x/imports/openclaw/install channels written to storage_root/channels/{id}.json.
  - POST /v1/x/imports/openclaw/install channel id has "ch_" prefix.
  - POST /v1/x/imports/openclaw/install sessions written to sessions/{id}/manifest.json.
  - POST /v1/x/imports/openclaw/install session id has "sess_" prefix.
  - POST /v1/x/imports/openclaw/install session status is "archived".
  - POST /v1/x/imports/openclaw/install session openclaw_id stored in metadata.
  - POST /v1/x/imports/openclaw/install session events written as events.ndjson.
  - POST /v1/x/imports/openclaw/install session thread written to threads/{id}.json.
  - POST /v1/x/imports/openclaw/install no events.ndjson when events list is empty.
  - POST /v1/x/imports/openclaw/install empty session created_at returns 422.
  - POST /v1/x/imports/openclaw/install memory store written to memory_stores/{id}.json.
  - POST /v1/x/imports/openclaw/install memory store id has "memstore_" prefix.
  - POST /v1/x/imports/openclaw/install memory store scope is "agent".
  - POST /v1/x/imports/openclaw/install memory store metadata has from=openclaw.
  - POST /v1/x/imports/openclaw/install memory store raw content written to memory_stores/{id}/raw/.
  - POST /v1/x/imports/openclaw/install no memory_store written when memory list is empty.
  - POST /v1/x/imports/openclaw/install tools written to tools/{id}.json.
  - POST /v1/x/imports/openclaw/install tool id has "tool_" prefix.
  - POST /v1/x/imports/openclaw/install tool conservative caps applied when capabilities absent.
  - POST /v1/x/imports/openclaw/install tool allow_exec defaults to False.
  - POST /v1/x/imports/openclaw/install tool allow_network defaults to False.
  - POST /v1/x/imports/openclaw/install tool allow_file_write defaults to False.
  - POST /v1/x/imports/openclaw/install tool allow_file_read defaults to True.
  - POST /v1/x/imports/openclaw/install tool sandboxed defaults to True.
  - POST /v1/x/imports/openclaw/install tool source is "imported", source_type is "openclaw".
  - POST /v1/x/imports/openclaw/install tool openclaw_id stored in metadata.
  - POST /v1/x/imports/openclaw/install tool empty name returns 422.
  - POST /v1/x/imports/openclaw/install tool empty id returns 422.
  - POST /v1/x/imports/openclaw/install is transactional: no files written on validation failure.
  - POST /v1/x/imports/openclaw/install validation failure returns 422 with import_record_invalid.
  - POST /v1/x/imports/openclaw/install failure writes import_failed to import audit log.
  - POST /v1/x/imports/openclaw/install failure writes to main audit.ndjson.
  - POST /v1/x/imports/openclaw/install OTel span "import.openclaw_install" emitted on success.
  - create_app wires /v1/x/imports/openclaw/install route when storage_root is supplied.
  - create_app wires imports router when storage_root is supplied.
  - create_app omits imports routes when storage_root is None.
  - OTel span "import.openclaw" emitted on success.
  - OTel span "import.openclaw" emitted on failure.
  - OTel span "import.hermes" emitted on success.
  - OTel span "import.hermes" emitted on failure.
  - POST /v1/x/imports/hermes/install returns 201 on success.
  - POST /v1/x/imports/hermes/install response has audit_path and per-subsystem keys.
  - POST /v1/x/imports/hermes/install skills/environments/providers/sessions/user_profiles/cron/acp_registry imported counts correct.
  - POST /v1/x/imports/hermes/install empty subsystems returns zeros.
  - POST /v1/x/imports/hermes/install creates meridian-import-*.audit.ndjson.
  - POST /v1/x/imports/hermes/install audit has import_started as first entry.
  - POST /v1/x/imports/hermes/install audit has record_translated for each record across all subsystems.
  - POST /v1/x/imports/hermes/install record_translated kinds cover all subsystem types.
  - POST /v1/x/imports/hermes/install audit ends with import_completed.
  - POST /v1/x/imports/hermes/install audit has checklist entry.
  - POST /v1/x/imports/hermes/install environments written to storage_root/environments/{id}.json.
  - POST /v1/x/imports/hermes/install environment id has "env_" prefix.
  - POST /v1/x/imports/hermes/install environment backend preserved.
  - POST /v1/x/imports/hermes/install environment hermes_id stored in metadata.
  - POST /v1/x/imports/hermes/install unknown backend marks lossy.
  - POST /v1/x/imports/hermes/install empty env name returns 422.
  - POST /v1/x/imports/hermes/install providers written to providers-imported.json fragment.
  - POST /v1/x/imports/hermes/install provider fragment is JSON array.
  - POST /v1/x/imports/hermes/install provider auth is null.
  - POST /v1/x/imports/hermes/install provider always lossy due to auth.
  - POST /v1/x/imports/hermes/install provider hermes_id in metadata.
  - POST /v1/x/imports/hermes/install empty provider name returns 422.
  - POST /v1/x/imports/hermes/install session manifest written to sessions/{id}/manifest.json.
  - POST /v1/x/imports/hermes/install session id has "sess_" prefix.
  - POST /v1/x/imports/hermes/install session status is "archived".
  - POST /v1/x/imports/hermes/install session hermes_id in metadata.
  - POST /v1/x/imports/hermes/install session events written as events.ndjson.
  - POST /v1/x/imports/hermes/install session thread written to threads/{id}.json.
  - POST /v1/x/imports/hermes/install no events.ndjson when events list is empty.
  - POST /v1/x/imports/hermes/install empty session created_at returns 422.
  - POST /v1/x/imports/hermes/install user_profile written to user_profiles/{id}.json.
  - POST /v1/x/imports/hermes/install user_profile id has "user_" prefix.
  - POST /v1/x/imports/hermes/install user_profile is_primary is False.
  - POST /v1/x/imports/hermes/install user_profile hermes_id in metadata.
  - POST /v1/x/imports/hermes/install memories stored verbatim and marked lossy.
  - POST /v1/x/imports/hermes/install empty username returns 422.
  - POST /v1/x/imports/hermes/install cron written to cron/{id}.json.
  - POST /v1/x/imports/hermes/install cron id has "cron_" prefix.
  - POST /v1/x/imports/hermes/install cron status is "active".
  - POST /v1/x/imports/hermes/install cron hermes_id in metadata.
  - POST /v1/x/imports/hermes/install invalid trigger_type returns 422.
  - POST /v1/x/imports/hermes/install empty session_id returns 422.
  - POST /v1/x/imports/hermes/install timestamp trigger sets next_fire_at.
  - POST /v1/x/imports/hermes/install acp fragment written to acp-peers-imported.json.
  - POST /v1/x/imports/hermes/install acp fragment is JSON array.
  - POST /v1/x/imports/hermes/install acp peer_id preserved.
  - POST /v1/x/imports/hermes/install acp base_url preserved.
  - POST /v1/x/imports/hermes/install acp hermes_id in metadata.
  - POST /v1/x/imports/hermes/install empty peer_id returns 422.
  - POST /v1/x/imports/hermes/install validation failure returns 422 with import_record_invalid.
  - POST /v1/x/imports/hermes/install no files written on validation failure.
  - POST /v1/x/imports/hermes/install failure writes import_failed to import audit log.
  - POST /v1/x/imports/hermes/install failure writes to main audit.ndjson.
  - create_app wires /v1/x/imports/hermes/install route when storage_root is supplied.
  - OTel span "import.hermes_install" emitted on success.
  - OTel span "import.hermes_install" emitted on failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._imports import make_imports_router

import tests._otel_shared as _otel_shared


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _main_audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _import_audit_records(storage_root: Path) -> list[dict]:
    files = sorted(storage_root.glob("meridian-import-*.audit.ndjson"))
    if not files:
        return []
    return [json.loads(line) for line in files[-1].read_text().splitlines() if line.strip()]


def _import_audit_path(storage_root: Path) -> Path | None:
    files = sorted(storage_root.glob("meridian-import-*.audit.ndjson"))
    return files[-1] if files else None


def _openclaw_body(**overrides: Any) -> dict:
    base: dict = {
        "records": [
            {
                "id": "oc_001",
                "kind": "telegram",
                "name": "My Channel",
                "config": {"token_vault_ref": "vlt_abc"},
            }
        ]
    }
    base.update(overrides)
    return base


def _hermes_body(**overrides: Any) -> dict:
    base: dict = {
        "records": [
            {
                "id": "h_001",
                "name": "my-skill",
                "description": "Does something",
                "instructions": "Step 1: do it.",
                "tools": [{"name": "bash", "description": "Run shell"}],
            }
        ]
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# OpenClaw – success
# ---------------------------------------------------------------------------


class TestOpenClawImportSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        assert resp.status_code == 201

    def test_response_has_expected_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json=_openclaw_body()).json()
        assert "imported" in body
        assert "lossy_count" in body
        assert "audit_path" in body
        assert "channel_ids" in body

    def test_imported_count_matches_records(self, storage_root: Path) -> None:
        records = [
            {"id": "oc_001", "kind": "telegram", "name": "Ch1", "config": {"token_vault_ref": "v1"}},
            {"id": "oc_002", "kind": "slack", "name": "Ch2", "config": {"token_vault_ref": "v2"}},
        ]
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": records}).json()
        assert body["imported"] == 2
        assert len(body["channel_ids"]) == 2

    def test_empty_records_returns_201_with_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": []}).json()
        assert body["imported"] == 0
        assert body["channel_ids"] == []


# ---------------------------------------------------------------------------
# OpenClaw – import audit log
# ---------------------------------------------------------------------------


class TestOpenClawAuditLog:
    def test_audit_file_created(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        files = list(storage_root.glob("meridian-import-*.audit.ndjson"))
        assert len(files) == 1

    def test_audit_path_in_response_matches_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json=_openclaw_body()).json()
        assert (storage_root / body["audit_path"]).exists()

    def test_first_entry_is_import_started(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        assert entries[0]["type"] == "import_started"
        assert entries[0]["source"] == "openclaw"

    def test_import_started_has_record_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        assert entries[0]["record_count"] == 1

    def test_record_translated_entries_present(self, storage_root: Path) -> None:
        records = [
            {"id": "oc_1", "kind": "telegram", "config": {"token_vault_ref": "v1"}},
            {"id": "oc_2", "kind": "slack", "config": {"token_vault_ref": "v2"}},
        ]
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": records})
        entries = _import_audit_records(storage_root)
        translated = [e for e in entries if e["type"] == "record_translated"]
        assert len(translated) == 2

    def test_record_translated_has_required_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert "seq" in t
        assert "source_id" in t
        assert "target_id" in t
        assert "kind" in t
        assert "lossy_fields" in t

    def test_record_translated_kind_is_channel(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert t["kind"] == "channel"

    def test_record_translated_source_id_matches_input(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert t["source_id"] == "oc_001"

    def test_record_translated_target_id_has_ch_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert t["target_id"].startswith("ch_")

    def test_checklist_entry_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        checklist = [e for e in entries if e["type"] == "checklist"]
        assert len(checklist) == 1

    def test_checklist_has_items_list(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "checklist")
        assert isinstance(c["items"], list)
        assert len(c["items"]) > 0

    def test_checklist_includes_token_vault_ref_missing(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [{"id": "oc_1", "kind": "telegram"}]})
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "checklist")
        assert any("token_vault_ref" in item for item in c["items"])

    def test_checklist_includes_webhook_url_item(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "telegram", "webhook_url": "https://example.com/hook"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "checklist")
        assert any("webhook_url" in item for item in c["items"])

    def test_checklist_includes_kind_remapped_item(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "carrier_pigeon"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "checklist")
        assert any("carrier_pigeon" in item for item in c["items"])

    def test_last_entry_is_import_completed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        assert entries[-1]["type"] == "import_completed"

    def test_import_completed_has_total_and_lossy_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "import_completed")
        assert "total" in c
        assert "lossy_count" in c

    def test_ordering_is_started_translated_checklist_completed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        entries = _import_audit_records(storage_root)
        types = [e["type"] for e in entries]
        assert types[0] == "import_started"
        assert "record_translated" in types
        checklist_idx = types.index("checklist")
        completed_idx = types.index("import_completed")
        first_translated_idx = types.index("record_translated")
        assert first_translated_idx < checklist_idx < completed_idx


# ---------------------------------------------------------------------------
# OpenClaw – channel record content
# ---------------------------------------------------------------------------


class TestOpenClawChannelRecords:
    def test_channel_written_to_disk(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json=_openclaw_body()).json()
        ch_id = body["channel_ids"][0]
        assert (storage_root / "channels" / f"{ch_id}.json").exists()

    def test_channel_id_has_ch_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json=_openclaw_body()).json()
        assert body["channel_ids"][0].startswith("ch_")

    def test_known_kind_passed_through(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "slack", "config": {"token_vault_ref": "v1"}}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": [rec]}).json()
        ch = json.loads((storage_root / "channels" / f"{body['channel_ids'][0]}.json").read_text())
        assert ch["kind"] == "slack"

    def test_unknown_kind_mapped_to_generic(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "carrier_pigeon"}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": [rec]}).json()
        ch = json.loads((storage_root / "channels" / f"{body['channel_ids'][0]}.json").read_text())
        assert ch["kind"] == "generic"

    def test_unknown_kind_lossy_field_recorded(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "carrier_pigeon"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert "kind_remapped_to_generic" in t["lossy_fields"]

    def test_webhook_url_stored_in_config(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "telegram", "webhook_url": "https://example.com/hook"}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": [rec]}).json()
        ch = json.loads((storage_root / "channels" / f"{body['channel_ids'][0]}.json").read_text())
        assert ch["config"]["webhook_url"] == "https://example.com/hook"

    def test_webhook_url_lossy_field_recorded(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "telegram", "webhook_url": "https://example.com/hook"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert "webhook_url" in t["lossy_fields"]

    def test_source_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json=_openclaw_body()).json()
        ch = json.loads((storage_root / "channels" / f"{body['channel_ids'][0]}.json").read_text())
        assert ch["metadata"]["openclaw_id"] == "oc_001"

    def test_name_in_metadata(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "telegram", "name": "My Channel"}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": [rec]}).json()
        ch = json.loads((storage_root / "channels" / f"{body['channel_ids'][0]}.json").read_text())
        assert ch["metadata"]["openclaw_name"] == "My Channel"

    def test_metadata_keys_stored_with_prefix(self, storage_root: Path) -> None:
        rec = {"id": "oc_1", "kind": "telegram", "metadata": {"team": "platform", "env": "prod"}}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": [rec]}).json()
        ch = json.loads((storage_root / "channels" / f"{body['channel_ids'][0]}.json").read_text())
        assert ch["metadata"]["openclaw_meta_team"] == "platform"
        assert ch["metadata"]["openclaw_meta_env"] == "prod"

    def test_channel_has_created_at(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json=_openclaw_body()).json()
        ch = json.loads((storage_root / "channels" / f"{body['channel_ids'][0]}.json").read_text())
        assert "created_at" in ch


# ---------------------------------------------------------------------------
# OpenClaw – transactional / error handling
# ---------------------------------------------------------------------------


class TestOpenClawTransactional:
    def test_validation_failure_returns_422(self, storage_root: Path) -> None:
        rec = {"id": "", "kind": "telegram"}  # empty id
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        assert resp.status_code == 422

    def test_validation_failure_error_code(self, storage_root: Path) -> None:
        rec = {"id": "", "kind": "telegram"}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw", json={"records": [rec]}).json()
        assert body["error"]["code"] == "import_record_invalid"

    def test_no_channel_files_written_on_failure(self, storage_root: Path) -> None:
        rec = {"id": "", "kind": "telegram"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        channels_dir = storage_root / "channels"
        if channels_dir.exists():
            assert list(channels_dir.glob("*.json")) == []

    def test_import_failed_entry_in_import_audit_log(self, storage_root: Path) -> None:
        rec = {"id": "", "kind": "telegram"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        assert any(e["type"] == "import_failed" for e in entries)

    def test_import_failed_entry_has_code(self, storage_root: Path) -> None:
        rec = {"id": "", "kind": "telegram"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        failed = next(e for e in entries if e["type"] == "import_failed")
        assert failed["code"] == "import_record_invalid"

    def test_failure_written_to_main_audit_log(self, storage_root: Path) -> None:
        rec = {"id": "", "kind": "telegram"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [rec]})
        records = _main_audit_records(storage_root)
        assert any(r.get("event") == "import.openclaw.failed" for r in records)

    def test_second_invalid_record_rolls_back_first(self, storage_root: Path) -> None:
        records = [
            {"id": "oc_1", "kind": "telegram"},
            {"id": "", "kind": "slack"},  # invalid: empty id
        ]
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": records})
        channels_dir = storage_root / "channels"
        if channels_dir.exists():
            assert list(channels_dir.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Hermes – success
# ---------------------------------------------------------------------------


class TestHermesImportSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes", json=_hermes_body())
        assert resp.status_code == 201

    def test_response_has_expected_fields(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        assert "imported" in body
        assert "lossy_count" in body
        assert "audit_path" in body
        assert "skill_ids" in body

    def test_imported_count_matches_records(self, storage_root: Path) -> None:
        records = [
            {"id": "h_1", "name": "skill-a", "description": "d", "instructions": "i", "tools": [{"name": "t"}]},
            {"id": "h_2", "name": "skill-b", "description": "d", "instructions": "i", "tools": [{"name": "t"}]},
        ]
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json={"records": records}).json()
        assert body["imported"] == 2
        assert len(body["skill_ids"]) == 2

    def test_empty_records_returns_201_with_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json={"records": []}).json()
        assert body["imported"] == 0
        assert body["skill_ids"] == []


# ---------------------------------------------------------------------------
# Hermes – import audit log
# ---------------------------------------------------------------------------


class TestHermesAuditLog:
    def test_audit_file_created(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        files = list(storage_root.glob("meridian-import-*.audit.ndjson"))
        assert len(files) == 1

    def test_audit_path_in_response_matches_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        assert (storage_root / body["audit_path"]).exists()

    def test_first_entry_is_import_started(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        entries = _import_audit_records(storage_root)
        assert entries[0]["type"] == "import_started"
        assert entries[0]["source"] == "hermes"

    def test_record_translated_entries_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        entries = _import_audit_records(storage_root)
        translated = [e for e in entries if e["type"] == "record_translated"]
        assert len(translated) == 1

    def test_record_translated_kind_is_skill(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert t["kind"] == "skill"

    def test_checklist_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        entries = _import_audit_records(storage_root)
        assert any(e["type"] == "checklist" for e in entries)

    def test_checklist_includes_version_tag_item(self, storage_root: Path) -> None:
        rec = {**_hermes_body()["records"][0], "version_tag": "v2.1.0"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "checklist")
        assert any("version_tag" in item.lower() or "version" in item.lower() for item in c["items"])

    def test_checklist_includes_tags_item(self, storage_root: Path) -> None:
        rec = {**_hermes_body()["records"][0], "tags": ["ml", "nlp"]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "checklist")
        assert any("tags" in item.lower() for item in c["items"])

    def test_checklist_includes_is_public_item(self, storage_root: Path) -> None:
        rec = {**_hermes_body()["records"][0], "is_public": True}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "checklist")
        assert any("public" in item.lower() or "visibility" in item.lower() for item in c["items"])

    def test_last_entry_is_import_completed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        entries = _import_audit_records(storage_root)
        assert entries[-1]["type"] == "import_completed"

    def test_import_completed_source_is_hermes(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        entries = _import_audit_records(storage_root)
        c = next(e for e in entries if e["type"] == "import_completed")
        assert c["source"] == "hermes"


# ---------------------------------------------------------------------------
# Hermes – skill record content
# ---------------------------------------------------------------------------


class TestHermesSkillRecords:
    def test_skill_written_to_disk(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        skill_id = body["skill_ids"][0]
        assert (storage_root / "skills" / f"{skill_id}.json").exists()

    def test_version_written_to_disk(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        ver_id = skill["version"]["id"]
        assert (storage_root / "skill_versions" / f"{ver_id}.json").exists()

    def test_skill_id_has_skill_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        assert body["skill_ids"][0].startswith("skill_")

    def test_version_id_has_skillver_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["version"]["id"].startswith("skillver_")

    def test_version_number_is_1(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["version"]["version_number"] == 1

    def test_version_source_type_is_hermes(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["version"]["source_type"] == "hermes"

    def test_version_source_is_imported(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["version"]["source"] == "imported"

    def test_hermes_id_stored_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json=_hermes_body()).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["metadata"]["hermes_id"] == "h_001"

    def test_version_tag_stored_in_metadata_as_lossy(self, storage_root: Path) -> None:
        rec = {**_hermes_body()["records"][0], "version_tag": "v1.2.3"}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json={"records": [rec]}).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["metadata"]["hermes_version_tag"] == "v1.2.3"

    def test_version_tag_lossy_field_recorded(self, storage_root: Path) -> None:
        rec = {**_hermes_body()["records"][0], "version_tag": "v1.2.3"}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        t = next(e for e in entries if e["type"] == "record_translated")
        assert "version_tag" in t["lossy_fields"]

    def test_tags_stored_in_metadata_as_lossy(self, storage_root: Path) -> None:
        rec = {**_hermes_body()["records"][0], "tags": ["ml", "nlp"]}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json={"records": [rec]}).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["metadata"]["hermes_tags"] == ["ml", "nlp"]

    def test_is_public_stored_in_metadata_as_lossy(self, storage_root: Path) -> None:
        rec = {**_hermes_body()["records"][0], "is_public": True}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json={"records": [rec]}).json()
        skill_id = body["skill_ids"][0]
        skill = json.loads((storage_root / "skills" / f"{skill_id}.json").read_text())
        assert skill["metadata"]["hermes_is_public"] is True


# ---------------------------------------------------------------------------
# Hermes – transactional / error handling
# ---------------------------------------------------------------------------


class TestHermesTransactional:
    def test_empty_id_returns_422(self, storage_root: Path) -> None:
        rec = {"id": "", "name": "skill", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes", json={"records": [rec]})
        assert resp.status_code == 422

    def test_empty_name_returns_422(self, storage_root: Path) -> None:
        rec = {"id": "h_1", "name": "", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes", json={"records": [rec]})
        assert resp.status_code == 422

    def test_empty_instructions_returns_422(self, storage_root: Path) -> None:
        rec = {"id": "h_1", "name": "s", "description": "d", "instructions": "", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes", json={"records": [rec]})
        assert resp.status_code == 422

    def test_empty_tools_returns_422(self, storage_root: Path) -> None:
        rec = {"id": "h_1", "name": "s", "description": "d", "instructions": "i", "tools": []}
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes", json={"records": [rec]})
        assert resp.status_code == 422

    def test_validation_failure_error_code(self, storage_root: Path) -> None:
        rec = {"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes", json={"records": [rec]}).json()
        assert body["error"]["code"] == "import_record_invalid"

    def test_no_skill_files_written_on_failure(self, storage_root: Path) -> None:
        rec = {"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        skills_dir = storage_root / "skills"
        if skills_dir.exists():
            assert list(skills_dir.glob("*.json")) == []

    def test_import_failed_entry_in_import_audit_log(self, storage_root: Path) -> None:
        rec = {"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        entries = _import_audit_records(storage_root)
        assert any(e["type"] == "import_failed" for e in entries)

    def test_failure_written_to_main_audit_log(self, storage_root: Path) -> None:
        rec = {"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        records = _main_audit_records(storage_root)
        assert any(r.get("event") == "import.hermes.failed" for r in records)


# ---------------------------------------------------------------------------
# App factory wiring
# ---------------------------------------------------------------------------


class TestAppFactoryWiring:
    def test_openclaw_route_present_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/imports/openclaw" in paths

    def test_hermes_route_present_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/imports/hermes" in paths

    def test_import_routes_absent_without_storage_root(self) -> None:
        from core_errors import NoopAuditLog
        app = create_app(NoopAuditLog())
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/imports/openclaw" not in paths
        assert "/v1/x/imports/hermes" not in paths


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOTelSpans:
    def setup_method(self) -> None:
        _otel_shared.otel_exporter.clear()

    def test_openclaw_span_emitted_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json=_openclaw_body())
        span_names = [s.name for s in _otel_shared.otel_exporter.get_finished_spans()]
        assert any("import.openclaw" in n for n in span_names)

    def test_openclaw_span_emitted_on_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw", json={"records": [{"id": "", "kind": "telegram"}]})
        span_names = [s.name for s in _otel_shared.otel_exporter.get_finished_spans()]
        assert any("import.openclaw" in n for n in span_names)

    def test_hermes_span_emitted_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json=_hermes_body())
        span_names = [s.name for s in _otel_shared.otel_exporter.get_finished_spans()]
        assert any("import.hermes" in n for n in span_names)

    def test_hermes_span_emitted_on_failure(self, storage_root: Path) -> None:
        rec = {"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes", json={"records": [rec]})
        span_names = [s.name for s in _otel_shared.otel_exporter.get_finished_spans()]
        assert any("import.hermes" in n for n in span_names)

    def test_hermes_install_span_emitted_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        span_names = [s.name for s in _otel_shared.otel_exporter.get_finished_spans()]
        assert any("import.hermes_install" in n for n in span_names)

    def test_hermes_install_span_emitted_on_failure(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json={"skills": [{"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}]})
        span_names = [s.name for s in _otel_shared.otel_exporter.get_finished_spans()]
        assert any("import.hermes_install" in n for n in span_names)


# ---------------------------------------------------------------------------
# Hermes install — helpers
# ---------------------------------------------------------------------------


def _hermes_install_body(**overrides: Any) -> dict:
    base: dict = {
        "skills": [
            {
                "id": "h_s001",
                "name": "my-skill",
                "description": "Does something",
                "instructions": "Step 1: do it.",
                "tools": [{"name": "bash", "description": "Run shell"}],
            }
        ],
        "environments": [
            {"id": "h_e001", "name": "dev-env", "backend": "docker", "image": "ubuntu:24.04"}
        ],
        "providers": [
            {"id": "h_p001", "name": "anthropic-main", "kind": "anthropic"}
        ],
        "sessions": [
            {
                "id": "h_sess001",
                "title": "Test session",
                "created_at": "2024-01-01T00:00:00+00:00",
                "events": [{"type": "session.created", "ts": "2024-01-01T00:00:00+00:00"}],
            }
        ],
        "user_profiles": [
            {"id": "h_u001", "username": "alice", "display_name": "Alice"}
        ],
        "cron": [
            {"id": "h_c001", "trigger_type": "interval", "session_id": "sess_abc", "interval": "1h"}
        ],
        "acp_registry": [
            {"id": "h_a001", "peer_id": "peer_xyz", "base_url": "https://peer.example.com"}
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Hermes install — success
# ---------------------------------------------------------------------------


class TestHermesInstallSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        assert resp.status_code == 201

    def test_response_has_audit_path(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert "audit_path" in body

    def test_response_has_all_subsystem_keys(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        for key in ("skills", "environments", "providers", "sessions", "user_profiles", "cron", "acp_registry"):
            assert key in body, f"missing key: {key}"

    def test_skills_imported_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["skills"]["imported"] == 1

    def test_environments_imported_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["environments"]["imported"] == 1

    def test_providers_imported_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["providers"]["imported"] == 1

    def test_sessions_imported_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["sessions"]["imported"] == 1

    def test_user_profiles_imported_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["user_profiles"]["imported"] == 1

    def test_cron_imported_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["cron"]["imported"] == 1

    def test_acp_registry_imported_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["acp_registry"]["imported"] == 1

    def test_empty_subsystems_returns_zeros(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json={}).json()
        assert body["skills"]["imported"] == 0
        assert body["environments"]["imported"] == 0

    def test_creates_import_audit_ndjson(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        assert _import_audit_path(storage_root) is not None

    def test_audit_log_has_import_started(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        entries = _import_audit_records(storage_root)
        assert entries[0]["type"] == "import_started"

    def test_audit_log_has_record_translated_for_each_subsystem_record(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        entries = _import_audit_records(storage_root)
        translated = [e for e in entries if e["type"] == "record_translated"]
        # 1 skill + 1 env + 1 provider + 1 session + 1 user_profile + 1 cron + 1 acp = 7
        assert len(translated) == 7

    def test_audit_log_record_translated_kinds(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        entries = _import_audit_records(storage_root)
        kinds = {e["kind"] for e in entries if e["type"] == "record_translated"}
        assert kinds == {"skill", "environment", "provider", "session", "user_profile", "cron", "acp_peer"}

    def test_audit_log_ends_with_import_completed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        entries = _import_audit_records(storage_root)
        assert entries[-1]["type"] == "import_completed"

    def test_audit_log_has_checklist(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=_hermes_install_body())
        entries = _import_audit_records(storage_root)
        assert any(e["type"] == "checklist" for e in entries)


# ---------------------------------------------------------------------------
# Hermes install — environments
# ---------------------------------------------------------------------------


class TestHermesInstallEnvironments:
    def test_env_written_to_environments_dir(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        env_ids = body["environments"]["ids"]
        assert len(env_ids) == 1
        env_file = storage_root / "environments" / f"{env_ids[0]}.json"
        assert env_file.exists()

    def test_env_id_has_env_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["environments"]["ids"][0].startswith("env_")

    def test_env_record_has_backend(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        env_file = storage_root / "environments" / f"{body['environments']['ids'][0]}.json"
        rec = json.loads(env_file.read_text())
        assert rec["backend"] == "docker"

    def test_env_hermes_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        env_file = storage_root / "environments" / f"{body['environments']['ids'][0]}.json"
        rec = json.loads(env_file.read_text())
        assert rec["metadata"]["hermes_id"] == "h_e001"

    def test_env_unknown_backend_marks_lossy(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["environments"][0]["backend"] = "obscure_vm"
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=install).json()
        assert body["environments"]["lossy_count"] == 1

    def test_env_empty_name_returns_422(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["environments"] = [{"id": "e1", "name": "", "backend": "docker"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Hermes install — providers
# ---------------------------------------------------------------------------


class TestHermesInstallProviders:
    def test_providers_written_to_fragment_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["providers"]["fragment_path"]
        assert fragment.exists()

    def test_provider_fragment_is_valid_json_array(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["providers"]["fragment_path"]
        data = json.loads(fragment.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_provider_auth_is_null(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["providers"]["fragment_path"]
        prov = json.loads(fragment.read_text())[0]
        assert prov["auth"] is None

    def test_provider_always_lossy_due_to_auth(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["providers"]["lossy_count"] == 1

    def test_provider_hermes_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["providers"]["fragment_path"]
        prov = json.loads(fragment.read_text())[0]
        assert prov["metadata"]["hermes_id"] == "h_p001"

    def test_provider_empty_name_returns_422(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["providers"] = [{"id": "p1", "name": "", "kind": "anthropic"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Hermes install — sessions
# ---------------------------------------------------------------------------


class TestHermesInstallSessions:
    def test_session_manifest_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        sess_id = body["sessions"]["ids"][0]
        manifest_path = storage_root / "sessions" / sess_id / "manifest.json"
        assert manifest_path.exists()

    def test_session_id_has_sess_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["sessions"]["ids"][0].startswith("sess_")

    def test_session_status_is_archived(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        sess_id = body["sessions"]["ids"][0]
        manifest = json.loads((storage_root / "sessions" / sess_id / "manifest.json").read_text())
        assert manifest["status"] == "archived"

    def test_session_hermes_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        sess_id = body["sessions"]["ids"][0]
        manifest = json.loads((storage_root / "sessions" / sess_id / "manifest.json").read_text())
        assert manifest["metadata"]["hermes_id"] == "h_sess001"

    def test_session_events_written_as_ndjson(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        sess_id = body["sessions"]["ids"][0]
        elog = storage_root / "sessions" / sess_id / "events.ndjson"
        assert elog.exists()
        lines = [l for l in elog.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_session_thread_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        sess_id = body["sessions"]["ids"][0]
        threads_dir = storage_root / "sessions" / sess_id / "threads"
        assert threads_dir.exists()
        assert len(list(threads_dir.glob("*.json"))) == 1

    def test_session_no_events_file_when_events_empty(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["sessions"] = [
            {"id": "h_sess002", "created_at": "2024-01-02T00:00:00+00:00", "events": []}
        ]
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=install).json()
        sess_id = body["sessions"]["ids"][0]
        elog = storage_root / "sessions" / sess_id / "events.ndjson"
        assert not elog.exists()

    def test_session_empty_created_at_returns_422(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["sessions"] = [{"id": "h_sess003", "created_at": ""}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Hermes install — user_profiles
# ---------------------------------------------------------------------------


class TestHermesInstallUserProfiles:
    def test_profile_written_to_user_profiles_dir(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        uid = body["user_profiles"]["ids"][0]
        assert (storage_root / "user_profiles" / f"{uid}.json").exists()

    def test_profile_id_has_user_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["user_profiles"]["ids"][0].startswith("user_")

    def test_profile_is_not_primary(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        uid = body["user_profiles"]["ids"][0]
        rec = json.loads((storage_root / "user_profiles" / f"{uid}.json").read_text())
        assert rec["is_primary"] is False

    def test_profile_hermes_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        uid = body["user_profiles"]["ids"][0]
        rec = json.loads((storage_root / "user_profiles" / f"{uid}.json").read_text())
        assert rec["metadata"]["hermes_id"] == "h_u001"

    def test_profile_memories_stored_verbatim(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["user_profiles"] = [
            {"id": "h_u002", "username": "bob", "memories": ["Remember to call mom"]}
        ]
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=install).json()
        uid = body["user_profiles"]["ids"][0]
        rec = json.loads((storage_root / "user_profiles" / f"{uid}.json").read_text())
        assert rec["memories"] == ["Remember to call mom"]
        assert body["user_profiles"]["lossy_count"] == 1

    def test_profile_empty_username_returns_422(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["user_profiles"] = [{"id": "h_u003", "username": ""}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Hermes install — cron
# ---------------------------------------------------------------------------


class TestHermesInstallCron:
    def test_cron_written_to_cron_dir(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        cron_id = body["cron"]["ids"][0]
        assert (storage_root / "cron" / f"{cron_id}.json").exists()

    def test_cron_id_has_cron_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        assert body["cron"]["ids"][0].startswith("cron_")

    def test_cron_status_is_active(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        cron_id = body["cron"]["ids"][0]
        rec = json.loads((storage_root / "cron" / f"{cron_id}.json").read_text())
        assert rec["status"] == "active"

    def test_cron_hermes_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        cron_id = body["cron"]["ids"][0]
        rec = json.loads((storage_root / "cron" / f"{cron_id}.json").read_text())
        assert rec["metadata"]["hermes_id"] == "h_c001"

    def test_cron_invalid_trigger_type_returns_422(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["cron"] = [{"id": "h_c002", "trigger_type": "bogus", "session_id": "sess_abc"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422

    def test_cron_empty_session_id_returns_422(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["cron"] = [{"id": "h_c003", "trigger_type": "interval", "session_id": "", "interval": "5m"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422

    def test_cron_timestamp_trigger_sets_next_fire_at(self, storage_root: Path) -> None:
        ts = "2025-01-01T12:00:00+00:00"
        install = _hermes_install_body()
        install["cron"] = [{"id": "h_c004", "trigger_type": "timestamp", "session_id": "sess_abc", "timestamp": ts}]
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=install).json()
        cron_id = body["cron"]["ids"][0]
        rec = json.loads((storage_root / "cron" / f"{cron_id}.json").read_text())
        assert rec["next_fire_at"] == ts


# ---------------------------------------------------------------------------
# Hermes install — ACP registry
# ---------------------------------------------------------------------------


class TestHermesInstallAcpRegistry:
    def test_acp_fragment_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["acp_registry"]["fragment_path"]
        assert fragment.exists()

    def test_acp_fragment_is_valid_json_array(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["acp_registry"]["fragment_path"]
        data = json.loads(fragment.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_acp_peer_id_preserved(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["acp_registry"]["fragment_path"]
        peer = json.loads(fragment.read_text())[0]
        assert peer["peer_id"] == "peer_xyz"

    def test_acp_base_url_preserved(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["acp_registry"]["fragment_path"]
        peer = json.loads(fragment.read_text())[0]
        assert peer["base_url"] == "https://peer.example.com"

    def test_acp_hermes_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=_hermes_install_body()).json()
        fragment = storage_root / body["acp_registry"]["fragment_path"]
        peer = json.loads(fragment.read_text())[0]
        assert peer["metadata"]["hermes_id"] == "h_a001"

    def test_acp_empty_peer_id_returns_422(self, storage_root: Path) -> None:
        install = _hermes_install_body()
        install["acp_registry"] = [{"id": "a1", "peer_id": "", "base_url": "https://x.example.com"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Hermes install — transactional / failure
# ---------------------------------------------------------------------------


class TestHermesInstallFailure:
    def test_validation_failure_returns_422(self, storage_root: Path) -> None:
        install = {"skills": [{"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}]}
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/hermes/install", json=install)
        assert resp.status_code == 422

    def test_validation_failure_error_code(self, storage_root: Path) -> None:
        install = {"skills": [{"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}]}
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/hermes/install", json=install).json()
        assert body["error"]["code"] == "import_record_invalid"

    def test_no_files_written_on_failure(self, storage_root: Path) -> None:
        install = {"skills": [{"id": "", "name": "s", "description": "d", "instructions": "i", "tools": [{"name": "t"}]}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=install)
        if (storage_root / "skills").exists():
            assert list((storage_root / "skills").glob("*.json")) == []

    def test_failure_writes_import_failed_to_audit(self, storage_root: Path) -> None:
        install = {"environments": [{"id": "", "name": "e", "backend": "docker"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=install)
        entries = _import_audit_records(storage_root)
        assert any(e["type"] == "import_failed" for e in entries)

    def test_failure_writes_to_main_audit_log(self, storage_root: Path) -> None:
        install = {"environments": [{"id": "", "name": "e", "backend": "docker"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/hermes/install", json=install)
        records = _main_audit_records(storage_root)
        assert any(r.get("event") == "import.hermes_install.failed" for r in records)

    def test_install_route_present_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/imports/hermes/install" in paths


# ---------------------------------------------------------------------------
# OpenClaw install — helpers
# ---------------------------------------------------------------------------


def _openclaw_install_body(**overrides: Any) -> dict:
    base: dict = {
        "channels": [
            {
                "id": "oc_ch_001",
                "kind": "telegram",
                "name": "Main Channel",
                "config": {"token_vault_ref": "vlt_abc"},
            }
        ],
        "sessions": [
            {
                "id": "oc_sess_001",
                "title": "Session One",
                "created_at": "2026-01-01T00:00:00+00:00",
                "events": [{"type": "message", "role": "user", "content": "hello"}],
            }
        ],
        "memory": [
            {"key": "MEMORY.md", "content": "# Agent Memory\n\nUser prefers dark mode."}
        ],
        "tools": [
            {
                "id": "oc_tool_001",
                "name": "search_web",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "handler_kind": "http",
            }
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# OpenClaw install — success
# ---------------------------------------------------------------------------


class TestOpenClawInstallSuccess:
    def test_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        assert resp.status_code == 201

    def test_response_has_audit_path_and_subsystem_keys(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body()).json()
        assert "audit_path" in body
        assert "channels" in body
        assert "sessions" in body
        assert "memory_stores" in body
        assert "tools" in body

    def test_counts_correct(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body()).json()
        assert body["channels"]["imported"] == 1
        assert body["sessions"]["imported"] == 1
        assert body["memory_stores"]["imported"] == 1
        assert body["tools"]["imported"] == 1

    def test_empty_subsystems_returns_zeros(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        body = client.post(
            "/v1/x/imports/openclaw/install",
            json={"channels": [], "sessions": [], "memory": [], "tools": []},
        ).json()
        assert body["channels"]["imported"] == 0
        assert body["sessions"]["imported"] == 0
        assert body["memory_stores"]["imported"] == 0
        assert body["tools"]["imported"] == 0

    def test_creates_import_audit_file(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        assert _import_audit_path(storage_root) is not None

    def test_audit_starts_with_import_started(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        entries = _import_audit_records(storage_root)
        assert entries[0]["type"] == "import_started"

    def test_audit_has_record_translated_for_each_record(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        entries = _import_audit_records(storage_root)
        translated = [e for e in entries if e["type"] == "record_translated"]
        # 1 channel + 1 session + 1 memory_store + 1 tool = 4
        assert len(translated) == 4

    def test_audit_record_translated_kinds(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        entries = _import_audit_records(storage_root)
        kinds = {e["kind"] for e in entries if e["type"] == "record_translated"}
        assert kinds == {"channel", "session", "memory_store", "tool"}

    def test_audit_ends_with_import_completed(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        entries = _import_audit_records(storage_root)
        assert entries[-1]["type"] == "import_completed"

    def test_audit_has_checklist_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        entries = _import_audit_records(storage_root)
        assert any(e["type"] == "checklist" for e in entries)


# ---------------------------------------------------------------------------
# OpenClaw install — channels
# ---------------------------------------------------------------------------


class TestOpenClawInstallChannels:
    def test_channel_written_to_channels_dir(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        ch_id = resp.json()["channels"]["ids"][0]
        assert (storage_root / "channels" / f"{ch_id}.json").exists()

    def test_channel_id_has_ch_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        assert resp.json()["channels"]["ids"][0].startswith("ch_")


# ---------------------------------------------------------------------------
# OpenClaw install — sessions
# ---------------------------------------------------------------------------


class TestOpenClawInstallSessions:
    def test_session_manifest_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        sess_id = resp.json()["sessions"]["ids"][0]
        assert (storage_root / "sessions" / sess_id / "manifest.json").exists()

    def test_session_id_has_sess_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        assert resp.json()["sessions"]["ids"][0].startswith("sess_")

    def test_session_status_is_archived(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        sess_id = resp.json()["sessions"]["ids"][0]
        manifest = json.loads(
            (storage_root / "sessions" / sess_id / "manifest.json").read_text()
        )
        assert manifest["status"] == "archived"

    def test_session_openclaw_id_in_metadata(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        sess_id = resp.json()["sessions"]["ids"][0]
        manifest = json.loads(
            (storage_root / "sessions" / sess_id / "manifest.json").read_text()
        )
        assert manifest["metadata"]["openclaw_id"] == "oc_sess_001"

    def test_session_events_written_as_ndjson(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        sess_id = resp.json()["sessions"]["ids"][0]
        elog = storage_root / "sessions" / sess_id / "events.ndjson"
        assert elog.exists()
        lines = [json.loads(l) for l in elog.read_text().splitlines() if l.strip()]
        assert lines[0]["type"] == "message"

    def test_session_thread_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        sess_id = resp.json()["sessions"]["ids"][0]
        threads = list((storage_root / "sessions" / sess_id / "threads").glob("*.json"))
        assert len(threads) == 1

    def test_no_events_ndjson_when_events_empty(self, storage_root: Path) -> None:
        body = _openclaw_install_body()
        body["sessions"] = [{"id": "s1", "created_at": "2026-01-01T00:00:00+00:00"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=body)
        sess_id = resp.json()["sessions"]["ids"][0]
        assert not (storage_root / "sessions" / sess_id / "events.ndjson").exists()

    def test_empty_created_at_returns_422(self, storage_root: Path) -> None:
        body = _openclaw_install_body()
        body["sessions"] = [{"id": "s1", "created_at": ""}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=body)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# OpenClaw install — memory stores
# ---------------------------------------------------------------------------


class TestOpenClawInstallMemoryStores:
    def test_memory_store_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        store_id = resp.json()["memory_stores"]["ids"][0]
        assert (storage_root / "memory_stores" / f"{store_id}.json").exists()

    def test_memory_store_id_has_memstore_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        assert resp.json()["memory_stores"]["ids"][0].startswith("memstore_")

    def test_memory_store_scope_is_agent(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        store_id = resp.json()["memory_stores"]["ids"][0]
        record = json.loads(
            (storage_root / "memory_stores" / f"{store_id}.json").read_text()
        )
        assert record["scope"] == "agent"

    def test_memory_store_metadata_tagged_from_openclaw(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        store_id = resp.json()["memory_stores"]["ids"][0]
        record = json.loads(
            (storage_root / "memory_stores" / f"{store_id}.json").read_text()
        )
        assert record["metadata"]["from"] == "openclaw"

    def test_memory_store_raw_content_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        store_id = resp.json()["memory_stores"]["ids"][0]
        raw_dir = storage_root / "memory_stores" / store_id / "raw"
        assert raw_dir.exists()
        raw_files = list(raw_dir.glob("*.md"))
        assert len(raw_files) == 1

    def test_no_memory_store_when_memory_list_empty(self, storage_root: Path) -> None:
        body = _openclaw_install_body()
        body["memory"] = []
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=body)
        assert resp.json()["memory_stores"]["imported"] == 0
        assert not (storage_root / "memory_stores").exists() or \
            list((storage_root / "memory_stores").glob("*.json")) == []


# ---------------------------------------------------------------------------
# OpenClaw install — tools
# ---------------------------------------------------------------------------


class TestOpenClawInstallTools:
    def _tool_record(self, storage_root: Path) -> dict:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        tool_id = resp.json()["tools"]["ids"][0]
        return json.loads((storage_root / "tools" / f"{tool_id}.json").read_text())

    def test_tool_written_to_tools_dir(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        tool_id = resp.json()["tools"]["ids"][0]
        assert (storage_root / "tools" / f"{tool_id}.json").exists()

    def test_tool_id_has_tool_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        assert resp.json()["tools"]["ids"][0].startswith("tool_")

    def test_tool_conservative_caps_applied_when_absent(self, storage_root: Path) -> None:
        body = _openclaw_install_body()
        body["tools"] = [{"id": "t1", "name": "my_tool"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=body)
        tool_id = resp.json()["tools"]["ids"][0]
        record = json.loads((storage_root / "tools" / f"{tool_id}.json").read_text())
        caps = record["capabilities"]
        assert caps["allow_exec"] is False
        assert caps["allow_network"] is False
        assert caps["allow_file_write"] is False
        assert caps["allow_file_read"] is True
        assert caps["sandboxed"] is True

    def test_tool_allow_exec_defaults_false(self, storage_root: Path) -> None:
        tr = self._tool_record(storage_root)
        assert tr["capabilities"]["allow_exec"] is False

    def test_tool_allow_network_defaults_false(self, storage_root: Path) -> None:
        tr = self._tool_record(storage_root)
        assert tr["capabilities"]["allow_network"] is False

    def test_tool_allow_file_write_defaults_false(self, storage_root: Path) -> None:
        tr = self._tool_record(storage_root)
        assert tr["capabilities"]["allow_file_write"] is False

    def test_tool_allow_file_read_defaults_true(self, storage_root: Path) -> None:
        tr = self._tool_record(storage_root)
        assert tr["capabilities"]["allow_file_read"] is True

    def test_tool_sandboxed_defaults_true(self, storage_root: Path) -> None:
        tr = self._tool_record(storage_root)
        assert tr["capabilities"]["sandboxed"] is True

    def test_tool_source_and_source_type(self, storage_root: Path) -> None:
        tr = self._tool_record(storage_root)
        assert tr["source"] == "imported"
        assert tr["source_type"] == "openclaw"

    def test_tool_openclaw_id_in_metadata(self, storage_root: Path) -> None:
        tr = self._tool_record(storage_root)
        assert tr["metadata"]["openclaw_id"] == "oc_tool_001"

    def test_tool_empty_name_returns_422(self, storage_root: Path) -> None:
        body = _openclaw_install_body()
        body["tools"] = [{"id": "t1", "name": ""}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=body)
        assert resp.status_code == 422

    def test_tool_empty_id_returns_422(self, storage_root: Path) -> None:
        body = _openclaw_install_body()
        body["tools"] = [{"id": "", "name": "my_tool"}]
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=body)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# OpenClaw install — transactional / failure
# ---------------------------------------------------------------------------


class TestOpenClawInstallFailure:
    def test_validation_failure_returns_422(self, storage_root: Path) -> None:
        body = {"sessions": [{"id": "", "created_at": "2026-01-01T00:00:00+00:00"}]}
        client = _make_client(storage_root)
        resp = client.post("/v1/x/imports/openclaw/install", json=body)
        assert resp.status_code == 422

    def test_validation_failure_error_code(self, storage_root: Path) -> None:
        body = {"sessions": [{"id": "", "created_at": "2026-01-01T00:00:00+00:00"}]}
        client = _make_client(storage_root)
        resp_body = client.post("/v1/x/imports/openclaw/install", json=body).json()
        assert resp_body["error"]["code"] == "import_record_invalid"

    def test_no_files_written_on_failure(self, storage_root: Path) -> None:
        body = {"sessions": [{"id": "", "created_at": "2026-01-01T00:00:00+00:00"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=body)
        if (storage_root / "sessions").exists():
            assert list((storage_root / "sessions").glob("*/manifest.json")) == []

    def test_failure_writes_import_failed_to_audit(self, storage_root: Path) -> None:
        body = {"tools": [{"id": "", "name": "t"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=body)
        entries = _import_audit_records(storage_root)
        assert any(e["type"] == "import_failed" for e in entries)

    def test_failure_writes_to_main_audit_log(self, storage_root: Path) -> None:
        body = {"tools": [{"id": "", "name": "t"}]}
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=body)
        records = _main_audit_records(storage_root)
        assert any(r.get("event") == "import.openclaw_install.failed" for r in records)

    def test_install_route_present_with_storage_root(self, storage_root: Path) -> None:
        app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
        paths = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/v1/x/imports/openclaw/install" in paths


# ---------------------------------------------------------------------------
# OpenClaw install — OTel
# ---------------------------------------------------------------------------


class TestOpenClawInstallOTel:
    def test_otel_span_emitted_on_success(self, storage_root: Path) -> None:
        _otel_shared.otel_exporter.clear()
        client = _make_client(storage_root)
        client.post("/v1/x/imports/openclaw/install", json=_openclaw_install_body())
        span_names = [s.name for s in _otel_shared.otel_exporter.get_finished_spans()]
        assert "import.openclaw_install" in span_names
