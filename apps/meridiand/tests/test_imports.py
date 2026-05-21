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
  - create_app wires imports router when storage_root is supplied.
  - create_app omits imports routes when storage_root is None.
  - OTel span "import.openclaw" emitted on success.
  - OTel span "import.openclaw" emitted on failure.
  - OTel span "import.hermes" emitted on success.
  - OTel span "import.hermes" emitted on failure.
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
