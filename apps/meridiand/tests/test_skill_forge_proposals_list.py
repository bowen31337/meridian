"""
Skill Forge proposals list endpoint conformance suite.

Tests cover:
  - GET /v1/x/skill_forge/proposals returns 200 on success.
  - Returns empty items list when no proposals exist.
  - Returns only PROPOSAL-status proposals (not REJECTED or PROMOTED).
  - Response has items, next_cursor, limit fields.
  - Items include trajectory provenance fields (run_id, job_id, derived_from_session_ids).
  - Items are sorted newest first (descending created_at).
  - Cursor pagination: next_cursor is null when all items fit on one page.
  - Cursor pagination: next_cursor is set when more items exist.
  - Cursor pagination: Link response header set when next_cursor is present
    (middleware converts X-Next-Cursor).
  - Cursor pagination: cursor query param advances to next page.
  - limit query param restricts items returned.
  - include_efficacy=false (default) does not attach efficacy field to items.
  - include_efficacy=true attaches efficacy record when efficacy file exists.
  - include_efficacy=true attaches efficacy=null when no efficacy file exists.
  - OTel span "skill_forge.proposal.list" emitted on success.
  - OTel span carries skill_forge.include_efficacy attribute.
  - OTel span sets skill_forge.proposal.list.success=True on success.
  - OTel span sets skill_forge.proposal.list.count attribute on success.
  - OTel span "skill_forge.proposal.list" emitted on cursor error.
  - OTel span sets skill_forge.proposal.list.success=False on cursor error.
  - Success writes audit entry with event "skill_forge.proposal.listed".
  - Success audit entry level is "info".
  - Success audit detail has count and include_efficacy.
  - Invalid cursor returns 400 with code "cursor_invalid".
  - Invalid cursor writes audit entry with event "skill_forge.proposal.list.failed".
  - Invalid cursor audit entry level is "error".
  - create_app wires proposals list route when storage_root is supplied.
  - create_app omits proposals list route when storage_root is None.
  - SkillForgeProposalListError has http_status 500.
  - SkillForgeProposalListError has code "skill_forge_proposal_list_failed".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._skill_forge_proposals import SkillForgeProposalListError
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path) -> TestClient:
    app = create_app(FileAuditLog(storage_root), storage_root=storage_root)
    return TestClient(app, raise_server_exceptions=False)


def _write_proposal(
    storage_root: Path,
    proposal_id: str = "skillver_abc123",
    skill_id: str = "skill_xyz",
    status: str = "PROPOSAL",
    created_at: str = "2024-01-01T00:00:00+00:00",
    run_id: str = "sfrun_test",
    job_id: str = "sfjob_test",
    derived_from_session_ids: list[str] | None = None,
) -> dict[str, Any]:
    proposals_dir = storage_root / "skill_forge" / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": proposal_id,
        "skill_id": skill_id,
        "instructions": "Do something.",
        "tools": [],
        "tests": [],
        "source": "forge",
        "source_type": "forge",
        "source_url": None,
        "derived_from_session_ids": derived_from_session_ids,
        "run_id": run_id,
        "job_id": job_id,
        "status": status,
        "created_at": created_at,
    }
    (proposals_dir / f"{proposal_id}.json").write_text(json.dumps(record))
    return record


def _write_efficacy(
    storage_root: Path,
    proposal_id: str,
    lift: float = 0.4,
) -> dict[str, Any]:
    efficacy_dir = storage_root / "skill_forge" / "efficacy"
    efficacy_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "id": f"skefficacy_{proposal_id}",
        "proposal_id": proposal_id,
        "skill_id": "skill_xyz",
        "test_case_count": 2,
        "pass_rate_without_skill": 0.0,
        "pass_rate_with_skill": lift,
        "lift": lift,
        "case_results": [],
        "created_at": "2024-01-01T01:00:00+00:00",
    }
    (efficacy_dir / f"{proposal_id}_efficacy.json").write_text(json.dumps(record))
    return record


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _list(
    client: TestClient,
    cursor: str | None = None,
    limit: int | None = None,
    include_efficacy: bool | None = None,
) -> Any:
    params: dict[str, Any] = {}
    if cursor is not None:
        params["cursor"] = cursor
    if limit is not None:
        params["limit"] = limit
    if include_efficacy is not None:
        params["include_efficacy"] = include_efficacy
    return client.get("/v1/x/skill_forge/proposals", params=params)


# ---------------------------------------------------------------------------
# Success: HTTP and response shape
# ---------------------------------------------------------------------------


class TestListSuccess:
    def test_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.status_code == 200

    def test_empty_items_when_no_proposals(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.json()["items"] == []

    def test_response_has_items_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client)
        assert "items" in resp.json()

    def test_response_has_next_cursor_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client)
        assert "next_cursor" in resp.json()

    def test_response_has_limit_field(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client)
        assert "limit" in resp.json()

    def test_returns_proposal_status_proposals(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_a", status="PROPOSAL")
        client = _make_client(storage_root)
        resp = _list(client)
        ids = [item["id"] for item in resp.json()["items"]]
        assert "skillver_a" in ids

    def test_excludes_rejected_proposals(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_r", status="REJECTED")
        client = _make_client(storage_root)
        resp = _list(client)
        ids = [item["id"] for item in resp.json()["items"]]
        assert "skillver_r" not in ids

    def test_excludes_promoted_proposals(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_p", status="PROMOTED")
        client = _make_client(storage_root)
        resp = _list(client)
        ids = [item["id"] for item in resp.json()["items"]]
        assert "skillver_p" not in ids

    def test_only_proposal_status_returned_among_mixed(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_q", status="PROPOSAL")
        _write_proposal(storage_root, proposal_id="skillver_r", status="REJECTED")
        _write_proposal(storage_root, proposal_id="skillver_p", status="PROMOTED")
        client = _make_client(storage_root)
        resp = _list(client)
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["id"] == "skillver_q"


# ---------------------------------------------------------------------------
# Trajectory provenance fields
# ---------------------------------------------------------------------------


class TestTrajectoryProvenance:
    def test_item_has_run_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, run_id="sfrun_prov")
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.json()["items"][0]["run_id"] == "sfrun_prov"

    def test_item_has_job_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, job_id="sfjob_prov")
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.json()["items"][0]["job_id"] == "sfjob_prov"

    def test_item_has_derived_from_session_ids_null_when_none(self, storage_root: Path) -> None:
        _write_proposal(storage_root, derived_from_session_ids=None)
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.json()["items"][0]["derived_from_session_ids"] is None

    def test_item_has_derived_from_session_ids_when_set(self, storage_root: Path) -> None:
        _write_proposal(storage_root, derived_from_session_ids=["sess_1", "sess_2"])
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.json()["items"][0]["derived_from_session_ids"] == [
            "sess_1",
            "sess_2",
        ]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_items_ordered_newest_first(self, storage_root: Path) -> None:
        _write_proposal(
            storage_root, proposal_id="skillver_old", created_at="2024-01-01T00:00:00+00:00"
        )
        _write_proposal(
            storage_root, proposal_id="skillver_new", created_at="2024-06-01T00:00:00+00:00"
        )
        client = _make_client(storage_root)
        resp = _list(client)
        ids = [item["id"] for item in resp.json()["items"]]
        assert ids[0] == "skillver_new"
        assert ids[1] == "skillver_old"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    def test_next_cursor_null_when_all_fit(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_a")
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.json()["next_cursor"] is None

    def test_no_link_header_when_all_fit(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_a")
        client = _make_client(storage_root)
        resp = _list(client)
        assert "link" not in {k.lower() for k in resp.headers}

    def test_next_cursor_set_when_more_items_exist(self, storage_root: Path) -> None:
        _write_proposal(
            storage_root, proposal_id="skillver_a", created_at="2024-01-02T00:00:00+00:00"
        )
        _write_proposal(
            storage_root, proposal_id="skillver_b", created_at="2024-01-01T00:00:00+00:00"
        )
        client = _make_client(storage_root)
        resp = _list(client, limit=1)
        assert resp.json()["next_cursor"] is not None

    def test_link_header_set_when_more_items_exist(self, storage_root: Path) -> None:
        _write_proposal(
            storage_root, proposal_id="skillver_a", created_at="2024-01-02T00:00:00+00:00"
        )
        _write_proposal(
            storage_root, proposal_id="skillver_b", created_at="2024-01-01T00:00:00+00:00"
        )
        client = _make_client(storage_root)
        resp = _list(client, limit=1)
        assert "link" in {k.lower() for k in resp.headers}

    def test_link_header_contains_rel_next(self, storage_root: Path) -> None:
        _write_proposal(
            storage_root, proposal_id="skillver_a", created_at="2024-01-02T00:00:00+00:00"
        )
        _write_proposal(
            storage_root, proposal_id="skillver_b", created_at="2024-01-01T00:00:00+00:00"
        )
        client = _make_client(storage_root)
        resp = _list(client, limit=1)
        assert 'rel="next"' in resp.headers.get("link", "")

    def test_cursor_advances_to_next_page(self, storage_root: Path) -> None:
        _write_proposal(
            storage_root, proposal_id="skillver_a", created_at="2024-01-02T00:00:00+00:00"
        )
        _write_proposal(
            storage_root, proposal_id="skillver_b", created_at="2024-01-01T00:00:00+00:00"
        )
        client = _make_client(storage_root)
        page1 = _list(client, limit=1)
        cursor = page1.json()["next_cursor"]
        page2 = _list(client, cursor=cursor, limit=1)
        ids = [item["id"] for item in page2.json()["items"]]
        assert "skillver_b" in ids
        assert "skillver_a" not in ids

    def test_second_page_next_cursor_null_when_exhausted(self, storage_root: Path) -> None:
        _write_proposal(
            storage_root, proposal_id="skillver_a", created_at="2024-01-02T00:00:00+00:00"
        )
        _write_proposal(
            storage_root, proposal_id="skillver_b", created_at="2024-01-01T00:00:00+00:00"
        )
        client = _make_client(storage_root)
        page1 = _list(client, limit=1)
        cursor = page1.json()["next_cursor"]
        page2 = _list(client, cursor=cursor, limit=1)
        assert page2.json()["next_cursor"] is None

    def test_limit_restricts_items_returned(self, storage_root: Path) -> None:
        for i in range(5):
            _write_proposal(
                storage_root,
                proposal_id=f"skillver_{i:03d}",
                created_at=f"2024-01-0{i + 1}T00:00:00+00:00",
            )
        client = _make_client(storage_root)
        resp = _list(client, limit=2)
        assert len(resp.json()["items"]) == 2

    def test_limit_reflected_in_response(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client, limit=7)
        assert resp.json()["limit"] == 7


# ---------------------------------------------------------------------------
# Efficacy (A/B comparison)
# ---------------------------------------------------------------------------


class TestEfficacy:
    def test_no_efficacy_field_by_default(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _list(client)
        assert "efficacy" not in resp.json()["items"][0]

    def test_include_efficacy_false_no_efficacy_field(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        resp = _list(client, include_efficacy=False)
        assert "efficacy" not in resp.json()["items"][0]

    def test_include_efficacy_true_attaches_efficacy_record(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_abc123")
        _write_efficacy(storage_root, proposal_id="skillver_abc123", lift=0.3)
        client = _make_client(storage_root)
        resp = _list(client, include_efficacy=True)
        item = resp.json()["items"][0]
        assert "efficacy" in item
        assert item["efficacy"] is not None

    def test_include_efficacy_true_efficacy_has_lift(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_abc123")
        _write_efficacy(storage_root, proposal_id="skillver_abc123", lift=0.3)
        client = _make_client(storage_root)
        resp = _list(client, include_efficacy=True)
        item = resp.json()["items"][0]
        assert item["efficacy"]["lift"] == pytest.approx(0.3)

    def test_include_efficacy_true_null_when_no_efficacy_file(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_noefficacy")
        client = _make_client(storage_root)
        resp = _list(client, include_efficacy=True)
        item = resp.json()["items"][0]
        assert "efficacy" in item
        assert item["efficacy"] is None

    def test_include_efficacy_true_efficacy_has_proposal_id(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_abc123")
        _write_efficacy(storage_root, proposal_id="skillver_abc123")
        client = _make_client(storage_root)
        resp = _list(client, include_efficacy=True)
        item = resp.json()["items"][0]
        assert item["efficacy"]["proposal_id"] == "skillver_abc123"

    def test_include_efficacy_true_span_has_include_efficacy_true(self, storage_root: Path) -> None:
        _write_proposal(storage_root)
        client = _make_client(storage_root)
        _otel_exporter.clear()
        _list(client, include_efficacy=True)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("skill_forge.proposal.list")
        assert span is not None
        assert span.attributes["skill_forge.include_efficacy"] is True


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.proposal.list")

    def test_emits_list_span_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.proposal.list" in span_names

    def test_span_has_include_efficacy_attribute(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client, include_efficacy=False)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.include_efficacy"] is False

    def test_span_success_attribute_true_on_success(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.list.success"] is True

    def test_span_has_count_attribute_on_success(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_a")
        client = _make_client(storage_root)
        _list(client)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.list.count"] == 1

    def test_emits_list_span_on_cursor_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client, cursor="not-valid-base64!!!")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.proposal.list" in span_names

    def test_span_success_attribute_false_on_cursor_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client, cursor="not-valid-base64!!!")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.proposal.list.success"] is False


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_success_writes_proposal_listed_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.listed" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.listed"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_count(self, storage_root: Path) -> None:
        _write_proposal(storage_root, proposal_id="skillver_a")
        client = _make_client(storage_root)
        _list(client)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.listed"
        )
        assert record["detail"]["count"] == 1

    def test_success_audit_detail_has_include_efficacy(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client, include_efficacy=False)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.listed"
        )
        assert record["detail"]["include_efficacy"] is False

    def test_cursor_error_writes_list_failed_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client, cursor="not-valid-base64!!!")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.proposal.list.failed" for r in records)

    def test_cursor_error_audit_level_is_error(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        _list(client, cursor="not-valid-base64!!!")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.proposal.list.failed"
        )
        assert record["level"] == "error"


# ---------------------------------------------------------------------------
# Invalid cursor
# ---------------------------------------------------------------------------


class TestInvalidCursor:
    def test_invalid_cursor_returns_400(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client, cursor="not-valid-base64!!!")
        assert resp.status_code == 400

    def test_invalid_cursor_error_code_is_cursor_invalid(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client, cursor="not-valid-base64!!!")
        assert resp.json()["error"]["code"] == "cursor_invalid"


# ---------------------------------------------------------------------------
# Router wiring
# ---------------------------------------------------------------------------


class TestRouterWiring:
    def test_route_present_with_storage_root(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        resp = _list(client)
        assert resp.status_code == 200

    def test_route_absent_without_storage_root(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(FileAuditLog(Path(tmp)), storage_root=None)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/v1/x/skill_forge/proposals")
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Error class contracts
# ---------------------------------------------------------------------------


class TestErrorClasses:
    def test_list_error_http_status(self) -> None:
        err = SkillForgeProposalListError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.http_status() == 500

    def test_list_error_code(self) -> None:
        err = SkillForgeProposalListError(message="boom", timestamp="2024-01-01T00:00:00+00:00")
        assert err.code == "skill_forge_proposal_list_failed"
