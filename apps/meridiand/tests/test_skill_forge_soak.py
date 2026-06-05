"""
Skill Forge soak test conformance suite.

Tests cover:
  - POST /v1/x/skill-forge/soak-run returns 200 when no fixtures directory exists.
  - Returns precision=0.0 and fixture_count=0 when no fixtures directory.
  - Returns 200 (vacuous pass) when fixtures directory is empty.
  - Malformed JSON fixture files are silently skipped.
  - A fixture whose forge result exactly matches expected_proposal is a "hit".
  - A fixture whose forge result does not match expected_proposal is a "miss".
  - precision = hit_count / fixture_count.
  - Returns 200 when precision equals the 50% threshold exactly.
  - Returns 422 with code "skill_forge_soak_failed" when precision < 50%.
  - Error message mentions the precision percentage.
  - Error message mentions the threshold.
  - Error message mentions hit_count / fixture_count.
  - On failure: audit log entry "skill_forge.soak.run.failed" written.
  - On failure: audit entry level is "error".
  - On failure: audit detail has run_id, precision, hit_count, fixture_count.
  - On success: audit log entry "skill_forge.soak.ran" written.
  - On success: audit entry level is "info".
  - On success: audit detail has run_id, precision, hit_count, fixture_count.
  - OTel span "skill_forge.soak.run" emitted on success.
  - OTel span "skill_forge.soak.run" emitted on failure.
  - OTel span set to ERROR status on failure.
  - Span carries skill_forge.soak.fixture_count, hit_count, precision attributes.
  - Response body has run_id, status, precision, hit_count, fixture_count, fixtures.
  - fixtures list has fixture_id, status, result, expected_proposal for each fixture.
  - Provider errors on individual fixtures are recorded as "error" in fixtures list.
  - create_app wires the soak router when storage_root is supplied.
  - create_app omits the soak route when storage_root is None.
  - _proposal_matches returns True for identical strings.
  - _proposal_matches returns False for differing strings.
  - SkillForgeSoakError has http_status 422.
  - PRECISION_THRESHOLD is 0.50.
  - run_id has "soak_" prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridiand._skill_forge_soak import (
    PRECISION_THRESHOLD,
    SkillForgeSoakError,
    _proposal_matches,
    make_skill_forge_soak_router,
)
import pytest

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedProvider:
    """Always returns the same result string."""

    def __init__(self, result: str) -> None:
        self._result = result

    async def forge(self, skill: dict[str, Any], job_type: str) -> str:
        return self._result


class _MappingProvider:
    """Returns different results based on job_type."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    async def forge(self, skill: dict[str, Any], job_type: str) -> str:
        return self._mapping.get(job_type, "")


class _ErrorProvider:
    """Always raises RuntimeError."""

    async def forge(self, skill: dict[str, Any], job_type: str) -> str:
        raise RuntimeError("provider boom")


def _write_fixture(
    fixtures_dir: Path,
    name: str,
    *,
    fixture_id: str | None = None,
    skill: dict | None = None,
    job_type: str = "validate_tests",
    expected_proposal: str = "expected",
) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "job_type": job_type,
        "expected_proposal": expected_proposal,
        "skill": skill or {"name": "test-skill"},
    }
    if fixture_id is not None:
        record["id"] = fixture_id
    (fixtures_dir / name).write_text(json.dumps(record))


def _fixtures_dir(storage_root: Path) -> Path:
    return storage_root / "skill_forge" / "soak_fixtures"


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_client(
    storage_root: Path,
    provider=None,
) -> TestClient:
    from core_errors import HandlerOptions, install_error_handler
    from fastapi import FastAPI

    audit = FileAuditLog(storage_root)
    router = make_skill_forge_soak_router(
        audit_log=audit,
        storage_root=storage_root,
        provider=provider,
    )
    app = FastAPI()
    app.include_router(router)
    install_error_handler(app, HandlerOptions(audit_log=audit))
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Unit: _proposal_matches
# ---------------------------------------------------------------------------


class TestProposalMatches:
    def test_identical_strings_return_true(self) -> None:
        assert _proposal_matches("hello", "hello") is True

    def test_empty_strings_return_true(self) -> None:
        assert _proposal_matches("", "") is True

    def test_differing_strings_return_false(self) -> None:
        assert _proposal_matches("foo", "bar") is False

    def test_partial_match_returns_false(self) -> None:
        assert _proposal_matches("foo bar", "foo") is False

    def test_case_sensitive(self) -> None:
        assert _proposal_matches("Hello", "hello") is False


# ---------------------------------------------------------------------------
# Unit: SkillForgeSoakError
# ---------------------------------------------------------------------------


class TestSkillForgeSoakError:
    def test_http_status_is_422(self) -> None:
        err = SkillForgeSoakError(message="bad", timestamp="2024-01-01T00:00:00+00:00")
        assert err.http_status() == 422

    def test_code_is_skill_forge_soak_failed(self) -> None:
        err = SkillForgeSoakError(message="bad", timestamp="2024-01-01T00:00:00+00:00")
        assert err.code == "skill_forge_soak_failed"


# ---------------------------------------------------------------------------
# Unit: PRECISION_THRESHOLD
# ---------------------------------------------------------------------------


class TestPrecisionThreshold:
    def test_threshold_is_0_50(self) -> None:
        assert PRECISION_THRESHOLD == 0.50


# ---------------------------------------------------------------------------
# Endpoint: no fixtures
# ---------------------------------------------------------------------------


class TestSoakRunNoFixtures:
    def test_no_fixtures_dir_returns_200(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        resp = client.post("/v1/x/skill-forge/soak-run")
        assert resp.status_code == 200

    def test_no_fixtures_dir_precision_is_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["precision"] == 0.0

    def test_no_fixtures_dir_fixture_count_is_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["fixture_count"] == 0

    def test_no_fixtures_dir_hit_count_is_zero(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["hit_count"] == 0

    def test_empty_fixtures_dir_returns_200(self, storage_root: Path) -> None:
        _fixtures_dir(storage_root).mkdir(parents=True, exist_ok=True)
        client = _make_client(storage_root, _FixedProvider("x"))
        resp = client.post("/v1/x/skill-forge/soak-run")
        assert resp.status_code == 200

    def test_run_id_has_soak_prefix(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["run_id"].startswith("soak_")


# ---------------------------------------------------------------------------
# Endpoint: fixture matching
# ---------------------------------------------------------------------------


class TestSoakRunFixtureMatching:
    def test_matching_fixture_is_hit(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", fixture_id="f1", expected_proposal="match")
        client = _make_client(storage_root, _FixedProvider("match"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["hit_count"] == 1
        assert body["fixtures"][0]["status"] == "hit"

    def test_non_matching_fixture_is_miss(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        # f1 misses, f2 hits → precision=50% → 200 with fixtures list
        _write_fixture(
            fd, "f1.json", fixture_id="f1", job_type="miss_type", expected_proposal="expected"
        )
        _write_fixture(fd, "f2.json", fixture_id="f2", job_type="hit_type", expected_proposal="hit")
        provider = _MappingProvider({"miss_type": "wrong", "hit_type": "hit"})
        client = _make_client(storage_root, provider)
        body = client.post("/v1/x/skill-forge/soak-run").json()
        miss = next(f for f in body["fixtures"] if f["fixture_id"] == "f1")
        assert miss["status"] == "miss"

    def test_fixture_result_in_response(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        # f1 misses with result "actual", f2 hits to keep precision at 50%
        _write_fixture(fd, "f1.json", fixture_id="f1", job_type="a", expected_proposal="expected")
        _write_fixture(fd, "f2.json", fixture_id="f2", job_type="b", expected_proposal="hit")
        provider = _MappingProvider({"a": "actual", "b": "hit"})
        client = _make_client(storage_root, provider)
        body = client.post("/v1/x/skill-forge/soak-run").json()
        f1 = next(f for f in body["fixtures"] if f["fixture_id"] == "f1")
        assert f1["result"] == "actual"

    def test_fixture_expected_proposal_in_response(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        # f1 misses, f2 hits to keep precision at 50%
        _write_fixture(
            fd, "f1.json", fixture_id="f1", job_type="a", expected_proposal="my-expected"
        )
        _write_fixture(fd, "f2.json", fixture_id="f2", job_type="b", expected_proposal="hit")
        provider = _MappingProvider({"a": "actual", "b": "hit"})
        client = _make_client(storage_root, provider)
        body = client.post("/v1/x/skill-forge/soak-run").json()
        f1 = next(f for f in body["fixtures"] if f["fixture_id"] == "f1")
        assert f1["expected_proposal"] == "my-expected"

    def test_fixture_id_from_id_field(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", fixture_id="my-fixture-id", expected_proposal="x")
        client = _make_client(storage_root, _FixedProvider("x"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["fixtures"][0]["fixture_id"] == "my-fixture-id"

    def test_fixture_id_falls_back_to_stem_when_no_id_field(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "fixture_007.json", expected_proposal="x")  # no fixture_id
        client = _make_client(storage_root, _FixedProvider("x"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["fixtures"][0]["fixture_id"] == "fixture_007"

    def test_malformed_json_fixture_is_skipped(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        fd.mkdir(parents=True, exist_ok=True)
        (fd / "bad.json").write_text("not valid json{{{")
        client = _make_client(storage_root, _FixedProvider("x"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["fixture_count"] == 0

    def test_malformed_json_does_not_stop_other_fixtures(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        fd.mkdir(parents=True, exist_ok=True)
        (fd / "aaa_bad.json").write_text("{invalid")
        _write_fixture(fd, "zzz_good.json", fixture_id="good", expected_proposal="ok")
        client = _make_client(storage_root, _FixedProvider("ok"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["fixture_count"] == 1
        assert body["hit_count"] == 1

    def test_provider_error_recorded_as_error_status(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        # f1 errors, f2 succeeds → fixture_count=2, hit_count=1, precision=50% → 200
        _write_fixture(fd, "f1.json", fixture_id="f1", job_type="bad", expected_proposal="x")
        _write_fixture(fd, "f2.json", fixture_id="f2", job_type="good", expected_proposal="hit")

        class _PartialErrorProvider:
            async def forge(self, skill: dict, job_type: str) -> str:
                if job_type == "bad":
                    raise RuntimeError("boom")
                return "hit"

        client = _make_client(storage_root, _PartialErrorProvider())
        body = client.post("/v1/x/skill-forge/soak-run").json()
        error_fixture = next(f for f in body["fixtures"] if f["fixture_id"] == "f1")
        assert error_fixture["status"] == "error"
        assert "error" in error_fixture

    def test_provider_error_fixture_not_counted_as_hit(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        # f1 errors, f2 succeeds → only f2 is a hit
        _write_fixture(fd, "f1.json", fixture_id="f1", job_type="bad", expected_proposal="x")
        _write_fixture(fd, "f2.json", fixture_id="f2", job_type="good", expected_proposal="hit")

        class _PartialErrorProvider:
            async def forge(self, skill: dict, job_type: str) -> str:
                if job_type == "bad":
                    raise RuntimeError("boom")
                return "hit"

        client = _make_client(storage_root, _PartialErrorProvider())
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["hit_count"] == 1


# ---------------------------------------------------------------------------
# Endpoint: precision calculation and threshold
# ---------------------------------------------------------------------------


class TestSoakRunPrecision:
    def test_precision_equals_hit_over_total(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", job_type="a", expected_proposal="hit")
        _write_fixture(fd, "f2.json", job_type="b", expected_proposal="miss_expected")
        provider = _MappingProvider({"a": "hit", "b": "wrong"})
        client = _make_client(storage_root, provider)
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["precision"] == pytest.approx(0.5)
        assert body["hit_count"] == 1
        assert body["fixture_count"] == 2

    def test_exactly_50_percent_returns_200(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", job_type="a", expected_proposal="hit")
        _write_fixture(fd, "f2.json", job_type="b", expected_proposal="nope")
        provider = _MappingProvider({"a": "hit", "b": "wrong"})
        client = _make_client(storage_root, provider)
        resp = client.post("/v1/x/skill-forge/soak-run")
        assert resp.status_code == 200

    def test_below_50_percent_returns_422(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        for i in range(3):
            _write_fixture(fd, f"f{i}.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        resp = client.post("/v1/x/skill-forge/soak-run")
        assert resp.status_code == 422

    def test_below_threshold_error_code(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["error"]["code"] == "skill_forge_soak_failed"

    def test_error_message_mentions_precision(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert "0.00%" in body["error"]["message"] or "0%" in body["error"]["message"]

    def test_error_message_mentions_threshold(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert "50%" in body["error"]["message"]

    def test_error_message_mentions_hit_slash_fixture_count(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert "0/1" in body["error"]["message"]

    def test_all_hits_returns_precision_1(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        for i in range(4):
            _write_fixture(fd, f"f{i}.json", expected_proposal="perfect")
        client = _make_client(storage_root, _FixedProvider("perfect"))
        body = client.post("/v1/x/skill-forge/soak-run").json()
        assert body["precision"] == pytest.approx(1.0)
        assert body["status"] == "passed"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestSoakRunAudit:
    def test_success_writes_ran_audit_entry(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.soak.ran" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.soak.ran"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_run_id(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.soak.ran"
        )
        assert "run_id" in record["detail"] and record["detail"]["run_id"]

    def test_success_audit_detail_has_precision(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.soak.ran"
        )
        assert "precision" in record["detail"]

    def test_success_audit_detail_has_hit_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.soak.ran"
        )
        assert "hit_count" in record["detail"]

    def test_success_audit_detail_has_fixture_count(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "skill_forge.soak.ran"
        )
        assert "fixture_count" in record["detail"]

    def test_failure_writes_failed_audit_entry(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.soak.run.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.soak.run.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_run_id(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.soak.run.failed"
        )
        assert "run_id" in record["detail"] and record["detail"]["run_id"]

    def test_failure_audit_detail_has_precision(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.soak.run.failed"
        )
        assert "precision" in record["detail"]

    def test_failure_audit_detail_has_hit_count(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.soak.run.failed"
        )
        assert "hit_count" in record["detail"]

    def test_failure_audit_detail_has_fixture_count(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.soak.run.failed"
        )
        assert "fixture_count" in record["detail"]


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestSoakRunOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.soak.run")

    def test_success_emits_soak_run_span(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.soak.run" in span_names

    def test_failure_emits_soak_run_span(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.soak.run" in span_names

    def test_failure_span_has_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="expected")
        client = _make_client(storage_root, _FixedProvider("wrong"))
        client.post("/v1/x/skill-forge/soak-run")
        span = self._get_span()
        assert span is not None
        assert span.status.status_code == StatusCode.ERROR

    def test_success_span_has_non_error_status(self, storage_root: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_client(storage_root, _FixedProvider("x"))
        client.post("/v1/x/skill-forge/soak-run")
        span = self._get_span()
        assert span is not None
        assert span.status.status_code != StatusCode.ERROR

    def test_span_has_fixture_count_attribute(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="ok")
        _write_fixture(fd, "f2.json", expected_proposal="ok")
        client = _make_client(storage_root, _FixedProvider("ok"))
        client.post("/v1/x/skill-forge/soak-run")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.soak.fixture_count"] == 2

    def test_span_has_hit_count_attribute(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="ok")
        client = _make_client(storage_root, _FixedProvider("ok"))
        client.post("/v1/x/skill-forge/soak-run")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.soak.hit_count"] == 1

    def test_span_has_precision_attribute(self, storage_root: Path) -> None:
        fd = _fixtures_dir(storage_root)
        _write_fixture(fd, "f1.json", expected_proposal="ok")
        client = _make_client(storage_root, _FixedProvider("ok"))
        client.post("/v1/x/skill-forge/soak-run")
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.soak.precision"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# create_app integration
# ---------------------------------------------------------------------------


class TestCreateAppIntegration:
    def test_soak_route_present_when_storage_root_set(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit, storage_root=storage_root)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/skill-forge/soak-run")
        assert resp.status_code != 404

    def test_soak_route_absent_when_no_storage_root(self, storage_root: Path) -> None:
        audit = FileAuditLog(storage_root)
        app = create_app(audit)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/x/skill-forge/soak-run")
        assert resp.status_code == 404
