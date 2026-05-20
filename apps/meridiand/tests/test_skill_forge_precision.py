"""
Skill Forge precision metric conformance suite.

Tests cover:
  - compute_precision_metric stores metric record in precision_dir.
  - Metric record id has "skprecision_" prefix.
  - Metric record has total_proposals equal to count of proposal files.
  - Metric record has approved_count equal to count of approved proposals.
  - Metric record has precision equal to approved_count / total_proposals.
  - Metric record has computed_at timestamp.
  - compute_precision_metric returns the metric record dict.
  - When no proposals exist, precision is 0.0 and total_proposals is 0.
  - When all proposals are approved, precision is 1.0.
  - When half of proposals are approved, precision is 0.5.
  - When proposals exist but no activations exist, approved_count is 0.
  - A proposal is approved when its id matches an activation skill_version_id with status "active".
  - Pending activations (status "pending") do not count as approved.
  - Revoked activations (status "revoked") do not count as approved.
  - Each proposal counted at most once even if multiple active activations reference it.
  - Missing proposals_dir and activations_dir → total_proposals=0, approved_count=0.
  - Writes audit entry "skill_forge.precision.computed" on success.
  - Audit entry level is "info" on success.
  - Audit detail contains metric_id, total_proposals, approved_count, precision.
  - On failure, raises SkillForgePrecisionError.
  - SkillForgePrecisionError has http_status 500.
  - SkillForgePrecisionError code is "skill_forge_precision_failed".
  - On failure, writes audit entry "skill_forge.precision.compute.failed".
  - Failed audit entry level is "error".
  - Failed audit detail contains metric_id and message.
  - No metric record written on failure.
  - OTel span "skill_forge.precision_metric" emitted on success.
  - OTel span carries metric_id, total_proposals, approved_count, precision attributes.
  - OTel span sets skill_forge.precision_metric.success=True on success.
  - OTel span "skill_forge.precision_metric" emitted on failure.
  - OTel span sets skill_forge.precision_metric.success=False on failure.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from meridiand._audit import FileAuditLog
from meridiand._skill_forge_precision import (
    SkillForgePrecisionError,
    compute_precision_metric,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_proposal(proposals_dir: Path, proposal_id: str) -> None:
    proposals_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": proposal_id,
        "skill_id": "skill_test",
        "status": "PROPOSAL",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    (proposals_dir / f"{proposal_id}.json").write_text(json.dumps(record))


def _write_activation(
    activations_dir: Path,
    *,
    skill_version_id: str,
    status: str = "active",
) -> None:
    activations_dir.mkdir(parents=True, exist_ok=True)
    act_id = f"skillact_{uuid.uuid4().hex}"
    record = {
        "id": act_id,
        "agent_id": "agent_test",
        "skill_id": "skill_test",
        "skill_version_id": skill_version_id,
        "status": status,
        "requested_at": "2026-01-01T00:00:00+00:00",
        "approved_at": "2026-01-01T01:00:00+00:00" if status == "active" else None,
        "revoked_at": None,
    }
    (activations_dir / f"{act_id}.json").write_text(json.dumps(record))


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _precision_records(precision_dir: Path) -> list[dict[str, Any]]:
    if not precision_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in precision_dir.glob("*_precision.json")]


def _call(
    storage_root: Path,
    *,
    proposals_dir: Path | None = None,
    activations_dir: Path | None = None,
    precision_dir: Path | None = None,
) -> dict[str, Any]:
    audit_log = FileAuditLog(storage_root)
    return compute_precision_metric(
        proposals_dir=proposals_dir
        if proposals_dir is not None
        else storage_root / "skill_forge" / "proposals",
        activations_dir=activations_dir
        if activations_dir is not None
        else storage_root / "skill_activations",
        precision_dir=precision_dir
        if precision_dir is not None
        else storage_root / "skill_forge" / "precision",
        audit_log=audit_log,
    )


# ---------------------------------------------------------------------------
# Metric record: storage and shape
# ---------------------------------------------------------------------------


class TestPrecisionRecordStorage:
    def test_stores_metric_record_in_precision_dir(self, storage_root: Path) -> None:
        prec_dir = storage_root / "precision"
        _call(storage_root, precision_dir=prec_dir)
        assert len(_precision_records(prec_dir)) == 1

    def test_metric_record_id_has_skprecision_prefix(
        self, storage_root: Path
    ) -> None:
        result = _call(storage_root)
        assert result["id"].startswith("skprecision_")

    def test_metric_record_has_total_proposals(self, storage_root: Path) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        _write_proposal(props_dir, "skillver_aaa")
        _write_proposal(props_dir, "skillver_bbb")
        result = _call(storage_root)
        assert result["total_proposals"] == 2

    def test_metric_record_has_approved_count(self, storage_root: Path) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_aaa")
        _write_activation(acts_dir, skill_version_id="skillver_aaa")
        result = _call(storage_root)
        assert result["approved_count"] == 1

    def test_metric_record_has_precision(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert "precision" in result

    def test_metric_record_has_computed_at(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert result.get("computed_at")

    def test_compute_returns_metric_record_dict(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert isinstance(result, dict)
        assert "id" in result and "precision" in result


# ---------------------------------------------------------------------------
# Metric computation correctness
# ---------------------------------------------------------------------------


class TestMetricComputation:
    def test_no_proposals_gives_zero_precision(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert result["precision"] == 0.0
        assert result["total_proposals"] == 0

    def test_no_proposals_gives_zero_approved_count(self, storage_root: Path) -> None:
        result = _call(storage_root)
        assert result["approved_count"] == 0

    def test_all_proposals_approved_gives_precision_one(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_p1")
        _write_proposal(props_dir, "skillver_p2")
        _write_activation(acts_dir, skill_version_id="skillver_p1")
        _write_activation(acts_dir, skill_version_id="skillver_p2")
        result = _call(storage_root)
        assert result["precision"] == pytest.approx(1.0)

    def test_half_proposals_approved_gives_precision_half(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_p1")
        _write_proposal(props_dir, "skillver_p2")
        _write_activation(acts_dir, skill_version_id="skillver_p1")
        result = _call(storage_root)
        assert result["precision"] == pytest.approx(0.5)

    def test_proposals_with_no_activations_gives_approved_count_zero(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        _write_proposal(props_dir, "skillver_p1")
        result = _call(storage_root)
        assert result["approved_count"] == 0

    def test_active_activation_matching_proposal_id_counts_as_approved(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_match")
        _write_activation(acts_dir, skill_version_id="skillver_match", status="active")
        result = _call(storage_root)
        assert result["approved_count"] == 1

    def test_pending_activation_does_not_count_as_approved(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_pending")
        _write_activation(
            acts_dir, skill_version_id="skillver_pending", status="pending"
        )
        result = _call(storage_root)
        assert result["approved_count"] == 0

    def test_revoked_activation_does_not_count_as_approved(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_revoked")
        _write_activation(
            acts_dir, skill_version_id="skillver_revoked", status="revoked"
        )
        result = _call(storage_root)
        assert result["approved_count"] == 0

    def test_multiple_active_activations_for_same_proposal_count_once(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_dup")
        _write_activation(acts_dir, skill_version_id="skillver_dup")
        _write_activation(acts_dir, skill_version_id="skillver_dup")
        result = _call(storage_root)
        assert result["approved_count"] == 1

    def test_precision_equals_approved_count_over_total(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        acts_dir = storage_root / "skill_activations"
        _write_proposal(props_dir, "skillver_x1")
        _write_proposal(props_dir, "skillver_x2")
        _write_proposal(props_dir, "skillver_x3")
        _write_activation(acts_dir, skill_version_id="skillver_x1")
        result = _call(storage_root)
        expected = result["approved_count"] / result["total_proposals"]
        assert result["precision"] == pytest.approx(expected)

    def test_missing_proposals_dir_gives_zero_total(
        self, storage_root: Path
    ) -> None:
        result = _call(
            storage_root,
            proposals_dir=storage_root / "nonexistent_proposals",
            activations_dir=storage_root / "nonexistent_activations",
        )
        assert result["total_proposals"] == 0
        assert result["approved_count"] == 0
        assert result["precision"] == 0.0

    def test_missing_activations_dir_still_computes_total(
        self, storage_root: Path
    ) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        _write_proposal(props_dir, "skillver_noa")
        result = _call(
            storage_root,
            activations_dir=storage_root / "nonexistent_activations",
        )
        assert result["total_proposals"] == 1
        assert result["approved_count"] == 0


# ---------------------------------------------------------------------------
# Audit log: success
# ---------------------------------------------------------------------------


class TestAuditLogSuccess:
    def test_writes_computed_audit_entry(self, storage_root: Path) -> None:
        _call(storage_root)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.precision.computed" for r in records)

    def test_audit_level_is_info(self, storage_root: Path) -> None:
        _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.computed"
        )
        assert record["level"] == "info"

    def test_audit_detail_has_metric_id(self, storage_root: Path) -> None:
        result = _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.computed"
        )
        assert record["detail"]["metric_id"] == result["id"]

    def test_audit_detail_has_total_proposals(self, storage_root: Path) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        _write_proposal(props_dir, "skillver_aud")
        _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.computed"
        )
        assert record["detail"]["total_proposals"] == 1

    def test_audit_detail_has_approved_count(self, storage_root: Path) -> None:
        _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.computed"
        )
        assert "approved_count" in record["detail"]

    def test_audit_detail_has_precision(self, storage_root: Path) -> None:
        _call(storage_root)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.computed"
        )
        assert "precision" in record["detail"]


# ---------------------------------------------------------------------------
# Failure: unwritable precision_dir
# ---------------------------------------------------------------------------


class TestFailureUnwritable:
    def test_raises_precision_error_on_write_failure(
        self, storage_root: Path
    ) -> None:
        # Place a regular file at the precision_dir path so mkdir raises.
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)

    def test_error_http_status_is_500(self) -> None:
        err = SkillForgePrecisionError(
            message="boom", timestamp="2026-01-01T00:00:00+00:00"
        )
        assert err.http_status() == 500

    def test_error_code_is_skill_forge_precision_failed(self) -> None:
        err = SkillForgePrecisionError(
            message="boom", timestamp="2026-01-01T00:00:00+00:00"
        )
        assert err.code == "skill_forge_precision_failed"

    def test_failure_writes_compute_failed_audit_entry(
        self, storage_root: Path
    ) -> None:
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)
        records = _audit_records(storage_root)
        assert any(
            r.get("event") == "skill_forge.precision.compute.failed" for r in records
        )

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.compute.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_metric_id(self, storage_root: Path) -> None:
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.compute.failed"
        )
        assert record["detail"]["metric_id"].startswith("skprecision_")

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)
        record = next(
            r
            for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.precision.compute.failed"
        )
        assert "message" in record["detail"] and record["detail"]["message"]

    def test_no_metric_record_written_on_failure(self, storage_root: Path) -> None:
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)
        # prec_dir is a file, not a dir → no precision records
        assert not any(True for _ in storage_root.rglob("*_precision.json"))


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestOtelSpans:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.precision_metric")

    def test_emits_precision_metric_span_on_success(
        self, storage_root: Path
    ) -> None:
        _call(storage_root)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.precision_metric" in span_names

    def test_span_has_metric_id_attribute(self, storage_root: Path) -> None:
        result = _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.precision.metric_id"] == result["id"]

    def test_span_has_total_proposals_attribute(self, storage_root: Path) -> None:
        props_dir = storage_root / "skill_forge" / "proposals"
        _write_proposal(props_dir, "skillver_otel1")
        _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.precision.total_proposals"] == 1

    def test_span_has_approved_count_attribute(self, storage_root: Path) -> None:
        _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.precision.approved_count"] == 0

    def test_span_has_precision_attribute(self, storage_root: Path) -> None:
        _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.precision.precision"] == pytest.approx(0.0)

    def test_span_success_attribute_true_on_success(
        self, storage_root: Path
    ) -> None:
        _call(storage_root)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.precision_metric.success"] is True

    def test_emits_precision_metric_span_on_failure(
        self, storage_root: Path
    ) -> None:
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.precision_metric" in span_names

    def test_span_success_attribute_false_on_failure(
        self, storage_root: Path
    ) -> None:
        prec_dir = storage_root / "precision"
        prec_dir.write_text("not_a_directory")
        with pytest.raises(SkillForgePrecisionError):
            _call(storage_root, precision_dir=prec_dir)
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.precision_metric.success"] is False
