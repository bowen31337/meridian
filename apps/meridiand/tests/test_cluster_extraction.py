"""
Cluster extraction conformance suite.

Tests cover:
  - extract_cluster returns a ClusterExtractionResult on success.
  - Result cluster_id matches the input cluster ID.
  - Result name is parsed from the model response.
  - Result description is parsed from the model response.
  - Result instructions are parsed from the model response.
  - Result tools are parsed from the model response (name + capabilities).
  - Result tests are parsed from the model response (description + steps).
  - extract_cluster emits OTel span "skill_forge.cluster.extract".
  - OTel span carries skill_forge.cluster.id attribute.
  - OTel span carries skill_forge.cluster.size attribute.
  - OTel span sets skill_forge.cluster.extract.success=True on success.
  - OTel span sets skill_forge.cluster.extract.success=False on failure.
  - extract_cluster writes audit entry "skill_forge.cluster.extracted" on success.
  - Success audit entry level is "info".
  - Success audit detail contains cluster_id.
  - Success audit detail contains cluster_size.
  - Success audit detail contains skill_name.
  - extract_cluster raises ClusterExtractionError on router failure.
  - extract_cluster raises ClusterExtractionError on invalid JSON response.
  - On failure: writes audit entry "skill_forge.cluster.extract.failed".
  - Failure audit entry level is "error".
  - Failure audit detail contains cluster_id.
  - Failure audit detail contains message.
  - ClusterExtractionError carries error code "cluster_extraction_failed".
  - User message sent to router includes cluster ID.
  - User message sent to router includes member count.
  - User message sent to router includes session tool calls.
  - System prompt is passed to the router opts.
  - Role "skill_forge_extractor" is set on the router opts.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from meridiand._audit import FileAuditLog
from meridiand._cluster_extraction import (
    Cluster,
    ClusterExtractionError,
    ClusterExtractionResult,
    ClusterMember,
    SkillTestCase,
    ToolRequirement,
    _build_user_message,
    _parse_response,
    extract_cluster,
)
from meridian_sdk_provider.types import (
    MessageStopEvent,
    ModelCallOpts,
    TextDeltaEvent,
)

from tests._otel_shared import otel_exporter as _otel_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RESPONSE = json.dumps(
    {
        "name": "run_tests",
        "description": "Runs the project test suite and reports results.",
        "instructions": "Execute the test runner and capture output.",
        "tools": [{"name": "Bash", "capabilities": ["shell"]}],
        "tests": [{"description": "Verifies tests run", "steps": ["invoke runner"]}],
    }
)


def _make_cluster(
    cluster_id: str = "cluster_abc",
    members: list[ClusterMember] | None = None,
) -> Cluster:
    if members is None:
        members = [
            ClusterMember(session_id="sess_1", tool_calls=["Bash", "Read"]),
            ClusterMember(session_id="sess_2", tool_calls=["Bash"]),
        ]
    return Cluster(id=cluster_id, members=members)


def _audit_records(storage_root: Path) -> list[dict[str, Any]]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class _OkRouter:
    """Stub router that streams a fixed text response."""

    def __init__(self, response: str = _VALID_RESPONSE) -> None:
        self._response = response
        self.last_opts: ModelCallOpts | None = None

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[Any]:
        self.last_opts = opts
        yield TextDeltaEvent(text=self._response)
        yield MessageStopEvent()


class _FailRouter:
    """Stub router that always raises on call."""

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[Any]:
        raise RuntimeError("router exploded")
        yield  # make it an async generator


# ---------------------------------------------------------------------------
# extract_cluster: success — result fields
# ---------------------------------------------------------------------------


class TestExtractClusterResult:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_returns_cluster_extraction_result(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        result = asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert isinstance(result, ClusterExtractionResult)

    def test_result_cluster_id_matches_input(self, storage_root: Path) -> None:
        cluster = _make_cluster(cluster_id="cluster_xyz")
        result = asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert result.cluster_id == "cluster_xyz"

    def test_result_name_parsed(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        result = asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert result.name == "run_tests"

    def test_result_description_parsed(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        result = asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert "test suite" in result.description

    def test_result_instructions_parsed(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        result = asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert "test runner" in result.instructions

    def test_result_tools_parsed(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        result = asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert len(result.tools) == 1
        assert isinstance(result.tools[0], ToolRequirement)
        assert result.tools[0].name == "Bash"
        assert result.tools[0].capabilities == ["shell"]

    def test_result_tests_parsed(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        result = asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert len(result.tests) == 1
        assert isinstance(result.tests[0], SkillTestCase)
        assert "Verifies" in result.tests[0].description
        assert result.tests[0].steps == ["invoke runner"]


# ---------------------------------------------------------------------------
# extract_cluster: OTel
# ---------------------------------------------------------------------------


class TestExtractClusterOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def _get_span(self) -> Any:
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        return spans.get("skill_forge.cluster.extract")

    def test_emits_cluster_extract_span(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "skill_forge.cluster.extract" in span_names

    def test_span_has_cluster_id_attribute(self, storage_root: Path) -> None:
        cluster = _make_cluster(cluster_id="cluster_otel")
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.cluster.id"] == "cluster_otel"

    def test_span_has_cluster_size_attribute(self, storage_root: Path) -> None:
        cluster = _make_cluster(
            members=[
                ClusterMember(session_id="s1", tool_calls=["Bash"]),
                ClusterMember(session_id="s2", tool_calls=["Read"]),
                ClusterMember(session_id="s3", tool_calls=[]),
            ]
        )
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.cluster.size"] == 3

    def test_span_success_true_on_success(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.cluster.extract.success"] is True

    def test_span_success_false_on_failure(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        with pytest.raises(ClusterExtractionError):
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_FailRouter(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        span = self._get_span()
        assert span is not None
        assert span.attributes["skill_forge.cluster.extract.success"] is False


# ---------------------------------------------------------------------------
# extract_cluster: audit log
# ---------------------------------------------------------------------------


class TestExtractClusterAudit:
    def test_success_writes_extracted_audit_entry(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.cluster.extracted" for r in records)

    def test_success_audit_level_is_info(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.cluster.extracted"
        )
        assert record["level"] == "info"

    def test_success_audit_detail_has_cluster_id(self, storage_root: Path) -> None:
        cluster = _make_cluster(cluster_id="cluster_audit")
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.cluster.extracted"
        )
        assert record["detail"]["cluster_id"] == "cluster_audit"

    def test_success_audit_detail_has_cluster_size(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.cluster.extracted"
        )
        assert record["detail"]["cluster_size"] == len(cluster.members)

    def test_success_audit_detail_has_skill_name(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        asyncio.run(
            extract_cluster(
                cluster,
                router=_OkRouter(),
                audit_log=FileAuditLog(storage_root),
            )
        )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.cluster.extracted"
        )
        assert record["detail"]["skill_name"] == "run_tests"

    def test_failure_raises_cluster_extraction_error(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        with pytest.raises(ClusterExtractionError):
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_FailRouter(),
                    audit_log=FileAuditLog(storage_root),
                )
            )

    def test_failure_on_invalid_json_raises_cluster_extraction_error(
        self, storage_root: Path
    ) -> None:
        cluster = _make_cluster()
        with pytest.raises(ClusterExtractionError):
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_OkRouter(response="not valid json {{{"),
                    audit_log=FileAuditLog(storage_root),
                )
            )

    def test_failure_writes_failed_audit_entry(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        with pytest.raises(ClusterExtractionError):
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_FailRouter(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        records = _audit_records(storage_root)
        assert any(r.get("event") == "skill_forge.cluster.extract.failed" for r in records)

    def test_failure_audit_level_is_error(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        with pytest.raises(ClusterExtractionError):
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_FailRouter(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.cluster.extract.failed"
        )
        assert record["level"] == "error"

    def test_failure_audit_detail_has_cluster_id(self, storage_root: Path) -> None:
        cluster = _make_cluster(cluster_id="cluster_fail")
        with pytest.raises(ClusterExtractionError):
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_FailRouter(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.cluster.extract.failed"
        )
        assert record["detail"]["cluster_id"] == "cluster_fail"

    def test_failure_audit_detail_has_message(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        with pytest.raises(ClusterExtractionError):
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_FailRouter(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        record = next(
            r for r in _audit_records(storage_root)
            if r.get("event") == "skill_forge.cluster.extract.failed"
        )
        assert "message" in record["detail"] and record["detail"]["message"]

    def test_error_code_is_cluster_extraction_failed(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        with pytest.raises(ClusterExtractionError) as exc_info:
            asyncio.run(
                extract_cluster(
                    cluster,
                    router=_FailRouter(),
                    audit_log=FileAuditLog(storage_root),
                )
            )
        assert exc_info.value.code == "cluster_extraction_failed"


# ---------------------------------------------------------------------------
# extract_cluster: router opts
# ---------------------------------------------------------------------------


class TestExtractClusterRouterOpts:
    def test_user_message_includes_cluster_id(self, storage_root: Path) -> None:
        cluster = _make_cluster(cluster_id="cluster_prompt_test")
        router = _OkRouter()
        asyncio.run(
            extract_cluster(
                cluster,
                router=router,
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert router.last_opts is not None
        msg = router.last_opts.messages[0]
        assert "cluster_prompt_test" in (msg.content if isinstance(msg.content, str) else "")

    def test_user_message_includes_member_count(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        router = _OkRouter()
        asyncio.run(
            extract_cluster(
                cluster,
                router=router,
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert router.last_opts is not None
        msg = router.last_opts.messages[0]
        assert str(len(cluster.members)) in (
            msg.content if isinstance(msg.content, str) else ""
        )

    def test_user_message_includes_tool_calls(self, storage_root: Path) -> None:
        cluster = _make_cluster(
            members=[ClusterMember(session_id="sess_x", tool_calls=["Glob", "Write"])]
        )
        router = _OkRouter()
        asyncio.run(
            extract_cluster(
                cluster,
                router=router,
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert router.last_opts is not None
        msg = router.last_opts.messages[0]
        content = msg.content if isinstance(msg.content, str) else ""
        assert "Glob" in content
        assert "Write" in content

    def test_system_prompt_is_set(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        router = _OkRouter()
        asyncio.run(
            extract_cluster(
                cluster,
                router=router,
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert router.last_opts is not None
        assert router.last_opts.system is not None
        assert len(router.last_opts.system) > 0

    def test_role_is_skill_forge_extractor(self, storage_root: Path) -> None:
        cluster = _make_cluster()
        router = _OkRouter()
        asyncio.run(
            extract_cluster(
                cluster,
                router=router,
                audit_log=FileAuditLog(storage_root),
            )
        )
        assert router.last_opts is not None
        assert router.last_opts.role == "skill_forge_extractor"


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    def test_contains_cluster_id(self) -> None:
        cluster = _make_cluster(cluster_id="cid_xyz")
        msg = _build_user_message(cluster)
        assert "cid_xyz" in msg

    def test_contains_member_count(self) -> None:
        cluster = _make_cluster()
        msg = _build_user_message(cluster)
        assert str(len(cluster.members)) in msg

    def test_contains_session_id(self) -> None:
        cluster = _make_cluster(
            members=[ClusterMember(session_id="sess_unique", tool_calls=[])]
        )
        msg = _build_user_message(cluster)
        assert "sess_unique" in msg

    def test_contains_tool_call_names(self) -> None:
        cluster = _make_cluster(
            members=[ClusterMember(session_id="s1", tool_calls=["Edit", "Glob"])]
        )
        msg = _build_user_message(cluster)
        assert "Edit" in msg
        assert "Glob" in msg

    def test_empty_tool_calls_shows_none(self) -> None:
        cluster = _make_cluster(
            members=[ClusterMember(session_id="s1", tool_calls=[])]
        )
        msg = _build_user_message(cluster)
        assert "(none)" in msg


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_parses_name(self) -> None:
        result = _parse_response(_VALID_RESPONSE, "c1")
        assert result.name == "run_tests"

    def test_parses_description(self) -> None:
        result = _parse_response(_VALID_RESPONSE, "c1")
        assert result.description != ""

    def test_parses_instructions(self) -> None:
        result = _parse_response(_VALID_RESPONSE, "c1")
        assert result.instructions != ""

    def test_parses_tools(self) -> None:
        result = _parse_response(_VALID_RESPONSE, "c1")
        assert len(result.tools) == 1
        assert result.tools[0].name == "Bash"

    def test_parses_tests(self) -> None:
        result = _parse_response(_VALID_RESPONSE, "c1")
        assert len(result.tests) == 1

    def test_cluster_id_set(self) -> None:
        result = _parse_response(_VALID_RESPONSE, "my_cluster")
        assert result.cluster_id == "my_cluster"

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_response("not json {{", "c1")
