"""
POST /v1/messages single-shot inference endpoint conformance suite.

Tests cover:
  - Returns 200 on success with a valid body.
  - Response has id, type, role, model, content, stop_reason, usage fields.
  - Text content is correctly assembled from TextDeltaEvent deltas.
  - usage.input_tokens and usage.output_tokens reflect fixture values.
  - Tool use blocks assembled from ToolUseStartEvent and ToolInputDeltaEvent.
  - Missing required fields (model, messages, max_tokens) returns 422.
  - ProviderCallError returns 502 with error.code "inference_error".
  - NoProviderFoundError returns 502 with error.code "inference_error".
  - On failure, audit log entry written with event "messages.infer.failed".
  - Audit level is "error" and detail includes model on failure.
  - OTel span "messages.infer" emitted on success.
  - Span has "model" attribute matching request.
  - Span is set to ERROR status on provider failure.
  - create_app wires messages route when model_router is supplied.
  - create_app omits messages route when model_router is None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from meridiand._app import create_app
from meridiand._audit import FileAuditLog
from meridian_sdk_provider import (
    FakeModelAdapter,
    ModelCallOpts,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    ProviderCapabilities,
    write_model_fixture,
)
from meridian_sdk_provider.errors import ProviderCallError
from meridian_sdk_provider.types import ModelCountReq, TokenCount

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

_DEFAULT_EVENTS: list[dict[str, Any]] = [
    {"type": "message_start", "model": "test-model", "provider": "fake", "input_tokens": 15},
    {"type": "text_delta", "text": "Hello, "},
    {"type": "text_delta", "text": "world!"},
    {"type": "message_stop", "input_tokens": 15, "output_tokens": 4, "stop_reason": "end_turn"},
]

_TOOL_EVENTS: list[dict[str, Any]] = [
    {"type": "message_start", "model": "test-model", "provider": "fake", "input_tokens": 20},
    {
        "type": "tool_use_start",
        "id": "tu_abc123",
        "name": "get_weather",
    },
    {"type": "tool_input_delta", "id": "tu_abc123", "partial_json": '{"location":'},
    {"type": "tool_input_delta", "id": "tu_abc123", "partial_json": '"Paris"}'},
    {"type": "message_stop", "input_tokens": 20, "output_tokens": 10, "stop_reason": "tool_use"},
]

_DEFAULT_REQUEST: dict[str, Any] = {
    "model": "fake:test-model",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 256,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(fixtures_dir: Path) -> ModelRouter:
    adapter = FakeModelAdapter(fixtures_dir=fixtures_dir, name="fake")
    policy = ModelRoutingPolicy(rules=[ModelRoutingRule(model="fake:test-model")])
    return ModelRouter(policy=policy, providers={"fake": adapter})


def _make_client(tmp_path: Path) -> TestClient:
    fixtures_dir = tmp_path / "fixtures"
    write_model_fixture(fixtures_dir / "test-model.ndjson", _DEFAULT_EVENTS)
    audit_log = FileAuditLog(tmp_path)
    app = create_app(audit_log, model_router=_make_router(fixtures_dir))
    return TestClient(app, raise_server_exceptions=False)


def _audit_records(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Error adapter for provider-failure tests
# ---------------------------------------------------------------------------


class _ErrorAdapter:
    name = "error"
    kind = "error"
    capabilities = ProviderCapabilities()

    async def call(self, opts: ModelCallOpts):
        raise ProviderCallError("provider failed", provider_name="error")
        yield  # pragma: no cover

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        return TokenCount(input_tokens=0)

    async def close(self) -> None:
        pass


def _make_error_client(tmp_path: Path) -> TestClient:
    adapter = _ErrorAdapter()
    policy = ModelRoutingPolicy(rules=[ModelRoutingRule(model="error:any-model")])
    router = ModelRouter(policy=policy, providers={"error": adapter})
    audit_log = FileAuditLog(tmp_path)
    app = create_app(audit_log, model_router=router)
    return TestClient(app, raise_server_exceptions=False)


def _error_request() -> dict[str, Any]:
    return {
        "model": "error:any-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 256,
    }


# ---------------------------------------------------------------------------
# Success: status and shape
# ---------------------------------------------------------------------------


class TestMessagesSuccess:
    def test_returns_200(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/v1/messages", json=_DEFAULT_REQUEST)
        assert resp.status_code == 200

    def test_response_has_id(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert "id" in body
        assert body["id"].startswith("msg_")

    def test_response_id_is_unique_per_call(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        id1 = client.post("/v1/messages", json=_DEFAULT_REQUEST).json()["id"]
        id2 = client.post("/v1/messages", json=_DEFAULT_REQUEST).json()["id"]
        assert id1 != id2

    def test_response_type_is_message(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert body["type"] == "message"

    def test_response_role_is_assistant(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert body["role"] == "assistant"

    def test_response_has_model(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert "model" in body
        assert isinstance(body["model"], str)

    def test_response_has_content_list(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert "content" in body
        assert isinstance(body["content"], list)

    def test_response_has_stop_reason(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert "stop_reason" in body

    def test_response_has_usage(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert "usage" in body

    def test_response_usage_has_input_tokens(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert "input_tokens" in body["usage"]

    def test_response_usage_has_output_tokens(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert "output_tokens" in body["usage"]

    def test_stop_reason_from_fixture(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert body["stop_reason"] == "end_turn"

    def test_input_tokens_from_fixture(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert body["usage"]["input_tokens"] == 15

    def test_output_tokens_from_fixture(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert body["usage"]["output_tokens"] == 4


# ---------------------------------------------------------------------------
# Content assembly
# ---------------------------------------------------------------------------


class TestMessagesContentAssembly:
    def test_text_deltas_joined_into_single_block(self, tmp_path: Path) -> None:
        body = _make_client(tmp_path).post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert len(body["content"]) == 1
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"] == "Hello, world!"

    def test_tool_use_block_assembled(self, tmp_path: Path) -> None:
        fixtures_dir = tmp_path / "fixtures"
        write_model_fixture(fixtures_dir / "test-model.ndjson", _TOOL_EVENTS)
        audit_log = FileAuditLog(tmp_path)
        app = create_app(audit_log, model_router=_make_router(fixtures_dir))
        client = TestClient(app, raise_server_exceptions=False)

        body = client.post("/v1/messages", json=_DEFAULT_REQUEST).json()
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == "tu_abc123"
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"location": "Paris"}

    def test_tool_stop_reason_is_tool_use(self, tmp_path: Path) -> None:
        fixtures_dir = tmp_path / "fixtures"
        write_model_fixture(fixtures_dir / "test-model.ndjson", _TOOL_EVENTS)
        audit_log = FileAuditLog(tmp_path)
        app = create_app(audit_log, model_router=_make_router(fixtures_dir))
        client = TestClient(app, raise_server_exceptions=False)
        body = client.post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert body["stop_reason"] == "tool_use"

    def test_empty_content_when_no_deltas(self, tmp_path: Path) -> None:
        fixtures_dir = tmp_path / "fixtures"
        write_model_fixture(
            fixtures_dir / "test-model.ndjson",
            [
                {"type": "message_start", "model": "test-model", "provider": "fake"},
                {"type": "message_stop", "stop_reason": "end_turn"},
            ],
        )
        audit_log = FileAuditLog(tmp_path)
        app = create_app(audit_log, model_router=_make_router(fixtures_dir))
        client = TestClient(app, raise_server_exceptions=False)
        body = client.post("/v1/messages", json=_DEFAULT_REQUEST).json()
        assert body["content"] == []


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestMessagesSchemaValidation:
    def test_missing_model_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64},
        )
        assert resp.status_code == 422

    def test_missing_messages_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post(
            "/v1/messages",
            json={"model": "fake:test-model", "max_tokens": 64},
        )
        assert resp.status_code == 422

    def test_missing_max_tokens_returns_422(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post(
            "/v1/messages",
            json={"model": "fake:test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------


class TestMessagesProviderError:
    def test_provider_error_returns_502(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        resp = client.post("/v1/messages", json=_error_request())
        assert resp.status_code == 502

    def test_provider_error_code_is_inference_error(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        body = client.post("/v1/messages", json=_error_request()).json()
        assert body["error"]["code"] == "inference_error"

    def test_provider_error_message_in_response(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        body = client.post("/v1/messages", json=_error_request()).json()
        assert "message" in body["error"]
        assert len(body["error"]["message"]) > 0

    def test_no_provider_found_returns_502(self, tmp_path: Path) -> None:
        policy = ModelRoutingPolicy(rules=[ModelRoutingRule(model="missing:model")])
        router = ModelRouter(policy=policy, providers={})
        audit_log = FileAuditLog(tmp_path)
        app = create_app(audit_log, model_router=router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "missing:model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
            },
        )
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestMessagesAuditLog:
    def test_provider_failure_writes_audit_entry(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        client.post("/v1/messages", json=_error_request())
        records = _audit_records(tmp_path)
        assert any(r.get("event") == "messages.infer.failed" for r in records)

    def test_audit_level_is_error(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        client.post("/v1/messages", json=_error_request())
        record = next(
            r for r in _audit_records(tmp_path) if r.get("event") == "messages.infer.failed"
        )
        assert record["level"] == "error"

    def test_audit_code_is_inference_error(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        client.post("/v1/messages", json=_error_request())
        record = next(
            r for r in _audit_records(tmp_path) if r.get("event") == "messages.infer.failed"
        )
        assert record["code"] == "inference_error"

    def test_audit_detail_has_model(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        client.post("/v1/messages", json=_error_request())
        record = next(
            r for r in _audit_records(tmp_path) if r.get("event") == "messages.infer.failed"
        )
        assert "model" in record["detail"]

    def test_audit_detail_model_matches_request(self, tmp_path: Path) -> None:
        client = _make_error_client(tmp_path)
        req = _error_request()
        client.post("/v1/messages", json=req)
        record = next(
            r for r in _audit_records(tmp_path) if r.get("event") == "messages.infer.failed"
        )
        assert record["detail"]["model"] == req["model"]


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestMessagesOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_success_emits_messages_infer_span(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/messages", json=_DEFAULT_REQUEST)
        span_names = [s.name for s in _otel_exporter.get_finished_spans()]
        assert "messages.infer" in span_names

    def test_span_has_model_attribute(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/messages", json=_DEFAULT_REQUEST)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        infer_span = spans.get("messages.infer")
        assert infer_span is not None
        assert infer_span.attributes["model"] == _DEFAULT_REQUEST["model"]

    def test_span_has_max_tokens_attribute(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post("/v1/messages", json=_DEFAULT_REQUEST)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        infer_span = spans.get("messages.infer")
        assert infer_span is not None
        assert infer_span.attributes["max_tokens"] == _DEFAULT_REQUEST["max_tokens"]

    def test_provider_failure_span_has_error_status(self, tmp_path: Path) -> None:
        from opentelemetry.trace import StatusCode

        client = _make_error_client(tmp_path)
        client.post("/v1/messages", json=_error_request())
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        infer_span = spans.get("messages.infer")
        assert infer_span is not None
        assert infer_span.status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# Route wiring
# ---------------------------------------------------------------------------


class TestMessagesRouteWiring:
    def test_no_model_router_no_route(self, tmp_path: Path) -> None:
        audit_log = FileAuditLog(tmp_path)
        app = create_app(audit_log)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/messages", json=_DEFAULT_REQUEST)
        assert resp.status_code == 404

    def test_with_model_router_route_exists(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.post("/v1/messages", json=_DEFAULT_REQUEST)
        assert resp.status_code != 404
