"""
Conformance suite for GET /v1/models.

Tests cover:
  - 200 status, non-empty list, correct response schema.
  - Each model entry has provider, model, context_window, and all capability flags.
  - OTel span: correct name, invocation event with accurate count, span ended.
  - Audit log: not written on success; written with correct fields on failure.
  - OTel on failure: span ERROR status, models.error event, exception recorded.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from meridian_sdk_provider.protocol import ModelCapabilities, ModelEntry
from opentelemetry.trace import StatusCode

from tests.conftest import CapturingAuditLog, MockTracer, make_app, make_test_model_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(
    audit_log: CapturingAuditLog | None = None,
) -> TestClient:
    return TestClient(make_app(audit_log=audit_log), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /v1/models — status and schema
# ---------------------------------------------------------------------------


class TestListStatus:
    def test_returns_200(self, mock_tracer: MockTracer) -> None:
        assert _client().get("/v1/models").status_code == 200

    def test_response_has_models_key(self, mock_tracer: MockTracer) -> None:
        assert "models" in _client().get("/v1/models").json()

    def test_models_is_list(self, mock_tracer: MockTracer) -> None:
        assert isinstance(_client().get("/v1/models").json()["models"], list)

    def test_models_non_empty(self, mock_tracer: MockTracer) -> None:
        assert len(_client().get("/v1/models").json()["models"]) > 0


class TestModelSchema:
    def test_each_model_has_required_fields(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert {"provider", "model", "context_window", "capabilities"} <= m.keys()

    def test_capabilities_has_streaming(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert "streaming" in m["capabilities"]

    def test_capabilities_has_thinking(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert "thinking" in m["capabilities"]

    def test_capabilities_has_vision(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert "vision" in m["capabilities"]

    def test_capabilities_has_tools(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert "tools" in m["capabilities"]

    def test_capabilities_has_cache(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert "cache" in m["capabilities"]

    def test_context_window_is_positive_int(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert isinstance(m["context_window"], int)
            assert m["context_window"] > 0

    def test_provider_is_string(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert isinstance(m["provider"], str)

    def test_model_is_string(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            assert isinstance(m["model"], str)


class TestModelValues:
    def test_capability_flags_are_bool(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        for m in body["models"]:
            caps = m["capabilities"]
            for flag in ("streaming", "thinking", "vision", "tools", "cache"):
                assert isinstance(caps[flag], bool), f"{flag} should be bool"

    def test_thinking_flag_reflects_provider(self, mock_tracer: MockTracer) -> None:
        # claude-opus-4-7 has thinking=True; claude-haiku-4-5 has thinking=False
        body = _client().get("/v1/models").json()
        models_by_name = {m["model"]: m for m in body["models"]}
        assert models_by_name["claude-opus-4-7"]["capabilities"]["thinking"] is True
        assert models_by_name["claude-haiku-4-5"]["capabilities"]["thinking"] is False

    def test_count_matches_registered_models(self, mock_tracer: MockTracer) -> None:
        body = _client().get("/v1/models").json()
        assert len(body["models"]) == 2

    def test_multiple_providers_aggregated(self, mock_tracer: MockTracer) -> None:
        from meridian_sdk_provider import (
            FakeModelAdapter,
            ModelRouter,
            ModelRoutingPolicy,
            ModelRoutingRule,
        )

        router = ModelRouter(policy=ModelRoutingPolicy(rules=[ModelRoutingRule(model="p1:m1")]))
        router.register_provider(
            FakeModelAdapter(
                name="p1",
                models=[
                    ModelEntry(
                        provider="p1",
                        model="m1",
                        context_window=4096,
                        capabilities=ModelCapabilities(),
                    ),
                ],
            )
        )
        router.register_provider(
            FakeModelAdapter(
                name="p2",
                models=[
                    ModelEntry(
                        provider="p2",
                        model="m2",
                        context_window=8192,
                        capabilities=ModelCapabilities(thinking=True),
                    ),
                ],
            )
        )
        from api_models._routes import make_router
        from core_errors import install_error_handler
        from fastapi import FastAPI

        app = FastAPI()
        install_error_handler(app)
        app.include_router(make_router(model_router=router))
        client = TestClient(app, raise_server_exceptions=False)
        body = client.get("/v1/models").json()
        providers = {m["provider"] for m in body["models"]}
        assert {"p1", "p2"} == providers

    def test_empty_provider_returns_empty_list(self, mock_tracer: MockTracer) -> None:
        router = make_test_model_router(models=[])
        from api_models._routes import make_router
        from core_errors import install_error_handler
        from fastapi import FastAPI

        app = FastAPI()
        install_error_handler(app)
        app.include_router(make_router(model_router=router))
        client = TestClient(app, raise_server_exceptions=False)
        body = client.get("/v1/models").json()
        assert body["models"] == []


# ---------------------------------------------------------------------------
# GET /v1/models — OTel
# ---------------------------------------------------------------------------


class TestListTelemetry:
    def test_span_name(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/models")
        assert mock_tracer.spans[0].name == "models.list"

    def test_span_has_invocation_event(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/models")
        event_names = [e[0] for e in mock_tracer.spans[0].events]
        assert "models.list.invocation" in event_names

    def test_invocation_event_count_matches_response(self, mock_tracer: MockTracer) -> None:
        resp = _client().get("/v1/models")
        ev = next(e for e in mock_tracer.spans[0].events if e[0] == "models.list.invocation")
        assert ev[1]["count"] == len(resp.json()["models"])

    def test_invocation_event_count_is_non_negative(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/models")
        ev = next(e for e in mock_tracer.spans[0].events if e[0] == "models.list.invocation")
        assert ev[1]["count"] >= 0

    def test_span_ended(self, mock_tracer: MockTracer) -> None:
        _client().get("/v1/models")
        assert mock_tracer.spans[0].ended


# ---------------------------------------------------------------------------
# GET /v1/models — audit log
# ---------------------------------------------------------------------------


class TestListAuditLog:
    def test_no_audit_on_success(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        _client(audit_log=audit_log).get("/v1/models")
        assert len(audit_log.entries) == 0


# ---------------------------------------------------------------------------
# GET /v1/models — OTel on failure
# ---------------------------------------------------------------------------


class TestListTelemetryOnFailure:
    def _make_failing_client(self, mock_tracer: MockTracer) -> TestClient:
        """Return a client whose model_router.list_models() always raises."""
        from unittest.mock import MagicMock

        from api_models._routes import make_router
        from core_errors import install_error_handler
        from fastapi import FastAPI

        broken_router = MagicMock()
        broken_router.list_models.side_effect = RuntimeError("catalog unavailable")
        app = FastAPI()
        install_error_handler(app)
        app.include_router(make_router(model_router=broken_router))
        return TestClient(app, raise_server_exceptions=False)

    def test_span_status_error_on_failure(self, mock_tracer: MockTracer) -> None:
        client = self._make_failing_client(mock_tracer)
        client.get("/v1/models")
        assert mock_tracer.spans[0].status is not None
        assert mock_tracer.spans[0].status.status_code == StatusCode.ERROR

    def test_span_has_error_event_on_failure(self, mock_tracer: MockTracer) -> None:
        client = self._make_failing_client(mock_tracer)
        client.get("/v1/models")
        event_names = [e[0] for e in mock_tracer.spans[0].events]
        assert "models.error" in event_names

    def test_span_error_event_operation(self, mock_tracer: MockTracer) -> None:
        client = self._make_failing_client(mock_tracer)
        client.get("/v1/models")
        ev = next(e for e in mock_tracer.spans[0].events if e[0] == "models.error")
        assert ev[1]["operation"] == "list"

    def test_exception_recorded_on_span(self, mock_tracer: MockTracer) -> None:
        client = self._make_failing_client(mock_tracer)
        client.get("/v1/models")
        assert len(mock_tracer.spans[0].recorded_exceptions) == 1

    def test_span_ended_on_failure(self, mock_tracer: MockTracer) -> None:
        client = self._make_failing_client(mock_tracer)
        client.get("/v1/models")
        assert mock_tracer.spans[0].ended

    def test_audit_written_on_failure(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        from unittest.mock import MagicMock

        from api_models._routes import make_router
        from core_errors import install_error_handler
        from fastapi import FastAPI

        broken_router = MagicMock()
        broken_router.list_models.side_effect = RuntimeError("catalog unavailable")
        app = FastAPI()
        install_error_handler(app)
        app.include_router(make_router(model_router=broken_router, audit_log=audit_log))
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/v1/models")
        assert len(audit_log.entries) == 1

    def test_audit_entry_level_error(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        from unittest.mock import MagicMock

        from api_models._routes import make_router
        from core_errors import install_error_handler
        from fastapi import FastAPI

        broken_router = MagicMock()
        broken_router.list_models.side_effect = RuntimeError("catalog unavailable")
        app = FastAPI()
        install_error_handler(app)
        app.include_router(make_router(model_router=broken_router, audit_log=audit_log))
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/v1/models")
        assert audit_log.entries[0].level == "error"

    def test_audit_entry_event_name(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        from unittest.mock import MagicMock

        from api_models._routes import make_router
        from core_errors import install_error_handler
        from fastapi import FastAPI

        broken_router = MagicMock()
        broken_router.list_models.side_effect = RuntimeError("catalog unavailable")
        app = FastAPI()
        install_error_handler(app)
        app.include_router(make_router(model_router=broken_router, audit_log=audit_log))
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/v1/models")
        assert audit_log.entries[0].event == "models.list.failed"

    def test_audit_entry_operation(
        self, mock_tracer: MockTracer, audit_log: CapturingAuditLog
    ) -> None:
        from unittest.mock import MagicMock

        from api_models._routes import make_router
        from core_errors import install_error_handler
        from fastapi import FastAPI

        broken_router = MagicMock()
        broken_router.list_models.side_effect = RuntimeError("catalog unavailable")
        app = FastAPI()
        install_error_handler(app)
        app.include_router(make_router(model_router=broken_router, audit_log=audit_log))
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/v1/models")
        assert audit_log.entries[0].operation == "list"
