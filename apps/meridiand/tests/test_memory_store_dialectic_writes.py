"""
Honcho-style dialectic write conformance suite.

Tests cover:
  - dialectic=False (default): existing insert/update behaviour unchanged.
  - dialectic=True with no model_router: falls back to regular write.
  - dialectic=True, empty store: action "inserted" without model call.
  - dialectic=True, model returns "duplicate": action is "deduplicated".
  - dialectic=True, model returns "duplicate": key NOT written to store.
  - dialectic=True, model returns "refinement": action is "merged".
  - dialectic=True, model returns "refinement": content is merged_content from model.
  - dialectic=True, model returns "contradiction": action is "superseded".
  - dialectic=True, model returns "contradiction": provenance sidecar written.
  - dialectic=True, model returns "contradiction": provenance contains key + match_key.
  - dialectic=True, model returns "net-new": action is "inserted".
  - response includes dialectic_label field (None when dialectic=False).
  - response includes dialectic_match_key field (None when dialectic=False).
  - dialectic_label matches classifier output in response.
  - classification failure raises 500 with code "memory_store_dialectic_failed".
  - classification failure writes audit entry "memory_store.write.failed".
  - classification failure audit code is "memory_store_dialectic_failed".
  - OTel span carries "memory_store.write.dialectic" attribute.
  - OTel span carries "memory_store.write.dialectic_label" when classified.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from pathlib import Path

from fastapi.testclient import TestClient
from meridian_sdk_provider import (
    ModelCallOpts,
    ModelCountReq,
    ModelRouter,
    ModelRoutingPolicy,
    ModelRoutingRule,
    TextDeltaEvent,
    TokenCount,
)
from meridian_sdk_provider.protocol import ProviderCapabilities
from meridian_sdk_provider.types import MessageStartEvent, MessageStopEvent, ModelEvent
from meridiand._app import create_app
from meridiand._audit import FileAuditLog

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------


class _StubProvider:
    """Synchronous stub returning a configurable JSON text response."""

    def __init__(self, response_json: str, *, raise_on_call: Exception | None = None) -> None:
        self.name = "stub"
        self.kind = "stub"
        self.capabilities = ProviderCapabilities()
        self._response = response_json
        self._raise = raise_on_call

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        if self._raise is not None:
            raise self._raise
        yield MessageStartEvent(model="stub", provider="stub")  # type: ignore[misc]
        yield TextDeltaEvent(text=self._response)  # type: ignore[misc]
        yield MessageStopEvent(stop_reason="end_turn")  # type: ignore[misc]

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        return TokenCount(input_tokens=0)

    async def close(self) -> None:
        pass


_LABEL_RESPONSES: dict[str, str] = {
    "duplicate": json.dumps(
        {
            "label": "duplicate",
            "match_key": "existing_key",
            "merged_content": None,
            "explanation": "Same information.",
        }
    ),
    "refinement": json.dumps(
        {
            "label": "refinement",
            "match_key": "existing_key",
            "merged_content": "merged alpha beta",
            "explanation": "New info compatible.",
        }
    ),
    "contradiction": json.dumps(
        {
            "label": "contradiction",
            "match_key": "existing_key",
            "merged_content": None,
            "explanation": "Contradicts existing.",
        }
    ),
    "net-new": json.dumps(
        {
            "label": "net-new",
            "match_key": None,
            "merged_content": None,
            "explanation": "No similar memories.",
        }
    ),
}


def _make_router(label: str) -> ModelRouter:
    policy = ModelRoutingPolicy(rules=[ModelRoutingRule(model="stub:memory_classifier")])
    provider = _StubProvider(_LABEL_RESPONSES[label])
    return ModelRouter(policy=policy, providers={"stub": provider})


def _make_failing_router() -> ModelRouter:
    policy = ModelRoutingPolicy(rules=[ModelRoutingRule(model="stub:memory_classifier")])
    provider = _StubProvider("", raise_on_call=RuntimeError("provider down"))
    return ModelRouter(policy=policy, providers={"stub": provider})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(storage_root: Path, model_router: ModelRouter | None = None) -> TestClient:
    app = create_app(
        FileAuditLog(storage_root),
        storage_root=storage_root,
        model_router=model_router,
    )
    return TestClient(app, raise_server_exceptions=False)


def _create_store(client: TestClient) -> str:
    resp = client.post(
        "/v1/memory_stores",
        json={"name": "test-store", "backend": "sqlite-vec", "scope": "global"},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def _write(client: TestClient, store_id: str, **overrides) -> dict:
    body: dict = {"key": "k1", "content": "alpha beta gamma"}
    body.update(overrides)
    return client.post(f"/v1/memory_stores/{store_id}/write", json=body).json()


def _write_resp(client: TestClient, store_id: str, **overrides):
    body: dict = {"key": "k1", "content": "alpha beta gamma"}
    body.update(overrides)
    return client.post(f"/v1/memory_stores/{store_id}/write", json=body)


def _audit_records(storage_root: Path) -> list[dict]:
    path = storage_root / "audit.ndjson"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_existing(client: TestClient, store_id: str, key: str = "existing_key") -> None:
    """Write an existing memory so the store is non-empty before dialectic write."""
    resp = client.post(
        f"/v1/memory_stores/{store_id}/write",
        json={"key": key, "content": "alpha beta existing content"},
    )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Backward compatibility: dialectic=False (default)
# ---------------------------------------------------------------------------


class TestDialecticFalseBackwardCompat:
    def test_default_action_inserted(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = _write(client, store_id)
        assert body["action"] == "inserted"

    def test_default_action_updated_on_second_write(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        _write(client, store_id, key="k1")
        body = _write(client, store_id, key="k1")
        assert body["action"] == "updated"

    def test_dialectic_label_is_none_when_false(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = _write(client, store_id, dialectic=False)
        assert body["dialectic_label"] is None

    def test_dialectic_match_key_is_none_when_false(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = _write(client, store_id, dialectic=False)
        assert body["dialectic_match_key"] is None


# ---------------------------------------------------------------------------
# No model_router: dialectic=True falls back silently
# ---------------------------------------------------------------------------


class TestDialecticNoRouter:
    def test_dialectic_true_no_router_returns_201(self, storage_root: Path) -> None:
        client = _make_client(storage_root)  # no model_router
        store_id = _create_store(client)
        resp = _write_resp(client, store_id, dialectic=True)
        assert resp.status_code == 201

    def test_dialectic_true_no_router_inserts(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = _write(client, store_id, dialectic=True)
        assert body["action"] == "inserted"

    def test_dialectic_true_no_router_label_is_none(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        body = _write(client, store_id, dialectic=True)
        assert body["dialectic_label"] is None


# ---------------------------------------------------------------------------
# Empty store: net-new without model call
# ---------------------------------------------------------------------------


class TestDialecticEmptyStore:
    def test_empty_store_action_inserted(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("net-new"))
        store_id = _create_store(client)
        body = _write(client, store_id, dialectic=True)
        # No existing memories → short-circuit to net-new without model call.
        assert body["action"] == "inserted"

    def test_empty_store_dialectic_label_is_net_new(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("net-new"))
        store_id = _create_store(client)
        body = _write(client, store_id, dialectic=True)
        assert body["dialectic_label"] == "net-new"


# ---------------------------------------------------------------------------
# duplicate → deduplicated
# ---------------------------------------------------------------------------


class TestDialecticDuplicate:
    def test_action_is_deduplicated(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("duplicate"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["action"] == "deduplicated"

    def test_key_not_written_to_store(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("duplicate"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write(client, store_id, key="k_new", dialectic=True)
        # The deduplicated key should not appear in query results.
        result = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json={"query": "k_new"},
        ).json()
        keys = [r["file_path"] for r in result["results"]]
        assert "k_new" not in keys

    def test_dialectic_label_is_duplicate(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("duplicate"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["dialectic_label"] == "duplicate"

    def test_dialectic_match_key_present(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("duplicate"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["dialectic_match_key"] == "existing_key"


# ---------------------------------------------------------------------------
# refinement → merged
# ---------------------------------------------------------------------------


class TestDialecticRefinement:
    def test_action_is_merged(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("refinement"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["action"] == "merged"

    def test_content_is_merged_content_from_model(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("refinement"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", content="incoming", dialectic=True)
        assert body["content"] == "merged alpha beta"

    def test_merged_content_is_queryable(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("refinement"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write(client, store_id, key="k_new", content="incoming", dialectic=True)
        result = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json={"query": "merged alpha beta"},
        ).json()
        contents = [r["content"] for r in result["results"]]
        assert any("merged alpha beta" in c for c in contents)

    def test_dialectic_label_is_refinement(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("refinement"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["dialectic_label"] == "refinement"


# ---------------------------------------------------------------------------
# contradiction → superseded
# ---------------------------------------------------------------------------


class TestDialecticContradiction:
    def test_action_is_superseded(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("contradiction"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["action"] == "superseded"

    def test_provenance_sidecar_written(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("contradiction"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write(client, store_id, key="k_new", dialectic=True)
        stores_dir = storage_root / "memory_stores"
        prov_file = stores_dir / store_id / "provenance" / "k_new.json"
        assert prov_file.exists()

    def test_provenance_contains_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("contradiction"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write(client, store_id, key="k_new", dialectic=True)
        stores_dir = storage_root / "memory_stores"
        prov = json.loads((stores_dir / store_id / "provenance" / "k_new.json").read_text())
        assert prov["key"] == "k_new"

    def test_provenance_contains_match_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("contradiction"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write(client, store_id, key="k_new", dialectic=True)
        stores_dir = storage_root / "memory_stores"
        prov = json.loads((stores_dir / store_id / "provenance" / "k_new.json").read_text())
        assert prov["superseded_match_key"] == "existing_key"

    def test_dialectic_label_is_contradiction(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("contradiction"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["dialectic_label"] == "contradiction"

    def test_superseded_content_is_queryable(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("contradiction"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write(client, store_id, key="k_new", content="new contradicting fact", dialectic=True)
        result = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json={"query": "new contradicting fact"},
        ).json()
        contents = [r["content"] for r in result["results"]]
        assert any("new contradicting fact" in c for c in contents)


# ---------------------------------------------------------------------------
# net-new → inserted
# ---------------------------------------------------------------------------


class TestDialecticNetNew:
    def test_action_is_inserted(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("net-new"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["action"] == "inserted"

    def test_dialectic_label_is_net_new(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("net-new"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["dialectic_label"] == "net-new"

    def test_content_queryable_after_insert(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("net-new"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write(client, store_id, key="k_new", content="unique xyzzy phrase", dialectic=True)
        result = client.post(
            f"/v1/memory_stores/{store_id}/query_runs",
            json={"query": "xyzzy"},
        ).json()
        assert result["count"] >= 1


# ---------------------------------------------------------------------------
# Classification failure
# ---------------------------------------------------------------------------


class TestDialecticClassificationFailure:
    def test_failure_returns_500(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_failing_router())
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        resp = _write_resp(client, store_id, key="k_new", dialectic=True)
        assert resp.status_code == 500

    def test_failure_error_code_is_dialectic_failed(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_failing_router())
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        body = _write(client, store_id, key="k_new", dialectic=True)
        assert body["error"]["code"] == "memory_store_dialectic_failed"

    def test_failure_writes_audit_log(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_failing_router())
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write_resp(client, store_id, key="k_new", dialectic=True)
        records = _audit_records(storage_root)
        assert any(r.get("event") == "memory_store.write.failed" for r in records)

    def test_failure_audit_code_is_dialectic_failed(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_failing_router())
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write_resp(client, store_id, key="k_new", dialectic=True)
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "memory_store.write.failed"
        )
        assert record["code"] == "memory_store_dialectic_failed"

    def test_failure_audit_detail_has_key(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_failing_router())
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _write_resp(client, store_id, key="k_new", dialectic=True)
        record = next(
            r for r in _audit_records(storage_root) if r.get("event") == "memory_store.write.failed"
        )
        assert record["detail"]["key"] == "k_new"


# ---------------------------------------------------------------------------
# OTel spans
# ---------------------------------------------------------------------------


class TestDialecticOtel:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    def test_span_has_dialectic_attribute_false(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        _otel_exporter.clear()
        _write(client, store_id, dialectic=False)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.write")
        assert span is not None
        assert span.attributes.get("memory_store.write.dialectic") is False

    def test_span_has_dialectic_attribute_true(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("net-new"))
        store_id = _create_store(client)
        _otel_exporter.clear()
        _write(client, store_id, dialectic=True)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.write")
        assert span is not None
        assert span.attributes.get("memory_store.write.dialectic") is True

    def test_span_has_dialectic_label_when_classified(self, storage_root: Path) -> None:
        client = _make_client(storage_root, _make_router("net-new"))
        store_id = _create_store(client)
        _seed_existing(client, store_id)
        _otel_exporter.clear()
        _write(client, store_id, key="k_new", dialectic=True)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.write")
        assert span is not None
        assert span.attributes.get("memory_store.write.dialectic_label") == "net-new"

    def test_span_no_dialectic_label_when_not_classified(self, storage_root: Path) -> None:
        client = _make_client(storage_root)
        store_id = _create_store(client)
        _otel_exporter.clear()
        _write(client, store_id, dialectic=False)
        spans = {s.name: s for s in _otel_exporter.get_finished_spans()}
        span = spans.get("memory_store.write")
        assert span is not None
        assert "memory_store.write.dialectic_label" not in span.attributes
