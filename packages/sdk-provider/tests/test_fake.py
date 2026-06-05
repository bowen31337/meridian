"""Tests for FakeModelAdapter — fixture loading, OTel span, audit log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from meridian_sdk_provider import (
    AuditLogEntry,
    MessageStartEvent,
    MessageStopEvent,
    ModelProvider,
    TextDeltaEvent,
)
from meridian_sdk_provider.fake import FakeModelAdapter, write_model_fixture

from .conftest import CollectingAuditLog, make_opts

# ---------------------------------------------------------------------------
# OTel mock (mirrors the pattern used in sdk-sandbox conformance tests)
# ---------------------------------------------------------------------------


class MockSpan:
    def __init__(self) -> None:
        self.name: str = ""
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status: Any = None
        self.recorded_exceptions: list[BaseException] = []
        self.ended: bool = False

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: BaseException, **_: Any) -> None:
        self.recorded_exceptions.append(exc)

    def __enter__(self) -> MockSpan:
        return self

    def __exit__(self, *_: Any) -> bool:
        self.ended = True
        return False


class MockTracer:
    def __init__(self) -> None:
        self.span = MockSpan()

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None, **_: Any
    ) -> MockSpan:
        self.span.name = name
        if attributes:
            self.span.attributes.update(attributes)
        return self.span


@pytest.fixture()
def mock_tracer(monkeypatch: pytest.MonkeyPatch) -> MockTracer:
    tracer = MockTracer()
    monkeypatch.setattr("meridian_sdk_provider.fake.get_tracer", lambda: tracer)
    return tracer


@pytest.fixture()
def mock_span(mock_tracer: MockTracer) -> MockSpan:
    return mock_tracer.span


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_DEFAULT_EVENTS: list[dict[str, Any]] = [
    {"type": "message_start", "model": "test-model", "provider": "fake"},
    {"type": "text_delta", "text": "hello"},
    {"type": "message_stop", "stop_reason": "end_turn"},
]


def make_adapter(
    fixtures_dir: Path, audit_log: CollectingAuditLog | None = None
) -> FakeModelAdapter:
    return FakeModelAdapter(fixtures_dir, audit_log=audit_log)


def make_fixture(
    fixtures_dir: Path, model: str, events: list[dict[str, Any]] | None = None
) -> Path:
    slug = model.replace(":", "_").replace("/", "_")
    path = fixtures_dir / f"{slug}.ndjson"
    write_model_fixture(path, events or _DEFAULT_EVENTS)
    return path


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_model_provider_protocol(self, tmp_path: Path) -> None:
        adapter = make_adapter(tmp_path)
        assert isinstance(adapter, ModelProvider)

    def test_name_attribute(self, tmp_path: Path) -> None:
        adapter = FakeModelAdapter(tmp_path, name="my-fake")
        assert adapter.name == "my-fake"

    def test_kind_is_fake(self, tmp_path: Path) -> None:
        adapter = make_adapter(tmp_path)
        assert adapter.kind == "fake"

    def test_capabilities_defaults(self, tmp_path: Path) -> None:
        adapter = make_adapter(tmp_path)
        assert adapter.capabilities.streaming is True
        assert adapter.capabilities.thinking is False
        assert adapter.capabilities.cache_control is False
        assert adapter.capabilities.count_tokens is False


# ---------------------------------------------------------------------------
# call() — happy path
# ---------------------------------------------------------------------------


class TestCallHappyPath:
    async def test_yields_correct_event_count(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        events = [e async for e in adapter.call(opts)]
        assert len(events) == 3

    async def test_yields_typed_events(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        events = [e async for e in adapter.call(opts)]
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[1], TextDeltaEvent)
        assert isinstance(events[2], MessageStopEvent)

    async def test_text_delta_content(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        events = [e async for e in adapter.call(opts)]
        assert isinstance(events[1], TextDeltaEvent)
        assert events[1].text == "hello"

    async def test_replay_is_stateless(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        first = [e async for e in adapter.call(opts)]
        second = [e async for e in adapter.call(opts)]
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert type(a) is type(b)

    async def test_skips_blank_lines(self, tmp_path: Path, mock_span: MockSpan) -> None:
        slug = "test-provider_test-model"
        path = tmp_path / f"{slug}.ndjson"
        path.write_text(
            json.dumps({"type": "message_start", "model": "m", "provider": "fake"})
            + "\n\n"
            + json.dumps({"type": "message_stop", "stop_reason": "end_turn"})
            + "\n",
            encoding="utf-8",
        )
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        events = [e async for e in adapter.call(opts)]
        assert len(events) == 2

    async def test_no_audit_entries_on_success(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        audit = CollectingAuditLog()
        adapter = make_adapter(tmp_path, audit)
        opts = make_opts(model="test-provider:test-model")
        [e async for e in adapter.call(opts)]
        assert audit.entries == []


# ---------------------------------------------------------------------------
# call() — OTel span
# ---------------------------------------------------------------------------


class TestCallOTelSpan:
    async def test_span_name(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        [e async for e in adapter.call(make_opts(model="test-provider:test-model"))]
        assert mock_span.name == "fake.model.call"

    async def test_span_attributes_provider_name(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = FakeModelAdapter(tmp_path, name="acme-fake")
        [e async for e in adapter.call(make_opts(model="test-provider:test-model"))]
        assert mock_span.attributes["provider.name"] == "acme-fake"

    async def test_span_attributes_model(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        [e async for e in adapter.call(make_opts(model="test-provider:test-model"))]
        assert mock_span.attributes["model"] == "test-provider:test-model"

    async def test_invocation_event_attached(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        [e async for e in adapter.call(make_opts(model="test-provider:test-model"))]
        event_names = [e[0] for e in mock_span.events]
        assert "provider.invocation" in event_names

    async def test_invocation_event_provider_kind(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        [e async for e in adapter.call(make_opts(model="test-provider:test-model"))]
        inv = next(e for e in mock_span.events if e[0] == "provider.invocation")
        assert inv[1]["provider.kind"] == "fake"

    async def test_span_ended_on_success(self, tmp_path: Path, mock_span: MockSpan) -> None:
        make_fixture(tmp_path, "test-provider:test-model")
        adapter = make_adapter(tmp_path)
        [e async for e in adapter.call(make_opts(model="test-provider:test-model"))]
        assert mock_span.ended


# ---------------------------------------------------------------------------
# call() — fixture failure
# ---------------------------------------------------------------------------


class TestCallFixtureFailure:
    async def test_raises_on_missing_fixture(self, tmp_path: Path, mock_span: MockSpan) -> None:
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(FileNotFoundError, match="Fixture not found"):
            [e async for e in adapter.call(opts)]

    async def test_raises_on_invalid_json(self, tmp_path: Path, mock_span: MockSpan) -> None:
        slug = "test-provider_test-model"
        (tmp_path / f"{slug}.ndjson").write_text("not-json\n", encoding="utf-8")
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(ValueError, match="invalid ModelEvent"):
            [e async for e in adapter.call(opts)]

    async def test_raises_on_wrong_event_type(self, tmp_path: Path, mock_span: MockSpan) -> None:
        slug = "test-provider_test-model"
        (tmp_path / f"{slug}.ndjson").write_text(
            json.dumps({"type": "unknown_type_xyz"}) + "\n", encoding="utf-8"
        )
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(ValueError, match="invalid ModelEvent"):
            [e async for e in adapter.call(opts)]

    async def test_audit_entry_written_on_missing_fixture(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        audit = CollectingAuditLog()
        adapter = make_adapter(tmp_path, audit)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(FileNotFoundError):
            [e async for e in adapter.call(opts)]
        assert len(audit.entries) == 1
        entry: AuditLogEntry = audit.entries[0]
        assert entry.level == "error"
        assert entry.event == "fake_model.fixture.failed"

    async def test_audit_entry_detail_contains_fixture_path(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        audit = CollectingAuditLog()
        adapter = make_adapter(tmp_path, audit)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(FileNotFoundError):
            [e async for e in adapter.call(opts)]
        assert "fixture" in audit.entries[0].detail

    async def test_span_marked_error_on_missing_fixture(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(FileNotFoundError):
            [e async for e in adapter.call(opts)]
        assert mock_span.status is not None
        assert mock_span.status.status_code == StatusCode.ERROR

    async def test_span_records_exception_on_failure(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(FileNotFoundError) as exc_info:
            [e async for e in adapter.call(opts)]
        assert exc_info.value in mock_span.recorded_exceptions

    async def test_span_ended_on_failure(self, tmp_path: Path, mock_span: MockSpan) -> None:
        adapter = make_adapter(tmp_path)
        opts = make_opts(model="test-provider:test-model")
        with pytest.raises(FileNotFoundError):
            [e async for e in adapter.call(opts)]
        assert mock_span.ended


# ---------------------------------------------------------------------------
# Model slug derivation
# ---------------------------------------------------------------------------


class TestModelSlugDerivation:
    async def test_colon_replaced_with_underscore(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        (tmp_path / "provider_model.ndjson").write_text(
            json.dumps({"type": "message_stop", "stop_reason": "end_turn"}) + "\n",
            encoding="utf-8",
        )
        adapter = make_adapter(tmp_path)
        events = [e async for e in adapter.call(make_opts(model="provider:model"))]
        assert len(events) == 1

    async def test_slash_replaced_with_underscore(
        self, tmp_path: Path, mock_span: MockSpan
    ) -> None:
        (tmp_path / "org_model.ndjson").write_text(
            json.dumps({"type": "message_stop", "stop_reason": "end_turn"}) + "\n",
            encoding="utf-8",
        )
        adapter = make_adapter(tmp_path)
        events = [e async for e in adapter.call(make_opts(model="org/model"))]
        assert len(events) == 1


# ---------------------------------------------------------------------------
# count_tokens and close
# ---------------------------------------------------------------------------


class TestCountTokensAndClose:
    async def test_count_tokens_returns_zero_input(self, tmp_path: Path) -> None:
        from meridian_sdk_provider import ModelCountReq

        adapter = make_adapter(tmp_path)
        req = ModelCountReq(model="fake:m", messages=[{"role": "user", "content": "hi"}])
        result = await adapter.count_tokens(req)
        assert result.input_tokens == 0

    async def test_close_is_noop(self, tmp_path: Path) -> None:
        adapter = make_adapter(tmp_path)
        await adapter.close()


# ---------------------------------------------------------------------------
# write_model_fixture helper
# ---------------------------------------------------------------------------


class TestWriteModelFixture:
    def test_creates_ndjson_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.ndjson"
        write_model_fixture(path, [{"type": "message_stop", "stop_reason": "end_turn"}])
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "out.ndjson"
        write_model_fixture(path, [{"type": "message_stop", "stop_reason": "end_turn"}])
        assert path.exists()

    def test_each_event_on_own_line(self, tmp_path: Path) -> None:
        path = tmp_path / "out.ndjson"
        write_model_fixture(
            path,
            [
                {"type": "message_start", "model": "m", "provider": "fake"},
                {"type": "message_stop", "stop_reason": "end_turn"},
            ],
        )
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "message_start"
        assert json.loads(lines[1])["type"] == "message_stop"
