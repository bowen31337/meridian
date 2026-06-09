"""
Tests for the in-daemon ChannelRuntime factory (_channel_factory).

Covers:
  - All five v1 channel drivers registered (cli / telegram / slack / discord /
    webhook), capabilities reachable per kind.
  - SecretRefResolver -> channel SecretResolver adapter: returns the resolved
    value, and returns None (instead of raising) when the inner resolver fails.
  - Bundle wiring: drivers receive the adapter; resolver is None when no inner
    resolver is provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from meridiand._audit import FileAuditLog
from meridiand._channel_factory import (
    ChannelRuntimeBundle,
    _SecretRefChannelResolver,
    build_channel_runtime,
    start_configured_channels,
    stop_channels,
)
from meridiand._secret_ref import SecretRefResolver
from meridiand._telegram_channel_driver import TelegramLongPollClient
import pytest
from sdk_channel import (
    ChannelCapabilities,
    ChannelDriver,
    ChannelRuntime,
    SendRequest,
    SendResult,
    StartRequest,
    StopRequest,
)

_V1_KINDS = [
    "meridian.cli",
    "meridian.telegram",
    "meridian.slack",
    "meridian.discord",
    "meridian.webhook",
]


def _audit(tmp_path: Path) -> FileAuditLog:
    return FileAuditLog(tmp_path)


class TestBuildChannelRuntime:
    def test_registers_all_v1_kinds(self, tmp_path: Path) -> None:
        bundle = build_channel_runtime(storage_root=tmp_path, audit_log=_audit(tmp_path))
        for kind in _V1_KINDS:
            assert bundle.runtime.get(kind) is not None, f"{kind} not registered"

    def test_capabilities_reachable_per_kind(self, tmp_path: Path) -> None:
        bundle = build_channel_runtime(storage_root=tmp_path, audit_log=_audit(tmp_path))
        for kind in _V1_KINDS:
            caps = bundle.runtime.capabilities(kind)
            assert isinstance(caps, ChannelCapabilities)

    def test_returns_bundle(self, tmp_path: Path) -> None:
        bundle = build_channel_runtime(storage_root=tmp_path, audit_log=_audit(tmp_path))
        assert isinstance(bundle, ChannelRuntimeBundle)

    def test_secret_resolver_none_when_not_provided(self, tmp_path: Path) -> None:
        bundle = build_channel_runtime(storage_root=tmp_path, audit_log=_audit(tmp_path))
        assert bundle.secret_resolver is None

    def test_secret_resolver_present_when_provided(self, tmp_path: Path) -> None:
        inner = SecretRefResolver(storage_root=tmp_path)
        bundle = build_channel_runtime(
            storage_root=tmp_path, audit_log=_audit(tmp_path), secret_resolver=inner
        )
        assert isinstance(bundle.secret_resolver, _SecretRefChannelResolver)

    def test_driver_receives_adapter(self, tmp_path: Path) -> None:
        inner = SecretRefResolver(storage_root=tmp_path)
        bundle = build_channel_runtime(
            storage_root=tmp_path, audit_log=_audit(tmp_path), secret_resolver=inner
        )
        telegram = bundle.runtime.get("meridian.telegram")
        assert telegram is not None
        assert isinstance(telegram._resolver, _SecretRefChannelResolver)  # type: ignore[attr-defined]


class TestSecretRefChannelResolver:
    def test_returns_resolved_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        inner = SecretRefResolver(storage_root=tmp_path)
        monkeypatch.setattr(inner, "resolve", lambda ref: "the-secret")
        adapter = _SecretRefChannelResolver(inner)
        assert adapter.resolve("secret_ref://vault/x/y") == "the-secret"

    def test_returns_none_when_inner_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = SecretRefResolver(storage_root=tmp_path)

        def _boom(ref: str) -> str:
            raise RuntimeError("unresolvable")

        monkeypatch.setattr(inner, "resolve", _boom)
        adapter = _SecretRefChannelResolver(inner)
        assert adapter.resolve("secret_ref://vault/x/y") is None


class _RecordingSink:
    async def dispatch(
        self, *, channel_id: str, sender_id: str, content: str, content_type: str
    ) -> None:
        pass


class TestInboundSinkWiring:
    def test_no_factory_when_sink_absent(self, tmp_path: Path) -> None:
        bundle = build_channel_runtime(storage_root=tmp_path, audit_log=_audit(tmp_path))
        telegram = bundle.runtime.get("meridian.telegram")
        assert telegram is not None
        assert telegram._long_poll_client_factory is None  # type: ignore[attr-defined]

    def test_factory_wired_when_sink_present(self, tmp_path: Path) -> None:
        bundle = build_channel_runtime(
            storage_root=tmp_path, audit_log=_audit(tmp_path), inbound_sink=_RecordingSink()
        )
        telegram = bundle.runtime.get("meridian.telegram")
        assert telegram is not None
        factory = telegram._long_poll_client_factory  # type: ignore[attr-defined]
        assert factory is not None
        client = factory("ch_abc")
        assert isinstance(client, TelegramLongPollClient)
        assert client._channel_id == "ch_abc"  # type: ignore[attr-defined]


class _StubDriver(ChannelDriver):
    def __init__(self, kind: str, *, fail: bool = False) -> None:
        self._kind = kind
        self._fail = fail
        self.started: list[str] = []
        self.stopped: list[str] = []

    @property
    def kind(self) -> str:
        return self._kind

    async def start(self, request: StartRequest) -> None:
        if self._fail:
            raise RuntimeError("start boom")
        self.started.append(request.channel_id)

    async def send(self, request: SendRequest) -> SendResult:
        return SendResult(message_id="m", timestamp="t", delivered=True)

    async def stop(self, request: StopRequest) -> None:
        if self._fail:
            raise RuntimeError("stop boom")
        self.stopped.append(request.channel_id)

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities()


def _write_channel(storage_root: Path, channel_id: str, kind: str) -> None:
    channels_dir = storage_root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"id": channel_id, "kind": kind, "config": {}}
    (channels_dir / f"{channel_id}.json").write_text(json.dumps(record))


class TestStartConfiguredChannels:
    async def test_no_channels_dir_returns_empty(self, tmp_path: Path) -> None:
        runtime = ChannelRuntime()
        started = await start_configured_channels(
            runtime=runtime, storage_root=tmp_path, audit_log=_audit(tmp_path)
        )
        assert started == []

    async def test_starts_each_channel(self, tmp_path: Path) -> None:
        driver = _StubDriver("meridian.stub")
        runtime = ChannelRuntime()
        runtime.register(driver)
        _write_channel(tmp_path, "ch_a", "meridian.stub")
        _write_channel(tmp_path, "ch_b", "meridian.stub")

        started = await start_configured_channels(
            runtime=runtime, storage_root=tmp_path, audit_log=_audit(tmp_path)
        )

        assert sorted(started) == [("ch_a", "meridian.stub"), ("ch_b", "meridian.stub")]
        assert sorted(driver.started) == ["ch_a", "ch_b"]

    async def test_failure_is_audited_and_does_not_abort(self, tmp_path: Path) -> None:
        driver = _StubDriver("meridian.stub", fail=True)
        runtime = ChannelRuntime()
        runtime.register(driver)
        _write_channel(tmp_path, "ch_a", "meridian.stub")
        audit = _audit(tmp_path)

        started = await start_configured_channels(
            runtime=runtime, storage_root=tmp_path, audit_log=audit
        )

        assert started == []
        audit_text = (tmp_path / "audit.ndjson").read_text()
        assert "channel.autostart.failed" in audit_text


class TestStopChannels:
    async def test_stops_each(self, tmp_path: Path) -> None:
        driver = _StubDriver("meridian.stub")
        runtime = ChannelRuntime()
        runtime.register(driver)
        await stop_channels(
            runtime=runtime,
            started=[("ch_a", "meridian.stub"), ("ch_b", "meridian.stub")],
            audit_log=_audit(tmp_path),
        )
        assert sorted(driver.stopped) == ["ch_a", "ch_b"]

    async def test_failure_is_audited(self, tmp_path: Path) -> None:
        driver = _StubDriver("meridian.stub", fail=True)
        runtime = ChannelRuntime()
        runtime.register(driver)
        audit = _audit(tmp_path)
        await stop_channels(runtime=runtime, started=[("ch_a", "meridian.stub")], audit_log=audit)
        audit_text = (tmp_path / "audit.ndjson").read_text()
        assert "channel.autostop.failed" in audit_text


class TestLifespanAutostart:
    def test_app_lifespan_starts_and_stops_channels(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient
        from meridiand._app import create_app

        driver = _StubDriver("meridian.stub")
        runtime = ChannelRuntime()
        runtime.register(driver)
        _write_channel(tmp_path, "ch_a", "meridian.stub")

        app = create_app(FileAuditLog(tmp_path), storage_root=tmp_path, channel_runtime=runtime)
        with TestClient(app):
            pass  # entering runs startup (autostart), exiting runs shutdown (stop)

        assert driver.started == ["ch_a"]
        assert driver.stopped == ["ch_a"]
