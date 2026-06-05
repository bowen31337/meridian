"""
SIGHUP signal handler conformance suite.

Tests cover:
  - _do_reload: emits OTel span "system.config.reload" on every invocation.
  - _do_reload: span carries structured invocation event with code "system_config_reload".
  - _do_reload: writes "system.config.reload.ok" audit entry on success.
  - _do_reload: success audit entry level is "info" and detail has provider_count.
  - _do_reload: updates model_router routing policy on success.
  - _do_reload: hot-swaps registry providers on success when registry is present.
  - _do_reload: success with no registry skips swap_all and still writes ok entry.
  - _do_reload: parse failure writes "system.config.reload.failed" with stage "parse".
  - _do_reload: parse failure sets span status to ERROR.
  - _do_reload: parse failure leaves old config in effect (no swap, no policy update).
  - _do_reload: validate failure writes "system.config.reload.failed" with stage "validate".
  - _do_reload: validate failure detail contains errors list.
  - _do_reload: validate failure sets span status to ERROR.
  - _do_reload: validate failure leaves old config in effect (no swap, no policy update).
  - _do_reload: provider_build failure writes "system.config.reload.failed" with stage
    "provider_build".
  - _do_reload: provider_build failure sets span status to ERROR.
  - _do_reload: provider_build failure leaves routing policy unchanged.
  - install_sighup_handler: registers a handler on the event loop via add_signal_handler.
  - remove_sighup_handler: unregisters the handler via remove_signal_handler.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import signal
from unittest.mock import AsyncMock, MagicMock, patch

from core_errors import AuditLog, AuditLogEntry, NoopAuditLog
from meridian_sdk_provider import ModelRouter, ModelRoutingPolicy, ProviderRegistry
from meridiand._sighup import _do_reload, install_sighup_handler, remove_sighup_handler
from opentelemetry.trace import StatusCode
import yaml

from tests._otel_shared import otel_exporter as _otel_exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingAuditLog(AuditLog):
    def __init__(self) -> None:
        self.entries: list[AuditLogEntry] = []

    def write(self, entry: AuditLogEntry) -> None:
        self.entries.append(entry)


def _write_cfg(tmp_path: Path, **extra: object) -> Path:
    cfg = tmp_path / "config.yaml"
    data: dict[str, object] = {
        "version": 2,
        "storage_root": str(tmp_path / "storage"),
    }
    data.update(extra)
    cfg.write_text(yaml.dump(data))
    return cfg


def _make_router(registry: ProviderRegistry | None = None) -> ModelRouter:
    return ModelRouter(
        policy=ModelRoutingPolicy(rules=[], fallbacks=[]),
        registry=registry,
    )


# ---------------------------------------------------------------------------
# TestDoReloadSuccess
# ---------------------------------------------------------------------------


class TestDoReloadSuccess:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_emits_otel_span(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        spans = _otel_exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert "system.config.reload" in names

    async def test_span_has_invocation_event(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        spans = _otel_exporter.get_finished_spans()
        reload_span = next(s for s in spans if s.name == "system.config.reload")
        event_names = [e.name for e in reload_span.events]
        assert "meridian.error.invocation" in event_names

    async def test_writes_ok_audit_entry(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        ok_entries = [e for e in audit.entries if e.event == "system.config.reload.ok"]
        assert len(ok_entries) == 1

    async def test_ok_entry_level_is_info(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        entry = next(e for e in audit.entries if e.event == "system.config.reload.ok")
        assert entry.level == "info"

    async def test_ok_entry_detail_has_provider_count(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        entry = next(e for e in audit.entries if e.event == "system.config.reload.ok")
        assert "provider_count" in (entry.detail or {})

    async def test_updates_routing_policy(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router()
        old_policy = router._policy
        await _do_reload(cfg, router, audit, None)
        # Policy object is replaced (new instance) even for empty routing.
        assert router._policy is not old_policy

    async def test_no_registry_skips_swap_and_writes_ok(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router(registry=None)
        await _do_reload(cfg, router, audit, None)
        ok_entries = [e for e in audit.entries if e.event == "system.config.reload.ok"]
        assert len(ok_entries) == 1
        assert ok_entries[0].detail == {"provider_count": 0}

    async def test_registry_swap_all_called_with_providers(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "local1", "kind": "ollama"}],
        )
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)

        fake_provider = MagicMock()
        with (
            patch("meridiand._sighup._resolve_auth", return_value=None),
            patch("meridiand._sighup._build_provider", return_value=fake_provider),
        ):
            await _do_reload(cfg, router, audit, None)

        registry.swap_all.assert_awaited_once()
        swapped = registry.swap_all.call_args[0][0]
        assert "local1" in swapped

    async def test_ok_entry_provider_count_matches_providers(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "local1", "kind": "ollama"}],
        )
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)

        fake_provider = MagicMock()
        with (
            patch("meridiand._sighup._resolve_auth", return_value=None),
            patch("meridiand._sighup._build_provider", return_value=fake_provider),
        ):
            await _do_reload(cfg, router, audit, None)

        entry = next(e for e in audit.entries if e.event == "system.config.reload.ok")
        assert entry.detail == {"provider_count": 1}


# ---------------------------------------------------------------------------
# TestDoReloadParseFailure
# ---------------------------------------------------------------------------


class TestDoReloadParseFailure:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_bad_yaml_writes_failed_entry(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(": bad: yaml: [\n")
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        failed = [e for e in audit.entries if e.event == "system.config.reload.failed"]
        assert len(failed) >= 1

    async def test_bad_yaml_entry_stage_is_parse(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(": bad: yaml: [\n")
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        failed = next(e for e in audit.entries if e.event == "system.config.reload.failed")
        assert (failed.detail or {}).get("stage") == "parse"

    async def test_bad_yaml_sets_span_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(": bad: yaml: [\n")
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        spans = _otel_exporter.get_finished_spans()
        reload_span = next(s for s in spans if s.name == "system.config.reload")
        assert reload_span.status.status_code == StatusCode.ERROR

    async def test_parse_failure_no_policy_update(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(": bad: yaml: [\n")
        audit = _CapturingAuditLog()
        router = _make_router()
        original_policy = router._policy
        await _do_reload(cfg, router, audit, None)
        assert router._policy is original_policy

    async def test_parse_failure_no_swap(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(": bad: yaml: [\n")
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)
        await _do_reload(cfg, router, audit, None)
        registry.swap_all.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestDoReloadValidateFailure
# ---------------------------------------------------------------------------


class TestDoReloadValidateFailure:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_invalid_provider_kind_writes_failed_entry(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "bad", "kind": "unknown_kind"}],
        )
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        failed = [e for e in audit.entries if e.event == "system.config.reload.failed"]
        assert len(failed) >= 1

    async def test_validate_failure_entry_stage_is_validate(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "bad", "kind": "unknown_kind"}],
        )
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        failed = next(e for e in audit.entries if e.event == "system.config.reload.failed")
        assert (failed.detail or {}).get("stage") == "validate"

    async def test_validate_failure_detail_has_errors(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "bad", "kind": "unknown_kind"}],
        )
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        failed = next(e for e in audit.entries if e.event == "system.config.reload.failed")
        assert "errors" in (failed.detail or {})

    async def test_validate_failure_sets_span_error(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "bad", "kind": "unknown_kind"}],
        )
        audit = _CapturingAuditLog()
        router = _make_router()
        await _do_reload(cfg, router, audit, None)
        spans = _otel_exporter.get_finished_spans()
        reload_span = next(s for s in spans if s.name == "system.config.reload")
        assert reload_span.status.status_code == StatusCode.ERROR

    async def test_validate_failure_no_swap(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "bad", "kind": "unknown_kind"}],
        )
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)
        await _do_reload(cfg, router, audit, None)
        registry.swap_all.assert_not_awaited()

    async def test_validate_failure_no_policy_update(self, tmp_path: Path) -> None:
        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "bad", "kind": "unknown_kind"}],
        )
        audit = _CapturingAuditLog()
        router = _make_router()
        original_policy = router._policy
        await _do_reload(cfg, router, audit, None)
        assert router._policy is original_policy


# ---------------------------------------------------------------------------
# TestDoReloadProviderBuildFailure
# ---------------------------------------------------------------------------


class TestDoReloadProviderBuildFailure:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_provider_build_error_writes_failed_entry(self, tmp_path: Path) -> None:
        from meridiand._provider_factory import ProviderFactoryError

        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "local1", "kind": "ollama"}],
        )
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)

        exc = ProviderFactoryError(message="build failed", timestamp="2026-01-01T00:00:00+00:00")
        with (
            patch("meridiand._sighup._resolve_auth", return_value=None),
            patch("meridiand._sighup._build_provider", side_effect=exc),
        ):
            await _do_reload(cfg, router, audit, None)

        failed = [e for e in audit.entries if e.event == "system.config.reload.failed"]
        assert len(failed) >= 1

    async def test_provider_build_error_stage_is_provider_build(self, tmp_path: Path) -> None:
        from meridiand._provider_factory import ProviderFactoryError

        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "local1", "kind": "ollama"}],
        )
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)

        exc = ProviderFactoryError(message="build failed", timestamp="2026-01-01T00:00:00+00:00")
        with (
            patch("meridiand._sighup._resolve_auth", return_value=None),
            patch("meridiand._sighup._build_provider", side_effect=exc),
        ):
            await _do_reload(cfg, router, audit, None)

        failed = next(e for e in audit.entries if e.event == "system.config.reload.failed")
        assert (failed.detail or {}).get("stage") == "provider_build"

    async def test_provider_build_error_sets_span_error(self, tmp_path: Path) -> None:
        from meridiand._provider_factory import ProviderFactoryError

        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "local1", "kind": "ollama"}],
        )
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)

        exc = ProviderFactoryError(message="build failed", timestamp="2026-01-01T00:00:00+00:00")
        with (
            patch("meridiand._sighup._resolve_auth", return_value=None),
            patch("meridiand._sighup._build_provider", side_effect=exc),
        ):
            await _do_reload(cfg, router, audit, None)

        spans = _otel_exporter.get_finished_spans()
        reload_span = next(s for s in spans if s.name == "system.config.reload")
        assert reload_span.status.status_code == StatusCode.ERROR

    async def test_provider_build_error_no_policy_update(self, tmp_path: Path) -> None:
        from meridiand._provider_factory import ProviderFactoryError

        cfg = _write_cfg(
            tmp_path,
            providers=[{"name": "local1", "kind": "ollama"}],
        )
        audit = _CapturingAuditLog()
        registry = MagicMock(spec=ProviderRegistry)
        registry.swap_all = AsyncMock()
        router = _make_router(registry=registry)
        original_policy = router._policy

        exc = ProviderFactoryError(message="build failed", timestamp="2026-01-01T00:00:00+00:00")
        with (
            patch("meridiand._sighup._resolve_auth", return_value=None),
            patch("meridiand._sighup._build_provider", side_effect=exc),
        ):
            await _do_reload(cfg, router, audit, None)

        assert router._policy is original_policy


# ---------------------------------------------------------------------------
# TestSignalHandlerRegistration
# ---------------------------------------------------------------------------


class TestSignalHandlerRegistration:
    def setup_method(self) -> None:
        _otel_exporter.clear()

    async def test_install_calls_add_signal_handler(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        router = _make_router()
        audit = NoopAuditLog()

        loop = asyncio.get_event_loop()
        registered: list[tuple] = []
        original = loop.add_signal_handler

        def _capture(sig, cb):
            registered.append((sig, cb))
            original(sig, cb)

        loop.add_signal_handler = _capture  # type: ignore[method-assign]
        try:
            install_sighup_handler(
                config_path=cfg,
                model_router=router,
                audit_log=audit,
            )
            assert any(sig == signal.SIGHUP for sig, _ in registered)
        finally:
            loop.add_signal_handler = original  # type: ignore[method-assign]
            loop.remove_signal_handler(signal.SIGHUP)

    async def test_remove_calls_remove_signal_handler(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        router = _make_router()
        audit = NoopAuditLog()

        loop = asyncio.get_event_loop()
        install_sighup_handler(
            config_path=cfg,
            model_router=router,
            audit_log=audit,
        )

        removed: list[int] = []
        original = loop.remove_signal_handler

        def _capture(sig):
            removed.append(sig)
            return original(sig)

        loop.remove_signal_handler = _capture  # type: ignore[method-assign]
        try:
            remove_sighup_handler()
            assert signal.SIGHUP in removed
        finally:
            loop.remove_signal_handler = original  # type: ignore[method-assign]

    async def test_install_schedules_reload_task_on_signal(self, tmp_path: Path) -> None:
        cfg = _write_cfg(tmp_path)
        audit = _CapturingAuditLog()
        router = _make_router()

        loop = asyncio.get_event_loop()
        captured_cb: list = []
        original_add = loop.add_signal_handler

        def _capture(sig, cb):
            captured_cb.append(cb)
            original_add(sig, cb)

        loop.add_signal_handler = _capture  # type: ignore[method-assign]
        try:
            install_sighup_handler(
                config_path=cfg,
                model_router=router,
                audit_log=audit,
            )
            assert len(captured_cb) == 1
            captured_cb[0]()  # invoke the registered callback directly
            # drain: the callback creates a task; yield until it finishes
            pending = asyncio.all_tasks(loop) - {asyncio.current_task()}
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            ok_entries = [e for e in audit.entries if e.event == "system.config.reload.ok"]
            assert len(ok_entries) == 1
        finally:
            loop.add_signal_handler = original_add  # type: ignore[method-assign]
            loop.remove_signal_handler(signal.SIGHUP)
