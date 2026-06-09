"""
Tests for the sdk-provider -> core_errors audit bridge in _provider_factory.

The ModelRouter/ProviderRegistry write meridian_sdk_provider AuditLogEntry
objects (no `code` field); the bridge adapts them to the core_errors audit sink.
"""

from __future__ import annotations

from typing import Any

from meridian_sdk_provider.audit import AuditLogEntry as SdkAuditLogEntry
from meridiand._provider_factory import _bridge_audit, _SdkProviderAuditBridge


class _Recording:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    def write(self, entry: Any) -> None:
        self.entries.append(entry)


class TestSdkProviderAuditBridge:
    def test_maps_error_entry(self) -> None:
        rec = _Recording()
        _SdkProviderAuditBridge(rec).write(
            SdkAuditLogEntry(
                level="error",
                event="provider.call.failed",
                provider_name="claude",
                provider_kind="claude_code_oauth",
                model="claude-opus-4-7",
                session_id="sess_1",
                timestamp="2026-01-01T00:00:00+00:00",
                detail={"error_type": "CliSubprocessError", "error": "boom"},
            )
        )
        entry = rec.entries[0]
        assert entry.level == "error"
        assert entry.event == "provider.call.failed"
        assert entry.code == "CliSubprocessError"  # synthesized from error_type
        assert entry.detail["provider_name"] == "claude"
        assert entry.detail["model"] == "claude-opus-4-7"
        assert entry.detail["session_id"] == "sess_1"

    def test_warning_level_and_code_fallback(self) -> None:
        rec = _Recording()
        _SdkProviderAuditBridge(rec).write(
            SdkAuditLogEntry(
                level="warning",
                event="router.failover",
                provider_name="p",
                provider_kind="k",
                timestamp="t",
                detail={},
            )
        )
        entry = rec.entries[0]
        assert entry.level == "warn"  # "warning" -> "warn"
        assert entry.code == "router_failover"  # event with dots -> underscores
        assert entry.detail["provider_name"] == "p"


class TestBridgeAudit:
    def test_none_passthrough(self) -> None:
        assert _bridge_audit(None) is None

    def test_wraps_non_none(self) -> None:
        bridge = _bridge_audit(_Recording())
        assert isinstance(bridge, _SdkProviderAuditBridge)
