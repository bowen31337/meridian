"""Shared fixtures for meridian-cli tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from meridian_cli._client import DaemonClient
import pytest


@pytest.fixture()
def mock_client() -> MagicMock:
    """A MagicMock that satisfies the DaemonClient interface."""
    client = MagicMock(spec=DaemonClient)
    client.request.return_value = {"id": "abc123", "name": "test"}
    return client


@pytest.fixture(autouse=True)
def _otel_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch OTel tracer in both _resource and files modules so no SDK is required."""
    mock_span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = lambda *_: mock_span
    tracer.start_as_current_span.return_value.__exit__ = lambda *_: False

    monkeypatch.setattr("meridian_cli._resource.get_tracer", lambda: tracer)
    monkeypatch.setattr("meridian_cli.files.get_tracer", lambda: tracer)
