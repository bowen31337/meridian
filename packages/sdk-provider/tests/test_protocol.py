"""Tests for ModelProvider Protocol conformance and ProviderCapabilities defaults."""

from __future__ import annotations

from meridian_sdk_provider import ModelProvider, ProviderCapabilities
from tests.conftest import FakeProvider


def test_fake_provider_satisfies_protocol() -> None:
    provider = FakeProvider(name="p", kind="fake")
    assert isinstance(provider, ModelProvider)


def test_provider_capabilities_defaults() -> None:
    caps = ProviderCapabilities()
    assert caps.streaming is True
    assert caps.thinking is False
    assert caps.cache_control is False
    assert caps.count_tokens is False


def test_provider_capabilities_custom() -> None:
    caps = ProviderCapabilities(
        streaming=False, thinking=True, cache_control=True, count_tokens=True
    )
    assert caps.streaming is False
    assert caps.thinking is True
    assert caps.cache_control is True
    assert caps.count_tokens is True


def test_protocol_requires_name_and_kind() -> None:
    provider = FakeProvider(name="my-provider", kind="openai")
    assert provider.name == "my-provider"
    assert provider.kind == "openai"


def test_protocol_exposes_capabilities() -> None:
    caps = ProviderCapabilities(thinking=True)
    provider = FakeProvider(name="p", capabilities=caps)
    assert provider.capabilities.thinking is True
