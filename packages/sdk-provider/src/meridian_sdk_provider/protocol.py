from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .types import ModelCallOpts, ModelCountReq, ModelEvent, TokenCount


@dataclass
class ModelCapabilities:
    """Per-model capability flags surfaced by the GET /v1/models endpoint."""

    streaming: bool = True
    thinking: bool = False
    vision: bool = False
    tools: bool = False
    cache: bool = False


@dataclass
class ModelEntry:
    """A single model advertised by a provider."""

    provider: str
    model: str
    context_window: int
    capabilities: ModelCapabilities


@dataclass
class ProviderCapabilities:
    """Optional feature flags declared by a provider.

    The Router reads these before every call and strips opts that the provider
    cannot handle (e.g., clears ``stream=True`` when ``streaming=False``,
    removes ``cache_control`` headers when ``cache_control=False``).
    """

    streaming: bool = True
    thinking: bool = False
    cache_control: bool = False
    count_tokens: bool = False


@runtime_checkable
class ModelProvider(Protocol):
    """Contract that every model-provider adapter must satisfy.

    Providers are instantiated from the YAML config and registered in the
    ModelRouter by ``name``.  Multiple instances of the same ``kind`` are
    allowed (e.g., ``anthropic-oauth`` and ``anthropic-api`` are both
    ``kind="anthropic"``).

    Capability hints are declared via ``capabilities``; the Router enforces
    them — adapters must not rely on callers stripping unsupported opts.
    """

    name: str
    kind: str
    capabilities: ProviderCapabilities

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        """Stream model events for a single call.

        Implementors write this as an async generator::

            async def call(self, opts):
                async with client.stream(...) as stream:
                    async for raw in stream:
                        yield translate(raw)

        The caller iterates with ``async for event in provider.call(opts)``.
        """
        ...  # pragma: no cover

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        """Return a token count for the request without executing the call.

        Only called by the Router when ``capabilities.count_tokens`` is True.
        Providers that set ``count_tokens=False`` may raise NotImplementedError.
        """
        ...  # pragma: no cover

    def list_models(self) -> list[ModelEntry]:
        """Return the set of models this provider can serve.

        Each entry carries per-model capability flags and the context window
        size.  The ModelRouter aggregates these lists to populate GET /v1/models.
        """
        ...  # pragma: no cover

    async def close(self) -> None:
        """Release any resources held by the adapter (connections, subprocesses)."""
        ...  # pragma: no cover
