"""Load-balanced provider pool.

Wraps N member providers of the same kind (e.g. several API keys for one
endpoint) behind a single provider name. Each ``call()`` starts at the next
member in round-robin order — spreading load across the pool — and on a
pre-stream failure rotates through the remaining members so one exhausted or
rate-limited key never drops the call. Once a member streams its first event the
pool commits to it (a mid-stream failure is surfaced, not re-tried), matching the
ModelRouter's commit-on-first-event semantics.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from itertools import count

from .protocol import ModelEntry, ModelProvider
from .types import ModelCallOpts, ModelCountReq, ModelEvent, TokenCount


class LoadBalancedProvider:
    """Round-robin load balancer over a list of same-kind member providers."""

    def __init__(self, *, name: str, kind: str, members: Sequence[ModelProvider]) -> None:
        if not members:
            raise ValueError("LoadBalancedProvider requires at least one member")
        self.name = name
        self.kind = kind
        self._members: list[ModelProvider] = list(members)
        # All members share an endpoint/kind, so their capabilities match.
        self.capabilities = self._members[0].capabilities
        self._counter = count()

    def _start_index(self) -> int:
        return next(self._counter) % len(self._members)

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        n = len(self._members)
        start = self._start_index()
        last_exc: Exception | None = None
        for offset in range(n):
            member = self._members[(start + offset) % n]
            gen = member.call(opts)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                return
            except Exception as exc:  # noqa: BLE001 - rotate to the next member
                last_exc = exc
                continue
            # This member streamed — commit to it for the rest of the call.
            yield first
            async for event in gen:
                yield event
            return
        # Every member failed before streaming; surface the last error.
        assert last_exc is not None
        raise last_exc

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        return await self._members[0].count_tokens(req)

    def list_models(self) -> list[ModelEntry]:
        return self._members[0].list_models()

    async def close(self) -> None:
        for member in self._members:
            await member.close()
