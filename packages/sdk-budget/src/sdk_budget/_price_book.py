"""PriceBook — per-provider pricing config loaded from a JSON file.

JSON schema (dollars per 1 000 tokens, by provider → model → direction):

    {
        "openai": {
            "gpt-4o": {
                "input": 2.50,
                "output": 10.00,
                "cache_creation": 3.75,
                "cache_read": 1.25
            }
        }
    }

``cache_creation`` and ``cache_read`` are optional; both default to 0.0.
Missing provider/model entries return 0.0 cost (unknown model ≠ error).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    """Per-direction rates for one model, in dollars per 1 000 tokens."""

    input: float
    output: float
    cache_creation: float = 0.0
    cache_read: float = 0.0


class PriceBook:
    """Pricing config indexed by (provider, model).

    Unknown (provider, model) pairs return zero cost; callers that need
    to distinguish "priced" from "unpriced" should call ``has_pricing``.
    """

    def __init__(self, entries: dict[str, dict[str, ModelPricing]]) -> None:
        self._entries = entries

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> PriceBook:
        """Load from *path* (must be a UTF-8 JSON file)."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriceBook:
        """Build from a nested dict with the schema described above."""
        entries: dict[str, dict[str, ModelPricing]] = {}
        for provider, models in data.items():
            if not isinstance(models, dict):
                raise ValueError(
                    f"PriceBook: provider {provider!r} must map to an object, got {type(models).__name__}"
                )
            entries[provider] = {}
            for model, rates in models.items():
                if not isinstance(rates, dict):
                    raise ValueError(
                        f"PriceBook: {provider}/{model} must be an object, got {type(rates).__name__}"
                    )
                if "input" not in rates or "output" not in rates:
                    raise ValueError(
                        f"PriceBook: {provider}/{model} must have 'input' and 'output' keys"
                    )
                entries[provider][model] = ModelPricing(
                    input=float(rates["input"]),
                    output=float(rates["output"]),
                    cache_creation=float(rates.get("cache_creation", 0.0)),
                    cache_read=float(rates.get("cache_read", 0.0)),
                )
        return cls(entries)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def has_pricing(self, provider: str, model: str) -> bool:
        """Return True if (provider, model) has an entry."""
        return model in self._entries.get(provider, {})

    def pricing_for(self, provider: str, model: str) -> ModelPricing | None:
        """Return ``ModelPricing`` for (provider, model), or ``None`` if unknown."""
        return self._entries.get(provider, {}).get(model)

    def cost_for_delta(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Compute total dollar cost for one usage.delta.

        Returns 0.0 if (provider, model) is not in the price book.
        All token counts are divided by 1 000 before multiplying by the rate.
        """
        pricing = self.pricing_for(provider, model)
        if pricing is None:
            return 0.0
        return (
            prompt_tokens * pricing.input / 1_000
            + completion_tokens * pricing.output / 1_000
            + cache_creation_tokens * pricing.cache_creation / 1_000
            + cache_read_tokens * pricing.cache_read / 1_000
        )
