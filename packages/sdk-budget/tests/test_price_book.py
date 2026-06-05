"""
PriceBook conformance suite.

Covers:

  from_dict():
    - Parses provider → model → direction rates.
    - cache_creation and cache_read default to 0.0 when absent.
    - Raises ValueError when a provider entry is not a dict.
    - Raises ValueError when a model entry is not a dict.
    - Raises ValueError when input or output key is missing.

  has_pricing():
    - Returns True for a known (provider, model) pair.
    - Returns False for an unknown provider.
    - Returns False for an unknown model under a known provider.

  pricing_for():
    - Returns ModelPricing for a known pair.
    - Returns None for an unknown pair.

  cost_for_delta():
    - Returns 0.0 for unknown (provider, model).
    - Charges prompt_tokens at the input rate / 1 000.
    - Charges completion_tokens at the output rate / 1 000.
    - Charges cache_creation_tokens at the cache_creation rate / 1 000.
    - Charges cache_read_tokens at the cache_read rate / 1 000.
    - Sums all four directions in one call.
    - Zero tokens yields 0.0 cost.
    - Fractional token counts are handled correctly.

  from_file():
    - Loads the same config as from_dict() given an equivalent JSON file.
    - Raises FileNotFoundError for a missing path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sdk_budget import ModelPricing, PriceBook

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


SAMPLE_DICT = {
    "openai": {
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00,
            "cache_creation": 3.75,
            "cache_read": 1.25,
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50,
        },
    },
    "anthropic": {
        "claude-3-5-sonnet": {
            "input": 3.00,
            "output": 15.00,
            "cache_creation": 3.75,
            "cache_read": 0.30,
        },
    },
}


def book() -> PriceBook:
    return PriceBook.from_dict(SAMPLE_DICT)


# ---------------------------------------------------------------------------
# from_dict — construction
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_parses_input_rate(self) -> None:
        b = book()
        assert b.pricing_for("openai", "gpt-4o") is not None
        assert b.pricing_for("openai", "gpt-4o").input == 2.50  # type: ignore[union-attr]

    def test_parses_output_rate(self) -> None:
        b = book()
        assert b.pricing_for("openai", "gpt-4o").output == 10.00  # type: ignore[union-attr]

    def test_parses_cache_creation_rate(self) -> None:
        b = book()
        assert b.pricing_for("openai", "gpt-4o").cache_creation == 3.75  # type: ignore[union-attr]

    def test_parses_cache_read_rate(self) -> None:
        b = book()
        assert b.pricing_for("openai", "gpt-4o").cache_read == 1.25  # type: ignore[union-attr]

    def test_cache_creation_defaults_to_zero(self) -> None:
        b = book()
        p = b.pricing_for("openai", "gpt-3.5-turbo")
        assert p is not None
        assert p.cache_creation == 0.0

    def test_cache_read_defaults_to_zero(self) -> None:
        b = book()
        p = b.pricing_for("openai", "gpt-3.5-turbo")
        assert p is not None
        assert p.cache_read == 0.0

    def test_raises_when_provider_value_is_not_dict(self) -> None:
        with pytest.raises(ValueError, match="must map to an object"):
            PriceBook.from_dict({"openai": "bad"})

    def test_raises_when_model_value_is_not_dict(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            PriceBook.from_dict({"openai": {"gpt-4o": "bad"}})

    def test_raises_when_input_key_missing(self) -> None:
        with pytest.raises(ValueError, match="must have 'input' and 'output'"):
            PriceBook.from_dict({"openai": {"gpt-4o": {"output": 10.0}}})

    def test_raises_when_output_key_missing(self) -> None:
        with pytest.raises(ValueError, match="must have 'input' and 'output'"):
            PriceBook.from_dict({"openai": {"gpt-4o": {"input": 2.5}}})

    def test_empty_dict_produces_empty_book(self) -> None:
        b = PriceBook.from_dict({})
        assert b.has_pricing("openai", "gpt-4o") is False


# ---------------------------------------------------------------------------
# has_pricing
# ---------------------------------------------------------------------------


class TestHasPricing:
    def test_known_pair_returns_true(self) -> None:
        assert book().has_pricing("openai", "gpt-4o") is True

    def test_unknown_provider_returns_false(self) -> None:
        assert book().has_pricing("unknown-provider", "gpt-4o") is False

    def test_unknown_model_returns_false(self) -> None:
        assert book().has_pricing("openai", "gpt-99") is False

    def test_known_second_provider(self) -> None:
        assert book().has_pricing("anthropic", "claude-3-5-sonnet") is True


# ---------------------------------------------------------------------------
# pricing_for
# ---------------------------------------------------------------------------


class TestPricingFor:
    def test_returns_model_pricing_for_known_pair(self) -> None:
        p = book().pricing_for("openai", "gpt-4o")
        assert isinstance(p, ModelPricing)

    def test_returns_none_for_unknown_provider(self) -> None:
        assert book().pricing_for("does-not-exist", "gpt-4o") is None

    def test_returns_none_for_unknown_model(self) -> None:
        assert book().pricing_for("openai", "does-not-exist") is None


# ---------------------------------------------------------------------------
# cost_for_delta
# ---------------------------------------------------------------------------


class TestCostForDelta:
    def test_unknown_pair_returns_zero(self) -> None:
        cost = book().cost_for_delta(
            provider="mystery",
            model="x",
            prompt_tokens=1000,
            completion_tokens=1000,
        )
        assert cost == 0.0

    def test_charges_prompt_tokens_at_input_rate(self) -> None:
        # 1000 prompt tokens * $2.50/1k = $2.50
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=0,
        )
        assert cost == pytest.approx(2.50)

    def test_charges_completion_tokens_at_output_rate(self) -> None:
        # 500 completion tokens * $10.00/1k = $5.00
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=0,
            completion_tokens=500,
        )
        assert cost == pytest.approx(5.00)

    def test_charges_cache_creation_tokens(self) -> None:
        # 2000 cache_creation tokens * $3.75/1k = $7.50
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=0,
            completion_tokens=0,
            cache_creation_tokens=2000,
        )
        assert cost == pytest.approx(7.50)

    def test_charges_cache_read_tokens(self) -> None:
        # 4000 cache_read tokens * $1.25/1k = $5.00
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_tokens=4000,
        )
        assert cost == pytest.approx(5.00)

    def test_sums_all_four_directions(self) -> None:
        # 1000 * 2.50/1k + 500 * 10.00/1k + 200 * 3.75/1k + 100 * 1.25/1k
        # = 2.50 + 5.00 + 0.75 + 0.125 = 8.375
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
            cache_creation_tokens=200,
            cache_read_tokens=100,
        )
        assert cost == pytest.approx(8.375)

    def test_zero_tokens_yields_zero_cost(self) -> None:
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert cost == 0.0

    def test_model_without_cache_rates_zero_cache_cost(self) -> None:
        # gpt-3.5-turbo has no cache_creation/cache_read in the config
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-3.5-turbo",
            prompt_tokens=0,
            completion_tokens=0,
            cache_creation_tokens=5000,
            cache_read_tokens=5000,
        )
        assert cost == 0.0

    def test_sub_thousand_tokens(self) -> None:
        # 100 prompt tokens * $2.50/1k = $0.25
        cost = book().cost_for_delta(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=0,
        )
        assert cost == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------


class TestFromFile:
    def test_loads_equivalent_to_from_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "prices.json"
        path.write_text(json.dumps(SAMPLE_DICT), encoding="utf-8")
        b = PriceBook.from_file(path)
        assert b.has_pricing("openai", "gpt-4o")
        assert b.pricing_for("openai", "gpt-4o") == PriceBook.from_dict(SAMPLE_DICT).pricing_for(
            "openai", "gpt-4o"
        )

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PriceBook.from_file(tmp_path / "nonexistent.json")
