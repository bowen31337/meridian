"""
Tests for the pluggable embedder selection in _kb (hash vs fastembed).

The "fastembed" path is exercised with an injected fake model (reporting its own
dimension) so the heavy ONNX dependency / model download is never required, and
the vector dimension is auto-derived from whatever model is loaded.
"""

from __future__ import annotations

from typing import Any

from meridiand import _kb
import pytest

_FAKE_DIM = 1024  # simulate a stronger model (e.g. mxbai/bge-large)


class _FakeFastembed:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _FAKE_DIM for _ in texts]

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.2] * _FAKE_DIM for _ in texts]


def _use(monkeypatch: pytest.MonkeyPatch, kind: str | None, *, model: Any = None) -> None:
    monkeypatch.setattr(_kb, "_embedder_kind_cache", None)
    monkeypatch.setattr(_kb, "_fastembed_model", model)
    monkeypatch.setattr(_kb, "_fastembed_dim_cache", None)
    if kind is None:
        monkeypatch.delenv("MERIDIAN_EMBEDDER", raising=False)
    else:
        monkeypatch.setenv("MERIDIAN_EMBEDDER", kind)


class TestEmbedderSelection:
    def test_default_is_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, None)
        assert _kb._embedder_kind() == "hash"
        assert _kb._embed_dim() == _kb._EMBED_DIM == 128
        assert "FLOAT[128]" in _kb._create_vec_sql()

    def test_fastembed_dim_auto_derived(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, "fastembed", model=_FakeFastembed())
        assert _kb._embedder_kind() == "fastembed"
        assert _kb._embed_dim() == _FAKE_DIM  # derived from the model, not hardcoded
        assert f"FLOAT[{_FAKE_DIM}]" in _kb._create_vec_sql()

    def test_dim_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, "fastembed", model=_FakeFastembed())
        assert _kb._embed_dim() == _FAKE_DIM
        # second call uses the cache (model is irrelevant now)
        assert _kb._embed_dim() == _FAKE_DIM

    def test_kind_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, "fastembed", model=_FakeFastembed())
        assert _kb._embedder_kind() == "fastembed"
        monkeypatch.setenv("MERIDIAN_EMBEDDER", "hash")  # cache already set -> ignored
        assert _kb._embedder_kind() == "fastembed"


class TestActiveEmbedderId:
    def test_hash_id_encodes_dimension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, None)
        assert _kb.active_embedder_id() == "hash-128"

    def test_fastembed_id_uses_configured_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, "fastembed", model=_FakeFastembed())
        monkeypatch.setenv("MERIDIAN_EMBEDDER_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
        assert _kb.active_embedder_id() == "fastembed:mixedbread-ai/mxbai-embed-large-v1"

    def test_fastembed_id_falls_back_to_default_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(monkeypatch, "fastembed", model=_FakeFastembed())
        monkeypatch.delenv("MERIDIAN_EMBEDDER_MODEL", raising=False)
        assert _kb.active_embedder_id() == f"fastembed:{_kb._DEFAULT_FASTEMBED_MODEL}"

    def test_id_does_not_load_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Resolving the id must not force the heavy ONNX model to load.
        _use(monkeypatch, "fastembed", model=None)  # no model injected
        monkeypatch.setenv("MERIDIAN_EMBEDDER_MODEL", "some/model")
        assert _kb.active_embedder_id() == "fastembed:some/model"  # no probe, no load


class TestEmbedDispatch:
    def test_hash_document_and_query_are_128(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, None)
        assert len(_kb._embed_document("hello world")) == 128 * 4
        assert len(_kb._embed_query("hello world")) == 128 * 4

    def test_fastembed_document_and_query_match_model_dim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _use(monkeypatch, "fastembed", model=_FakeFastembed())
        assert len(_kb._embed_document("a document")) == _FAKE_DIM * 4
        assert len(_kb._embed_query("a query")) == _FAKE_DIM * 4

    def test_get_fastembed_returns_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake: Any = _FakeFastembed()
        monkeypatch.setattr(_kb, "_fastembed_model", fake)
        assert _kb._get_fastembed() is fake
