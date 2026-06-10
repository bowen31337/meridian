"""
Tests for the pluggable embedder selection in _kb (hash vs fastembed).

The "fastembed" path is exercised with an injected fake model so the heavy ONNX
dependency / model download is never required.
"""

from __future__ import annotations

from typing import Any

from meridiand import _kb
import pytest


class _FakeFastembed:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _kb._FASTEMBED_DIM for _ in texts]

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.2] * _kb._FASTEMBED_DIM for _ in texts]


def _use(monkeypatch: pytest.MonkeyPatch, kind: str | None) -> None:
    monkeypatch.setattr(_kb, "_embedder_kind_cache", None)
    monkeypatch.setattr(_kb, "_fastembed_model", None)
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

    def test_fastembed_dim_and_ddl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, "fastembed")
        assert _kb._embedder_kind() == "fastembed"
        assert _kb._embed_dim() == 384
        assert "FLOAT[384]" in _kb._create_vec_sql()

    def test_kind_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, "fastembed")
        assert _kb._embedder_kind() == "fastembed"
        monkeypatch.setenv("MERIDIAN_EMBEDDER", "hash")  # cache already set -> ignored
        assert _kb._embedder_kind() == "fastembed"


class TestEmbedDispatch:
    def test_hash_document_and_query_are_128(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, None)
        assert len(_kb._embed_document("hello world")) == 128 * 4
        assert len(_kb._embed_query("hello world")) == 128 * 4

    def test_fastembed_document_and_query_are_384(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use(monkeypatch, "fastembed")
        monkeypatch.setattr(_kb, "_fastembed_model", _FakeFastembed())
        assert len(_kb._embed_document("a document")) == 384 * 4
        assert len(_kb._embed_query("a query")) == 384 * 4

    def test_get_fastembed_returns_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake: Any = _FakeFastembed()
        monkeypatch.setattr(_kb, "_fastembed_model", fake)
        assert _kb._get_fastembed() is fake
